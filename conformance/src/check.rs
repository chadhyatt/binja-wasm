//! The checks themselves

use std::path::Path;

use binja_wasm::{cfg, insn, lift, module};
use wasmparser::{
    FuncValidator, FuncValidatorAllocations, FunctionBody, Operator, Parser, Payload, ValidPayload,
    Validator, WasmFeatures, WasmModuleResources,
};

use crate::report::Report;

pub fn module(file: &Path, image: &[u8], report: &mut Report) {
    let mut validator = Validator::new_with_features(WasmFeatures::all());
    if validator.validate_all(image).is_err() {
        // Malformed by design, or using something the validator rejects; counted so a corpus that
        // stopped validating at all cannot read as a clean sweep
        report.invalid += 1;
        survive(image);
        return;
    }

    // A component's nested modules each number their own index spaces, so each is checked on its
    // own, against the same spans the plugin reads
    for span in module::core_module_spans(image) {
        if let Some(nested) = image.get(span) {
            core_module(file, nested, report);
        }
    }
}

fn survive(image: &[u8]) {
    for module in module::parse_all(image, 0) {
        for (_, info) in module.functions() {
            let Some(code) = image.get(info.entry as usize..info.end as usize) else {
                continue;
            };
            let flow = cfg::recover(code, info.entry, Some(&module));
            let _ = flow.blocks();
            for (at, _) in &flow.instructions {
                let Some(insn) = insn::decode_any(&code[(at - info.entry) as usize..]) else {
                    continue;
                };
                let _ = lift::stack_effect(&insn, module.resolve(&insn.op).as_ref());
            }
        }
    }
}

fn core_module(file: &Path, image: &[u8], report: &mut Report) {
    report.modules += 1;

    let read = module::parse(image, 0);
    let mut bodies = Vec::new();

    let mut validator = Validator::new_with_features(WasmFeatures::all());
    let mut allocations = FuncValidatorAllocations::default();

    for payload in Parser::new(0).parse_all(image) {
        // A payload this build cannot read ends the walk, but what was collected still gets
        // checked, or a walk that gave up would look like one that verified everything
        let Ok(payload) = payload else { break };

        if let Payload::CodeSectionEntry(body) = &payload {
            report.functions += 1;
            if let Ok(reader) = body.get_operators_reader() {
                bodies.push((reader.original_position(), body.range().end));
            }
        }

        match validator.payload(&payload) {
            Ok(ValidPayload::Func(to_validate, body)) => {
                let mut func = to_validate.into_validator(std::mem::take(&mut allocations));
                function(file, image, &body, &mut func, read.as_ref(), report);
                allocations = func.into_allocations();
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }

    module_layout(file, read.as_ref(), &bodies, report);
}

/// The reader is what turns `call 3` into an address, so a body it misses or misplaces is a call
/// edge pointing at nothing, or worse at the wrong function
fn module_layout(
    file: &Path,
    read: Option<&module::Module>,
    bodies: &[(u64, u64)],
    report: &mut Report,
) {
    let Some(read) = read else {
        if !bodies.is_empty() {
            report.fail(
                file,
                format!("module reader found none of {} bodies", bodies.len()),
            );
        }
        return;
    };

    let found: Vec<_> = read
        .functions()
        .map(|(_, info)| (info.entry, info.end))
        .collect();
    if found.len() != bodies.len() {
        report.fail(
            file,
            format!(
                "module reader found {} bodies, parser found {}",
                found.len(),
                bodies.len()
            ),
        );
        return;
    }

    for (nth, (ours, theirs)) in found.iter().zip(bodies).enumerate() {
        if ours != theirs {
            report.fail(
                file,
                format!(
                    "body {nth} read as {:#x}..{:#x}, parser says {:#x}..{:#x}",
                    ours.0, ours.1, theirs.0, theirs.1
                ),
            );
            return;
        }
    }
}

fn function(
    file: &Path,
    image: &[u8],
    body: &FunctionBody,
    validator: &mut FuncValidator<impl WasmModuleResources>,
    read: Option<&module::Module>,
    report: &mut Report,
) {
    let Ok(mut reader) = body.get_operators_reader() else {
        report.abandoned += 1;
        return;
    };

    // Without the body's locals the validator believes the function has only its parameters and
    // rejects the first `local.get` past them, which abandoned nearly three hundred bodies
    if let Ok(locals) = body.get_locals_reader() {
        for local in locals.into_iter() {
            let Ok((count, ty)) = local else {
                report.abandoned += 1;
                return;
            };
            if validator.define_locals(0, count, ty).is_err() {
                report.abandoned += 1;
                return;
            }
        }
    }

    // Instructions start after the locals declaration, which is not code
    let start = reader.original_position();
    let end = body.range().end;
    let code = usize::try_from(start)
        .ok()
        .zip(usize::try_from(end).ok())
        .and_then(|(start, end)| image.get(start..end));
    let Some(code) = code else {
        report.abandoned += 1;
        return;
    };

    // Walk the body with the validator so every operator's real arity is available
    let mut offset = 0usize;
    // The validator's own height per instruction, which recovery is compared against afterwards
    let mut truth_heights: Vec<(u64, Option<u64>)> = Vec::new();
    while !reader.eof() {
        let at = reader.original_position();
        let Ok(op) = reader.read() else {
            report.abandoned += 1;
            return;
        };
        let consumed = reader.original_position() - at;

        // The core caps how many bytes it asks an architecture about, so a `br_table` past that is
        // one no plugin could disassemble and is counted rather than failed
        if consumed > insn::MAX_INSTR_LEN as u64 {
            report.oversized += 1;
        }

        let Some(decoded) = insn::decode_any(&code[offset..]) else {
            report.fail(file, format!("{at:#x}: no decode for {op:?}"));
            return;
        };
        if decoded.len as u64 != consumed {
            report.fail(
                file,
                format!(
                    "{at:#x}: decoded {} bytes, wasmparser read {consumed}",
                    decoded.len
                ),
            );
            return;
        }

        report.instructions += 1;

        // Only where control reaches the instruction, since past a branch the validator switches
        // to a polymorphic stack whose height is not a number this layer claims either
        let reachable = validator
            .get_control_frame(0)
            .is_some_and(|frame| !frame.unreachable);
        truth_heights.push((
            at,
            reachable.then(|| u64::from(validator.operand_stack_height())),
        ));

        let call = read.and_then(|read| read.resolve(&op));
        let truth = op.operator_arity(validator);
        if truth.is_none() {
            // Nothing to compare against, so neither check ran, which is worth a number rather
            // than silence
            report.unchecked += 1;
        }
        stack_effect(file, at, &decoded, call, truth, report);
        call_target(file, at, &op, truth, read, report);

        if validator.op(at, &op).is_err() {
            report.abandoned += 1;
            return;
        }
        offset += decoded.len;
    }

    if offset != code.len() {
        report.fail(
            file,
            format!("body at {start:#x} left {} bytes over", code.len() - offset),
        );
    }

    control_flow(file, code, start, read, &truth_heights, report);
}

/// Every [`cfg::Unwind`] is derived from this
///
/// Only compared where control reaches the instruction and the walk claims to know the height,
/// since claiming to know one that disagrees is the failure
fn heights(
    file: &Path,
    flow: &cfg::ControlFlow,
    truth: &[(u64, Option<u64>)],
    report: &mut Report,
) {
    for ((addr, _), ours) in flow.instructions.iter().zip(&flow.heights) {
        let Some(ours) = ours else {
            report.unwound(false);
            continue;
        };
        let Ok(nth) = truth.binary_search_by_key(addr, |(at, _)| *at) else {
            continue;
        };
        let Some(theirs) = truth[nth].1 else { continue };
        report.unwound(true);
        if *ours != theirs as i64 {
            report.fail(
                file,
                format!("{addr:#x}: stack height recovered as {ours}, validator says {theirs}"),
            );
            return;
        }
    }
}

fn landings(
    file: &Path,
    code: &[u8],
    start: u64,
    flow: &cfg::ControlFlow,
    truth: &[(u64, Option<u64>)],
    report: &mut Report,
) {
    let validator = |addr: u64| {
        let nth = truth.binary_search_by_key(&addr, |(at, _)| *at).ok()?;
        truth[nth].1.map(|height| height as i64)
    };
    for ((addr, _), ours) in flow.instructions.iter().zip(&flow.heights) {
        let (Some(before), Some(terminator)) = (ours, flow.terminators.get(addr)) else {
            continue;
        };
        let Some(insn) = code
            .get((addr - start) as usize..)
            .and_then(insn::decode_any)
        else {
            continue;
        };
        let taken = match insn.flow() {
            insn::Flow::Branch => *before,
            insn::Flow::ConditionalBranch => before - lift::taken_pops(&insn.op),
            insn::Flow::IndirectBranch => before - 1,
            _ if insn::resume_table(&insn.op).is_some() => *before,
            _ => continue,
        };
        let unwind = |index: usize| {
            flow.unwinds
                .get(addr)
                .and_then(|unwinds| unwinds.get(index))
                .copied()
                .unwrap_or_default()
        };
        let edges: Vec<(u64, i64)> = match terminator {
            cfg::Terminator::Jump(to) | cfg::Terminator::Branch { taken: to, .. } => {
                vec![(*to, taken - i64::from(unwind(0).drop))]
            }
            cfg::Terminator::Table { targets, default } => targets
                .iter()
                .chain([default])
                .enumerate()
                .filter_map(|(nth, to)| Some(((*to)?, taken - i64::from(unwind(nth).drop))))
                .collect(),
            cfg::Terminator::Suspend { handlers, .. } => handlers
                .iter()
                .enumerate()
                .filter_map(|(nth, (_, to))| {
                    let unwind = unwind(nth);
                    Some((
                        (*to)?,
                        taken + i64::from(unwind.keep) - i64::from(unwind.drop),
                    ))
                })
                .collect(),
            _ => continue,
        };
        for (to, lands) in edges {
            let Some(expected) = validator(to) else {
                continue;
            };
            report.landings += 1;
            if lands != expected {
                report.fail(
                    file,
                    format!(
                        "{addr:#x}: {} lands at {to:#x} with {lands} on the stack, validator says \
                         {expected}",
                        insn.mnemonic()
                    ),
                );
                return;
            }
        }
    }
    for (addr, dispatch) in &flow.dispatch {
        for clause in dispatch.offered().clauses {
            let (Some(to), Some(base)) = (clause.target, clause.base) else {
                continue;
            };
            let Some(expected) = validator(to) else {
                continue;
            };
            let lands = base + i64::from(clause.carried) + i64::from(clause.exnref);
            report.landings += 1;
            if lands != expected {
                report.fail(
                    file,
                    format!(
                        "{addr:#x}: a handler lands at {to:#x} with {lands} on the stack, \
                         validator says {expected}"
                    ),
                );
                return;
            }
        }
    }
}

fn stack_effect(
    file: &Path,
    at: u64,
    decoded: &insn::Instruction,
    call: Option<module::Resolved>,
    truth: Option<(u32, u32)>,
    report: &mut Report,
) {
    match (lift::stack_effect(decoded, call.as_ref()), truth) {
        (Some(ours), Some((pops, pushes))) => {
            let expected = i64::from(pops) - i64::from(pushes);
            if ours != expected {
                report.fail(
                    file,
                    format!(
                        "{at:#x}: {} moves the stack by {ours}, should be {expected}",
                        decoded.mnemonic()
                    ),
                );
            }
        }
        // `None` is a claim rather than an omission where control leaves the instruction, arms
        // included; anything else returning `None` is a real gap
        (None, _) if !decoded.flow().falls_through() => report.leaves(decoded.mnemonic()),
        (None, _) => report.unmodelled(decoded.mnemonic()),
        (Some(_), None) => {}
    }
}

/// The check that pins the function index space, since miscounting imports shifts every index and
/// the arities stop lining up almost immediately
fn call_target(
    file: &Path,
    at: u64,
    op: &Operator,
    truth: Option<(u32, u32)>,
    read: Option<&module::Module>,
    report: &mut Report,
) {
    let Operator::Call { function_index } = op else {
        return;
    };
    let Some(read) = read else { return };
    let index = *function_index;

    match (read.arity(index), truth) {
        (Some(ours), Some((pops, pushes))) if (ours.pops, ours.pushes) != (pops, pushes) => {
            report.fail(
                file,
                format!(
                    "{at:#x}: call {index} reads as {}->{}, validator says {pops}->{pushes}",
                    ours.pops, ours.pushes
                ),
            );
        }
        (None, Some(_)) => report.fail(file, format!("{at:#x}: call {index} has no signature")),
        _ => {}
    }
}

/// The rest of `control_flow` only checks that recovery is self-consistent, which an operator it
/// ignores entirely still satisfies: no terminator means no leader, no leader means one big block,
/// and one big block tiles the body perfectly
fn terminators_exist(
    file: &Path,
    code: &[u8],
    start: u64,
    flow: &cfg::ControlFlow,
    report: &mut Report,
) {
    for (addr, _) in &flow.instructions {
        let Some(insn) = code
            .get((addr - start) as usize..)
            .and_then(insn::decode_any)
        else {
            continue;
        };
        let decides = matches!(
            insn.flow(),
            insn::Flow::Branch
                | insn::Flow::ConditionalBranch
                | insn::Flow::IndirectBranch
                | insn::Flow::Return
                | insn::Flow::Trap
                | insn::Flow::TailCall
        );
        if decides && !flow.terminators.contains_key(addr) {
            report.fail(
                file,
                format!("{addr:#x}: {} was given no terminator", insn.mnemonic()),
            );
            return;
        }
    }
}

/// The recovered control flow has to describe exactly the bytes it was given
fn control_flow(
    file: &Path,
    code: &[u8],
    start: u64,
    read: Option<&module::Module>,
    truth: &[(u64, Option<u64>)],
    report: &mut Report,
) {
    let flow = cfg::recover(code, start, read);
    let end = start + code.len() as u64;

    terminators_exist(file, code, start, &flow, report);
    heights(file, &flow, truth, report);
    landings(file, code, start, &flow, truth, report);
    report.unwinding += flow
        .unwinds
        .values()
        .flatten()
        .filter(|unwind| !unwind.is_empty())
        .count() as u64;

    if flow.truncated {
        report.fail(file, format!("body at {start:#x} did not close"));
        return;
    }
    if flow.end != end {
        report.fail(
            file,
            format!(
                "body at {start:#x} recovered to {:#x}, body ends at {end:#x}",
                flow.end
            ),
        );
        return;
    }

    let blocks = flow.blocks();
    if blocks.is_empty() {
        report.fail(file, format!("body at {start:#x} produced no blocks"));
        return;
    }
    if blocks[0].start != start || blocks[blocks.len() - 1].end != end {
        report.fail(file, format!("blocks at {start:#x} do not span the body"));
        return;
    }

    for pair in blocks.windows(2) {
        if pair[0].end != pair[1].start || pair[0].start >= pair[0].end {
            report.fail(file, format!("blocks at {start:#x} do not tile"));
            return;
        }
    }

    for block in &blocks {
        for edge in &block.edges {
            let target = match edge {
                cfg::Edge::Unconditional(to) | cfg::Edge::True(to) | cfg::Edge::False(to) => *to,
                _ => continue,
            };
            if !blocks.iter().any(|block| block.start == target) {
                report.fail(
                    file,
                    format!(
                        "edge from {:#x} to {target:#x} is not a block start",
                        block.start
                    ),
                );
                return;
            }
        }
    }

    dispatch(file, &flow, &blocks, report);
}

/// Every handler a raise can reach has to start a block, and one reached from live code has to be
/// handed to the core, or its code is never analysed
fn dispatch(file: &Path, flow: &cfg::ControlFlow, blocks: &[cfg::Block], report: &mut Report) {
    let reachable = flow.reachable_blocks();
    if reachable.first().map(|block| block.start) != Some(flow.start)
        || reachable.iter().any(|block| !blocks.contains(block))
    {
        report.fail(
            file,
            format!(
                "the reachable blocks at {:#x} are not blocks of the body",
                flow.start
            ),
        );
        return;
    }
    let reached = |at: u64| {
        reachable
            .iter()
            .any(|block| (block.start..block.end).contains(&at))
    };
    report.dispatching += flow.dispatch.len() as u64;
    for (at, dispatch) in &flow.dispatch {
        if flow
            .instructions
            .binary_search_by_key(at, |(start, _)| *start)
            .is_err()
        {
            report.fail(
                file,
                format!("{at:#x}: dispatch where no instruction starts"),
            );
            return;
        }
        let offered = dispatch.offered().clauses;
        for target in offered.into_iter().filter_map(|clause| clause.target) {
            if !blocks.iter().any(|block| block.start == target) {
                report.fail(
                    file,
                    format!("{at:#x}: the handler at {target:#x} is not a block start"),
                );
                return;
            }
            if reached(*at) && !reached(target) {
                report.fail(
                    file,
                    format!("{at:#x}: the handler at {target:#x} is dropped though this is live"),
                );
                return;
            }
        }
    }
}
