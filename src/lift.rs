//! LLIL lifting for the WebAssembly operand stack
//!
//! The operand stack is memory below [`RegKind::Sp`], one [`SLOT`] per value, with the locals in a
//! frame below [`RegKind::Fp`] and the globals in a fixed region
//!
//! Values are written at their own width rather than the slot's, which is safe because a slot is
//! statically typed and only ever read back at the type that wrote it

use binaryninja::low_level_il::expression::ValueExpr;
use binaryninja::low_level_il::lifting::LowLevelILLabel;
use binaryninja::low_level_il::{
    LowLevelILMutableExpression, LowLevelILMutableFunction, LowLevelILRegisterKind,
    LowLevelILTempRegister,
};
use wasmparser::Operator;

use crate::arch::{RegKind, WasmIntrinsic, WasmRegister};
use crate::cfg::{Recovered, Terminator, Unwind};
use crate::insn::{Arity, Flow, Instruction};
use crate::module::{Call, Resolved, ValueKind};

/// Wide enough for `i64` and `f64`; a `v128` does not fit and is left unlifted
pub const SLOT: u64 = 8;

/// An address is as wide as the module makes it, and where the globals live follows the file
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    pub addr: usize,
    pub global_base: u64,
    pub table_base: u64,
    /// How many of the current function's locals are parameters, and how many there are in total
    pub frame: Frame,
}

/// The parameters are the frame the caller built, first at the base and the rest above it; the
/// ones the body declares for itself go below, in the room the prologue makes
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Frame {
    pub params: u32,
    pub locals: u32,
    pub results: u32,
    pub result: Option<ValueKind>,
    pub entry: bool,
}

impl Frame {
    /// Parameters run up from the base in declaration order, where [`argument_frame`] puts them,
    /// and the declared locals go below
    fn offset(self, index: u32) -> i64 {
        match index.checked_sub(self.params) {
            None => i64::from(index) * SLOT as i64,
            Some(declared) => -(i64::from(declared) + 1) * SLOT as i64,
        }
    }

    fn reserved(self) -> i64 {
        i64::from(self.locals.saturating_sub(self.params)) * SLOT as i64
    }
}

impl Default for Model {
    fn default() -> Self {
        Self {
            addr: 4,
            global_base: 0,
            table_base: 0,
            frame: Frame::default(),
        }
    }
}

impl Model {
    fn reg(self, kind: RegKind) -> LowLevelILRegisterKind<WasmRegister> {
        LowLevelILRegisterKind::Arch(WasmRegister::new(kind, self.addr))
    }

    /// Where a return goes, since wasm keeps its call stack out of reach of the program
    pub fn link<'a>(
        self,
        il: &'a LowLevelILMutableFunction,
    ) -> LowLevelILMutableExpression<'a, ValueExpr> {
        il.reg(self.addr, self.reg(RegKind::Lr))
    }
}

/// Without `recovered` and `resolved` a branch has no address to name and a call no signature, and
/// the best the IL can say is that control goes somewhere unknown
pub fn lift(
    il: &LowLevelILMutableFunction,
    model: Model,
    insn: &Instruction,
    recovered: Option<&Recovered>,
    resolved: Option<&Resolved>,
    width: Option<usize>,
) {
    // Moving a local at the slot width writes eight bytes into a slot the next operator reads four
    // out of, and the core warns on every such access
    let moved = width.unwrap_or(SLOT as usize);
    if model.frame.entry {
        prologue(il, model);
    }
    // Calls come first, since a tail call is a terminator too and lifting it as one loses the
    // callee
    if let Some(Resolved::Call(call)) = resolved {
        if lift_call(il, model, insn, call) {
            return;
        }
    }

    if let Some(recovered) = recovered {
        if lift_terminator(il, model, insn, recovered) {
            return;
        }
    }

    match semantics(insn) {
        Sem::Nop => il.add_instruction(il.nop()),
        Sem::Trap => il.add_instruction(il.no_ret()),
        Sem::Return => leave(il, model),
        Sem::Const { size, value } => {
            push(il, model);
            let value = il.const_int(size, value);
            il.add_instruction(il.store(size, slot(il, model, 0), value));
        }
        Sem::Unary { size, kind } => {
            let value = peek(il, model, 0, size);
            let result = unary(il, kind, size, value);
            il.add_instruction(il.store(size, slot(il, model, 0), result));
        }
        Sem::Binary { size, kind } => {
            let left = peek(il, model, 1, size);
            let right = peek(il, model, 0, size);
            let result = binary(il, kind, size, left, right);
            il.add_instruction(il.store(size, slot(il, model, 1), result));
            pop(il, model, 1);
        }
        Sem::Compare { size, kind } => {
            let left = peek(il, model, 1, size);
            let right = peek(il, model, 0, size);
            // A comparison is built at the width of what it compares, but the answer is an `i32`
            let result = compare(il, kind, size, left, right);
            let result = il.expression(il.bool_to_int(RESULT_I32, result));
            il.add_instruction(il.store(RESULT_I32, slot(il, model, 1), result));
            pop(il, model, 1);
        }
        Sem::TestZero { size } => {
            let value = peek(il, model, 0, size);
            let zero = il.const_int(size, 0);
            let result = compare(il, Cmp::Eq, size, value, zero);
            let result = il.expression(il.bool_to_int(RESULT_I32, result));
            il.add_instruction(il.store(RESULT_I32, slot(il, model, 0), result));
        }
        Sem::Convert { from, to, kind } => {
            let value = peek(il, model, 0, from);
            let result = convert(il, kind, from, to, value);
            il.add_instruction(il.store(to, slot(il, model, 0), result));
        }
        Sem::Load {
            access,
            result,
            extend,
            offset,
        } => {
            let address = linear_address(il, model, 0, offset);
            let loaded = il.expression(il.load(access, address));
            let value = match extend {
                Extend::None => loaded,
                Extend::Sign => il.expression(il.sx(result, loaded)),
                Extend::Zero => il.expression(il.zx(result, loaded)),
            };
            il.add_instruction(il.store(result, slot(il, model, 0), value));
        }
        Sem::Store {
            access,
            value: value_size,
            offset,
        } => {
            let address = linear_address(il, model, 1, offset);
            let value = peek(il, model, 0, value_size);
            let value = if access < value_size {
                il.expression(il.low_part(access, value))
            } else {
                value
            };
            il.add_instruction(il.store(access, address, value));
            pop(il, model, 2);
        }
        Sem::LocalGet(index) => {
            push(il, model);
            let value = il.expression(il.load(moved, frame_slot(il, model, index)));
            il.add_instruction(il.store(moved, slot(il, model, 0), value));
        }
        Sem::LocalSet(index) => {
            let value = peek(il, model, 0, moved);
            il.add_instruction(il.store(moved, frame_slot(il, model, index), value));
            pop(il, model, 1);
        }
        Sem::LocalTee(index) => {
            let value = peek(il, model, 0, moved);
            il.add_instruction(il.store(moved, frame_slot(il, model, index), value));
        }
        Sem::GlobalGet(index) => {
            push(il, model);
            let value = il.expression(il.load(moved, global_slot(il, model, index)));
            il.add_instruction(il.store(moved, slot(il, model, 0), value));
        }
        Sem::GlobalSet(index) => {
            let value = peek(il, model, 0, moved);
            il.add_instruction(il.store(moved, global_slot(il, model, index), value));
            pop(il, model, 1);
        }
        Sem::Drop => pop(il, model, 1),
        Sem::CondBranch => {
            pop(il, model, 1);
            il.add_instruction(il.unimplemented());
        }
        // Where it goes needs the block stack, but what it does to the operand stack does not
        Sem::RefBranch => il.add_instruction(il.unimplemented()),
        // Not an intrinsic: the count is per instruction and an intrinsic's prototype is per
        // operator, so the two could not agree
        Sem::Aggregate { pops, pushes } => aggregate(il, model, Arity { pops, pushes }),
        // An aggregate whose width only the module knows, so `semantics` could not see it
        _ if matches!(resolved, Some(Resolved::Aggregate(_))) => {
            let Some(Resolved::Aggregate(arity)) = resolved else {
                return;
            };
            aggregate(il, model, *arity);
        }
        Sem::Opaque => opaque(il, model, insn),
    }
}

fn aggregate(il: &LowLevelILMutableFunction, model: Model, arity: Arity) {
    let net = i64::from(arity.pops) - i64::from(arity.pushes);
    if net != 0 {
        adjust_sp(il, model, net * SLOT as i64);
    }
    for depth in 0..arity.pushes {
        let value = il.undefined();
        il.add_instruction(il.store(SLOT as usize, slot(il, model, u64::from(depth)), value));
    }
}

/// The operands a branch consumes are popped first, so the stack is right on every edge
fn lift_terminator(
    il: &LowLevelILMutableFunction,
    model: Model,
    insn: &Instruction,
    recovered: &Recovered,
) -> bool {
    match &recovered.terminator {
        Terminator::Jump(target) => {
            // `else` and `end` mark a place rather than consuming anything, but a real `br`
            // reaches here too and that one unwinds
            unwind(il, model, recovered.unwind(0));
            go(il, model, Some(*target));
            true
        }
        Terminator::Branch { taken, not_taken } => {
            let condition = branch_condition(il, model, insn);

            // Some reference tests hand their operand to the label and drop it otherwise, so the
            // fallthrough needs a pop the taken edge must not get
            let dropped = fallthrough_pops(insn);
            let leaving = recovered.unwind(0);
            let fixup = dropped != 0 || !leaving.is_empty();

            match (
                il.label_for_address(*taken),
                il.label_for_address(*not_taken),
            ) {
                (Some(mut hit), Some(mut miss)) if !fixup => {
                    il.add_instruction(il.if_expr(condition, &mut hit, &mut miss));
                }
                // Either an edge needs work before it is taken or the core has no label for one of
                // these addresses; both go somewhere known, so both get a branch
                _ => {
                    let mut hit = LowLevelILLabel::new();
                    let mut miss = LowLevelILLabel::new();
                    il.add_instruction(il.if_expr(condition, &mut hit, &mut miss));

                    il.mark_label(&mut hit);
                    unwind(il, model, leaving);
                    go(il, model, Some(*taken));

                    il.mark_label(&mut miss);
                    if dropped != 0 {
                        pop(il, model, dropped);
                    }
                    go(il, model, Some(*not_taken));
                }
            }
            true
        }
        Terminator::Table { targets, default } => {
            lift_table(il, model, targets, *default, recovered);
            true
        }
        Terminator::Return => {
            // A `return` leaves what the function produces on the stack, and the frame the caller
            // restores is its own business
            leave(il, model);
            true
        }
        Terminator::ConditionalReturn { .. } => {
            // Two labels of its own rather than addresses, since the return is not an instruction
            // anywhere in the function
            let condition = branch_condition(il, model, insn);
            let mut leaving = LowLevelILLabel::new();
            let mut carry_on = LowLevelILLabel::new();
            il.add_instruction(il.if_expr(condition, &mut leaving, &mut carry_on));

            il.mark_label(&mut leaving);
            leave(il, model);
            il.mark_label(&mut carry_on);
            true
        }
        Terminator::Halt => {
            il.add_instruction(il.no_ret());
            true
        }
        Terminator::Unresolved => {
            // Not knowing where it goes is no reason to leave its condition on the stack
            if matches!(insn.op, Operator::BrIf { .. } | Operator::If { .. }) {
                pop(il, model, 1);
            }
            il.add_instruction(il.jump(il.unimplemented()));
            true
        }
    }
}

fn branch_condition<'a>(
    il: &'a LowLevelILMutableFunction,
    model: Model,
    insn: &Instruction,
) -> LowLevelILMutableExpression<'a, ValueExpr> {
    match insn.op {
        // The value moves to a temporary first, since the test is emitted as part of the branch
        // after the pop, by which point `sp` no longer points at the slot it came from
        Operator::BrIf { .. } | Operator::If { .. } => {
            let held = temp(0);
            let value = peek(il, model, 0, RESULT_I32);
            il.add_instruction(il.set_reg(RESULT_I32, held, value));
            pop(il, model, 1);

            let condition = il.reg(RESULT_I32, held);
            let zero = il.const_int(RESULT_I32, 0);
            il.expression(il.cmp_ne(RESULT_I32, condition, zero))
        }
        // Which edge keeps the reference is decided by the type of the block branched to, which
        // this layer cannot see
        Operator::BrOnNull { .. } => {
            let value = peek(il, model, 0, model.addr);
            let null = il.const_int(model.addr, 0);
            il.expression(il.cmp_e(model.addr, value, null))
        }
        Operator::BrOnNonNull { .. } => {
            let value = peek(il, model, 0, model.addr);
            let null = il.const_int(model.addr, 0);
            il.expression(il.cmp_ne(model.addr, value, null))
        }
        // A type is not a value the IL has, and an unknown condition keeps both edges live
        _ => il.unimplemented(),
    }
}

/// The arguments are consumed here rather than by the callee, whose locals are lifted against `fp`
/// so it never reads the caller's operand stack
fn lift_call(
    il: &LowLevelILMutableFunction,
    model: Model,
    insn: &Instruction,
    call: &Call,
) -> bool {
    let target = match insn.op {
        Operator::Call { .. } | Operator::ReturnCall { .. } => match call.target {
            Some(entry) => il.const_ptr(entry),
            // An import has no body in this image, and an unknown target keeps the call in the IL
            // without pointing it at an address that stands for nothing
            None => il.unimplemented(),
        },
        // A reference is the callee itself, so it is read before the pop moves the stack out from
        // under the call
        Operator::CallRef { .. } | Operator::ReturnCallRef { .. } => {
            let held = temp(0);
            let callee = peek(il, model, 0, model.addr);
            il.add_instruction(il.set_reg(model.addr, held, callee));
            pop(il, model, 1);
            il.reg(model.addr, held)
        }
        // An index is not an address: the callee is whatever the slot holds, and naming the index
        // would point the call into linear memory
        Operator::CallIndirect { .. } | Operator::ReturnCallIndirect { .. } => {
            let held = temp(0);
            let index = peek(il, model, 0, model.addr);
            il.add_instruction(il.set_reg(model.addr, held, index));
            pop(il, model, 1);

            let index = il.reg(model.addr, held);
            let stride = il.const_int(model.addr, model.addr as u64);
            let offset = il.expression(il.mul(model.addr, index, stride));
            let base = il.const_ptr_sized(model.addr, model.table_base);
            let at = il.expression(il.add(model.addr, base, offset));
            il.expression(il.load(model.addr, at))
        }
        _ => return false,
    };

    // What is left of the arity after the callee operand is the arguments and the results
    let arguments = i64::from(call.arity.pops) - i64::from(indirect(insn));
    argument_frame(il, model, arguments, &call.params);

    // A tail call replaces the frame, so there is no stack left to put results back on
    if !insn.flow().falls_through() {
        il.add_instruction(il.tailcall(target));
        return true;
    }

    il.add_instruction(il.call(target));

    // Drop the argument frame and the arguments, and make room for what came back
    let net = arguments - i64::from(call.arity.pushes);
    adjust_sp(il, model, (arguments + net) * SLOT as i64);

    // The first result comes back in the register the convention names, which is what ties the
    // value to the call; anything past it has nowhere to come from
    for depth in 0..call.arity.pushes {
        let (width, value) = if depth == call.arity.pushes - 1 {
            let width = call.result.map_or(SLOT as usize, moved_width);
            (width, il.reg(width, model.reg(RegKind::Rv)))
        } else {
            (SLOT as usize, il.expression(il.undefined()))
        };
        il.add_instruction(il.store(width, slot(il, model, u64::from(depth)), value));
    }

    true
}

/// A wasm call pushes its arguments onto a stack that grows down, so the first ends up highest,
/// while the core lays a stack parameter list out the other way, and nothing in the type system can
/// say so, since an explicit location is dropped when the type reaches a function
///
/// The core folds the copies into the call the way it folds a cdecl push
fn argument_frame(
    il: &LowLevelILMutableFunction,
    model: Model,
    arguments: i64,
    params: &[ValueKind],
) {
    if arguments <= 0 {
        return;
    }
    for nth in 0..arguments {
        // Each at its own width, since copying a slot the caller wrote four bytes into as eight
        // claims the other half meant something
        let width = params
            .get(nth as usize)
            .map_or(SLOT as usize, |kind| moved_width(*kind));
        let value = peek(il, model, (arguments - 1 - nth) as u64, width);
        let at = signed_offset_from(il, model, RegKind::Sp, -(arguments - nth) * SLOT as i64);
        il.add_instruction(il.store(width, at, value));
    }
    adjust_sp(il, model, -arguments * SLOT as i64);
}

fn moved_width(kind: ValueKind) -> usize {
    kind.size().min(SLOT as usize)
}

/// A reference test passes what it tested to the label, so only the fallthrough drops anything
fn fallthrough_pops(insn: &Instruction) -> u64 {
    match insn.op {
        Operator::BrOnNonNull { .. }
        | Operator::BrOnCastDescEq { .. }
        | Operator::BrOnCastDescEqFail { .. } => 1,
        _ => 0,
    }
}

/// A `br_on_null` discards the reference it tested when it branches and keeps it when it does not,
/// and `br_on_non_null` does the reverse, which decides the height a branch unwinds from
pub fn taken_pops(op: &Operator) -> i64 {
    match op {
        Operator::BrIf { .. }
        | Operator::BrOnNull { .. }
        | Operator::BrOnCastDescEq { .. }
        | Operator::BrOnCastDescEqFail { .. } => 1,
        _ => 0,
    }
}

fn indirect(insn: &Instruction) -> u32 {
    u32::from(matches!(
        insn.op,
        Operator::CallIndirect { .. }
            | Operator::ReturnCallIndirect { .. }
            | Operator::CallRef { .. }
            | Operator::ReturnCallRef { .. }
    ))
}

/// LLIL exposes no jump table operation here, and a chain of tests says the same thing while
/// keeping every edge visible; the index is copied to a temporary first, since popping it moves the
/// stack out from under the later comparisons
fn lift_table(
    il: &LowLevelILMutableFunction,
    model: Model,
    targets: &[Option<u64>],
    default: Option<u64>,
    recovered: &Recovered,
) {
    let index = temp(0);
    let value = peek(il, model, 0, RESULT_I32);
    il.add_instruction(il.set_reg(RESULT_I32, index, value));
    pop(il, model, 1);

    for (entry, target) in targets.iter().enumerate() {
        let mut miss = LowLevelILLabel::new();
        let selector = il.reg(RESULT_I32, index);
        let wanted = il.const_int(RESULT_I32, entry as u64);
        let matches = il.expression(il.cmp_e(RESULT_I32, selector, wanted));

        // Each entry names a label of its own, so each unwinds by an amount of its own
        let leaving = recovered.unwind(entry);
        let direct = leaving
            .is_empty()
            .then(|| target.and_then(|to| il.label_for_address(to)))
            .flatten();
        match direct {
            Some(mut label) => il.add_instruction(il.if_expr(matches, &mut label, &mut miss)),
            // An entry that unwinds, returns, or lands outside this function needs somewhere of
            // its own to put the transfer
            None => {
                let mut hit = LowLevelILLabel::new();
                il.add_instruction(il.if_expr(matches, &mut hit, &mut miss));
                il.mark_label(&mut hit);
                unwind(il, model, leaving);
                go(il, model, *target);
            }
        }
        il.mark_label(&mut miss);
    }

    unwind(il, model, recovered.unwind(targets.len()));
    go(il, model, default);
}

/// The kept values move down over the discarded ones, highest destination first so a value is read
/// before anything is written over it
fn unwind(il: &LowLevelILMutableFunction, model: Model, unwind: Unwind) {
    if unwind.is_empty() {
        return;
    }

    let drop = u64::from(unwind.drop);
    for depth in (0..u64::from(unwind.keep)).rev() {
        let value = peek(il, model, depth, SLOT as usize);
        il.add_instruction(il.store(SLOT as usize, slot(il, model, depth + drop), value));
    }
    pop(il, model, drop);
}

/// A calling convention cannot describe results on the operand stack, so the first one moves to the
/// register [`crate::arch::WasmCallingConvention`] names; without this the callee computes its
/// result and drops it, and the caller's value comes from nowhere
fn leave(il: &LowLevelILMutableFunction, model: Model) {
    // The first result, which is the deepest, since taking the top would hand a multi-value
    // function's last result to the slot standing for its first
    if let Some(depth) = model.frame.results.checked_sub(1) {
        let result = peek(il, model, u64::from(depth), SLOT as usize);
        il.add_instruction(il.set_reg(SLOT as usize, model.reg(RegKind::Rv), result));
    }
    // `fp` is still this frame's own, so the slot the prologue wrote is a fixed distance from it
    let saved = signed_offset_from(il, model, RegKind::Fp, -saved_frame(model));
    let caller = il.expression(il.load(model.addr, saved));
    il.add_instruction(il.set_reg(model.addr, model.reg(RegKind::Fp), caller));

    let address = model.link(il);
    il.add_instruction(il.ret(address));
}

fn go(il: &LowLevelILMutableFunction, model: Model, target: Option<u64>) {
    let Some(to) = target else {
        leave(il, model);
        return;
    };

    match il.label_for_address(to) {
        Some(mut label) => il.add_instruction(il.goto(&mut label)),
        None => il.add_instruction(il.jump(il.const_ptr(to))),
    }
}

const RESULT_I32: usize = 4;

/// In slots, positive when the stack shrinks, and `None` where the lifter deliberately does not
/// move it; the conformance harness checks this against a real validator, since moving the stack by
/// the wrong amount silently corrupts every instruction after it
pub fn stack_effect(insn: &Instruction, resolved: Option<&Resolved>) -> Option<i64> {
    let net = |pops: i64, pushes: i64| Some(pops - pushes);

    // Control does not fall into an arm, so there is no fallthrough effect to state; what a
    // validator reports there is a type-level transition between two edges
    if matches!(insn.flow(), Flow::Arm) {
        return None;
    }

    // What the module resolved is the one thing the instruction alone could not have known
    match resolved {
        Some(Resolved::Call(Call { arity, .. })) => {
            return insn
                .flow()
                .falls_through()
                .then_some(i64::from(arity.pops) - i64::from(arity.pushes));
        }
        Some(Resolved::Aggregate(arity)) => {
            return net(i64::from(arity.pops), i64::from(arity.pushes));
        }
        None => {}
    }

    match semantics(insn) {
        Sem::Nop => net(0, 0),
        Sem::Trap | Sem::Return => None,
        Sem::Const { .. } | Sem::LocalGet(_) | Sem::GlobalGet(_) => net(0, 1),
        Sem::Unary { .. }
        | Sem::TestZero { .. }
        | Sem::Convert { .. }
        | Sem::Load { .. }
        | Sem::LocalTee(_) => net(1, 1),
        Sem::Binary { .. } | Sem::Compare { .. } => net(2, 1),
        Sem::Store { .. } => net(2, 0),
        Sem::LocalSet(_) | Sem::GlobalSet(_) | Sem::Drop | Sem::CondBranch => net(1, 0),
        Sem::RefBranch => net(0, 0),
        Sem::Aggregate { pops, pushes } => net(i64::from(pops), i64::from(pushes)),
        Sem::Opaque if !insn.flow().falls_through() => None,
        Sem::Opaque => {
            let Arity { pops, pushes } = insn.arity()?;
            net(i64::from(pops), i64::from(pushes))
        }
    }
}

/// Everything without a modelled meaning still has a known stack effect, so the stack pointer stays
/// honest and each produced value is marked undefined rather than silently reused
fn opaque(il: &LowLevelILMutableFunction, model: Model, insn: &Instruction) {
    if !insn.flow().falls_through() {
        match insn.flow() {
            Flow::Trap => il.add_instruction(il.no_ret()),
            _ => il.add_instruction(il.unimplemented()),
        }
        return;
    }

    let (Some(arity), Some(operator)) = (insn.arity(), insn.operator_id()) else {
        il.add_instruction(il.unimplemented());
        return;
    };

    // Arguments were pushed in order, so the first one sits deepest
    let slots = usize::try_from(arity.pops).unwrap_or(0);
    let arguments: Vec<_> = (0..slots)
        .map(|index| peek(il, model, (slots - 1 - index) as u64, SLOT as usize))
        .collect();

    let results: Vec<_> = (0..arity.pushes).map(temp).collect();

    il.add_instruction(il.intrinsic(results.clone(), WasmIntrinsic(operator), arguments));

    let net = i64::from(arity.pops) - i64::from(arity.pushes);
    if net != 0 {
        adjust_sp(il, model, net * SLOT as i64);
    }

    for (index, result) in results.into_iter().enumerate() {
        let depth = (results_len(arity.pushes) - 1 - index) as u64;
        let value = il.reg(SLOT as usize, result);
        il.add_instruction(il.store(SLOT as usize, slot(il, model, depth), value));
    }
}

fn results_len(pushes: u32) -> usize {
    usize::try_from(pushes).unwrap_or(0)
}

fn slot(
    il: &LowLevelILMutableFunction,
    model: Model,
    depth: u64,
) -> LowLevelILMutableExpression<'_, ValueExpr> {
    offset_from(il, model, RegKind::Sp, depth * SLOT)
}

fn frame_slot(
    il: &LowLevelILMutableFunction,
    model: Model,
    index: u32,
) -> LowLevelILMutableExpression<'_, ValueExpr> {
    signed_offset_from(il, model, RegKind::Fp, model.frame.offset(index))
}

fn global_slot(
    il: &LowLevelILMutableFunction,
    model: Model,
    index: u32,
) -> LowLevelILMutableExpression<'_, ValueExpr> {
    il.const_ptr_sized(
        model.addr,
        model.global_base + u64::from(index) * crate::module::GLOBAL_STRIDE,
    )
}

fn offset_from(
    il: &LowLevelILMutableFunction,
    model: Model,
    reg: RegKind,
    offset: u64,
) -> LowLevelILMutableExpression<'_, ValueExpr> {
    let base = il.reg(model.addr, model.reg(reg));
    if offset == 0 {
        base
    } else {
        let delta = il.const_int(model.addr, offset);
        il.expression(il.add(model.addr, base, delta))
    }
}

fn signed_offset_from(
    il: &LowLevelILMutableFunction,
    model: Model,
    reg: RegKind,
    offset: i64,
) -> LowLevelILMutableExpression<'_, ValueExpr> {
    let base = il.reg(model.addr, model.reg(reg));
    if offset == 0 {
        return base;
    }
    let delta = il.const_int(model.addr, offset.unsigned_abs());
    if offset < 0 {
        il.expression(il.sub(model.addr, base, delta))
    } else {
        il.expression(il.add(model.addr, base, delta))
    }
}

/// `fp` takes the stack pointer as the caller left it, which is what puts the parameters inside the
/// frame the core tracks; without it a function reads as taking an opaque pointer and dereferencing
/// it, and the prototype the module declares has nowhere to live
fn prologue(il: &LowLevelILMutableFunction, model: Model) {
    // The caller's own frame base goes just below where this one's locals will, since the
    // convention calls `fp` callee saved and a caller's locals are addressed off it
    let saved = signed_offset_from(il, model, RegKind::Sp, -saved_frame(model));
    let caller = il.reg(model.addr, model.reg(RegKind::Fp));
    il.add_instruction(il.store(model.addr, saved, caller));

    let sp = il.reg(model.addr, model.reg(RegKind::Sp));
    il.add_instruction(il.set_reg(model.addr, model.reg(RegKind::Fp), sp));
    adjust_sp(il, model, -saved_frame(model));
}

fn saved_frame(model: Model) -> i64 {
    model.frame.reserved() + SLOT as i64
}

fn peek(
    il: &LowLevelILMutableFunction,
    model: Model,
    depth: u64,
    size: usize,
) -> LowLevelILMutableExpression<'_, ValueExpr> {
    let address = slot(il, model, depth);
    il.expression(il.load(size, address))
}

fn linear_address(
    il: &LowLevelILMutableFunction,
    model: Model,
    depth: u64,
    offset: u64,
) -> LowLevelILMutableExpression<'_, ValueExpr> {
    // The operand is an offset into linear memory rather than an address in the file, so adding
    // the base memory is mapped at is what makes a load of a constant reach the bytes it reads
    let base = peek(il, model, depth, model.addr);
    let delta = il.const_int(model.addr, offset);
    il.expression(il.add(model.addr, base, delta))
}

/// The stack grows down, as everywhere else, so the core's stack analysis reads the right way round
fn push(il: &LowLevelILMutableFunction, model: Model) {
    adjust_sp(il, model, -(SLOT as i64));
}

fn pop(il: &LowLevelILMutableFunction, model: Model, slots: u64) {
    adjust_sp(il, model, (slots * SLOT) as i64);
}

/// A scratch register with no architectural counterpart, for a value that has to pass through
/// something nameable
fn temp(index: u32) -> LowLevelILRegisterKind<WasmRegister> {
    LowLevelILRegisterKind::Temp(LowLevelILTempRegister::new(index))
}

fn adjust_sp(il: &LowLevelILMutableFunction, model: Model, delta: i64) {
    let sp = il.reg(model.addr, model.reg(RegKind::Sp));
    let amount = il.const_int(model.addr, delta.unsigned_abs());
    let updated = if delta < 0 {
        il.sub(model.addr, sp, amount)
    } else {
        il.add(model.addr, sp, amount)
    };
    il.add_instruction(il.set_reg(model.addr, model.reg(RegKind::Sp), updated));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// The only unary numeric operators are the floating point ones; integers reach the same effects
/// through binary operators against a constant
enum Un {
    Neg,
    Abs,
    Sqrt,
    Ceil,
    Floor,
    Trunc,
    Nearest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bin {
    Add,
    Sub,
    Mul,
    DivS,
    DivU,
    RemS,
    RemU,
    And,
    Or,
    Xor,
    Shl,
    ShrS,
    ShrU,
    Rotl,
    Rotr,
    FAdd,
    FSub,
    FMul,
    FDiv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cmp {
    Eq,
    Ne,
    LtS,
    LtU,
    GtS,
    GtU,
    LeS,
    LeU,
    GeS,
    GeU,
    FEq,
    FNe,
    FLt,
    FGt,
    FLe,
    FGe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Conv {
    Wrap,
    SignExtend,
    ZeroExtend,
    /// LLIL has no unsigned form, so the unsigned wasm operators lift to the signed one
    FloatToInt,
    IntToFloat {
        signed: bool,
    },
    FloatResize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Extend {
    None,
    Sign,
    Zero,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Sem {
    Nop,
    Trap,
    Return,
    Drop,
    Const {
        size: usize,
        value: u64,
    },
    Unary {
        size: usize,
        kind: Un,
    },
    Binary {
        size: usize,
        kind: Bin,
    },
    Compare {
        size: usize,
        kind: Cmp,
    },
    TestZero {
        size: usize,
    },
    Convert {
        from: usize,
        to: usize,
        kind: Conv,
    },
    Load {
        access: usize,
        result: usize,
        extend: Extend,
        offset: u64,
    },
    Store {
        access: usize,
        value: usize,
        offset: u64,
    },
    CondBranch,
    /// A reference test that branches, leaving its operand where it was
    RefBranch,
    /// Moves a run of operands whose count is part of the instruction rather than the opcode
    Aggregate {
        pops: u32,
        pushes: u32,
    },
    LocalGet(u32),
    LocalSet(u32),
    LocalTee(u32),
    GlobalGet(u32),
    GlobalSet(u32),
    Opaque,
}

fn unary<'a>(
    il: &'a LowLevelILMutableFunction,
    kind: Un,
    size: usize,
    value: LowLevelILMutableExpression<'a, ValueExpr>,
) -> LowLevelILMutableExpression<'a, ValueExpr> {
    match kind {
        Un::Neg => il.expression(il.fneg(size, value)),
        Un::Abs => il.expression(il.fabs(size, value)),
        Un::Sqrt => il.expression(il.fsqrt(size, value)),
        Un::Ceil => il.expression(il.ceil(size, value)),
        Un::Floor => il.expression(il.floor(size, value)),
        Un::Trunc => il.expression(il.ftrunc(size, value)),
        Un::Nearest => il.expression(il.round_to_int(size, value)),
    }
}

fn binary<'a>(
    il: &'a LowLevelILMutableFunction,
    kind: Bin,
    size: usize,
    left: LowLevelILMutableExpression<'a, ValueExpr>,
    right: LowLevelILMutableExpression<'a, ValueExpr>,
) -> LowLevelILMutableExpression<'a, ValueExpr> {
    match kind {
        Bin::Add => il.expression(il.add(size, left, right)),
        Bin::Sub => il.expression(il.sub(size, left, right)),
        Bin::Mul => il.expression(il.mul(size, left, right)),
        Bin::DivS => il.expression(il.divs(size, left, right)),
        Bin::DivU => il.expression(il.divu(size, left, right)),
        Bin::RemS => il.expression(il.mods(size, left, right)),
        Bin::RemU => il.expression(il.modu(size, left, right)),
        Bin::And => il.expression(il.and(size, left, right)),
        Bin::Or => il.expression(il.or(size, left, right)),
        Bin::Xor => il.expression(il.xor(size, left, right)),
        Bin::Shl => il.expression(il.lsl(size, left, right)),
        Bin::ShrS => il.expression(il.asr(size, left, right)),
        Bin::ShrU => il.expression(il.lsr(size, left, right)),
        Bin::Rotl => il.expression(il.rol(size, left, right)),
        Bin::Rotr => il.expression(il.ror(size, left, right)),
        Bin::FAdd => il.expression(il.fadd(size, left, right)),
        Bin::FSub => il.expression(il.fsub(size, left, right)),
        Bin::FMul => il.expression(il.fmul(size, left, right)),
        Bin::FDiv => il.expression(il.fdiv(size, left, right)),
    }
}

fn compare<'a>(
    il: &'a LowLevelILMutableFunction,
    kind: Cmp,
    size: usize,
    left: LowLevelILMutableExpression<'a, ValueExpr>,
    right: LowLevelILMutableExpression<'a, ValueExpr>,
) -> LowLevelILMutableExpression<'a, ValueExpr> {
    let condition = match kind {
        Cmp::Eq => il.expression(il.cmp_e(size, left, right)),
        Cmp::Ne => il.expression(il.cmp_ne(size, left, right)),
        Cmp::LtS => il.expression(il.cmp_slt(size, left, right)),
        Cmp::LtU => il.expression(il.cmp_ult(size, left, right)),
        Cmp::GtS => il.expression(il.cmp_sgt(size, left, right)),
        Cmp::GtU => il.expression(il.cmp_ugt(size, left, right)),
        Cmp::LeS => il.expression(il.cmp_sle(size, left, right)),
        Cmp::LeU => il.expression(il.cmp_ule(size, left, right)),
        Cmp::GeS => il.expression(il.cmp_sge(size, left, right)),
        Cmp::GeU => il.expression(il.cmp_uge(size, left, right)),
        Cmp::FEq => il.expression(il.fcmp_e(size, left, right)),
        Cmp::FNe => il.expression(il.fcmp_ne(size, left, right)),
        Cmp::FLt => il.expression(il.fcmp_lt(size, left, right)),
        Cmp::FGt => il.expression(il.fcmp_gt(size, left, right)),
        Cmp::FLe => il.expression(il.fcmp_le(size, left, right)),
        Cmp::FGe => il.expression(il.fcmp_ge(size, left, right)),
    };
    il.expression(il.bool_to_int(RESULT_I32, condition))
}

fn convert<'a>(
    il: &'a LowLevelILMutableFunction,
    kind: Conv,
    from: usize,
    to: usize,
    value: LowLevelILMutableExpression<'a, ValueExpr>,
) -> LowLevelILMutableExpression<'a, ValueExpr> {
    match kind {
        Conv::Wrap => il.expression(il.low_part(to, value)),
        Conv::SignExtend => il.expression(il.sx(to, value)),
        Conv::ZeroExtend => il.expression(il.zx(to, value)),
        Conv::FloatToInt => il.expression(il.float_to_int(to, value)),
        Conv::IntToFloat { signed } => {
            // The operand has to be as wide as the float it becomes, and this is where the
            // operator's signedness lands: `f64.convert_i32_u` of `0xffffffff` is 4294967295
            let widened = if from < to {
                if signed {
                    il.expression(il.sx(to, value))
                } else {
                    il.expression(il.zx(to, value))
                }
            } else {
                value
            };
            il.expression(il.int_to_float(to, widened))
        }
        Conv::FloatResize => il.expression(il.float_conv(to, value)),
    }
}

/// Anything not named here is [`Sem::Opaque`] and is lifted from its stack effect alone, which
/// covers the proposals whose values do not fit a scalar slot
fn semantics(insn: &Instruction) -> Sem {
    use Operator as O;

    match &insn.op {
        O::Nop => Sem::Nop,
        O::Unreachable => Sem::Trap,
        O::Return => Sem::Return,
        O::Drop => Sem::Drop,

        // Structured control flow has no runtime effect of its own
        O::Block { .. } | O::Loop { .. } | O::End | O::Else => Sem::Nop,
        // The exception handling block openers are structure too, and `delegate` closes one the
        // way `end` does; the handlers are arms, reached along a throw edge
        O::Try { .. } | O::TryTable { .. } | O::Delegate { .. } => Sem::Nop,

        // A reference test hands its operand to the label when it branches, and on the fallthrough
        // `br_on_non_null` is the one that drops it
        O::BrOnNonNull { .. } | O::BrOnCastDescEq { .. } | O::BrOnCastDescEqFail { .. } => {
            Sem::CondBranch
        }
        O::BrOnNull { .. } | O::BrOnCast { .. } | O::BrOnCastFail { .. } => Sem::RefBranch,

        // The generated tables have no arity for this one, but the instruction carries the count
        O::ArrayNewFixed { array_size, .. } => Sem::Aggregate {
            pops: *array_size,
            pushes: 1,
        },

        // A multi-value `select` chooses between two runs of a shape the instruction spells out
        O::TypedSelectMulti { tys } => {
            let values = tys.len() as u32;
            Sem::Aggregate {
                pops: values.saturating_mul(2).saturating_add(1),
                pushes: values,
            }
        }

        // The arity table calls these variable, since entering a block also moves its parameters,
        // but the condition on top is always there
        O::If { .. } | O::BrIf { .. } => Sem::CondBranch,

        O::I32Const { value } => Sem::Const {
            size: 4,
            value: *value as u32 as u64,
        },
        O::I64Const { value } => Sem::Const {
            size: 8,
            value: *value as u64,
        },
        O::F32Const { value } => Sem::Const {
            size: 4,
            value: u64::from(value.bits()),
        },
        O::F64Const { value } => Sem::Const {
            size: 8,
            value: value.bits(),
        },

        O::LocalGet { local_index } => Sem::LocalGet(*local_index),
        O::LocalSet { local_index } => Sem::LocalSet(*local_index),
        O::LocalTee { local_index } => Sem::LocalTee(*local_index),
        O::GlobalGet { global_index } => Sem::GlobalGet(*global_index),
        O::GlobalSet { global_index } => Sem::GlobalSet(*global_index),

        O::I32Eqz => Sem::TestZero { size: 4 },
        O::I64Eqz => Sem::TestZero { size: 8 },

        O::I32Eq => int_compare(4, Cmp::Eq),
        O::I32Ne => int_compare(4, Cmp::Ne),
        O::I32LtS => int_compare(4, Cmp::LtS),
        O::I32LtU => int_compare(4, Cmp::LtU),
        O::I32GtS => int_compare(4, Cmp::GtS),
        O::I32GtU => int_compare(4, Cmp::GtU),
        O::I32LeS => int_compare(4, Cmp::LeS),
        O::I32LeU => int_compare(4, Cmp::LeU),
        O::I32GeS => int_compare(4, Cmp::GeS),
        O::I32GeU => int_compare(4, Cmp::GeU),
        O::I64Eq => int_compare(8, Cmp::Eq),
        O::I64Ne => int_compare(8, Cmp::Ne),
        O::I64LtS => int_compare(8, Cmp::LtS),
        O::I64LtU => int_compare(8, Cmp::LtU),
        O::I64GtS => int_compare(8, Cmp::GtS),
        O::I64GtU => int_compare(8, Cmp::GtU),
        O::I64LeS => int_compare(8, Cmp::LeS),
        O::I64LeU => int_compare(8, Cmp::LeU),
        O::I64GeS => int_compare(8, Cmp::GeS),
        O::I64GeU => int_compare(8, Cmp::GeU),
        O::F32Eq => int_compare(4, Cmp::FEq),
        O::F32Ne => int_compare(4, Cmp::FNe),
        O::F32Lt => int_compare(4, Cmp::FLt),
        O::F32Gt => int_compare(4, Cmp::FGt),
        O::F32Le => int_compare(4, Cmp::FLe),
        O::F32Ge => int_compare(4, Cmp::FGe),
        O::F64Eq => int_compare(8, Cmp::FEq),
        O::F64Ne => int_compare(8, Cmp::FNe),
        O::F64Lt => int_compare(8, Cmp::FLt),
        O::F64Gt => int_compare(8, Cmp::FGt),
        O::F64Le => int_compare(8, Cmp::FLe),
        O::F64Ge => int_compare(8, Cmp::FGe),

        O::I32Add => int_binary(4, Bin::Add),
        O::I32Sub => int_binary(4, Bin::Sub),
        O::I32Mul => int_binary(4, Bin::Mul),
        O::I32DivS => int_binary(4, Bin::DivS),
        O::I32DivU => int_binary(4, Bin::DivU),
        O::I32RemS => int_binary(4, Bin::RemS),
        O::I32RemU => int_binary(4, Bin::RemU),
        O::I32And => int_binary(4, Bin::And),
        O::I32Or => int_binary(4, Bin::Or),
        O::I32Xor => int_binary(4, Bin::Xor),
        O::I32Shl => int_binary(4, Bin::Shl),
        O::I32ShrS => int_binary(4, Bin::ShrS),
        O::I32ShrU => int_binary(4, Bin::ShrU),
        O::I32Rotl => int_binary(4, Bin::Rotl),
        O::I32Rotr => int_binary(4, Bin::Rotr),
        O::I64Add => int_binary(8, Bin::Add),
        O::I64Sub => int_binary(8, Bin::Sub),
        O::I64Mul => int_binary(8, Bin::Mul),
        O::I64DivS => int_binary(8, Bin::DivS),
        O::I64DivU => int_binary(8, Bin::DivU),
        O::I64RemS => int_binary(8, Bin::RemS),
        O::I64RemU => int_binary(8, Bin::RemU),
        O::I64And => int_binary(8, Bin::And),
        O::I64Or => int_binary(8, Bin::Or),
        O::I64Xor => int_binary(8, Bin::Xor),
        O::I64Shl => int_binary(8, Bin::Shl),
        O::I64ShrS => int_binary(8, Bin::ShrS),
        O::I64ShrU => int_binary(8, Bin::ShrU),
        O::I64Rotl => int_binary(8, Bin::Rotl),
        O::I64Rotr => int_binary(8, Bin::Rotr),
        O::F32Add => int_binary(4, Bin::FAdd),
        O::F32Sub => int_binary(4, Bin::FSub),
        O::F32Mul => int_binary(4, Bin::FMul),
        O::F32Div => int_binary(4, Bin::FDiv),
        O::F64Add => int_binary(8, Bin::FAdd),
        O::F64Sub => int_binary(8, Bin::FSub),
        O::F64Mul => int_binary(8, Bin::FMul),
        O::F64Div => int_binary(8, Bin::FDiv),

        O::F32Neg => int_unary(4, Un::Neg),
        O::F32Abs => int_unary(4, Un::Abs),
        O::F32Sqrt => int_unary(4, Un::Sqrt),
        O::F32Ceil => int_unary(4, Un::Ceil),
        O::F32Floor => int_unary(4, Un::Floor),
        O::F32Trunc => int_unary(4, Un::Trunc),
        O::F32Nearest => int_unary(4, Un::Nearest),
        O::F64Neg => int_unary(8, Un::Neg),
        O::F64Abs => int_unary(8, Un::Abs),
        O::F64Sqrt => int_unary(8, Un::Sqrt),
        O::F64Ceil => int_unary(8, Un::Ceil),
        O::F64Floor => int_unary(8, Un::Floor),
        O::F64Trunc => int_unary(8, Un::Trunc),
        O::F64Nearest => int_unary(8, Un::Nearest),

        O::I32WrapI64 => convert_sem(8, 4, Conv::Wrap),
        O::I64ExtendI32S => convert_sem(4, 8, Conv::SignExtend),
        O::I64ExtendI32U => convert_sem(4, 8, Conv::ZeroExtend),
        O::I32Extend8S => convert_sem(1, 4, Conv::SignExtend),
        O::I32Extend16S => convert_sem(2, 4, Conv::SignExtend),
        O::I64Extend8S => convert_sem(1, 8, Conv::SignExtend),
        O::I64Extend16S => convert_sem(2, 8, Conv::SignExtend),
        O::I64Extend32S => convert_sem(4, 8, Conv::SignExtend),

        O::I32TruncF32S | O::I32TruncF32U | O::I32TruncSatF32S | O::I32TruncSatF32U => {
            convert_sem(4, 4, Conv::FloatToInt)
        }
        O::I32TruncF64S | O::I32TruncF64U | O::I32TruncSatF64S | O::I32TruncSatF64U => {
            convert_sem(8, 4, Conv::FloatToInt)
        }
        O::I64TruncF32S | O::I64TruncF32U | O::I64TruncSatF32S | O::I64TruncSatF32U => {
            convert_sem(4, 8, Conv::FloatToInt)
        }
        O::I64TruncF64S | O::I64TruncF64U | O::I64TruncSatF64S | O::I64TruncSatF64U => {
            convert_sem(8, 8, Conv::FloatToInt)
        }
        O::F32ConvertI32S => convert_sem(4, 4, Conv::IntToFloat { signed: true }),
        O::F32ConvertI32U => convert_sem(4, 4, Conv::IntToFloat { signed: false }),
        O::F32ConvertI64S => convert_sem(8, 4, Conv::IntToFloat { signed: true }),
        O::F32ConvertI64U => convert_sem(8, 4, Conv::IntToFloat { signed: false }),
        O::F64ConvertI32S => convert_sem(4, 8, Conv::IntToFloat { signed: true }),
        O::F64ConvertI32U => convert_sem(4, 8, Conv::IntToFloat { signed: false }),
        O::F64ConvertI64S => convert_sem(8, 8, Conv::IntToFloat { signed: true }),
        O::F64ConvertI64U => convert_sem(8, 8, Conv::IntToFloat { signed: false }),
        O::F32DemoteF64 => convert_sem(8, 4, Conv::FloatResize),
        O::F64PromoteF32 => convert_sem(4, 8, Conv::FloatResize),

        // Reinterpretation only changes how the bits are read, and the slot already holds them
        O::I32ReinterpretF32
        | O::F32ReinterpretI32
        | O::I64ReinterpretF64
        | O::F64ReinterpretI64 => Sem::Nop,

        O::I32Load { memarg } => load(4, 4, Extend::None, memarg.offset),
        O::I64Load { memarg } => load(8, 8, Extend::None, memarg.offset),
        O::F32Load { memarg } => load(4, 4, Extend::None, memarg.offset),
        O::F64Load { memarg } => load(8, 8, Extend::None, memarg.offset),
        O::I32Load8S { memarg } => load(1, 4, Extend::Sign, memarg.offset),
        O::I32Load8U { memarg } => load(1, 4, Extend::Zero, memarg.offset),
        O::I32Load16S { memarg } => load(2, 4, Extend::Sign, memarg.offset),
        O::I32Load16U { memarg } => load(2, 4, Extend::Zero, memarg.offset),
        O::I64Load8S { memarg } => load(1, 8, Extend::Sign, memarg.offset),
        O::I64Load8U { memarg } => load(1, 8, Extend::Zero, memarg.offset),
        O::I64Load16S { memarg } => load(2, 8, Extend::Sign, memarg.offset),
        O::I64Load16U { memarg } => load(2, 8, Extend::Zero, memarg.offset),
        O::I64Load32S { memarg } => load(4, 8, Extend::Sign, memarg.offset),
        O::I64Load32U { memarg } => load(4, 8, Extend::Zero, memarg.offset),

        O::I32Store { memarg } => store(4, 4, memarg.offset),
        O::I64Store { memarg } => store(8, 8, memarg.offset),
        O::F32Store { memarg } => store(4, 4, memarg.offset),
        O::F64Store { memarg } => store(8, 8, memarg.offset),
        O::I32Store8 { memarg } => store(1, 4, memarg.offset),
        O::I32Store16 { memarg } => store(2, 4, memarg.offset),
        O::I64Store8 { memarg } => store(1, 8, memarg.offset),
        O::I64Store16 { memarg } => store(2, 8, memarg.offset),
        O::I64Store32 { memarg } => store(4, 8, memarg.offset),

        _ => Sem::Opaque,
    }
}

fn int_unary(size: usize, kind: Un) -> Sem {
    Sem::Unary { size, kind }
}

fn int_binary(size: usize, kind: Bin) -> Sem {
    Sem::Binary { size, kind }
}

fn int_compare(size: usize, kind: Cmp) -> Sem {
    Sem::Compare { size, kind }
}

fn convert_sem(from: usize, to: usize, kind: Conv) -> Sem {
    Sem::Convert { from, to, kind }
}

fn load(access: usize, result: usize, extend: Extend, offset: u64) -> Sem {
    Sem::Load {
        access,
        result,
        extend,
        offset,
    }
}

fn store(access: usize, value: usize, offset: u64) -> Sem {
    Sem::Store {
        access,
        value,
        offset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::insn::decode;

    fn sem(data: &[u8]) -> Sem {
        semantics(&decode(data).expect("decodes"))
    }

    #[test]
    fn constants_carry_their_encoded_bits() {
        assert_eq!(
            sem(&[0x41, 0x7f]),
            Sem::Const {
                size: 4,
                value: 0xffff_ffff
            }
        );
        assert_eq!(
            sem(&[0x42, 0x7f]),
            Sem::Const {
                size: 8,
                value: u64::MAX
            }
        );
        assert_eq!(
            sem(&[0x43, 0x00, 0x00, 0x80, 0x3f]),
            Sem::Const {
                size: 4,
                value: 0x3f80_0000
            }
        );
    }

    #[test]
    fn arithmetic_picks_the_right_width() {
        assert_eq!(
            sem(&[0x6a]),
            Sem::Binary {
                size: 4,
                kind: Bin::Add
            }
        );
        assert_eq!(
            sem(&[0x7c]),
            Sem::Binary {
                size: 8,
                kind: Bin::Add
            }
        );
        assert_eq!(
            sem(&[0x92]),
            Sem::Binary {
                size: 4,
                kind: Bin::FAdd
            }
        );
        assert_eq!(
            sem(&[0xa0]),
            Sem::Binary {
                size: 8,
                kind: Bin::FAdd
            }
        );
    }

    #[test]
    fn shifts_distinguish_arithmetic_from_logical() {
        assert_eq!(
            sem(&[0x75]),
            Sem::Binary {
                size: 4,
                kind: Bin::ShrS
            }
        );
        assert_eq!(
            sem(&[0x76]),
            Sem::Binary {
                size: 4,
                kind: Bin::ShrU
            }
        );
    }

    #[test]
    fn comparisons_keep_their_signedness() {
        assert_eq!(
            sem(&[0x48]),
            Sem::Compare {
                size: 4,
                kind: Cmp::LtS
            }
        );
        assert_eq!(
            sem(&[0x49]),
            Sem::Compare {
                size: 4,
                kind: Cmp::LtU
            }
        );
        assert_eq!(sem(&[0x45]), Sem::TestZero { size: 4 });
        assert_eq!(sem(&[0x50]), Sem::TestZero { size: 8 });
    }

    #[test]
    fn loads_extend_as_the_opcode_says() {
        assert_eq!(
            sem(&[0x2c, 0x00, 0x10]),
            Sem::Load {
                access: 1,
                result: 4,
                extend: Extend::Sign,
                offset: 16
            }
        );
        assert_eq!(
            sem(&[0x2d, 0x00, 0x00]),
            Sem::Load {
                access: 1,
                result: 4,
                extend: Extend::Zero,
                offset: 0
            }
        );
        assert_eq!(
            sem(&[0x29, 0x03, 0x00]),
            Sem::Load {
                access: 8,
                result: 8,
                extend: Extend::None,
                offset: 0
            }
        );
    }

    #[test]
    fn stores_narrow_to_the_access_width() {
        assert_eq!(
            sem(&[0x3a, 0x00, 0x04]),
            Sem::Store {
                access: 1,
                value: 4,
                offset: 4
            }
        );
        assert_eq!(
            sem(&[0x37, 0x03, 0x00]),
            Sem::Store {
                access: 8,
                value: 8,
                offset: 0
            }
        );
    }

    #[test]
    fn conversions() {
        assert_eq!(sem(&[0xa7]), convert_sem(8, 4, Conv::Wrap));
        assert_eq!(sem(&[0xac]), convert_sem(4, 8, Conv::SignExtend));
        assert_eq!(sem(&[0xad]), convert_sem(4, 8, Conv::ZeroExtend));
        assert_eq!(sem(&[0xc0]), convert_sem(1, 4, Conv::SignExtend));
        assert_eq!(sem(&[0xb6]), convert_sem(8, 4, Conv::FloatResize));
        assert_eq!(sem(&[0xbb]), convert_sem(4, 8, Conv::FloatResize));
    }

    #[test]
    fn no_op_semantics() {
        for encoding in [
            &[0xbc][..],
            &[0xbd],
            &[0xbe],
            &[0xbf],
            &[0x0b],
            &[0x02, 0x40],
            &[0x03, 0x40],
        ] {
            assert_eq!(sem(encoding), Sem::Nop, "{encoding:02x?}");
        }
    }

    #[test]
    fn variable_access() {
        assert_eq!(sem(&[0x20, 0x03]), Sem::LocalGet(3));
        assert_eq!(sem(&[0x21, 0x03]), Sem::LocalSet(3));
        assert_eq!(sem(&[0x22, 0x03]), Sem::LocalTee(3));
        assert_eq!(sem(&[0x23, 0x01]), Sem::GlobalGet(1));
        assert_eq!(sem(&[0x24, 0x01]), Sem::GlobalSet(1));
    }

    #[test]
    fn terminators() {
        assert_eq!(sem(&[0x00]), Sem::Trap);
        assert_eq!(sem(&[0x0f]), Sem::Return);
        assert_eq!(sem(&[0x01]), Sem::Nop);
        assert_eq!(sem(&[0x1a]), Sem::Drop);
    }

    #[test]
    fn parameters_sit_where_the_caller_left_them() {
        let frame = Frame {
            params: 3,
            locals: 5,
            results: 1,
            result: Some(ValueKind::I32),
            entry: false,
        };
        assert_eq!(frame.offset(0), 0);
        assert_eq!(frame.offset(1), SLOT as i64);
        assert_eq!(frame.offset(2), 2 * SLOT as i64);
        // The two the body declared go below the base, in the space the prologue reserves
        assert_eq!(frame.offset(3), -(SLOT as i64));
        assert_eq!(frame.offset(4), -2 * (SLOT as i64));
        assert_eq!(frame.reserved(), 2 * SLOT as i64);
    }

    #[test]
    fn a_frame_with_no_parameters_reserves_all_of_its_locals() {
        let frame = Frame {
            params: 0,
            locals: 2,
            results: 0,
            result: None,
            entry: true,
        };
        assert_eq!(frame.offset(0), -(SLOT as i64));
        assert_eq!(frame.offset(1), -2 * (SLOT as i64));
        assert_eq!(frame.reserved(), 2 * SLOT as i64);
        assert_eq!(Frame::default().reserved(), 0);
    }

    #[test]
    fn conditional_branches_still_consume_their_condition() {
        assert_eq!(sem(&[0x04, 0x40]), Sem::CondBranch);
        assert_eq!(sem(&[0x0d, 0x00]), Sem::CondBranch);
    }

    #[test]
    fn every_operator_leaves_the_stack_accounted_for() {
        let mut unaccounted = Vec::new();
        let mut modelled = 0;

        for prefix in 0u8..=0xff {
            for second in 0u8..=0xff {
                let mut buffer = [0u8; crate::insn::MAX_INSTR_LEN];
                buffer[0] = prefix;
                buffer[1] = second;
                let Some(insn) = decode(&buffer) else {
                    continue;
                };

                let semantics = semantics(&insn);
                if semantics != Sem::Opaque {
                    modelled += 1;
                    continue;
                }
                if insn.arity().is_some() || !insn.flow().falls_through() {
                    continue;
                }
                unaccounted.push(insn.mnemonic());
            }
        }

        unaccounted.sort();
        unaccounted.dedup();
        assert_eq!(
            unaccounted,
            [
                "call",
                "call_indirect",
                "call_ref",
                "catch",
                "catch_all",
                "cont.bind",
                "resume",
                "resume_throw",
                "resume_throw_ref",
                "struct.new",
                "struct.new_desc",
                "suspend",
                "switch",
            ]
        );
        assert!(modelled > 1000, "only {modelled} encodings modelled");
    }

    #[test]
    fn a_resolved_call_is_what_makes_a_call_measurable() {
        let call = decode(&[0x10, 0x00]).expect("call decodes");
        assert_eq!(stack_effect(&call, None), None);

        let takes_two = Resolved::Call(Call {
            target: Some(0x100),
            arity: Arity { pops: 2, pushes: 1 },
            params: vec![ValueKind::I32, ValueKind::I32],
            result: Some(ValueKind::I32),
        });
        assert_eq!(stack_effect(&call, Some(&takes_two)), Some(1));

        let aggregate = Resolved::Aggregate(Arity { pops: 3, pushes: 1 });
        assert_eq!(stack_effect(&call, Some(&aggregate)), Some(2));

        // A tail call replaces the frame, so nothing is left to measure
        let tail = decode(&[0x12, 0x00]).expect("return_call decodes");
        assert_eq!(stack_effect(&tail, Some(&takes_two)), None);
    }

    #[test]
    fn an_arm_has_no_fallthrough_stack_effect() {
        for encoding in [
            &[0x05][..],   // else
            &[0x07, 0x00], // catch
            &[0x19],       // catch_all
        ] {
            let insn = decode(encoding).expect("arm decodes");
            assert_eq!(insn.flow(), Flow::Arm, "{}", insn.mnemonic());
            assert_eq!(stack_effect(&insn, None), None, "{}", insn.mnemonic());

            // What a handler is handed still reaches the lifter, but it is not a stack effect
            let carried = Resolved::Aggregate(Arity { pops: 0, pushes: 1 });
            assert_eq!(
                stack_effect(&insn, Some(&carried)),
                None,
                "{}",
                insn.mnemonic()
            );
        }
    }

    #[test]
    fn unmodelled_operators_are_opaque() {
        for encoding in [
            &[0xfd, 0x6e][..], // i8x16.add
            &[0x10, 0x00],     // call
            &[0x1b],           // select
            &[0xfc, 0x0a, 0x00, 0x00],
        ] {
            assert_eq!(sem(encoding), Sem::Opaque, "{encoding:02x?}");
        }
    }
}
