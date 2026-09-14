//! Whole-function control flow recovery
//!
//! A branch names a label relative to its enclosing block, so where it goes is the end of the
//! frame that label names, or the top of the body for a `loop`

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{LazyLock, RwLock};

use wasmparser::{BlockType, Operator};

use crate::ViewId;
use crate::insn;
use crate::lift;
use crate::module::Module;

/// Bodies whose operand stack went below empty, by where their code starts
static UNBALANCED: LazyLock<RwLock<BTreeSet<(ViewId, u64)>>> = LazyLock::new(Default::default);

pub fn note_unbalanced(view: ViewId, start: u64) {
    let mut unbalanced = match UNBALANCED.write() {
        Ok(unbalanced) => unbalanced,
        Err(poisoned) => poisoned.into_inner(),
    };
    unbalanced.insert((view, start));
}

pub fn is_unbalanced(view: ViewId, start: u64) -> bool {
    let unbalanced = match UNBALANCED.read() {
        Ok(unbalanced) => unbalanced,
        Err(poisoned) => poisoned.into_inner(),
    };
    unbalanced.contains(&(view, start))
}

/// Recovery needs a whole function, but the core asks one instruction at a time with no context,
/// so analysis leaves the answers here
///
/// Keyed by address before file, so both looking one up in a known file and asking which files
/// know an address at all stay logarithmic
static RECOVERED: LazyLock<RwLock<BTreeMap<(u64, ViewId), Recovered>>> =
    LazyLock::new(Default::default);

pub fn install(view: ViewId, flow: &ControlFlow) {
    let mut recovered = match RECOVERED.write() {
        Ok(recovered) => recovered,
        // The map is a cache, so a poisoned lock is better carried on with than propagated
        Err(poisoned) => poisoned.into_inner(),
    };

    let stale: Vec<(u64, ViewId)> = recovered
        .range((flow.start, ViewId::MIN)..(flow.end, ViewId::MIN))
        .map(|(key, _)| *key)
        .filter(|(_, owner)| *owner == view)
        .collect();
    for key in stale {
        recovered.remove(&key);
    }
    for (addr, terminator) in &flow.terminators {
        recovered.insert(
            (*addr, view),
            Recovered {
                terminator: terminator.clone(),
                unwinds: flow.unwinds.get(addr).cloned().unwrap_or_default(),
            },
        );
    }
}

pub fn lookup(view: ViewId, addr: u64) -> Option<Recovered> {
    let recovered = match RECOVERED.read() {
        Ok(recovered) => recovered,
        Err(poisoned) => poisoned.into_inner(),
    };
    recovered.get(&(addr, view)).cloned()
}

/// For the callbacks handed an address and no file: answering only when every file that knows the
/// address agrees is exact for one open file and silent rather than wrong for two
pub fn lookup_anywhere(addr: u64) -> Option<Recovered> {
    let recovered = match RECOVERED.read() {
        Ok(recovered) => recovered,
        Err(poisoned) => poisoned.into_inner(),
    };

    let mut found: Option<&Recovered> = None;
    for (_, entry) in recovered.range((addr, ViewId::MIN)..=(addr, ViewId::MAX)) {
        match found {
            Some(seen) if seen != entry => return None,
            _ => found = Some(entry),
        }
    }
    found.cloned()
}

/// How much of a view to read looking for the end of a body, in case the bytes never balance
pub const MAX_BODY_LEN: usize = 8 << 20;

/// Instructions absent from [`ControlFlow::terminators`] fall through
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    Jump(u64),
    Branch {
        taken: u64,
        not_taken: u64,
    },
    Table {
        /// `None` for an entry naming the outermost label, which leaves the function
        targets: Vec<Option<u64>>,
        default: Option<u64>,
    },
    Return,
    ConditionalReturn {
        not_taken: u64,
    },
    /// Traps and throws, which have no successor inside the function
    Halt,
    /// A target that could not be recovered, reported as such rather than guessed at
    Unresolved,
}

/// Branching out of a block unwinds it, and leaving that out has the two edges meet again holding
/// different stack pointers, which puts every slot addressed off `sp` past that point out by the
/// difference
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Unwind {
    /// Values carried to the label, which move down over the discarded slots
    pub keep: u32,
    /// Slots discarded from beneath them
    pub drop: u32,
}

impl Unwind {
    pub fn is_empty(self) -> bool {
        self.drop == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovered {
    pub terminator: Terminator,
    /// One per edge that can unwind: a branch's taken edge, or every table entry then its default
    pub unwinds: Vec<Unwind>,
}

impl Recovered {
    pub fn unwind(&self, index: usize) -> Unwind {
        self.unwinds.get(index).copied().unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    Unconditional(u64),
    True(u64),
    False(u64),
    FunctionReturn,
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub start: u64,
    /// One past the last byte, as the core means it
    pub end: u64,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ControlFlow {
    pub start: u64,
    /// One past the last byte of the body
    pub end: u64,
    pub terminators: BTreeMap<u64, Terminator>,
    /// What each branch discards, for the ones that discard anything
    pub unwinds: BTreeMap<u64, Vec<Unwind>>,
    /// Instruction boundaries in address order
    pub instructions: Vec<(u64, usize)>,
    /// Height before each instruction, parallel to `instructions`, `None` where the walk could not
    /// know it; every [`Unwind`] is derived from this
    pub heights: Vec<Option<i64>>,
    /// Set on an undecodable byte, or on running out of input before the body closed
    pub truncated: bool,
    /// No module an engine would load can go below empty, so the file was built by hand and its
    /// declared signature is a claim rather than a fact
    pub underflow: bool,
}

impl ControlFlow {
    fn next_address(&self, addr: u64) -> Option<u64> {
        let index = self
            .instructions
            .binary_search_by_key(&addr, |(at, _)| *at)
            .ok()?;
        let (at, len) = self.instructions[index];
        Some(at + len as u64)
    }

    fn contains(&self, addr: u64) -> bool {
        (self.start..self.end).contains(&addr)
            && self
                .instructions
                .binary_search_by_key(&addr, |(at, _)| *at)
                .is_ok()
    }

    /// The core makes functions mid-body, and recovering from there cannot work: a `br 4` needs
    /// five enclosing frames and a mid-body walk has one, so walk the whole body then cut
    pub fn restricted_to(mut self, start: u64) -> Self {
        if start <= self.start {
            return self;
        }

        let kept = self
            .instructions
            .iter()
            .position(|(at, _)| *at >= start)
            .unwrap_or(self.instructions.len());
        self.instructions.drain(..kept);
        self.heights.drain(..kept.min(self.heights.len()));
        self.terminators.retain(|at, _| *at >= start);
        self.unwinds.retain(|at, _| *at >= start);
        self.start = start;
        self
    }

    /// A block starts at the entry, at any branch target, and after any instruction that does not
    /// fall into the next one
    pub fn blocks(&self) -> Vec<Block> {
        if self.instructions.is_empty() {
            return Vec::new();
        }

        let mut leaders = BTreeSet::from([self.start]);
        for (addr, terminator) in &self.terminators {
            for target in terminator.targets() {
                if self.contains(target) {
                    leaders.insert(target);
                }
            }
            if let Some(next) = self.next_address(*addr)
                && self.contains(next)
            {
                leaders.insert(next);
            }
        }

        let boundaries: Vec<u64> = leaders.into_iter().collect();

        // Which instruction ends each block, in one pass: rescanning per block is quadratic, and
        // a large body then takes seconds on the analysis thread
        let mut last = vec![None; boundaries.len()];
        let mut block = 0usize;
        for (addr, _) in &self.instructions {
            while block + 1 < boundaries.len() && *addr >= boundaries[block + 1] {
                block += 1;
            }
            if *addr >= boundaries[block] {
                last[block] = Some(*addr);
            }
        }

        boundaries
            .iter()
            .enumerate()
            .map(|(index, &start)| Block {
                start,
                end: boundaries.get(index + 1).copied().unwrap_or(self.end),
                edges: last[index].map_or_else(Vec::new, |at| self.edges_leaving(at)),
            })
            .collect()
    }

    fn edges_leaving(&self, last: u64) -> Vec<Edge> {
        let Some(terminator) = self.terminators.get(&last) else {
            return match self.next_address(last) {
                Some(next) if self.contains(next) => vec![Edge::Unconditional(next)],
                _ => Vec::new(),
            };
        };

        // A target outside the recovered body has no block to point at, so saying it is
        // unresolved beats naming an address the core will not find
        let mut edges: Vec<Edge> = Vec::new();
        for edge in terminator.edges() {
            let edge = match edge {
                Edge::Unconditional(to) | Edge::True(to) | Edge::False(to)
                    if !self.contains(to) =>
                {
                    Edge::Unresolved
                }
                edge => edge,
            };
            if !edges.contains(&edge) {
                edges.push(edge);
            }
        }
        edges
    }
}

impl Terminator {
    fn targets(&self) -> Vec<u64> {
        match self {
            Self::Jump(target) => vec![*target],
            Self::Branch { taken, not_taken } => vec![*taken, *not_taken],
            Self::Table { targets, default } => targets
                .iter()
                .chain([default])
                .filter_map(|to| *to)
                .collect(),
            Self::ConditionalReturn { not_taken } => vec![*not_taken],
            Self::Return | Self::Halt | Self::Unresolved => Vec::new(),
        }
    }

    fn edges(&self) -> Vec<Edge> {
        match self {
            Self::Jump(target) => vec![Edge::Unconditional(*target)],
            Self::Branch { taken, not_taken } => {
                vec![Edge::True(*taken), Edge::False(*not_taken)]
            }
            Self::Table { targets, default } => {
                let mut seen = BTreeSet::new();
                let mut leaves = false;
                let mut edges = Vec::new();
                for target in targets.iter().chain([default]) {
                    match target {
                        Some(to) if seen.insert(*to) => edges.push(Edge::Unconditional(*to)),
                        Some(_) => {}
                        None => leaves = true,
                    }
                }
                if leaves {
                    edges.push(Edge::FunctionReturn);
                }
                edges
            }
            Self::ConditionalReturn { not_taken } => {
                vec![Edge::FunctionReturn, Edge::False(*not_taken)]
            }
            Self::Return => vec![Edge::FunctionReturn],
            Self::Halt => Vec::new(),
            Self::Unresolved => vec![Edge::Unresolved],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameKind {
    /// The implicit block around the whole body, so branching to it is a return
    Function,
    Block,
    Loop,
    If,
    Try,
}

#[derive(Debug)]
struct Frame {
    kind: FrameKind,
    /// First instruction inside the frame, where a `loop` label points
    body: u64,
    /// Just past the matching `end`, once it has been seen
    end: Option<u64>,
    /// Just past the `else`, where a false `if` condition lands
    otherwise: Option<u64>,
    /// Height the label unwinds to, before the values it receives are put back
    base: Option<i64>,
    /// A `loop` label restarts the frame rather than leaving it, so it carries the parameters
    label_arity: u32,
    params: u32,
    results: u32,
}

impl Frame {
    /// Where the frame's `end` falls through to whatever encloses it
    fn end_height(&self) -> Option<i64> {
        self.base.map(|base| base + i64::from(self.results))
    }

    /// Where an arm begins, holding the frame's parameters again
    fn arm_height(&self) -> Option<i64> {
        self.base.map(|base| base + i64::from(self.params))
    }
}

/// A type index names a signature declared elsewhere, so a walk with no module says so rather than
/// guessing: a wrong height has every branch out of that frame unwind by the wrong amount
fn block_arity(blockty: &BlockType, module: Option<&Module>) -> Option<(u32, u32)> {
    match blockty {
        BlockType::Empty => Some((0, 0)),
        BlockType::Type(_) => Some((0, 1)),
        BlockType::FuncType(index) => module
            .and_then(|module| module.type_arity(*index))
            .map(|arity| (arity.pops, arity.pushes)),
    }
}

fn open_frame(
    frames: &mut Vec<Frame>,
    kind: FrameKind,
    body: u64,
    height: Option<i64>,
    blockty: &BlockType,
    module: Option<&Module>,
) -> usize {
    let arity = block_arity(blockty, module);
    let (params, results) = arity.unwrap_or((0, 0));
    frames.push(Frame {
        kind,
        body,
        end: None,
        otherwise: None,
        // The frame's parameters are already on the stack, so its label unwinds to below them, and
        // a block declaring more than the stack holds has no base rather than one below the frame
        base: arity
            .and(height)
            .map(|h| h - i64::from(params))
            .filter(|base| *base >= 0),
        label_arity: if kind == FrameKind::Loop {
            params
        } else {
            results
        },
        params,
        results,
    });
    frames.len() - 1
}

/// A branch recorded before the frame it names has closed
#[derive(Debug)]
enum Pending {
    Jump(usize),
    Branch {
        frame: usize,
        not_taken: u64,
    },
    /// An `if`, whose false edge lands after the `else` rather than at the frame end
    Conditional {
        frame: usize,
        taken: u64,
    },
    /// An `else`, only reached by falling out of the then-arm
    SkipElse(usize),
    Table {
        frames: Vec<usize>,
        default: usize,
    },
}

/// The walk stops at the `end` closing the implicit function block, so `code` may run past it
///
/// `module` resolves block types and calls; without one the labels still come out right and only
/// the [`Unwind`]s are lost
pub fn recover(code: &[u8], start: u64, module: Option<&Module>) -> ControlFlow {
    let mut flow = ControlFlow {
        start,
        end: start,
        ..Default::default()
    };

    let mut frames = vec![Frame {
        kind: FrameKind::Function,
        body: start,
        end: None,
        otherwise: None,
        // A function is entered with an empty operand stack whatever its parameters, since those
        // are locals here, in a frame of their own
        base: Some(0),
        label_arity: 0,
        params: 0,
        results: 0,
    }];
    let mut open = vec![0usize];
    let mut pending: Vec<(u64, Option<i64>, Pending)> = Vec::new();
    let mut offset = 0usize;

    // `None` past a point the walk could not account for; every frame boundary restores it, so an
    // unknown operator costs the rest of its block rather than the rest of the body
    let mut height: Option<i64> = Some(0);

    while offset < code.len() {
        // Uncapped: stopping at a `br_table` too long for the core to render would leave every
        // branch after it in the body with no target
        let Some(insn) = insn::decode_any(&code[offset..]) else {
            flow.truncated = true;
            break;
        };

        let addr = start + offset as u64;
        let next = addr + insn.len as u64;
        flow.instructions.push((addr, insn.len));
        flow.heights.push(height);
        flow.underflow |= height.is_some_and(|height| height < 0);
        flow.end = next;
        offset += insn.len;

        let frame_at = |open: &[usize], depth: u32| -> Option<usize> {
            let depth = usize::try_from(depth).ok()?;
            open.len().checked_sub(depth + 1).map(|index| open[index])
        };

        match &insn.op {
            Operator::Block { blockty }
            | Operator::Loop { blockty }
            | Operator::If { blockty }
            | Operator::Try { blockty } => {
                let kind = match insn.op {
                    Operator::Loop { .. } => FrameKind::Loop,
                    Operator::If { .. } => FrameKind::If,
                    Operator::Try { .. } => FrameKind::Try,
                    _ => FrameKind::Block,
                };
                // An `if` consumes its condition before the frame is entered
                if kind == FrameKind::If {
                    height = height.map(|h| h - 1);
                }
                let frame = open_frame(&mut frames, kind, next, height, blockty, module);
                flow.underflow |= frames[frame].base.is_none() && height.is_some();
                open.push(frame);
                if kind == FrameKind::If {
                    pending.push((addr, height, Pending::Conditional { frame, taken: next }));
                }
            }
            Operator::TryTable { try_table } => {
                let frame = open_frame(
                    &mut frames,
                    FrameKind::Try,
                    next,
                    height,
                    &try_table.ty,
                    module,
                );
                open.push(frame);
            }
            Operator::Else => {
                if let Some(&frame) = open.last() {
                    frames[frame].otherwise = Some(next);
                    pending.push((addr, height, Pending::SkipElse(frame)));
                    height = frames[frame].arm_height();
                }
            }
            // A legacy handler closes the protected body the way an `else` closes a then-arm
            Operator::Catch { .. } | Operator::CatchAll => {
                if let Some(&frame) = open.last() {
                    pending.push((addr, height, Pending::SkipElse(frame)));
                    // A handler holds what its tag carries, and `catch_all` names no tag
                    let carried = match &insn.op {
                        Operator::Catch { tag_index } => {
                            module.and_then(|module| module.tag_arity(*tag_index))
                        }
                        _ => Some(0),
                    };
                    height = frames[frame]
                        .base
                        .zip(carried)
                        .map(|(base, carried)| base + i64::from(carried));
                }
            }
            Operator::End | Operator::Delegate { .. } => {
                if let Some(frame) = open.pop() {
                    frames[frame].end = Some(next);
                    height = frames[frame].end_height();
                    if frames[frame].kind == FrameKind::Function {
                        flow.terminators.insert(addr, Terminator::Return);
                        break;
                    }
                } else {
                    // More `end`s than openers, so the bytes are not a function body
                    flow.truncated = true;
                    break;
                }
            }
            Operator::Br { relative_depth } => {
                match frame_at(&open, *relative_depth) {
                    Some(frame) => pending.push((addr, height, Pending::Jump(frame))),
                    None => {
                        flow.terminators.insert(addr, Terminator::Unresolved);
                    }
                }
                height = None;
            }
            // Every conditional branch names its label the same way; what it tests is the
            // lifter's problem rather than the graph's
            Operator::BrIf { relative_depth }
            | Operator::BrOnNull { relative_depth }
            | Operator::BrOnNonNull { relative_depth }
            | Operator::BrOnCast { relative_depth, .. }
            | Operator::BrOnCastFail { relative_depth, .. }
            | Operator::BrOnCastDescEq { relative_depth, .. }
            | Operator::BrOnCastDescEqFail { relative_depth, .. } => {
                // The two edges leave different amounts behind, so they count separately
                let taken = height.map(|h| h - lift::taken_pops(&insn.op));
                match frame_at(&open, *relative_depth) {
                    Some(frame) => pending.push((
                        addr,
                        taken,
                        Pending::Branch {
                            frame,
                            not_taken: next,
                        },
                    )),
                    None => {
                        flow.terminators.insert(addr, Terminator::Unresolved);
                    }
                }
                height = height.and_then(|h| lift::stack_effect(&insn, None).map(|net| h - net));
            }
            Operator::BrTable { targets } => {
                let depths: Vec<u32> = targets
                    .targets()
                    .filter_map(Result::ok)
                    .chain([targets.default()])
                    .collect();
                let resolved: Option<Vec<usize>> =
                    depths.iter().map(|depth| frame_at(&open, *depth)).collect();

                height = height.map(|h| h - 1);
                match resolved {
                    Some(mut frames) if !frames.is_empty() => {
                        let default = frames.pop().expect("the default is always present");
                        pending.push((addr, height, Pending::Table { frames, default }));
                    }
                    _ => {
                        flow.terminators.insert(addr, Terminator::Unresolved);
                    }
                }
                height = None;
            }
            Operator::Return
            | Operator::ReturnCall { .. }
            | Operator::ReturnCallIndirect { .. }
            | Operator::ReturnCallRef { .. } => {
                flow.terminators.insert(addr, Terminator::Return);
                height = None;
            }
            Operator::Unreachable
            | Operator::Throw { .. }
            | Operator::ThrowRef
            | Operator::Rethrow { .. } => {
                flow.terminators.insert(addr, Terminator::Halt);
                height = None;
            }
            // Everything else moves the stack by its own arity, which the lifter answers for so
            // there is one answer rather than two
            _ => {
                height = height.and_then(|h| {
                    let resolved = module.and_then(|module| module.resolve(&insn.op));
                    lift::stack_effect(&insn, resolved.as_ref()).map(|net| h - net)
                });
            }
        }
    }

    if open
        .iter()
        .any(|frame| frames[*frame].kind == FrameKind::Function)
    {
        flow.truncated = true;
    }

    resolve(&mut flow, &frames, pending);
    leaving_the_body_returns(&mut flow);
    flow
}

/// The outermost frame ends one byte past the body, so a label naming it is no address in this
/// function; falling out of that frame is a return, so that is what it becomes
fn leaving_the_body_returns(flow: &mut ControlFlow) {
    let end = flow.end;
    let outside = |target: u64| target >= end;

    for terminator in flow.terminators.values_mut() {
        match terminator {
            Terminator::Jump(target) if outside(*target) => *terminator = Terminator::Return,
            Terminator::Branch { taken, not_taken } => {
                match (outside(*taken), outside(*not_taken)) {
                    (true, false) => {
                        *terminator = Terminator::ConditionalReturn {
                            not_taken: *not_taken,
                        }
                    }
                    // Both edges leave, so nothing after this runs either way
                    (true, true) => *terminator = Terminator::Return,
                    _ => {}
                }
            }
            Terminator::ConditionalReturn { not_taken } if outside(*not_taken) => {
                *terminator = Terminator::Return
            }
            Terminator::Table { targets, default } => {
                for target in targets.iter_mut().chain(std::iter::once(default)) {
                    if target.is_some_and(outside) {
                        *target = None;
                    }
                }
            }
            _ => {}
        }
    }
}

/// Turns the recorded label references into addresses, now that every frame's end is known
fn resolve(flow: &mut ControlFlow, frames: &[Frame], pending: Vec<(u64, Option<i64>, Pending)>) {
    // A `loop` label restarts the body; everything else lands after its matching `end`
    let target = |index: usize| -> Option<u64> {
        let frame = &frames[index];
        match frame.kind {
            FrameKind::Loop => Some(frame.body),
            FrameKind::Function => None,
            _ => frame.end,
        }
    };
    let is_function = |index: usize| frames[index].kind == FrameKind::Function;

    // What a branch at `height` discards to land in `index`, and nothing where either height is
    // unknown or the two disagree, since a guessed unwind corrupts the stack rather than repairs it
    let unwind = |index: usize, height: Option<i64>| -> Unwind {
        let keep = frames[index].label_arity;
        let Some(depth) = height.zip(frames[index].base).map(|(at, base)| at - base) else {
            return Unwind::default();
        };
        u32::try_from(depth - i64::from(keep))
            .map(|drop| Unwind { keep, drop })
            .unwrap_or_default()
    };

    for (addr, height, branch) in pending {
        let mut unwinds = Vec::new();
        match &branch {
            // Only a real branch leaves a frame early; an `if` enters its own frame either way,
            // and an arm stands exactly where its target expects it
            Pending::Jump(frame) | Pending::Branch { frame, .. } if !is_function(*frame) => {
                unwinds.push(unwind(*frame, height));
            }
            Pending::Table {
                frames: entries,
                default,
            } => {
                unwinds.extend(entries.iter().chain([default]).map(|frame| {
                    if is_function(*frame) {
                        Unwind::default()
                    } else {
                        unwind(*frame, height)
                    }
                }));
            }
            _ => {}
        }
        if unwinds.iter().any(|unwind| !unwind.is_empty()) {
            flow.unwinds.insert(addr, unwinds);
        }

        let terminator = match branch {
            Pending::Jump(frame) if is_function(frame) => Terminator::Return,
            Pending::Jump(frame) => match target(frame) {
                Some(to) => Terminator::Jump(to),
                None => Terminator::Unresolved,
            },
            // Branching to the outermost label leaves the function, but the fallthrough is still
            // a real edge
            Pending::Branch { frame, not_taken } if is_function(frame) => {
                Terminator::ConditionalReturn { not_taken }
            }
            Pending::Branch { frame, not_taken } => match target(frame) {
                Some(taken) => Terminator::Branch { taken, not_taken },
                None => Terminator::Unresolved,
            },
            Pending::Conditional { frame, taken } => {
                match frames[frame].otherwise.or(frames[frame].end) {
                    Some(not_taken) => Terminator::Branch { taken, not_taken },
                    None => Terminator::Unresolved,
                }
            }
            Pending::SkipElse(frame) => match frames[frame].end {
                Some(to) => Terminator::Jump(to),
                None => Terminator::Unresolved,
            },
            Pending::Table {
                frames: entries,
                default,
            } => {
                // An entry naming the outermost label returns; one that fails to resolve makes
                // the whole table unusable, since no index can be marked as going nowhere
                let entry = |frame: usize| -> Option<Option<u64>> {
                    if is_function(frame) {
                        Some(None)
                    } else {
                        target(frame).map(Some)
                    }
                };

                let targets: Option<Vec<Option<u64>>> =
                    entries.iter().map(|frame| entry(*frame)).collect();
                match (targets, entry(default)) {
                    (Some(targets), Some(default)) => Terminator::Table { targets, default },
                    _ => Terminator::Unresolved,
                }
            }
        };

        flow.terminators.insert(addr, terminator);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recover(code: &[u8], start: u64) -> ControlFlow {
        super::recover(code, start, None)
    }

    #[test]
    fn a_branch_out_of_a_block_lands_after_its_end() {
        let flow = recover(&[0x02, 0x40, 0x0c, 0x00, 0x0b, 0x0b], 0x100);

        assert_eq!(flow.end, 0x106);
        assert!(!flow.truncated);
        assert_eq!(flow.terminators[&0x102], Terminator::Jump(0x105));
        assert_eq!(flow.terminators[&0x105], Terminator::Return);
    }

    #[test]
    fn a_branch_into_a_loop_lands_at_the_top_of_its_body() {
        let flow = recover(&[0x03, 0x40, 0x0c, 0x00, 0x0b, 0x0b], 0x100);

        assert_eq!(flow.terminators[&0x102], Terminator::Jump(0x102));
    }

    #[test]
    fn label_depth_counts_from_the_innermost_frame() {
        // block { block { br 1 } } end
        let code = [0x02, 0x40, 0x02, 0x40, 0x0c, 0x01, 0x0b, 0x0b, 0x0b];
        let flow = recover(&code, 0);

        // The outer block ends at 8, so its label is the byte after
        assert_eq!(flow.terminators[&4], Terminator::Jump(8));

        let inner = recover(&[0x02, 0x40, 0x02, 0x40, 0x0c, 0x00, 0x0b, 0x0b, 0x0b], 0);
        assert_eq!(inner.terminators[&4], Terminator::Jump(7));
    }

    #[test]
    fn a_branch_to_the_function_label_is_a_return() {
        let flow = recover(&[0x0c, 0x00, 0x0b], 0);
        assert_eq!(flow.terminators[&0], Terminator::Return);
    }

    #[test]
    fn br_if_keeps_its_fallthrough() {
        // block { br_if 0; nop } end
        let flow = recover(&[0x02, 0x40, 0x0d, 0x00, 0x01, 0x0b, 0x0b], 0);

        assert_eq!(
            flow.terminators[&2],
            Terminator::Branch {
                taken: 6,
                not_taken: 4
            }
        );
    }

    #[test]
    fn a_conditional_branch_to_the_function_label_keeps_both_edges() {
        // br_if 0; nop; end
        let flow = recover(&[0x0d, 0x00, 0x01, 0x0b], 0);

        assert_eq!(
            flow.terminators[&0],
            Terminator::ConditionalReturn { not_taken: 2 }
        );
        assert_eq!(
            flow.terminators[&0].edges(),
            [Edge::FunctionReturn, Edge::False(2)]
        );
    }

    #[test]
    fn every_conditional_branch_resolves_its_label() {
        // Each is `block { <branch> 0; ... } end`, with the branch at offset 2
        let bodies: [&[u8]; 4] = [
            &[0x02, 0x40, 0xd5, 0x00, 0x1a, 0x0b, 0x0b],
            &[0x02, 0x40, 0xd6, 0x00, 0x01, 0x0b, 0x0b],
            &[
                0x02, 0x40, 0xfb, 0x18, 0x01, 0x00, 0x6e, 0x6e, 0x1a, 0x0b, 0x0b,
            ],
            &[
                0x02, 0x40, 0xfb, 0x19, 0x01, 0x00, 0x6e, 0x6e, 0x1a, 0x0b, 0x0b,
            ],
        ];

        for body in bodies {
            let flow = recover(body, 0);
            let branch = insn::decode(&body[2..]).expect("the branch decodes");

            assert!(!flow.truncated, "{}", branch.mnemonic());
            assert_eq!(
                flow.terminators[&2],
                Terminator::Branch {
                    taken: body.len() as u64 - 1,
                    not_taken: 2 + branch.len as u64,
                },
                "{}",
                branch.mnemonic()
            );
        }
    }

    #[test]
    fn an_oversized_branch_table_does_not_end_the_walk() {
        let targets = 300u32;
        let mut code = vec![0x02, 0x40, 0x0e]; // block; br_table
        code.extend([0xac, 0x02]); // 300, as LEB128
        code.extend(std::iter::repeat_n(0x00, targets as usize + 1)); // targets, then default
        let table = code.len() - 2;
        code.extend([0x01, 0x0b, 0x0b]); // nop; end; end

        assert!(
            table > crate::insn::MAX_INSTR_LEN,
            "the table is meant to be too long for the core, at {table} bytes"
        );
        assert!(
            crate::insn::decode(&code[2..]).is_none(),
            "the core declines it"
        );

        let flow = recover(&code, 0);
        assert!(!flow.truncated, "{flow:?}");
        assert_eq!(flow.end, code.len() as u64);
        assert_eq!(
            flow.terminators[&(code.len() as u64 - 1)],
            Terminator::Return
        );
    }

    #[test]
    fn legacy_exception_handling_closes_its_frames() {
        // try; nop; catch 0; nop; end; end
        let flow = recover(&[0x06, 0x40, 0x01, 0x07, 0x00, 0x01, 0x0b, 0x0b], 0);
        assert!(!flow.truncated);
        assert_eq!(
            flow.terminators[&3],
            Terminator::Jump(7),
            "a handler is only reached by a throw, so falling into it skips to the end"
        );
        assert_eq!(flow.terminators[&7], Terminator::Return);

        // try; nop; catch_all; end; end
        let flow = recover(&[0x06, 0x40, 0x01, 0x19, 0x01, 0x0b, 0x0b], 0);
        assert!(!flow.truncated);
        assert_eq!(flow.terminators[&3], Terminator::Jump(6));

        // block { try { nop } delegate 0 } end, where `delegate` closes the try as `end` does
        let flow = recover(&[0x02, 0x40, 0x06, 0x40, 0x01, 0x18, 0x00, 0x0b, 0x0b], 0);
        assert!(!flow.truncated);
        assert_eq!(flow.terminators[&8], Terminator::Return);

        // block { try { br 1 } catch 0 { nop } end } end
        let code = [
            0x02, 0x40, 0x06, 0x40, 0x0c, 0x01, 0x07, 0x00, 0x01, 0x0b, 0x0b, 0x0b,
        ];
        let flow = recover(&code, 0);
        assert!(!flow.truncated);
        assert_eq!(
            flow.terminators[&4],
            Terminator::Jump(11),
            "a branch out of a try leaves the block enclosing it"
        );
    }

    #[test]
    fn an_if_with_an_else_splits_three_ways() {
        // if { nop } else { nop } end
        let flow = recover(&[0x04, 0x40, 0x01, 0x05, 0x01, 0x0b, 0x0b], 0);

        assert_eq!(
            flow.terminators[&0],
            Terminator::Branch {
                taken: 2,
                not_taken: 4
            }
        );
        assert_eq!(
            flow.terminators[&3],
            Terminator::Jump(6),
            "else skips ahead"
        );
    }

    #[test]
    fn an_if_without_an_else_falls_past_the_end() {
        let flow = recover(&[0x04, 0x40, 0x01, 0x0b, 0x0b], 0);

        assert_eq!(
            flow.terminators[&0],
            Terminator::Branch {
                taken: 2,
                not_taken: 4
            }
        );
    }

    #[test]
    fn br_table_resolves_every_entry() {
        // block { block { br_table 0 1 0 } } end
        let code = [
            0x02, 0x40, 0x02, 0x40, 0x0e, 0x02, 0x00, 0x01, 0x00, 0x0b, 0x0b, 0x0b,
        ];
        let flow = recover(&code, 0);

        assert_eq!(
            flow.terminators[&4],
            Terminator::Table {
                targets: vec![Some(10), Some(11)],
                default: Some(10)
            }
        );
    }

    #[test]
    fn a_br_table_entry_naming_the_function_label_returns() {
        // block { br_table 0 1 } end, so entry 0 leaves the block and entry 1 the function
        let code = [0x02, 0x40, 0x0e, 0x01, 0x00, 0x01, 0x0b, 0x0b];
        let flow = recover(&code, 0);

        assert_eq!(
            flow.terminators[&2],
            Terminator::Table {
                targets: vec![Some(7)],
                default: None
            }
        );
        assert_eq!(
            flow.terminators[&2].edges(),
            [Edge::Unconditional(7), Edge::FunctionReturn]
        );
    }

    #[test]
    fn terminators_that_end_the_function() {
        assert_eq!(recover(&[0x00, 0x0b], 0).terminators[&0], Terminator::Halt);
        assert_eq!(
            recover(&[0x0f, 0x0b], 0).terminators[&0],
            Terminator::Return
        );
        assert_eq!(
            recover(&[0x12, 0x00, 0x0b], 0).terminators[&0],
            Terminator::Return
        );
    }

    #[test]
    fn recovery_stops_at_the_end_of_the_body() {
        let flow = recover(&[0x01, 0x0b, 0x41, 0x00, 0x0b], 0x10);

        assert_eq!(flow.end, 0x12);
        assert_eq!(flow.instructions, [(0x10, 1), (0x11, 1)]);
    }

    #[test]
    fn an_unbalanced_body_is_reported_as_truncated() {
        assert!(recover(&[0x02, 0x40, 0x01], 0).truncated, "never closed");
        assert!(recover(&[0x02, 0x40, 0xff], 0).truncated, "bad opcode");
        assert!(!recover(&[0x01, 0x0b], 0).truncated);
    }

    #[test]
    fn an_out_of_range_label_is_unresolved() {
        let flow = recover(&[0x0c, 0x09, 0x0b], 0);
        assert_eq!(flow.terminators[&0], Terminator::Unresolved);
    }

    #[test]
    fn blocks_split_at_targets_and_after_terminators() {
        // block { br_if 0; nop } end
        let flow = recover(&[0x02, 0x40, 0x0d, 0x00, 0x01, 0x0b, 0x0b], 0);
        let blocks = flow.blocks();

        assert_eq!(
            blocks,
            [
                Block {
                    start: 0,
                    end: 4,
                    edges: vec![Edge::True(6), Edge::False(4)],
                },
                Block {
                    start: 4,
                    end: 6,
                    edges: vec![Edge::Unconditional(6)],
                },
                Block {
                    start: 6,
                    end: 7,
                    edges: vec![Edge::FunctionReturn],
                },
            ]
        );
    }

    #[test]
    fn a_straight_line_function_is_one_block() {
        let flow = recover(&[0x01, 0x01, 0x01, 0x0b], 0x40);
        let blocks = flow.blocks();

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].start, 0x40);
        assert_eq!(blocks[0].end, 0x44);
        assert_eq!(blocks[0].edges, [Edge::FunctionReturn]);
    }

    #[test]
    fn a_loop_produces_a_back_edge() {
        // loop { br 0 } end
        let flow = recover(&[0x03, 0x40, 0x0c, 0x00, 0x0b, 0x0b], 0);
        let blocks = flow.blocks();

        assert_eq!(blocks[0].start, 0);
        assert_eq!(blocks[0].edges, [Edge::Unconditional(2)]);
        assert_eq!(blocks[1].start, 2);
        assert_eq!(blocks[1].edges, [Edge::Unconditional(2)]);
    }

    #[test]
    fn a_trap_leaves_no_edges() {
        let flow = recover(&[0x00, 0x0b], 0);
        let blocks = flow.blocks();

        assert_eq!(blocks[0].edges, []);
    }

    #[test]
    fn blocks_tile_the_function_exactly() {
        let code = [
            0x02, 0x40, 0x04, 0x40, 0x01, 0x05, 0x0c, 0x01, 0x0b, 0x03, 0x40, 0x0d, 0x00, 0x0b,
            0x0b, 0x0b,
        ];
        let flow = recover(&code, 0x1000);
        let blocks = flow.blocks();

        assert!(!blocks.is_empty());
        assert_eq!(blocks[0].start, flow.start);
        assert_eq!(blocks.last().unwrap().end, flow.end);
        for pair in blocks.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
            assert!(pair[0].start < pair[0].end);
        }
        for target in blocks.iter().flat_map(|block| block.edges.iter()) {
            if let Edge::Unconditional(to) | Edge::True(to) | Edge::False(to) = target {
                assert!(
                    blocks.iter().any(|block| block.start == *to),
                    "{to:#x} is not a block start"
                );
            }
        }
    }

    #[test]
    fn a_branch_at_its_labels_own_height_unwinds_nothing() {
        let flow = recover(&[0x02, 0x40, 0x0c, 0x00, 0x0b, 0x0b], 0);
        assert!(flow.unwinds.is_empty(), "{:?}", flow.unwinds);
    }

    #[test]
    fn a_branch_out_of_a_block_discards_what_the_block_pushed() {
        let code = [0x02, 0x40, 0x41, 0x01, 0x41, 0x02, 0x0c, 0x00, 0x0b, 0x0b];
        let flow = recover(&code, 0);
        assert_eq!(flow.unwinds[&6], [Unwind { keep: 0, drop: 2 }]);
    }

    #[test]
    fn a_branch_carries_its_labels_results_over_what_it_discards() {
        let code = [
            0x02, 0x7f, 0x41, 0x01, 0x41, 0x02, 0x0c, 0x00, 0x0b, 0x1a, 0x0b,
        ];
        let flow = recover(&code, 0);
        assert_eq!(flow.unwinds[&6], [Unwind { keep: 1, drop: 1 }]);
    }

    #[test]
    fn a_conditional_branch_unwinds_from_below_its_own_condition() {
        let code = [
            0x02, 0x40, 0x41, 0x01, 0x41, 0x00, 0x0d, 0x00, 0x1a, 0x0b, 0x0b,
        ];
        let flow = recover(&code, 0);
        assert_eq!(flow.unwinds[&6], [Unwind { keep: 0, drop: 1 }]);
    }

    #[test]
    fn a_branch_to_a_loop_unwinds_to_the_top_of_it() {
        let code = [0x03, 0x40, 0x41, 0x01, 0x0c, 0x00, 0x0b, 0x0b];
        let flow = recover(&code, 0);
        assert_eq!(flow.unwinds[&4], [Unwind { keep: 0, drop: 1 }]);
    }

    #[test]
    fn the_height_recovers_after_code_nothing_reaches() {
        // (block (block i32.const 1 (br 0) i32.const 2) i32.const 3 (br 0))
        let code = [
            0x02, 0x40, 0x02, 0x40, 0x41, 0x01, 0x0c, 0x00, 0x41, 0x02, 0x0b, 0x41, 0x03, 0x0c,
            0x00, 0x0b, 0x0b,
        ];
        let flow = recover(&code, 0);
        assert_eq!(flow.unwinds[&6], [Unwind { keep: 0, drop: 1 }]);
        assert_eq!(
            flow.unwinds[&13],
            [Unwind { keep: 0, drop: 1 }],
            "the unreachable `i32.const 2` did not follow the walk out of its block"
        );
    }

    #[test]
    fn a_body_that_takes_more_than_it_was_given_says_so() {
        let flow = recover(&[0x1a, 0x0b], 0);
        assert!(flow.underflow, "{:?}", flow.heights);

        let flow = recover(&[0x41, 0x01, 0x1a, 0x0b], 0);
        assert!(!flow.underflow, "{:?}", flow.heights);
    }

    #[test]
    fn the_arms_of_an_if_stand_where_their_frame_begins() {
        // (i32.const 1 (if (then i32.const 2 drop) (else i32.const 3 drop)))
        let code = [
            0x41, 0x01, 0x04, 0x40, 0x41, 0x02, 0x1a, 0x05, 0x41, 0x03, 0x1a, 0x0b, 0x0b,
        ];
        let flow = recover(&code, 0);
        assert!(flow.unwinds.is_empty(), "{:?}", flow.unwinds);
    }
}
