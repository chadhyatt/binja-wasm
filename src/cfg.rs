//! Whole-function control flow recovery
//!
//! A branch names a label relative to its enclosing block, so where it goes is the end of the
//! frame that label names, or the top of the body for a `loop`

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::{Arc, LazyLock, PoisonError, RwLock};

use wasmparser::{BlockType, Catch, Handle, Operator};

use crate::ViewId;
use crate::arch::CAUGHT_REGISTERS;
use crate::insn;
use crate::lift;
use crate::module::Module;

/// Bodies whose operand stack went below empty, by where their code starts
static UNBALANCED: LazyLock<RwLock<BTreeSet<(ViewId, u64)>>> = LazyLock::new(Default::default);

pub fn forget(view: ViewId) {
    let mut unbalanced = UNBALANCED.write().unwrap_or_else(PoisonError::into_inner);
    unbalanced.retain(|(owner, _)| *owner != view);
    let mut recovered = RECOVERED.write().unwrap_or_else(PoisonError::into_inner);
    recovered.retain(|(_, owner), _| *owner != view);
    let mut dispatch = DISPATCH.write().unwrap_or_else(PoisonError::into_inner);
    dispatch.retain(|(_, owner), _| *owner != view);
    let mut heights = HEIGHTS.write().unwrap_or_else(PoisonError::into_inner);
    heights.retain(|(owner, _), _| *owner != view);
}

pub fn note_unbalanced(view: ViewId, start: u64) {
    let mut unbalanced = UNBALANCED.write().unwrap_or_else(PoisonError::into_inner);
    unbalanced.insert((view, start));
}

pub fn is_unbalanced(view: ViewId, start: u64) -> bool {
    let unbalanced = UNBALANCED.read().unwrap_or_else(PoisonError::into_inner);
    unbalanced.contains(&(view, start))
}

/// Recovery needs a whole function, but the core asks one instruction at a time with no context,
/// so analysis leaves the answers here
///
/// Keyed by address before file, so both looking one up in a known file and asking which files
/// know an address at all stay logarithmic
static RECOVERED: LazyLock<RwLock<BTreeMap<(u64, ViewId), Recovered>>> =
    LazyLock::new(Default::default);

static DISPATCH: LazyLock<RwLock<BTreeMap<(u64, ViewId), Dispatch>>> =
    LazyLock::new(Default::default);

#[derive(Debug)]
struct Heights {
    end: u64,
    at: Vec<(u64, Option<u32>)>,
}

static HEIGHTS: LazyLock<RwLock<BTreeMap<(ViewId, u64), Heights>>> =
    LazyLock::new(Default::default);

fn install_heights(view: ViewId, flow: &ControlFlow) {
    let at = flow
        .instructions
        .iter()
        .zip(&flow.heights)
        .map(|((addr, _), height)| (*addr, height.and_then(|height| u32::try_from(height).ok())))
        .collect();
    let mut heights = HEIGHTS.write().unwrap_or_else(PoisonError::into_inner);
    let stale: Vec<(ViewId, u64)> = heights
        .range((view, flow.start)..(view, flow.end))
        .map(|(key, _)| *key)
        .collect();
    for key in stale {
        heights.remove(&key);
    }
    heights.insert((view, flow.start), Heights { end: flow.end, at });
}

fn height_in(
    heights: &BTreeMap<(ViewId, u64), Heights>,
    view: ViewId,
    addr: u64,
) -> Option<Option<u32>> {
    let (_, found) = heights.range((view, 0)..=(view, addr)).next_back()?;
    if addr >= found.end {
        return None;
    }
    let index = found.at.binary_search_by_key(&addr, |(at, _)| *at).ok()?;
    Some(found.at[index].1)
}

pub fn height(view: ViewId, addr: u64) -> Option<u32> {
    let heights = HEIGHTS.read().unwrap_or_else(PoisonError::into_inner);
    height_in(&heights, view, addr).flatten()
}

pub fn height_anywhere(addr: u64) -> Option<u32> {
    let heights = HEIGHTS.read().unwrap_or_else(PoisonError::into_inner);
    let mut answer = None;
    let mut view = heights.keys().next().map(|(view, _)| *view);
    while let Some(current) = view {
        if let Some(height) = height_in(&heights, current, addr) {
            if answer.is_some_and(|seen| seen != height) {
                return None;
            }
            answer = Some(height);
        }
        view = current
            .checked_add(1)
            .and_then(|next| heights.range((next, 0)..).next())
            .map(|((view, _), _)| *view);
    }
    answer.flatten()
}

pub fn install(view: ViewId, flow: &ControlFlow) {
    install_heights(view, flow);
    // The map is a cache, so a poisoned lock is better carried on with than propagated
    let mut recovered = RECOVERED.write().unwrap_or_else(PoisonError::into_inner);
    replace(
        &mut recovered,
        view,
        flow,
        flow.terminators.iter().map(|(addr, terminator)| {
            (
                *addr,
                Recovered {
                    terminator: terminator.clone(),
                    unwinds: flow.unwinds.get(addr).cloned().unwrap_or_default(),
                },
            )
        }),
    );
    let mut dispatch = DISPATCH.write().unwrap_or_else(PoisonError::into_inner);
    replace(
        &mut dispatch,
        view,
        flow,
        flow.dispatch
            .iter()
            .map(|(addr, dispatch)| (*addr, dispatch.clone())),
    );
}

fn replace<T>(
    map: &mut BTreeMap<(u64, ViewId), T>,
    view: ViewId,
    flow: &ControlFlow,
    entries: impl Iterator<Item = (u64, T)>,
) {
    let stale: Vec<(u64, ViewId)> = map
        .range((flow.start, ViewId::MIN)..(flow.end, ViewId::MIN))
        .map(|(key, _)| *key)
        .filter(|(_, owner)| *owner == view)
        .collect();
    for key in stale {
        map.remove(&key);
    }
    for (addr, entry) in entries {
        map.insert((addr, view), entry);
    }
}

pub fn lookup(view: ViewId, addr: u64) -> Option<Recovered> {
    let recovered = RECOVERED.read().unwrap_or_else(PoisonError::into_inner);
    recovered.get(&(addr, view)).cloned()
}

pub fn dispatch(view: ViewId, addr: u64) -> Option<Dispatch> {
    let dispatch = DISPATCH.read().unwrap_or_else(PoisonError::into_inner);
    dispatch.get(&(addr, view)).cloned()
}

/// For the callbacks handed an address and no file: answering only when every file that knows the
/// address agrees is exact for one open file and silent rather than wrong for two
pub fn lookup_anywhere(addr: u64) -> Option<Recovered> {
    let recovered = RECOVERED.read().unwrap_or_else(PoisonError::into_inner);
    agreed(&recovered, addr)
}

pub fn dispatch_anywhere(addr: u64) -> Option<Dispatch> {
    let dispatch = DISPATCH.read().unwrap_or_else(PoisonError::into_inner);
    agreed(&dispatch, addr)
}

fn agreed<T: Clone + PartialEq>(map: &BTreeMap<(u64, ViewId), T>, addr: u64) -> Option<T> {
    let mut found: Option<&T> = None;
    for (_, entry) in map.range((addr, ViewId::MIN)..=(addr, ViewId::MAX)) {
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
    /// Traps and throws, which never fall through; where a throw lands is its [`Dispatch`]
    Halt,
    Suspend {
        handlers: Vec<(u32, Option<u64>)>,
        next: u64,
    },
    /// A target that could not be recovered, reported as such rather than guessed at
    Unresolved,
}

/// Branching out of a block unwinds it
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
pub struct Clause {
    pub tag: Option<u32>,
    pub target: Option<u64>,
    pub base: Option<i64>,
    pub carried: u32,
    pub exnref: bool,
    pub keeps: Option<u32>,
    pub restores: Option<u32>,
}

struct Handlers {
    clauses: Vec<Clause>,
    by_tag: HashMap<u32, usize>,
    unknown: bool,
    outer: Option<Arc<Handlers>>,
}

impl Handlers {
    fn chain(&self) -> impl Iterator<Item = &Handlers> {
        std::iter::successors(Some(self), |handlers| handlers.outer.as_deref())
    }

    fn catching(&self, tag: u32) -> &[Clause] {
        match self.by_tag.get(&tag) {
            Some(&at) => std::slice::from_ref(&self.clauses[at]),
            None => match self.clauses.last() {
                Some(all) if all.tag.is_none() => std::slice::from_ref(all),
                _ => &[],
            },
        }
    }
}

impl Drop for Handlers {
    fn drop(&mut self) {
        let mut outer = self.outer.take();
        while let Some(next) = outer {
            outer = Arc::into_inner(next).and_then(|mut handlers| handlers.outer.take());
        }
    }
}

impl PartialEq for Handlers {
    fn eq(&self, other: &Self) -> bool {
        let (mut left, mut right) = (self.chain(), other.chain());
        loop {
            match (left.next(), right.next()) {
                (None, None) => return true,
                (Some(ours), Some(theirs)) if std::ptr::eq(ours, theirs) => return true,
                (Some(ours), Some(theirs))
                    if ours.clauses == theirs.clauses && ours.unknown == theirs.unknown => {}
                _ => return false,
            }
        }
    }
}

impl Eq for Handlers {}

impl std::fmt::Debug for Handlers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(
                self.chain()
                    .map(|handlers| (&handlers.clauses, handlers.unknown)),
            )
            .finish()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Dispatch {
    pub tag: Option<u32>,
    pub carried: u32,
    pub caught: Option<u32>,
    handlers: Option<Arc<Handlers>>,
}

pub struct Offered<'a> {
    pub clauses: Vec<&'a Clause>,
    pub truncated: bool,
}

impl Dispatch {
    pub fn offered(&self) -> Offered<'_> {
        let mut offered = Offered {
            clauses: Vec::new(),
            truncated: false,
        };
        let mut seen = HashSet::new();
        let mut examined = 0usize;
        let mut exhausted = || {
            examined += 1;
            examined > MAX_SEARCH
        };
        for handlers in self.handlers.iter().flat_map(|handlers| handlers.chain()) {
            if exhausted() || handlers.unknown {
                offered.truncated = true;
                return offered;
            }
            let candidates = match self.tag {
                Some(tag) => handlers.catching(tag),
                None => &handlers.clauses[..],
            };
            for clause in candidates {
                if exhausted() {
                    offered.truncated = true;
                    return offered;
                }
                if clause.tag.is_some_and(|tag| !seen.insert(tag)) {
                    continue;
                }
                offered.clauses.push(clause);
                if clause.tag.is_none() || self.tag.is_some() {
                    return offered;
                }
            }
        }
        offered
    }

    fn landings(&self, visited: &mut HashSet<*const Handlers>) -> Vec<u64> {
        let mut landings = Vec::new();
        for handlers in self.handlers.iter().flat_map(|handlers| handlers.chain()) {
            if !visited.insert(std::ptr::from_ref(handlers)) {
                break;
            }
            landings.extend(handlers.clauses.iter().filter_map(|clause| clause.target));
        }
        landings
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovered {
    pub terminator: Terminator,
    /// One per edge that can unwind: a branch's taken edge, every table entry then its default, or
    /// every handler of a resume
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
    pub dispatch: BTreeMap<u64, Dispatch>,
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

    /// A block starts at the entry, at any branch target, and after any instruction that does not
    /// fall into the next one
    pub fn blocks(&self) -> Vec<Block> {
        if self.instructions.is_empty() {
            return Vec::new();
        }

        let mut leaders = BTreeSet::from([self.start]);
        leaders.extend(
            self.handlers()
                .into_iter()
                .filter(|target| self.contains(*target)),
        );
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

    fn handlers(&self) -> BTreeSet<u64> {
        let mut visited = HashSet::new();
        self.dispatch
            .values()
            .flat_map(|dispatch| dispatch.landings(&mut visited))
            .collect()
    }

    pub fn reachable_blocks(&self) -> Vec<Block> {
        let blocks = self.blocks();
        if self.truncated {
            return blocks;
        }
        let by_start: BTreeMap<u64, usize> = blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (block.start, index))
            .collect();
        let mut reached = vec![false; blocks.len()];
        let mut pending = vec![self.start];
        let mut visited = HashSet::new();
        while let Some(at) = pending.pop() {
            let Some(&index) = by_start.get(&at) else {
                continue;
            };
            if std::mem::replace(&mut reached[index], true) {
                continue;
            }
            let block = &blocks[index];
            for dispatch in self.dispatch.range(block.start..block.end).map(|(_, d)| d) {
                pending.extend(dispatch.landings(&mut visited));
            }
            for edge in &block.edges {
                if let Edge::Unconditional(to) | Edge::True(to) | Edge::False(to) = edge {
                    pending.push(*to);
                }
            }
        }
        blocks
            .into_iter()
            .zip(reached)
            .filter_map(|(block, reached)| reached.then_some(block))
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
            Self::Suspend { handlers, next } => handlers
                .iter()
                .filter_map(|(_, to)| *to)
                .chain([*next])
                .collect(),
            Self::Return | Self::Halt | Self::Unresolved => Vec::new(),
        }
    }

    pub(crate) fn edges(&self) -> Vec<Edge> {
        match self {
            Self::Jump(target) => vec![Edge::Unconditional(*target)],
            Self::Branch { taken, not_taken } => {
                vec![Edge::True(*taken), Edge::False(*not_taken)]
            }
            Self::Table { targets, default } => many_ways(targets.iter().chain([default]).copied()),
            Self::Suspend { handlers, next } => {
                many_ways(handlers.iter().map(|(_, to)| *to).chain([Some(*next)]))
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

fn many_ways(targets: impl Iterator<Item = Option<u64>>) -> Vec<Edge> {
    let mut seen = BTreeSet::new();
    let mut leaves = false;
    let mut edges = Vec::new();
    for target in targets {
        match target {
            Some(to) if seen.insert(to) => edges.push(Edge::Unconditional(to)),
            Some(_) => {}
            None => leaves = true,
        }
    }
    if leaves {
        edges.push(Edge::FunctionReturn);
    }
    edges
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameKind {
    /// The implicit block around the whole body, so branching to it is a return
    Function,
    Block,
    Loop,
    If,
    Try,
    TryTable,
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
    catches: Vec<(Option<u32>, u64)>,
    handling: Option<Option<u32>>,
    delegate: Option<Option<usize>>,
    clauses: Vec<(Option<u32>, bool, usize)>,
    unresolved: bool,
    nesting: u32,
    outside: Option<usize>,
    protecting: Option<usize>,
}

struct Protection {
    frame: usize,
    outer: Option<usize>,
}

#[derive(Default)]
struct Protections {
    chain: Vec<Protection>,
    head: Option<usize>,
}

impl Protections {
    fn enter(&mut self, frames: &mut [Frame], frame: usize) {
        self.chain.push(Protection {
            frame,
            outer: self.head,
        });
        self.head = Some(self.chain.len() - 1);
        frames[frame].protecting = self.head;
    }

    fn leave(&mut self, frames: &mut [Frame], frame: usize) {
        if let Some(node) = frames[frame].protecting.take() {
            self.head = self.chain[node].outer;
        }
    }
}

impl Frame {
    /// A `loop` label restarts the body; everything else lands after its matching `end`
    fn label(&self) -> Option<u64> {
        match self.kind {
            FrameKind::Loop => Some(self.body),
            FrameKind::Function => None,
            _ => self.end,
        }
    }

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
pub fn block_arity(blockty: &BlockType, module: Option<&Module>) -> Option<(u32, u32)> {
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
        catches: Vec::new(),
        handling: None,
        delegate: None,
        clauses: Vec::new(),
        unresolved: false,
        nesting: 0,
        outside: None,
        protecting: None,
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
    Suspend {
        handlers: Vec<(u32, usize)>,
        next: u64,
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

    let mut frames = Vec::new();
    // A function is entered with an empty operand stack whatever its parameters, since those are
    // locals here, in a frame of their own
    let function = open_frame(
        &mut frames,
        FrameKind::Function,
        start,
        Some(0),
        &BlockType::Empty,
        None,
    );
    let mut open = vec![function];
    let mut pending: Vec<(u64, Option<i64>, Pending)> = Vec::new();
    let mut raised: Vec<Raised> = Vec::new();
    let mut protections = Protections::default();
    let mut tries = 0u32;
    let mut offset = 0usize;
    let identity = |tag: u32| module.map_or(tag, |module| module.tag_identity(tag));

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
                frames[frame].outside = protections.head;
                if kind == FrameKind::Try {
                    frames[frame].nesting = tries;
                    tries += 1;
                    protections.enter(&mut frames, frame);
                }
                flow.underflow |= frames[frame].base.is_none() && height.is_some();
                open.push(frame);
                if kind == FrameKind::If {
                    pending.push((addr, height, Pending::Conditional { frame, taken: next }));
                }
            }
            Operator::TryTable { try_table } => {
                let mut unresolved = false;
                let clauses: Vec<(Option<u32>, bool, usize)> = try_table
                    .catches
                    .iter()
                    .filter_map(|catch| {
                        let (tag, exnref) = match *catch {
                            Catch::One { tag, .. } => (Some(identity(tag)), false),
                            Catch::OneRef { tag, .. } => (Some(identity(tag)), true),
                            Catch::All { .. } => (None, false),
                            Catch::AllRef { .. } => (None, true),
                        };
                        let label = frame_at(&open, insn::catch_label(catch));
                        unresolved |= label.is_none();
                        Some((tag, exnref, label?))
                    })
                    .collect();
                let frame = open_frame(
                    &mut frames,
                    FrameKind::TryTable,
                    next,
                    height,
                    &try_table.ty,
                    module,
                );
                frames[frame].clauses = clauses;
                frames[frame].unresolved = unresolved;
                frames[frame].outside = protections.head;
                protections.enter(&mut frames, frame);
                flow.underflow |= frames[frame].base.is_none() && height.is_some();
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
                    let tag = match insn.op {
                        Operator::Catch { tag_index } => Some(identity(tag_index)),
                        _ => None,
                    };
                    frames[frame].catches.push((tag, next));
                    frames[frame].handling = Some(tag);
                    protections.leave(&mut frames, frame);
                    // A handler holds what its tag carries, and `catch_all` names no tag
                    height = frames[frame]
                        .base
                        .zip(carried(module, tag))
                        .map(|(base, carried)| base + i64::from(carried));
                }
            }
            Operator::End | Operator::Delegate { .. } => {
                if let Some(frame) = open.pop() {
                    protections.leave(&mut frames, frame);
                    if frames[frame].kind == FrameKind::Try {
                        tries -= 1;
                    }
                    if let Operator::Delegate { relative_depth } = insn.op {
                        frames[frame].delegate = frame_at(&open, relative_depth)
                            .map(|target| frames[target].protecting.or(frames[target].outside));
                    }
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
            op if let Some(relative_depth) = insn::conditional_label(op) => {
                // The two edges leave different amounts behind, so they count separately
                let taken = height.map(|h| h - lift::taken_pops(&insn.op));
                match frame_at(&open, relative_depth) {
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
            Operator::Unreachable => {
                flow.terminators.insert(addr, Terminator::Halt);
                height = None;
            }
            Operator::Throw { .. } | Operator::ThrowRef | Operator::Rethrow { .. } => {
                let handler = match insn.op {
                    Operator::Rethrow { relative_depth } => frame_at(&open, relative_depth),
                    _ => None,
                };
                let tag = match insn.op {
                    Operator::Throw { tag_index } => Some(identity(tag_index)),
                    _ => handler.and_then(|frame| frames[frame].handling.flatten()),
                };
                raised.push(Raised {
                    at: addr,
                    tag,
                    caught: handler
                        .map(|frame| frames[frame].nesting)
                        .filter(|nesting| *nesting < CAUGHT_REGISTERS),
                    protected: protections.head,
                    throws: true,
                });
                flow.terminators.insert(addr, Terminator::Halt);
                height = None;
            }
            // Everything else moves the stack by its own arity, which the lifter answers for so
            // there is one answer rather than two
            _ => {
                if insn::raises(&insn.op) {
                    raised.push(Raised {
                        at: addr,
                        tag: None,
                        caught: None,
                        protected: protections.head,
                        throws: false,
                    });
                }
                if let Some(table) = insn::resume_table(&insn.op) {
                    let handlers: Option<Vec<(u32, usize)>> = table
                        .handlers
                        .iter()
                        .filter_map(|handle| match *handle {
                            Handle::OnLabel { tag, label } => {
                                Some(frame_at(&open, label).map(|frame| (identity(tag), frame)))
                            }
                            Handle::OnSwitch { .. } => None,
                        })
                        .collect();
                    match handlers {
                        Some(handlers) if handlers.is_empty() => {}
                        Some(handlers) => {
                            pending.push((addr, height, Pending::Suspend { handlers, next }));
                        }
                        None => {
                            flow.terminators.insert(addr, Terminator::Unresolved);
                        }
                    }
                }
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
    catch(&mut flow, &frames, &protections.chain, raised, module, code);
    leaving_the_body_returns(&mut flow);
    flow
}

struct Raised {
    at: u64,
    tag: Option<u32>,
    caught: Option<u32>,
    protected: Option<usize>,
    throws: bool,
}

fn carried(module: Option<&Module>, tag: Option<u32>) -> Option<u32> {
    match tag {
        Some(tag) => module.and_then(|module| module.tag_arity(tag)),
        None => Some(0),
    }
}

fn catch(
    flow: &mut ControlFlow,
    frames: &[Frame],
    chain: &[Protection],
    raised: Vec<Raised>,
    module: Option<&Module>,
    code: &[u8],
) {
    let handlers = handlers(frames, chain, module, code, flow.start);
    for Raised {
        at,
        tag,
        caught,
        protected,
        throws,
    } in raised
    {
        let handlers = protected.and_then(|node| handlers[node].clone());
        if throws || handlers.is_some() {
            flow.dispatch.insert(
                at,
                Dispatch {
                    tag,
                    carried: carried(module, tag).unwrap_or(0),
                    caught,
                    handlers,
                },
            );
        }
    }
}

const MAX_SEARCH: usize = 64;

fn handlers(
    frames: &[Frame],
    chain: &[Protection],
    module: Option<&Module>,
    code: &[u8],
    start: u64,
) -> Vec<Option<Arc<Handlers>>> {
    let mut built: Vec<Option<Arc<Handlers>>> = Vec::with_capacity(chain.len());
    for Protection { frame, outer } in chain {
        let frame = &frames[*frame];
        let outer = outer.and_then(|node| built[node].clone());
        let handlers = match frame.delegate {
            Some(target) => target.and_then(|node| built[node].clone()),
            None => {
                let mut clauses = Vec::new();
                let mut by_tag = HashMap::new();
                for clause in offered_by(frames, frame, module, code, start) {
                    if let Some(tag) = clause.tag {
                        if by_tag.contains_key(&tag) {
                            continue;
                        }
                        by_tag.insert(tag, clauses.len());
                    }
                    let last = clause.tag.is_none();
                    clauses.push(clause);
                    if last {
                        break;
                    }
                }
                if clauses.is_empty() && !frame.unresolved {
                    outer
                } else {
                    Some(Arc::new(Handlers {
                        clauses,
                        by_tag,
                        unknown: frame.unresolved,
                        outer,
                    }))
                }
            }
        };
        built.push(handlers);
    }
    built
}

fn offered_by(
    frames: &[Frame],
    frame: &Frame,
    module: Option<&Module>,
    code: &[u8],
    start: u64,
) -> Vec<Clause> {
    let carried = |tag: Option<u32>| carried(module, tag);
    let pointer = module.and_then(|module| module.stack_pointer);
    let restores = |target: Option<u64>, delivered: Option<u32>| {
        restored_from(code, start, target?, pointer?, delivered?)
    };
    if frame.kind == FrameKind::TryTable {
        frame
            .clauses
            .iter()
            .map(|&(clause, exnref, to)| {
                let arity = carried(clause);
                let target = frames[to].label();
                Clause {
                    tag: clause,
                    target,
                    base: arity.and(frames[to].base),
                    carried: arity.unwrap_or(0),
                    exnref,
                    keeps: None,
                    restores: restores(target, arity.map(|arity| arity + u32::from(exnref))),
                }
            })
            .collect()
    } else {
        frame
            .catches
            .iter()
            .map(|&(clause, handler)| {
                let arity = carried(clause);
                Clause {
                    tag: clause,
                    target: Some(handler),
                    base: arity.and(frame.base),
                    carried: arity.unwrap_or(0),
                    exnref: false,
                    keeps: Some(frame.nesting).filter(|n| *n < CAUGHT_REGISTERS),
                    restores: restores(Some(handler), arity),
                }
            })
            .collect()
    }
}

fn restored_from(
    code: &[u8],
    start: u64,
    target: u64,
    pointer: u32,
    delivered: u32,
) -> Option<u32> {
    let mut offset = usize::try_from(target.checked_sub(start)?).ok()?;
    let mut stored = HashSet::new();
    for _ in 0..=delivered {
        let insn = insn::decode_any(code.get(offset..)?)?;
        offset += insn.len;
        match insn.op {
            Operator::LocalSet { local_index } => {
                stored.insert(local_index);
            }
            Operator::Drop => {}
            Operator::LocalGet { local_index } if !stored.contains(&local_index) => {
                return match insn::decode_any(code.get(offset..)?)?.op {
                    Operator::GlobalSet { global_index } if global_index == pointer => {
                        Some(local_index)
                    }
                    _ => None,
                };
            }
            _ => return None,
        }
    }
    None
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
    let target = |index: usize| frames[index].label();
    let is_function = |index: usize| frames[index].kind == FrameKind::Function;
    // An entry naming the outermost label returns; one that fails to resolve makes the whole set
    // unusable, since no entry can be marked as going nowhere
    let entry = |frame: usize| -> Option<Option<u64>> {
        if is_function(frame) {
            Some(None)
        } else {
            target(frame).map(Some)
        }
    };

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
            Pending::Suspend { handlers, .. } => {
                unwinds.extend(handlers.iter().map(|(_, frame)| {
                    if is_function(*frame) {
                        Unwind::default()
                    } else {
                        let arity = i64::from(frames[*frame].label_arity);
                        unwind(*frame, height.map(|height| height + arity))
                    }
                }));
            }
            _ => {}
        }
        let carries = matches!(branch, Pending::Suspend { .. });
        if carries || unwinds.iter().any(|unwind| !unwind.is_empty()) {
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
                let targets: Option<Vec<Option<u64>>> =
                    entries.iter().map(|frame| entry(*frame)).collect();
                match (targets, entry(default)) {
                    (Some(targets), Some(default)) => Terminator::Table { targets, default },
                    _ => Terminator::Unresolved,
                }
            }
            Pending::Suspend { handlers, next } => handlers
                .iter()
                .map(|&(tag, frame)| Some((tag, entry(frame)?)))
                .collect::<Option<Vec<_>>>()
                .map_or(Terminator::Unresolved, |handlers| Terminator::Suspend {
                    handlers,
                    next,
                }),
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
    fn code_nothing_reaches_is_not_handed_over() {
        let flow = recover(&[0x02, 0x40, 0x0c, 0x00, 0x0b, 0x0b], 0x100);
        let all: Vec<u64> = flow.blocks().iter().map(|block| block.start).collect();
        let reached: Vec<u64> = flow
            .reachable_blocks()
            .iter()
            .map(|block| block.start)
            .collect();
        assert_eq!(all, [0x100, 0x104, 0x105]);
        assert_eq!(
            reached,
            [0x100, 0x105],
            "the end after the branch never runs"
        );
    }

    #[test]
    fn a_handler_is_handed_over_when_something_it_covers_can_throw() {
        let reached = |code: &[u8]| -> Vec<u64> {
            recover(code, 0x100)
                .reachable_blocks()
                .iter()
                .map(|block| block.start)
                .collect()
        };
        assert_eq!(
            reached(&[0x06, 0x40, 0x10, 0x00, 0x19, 0x01, 0x0b, 0x0b]),
            [0x100, 0x105, 0x107]
        );
        assert_eq!(
            reached(&[0x02, 0x40, 0x0d, 0x00, 0x01, 0xff]),
            [0x100, 0x104],
            "a body cut short keeps every block, whether or not an edge to it was recovered"
        );
        assert_eq!(
            reached(&[0x06, 0x40, 0x01, 0x19, 0x01, 0x0b, 0x0b]),
            [0x100, 0x106],
            "nothing in the body throws, so the handler never runs"
        );
        assert_eq!(
            reached(&[
                0x02, 0x40, 0x0c, 0x00, 0x06, 0x40, 0x08, 0x00, 0x19, 0x01, 0x0b, 0x0b, 0x0b
            ]),
            [0x100, 0x10c],
            "the only throw is never reached"
        );

        let (_, flow) = recover_last(
            r#"(module (type $ft (func)) (type $ct (cont $ft)) (func $f) (elem declare func $f)
                 (func
                   try
                     (resume $ct (cont.new $ct (ref.func $f)))
                   catch_all
                   end))"#,
        );
        let handler = dispatch_at(&flow, 0).offered().clauses[0]
            .target
            .expect("a handler");
        assert!(
            flow.reachable_blocks()
                .iter()
                .any(|block| block.start == handler),
            "a continuation can throw out of a resume"
        );
    }

    fn recover_last(text: &str) -> (Module, ControlFlow) {
        let image = wat::parse_str(text).expect("the fixture assembles");
        let module = crate::module::parse(&image, 0).expect("parses");
        let (_, info) = module.functions().last().expect("a function");
        let code = &image[info.entry as usize..info.end as usize];
        let flow = super::recover(code, info.entry, Some(&module));
        (module, flow)
    }

    fn dispatch_at(flow: &ControlFlow, nth: usize) -> &Dispatch {
        flow.dispatch.values().nth(nth).expect("a dispatch")
    }

    #[test]
    fn a_call_in_a_try_table_can_land_on_each_clause_label_with_the_payload() {
        let (_, flow) = recover_last(
            r#"(module (tag $e (param i32)) (func $f)
                 (func (result i32)
                   (block $h (result i32)
                     (drop
                       (block $any (result exnref)
                         (try_table (catch $e $h) (catch_all_ref $any) (call $f))
                         (return (i32.const 0))))
                     (i32.const 1))))"#,
        );
        let dispatch = dispatch_at(&flow, 0);
        assert_eq!(dispatch.tag, None);
        let [caught, all] = dispatch.offered().clauses[..] else {
            panic!("{dispatch:?}");
        };
        assert_eq!(
            (caught.tag, caught.base, caught.carried),
            (Some(0), Some(0), 1)
        );
        assert_eq!(
            (all.tag, all.base, all.carried, all.exnref),
            (None, Some(0), 0, true)
        );
        assert_ne!(caught.target, all.target);
        let starts: Vec<u64> = flow.blocks().iter().map(|block| block.start).collect();
        for clause in [caught, all] {
            assert!(
                starts.contains(&clause.target.expect("inside the body")),
                "a clause's label starts a block even though no branch names it"
            );
        }
    }

    #[test]
    fn a_resume_lands_on_each_handler_label_with_what_the_suspension_carries() {
        let (_, flow) = recover_last(
            r#"(module (type $f (func (result i32))) (type $c (cont $f)) (tag $y (param i64))
                 (func $g (result i32) (i32.const 0)) (elem declare func $g)
                 (func (result i32)
                   (block $h (result i64 (ref null $c))
                     (i32.const 7)
                     (resume $c (on $y $h) (cont.new $c (ref.func $g)))
                     (return))
                   (drop) (drop) (i32.const 1)))"#,
        );
        let (&at, terminator) = flow
            .terminators
            .iter()
            .find(|(_, terminator)| matches!(terminator, Terminator::Suspend { .. }))
            .expect("the resume");
        let Terminator::Suspend { handlers, next } = terminator else {
            unreachable!();
        };
        let [(0, Some(handler))] = handlers[..] else {
            panic!("{handlers:?}");
        };
        assert_eq!(flow.next_address(at), Some(*next));
        assert_eq!(
            flow.unwinds.get(&at).map(Vec::as_slice),
            Some(&[Unwind { keep: 2, drop: 2 }][..]),
            "an i64 and a continuation where the block began"
        );
        let index = flow
            .instructions
            .binary_search_by_key(next, |(start, _)| *start)
            .expect("an instruction");
        assert_eq!(
            flow.heights[index],
            Some(2),
            "the resume's result over the 7"
        );
        let reached: Vec<u64> = flow.reachable_blocks().iter().map(|b| b.start).collect();
        assert!(reached.contains(&handler) && reached.contains(next));
    }

    #[test]
    fn a_clause_naming_the_function_label_returns_the_payload() {
        let (_, flow) = recover_last(
            r#"(module (tag $e (param i32)) (func $f)
                 (func (result i32)
                   (try_table (catch $e 0) (call $f))
                   (i32.const 0)))"#,
        );
        let [clause] = dispatch_at(&flow, 0).offered().clauses[..] else {
            panic!("{flow:?}");
        };
        assert_eq!(
            (clause.target, clause.base, clause.carried),
            (None, Some(0), 1)
        );
    }

    #[test]
    fn a_delegated_try_hands_its_exceptions_to_the_try_it_names() {
        let (_, flow) = recover_last(
            r#"(module (tag $e (param i32)) (func $f)
                 (func
                   try
                     try
                       call $f
                     delegate 0
                   catch $e
                     drop
                   catch_all
                   end))"#,
        );
        let offered = dispatch_at(&flow, 0).offered().clauses;
        let tags: Vec<_> = offered.iter().map(|clause| clause.tag).collect();
        assert_eq!(tags, [Some(0), None]);
        assert_eq!(offered[0].carried, 1);
        let reached: Vec<u64> = flow.reachable_blocks().iter().map(|b| b.start).collect();
        for clause in offered {
            assert!(reached.contains(&clause.target.expect("a handler")));
        }
    }

    #[test]
    fn a_delegate_to_a_label_that_is_no_try_goes_to_the_handlers_around_it() {
        let tags = |inner: &str| {
            let (_, flow) = recover_last(&format!(
                r#"(module (tag $e) (func $f)
                     (func
                       try
                         block $b
                           try
                             try
                               call $f
                             delegate {inner}
                           catch $e
                           end
                         end
                       catch_all
                       end))"#
            ));
            flow.dispatch.values().next().map(|dispatch| {
                let offered = dispatch.offered();
                assert!(!offered.truncated);
                offered
                    .clauses
                    .iter()
                    .map(|clause| clause.tag)
                    .collect::<Vec<_>>()
            })
        };
        assert_eq!(tags("0"), Some(vec![Some(0), None]), "the try just outside");
        assert_eq!(
            tags("1"),
            Some(vec![None]),
            "a block, past the try inside it"
        );
        assert_eq!(
            tags("3"),
            None,
            "the function itself, so nothing here handles it"
        );
    }

    #[test]
    fn handlers_nested_past_the_limit_are_left_unknown() {
        let (_, flow) = recover_last(&format!(
            "(module (tag $e) (func $f) (func {} call $f {}))",
            "try ".repeat(MAX_SEARCH + 8),
            "catch $e end ".repeat(MAX_SEARCH + 8),
        ));
        let offered = dispatch_at(&flow, 0).offered();
        assert!(offered.truncated);
        assert_eq!(offered.clauses.len(), 1, "the one tag, caught innermost");
    }

    #[test]
    fn a_deep_chain_of_handlers_is_compared_and_freed_without_recursing() {
        let text = format!(
            "(module (func $f) (func {} call $f {}))",
            "try ".repeat(20_000),
            "catch_all end ".repeat(20_000),
        );
        let small = std::thread::Builder::new().stack_size(256 << 10);
        let run = small.spawn(move || {
            let (_, first) = recover_last(&text);
            let (_, second) = recover_last(&text);
            let same = dispatch_at(&first, 0) == dispatch_at(&second, 0);
            drop((first, second));
            same
        });
        assert!(run.expect("spawns").join().expect("did not overflow"));
    }

    #[test]
    fn handlers_past_the_search_limit_still_start_reachable_blocks() {
        let depth = MAX_SEARCH + 8;
        let tags: String = (0..depth).map(|nth| format!("(tag $t{nth})")).collect();
        let handlers: String = (0..depth)
            .rev()
            .map(|nth| format!("catch $t{nth} end "))
            .collect();
        let (_, flow) = recover_last(&format!(
            "(module {tags} (func $f) (func {} call $f {handlers}))",
            "try ".repeat(depth)
        ));
        let offered = dispatch_at(&flow, 0).offered();
        assert!(offered.truncated);
        let reached: BTreeSet<u64> = flow
            .reachable_blocks()
            .iter()
            .map(|block| block.start)
            .collect();
        let mut visited = HashSet::new();
        let landings = dispatch_at(&flow, 0).landings(&mut visited);
        assert_eq!(landings.len(), depth);
        assert!(landings.iter().all(|at| reached.contains(at)));
    }

    #[test]
    fn a_clause_whose_label_does_not_resolve_leaves_the_search_unknown() {
        let (_, flow) =
            recover_last("(module (func $f) (func (block (try_table (catch_all 5) (call $f)))))");
        let offered = dispatch_at(&flow, 0).offered();
        assert!(offered.truncated && offered.clauses.is_empty());
    }

    #[test]
    fn a_throw_goes_straight_to_the_first_clause_for_its_tag() {
        let (_, flow) = recover_last(
            r#"(module (tag $a) (tag $b (param i32 i64))
                 (func
                   try
                     i32.const 1
                     i64.const 2
                     throw $b
                   catch $a
                   catch $b
                     drop
                     drop
                   catch_all
                   end))"#,
        );
        let dispatch = dispatch_at(&flow, 0);
        assert_eq!((dispatch.tag, dispatch.carried), (Some(1), 2));
        let [only] = dispatch.offered().clauses[..] else {
            panic!("{dispatch:?}");
        };
        assert_eq!((only.tag, only.carried), (Some(1), 2));
    }

    #[test]
    fn imports_of_one_tag_are_caught_as_one_tag() {
        let caught = |second: &str| {
            let (_, flow) = recover_last(&format!(
                r#"(module (import "env" "e" (tag $a (param i32)))
                     (import "env" "{second}" (tag $b (param i32)))
                     (func
                       try
                         (throw $a (i32.const 1))
                       catch $b
                         drop
                       end))"#
            ));
            let dispatch = dispatch_at(&flow, 0);
            let offered = dispatch.offered();
            (
                dispatch.tag,
                offered
                    .clauses
                    .iter()
                    .map(|clause| clause.tag)
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(caught("e"), (Some(0), vec![Some(0)]));
        assert_eq!(
            caught("f"),
            (Some(0), vec![]),
            "another import is another tag"
        );
    }

    #[test]
    fn tries_with_no_handlers_of_their_own_share_the_ones_around_them() {
        let (_, flow) = recover_last(
            r#"(module (tag $e (param i32)) (func $f)
                 (func
                   try
                     try call $f end
                     try call $f end
                   catch $e
                     drop
                   catch_all
                   end))"#,
        );
        let [first, second] = [dispatch_at(&flow, 0), dispatch_at(&flow, 1)];
        let (Some(first_handlers), Some(second_handlers)) = (&first.handlers, &second.handlers)
        else {
            panic!("{first:?} {second:?}");
        };
        assert!(Arc::ptr_eq(first_handlers, second_handlers));
        assert_eq!(first.offered().clauses.len(), 2);
    }

    #[test]
    fn a_handler_that_puts_the_stack_pointer_back_first_says_from_which_local() {
        let restores = |body: &str| {
            let (_, flow) = recover_last(&format!(
                r#"(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096))
                     (global $other (mut i32) (i32.const 0)) (tag $e (param i32)) (func $f)
                     (func (local i32 i32 exnref) {body}))"#
            ));
            let offered = dispatch_at(&flow, 0).offered();
            offered
                .clauses
                .iter()
                .map(|clause| clause.restores)
                .collect::<Vec<_>>()
        };
        let legacy = |handler: &str| restores(&format!("try call $f {handler} end"));
        assert_eq!(
            legacy("catch_all local.get 0 global.set $__stack_pointer"),
            [Some(0)]
        );
        assert_eq!(
            legacy("catch $e local.set 1 local.get 0 global.set $__stack_pointer"),
            [Some(0)],
            "once what was caught is put away"
        );
        assert_eq!(
            legacy("catch $e drop local.get 0 global.set $__stack_pointer"),
            [Some(0)]
        );
        assert_eq!(
            legacy("catch $e local.set 0 local.get 0 global.set $__stack_pointer"),
            [None],
            "the local holds what was caught by then"
        );
        assert_eq!(
            legacy("catch_all call $f local.get 0 global.set $__stack_pointer"),
            [None],
            "a call first sees whatever the throw left"
        );
        assert_eq!(legacy("catch_all local.get 0 global.set $other"), [None]);
        assert_eq!(
            restores(
                "(block $h (result exnref) (try_table (catch_all_ref $h) (call $f)) (return))
                 (local.set 2) (local.get 1) (global.set $__stack_pointer)"
            ),
            [Some(1)],
            "a clause's label is where its handler starts"
        );
        assert_eq!(
            restores(
                "(block $h (result exnref) (try_table (catch_all_ref $h) (call $f)) (return))
                 (throw_ref)"
            ),
            [None]
        );
    }

    #[test]
    fn a_throw_nothing_catches_is_still_described() {
        let (_, flow) =
            recover_last(r#"(module (tag $e (param i32)) (func (throw $e (i32.const 1))))"#);
        let dispatch = dispatch_at(&flow, 0);
        assert_eq!((dispatch.tag, dispatch.carried), (Some(0), 1));
        assert!(dispatch.offered().clauses.is_empty());
    }

    #[test]
    fn a_rethrow_knows_the_tag_its_handler_caught_and_skips_that_handler() {
        let (_, flow) = recover_last(
            r#"(module (tag $a) (tag $b)
                 (func
                   try
                     try
                       call 0
                     catch $b
                       rethrow 0
                     end
                   catch $a
                   catch $b
                   end))"#,
        );
        let rethrow = flow
            .dispatch
            .values()
            .find(|dispatch| dispatch.tag.is_some())
            .expect("the rethrow");
        let [outer] = rethrow.offered().clauses[..] else {
            panic!("{rethrow:?}");
        };
        assert_eq!((rethrow.tag, outer.tag), (Some(1), Some(1)));
        assert_eq!(rethrow.caught, Some(1), "the inner handler is one try deep");
        assert_eq!(outer.keeps, Some(0));

        let call = flow
            .dispatch
            .values()
            .find(|dispatch| dispatch.tag.is_none())
            .expect("the call");
        let tags: Vec<_> = call
            .offered()
            .clauses
            .iter()
            .map(|clause| clause.tag)
            .collect();
        assert_eq!(
            tags,
            [Some(1), Some(0)],
            "the outer $b is shadowed by the inner one"
        );
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
