//! Shadow stack frames: which local holds a function's frame and how big it is, kept only where the
//! function gives back what it takes and never reads the stack pointer global once a call or a
//! throw may have moved it; a function that can return with the global moved moves it for callers,
//! an import gives it back as the C ABI has every callee do, and a call from code Asyncify left
//! alone never unwinds, so what it reaches is judged by how it returns normally

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use wasmparser::{BlockType, Catch, Operator, OperatorsReader};

use crate::cfg::block_arity;
use crate::insn::{self, Flow};
use crate::module::{Module, Resolved, Signature};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Symbol {
    Global(u32, i64),
    Constant(i64),
    NonZero,
    Unknown,
}

type Locals = BTreeMap<u32, Symbol>;

fn pop(stack: &mut Vec<Symbol>) -> Symbol {
    stack.pop().unwrap_or(Symbol::Unknown)
}

fn truth(value: Symbol) -> Option<bool> {
    match value {
        Symbol::Constant(value) => Some(value as i32 != 0),
        Symbol::NonZero => Some(true),
        Symbol::Global(..) | Symbol::Unknown => None,
    }
}

fn compare(stack: &mut Vec<Symbol>, equal: bool) {
    let (right, left) = (pop(stack), pop(stack));
    let same = match (left, right) {
        (Symbol::Constant(a), Symbol::Constant(b)) => Some(a as i32 == b as i32),
        (Symbol::NonZero, Symbol::Constant(zero)) | (Symbol::Constant(zero), Symbol::NonZero)
            if zero as i32 == 0 =>
        {
            Some(false)
        }
        _ => None,
    };
    stack.push(same.map_or(Symbol::Unknown, |same| {
        Symbol::Constant(i64::from(same == equal))
    }));
}

fn either(stack: &mut Vec<Symbol>) {
    let (right, left) = (pop(stack), pop(stack));
    stack.push(match (left, right) {
        (Symbol::Constant(a), Symbol::Constant(b)) => Symbol::Constant(i64::from((a | b) as i32)),
        _ if truth(left) == Some(true) || truth(right) == Some(true) => Symbol::NonZero,
        _ if truth(left) == Some(false) => right,
        _ if truth(right) == Some(false) => left,
        _ => Symbol::Unknown,
    });
}

fn push_unknown(stack: &mut Vec<Symbol>, count: u32) {
    stack.extend(std::iter::repeat_n(Symbol::Unknown, count as usize));
}

fn arithmetic(stack: &mut Vec<Symbol>, subtract: bool, wide: bool) {
    let apply = |a: i64, b: i64| {
        let value = if subtract {
            a.wrapping_sub(b)
        } else {
            a.wrapping_add(b)
        };
        if wide { value } else { i64::from(value as i32) }
    };
    let (right, left) = (pop(stack), pop(stack));
    stack.push(match (left, right) {
        (Symbol::Global(global, offset), Symbol::Constant(by)) => {
            Symbol::Global(global, apply(offset, by))
        }
        (Symbol::Constant(by), Symbol::Global(global, offset)) if !subtract => {
            Symbol::Global(global, apply(offset, by))
        }
        (Symbol::Constant(a), Symbol::Constant(b)) => Symbol::Constant(apply(a, b)),
        _ => Symbol::Unknown,
    });
}

fn evaluate(op: &Operator, stack: &mut Vec<Symbol>, locals: &mut Locals) -> bool {
    match *op {
        Operator::I32Const { value } => stack.push(Symbol::Constant(i64::from(value))),
        Operator::I64Const { value } => stack.push(Symbol::Constant(value)),
        Operator::I32Add => arithmetic(stack, false, false),
        Operator::I64Add => arithmetic(stack, false, true),
        Operator::I32Sub => arithmetic(stack, true, false),
        Operator::I64Sub => arithmetic(stack, true, true),
        Operator::I32Eqz => {
            let value = truth(pop(stack));
            stack.push(value.map_or(Symbol::Unknown, |value| Symbol::Constant(i64::from(!value))));
        }
        Operator::I32Eq => compare(stack, true),
        Operator::I32Ne => compare(stack, false),
        Operator::I32Or => either(stack),
        Operator::LocalGet { local_index } => {
            stack.push(locals.get(&local_index).copied().unwrap_or(Symbol::Unknown));
        }
        Operator::LocalSet { local_index } | Operator::LocalTee { local_index } => {
            let value = pop(stack);
            if matches!(op, Operator::LocalTee { .. }) {
                stack.push(value);
            }
            match value {
                Symbol::Unknown => locals.remove(&local_index),
                known => locals.insert(local_index, known),
            };
        }
        _ => {
            if insn::flow(op) != Flow::Normal {
                return false;
            }
            let Some(arity) = insn::arity(op) else {
                return false;
            };
            for _ in 0..arity.pops {
                pop(stack);
            }
            push_unknown(stack, arity.pushes);
        }
    }
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Save {
    pub global: u32,
    pub local: u32,
    pub size: u64,
    pub at: usize,
}

#[derive(Default)]
pub struct EntryRegion {
    stack: Vec<Symbol>,
    locals: Locals,
    moved: BTreeMap<u32, Symbol>,
}

impl EntryRegion {
    pub fn step(&mut self, op: &Operator, at: usize, saves: &mut Vec<Save>) -> bool {
        match *op {
            Operator::GlobalGet { global_index } => {
                let value = self.moved.get(&global_index).copied();
                self.stack
                    .push(value.unwrap_or(Symbol::Global(global_index, 0)));
                return true;
            }
            Operator::GlobalSet { global_index } => {
                let value = match pop(&mut self.stack) {
                    Symbol::Global(global, offset) if global == global_index => {
                        Symbol::Global(global, offset)
                    }
                    _ => Symbol::Unknown,
                };
                self.moved.insert(global_index, value);
                return true;
            }
            _ => {}
        }
        if !evaluate(op, &mut self.stack, &mut self.locals) {
            return false;
        }
        if let Operator::LocalSet { local_index } | Operator::LocalTee { local_index } = *op
            && let Some(&Symbol::Global(global, offset)) = self.locals.get(&local_index)
            && offset <= 0
        {
            record(
                saves,
                Save {
                    global,
                    local: local_index,
                    size: offset.unsigned_abs(),
                    at,
                },
            );
        }
        true
    }
}

fn record(saves: &mut Vec<Save>, found: Save) {
    match saves.iter_mut().find(|save| save.global == found.global) {
        None => saves.push(found),
        Some(saved) if saved.size == 0 && found.size != 0 => *saved = found,
        Some(_) => {}
    }
}

#[derive(Default)]
pub struct Saves {
    seen: Seen,
}

#[derive(Default)]
enum Seen {
    #[default]
    Nothing,
    Pointer(u32),
    Constant(u32, i64),
    Allocated(u32, u64),
}

impl Saves {
    pub fn step(&mut self, op: &Operator, at: usize, saves: &mut Vec<Save>) {
        let taken = std::mem::take(&mut self.seen);
        let (global, size, local) = match (taken, op) {
            (_, Operator::GlobalGet { global_index }) => {
                self.seen = Seen::Pointer(*global_index);
                return;
            }
            (Seen::Pointer(global), Operator::I32Const { value }) => {
                self.seen = Seen::Constant(global, i64::from(*value));
                return;
            }
            (Seen::Pointer(global), Operator::I64Const { value }) => {
                self.seen = Seen::Constant(global, *value);
                return;
            }
            (Seen::Constant(global, by), Operator::I32Sub | Operator::I64Sub) if by >= 0 => {
                self.seen = Seen::Allocated(global, by.unsigned_abs());
                return;
            }
            (Seen::Constant(global, by), Operator::I32Add | Operator::I64Add) if by <= 0 => {
                self.seen = Seen::Allocated(global, by.unsigned_abs());
                return;
            }
            (
                Seen::Pointer(global),
                Operator::LocalSet { local_index } | Operator::LocalTee { local_index },
            ) => (global, 0, *local_index),
            (
                Seen::Allocated(global, size),
                Operator::LocalSet { local_index } | Operator::LocalTee { local_index },
            ) => (global, size, *local_index),
            _ => return,
        };
        record(
            saves,
            Save {
                global,
                local,
                size,
                at,
            },
        );
    }
}

#[derive(Debug, Default)]
pub struct Effects {
    sets: BTreeSet<u32>,
    calls: BTreeSet<u32>,
    taken: BTreeSet<u32>,
    indirect: BTreeSet<u32>,
    catches: bool,
    unseen: bool,
}

enum Callee {
    Direct(u32),
    Table(u32),
    Reference(u32),
}

fn callee(op: &Operator) -> Option<Callee> {
    match *op {
        Operator::Call { function_index } | Operator::ReturnCall { function_index } => {
            Some(Callee::Direct(function_index))
        }
        Operator::CallIndirect { type_index, .. }
        | Operator::ReturnCallIndirect { type_index, .. } => Some(Callee::Table(type_index)),
        Operator::CallRef { type_index } | Operator::ReturnCallRef { type_index } => {
            Some(Callee::Reference(type_index))
        }
        _ => None,
    }
}

impl Effects {
    pub fn step(&mut self, op: &Operator) {
        if let Some(global) = written_global(op) {
            self.sets.insert(global);
        }
        match callee(op) {
            Some(Callee::Direct(function)) => {
                self.calls.insert(function);
            }
            Some(Callee::Table(ty) | Callee::Reference(ty)) => {
                self.indirect.insert(ty);
            }
            None => {}
        }
        match *op {
            Operator::RefFunc { function_index } => {
                self.taken.insert(function_index);
            }
            Operator::Catch { .. } | Operator::CatchAll => self.catches = true,
            Operator::TryTable { ref try_table } => {
                self.catches |= !try_table.catches.is_empty();
            }
            Operator::Suspend { .. }
            | Operator::Resume { .. }
            | Operator::ResumeThrow { .. }
            | Operator::ResumeThrowRef { .. }
            | Operator::Switch { .. } => self.unseen = true,
            _ => {}
        }
    }

    pub fn unread(&mut self) {
        self.unseen = true;
    }

    pub fn writes(&self, global: u32) -> bool {
        self.sets.contains(&global)
    }
}

fn written_global(op: &Operator) -> Option<u32> {
    match *op {
        Operator::GlobalSet { global_index }
        | Operator::GlobalAtomicSet { global_index, .. }
        | Operator::GlobalAtomicRmwAdd { global_index, .. }
        | Operator::GlobalAtomicRmwSub { global_index, .. }
        | Operator::GlobalAtomicRmwAnd { global_index, .. }
        | Operator::GlobalAtomicRmwOr { global_index, .. }
        | Operator::GlobalAtomicRmwXor { global_index, .. }
        | Operator::GlobalAtomicRmwXchg { global_index, .. }
        | Operator::GlobalAtomicRmwCmpxchg { global_index, .. } => Some(global_index),
        _ => None,
    }
}

struct CallGraph {
    callers: BTreeMap<u32, Vec<u32>>,
    indirect: Vec<(u32, BTreeSet<u32>)>,
    taken: BTreeSet<u32>,
    seeds: Vec<u32>,
}

impl CallGraph {
    fn new(
        effects: &BTreeMap<u32, Effects>,
        pointer: u32,
        mut taken: BTreeSet<u32>,
        aliases: &BTreeSet<u32>,
    ) -> Self {
        let mut callers: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (index, effect) in effects {
            for callee in &effect.calls {
                callers.entry(*callee).or_default().push(*index);
            }
            taken.extend(&effect.taken);
        }
        let seeds = effects
            .iter()
            .filter(|(_, effect)| {
                effect.unseen
                    || effect.catches
                    || effect.sets.contains(&pointer)
                    || !effect.sets.is_disjoint(aliases)
            })
            .map(|(index, _)| *index)
            .collect();
        Self {
            callers,
            indirect: effects
                .iter()
                .filter(|(_, effect)| !effect.indirect.is_empty())
                .map(|(index, effect)| (*index, effect.indirect.clone()))
                .collect(),
            taken,
            seeds,
        }
    }
}

const MAX_RECHECKS: u32 = 16;

const REWALKS: usize = 4;

pub fn frames(
    module: &Module,
    pointer: u32,
    candidates: BTreeMap<u32, (OperatorsReader, Vec<Save>)>,
    effects: &BTreeMap<u32, Effects>,
    taken: BTreeSet<u32>,
    aliases: &BTreeSet<u32>,
    asyncify: Option<u32>,
) -> BTreeMap<u32, ((u32, u64), bool)> {
    let graph = CallGraph::new(effects, pointer, taken, aliases);
    let settled = |asyncify| settle(module, pointer, &candidates, &graph, aliases, asyncify);
    let unaware = settled(None);
    let Some(state) = asyncify else {
        return unaware;
    };
    let instrumented: BTreeSet<u32> = candidates
        .iter()
        .filter(|(_, (reader, _))| {
            reader.clone().into_iter().any(|op| {
                matches!(op, Ok(Operator::GlobalGet { global_index }) if global_index == state)
            })
        })
        .map(|(index, _)| *index)
        .collect();
    let mut framed: BTreeMap<_, _> = settled(Some(state))
        .into_iter()
        .filter(|(index, _)| !instrumented.contains(index))
        .collect();
    framed.extend(
        unaware
            .into_iter()
            .filter(|(index, _)| instrumented.contains(index)),
    );
    framed
}

fn settle(
    module: &Module,
    pointer: u32,
    candidates: &BTreeMap<u32, (OperatorsReader, Vec<Save>)>,
    graph: &CallGraph,
    aliases: &BTreeSet<u32>,
    asyncify: Option<u32>,
) -> BTreeMap<u32, ((u32, u64), bool)> {
    let frame = |index: u32, reach: &Reach| {
        let (reader, saves) = &candidates[&index];
        saves.iter().find_map(|save| {
            let even = consistent(
                reader.clone(),
                pointer,
                *save,
                module,
                reach,
                aliases,
                asyncify,
            )?;
            Some(((save.local, save.size), even))
        })
    };

    let mut movers = BTreeSet::new();
    let mut indirect = Indirect::default();
    let optimistic = Reach {
        movers: &movers,
        indirect: &indirect,
    };
    let mut framed = BTreeMap::new();
    let mut balanced = BTreeSet::new();
    for &index in candidates.keys() {
        if let Some((found, even)) = frame(index, &optimistic) {
            framed.insert(index, found);
            if even {
                balanced.insert(index);
            }
        }
    }
    let mut rechecked: HashMap<u32, u32> = HashMap::new();
    let size = |index: u32| {
        candidates[&index]
            .0
            .get_binary_reader()
            .bytes_remaining()
            .max(1)
    };
    let mut allowance = candidates
        .keys()
        .map(|&index| size(index))
        .sum::<usize>()
        .saturating_mul(REWALKS);
    let mut pending = graph.seeds.clone();
    loop {
        let mut recheck = BTreeSet::new();
        while let Some(index) = pending.pop() {
            if balanced.contains(&index) || !movers.insert(index) {
                continue;
            }
            let mut reached: Vec<u32> = graph.callers.get(&index).cloned().unwrap_or_default();
            if graph.taken.contains(&index) && indirect.admit(module, index) {
                reached.extend(
                    graph
                        .indirect
                        .iter()
                        .filter(|(_, types)| types.iter().any(|ty| indirect.reaches(module, *ty)))
                        .map(|(caller, _)| *caller),
                );
            }
            for caller in reached {
                if framed.contains_key(&caller) {
                    recheck.insert(caller);
                }
                if !balanced.contains(&caller) {
                    pending.push(caller);
                }
            }
        }
        if recheck.is_empty() {
            return framed
                .into_iter()
                .map(|(index, frame)| (index, (frame, balanced.contains(&index))))
                .collect();
        }
        let reach = Reach {
            movers: &movers,
            indirect: &indirect,
        };
        for index in recheck {
            let count = rechecked.entry(index).or_default();
            *count += 1;
            let affordable = allowance >= size(index);
            allowance = allowance.saturating_sub(size(index));
            let found = (*count <= MAX_RECHECKS && affordable)
                .then(|| frame(index, &reach))
                .flatten();
            match found {
                Some((found, even)) => {
                    framed.insert(index, found);
                    if !even {
                        balanced.remove(&index);
                    }
                }
                None => {
                    framed.remove(&index);
                    balanced.remove(&index);
                }
            }
            if !balanced.contains(&index) {
                pending.push(index);
            }
        }
    }
}

struct Reach<'a> {
    movers: &'a BTreeSet<u32>,
    indirect: &'a Indirect,
}

#[derive(Default)]
struct Indirect {
    any: bool,
    signatures: Vec<Signature>,
}

impl Indirect {
    fn reaches(&self, module: &Module, ty: u32) -> bool {
        self.any
            || module
                .type_signature(ty)
                .is_none_or(|signature| self.signatures.contains(signature))
    }

    fn admit(&mut self, module: &Module, function: u32) -> bool {
        match module.signature(function) {
            Some(signature) if !self.any && !self.signatures.contains(signature) => {
                self.signatures.push(signature.clone());
                true
            }
            Some(_) => false,
            None => !std::mem::replace(&mut self.any, true),
        }
    }
}

const WORK_PER_OPERATOR: usize = 256;

const MAX_OPERANDS: usize = 1 << 16;

fn consistent(
    reader: OperatorsReader,
    pointer: u32,
    save: Save,
    module: &Module,
    reach: &Reach,
    aliases: &BTreeSet<u32>,
    asyncify: Option<u32>,
) -> Option<bool> {
    let ops = reader.into_iter().collect::<Result<Vec<_>, _>>().ok()?;
    let size = i64::try_from(save.size).ok()?;
    let mut budget = ops.len().saturating_mul(WORK_PER_OPERATOR);
    let Writes { protected, looped } = writes(&ops, &mut budget)?;
    let mut walk = Walk {
        pointer,
        aliases,
        asyncify,
        holder: save.local,
        saved: (save.at, Symbol::Global(pointer, -size)),
        established: false,
        unbalanced: false,
        module,
        reach,
        protected,
        looped,
        varying: HashMap::new(),
        stale_loops: HashSet::new(),
        unsettled_loops: HashSet::new(),
        budget,
        stack: Vec::new(),
        state: Some(State {
            depth: Some(0),
            locals: Locals::new(),
            stale: false,
        }),
        controls: vec![Control {
            kind: Kind::Function,
            start: 0,
            height: 0,
            params: 0,
            results: 0,
            stack: Vec::new(),
            entry: None,
            arrived: None,
            otherwise: false,
        }],
    };
    let finished = walk.run(&ops).is_ok() && walk.established;
    finished.then_some(!walk.unbalanced)
}

struct Writes {
    protected: HashMap<usize, BTreeSet<u32>>,
    looped: HashMap<usize, BTreeSet<u32>>,
}

fn writes(ops: &[Operator], budget: &mut usize) -> Option<Writes> {
    #[derive(Default)]
    struct Open {
        opener: usize,
        written: BTreeSet<u32>,
        thrown: BTreeSet<u32>,
        pending: BTreeSet<u32>,
        throws: bool,
        caught: Option<BTreeSet<u32>>,
    }

    let mut protected = HashMap::new();
    let mut looped = HashMap::new();
    let mut open: Vec<Open> = Vec::new();
    for (at, op) in ops.iter().enumerate() {
        match op {
            Operator::Block { .. }
            | Operator::Loop { .. }
            | Operator::If { .. }
            | Operator::Try { .. }
            | Operator::TryTable { .. } => open.push(Open {
                opener: at,
                ..Open::default()
            }),
            Operator::Catch { .. } | Operator::CatchAll => {
                if let Some(innermost) = open.last_mut()
                    && innermost.caught.is_none()
                {
                    charge(budget, innermost.thrown.len())?;
                    innermost.caught = Some(innermost.thrown.clone());
                }
            }
            Operator::LocalSet { local_index } | Operator::LocalTee { local_index } => {
                if let Some(innermost) = open.last_mut() {
                    innermost.written.insert(*local_index);
                    innermost.pending.insert(*local_index);
                }
            }
            op if insn::raises(op) => {
                if let Some(innermost) = open.last_mut() {
                    let pending = std::mem::take(&mut innermost.pending);
                    merge(&mut innermost.thrown, pending, budget)?;
                    innermost.throws = true;
                }
            }
            Operator::End | Operator::Delegate { .. } => {
                let Some(mut closed) = open.pop() else {
                    continue;
                };
                let mut thrown = std::mem::take(&mut closed.thrown);
                match ops[closed.opener] {
                    Operator::Loop { .. } => {
                        charge(budget, closed.written.len())?;
                        if closed.throws {
                            merge(&mut thrown, closed.written.clone(), budget)?;
                        }
                        looped.insert(closed.opener, closed.written.clone());
                    }
                    Operator::Try { .. } | Operator::TryTable { .. } => {
                        charge(budget, thrown.len())?;
                        let exposed = closed.caught.take().unwrap_or_else(|| thrown.clone());
                        protected.insert(closed.opener, exposed);
                    }
                    _ => {}
                }
                if let Some(parent) = open.last_mut() {
                    if closed.throws {
                        let pending = std::mem::take(&mut parent.pending);
                        merge(&mut parent.thrown, pending, budget)?;
                        parent.throws = true;
                    }
                    merge(&mut parent.thrown, thrown, budget)?;
                    merge(&mut parent.pending, closed.pending, budget)?;
                    merge(&mut parent.written, closed.written, budget)?;
                }
            }
            _ => {}
        }
    }
    Some(Writes { protected, looped })
}

fn charge(budget: &mut usize, units: usize) -> Option<()> {
    *budget = budget.checked_sub(units)?;
    Some(())
}

fn merge(into: &mut BTreeSet<u32>, mut from: BTreeSet<u32>, budget: &mut usize) -> Option<()> {
    if from.len() > into.len() {
        std::mem::swap(into, &mut from);
    }
    charge(budget, from.len())?;
    into.extend(from);
    Some(())
}

#[derive(Debug, Clone)]
struct State {
    depth: Option<i64>,
    locals: Locals,
    stale: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Function,
    Block,
    Loop,
    If,
    Try,
}

struct Control {
    kind: Kind,
    start: usize,
    height: usize,
    params: u32,
    results: u32,
    stack: Vec<Symbol>,
    entry: Option<State>,
    arrived: Option<State>,
    otherwise: bool,
}

struct Inconsistent;

enum Closed {
    Function,
    Block,
    Again(usize),
}

fn join(into: &mut Option<State>, from: State) {
    let Some(existing) = into else {
        *into = Some(from);
        return;
    };
    if existing.depth != from.depth {
        existing.depth = None;
    }
    existing.stale |= from.stale;
    existing
        .locals
        .retain(|local, value| from.locals.get(local) == Some(value));
}

struct Walk<'a> {
    pointer: u32,
    aliases: &'a BTreeSet<u32>,
    asyncify: Option<u32>,
    holder: u32,
    saved: (usize, Symbol),
    established: bool,
    unbalanced: bool,
    module: &'a Module,
    reach: &'a Reach<'a>,
    protected: HashMap<usize, BTreeSet<u32>>,
    looped: HashMap<usize, BTreeSet<u32>>,
    varying: HashMap<usize, BTreeSet<u32>>,
    stale_loops: HashSet<usize>,
    unsettled_loops: HashSet<usize>,
    budget: usize,
    stack: Vec<Symbol>,
    state: Option<State>,
    controls: Vec<Control>,
}

impl Walk<'_> {
    fn charge(&mut self, units: usize) -> Result<(), Inconsistent> {
        charge(&mut self.budget, units).ok_or(Inconsistent)
    }

    fn pushed(&mut self, count: u32) -> Result<(), Inconsistent> {
        self.charge(count as usize)?;
        if self.stack.len().saturating_add(count as usize) > MAX_OPERANDS {
            return Err(Inconsistent);
        }
        push_unknown(&mut self.stack, count);
        Ok(())
    }

    fn charge_copies(&mut self, copies: usize) -> Result<(), Inconsistent> {
        let size = self.stack.len() + self.state.as_ref().map_or(0, |state| state.locals.len());
        self.charge(copies.saturating_mul(size))
    }

    fn depth(&self, value: Option<&Symbol>) -> Option<i64> {
        match value {
            Some(Symbol::Global(global, depth)) if *global == self.pointer => Some(*depth),
            _ => None,
        }
    }

    fn without(&self, forgotten: Option<&BTreeSet<u32>>) -> Option<State> {
        let mut state = self.state.clone()?;
        for local in forgotten.into_iter().flatten() {
            state.locals.remove(local);
        }
        Some(state)
    }

    fn handler(&self, entry: &State) -> State {
        State {
            depth: self.depth(entry.locals.get(&self.holder)),
            locals: entry.locals.clone(),
            stale: true,
        }
    }

    fn leaves(&mut self) -> Result<(), Inconsistent> {
        match &self.state {
            Some(state) if state.depth != Some(0) => Err(Inconsistent),
            Some(state) => {
                self.unbalanced |= state.stale;
                Ok(())
            }
            None => Ok(()),
        }
    }

    fn branch(&mut self, relative_depth: u32) -> Result<(), Inconsistent> {
        self.charge_copies(1)?;
        let Some(state) = self.state.clone() else {
            return Ok(());
        };
        let index = self
            .controls
            .len()
            .checked_sub(relative_depth as usize + 1)
            .ok_or(Inconsistent)?;
        if self.controls[index].kind == Kind::Function {
            return self.leaves();
        }
        join(&mut self.controls[index].arrived, state);
        Ok(())
    }

    fn open(&mut self, kind: Kind, blockty: &BlockType, at: usize) -> Result<(), Inconsistent> {
        let (params, results) = block_arity(blockty, Some(self.module)).ok_or(Inconsistent)?;
        let taken = match kind {
            Kind::If => truth(pop(&mut self.stack)),
            _ => None,
        };
        let forgotten = match kind {
            Kind::Loop => self.varying.get(&at),
            Kind::Try => self.protected.get(&at),
            _ => None,
        };
        self.charge(forgotten.map_or(0, BTreeSet::len))?;
        self.charge_copies(1)?;
        let entry = match kind {
            Kind::Loop => self.without(self.varying.get(&at)).map(|mut entry| {
                entry.stale |= self.stale_loops.contains(&at);
                if self.unsettled_loops.contains(&at) {
                    entry.depth = None;
                }
                entry
            }),
            Kind::Try => self.without(self.protected.get(&at)),
            Kind::If if taken == Some(true) => None,
            Kind::If => self.state.clone(),
            Kind::Block | Kind::Function => None,
        };
        if taken == Some(false) {
            self.state = None;
        }
        let height = self.stack.len().saturating_sub(params as usize);
        let stack = match kind {
            Kind::Loop => {
                self.state = entry.clone();
                self.stack.truncate(height);
                self.pushed(params)?;
                self.stack.clone()
            }
            _ => Vec::new(),
        };
        self.controls.push(Control {
            kind,
            start: at,
            height,
            params,
            results,
            stack,
            entry,
            arrived: None,
            otherwise: false,
        });
        Ok(())
    }

    fn catches(&mut self, catches: &[Catch], at: usize) -> Result<(), Inconsistent> {
        self.charge(self.protected.get(&at).map_or(0, BTreeSet::len))?;
        self.charge_copies(1)?;
        let Some(entry) = self.without(self.protected.get(&at)) else {
            return Ok(());
        };
        let reached = Some(self.handler(&entry));
        let current = std::mem::replace(&mut self.state, reached);
        for catch in catches {
            self.branch(insn::catch_label(catch))?;
        }
        self.state = current;
        Ok(())
    }

    fn arm(&mut self, handled: Option<u32>) -> Result<(), Inconsistent> {
        self.charge_copies(2)?;
        let arriving = self.state.take();
        let control = self.controls.last_mut().ok_or(Inconsistent)?;
        if let Some(state) = arriving {
            join(&mut control.arrived, state);
        }
        control.otherwise = true;
        let (height, pushed) = (control.height, handled.unwrap_or(control.params));
        let entry = control.entry.clone();
        self.state = match handled {
            Some(_) => entry.map(|entry| self.handler(&entry)),
            None => entry,
        };
        self.stack.truncate(height);
        self.pushed(pushed)
    }

    fn widen(&mut self) -> Option<usize> {
        let control = self.controls.last_mut()?;
        if control.kind != Kind::Loop {
            return None;
        }
        let back = control.arrived.take()?;
        let header = control.entry.as_mut()?;
        let mut varying: BTreeSet<u32> = header
            .locals
            .iter()
            .filter(|(local, value)| back.locals.get(local) != Some(value))
            .map(|(local, _)| *local)
            .collect();
        let staled = back.stale && !header.stale;
        let unsettled = header.depth.is_some() && back.depth != header.depth;
        if varying.is_empty() && !staled && !unsettled {
            return None;
        }
        if staled {
            header.stale = true;
            self.stale_loops.insert(control.start);
        }
        if unsettled {
            header.depth = None;
            self.unsettled_loops.insert(control.start);
        }
        if !varying.is_empty() && self.varying.contains_key(&control.start) {
            varying.extend(self.looped.get(&control.start).into_iter().flatten());
        }
        for local in &varying {
            header.locals.remove(local);
        }
        if !varying.is_empty() {
            self.varying
                .entry(control.start)
                .or_default()
                .extend(varying);
        }
        self.state = Some(header.clone());
        self.stack.clone_from(&control.stack);
        Some(control.start)
    }

    fn close(&mut self) -> Result<Closed, Inconsistent> {
        self.charge_copies(3)?;
        if let Some(start) = self.widen() {
            return Ok(Closed::Again(start));
        }
        let mut control = self.controls.pop().ok_or(Inconsistent)?;
        let falling = self.state.take();
        self.state = match control.kind {
            Kind::Loop => falling,
            _ => {
                if let Some(state) = falling {
                    join(&mut control.arrived, state);
                }
                if control.kind == Kind::If
                    && !control.otherwise
                    && let Some(entry) = control.entry.take()
                {
                    join(&mut control.arrived, entry);
                }
                control.arrived
            }
        };
        if control.kind == Kind::Function {
            self.leaves()?;
            return Ok(Closed::Function);
        }
        self.stack.truncate(control.height);
        self.pushed(control.results)?;
        Ok(Closed::Block)
    }

    fn run(&mut self, ops: &[Operator]) -> Result<(), Inconsistent> {
        let mut at = 0;
        while let Some(op) = ops.get(at) {
            self.charge(1)?;
            match op {
                Operator::Block { blockty } => self.open(Kind::Block, blockty, at)?,
                Operator::Loop { blockty } => self.open(Kind::Loop, blockty, at)?,
                Operator::If { blockty } => self.open(Kind::If, blockty, at)?,
                Operator::Try { blockty } => self.open(Kind::Try, blockty, at)?,
                Operator::TryTable { try_table } => {
                    self.catches(&try_table.catches, at)?;
                    self.open(Kind::Block, &try_table.ty, at)?;
                }
                Operator::Else => self.arm(None)?,
                Operator::Catch { tag_index } => {
                    let carried = self.module.tag_arity(*tag_index).ok_or(Inconsistent)?;
                    self.arm(Some(carried))?;
                }
                Operator::CatchAll => self.arm(Some(0))?,
                Operator::End | Operator::Delegate { .. } => match self.close()? {
                    Closed::Function => return Ok(()),
                    Closed::Again(start) => at = start,
                    Closed::Block => {}
                },
                _ if self.state.is_some() => {
                    if at == self.saved.0 {
                        if self.stack.last() != Some(&self.saved.1) {
                            return Err(Inconsistent);
                        }
                        self.established = true;
                    }
                    self.step(op)?;
                }
                _ => {}
            }
            at += 1;
        }
        Err(Inconsistent)
    }

    fn step(&mut self, op: &Operator) -> Result<(), Inconsistent> {
        match op {
            Operator::Br { relative_depth } => {
                self.branch(*relative_depth)?;
                self.state = None;
            }
            Operator::BrIf { relative_depth } => match truth(pop(&mut self.stack)) {
                Some(false) => {}
                Some(true) => {
                    self.branch(*relative_depth)?;
                    self.state = None;
                }
                None => self.branch(*relative_depth)?,
            },
            Operator::BrTable { targets } => {
                pop(&mut self.stack);
                let depths: BTreeSet<u32> = targets
                    .targets()
                    .chain([Ok(targets.default())])
                    .collect::<Result<_, _>>()
                    .map_err(|_| Inconsistent)?;
                for depth in depths {
                    self.branch(depth)?;
                }
                self.state = None;
            }
            Operator::Return
            | Operator::ReturnCall { .. }
            | Operator::ReturnCallIndirect { .. }
            | Operator::ReturnCallRef { .. } => {
                self.leaves()?;
                self.unbalanced |= self.moves(op);
                self.state = None;
            }
            Operator::Unreachable
            | Operator::Throw { .. }
            | Operator::ThrowRef
            | Operator::Rethrow { .. } => self.state = None,
            Operator::Call { .. } | Operator::CallIndirect { .. } | Operator::CallRef { .. } => {
                let Some(Resolved::Call(call)) = self.module.resolve(op) else {
                    return Err(Inconsistent);
                };
                self.charge(call.arity.pops as usize)?;
                let first = self
                    .stack
                    .len()
                    .checked_sub(call.arity.pops as usize)
                    .and_then(|at| self.stack.get(at).copied());
                for _ in 0..call.arity.pops {
                    pop(&mut self.stack);
                }
                match first.filter(|_| call.returns_argument) {
                    Some(first) => self.stack.push(first),
                    None => self.pushed(call.arity.pushes)?,
                }
                if self.moves(op)
                    && let Some(state) = &mut self.state
                {
                    state.stale = true;
                }
            }
            Operator::GlobalGet { global_index } if *global_index == self.pointer => {
                if self.state.as_ref().is_some_and(|state| state.stale) {
                    return Err(Inconsistent);
                }
                let depth = self.state.as_ref().and_then(|state| state.depth);
                self.stack.push(match depth {
                    Some(depth) => Symbol::Global(self.pointer, depth),
                    None => Symbol::Unknown,
                });
            }
            Operator::GlobalSet { global_index } if *global_index == self.pointer => {
                let value = pop(&mut self.stack);
                let depth = self.depth(Some(&value));
                if let Some(state) = &mut self.state {
                    state.depth = depth;
                    state.stale = false;
                }
            }
            Operator::GlobalAtomicGet { global_index, .. } if *global_index == self.pointer => {
                return Err(Inconsistent);
            }
            Operator::GlobalGet { global_index } if Some(*global_index) == self.asyncify => {
                self.stack.push(Symbol::Constant(0));
            }
            _ if written_global(op)
                .is_some_and(|global| global == self.pointer || self.aliases.contains(&global)) =>
            {
                return Err(Inconsistent);
            }
            _ => self.evaluate(op)?,
        }
        Ok(())
    }

    fn moves(&self, op: &Operator) -> bool {
        match callee(op) {
            Some(Callee::Direct(function)) => self.reach.movers.contains(&function),
            Some(Callee::Table(ty) | Callee::Reference(ty)) => {
                self.reach.indirect.reaches(self.module, ty)
            }
            None => false,
        }
    }

    fn evaluate(&mut self, op: &Operator) -> Result<(), Inconsistent> {
        let Some(state) = &mut self.state else {
            return Ok(());
        };
        if !evaluate(op, &mut self.stack, &mut state.locals) {
            return Err(Inconsistent);
        }
        Ok(())
    }
}
