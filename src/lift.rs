//! LLIL lifting
//!
//! The operand stack lives in the `s` registers, one per height [`crate::cfg`] records for the
//! instruction, and the locals in the `l` registers; calls pass arguments in the `a` registers and
//! results in the `r` registers the calling convention names. `sp` is the module's stack pointer
//! global, in the functions that take their frame from it

use std::sync::atomic::{AtomicBool, Ordering};

use binaryninja::low_level_il::expression::ValueExpr;
use binaryninja::low_level_il::lifting::{
    LiftableLowLevelIL, LiftableLowLevelILWithSize, LowLevelILLabel,
};
use binaryninja::low_level_il::{
    LowLevelILMutableExpression, LowLevelILMutableFunction, LowLevelILRegisterKind,
};
use wasmparser::Operator;

use crate::arch::{
    ARGUMENT_REGISTERS, EXCEPTION_VALUE, LOCAL_REGISTERS, RESULT_REGISTERS, RETHROW, RegKind,
    STACK_REGISTERS, SUSPENDED, THROWN, WasmIntrinsic, WasmRegister, throw,
};
use crate::cfg::{Clause, Dispatch, Recovered, Terminator, Unwind};
use crate::insn::{Flow, Instruction, immediates, memarg};
use crate::module::{Call, Layout, Resolved, ValueKind};

/// Wide enough for `i64` and `f64`; a `v128` is carried as a slot-sized token between the
/// intrinsics that take and give it
pub const SLOT: u64 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// An address is as wide as the module makes it, and where the globals live follows the file
pub struct Model<'a> {
    pub addr: usize,
    pub layout: Layout,
    pub stack_pointer: Option<u32>,
    pub frame: Frame<'a>,
    pub height: Option<u32>,
    pub global: Option<ValueKind>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Frame<'a> {
    pub params: u32,
    pub locals: &'a [ValueKind],
    pub results: &'a [ValueKind],
    pub entry: bool,
    pub stack: Option<u32>,
    pub balanced: bool,
}

pub fn width(kind: ValueKind, pointer: usize) -> usize {
    match kind {
        ValueKind::Ref => pointer,
        ValueKind::V128 => SLOT as usize,
        kind => kind.size(),
    }
}

impl Model<'_> {
    fn register(self, kind: RegKind, width: usize) -> LowLevelILRegisterKind<WasmRegister> {
        LowLevelILRegisterKind::Arch(WasmRegister::sized(kind, width, self.addr))
    }

    fn stack(self, index: u32, width: usize) -> LowLevelILRegisterKind<WasmRegister> {
        self.register(RegKind::Stack(index), width)
    }

    fn argument(self, index: u32, width: usize) -> LowLevelILRegisterKind<WasmRegister> {
        self.register(RegKind::Argument(index), width)
    }

    fn result(self, index: u32, width: usize) -> LowLevelILRegisterKind<WasmRegister> {
        self.register(RegKind::Result(index), width)
    }

    fn intrinsic(self, id: u32) -> WasmIntrinsic {
        WasmIntrinsic {
            id,
            pointer: self.addr,
        }
    }

    fn local(self, index: u32, width: usize) -> LowLevelILRegisterKind<WasmRegister> {
        self.register(RegKind::Local(index), width)
    }

    fn pointer_register(self, kind: RegKind) -> LowLevelILRegisterKind<WasmRegister> {
        LowLevelILRegisterKind::Arch(WasmRegister::new(kind, self.addr))
    }

    fn tag(self, index: u32) -> u64 {
        self.layout.tag_address(index)
    }

    fn local_width(self, index: u32) -> usize {
        self.frame
            .locals
            .get(index as usize)
            .map_or(SLOT as usize, |kind| width(*kind, self.addr))
    }

    fn global_width(self) -> usize {
        self.global
            .map_or(SLOT as usize, |kind| width(kind, self.addr))
    }
}

fn slot<'a>(
    il: &'a LowLevelILMutableFunction,
    model: Model,
    index: u32,
    width: usize,
) -> LowLevelILMutableExpression<'a, ValueExpr> {
    if index < STACK_REGISTERS {
        il.reg(width, model.stack(index, width))
    } else {
        il.expression(il.undefined())
    }
}

fn set_slot(
    il: &LowLevelILMutableFunction,
    model: Model,
    index: u32,
    width: usize,
    value: LowLevelILMutableExpression<'_, ValueExpr>,
) {
    if index < STACK_REGISTERS {
        il.add_instruction(il.set_reg(width, model.stack(index, width), value));
    }
}

static CROWDED: AtomicBool = AtomicBool::new(false);
static SHORT: AtomicBool = AtomicBool::new(false);
static CROWDED_CALL: AtomicBool = AtomicBool::new(false);
static CROWDED_THROW: AtomicBool = AtomicBool::new(false);
static CROWDED_LOCALS: AtomicBool = AtomicBool::new(false);

fn warn_once(flag: &AtomicBool, message: std::fmt::Arguments) {
    if !flag.swap(true, Ordering::Relaxed) {
        tracing::warn!("{message}");
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
    dispatch: Option<&Dispatch>,
) {
    if model.frame.entry {
        prologue(il, model);
    }
    let sem = semantics(insn);

    // Calls come first, since a tail call is a terminator too and lifting it as one loses the
    // callee
    if let Some(Resolved::Call(call)) = resolved {
        lift_call(il, model, insn, call, dispatch);
        return;
    }
    if let Some(dispatch) = dispatch
        && matches!(
            insn.op,
            Operator::Throw { .. } | Operator::ThrowRef | Operator::Rethrow { .. }
        )
    {
        raise(il, model, insn, dispatch);
        return;
    }
    let suspends = recovered
        .is_some_and(|recovered| matches!(recovered.terminator, Terminator::Suspend { .. }));
    if let Some(recovered) = recovered
        && !suspends
    {
        lift_terminator(il, model, insn, recovered);
        return;
    }

    if let Sem::LocalGet(index) | Sem::LocalSet(index) | Sem::LocalTee(index) = sem
        && index >= LOCAL_REGISTERS
    {
        crowded_locals();
        if matches!(sem, Sem::LocalGet(_)) {
            give_up(il, model, insn, sem, resolved);
        } else {
            il.add_instruction(il.unimplemented());
        }
        return;
    }

    match model.height {
        Some(height) if fits(insn, sem, resolved, height) => {
            operate(il, model, insn, sem, resolved, height)
        }
        Some(height) => {
            let (flag, problem) = if height < demand(insn, sem, resolved).0 {
                (&SHORT, "fewer operands than it takes")
            } else {
                (
                    &CROWDED,
                    "an operand stack deeper than the registers that hold it",
                )
            };
            warn_once(
                flag,
                format_args!(
                    "wasm: an instruction with {problem} is not lifted, first at {:?}",
                    insn.mnemonic()
                ),
            );
            give_up(il, model, insn, sem, resolved);
        }
        None => give_up(il, model, insn, sem, resolved),
    }
    if let Some(dispatch) = dispatch
        && insn.flow().falls_through()
    {
        thrown_here(il, model, dispatch);
    }
    if let Some(recovered) = recovered {
        lift_terminator(il, model, insn, recovered);
    }
}

fn suspended(
    il: &LowLevelILMutableFunction,
    model: Model,
    recovered: &Recovered,
    handlers: &[(u32, Option<u64>)],
) {
    let tag = model.pointer_register(RegKind::Exception);
    il.add_instruction(il.intrinsic(
        [tag],
        model.intrinsic(SUSPENDED),
        std::iter::empty::<LowLevelILMutableExpression<'_, ValueExpr>>(),
    ));
    for (index, (handled, target)) in handlers.iter().enumerate() {
        let mut hit = LowLevelILLabel::new();
        let mut miss = LowLevelILLabel::new();
        let wanted = il.const_ptr_sized(model.addr, model.tag(*handled));
        il.add_instruction(il.if_expr(
            il.cmp_e(model.addr, il.reg(model.addr, tag), wanted),
            &mut hit,
            &mut miss,
        ));
        il.mark_label(&mut hit);
        let unwind = recovered.unwind(index);
        if let Some(base) = model
            .height
            .and_then(|height| height.checked_sub(unwind.drop))
        {
            for nth in 0..unwind.keep {
                let to = base.saturating_add(nth);
                set_slot(il, model, to, SLOT as usize, il.expression(il.undefined()));
            }
        }
        go(il, model, *target, None);
        il.mark_label(&mut miss);
    }
}

fn give_up(
    il: &LowLevelILMutableFunction,
    model: Model,
    insn: &Instruction,
    sem: Sem,
    resolved: Option<&Resolved>,
) {
    match insn.flow() {
        Flow::Trap => {
            il.add_instruction(il.no_ret());
            return;
        }
        Flow::Return => {
            leave(il, model, None);
            return;
        }
        _ => {}
    }
    if matches!(sem, Sem::Nop) && insn.flow().falls_through() {
        il.add_instruction(il.nop());
        return;
    }
    il.add_instruction(il.unimplemented());
    if let Some(height) = model.height {
        let (pops, pushes) = demand(insn, sem, resolved);
        let base = height.saturating_sub(pops);
        for index in (base..base.saturating_add(pushes)).take_while(|at| *at < STACK_REGISTERS) {
            set_slot(
                il,
                model,
                index,
                SLOT as usize,
                il.expression(il.undefined()),
            );
        }
    }
    if !matches!(insn.flow(), Flow::Normal | Flow::Call | Flow::IndirectCall) {
        il.add_instruction(il.jump(il.unimplemented()));
    }
}

fn operate(
    il: &LowLevelILMutableFunction,
    model: Model,
    insn: &Instruction,
    sem: Sem,
    resolved: Option<&Resolved>,
    height: u32,
) {
    let get = |index: u32, width: usize| il.reg(width, model.stack(index, width));
    let set = |index: u32, width: usize, value: LowLevelILMutableExpression<'_, ValueExpr>| {
        il.add_instruction(il.set_reg(width, model.stack(index, width), value));
    };

    match sem {
        Sem::Nop | Sem::Drop => il.add_instruction(il.nop()),
        Sem::Trap => il.add_instruction(il.no_ret()),
        Sem::Return => leave(il, model, Some(height)),
        Sem::Const { size, value } => set(height, size, il.const_int(size, value)),
        Sem::Null => set(height, model.addr, il.const_int(model.addr, 0)),
        Sem::IsNull => {
            let zero = il.const_int(model.addr, 0);
            let result = compare(il, Cmp::Eq, model.addr, get(height - 1, model.addr), zero);
            set(height - 1, RESULT_I32, result);
        }
        Sem::TableGet => {
            let slot = table_slot(il, model, height - 1);
            let value = il.expression(il.load(model.addr, slot));
            set(height - 1, model.addr, value);
        }
        Sem::TableSet => {
            let slot = table_slot(il, model, height - 2);
            let value = get(height - 1, model.addr);
            il.add_instruction(il.store(model.addr, slot, value));
        }
        Sem::Unary { size, kind } => {
            let result = unary(il, kind, size, get(height - 1, size));
            set(height - 1, size, result);
        }
        Sem::Binary { size, kind } => {
            let result = binary(il, kind, size, get(height - 2, size), get(height - 1, size));
            set(height - 2, size, result);
        }
        Sem::Compare { size, kind } => {
            // A comparison is built at the width of what it compares, but the answer is an `i32`
            let result = compare(il, kind, size, get(height - 2, size), get(height - 1, size));
            set(height - 2, RESULT_I32, result);
        }
        Sem::TestZero { size } => {
            let zero = il.const_int(size, 0);
            let result = compare(il, Cmp::Eq, size, get(height - 1, size), zero);
            set(height - 1, RESULT_I32, result);
        }
        Sem::Convert { from, to, kind } => {
            let value = if from < RESULT_I32 {
                il.expression(il.low_part(from, get(height - 1, RESULT_I32)))
            } else {
                get(height - 1, from)
            };
            set(height - 1, to, convert(il, kind, from, to, value));
        }
        Sem::Saturate(conversion) => {
            let value = get(height - 1, conversion.from);
            il.add_instruction(il.intrinsic(
                [model.stack(height - 1, conversion.to)],
                model.intrinsic(conversion.id),
                [value],
            ));
        }
        Sem::Load {
            access,
            result,
            extend,
            offset,
        } => {
            let address = linear_address(il, model, height - 1, offset);
            let loaded = il.expression(il.load(access, address));
            let value = match extend {
                Extend::None => loaded,
                Extend::Sign => il.expression(il.sx(result, loaded)),
                Extend::Zero => il.expression(il.zx(result, loaded)),
            };
            set(height - 1, result, value);
        }
        Sem::Store {
            access,
            value: size,
            offset,
        } => {
            let address = linear_address(il, model, height - 2, offset);
            let value = get(height - 1, size);
            let value = if access < size {
                il.expression(il.low_part(access, value))
            } else {
                value
            };
            il.add_instruction(il.store(access, address, value));
        }
        Sem::LocalGet(index) => {
            let width = model.local_width(index);
            set(height, width, il.reg(width, model.local(index, width)));
        }
        Sem::LocalSet(index) | Sem::LocalTee(index) => {
            let width = model.local_width(index);
            let value = get(height - 1, width);
            il.add_instruction(il.set_reg(width, model.local(index, width), value));
        }
        Sem::GlobalGet(index) if model.stack_pointer == Some(index) => {
            let sp = il.reg(model.addr, model.pointer_register(RegKind::Sp));
            set(height, model.addr, sp);
        }
        Sem::GlobalSet(index) if model.stack_pointer == Some(index) => {
            let value = get(height - 1, model.addr);
            il.add_instruction(il.set_reg(model.addr, model.pointer_register(RegKind::Sp), value));
        }
        Sem::GlobalGet(index) => {
            let width = model.global_width();
            let value = il.expression(il.load(width, global_address(il, model, index)));
            set(height, width, value);
        }
        Sem::GlobalSet(index) => {
            let width = model.global_width();
            let value = get(height - 1, width);
            il.add_instruction(il.store(width, global_address(il, model, index), value));
        }
        Sem::Select { values } => select(il, model, height, values),
        Sem::CondBranch | Sem::RefBranch => give_up(il, model, insn, sem, resolved),
        // Not an intrinsic: the count is per instruction and an intrinsic's prototype is per
        // operator, so the two could not agree
        Sem::Aggregate { pops, pushes } => aggregate(il, model, height, pops, pushes),
        Sem::Opaque => match resolved {
            // An aggregate whose width only the module knows, so `semantics` could not see it
            Some(Resolved::Aggregate(arity)) => {
                aggregate(il, model, height, arity.pops, arity.pushes)
            }
            Some(Resolved::Function(entry)) => {
                set(height, model.addr, il.const_ptr_sized(model.addr, *entry))
            }
            Some(Resolved::Switching(_)) => give_up(il, model, insn, sem, resolved),
            _ => opaque(il, model, insn, sem, height),
        },
        Sem::OtherMemory => give_up(il, model, insn, sem, resolved),
    }
}

fn fits(insn: &Instruction, sem: Sem, resolved: Option<&Resolved>, height: u32) -> bool {
    let (pops, pushes) = demand(insn, sem, resolved);
    height <= STACK_REGISTERS
        && height
            .checked_sub(pops)
            .is_some_and(|base| base.saturating_add(pushes) <= STACK_REGISTERS)
}

fn demand(insn: &Instruction, sem: Sem, resolved: Option<&Resolved>) -> (u32, u32) {
    match resolved {
        Some(Resolved::Call(call)) => (call.arity.pops, call.arity.pushes),
        Some(Resolved::Aggregate(arity) | Resolved::Switching(arity)) => (arity.pops, arity.pushes),
        Some(Resolved::Function(_)) => (0, 1),
        None => match sem {
            Sem::Aggregate { pops, pushes } => (pops, pushes),
            Sem::Select { values } => (values.saturating_mul(2).saturating_add(1), values),
            _ => {
                let fixed = insn
                    .arity()
                    .map_or((0, 0), |arity| (arity.pops, arity.pushes));
                (fixed.0.max(tested(&insn.op)), fixed.1)
            }
        },
    }
}

fn tested(op: &Operator) -> u32 {
    u32::from(
        matches!(op, Operator::If { .. })
            || matches!(
                crate::insn::flow(op),
                Flow::ConditionalBranch | Flow::IndirectBranch
            ),
    )
}

fn select(il: &LowLevelILMutableFunction, model: Model, height: u32, values: u32) {
    let first = height - 1 - 2 * values;
    let second = height - 1 - values;
    let condition = il.reg(RESULT_I32, model.stack(height - 1, RESULT_I32));
    let zero = il.const_int(RESULT_I32, 0);
    let chosen = il.expression(il.cmp_ne(RESULT_I32, condition, zero));

    let mut done = LowLevelILLabel::new();
    let mut other = LowLevelILLabel::new();
    il.add_instruction(il.if_expr(chosen, &mut done, &mut other));
    il.mark_label(&mut other);
    for nth in 0..values {
        let value = il.reg(SLOT as usize, model.stack(second + nth, SLOT as usize));
        il.add_instruction(il.set_reg(
            SLOT as usize,
            model.stack(first + nth, SLOT as usize),
            value,
        ));
    }
    il.add_instruction(il.goto(&mut done));
    il.mark_label(&mut done);
}

fn aggregate(il: &LowLevelILMutableFunction, model: Model, height: u32, pops: u32, pushes: u32) {
    let base = height - pops;
    for nth in 0..pushes {
        il.add_instruction(il.set_reg(
            SLOT as usize,
            model.stack(base + nth, SLOT as usize),
            il.undefined(),
        ));
    }
    if pushes == 0 {
        il.add_instruction(il.nop());
    }
}

fn lift_terminator(
    il: &LowLevelILMutableFunction,
    model: Model,
    insn: &Instruction,
    recovered: &Recovered,
) {
    let height = model.height;
    let consumed = u32::try_from(taken_pops(&insn.op)).unwrap_or(0);
    let taken_top = height.and_then(|height| height.checked_sub(consumed));

    match &recovered.terminator {
        Terminator::Jump(target) => {
            // `else`, `catch` and `catch_all` mark a place rather than consuming anything, but a
            // real `br` reaches here too and that one unwinds
            unwind(il, model, height, recovered.unwind(0));
            go(il, model, Some(*target), height);
        }
        Terminator::Branch { taken, not_taken } => {
            let condition = branch_condition(il, model, insn);
            let leaving = recovered.unwind(0);
            match (
                il.label_for_address(*taken),
                il.label_for_address(*not_taken),
            ) {
                (Some(mut hit), Some(mut miss)) if leaving.is_empty() => {
                    il.add_instruction(il.if_expr(condition, &mut hit, &mut miss));
                }
                // Either an edge needs work before it is taken or the core has no label for one of
                // these addresses; both go somewhere known, so both get a branch
                _ => {
                    let mut hit = LowLevelILLabel::new();
                    let mut miss = LowLevelILLabel::new();
                    il.add_instruction(il.if_expr(condition, &mut hit, &mut miss));

                    il.mark_label(&mut hit);
                    unwind(il, model, taken_top, leaving);
                    go(il, model, Some(*taken), taken_top);

                    il.mark_label(&mut miss);
                    go(il, model, Some(*not_taken), height);
                }
            }
        }
        Terminator::Table { targets, default } => {
            lift_table(il, model, targets, *default, recovered)
        }
        Terminator::Return if insn.flow() == Flow::TailCall => {
            il.add_instruction(il.tailcall(il.unimplemented()))
        }
        Terminator::Return => leave(il, model, taken_top),
        Terminator::ConditionalReturn { .. } => {
            let condition = branch_condition(il, model, insn);
            // Two labels of its own rather than addresses, since the return is not an instruction
            // anywhere in the function
            let mut leaving = LowLevelILLabel::new();
            let mut carry_on = LowLevelILLabel::new();
            il.add_instruction(il.if_expr(condition, &mut leaving, &mut carry_on));

            il.mark_label(&mut leaving);
            leave(il, model, taken_top);

            il.mark_label(&mut carry_on);
        }
        Terminator::Halt => il.add_instruction(il.no_ret()),
        Terminator::Unresolved => il.add_instruction(il.jump(il.unimplemented())),
        Terminator::Suspend { handlers, next } => {
            suspended(il, model, recovered, handlers);
            go(il, model, Some(*next), None);
        }
    }
}

fn branch_condition<'a>(
    il: &'a LowLevelILMutableFunction,
    model: Model,
    insn: &Instruction,
) -> LowLevelILMutableExpression<'a, ValueExpr> {
    let Some(top) = model.height.and_then(|height| height.checked_sub(1)) else {
        return il.unimplemented();
    };
    let (size, test_null) = match insn.op {
        Operator::BrIf { .. } | Operator::If { .. } => (RESULT_I32, false),
        Operator::BrOnNull { .. } => (model.addr, true),
        Operator::BrOnNonNull { .. } => (model.addr, false),
        // A type is not a value the IL has, and an unknown condition keeps both edges live
        _ => return il.unimplemented(),
    };
    let value = slot(il, model, top, size);
    let zero = il.const_int(size, 0);
    if test_null {
        il.expression(il.cmp_e(size, value, zero))
    } else {
        il.expression(il.cmp_ne(size, value, zero))
    }
}

fn lift_call(
    il: &LowLevelILMutableFunction,
    model: Model,
    insn: &Instruction,
    call: &Call,
    dispatch: Option<&Dispatch>,
) {
    let base = model
        .height
        .and_then(|height| height.checked_sub(call.arity.pops));
    let callee = model.height.and_then(|height| height.checked_sub(1));
    let target = match (&insn.op, callee) {
        (Operator::Call { .. } | Operator::ReturnCall { .. }, _) => match call.target {
            Some(entry) => il.const_ptr(entry),
            None => il.unimplemented(),
        },
        (Operator::CallRef { .. } | Operator::ReturnCallRef { .. }, Some(callee)) => {
            slot(il, model, callee, model.addr)
        }
        // An index is not an address: the callee is whatever the slot holds, and naming the index
        // would point the call into linear memory
        (
            Operator::CallIndirect { table_index: 0, .. }
            | Operator::ReturnCallIndirect { table_index: 0, .. },
            Some(callee),
        ) => {
            let at = table_slot(il, model, callee);
            il.expression(il.load(model.addr, at))
        }
        // Only the first table has a region, so an index into another one resolves to nothing
        _ => il.unimplemented(),
    };

    if call.params.len() > ARGUMENT_REGISTERS as usize || !returnable(&call.results) {
        crowded(il);
        if !insn.flow().falls_through() {
            stub(il, model.addr, &call.results);
            return;
        }
        if let Some(dispatch) = dispatch {
            thrown_here(il, model, dispatch);
        }
        if let Some(base) = base {
            for (nth, kind) in call.results.iter().enumerate() {
                let width = width(*kind, model.addr);
                let unknown = il.expression(il.undefined());
                set_slot(il, model, base + nth as u32, width, unknown);
            }
        }
        return;
    }
    for (nth, kind) in call.params.iter().enumerate() {
        let width = width(*kind, model.addr);
        let value = match base {
            Some(base) => slot(il, model, base + nth as u32, width),
            None => il.expression(il.undefined()),
        };
        il.add_instruction(il.set_reg(width, model.argument(nth as u32, width), value));
    }

    // A tail call replaces the frame, so there is no stack left to put results back on
    if !insn.flow().falls_through() {
        il.add_instruction(il.tailcall(target));
        return;
    }

    il.add_instruction(il.call(target));
    if let Some(dispatch) = dispatch {
        thrown_here(il, model, dispatch);
    }
    let Some(base) = base else {
        return;
    };
    for (nth, kind) in call
        .results
        .iter()
        .enumerate()
        .skip(usize::from(call.returns_argument))
    {
        let width = width(*kind, model.addr);
        let value = il.reg(width, model.result(nth as u32, width));
        set_slot(il, model, base + nth as u32, width, value);
    }
}

fn thrown_here(il: &LowLevelILMutableFunction, model: Model, dispatch: &Dispatch) {
    il.add_instruction(il.intrinsic(
        [model.pointer_register(RegKind::Exception)],
        model.intrinsic(THROWN),
        std::iter::empty::<LowLevelILMutableExpression<'_, ValueExpr>>(),
    ));
    let mut raised = LowLevelILLabel::new();
    let mut returned = LowLevelILLabel::new();
    let exception = il.reg(model.addr, model.pointer_register(RegKind::Exception));
    let none = il.const_int(model.addr, 0);
    il.add_instruction(il.if_expr(
        il.cmp_ne(model.addr, exception, none),
        &mut raised,
        &mut returned,
    ));
    il.mark_label(&mut raised);
    catch(il, model, dispatch, Carried::Registers);
    il.mark_label(&mut returned);
}

fn returnable(results: &[ValueKind]) -> bool {
    results.len() <= RESULT_REGISTERS as usize
}

fn crowded_locals() {
    warn_once(
        &CROWDED_LOCALS,
        format_args!("wasm: locals past the first {LOCAL_REGISTERS} are not lifted"),
    );
}

fn crowded(il: &LowLevelILMutableFunction) {
    warn_once(
        &CROWDED_CALL,
        format_args!(
            "wasm: calls and returns with more than {ARGUMENT_REGISTERS} arguments or \
             {RESULT_REGISTERS} results are not lifted"
        ),
    );
    il.add_instruction(il.unimplemented());
}

#[derive(Clone, Copy)]
enum Carried {
    Registers,
    Stack { tag: u32, top: Option<u32> },
}

fn raise(il: &LowLevelILMutableFunction, model: Model, insn: &Instruction, dispatch: &Dispatch) {
    let carried = match (&insn.op, dispatch.tag) {
        (Operator::Throw { .. }, Some(tag)) => Carried::Stack {
            tag,
            top: model.height.filter(|height| *height >= dispatch.carried),
        },
        (Operator::ThrowRef, _) => {
            let exnref = match model.height.and_then(|height| height.checked_sub(1)) {
                Some(top) => slot(il, model, top, model.addr),
                None => il.expression(il.undefined()),
            };
            let exception = model.pointer_register(RegKind::Exception);
            il.add_instruction(il.set_reg(model.addr, exception, exnref));
            let mut null = LowLevelILLabel::new();
            let mut thrown = LowLevelILLabel::new();
            let none = il.const_int(model.addr, 0);
            il.add_instruction(il.if_expr(
                il.cmp_e(model.addr, il.reg(model.addr, exception), none),
                &mut null,
                &mut thrown,
            ));
            il.mark_label(&mut null);
            il.add_instruction(il.no_ret());
            il.mark_label(&mut thrown);
            Carried::Registers
        }
        _ => {
            let rethrown = match (dispatch.caught, dispatch.tag) {
                (Some(caught), _) => {
                    il.reg(model.addr, model.pointer_register(RegKind::Caught(caught)))
                }
                (None, Some(tag)) => il.const_ptr_sized(model.addr, model.tag(tag)),
                (None, None) => il.expression(il.undefined()),
            };
            il.add_instruction(il.set_reg(
                model.addr,
                model.pointer_register(RegKind::Exception),
                rethrown,
            ));
            Carried::Registers
        }
    };

    catch(il, model, dispatch, carried);
}

fn propagate(il: &LowLevelILMutableFunction, model: Model, dispatch: &Dispatch, carried: Carried) {
    let none: Vec<LowLevelILRegisterKind<WasmRegister>> = Vec::new();
    match carried {
        Carried::Stack { tag, top } => match throw(dispatch.carried) {
            Some(id) => {
                let first = top.map(|top| top - dispatch.carried);
                let mut inputs = vec![il.const_ptr_sized(model.addr, model.tag(tag))];
                for nth in 0..dispatch.carried {
                    inputs.push(match first {
                        Some(first) => slot(il, model, first + nth, SLOT as usize),
                        None => il.expression(il.undefined()),
                    });
                }
                il.add_instruction(il.intrinsic(none, model.intrinsic(id), inputs));
            }
            None => {
                warn_once(
                    &CROWDED_THROW,
                    format_args!(
                        "wasm: throws carrying more than {ARGUMENT_REGISTERS} values are not lifted"
                    ),
                );
                il.add_instruction(il.unimplemented());
            }
        },
        Carried::Registers => {
            let exception = il.reg(model.addr, model.pointer_register(RegKind::Exception));
            il.add_instruction(il.intrinsic(none, model.intrinsic(RETHROW), [exception]));
        }
    }
    il.add_instruction(il.no_ret());
}

fn catch(il: &LowLevelILMutableFunction, model: Model, dispatch: &Dispatch, carried: Carried) {
    let offered = dispatch.offered();
    for clause in offered.clauses {
        let tested = clause.tag.filter(|_| dispatch.tag.is_none());
        let Some(tag) = tested else {
            deliver(il, model, dispatch, clause, carried);
            return;
        };
        let mut hit = LowLevelILLabel::new();
        let mut miss = LowLevelILLabel::new();
        let exception = il.reg(model.addr, model.pointer_register(RegKind::Exception));
        let wanted = il.const_ptr_sized(model.addr, model.tag(tag));
        il.add_instruction(il.if_expr(
            il.cmp_e(model.addr, exception, wanted),
            &mut hit,
            &mut miss,
        ));
        il.mark_label(&mut hit);
        deliver(il, model, dispatch, clause, carried);
        il.mark_label(&mut miss);
    }
    if offered.truncated {
        il.add_instruction(il.jump(il.unimplemented()));
    } else {
        propagate(il, model, dispatch, carried);
    }
}

fn deliver(
    il: &LowLevelILMutableFunction,
    model: Model,
    dispatch: &Dispatch,
    clause: &Clause,
    carried: Carried,
) {
    if let (Some(_), Some(holder)) = (model.stack_pointer, model.frame.stack) {
        let restored = clause
            .restores
            .or(model.frame.balanced.then_some(holder))
            .filter(|local| *local < LOCAL_REGISTERS);
        let frame = match restored {
            Some(local) => il.reg(model.addr, model.local(local, model.addr)),
            None => il.expression(il.undefined()),
        };
        il.add_instruction(il.set_reg(model.addr, model.pointer_register(RegKind::Sp), frame));
    }
    let exception = || match carried {
        Carried::Stack { tag, .. } => il.const_ptr_sized(model.addr, model.tag(tag)),
        Carried::Registers => il.reg(model.addr, model.pointer_register(RegKind::Exception)),
    };
    if let Some(keeps) = clause.keeps {
        il.add_instruction(il.set_reg(
            model.addr,
            model.pointer_register(RegKind::Caught(keeps)),
            exception(),
        ));
    }
    let Some(base) = clause.base.and_then(|base| u32::try_from(base).ok()) else {
        go(il, model, clause.target, None);
        return;
    };
    for nth in 0..clause.carried {
        let to = base.saturating_add(nth);
        match carried {
            Carried::Registers if to < STACK_REGISTERS => {
                let exception = il.reg(model.addr, model.pointer_register(RegKind::Exception));
                let index = il.const_int(4, u64::from(nth));
                il.add_instruction(il.intrinsic(
                    [model.stack(to, SLOT as usize)],
                    model.intrinsic(EXCEPTION_VALUE),
                    [exception, index],
                ));
            }
            Carried::Registers => {}
            Carried::Stack { top, .. } => {
                let value = match top {
                    Some(top) => slot(il, model, top - dispatch.carried + nth, SLOT as usize),
                    None => il.expression(il.undefined()),
                };
                set_slot(il, model, to, SLOT as usize, value);
            }
        }
    }
    let mut top = base.saturating_add(clause.carried);
    if clause.exnref {
        set_slot(il, model, top, model.addr, exception());
        top = top.saturating_add(1);
    }
    go(il, model, clause.target, Some(top));
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

/// LLIL exposes no jump table operation here, and a chain of tests says the same thing while
/// keeping every edge visible
fn lift_table(
    il: &LowLevelILMutableFunction,
    model: Model,
    targets: &[Option<u64>],
    default: Option<u64>,
    recovered: &Recovered,
) {
    let top = model.height.and_then(|height| height.checked_sub(1));

    for (entry, target) in targets.iter().enumerate() {
        let mut miss = LowLevelILLabel::new();
        let matches = match top {
            Some(top) => {
                let selector = slot(il, model, top, RESULT_I32);
                let wanted = il.const_int(RESULT_I32, entry as u64);
                il.expression(il.cmp_e(RESULT_I32, selector, wanted))
            }
            None => il.unimplemented(),
        };

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
                unwind(il, model, top, leaving);
                go(il, model, *target, top);
            }
        }
        il.mark_label(&mut miss);
    }

    unwind(il, model, top, recovered.unwind(targets.len()));
    go(il, model, default, top);
}

fn unwind(il: &LowLevelILMutableFunction, model: Model, top: Option<u32>, unwind: Unwind) {
    let Some(from) = top.and_then(|top| top.checked_sub(unwind.keep)) else {
        return;
    };
    let Some(to) = from.checked_sub(unwind.drop) else {
        return;
    };
    if to == from {
        return;
    }
    for nth in 0..unwind.keep {
        let value = slot(il, model, from + nth, SLOT as usize);
        set_slot(il, model, to + nth, SLOT as usize, value);
    }
}

fn leave(il: &LowLevelILMutableFunction, model: Model, top: Option<u32>) {
    let results = model.frame.results;
    if !returnable(results) {
        crowded(il);
    }
    let base = top
        .and_then(|top| top.checked_sub(results.len() as u32))
        .filter(|_| returnable(results));
    ret_with(il, model.addr, results, |nth, width| match base {
        Some(base) => slot(il, model, base + nth, width),
        None => il.expression(il.undefined()),
    });
}

pub fn stub(il: &LowLevelILMutableFunction, pointer: usize, results: &[ValueKind]) {
    ret_with(il, pointer, results, |_, _| il.expression(il.undefined()));
}

fn ret_with<'a>(
    il: &'a LowLevelILMutableFunction,
    pointer: usize,
    results: &[ValueKind],
    value: impl Fn(u32, usize) -> LowLevelILMutableExpression<'a, ValueExpr>,
) {
    for (nth, kind) in results.iter().enumerate().take(RESULT_REGISTERS as usize) {
        let width = width(*kind, pointer);
        let register = WasmRegister::sized(RegKind::Result(nth as u32), width, pointer);
        il.add_instruction(il.set_reg(
            width,
            LowLevelILRegisterKind::Arch(register),
            value(nth as u32, width),
        ));
    }
    let link = LowLevelILRegisterKind::Arch(WasmRegister::new(RegKind::Lr, pointer));
    il.add_instruction(il.ret(il.reg(pointer, link)));
}

fn go(il: &LowLevelILMutableFunction, model: Model, target: Option<u64>, top: Option<u32>) {
    let Some(to) = target else {
        leave(il, model, top);
        return;
    };

    match il.label_for_address(to) {
        Some(mut label) => il.add_instruction(il.goto(&mut label)),
        None => il.add_instruction(il.jump(il.const_ptr(to))),
    }
}

const RESULT_I32: usize = 4;

/// In slots, positive when the stack shrinks, and `None` where control does not fall through or the
/// effect is unknown; the conformance harness checks this against a real validator, since a wrong
/// amount silently puts every instruction after it at the wrong height
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
        Some(Resolved::Aggregate(arity) | Resolved::Switching(arity)) => {
            return net(i64::from(arity.pops), i64::from(arity.pushes));
        }
        Some(Resolved::Function(_)) => return net(0, 1),
        None => {}
    }

    match semantics(insn) {
        Sem::Nop => net(0, 0),
        Sem::Trap | Sem::Return => None,
        Sem::Const { .. } | Sem::LocalGet(_) | Sem::GlobalGet(_) | Sem::Null => net(0, 1),
        Sem::IsNull | Sem::TableGet => net(1, 1),
        Sem::TableSet => net(2, 0),
        Sem::Unary { .. }
        | Sem::TestZero { .. }
        | Sem::Convert { .. }
        | Sem::Saturate(_)
        | Sem::Load { .. }
        | Sem::LocalTee(_) => net(1, 1),
        Sem::Binary { .. } | Sem::Compare { .. } => net(2, 1),
        Sem::Store { .. } => net(2, 0),
        Sem::LocalSet(_) | Sem::GlobalSet(_) | Sem::Drop | Sem::CondBranch => net(1, 0),
        Sem::RefBranch => net(0, 0),
        Sem::Select { values } => net(2 * i64::from(values) + 1, i64::from(values)),
        Sem::Aggregate { pops, pushes } => net(i64::from(pops), i64::from(pushes)),
        Sem::Opaque | Sem::OtherMemory if !insn.flow().falls_through() => None,
        Sem::Opaque | Sem::OtherMemory => {
            let arity = insn.arity()?;
            net(i64::from(arity.pops), i64::from(arity.pushes))
        }
    }
}

fn opaque(il: &LowLevelILMutableFunction, model: Model, insn: &Instruction, sem: Sem, height: u32) {
    let (Some(arity), Some(operator), true) = (
        insn.arity(),
        insn.operator_id(),
        insn.flow().falls_through(),
    ) else {
        give_up(il, model, insn, sem, None);
        return;
    };

    // Arguments were pushed in order, so the first one sits deepest
    let base = height - arity.pops;
    let outputs: Vec<_> = (0..arity.pushes)
        .map(|nth| model.stack(base + nth, SLOT as usize))
        .collect();
    let offset = memarg(&insn.op).map(|memarg| memarg.offset);
    let inputs: Vec<_> = (0..arity.pops)
        .map(|nth| match offset {
            Some(offset) if nth == 0 => linear_address(il, model, base, offset),
            _ => il.reg(SLOT as usize, model.stack(base + nth, SLOT as usize)),
        })
        .chain(
            immediates(&insn.op)
                .into_iter()
                .map(|value| il.const_int(SLOT as usize, value)),
        )
        .collect();
    il.add_instruction(il.intrinsic(outputs, model.intrinsic(operator), inputs));
}

fn global_address<'a>(
    il: &'a LowLevelILMutableFunction,
    model: Model,
    index: u32,
) -> LowLevelILMutableExpression<'a, ValueExpr> {
    il.const_ptr_sized(model.addr, model.layout.global_address(index))
}

fn prologue(il: &LowLevelILMutableFunction, model: Model) {
    let frame = model.frame;
    for (index, kind) in frame.locals.iter().enumerate() {
        let index = index as u32;
        if index >= LOCAL_REGISTERS {
            crowded_locals();
            break;
        }
        let width = width(*kind, model.addr);
        let value = if index >= frame.params {
            il.const_int(width, 0)
        } else if index < ARGUMENT_REGISTERS {
            il.reg(width, model.argument(index, width))
        } else {
            il.expression(il.undefined())
        };
        il.add_instruction(il.set_reg(width, model.local(index, width), value));
    }
}

fn linear_address<'a>(
    il: &'a LowLevelILMutableFunction,
    model: Model,
    index: u32,
    offset: u64,
) -> LowLevelILMutableExpression<'a, ValueExpr> {
    let base = il.reg(model.addr, model.stack(index, model.addr));
    if offset == 0 {
        return base;
    }
    let delta = il.const_int(model.addr, offset);
    il.expression(il.add(model.addr, base, delta))
}

fn table_slot<'a>(
    il: &'a LowLevelILMutableFunction,
    model: Model,
    index: u32,
) -> LowLevelILMutableExpression<'a, ValueExpr> {
    let index = slot(il, model, index, model.addr);
    let stride = il.const_int(model.addr, model.addr as u64);
    let offset = il.expression(il.mul(model.addr, index, stride));
    let table = il.const_ptr_sized(model.addr, model.layout.table_base);
    il.expression(il.add(model.addr, table, offset))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// The floating point ones; `clz`, `ctz` and `popcnt` are intrinsics, and `eqz` and the extensions
/// have semantics of their own
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
    FloatToInt { signed: bool },
    IntToFloat { signed: bool },
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
    Null,
    IsNull,
    TableGet,
    TableSet,
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
    Saturate(&'static Saturating),
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
    /// A reference test that branches, leaving its operand where it was on the fallthrough
    RefBranch,
    Select {
        values: u32,
    },
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
    /// A memory other than the first, which has no addresses in the view
    OtherMemory,
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
        Bin::Shl => {
            let amount = shift_amount(il, size, right);
            il.expression(il.lsl(size, left, amount))
        }
        Bin::ShrS => {
            let amount = shift_amount(il, size, right);
            il.expression(il.asr(size, left, amount))
        }
        Bin::ShrU => {
            let amount = shift_amount(il, size, right);
            il.expression(il.lsr(size, left, amount))
        }
        Bin::Rotl => {
            let amount = shift_amount(il, size, right);
            il.expression(il.rol(size, left, amount))
        }
        Bin::Rotr => {
            let amount = shift_amount(il, size, right);
            il.expression(il.ror(size, left, amount))
        }
        Bin::FAdd => il.expression(il.fadd(size, left, right)),
        Bin::FSub => il.expression(il.fsub(size, left, right)),
        Bin::FMul => il.expression(il.fmul(size, left, right)),
        Bin::FDiv => il.expression(il.fdiv(size, left, right)),
    }
}

/// wasm shifts modulo the width and the IL does not, so `1 << 32` folded to 0. The mask folds away
/// with any amount already in range
fn shift_amount<'a>(
    il: &'a LowLevelILMutableFunction,
    size: usize,
    amount: LowLevelILMutableExpression<'a, ValueExpr>,
) -> LowLevelILMutableExpression<'a, ValueExpr> {
    let mask = il.const_int(size, (size as u64 * 8).saturating_sub(1));
    il.expression(il.and(size, amount, mask))
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
    // Once: the callers store this, and converting twice rendered as `? 1 : 0 ? 1 : 0`
    il.expression(il.bool_to_int(RESULT_I32, condition))
}

struct OtherWidth<'a>(LowLevelILMutableExpression<'a, ValueExpr>);

impl<'a> LiftableLowLevelIL<'a> for OtherWidth<'a> {
    type Result = ValueExpr;

    fn lift(
        _il: &'a LowLevelILMutableFunction,
        expr: Self,
    ) -> LowLevelILMutableExpression<'a, ValueExpr> {
        expr.0
    }
}

impl<'a> LiftableLowLevelILWithSize<'a> for OtherWidth<'a> {
    fn lift_with_size(
        _il: &'a LowLevelILMutableFunction,
        expr: Self,
        _size: usize,
    ) -> LowLevelILMutableExpression<'a, ValueExpr> {
        expr.0
    }
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
        Conv::FloatToInt { signed: true } => il.expression(il.float_to_int(to, OtherWidth(value))),
        Conv::FloatToInt { signed: false } => {
            let whole = il.expression(il.float_to_int(to * 2, OtherWidth(value)));
            il.expression(il.low_part(to, whole))
        }
        Conv::IntToFloat { signed: true } if from < to => {
            let widened = il.expression(il.sx(to, value));
            il.expression(il.int_to_float(to, widened))
        }
        Conv::IntToFloat { signed: true } => il.expression(il.int_to_float(to, OtherWidth(value))),
        Conv::IntToFloat { signed: false } => {
            let widened = il.expression(il.zx(from * 2, value));
            il.expression(il.int_to_float(to, OtherWidth(widened)))
        }
        Conv::FloatResize => il.expression(il.float_conv(to, OtherWidth(value))),
    }
}

/// Anything not named here is [`Sem::Opaque`], lifted as an intrinsic named after the operator that
/// takes its operands and immediates
fn semantics(insn: &Instruction) -> Sem {
    use Operator as O;

    if memarg(&insn.op).is_some_and(|memarg| memarg.memory != 0) {
        return Sem::OtherMemory;
    }

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

        // On the fallthrough `br_on_non_null` drops its reference and the descriptor forms their
        // descriptor, where the other reference tests leave the stack as it was
        O::BrOnNonNull { .. } | O::BrOnCastDescEq { .. } | O::BrOnCastDescEqFail { .. } => {
            Sem::CondBranch
        }
        O::BrOnNull { .. } | O::BrOnCast { .. } | O::BrOnCastFail { .. } => Sem::RefBranch,

        // The generated tables have no arity for this one, but the instruction carries the count
        O::ArrayNewFixed { array_size, .. } => Sem::Aggregate {
            pops: *array_size,
            pushes: 1,
        },

        O::Select | O::TypedSelect { .. } => Sem::Select { values: 1 },
        // A multi-value `select` chooses between two runs of a shape the instruction spells out
        O::TypedSelectMulti { tys } => Sem::Select {
            values: u32::try_from(tys.len()).unwrap_or(u32::MAX),
        },

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

        O::I32Eq => compare_sem(4, Cmp::Eq),
        O::I32Ne => compare_sem(4, Cmp::Ne),
        O::I32LtS => compare_sem(4, Cmp::LtS),
        O::I32LtU => compare_sem(4, Cmp::LtU),
        O::I32GtS => compare_sem(4, Cmp::GtS),
        O::I32GtU => compare_sem(4, Cmp::GtU),
        O::I32LeS => compare_sem(4, Cmp::LeS),
        O::I32LeU => compare_sem(4, Cmp::LeU),
        O::I32GeS => compare_sem(4, Cmp::GeS),
        O::I32GeU => compare_sem(4, Cmp::GeU),
        O::I64Eq => compare_sem(8, Cmp::Eq),
        O::I64Ne => compare_sem(8, Cmp::Ne),
        O::I64LtS => compare_sem(8, Cmp::LtS),
        O::I64LtU => compare_sem(8, Cmp::LtU),
        O::I64GtS => compare_sem(8, Cmp::GtS),
        O::I64GtU => compare_sem(8, Cmp::GtU),
        O::I64LeS => compare_sem(8, Cmp::LeS),
        O::I64LeU => compare_sem(8, Cmp::LeU),
        O::I64GeS => compare_sem(8, Cmp::GeS),
        O::I64GeU => compare_sem(8, Cmp::GeU),
        O::F32Eq => compare_sem(4, Cmp::FEq),
        O::F32Ne => compare_sem(4, Cmp::FNe),
        O::F32Lt => compare_sem(4, Cmp::FLt),
        O::F32Gt => compare_sem(4, Cmp::FGt),
        O::F32Le => compare_sem(4, Cmp::FLe),
        O::F32Ge => compare_sem(4, Cmp::FGe),
        O::F64Eq => compare_sem(8, Cmp::FEq),
        O::F64Ne => compare_sem(8, Cmp::FNe),
        O::F64Lt => compare_sem(8, Cmp::FLt),
        O::F64Gt => compare_sem(8, Cmp::FGt),
        O::F64Le => compare_sem(8, Cmp::FLe),
        O::F64Ge => compare_sem(8, Cmp::FGe),

        O::I32Add => binary_sem(4, Bin::Add),
        O::I32Sub => binary_sem(4, Bin::Sub),
        O::I32Mul => binary_sem(4, Bin::Mul),
        O::I32DivS => binary_sem(4, Bin::DivS),
        O::I32DivU => binary_sem(4, Bin::DivU),
        O::I32RemS => binary_sem(4, Bin::RemS),
        O::I32RemU => binary_sem(4, Bin::RemU),
        O::I32And => binary_sem(4, Bin::And),
        O::I32Or => binary_sem(4, Bin::Or),
        O::I32Xor => binary_sem(4, Bin::Xor),
        O::I32Shl => binary_sem(4, Bin::Shl),
        O::I32ShrS => binary_sem(4, Bin::ShrS),
        O::I32ShrU => binary_sem(4, Bin::ShrU),
        O::I32Rotl => binary_sem(4, Bin::Rotl),
        O::I32Rotr => binary_sem(4, Bin::Rotr),
        O::I64Add => binary_sem(8, Bin::Add),
        O::I64Sub => binary_sem(8, Bin::Sub),
        O::I64Mul => binary_sem(8, Bin::Mul),
        O::I64DivS => binary_sem(8, Bin::DivS),
        O::I64DivU => binary_sem(8, Bin::DivU),
        O::I64RemS => binary_sem(8, Bin::RemS),
        O::I64RemU => binary_sem(8, Bin::RemU),
        O::I64And => binary_sem(8, Bin::And),
        O::I64Or => binary_sem(8, Bin::Or),
        O::I64Xor => binary_sem(8, Bin::Xor),
        O::I64Shl => binary_sem(8, Bin::Shl),
        O::I64ShrS => binary_sem(8, Bin::ShrS),
        O::I64ShrU => binary_sem(8, Bin::ShrU),
        O::I64Rotl => binary_sem(8, Bin::Rotl),
        O::I64Rotr => binary_sem(8, Bin::Rotr),
        O::F32Add => binary_sem(4, Bin::FAdd),
        O::F32Sub => binary_sem(4, Bin::FSub),
        O::F32Mul => binary_sem(4, Bin::FMul),
        O::F32Div => binary_sem(4, Bin::FDiv),
        O::F64Add => binary_sem(8, Bin::FAdd),
        O::F64Sub => binary_sem(8, Bin::FSub),
        O::F64Mul => binary_sem(8, Bin::FMul),
        O::F64Div => binary_sem(8, Bin::FDiv),

        O::F32Neg => unary_sem(4, Un::Neg),
        O::F32Abs => unary_sem(4, Un::Abs),
        O::F32Sqrt => unary_sem(4, Un::Sqrt),
        O::F32Ceil => unary_sem(4, Un::Ceil),
        O::F32Floor => unary_sem(4, Un::Floor),
        O::F32Trunc => unary_sem(4, Un::Trunc),
        O::F32Nearest => unary_sem(4, Un::Nearest),
        O::F64Neg => unary_sem(8, Un::Neg),
        O::F64Abs => unary_sem(8, Un::Abs),
        O::F64Sqrt => unary_sem(8, Un::Sqrt),
        O::F64Ceil => unary_sem(8, Un::Ceil),
        O::F64Floor => unary_sem(8, Un::Floor),
        O::F64Trunc => unary_sem(8, Un::Trunc),
        O::F64Nearest => unary_sem(8, Un::Nearest),

        O::I32WrapI64 => convert_sem(8, 4, Conv::Wrap),
        O::I64ExtendI32S => convert_sem(4, 8, Conv::SignExtend),
        O::I64ExtendI32U => convert_sem(4, 8, Conv::ZeroExtend),
        O::I32Extend8S => convert_sem(1, 4, Conv::SignExtend),
        O::I32Extend16S => convert_sem(2, 4, Conv::SignExtend),
        O::I64Extend8S => convert_sem(1, 8, Conv::SignExtend),
        O::I64Extend16S => convert_sem(2, 8, Conv::SignExtend),
        O::I64Extend32S => convert_sem(4, 8, Conv::SignExtend),

        O::I32TruncF32S => convert_sem(4, 4, float_to_int(true)),
        O::I32TruncF32U => convert_sem(4, 4, float_to_int(false)),
        O::I32TruncF64S => convert_sem(8, 4, float_to_int(true)),
        O::I32TruncF64U => convert_sem(8, 4, float_to_int(false)),
        O::I64TruncF32S => convert_sem(4, 8, float_to_int(true)),
        O::I64TruncF32U => convert_sem(4, 8, float_to_int(false)),
        O::I64TruncF64S => convert_sem(8, 8, float_to_int(true)),
        O::I64TruncF64U => convert_sem(8, 8, float_to_int(false)),
        O::I32TruncSatF32S
        | O::I32TruncSatF32U
        | O::I32TruncSatF64S
        | O::I32TruncSatF64U
        | O::I64TruncSatF32S
        | O::I64TruncSatF32U
        | O::I64TruncSatF64S
        | O::I64TruncSatF64U => insn
            .operator_id()
            .and_then(saturating)
            .map_or(Sem::Opaque, Sem::Saturate),
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

        O::I32AtomicLoad { memarg } => load(4, 4, Extend::None, memarg.offset),
        O::I64AtomicLoad { memarg } => load(8, 8, Extend::None, memarg.offset),
        O::I32AtomicLoad8U { memarg } => load(1, 4, Extend::Zero, memarg.offset),
        O::I32AtomicLoad16U { memarg } => load(2, 4, Extend::Zero, memarg.offset),
        O::I64AtomicLoad8U { memarg } => load(1, 8, Extend::Zero, memarg.offset),
        O::I64AtomicLoad16U { memarg } => load(2, 8, Extend::Zero, memarg.offset),
        O::I64AtomicLoad32U { memarg } => load(4, 8, Extend::Zero, memarg.offset),
        O::I32AtomicStore { memarg } => store(4, 4, memarg.offset),
        O::I64AtomicStore { memarg } => store(8, 8, memarg.offset),
        O::I32AtomicStore8 { memarg } => store(1, 4, memarg.offset),
        O::I32AtomicStore16 { memarg } => store(2, 4, memarg.offset),
        O::I64AtomicStore8 { memarg } => store(1, 8, memarg.offset),
        O::I64AtomicStore16 { memarg } => store(2, 8, memarg.offset),
        O::I64AtomicStore32 { memarg } => store(4, 8, memarg.offset),

        O::RefNull { .. } => Sem::Null,
        O::RefIsNull => Sem::IsNull,
        O::TableGet { table: 0 } => Sem::TableGet,
        O::TableSet { table: 0 } => Sem::TableSet,

        _ => Sem::Opaque,
    }
}

fn unary_sem(size: usize, kind: Un) -> Sem {
    Sem::Unary { size, kind }
}

fn binary_sem(size: usize, kind: Bin) -> Sem {
    Sem::Binary { size, kind }
}

fn compare_sem(size: usize, kind: Cmp) -> Sem {
    Sem::Compare { size, kind }
}

fn float_to_int(signed: bool) -> Conv {
    Conv::FloatToInt { signed }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Saturating {
    pub id: u32,
    pub from: usize,
    pub to: usize,
    pub signed: bool,
}

const SATURATING: [Saturating; 8] = [
    saturating_from("visit_i32_trunc_sat_f32_s", 4, 4, true),
    saturating_from("visit_i32_trunc_sat_f32_u", 4, 4, false),
    saturating_from("visit_i32_trunc_sat_f64_s", 8, 4, true),
    saturating_from("visit_i32_trunc_sat_f64_u", 8, 4, false),
    saturating_from("visit_i64_trunc_sat_f32_s", 4, 8, true),
    saturating_from("visit_i64_trunc_sat_f32_u", 4, 8, false),
    saturating_from("visit_i64_trunc_sat_f64_s", 8, 8, true),
    saturating_from("visit_i64_trunc_sat_f64_u", 8, 8, false),
];

const fn saturating_from(visitor: &str, from: usize, to: usize, signed: bool) -> Saturating {
    Saturating {
        id: crate::insn::stable_id(visitor),
        from,
        to,
        signed,
    }
}

pub fn saturating(id: u32) -> Option<&'static Saturating> {
    SATURATING.iter().find(|conversion| conversion.id == id)
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
    use crate::insn::{Arity, decode};

    fn sem(data: &[u8]) -> Sem {
        semantics(&decode(data).expect("decodes"))
    }

    #[test]
    fn only_memory_zero_is_read_as_memory() {
        assert!(matches!(
            sem(&[0x28, 0x02, 0x00]),
            Sem::Load { access: 4, .. }
        ));
        assert!(
            matches!(sem(&[0x28, 0x42, 0x01, 0x00]), Sem::OtherMemory),
            "memory 1 is not mapped, so reading it as memory 0 would name the wrong bytes"
        );
        assert!(matches!(
            sem(&[0xfe, 0x10, 0x02, 0x00]),
            Sem::Load { access: 4, .. }
        ));
        assert!(matches!(
            sem(&[0xfe, 0x17, 0x02, 0x00]),
            Sem::Store { access: 4, .. }
        ));
    }

    #[test]
    fn references_and_the_first_table_are_values_and_slots() {
        assert!(matches!(sem(&[0xd0, 0x70]), Sem::Null));
        assert!(matches!(sem(&[0xd1]), Sem::IsNull));
        assert!(matches!(sem(&[0x25, 0x00]), Sem::TableGet));
        assert!(matches!(sem(&[0x26, 0x00]), Sem::TableSet));
        assert!(
            matches!(sem(&[0x25, 0x01]), Sem::Opaque),
            "only table 0 has a region"
        );
        let function = Resolved::Function(0x1234);
        assert_eq!(
            stack_effect(&decode(&[0xd2, 0x00]).expect("decodes"), Some(&function)),
            Some(-1)
        );
    }

    #[test]
    fn a_value_moves_at_the_width_of_its_type() {
        assert_eq!(width(ValueKind::I32, 4), 4);
        assert_eq!(width(ValueKind::F64, 4), 8);
        assert_eq!(width(ValueKind::Ref, 4), 4);
        assert_eq!(
            width(ValueKind::Ref, 8),
            8,
            "a reference is a pointer in memory64"
        );
        assert_eq!(width(ValueKind::V128, 4), SLOT as usize);
    }

    #[test]
    fn a_saturating_conversion_is_its_own_operator_rather_than_a_cast() {
        let widths = [
            (4, 4, true),
            (4, 4, false),
            (8, 4, true),
            (8, 4, false),
            (4, 8, true),
            (4, 8, false),
            (8, 8, true),
            (8, 8, false),
        ];
        for (sub, (from, to, signed)) in (0u8..).zip(widths) {
            let Sem::Saturate(conversion) = sem(&[0xfc, sub]) else {
                panic!("0xfc {sub:#x} is not lifted as a saturating conversion");
            };
            assert_eq!(
                (conversion.from, conversion.to, conversion.signed),
                (from, to, signed)
            );
            let id = decode(&[0xfc, sub]).and_then(|insn| insn.operator_id());
            assert_eq!(
                Some(conversion.id),
                id,
                "the intrinsic is the operator's own"
            );
        }
        assert_eq!(
            sem(&[0xa8]),
            Sem::Convert {
                from: 4,
                to: 4,
                kind: Conv::FloatToInt { signed: true }
            },
            "i32.trunc_f32_s traps instead, so no finished run sees a clamped value"
        );
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
    fn unsigned_conversions_stay_unsigned() {
        let unsigned = Conv::IntToFloat { signed: false };
        assert_eq!(
            sem(&[0xb3]),
            convert_sem(4, 4, unsigned),
            "f32.convert_i32_u"
        );
        assert_eq!(
            sem(&[0xb5]),
            convert_sem(8, 4, unsigned),
            "f32.convert_i64_u"
        );
        assert_eq!(
            sem(&[0xb8]),
            convert_sem(4, 8, unsigned),
            "f64.convert_i32_u"
        );
        assert_eq!(
            sem(&[0xba]),
            convert_sem(8, 8, unsigned),
            "f64.convert_i64_u"
        );
        assert_eq!(
            sem(&[0xa9]),
            convert_sem(4, 4, float_to_int(false)),
            "i32.trunc_f32_u"
        );
        assert_eq!(
            sem(&[0xab]),
            convert_sem(8, 4, float_to_int(false)),
            "i32.trunc_f64_u"
        );
        assert_eq!(
            sem(&[0xaa]),
            convert_sem(8, 4, float_to_int(true)),
            "i32.trunc_f64_s"
        );
        assert_eq!(
            sem(&[0xb1]),
            convert_sem(8, 8, float_to_int(false)),
            "i64.trunc_f64_u"
        );
    }

    #[test]
    fn select_chooses_between_runs_of_values() {
        assert_eq!(sem(&[0x1b]), Sem::Select { values: 1 });
        assert_eq!(sem(&[0x1c, 0x01, 0x7f]), Sem::Select { values: 1 });
        let select = decode(&[0x1b]).expect("select decodes");
        assert_eq!(stack_effect(&select, None), Some(2));
        assert_eq!(demand_alone(&select), (3, 1));
    }

    fn demand_alone(insn: &Instruction) -> (u32, u32) {
        demand(insn, semantics(insn), None)
    }

    fn fits_alone(insn: &Instruction, height: u32) -> bool {
        fits(insn, semantics(insn), None, height)
    }

    #[test]
    fn a_branch_that_tests_the_top_needs_it_there() {
        let br_if = decode(&[0x0d, 0x00]).expect("br_if decodes");
        assert_eq!(demand_alone(&br_if).0, 1);
        assert!(!fits_alone(&br_if, 0));
        assert!(fits_alone(&br_if, 1));
        let br_table = decode(&[0x0e, 0x00, 0x00]).expect("br_table decodes");
        assert_eq!(demand_alone(&br_table).0, 1);
    }

    #[test]
    fn an_operand_stack_deeper_than_the_register_file_does_not_fit() {
        let push = decode(&[0x41, 0x00]).expect("i32.const decodes");
        assert!(fits_alone(&push, STACK_REGISTERS - 1));
        assert!(!fits_alone(&push, STACK_REGISTERS));
        let add = decode(&[0x6a]).expect("i32.add decodes");
        assert!(fits_alone(&add, STACK_REGISTERS));
        assert!(!fits_alone(&add, 1), "a binary operator needs two operands");
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
    fn conditional_branches_still_consume_their_condition() {
        assert_eq!(sem(&[0x04, 0x40]), Sem::CondBranch);
        assert_eq!(sem(&[0x0d, 0x00]), Sem::CondBranch);
    }

    #[test]
    fn a_taken_branch_leaves_only_what_its_label_receives() {
        let (from_ref_type, to_ref_type) =
            (wasmparser::RefType::ANYREF, wasmparser::RefType::EQREF);
        let relative_depth = 0;
        let cases = [
            (Operator::BrIf { relative_depth }, 1, "the condition"),
            (
                Operator::BrOnNull { relative_depth },
                1,
                "the null reference",
            ),
            (
                Operator::BrOnNonNull { relative_depth },
                0,
                "the reference is carried",
            ),
            (
                Operator::BrOnCast {
                    relative_depth,
                    from_ref_type,
                    to_ref_type,
                },
                0,
                "the cast reference is carried",
            ),
            (
                Operator::BrOnCastFail {
                    relative_depth,
                    from_ref_type,
                    to_ref_type,
                },
                0,
                "the reference is carried",
            ),
            (
                Operator::BrOnCastDescEq {
                    relative_depth,
                    from_ref_type,
                    to_ref_type,
                },
                1,
                "the descriptor",
            ),
            (
                Operator::BrOnCastDescEqFail {
                    relative_depth,
                    from_ref_type,
                    to_ref_type,
                },
                1,
                "the descriptor",
            ),
        ];
        for (op, pops, what) in cases {
            assert_eq!(taken_pops(&op), pops, "{op:?}: {what}");
            assert_eq!(crate::insn::conditional_label(&op), Some(relative_depth));
        }
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
            results: vec![ValueKind::I32],
            returns_argument: false,
        });
        assert_eq!(stack_effect(&call, Some(&takes_two)), Some(1));

        let aggregate = Resolved::Aggregate(Arity { pops: 3, pushes: 1 });
        assert_eq!(stack_effect(&call, Some(&aggregate)), Some(2));

        // A tail call replaces the frame, so nothing is left to measure
        let tail = decode(&[0x12, 0x00]).expect("return_call decodes");
        assert_eq!(stack_effect(&tail, Some(&takes_two)), None);
    }

    #[test]
    fn switching_stacks_is_measured_by_what_the_module_resolves() {
        let suspend = decode(&[0xe2, 0x00]).expect("suspend decodes");
        assert_eq!(suspend.mnemonic(), "suspend");
        assert_eq!(stack_effect(&suspend, None), None);

        let switching = Resolved::Switching(Arity { pops: 2, pushes: 1 });
        assert_eq!(stack_effect(&suspend, Some(&switching)), Some(1));
        let sem = semantics(&suspend);
        assert_eq!(demand(&suspend, sem, Some(&switching)), (2, 1));
        assert!(fits(&suspend, sem, Some(&switching), 2));
        assert!(!fits(&suspend, sem, Some(&switching), 1));
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
        for encoding in [&[0xfd, 0x6e][..], &[0x10, 0x00], &[0xfc, 0x0a, 0x00, 0x00]] {
            assert_eq!(sem(encoding), Sem::Opaque, "{encoding:02x?}");
        }
    }
}
