//! The Binary Ninja `Architecture` implementation
//!
//! Glue only: decoding lives in [`crate::insn`], control flow recovery in [`crate::cfg`], and
//! semantics in [`crate::lift`]

use std::borrow::Cow;

use binaryninja::architecture::{
    Architecture, ArchitectureExt, BasicBlockAnalysisContext, BranchKind, BranchType,
    CoreArchitecture, CustomArchitectureHandle, ImplicitRegisterExtend, InstructionInfo, Intrinsic,
    IntrinsicId, Register, RegisterId, RegisterInfo, UnusedFlag, UnusedRegisterStack,
};
use binaryninja::basic_block::PendingBasicBlockEdge;
use binaryninja::binary_view::BinaryView;
use binaryninja::calling_convention::{
    CallingConvention, CoreCallingConvention, register_calling_convention,
};
use binaryninja::confidence::Conf;
use binaryninja::disassembly::{InstructionTextToken, InstructionTextTokenKind};
use binaryninja::function::Function;
use binaryninja::low_level_il::LowLevelILMutableFunction;
use binaryninja::rc::Ref;
use binaryninja::types::{NameAndType, Type};
use binaryninja::{Endianness, architecture};

use wasmparser::Operator;

use crate::ViewId;
use crate::asm;
use crate::cfg::{self, Edge, Terminator};
use crate::insn::{self, Flow, Operand};
use crate::lift::{self, Model, SLOT};
use crate::module;

pub const NAME: &str = "wasm";
pub const NAME64: &str = "wasm64";

/// Whether the core asks about an address at all is not visible from anywhere else: a function
/// that never disassembles looks the same whether the callback was never called, answered `None`,
/// or answered correctly and was ignored
mod trace {
    use std::sync::atomic::{AtomicUsize, Ordering};

    const LIMIT: usize = 12;
    static INFO: AtomicUsize = AtomicUsize::new(0);
    static TEXT: AtomicUsize = AtomicUsize::new(0);
    static LLIL: AtomicUsize = AtomicUsize::new(0);

    fn take(counter: &AtomicUsize) -> bool {
        counter.fetch_add(1, Ordering::Relaxed) < LIMIT
    }

    pub fn info(addr: u64, bytes: usize, decoded: Option<&str>) {
        if take(&INFO) {
            tracing::info!("wasm info @ {addr:#x}: {bytes} bytes, {decoded:?}");
        }
    }

    pub fn text(addr: u64, decoded: Option<&str>) {
        if take(&TEXT) {
            tracing::info!("wasm text @ {addr:#x}: {decoded:?}");
        }
    }

    pub fn llil(addr: u64, decoded: Option<&str>) {
        if take(&LLIL) {
            tracing::info!("wasm llil @ {addr:#x}: {decoded:?}");
        }
    }
}

/// Registers Binary Ninja needs that WebAssembly does not have
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegKind {
    /// Top of the operand stack
    Sp,
    /// Base of the current frame's locals
    Fp,
    /// Return address, since wasm keeps its call stack out of reach of the program
    Lr,
    /// A calling convention cannot describe results left on the operand stack, and without a
    /// register for it every call reads as producing nothing
    Rv,
}

impl RegKind {
    pub const ALL: [Self; 4] = [Self::Sp, Self::Fp, Self::Lr, Self::Rv];

    pub fn id(self) -> RegisterId {
        RegisterId(match self {
            Self::Sp => 0,
            Self::Fp => 1,
            Self::Lr => 2,
            Self::Rv => 3,
        })
    }

    fn from_id(id: RegisterId) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.id() == id)
    }
}

/// The width comes along because [`RegisterInfo::size`] is asked without an architecture to ask
/// about, and a register holding an address is as wide as an address is
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WasmRegister {
    pub kind: RegKind,
    /// Width of an address in the architecture this register belongs to
    pub pointer: usize,
}

impl WasmRegister {
    pub fn new(kind: RegKind, pointer: usize) -> Self {
        Self { kind, pointer }
    }
}

impl Register for WasmRegister {
    type InfoType = Self;

    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(match self.kind {
            RegKind::Sp => "sp",
            RegKind::Fp => "fp",
            RegKind::Lr => "lr",
            RegKind::Rv => "rv",
        })
    }

    fn info(&self) -> Self {
        *self
    }

    fn id(&self) -> RegisterId {
        self.kind.id()
    }
}

impl RegisterInfo for WasmRegister {
    type RegType = Self;

    fn parent(&self) -> Option<Self> {
        None
    }

    fn size(&self) -> usize {
        // A result is any wasm value, so it takes a whole slot rather than an address
        match self.kind {
            RegKind::Rv => lift::SLOT as usize,
            _ => self.pointer,
        }
    }

    fn offset(&self) -> usize {
        0
    }

    fn implicit_extend(&self) -> ImplicitRegisterExtend {
        ImplicitRegisterExtend::NoExtend
    }
}

/// Every operator gets one whether or not [`crate::lift`] models it, so an id read back from a
/// saved database always resolves; only the unmodelled ones are ever emitted
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WasmIntrinsic(pub u32);

impl Intrinsic for WasmIntrinsic {
    fn name(&self) -> Cow<'_, str> {
        Cow::Owned(insn::operator_name(self.0).unwrap_or_else(|| format!("op{}", self.0)))
    }

    fn id(&self) -> IntrinsicId {
        IntrinsicId(self.0)
    }

    /// A value's type belongs to the operator rather than the operand, and a slot is the same
    /// width either way
    fn inputs(&self) -> Vec<NameAndType> {
        let arity = insn::operator_arity(self.0).unwrap_or_default();
        (0..arity.pops)
            .map(|index| NameAndType::new(format!("arg{index}"), slot_type().into()))
            .collect()
    }

    fn outputs(&self) -> Vec<Conf<Ref<Type>>> {
        let arity = insn::operator_arity(self.0).unwrap_or_default();
        (0..arity.pushes).map(|_| slot_type().into()).collect()
    }
}

fn slot_type() -> Ref<Type> {
    Type::int(SLOT as usize, false)
}

/// Wasm passes arguments on the operand stack, so there is nothing here to name registers with,
/// and saying so beats leaving the core to guess from a register file that means nothing
struct WasmCallingConvention(CoreCallingConvention);

impl AsRef<CoreCallingConvention> for WasmCallingConvention {
    fn as_ref(&self) -> &CoreCallingConvention {
        &self.0
    }
}

impl CallingConvention for WasmCallingConvention {
    fn caller_saved_registers(&self) -> Vec<RegisterId> {
        vec![RegKind::Rv.id()]
    }

    fn callee_saved_registers(&self) -> Vec<RegisterId> {
        vec![RegKind::Fp.id(), RegKind::Lr.id()]
    }

    fn int_arg_registers(&self) -> Vec<RegisterId> {
        Vec::new()
    }

    fn float_arg_registers(&self) -> Vec<RegisterId> {
        Vec::new()
    }

    fn arg_registers_shared_index(&self) -> bool {
        false
    }

    fn reserved_stack_space_for_arg_registers(&self) -> bool {
        false
    }

    /// The callee's locals live in a frame of their own, so the adjustment is emitted at the call
    /// site; saying the callee did it has the core account for the same slots twice
    fn stack_adjusted_on_return(&self) -> bool {
        false
    }

    /// There are no argument registers to guess about, and letting the core guess anyway changed
    /// nothing about how many arguments a call site resolves
    fn is_eligible_for_heuristics(&self) -> bool {
        false
    }

    fn return_int_reg(&self) -> Option<RegisterId> {
        Some(RegKind::Rv.id())
    }

    fn return_hi_int_reg(&self) -> Option<RegisterId> {
        None
    }

    /// The same register: a result is a slot whatever its type, and [`lift::leave`] writes it there
    fn return_float_reg(&self) -> Option<RegisterId> {
        Some(RegKind::Rv.id())
    }

    fn global_pointer_reg(&self) -> Option<RegisterId> {
        None
    }

    fn implicitly_defined_registers(&self) -> Vec<RegisterId> {
        Vec::new()
    }

    fn are_argument_registers_used_for_var_args(&self) -> bool {
        false
    }
}

pub struct WasmArchitecture {
    handle: CustomArchitectureHandle<Self>,
    core: CoreArchitecture,
    /// Four for `wasm`, eight for `wasm64`
    pointer: usize,
}

impl WasmArchitecture {
    fn new(pointer: usize) -> impl Fn(CustomArchitectureHandle<Self>, CoreArchitecture) -> Self {
        move |handle, core| Self {
            handle,
            core,
            pointer,
        }
    }

    fn model(&self) -> Model {
        Model {
            addr: self.pointer,
            ..Model::default()
        }
    }
}

impl AsRef<CoreArchitecture> for WasmArchitecture {
    fn as_ref(&self) -> &CoreArchitecture {
        &self.core
    }
}

impl Architecture for WasmArchitecture {
    type Handle = CustomArchitectureHandle<Self>;
    type RegisterInfo = WasmRegister;
    type Register = WasmRegister;
    type RegisterStackInfo = UnusedRegisterStack<WasmRegister>;
    type RegisterStack = UnusedRegisterStack<WasmRegister>;
    type Flag = UnusedFlag;
    type FlagWrite = UnusedFlag;
    type FlagClass = UnusedFlag;
    type FlagGroup = UnusedFlag;
    type Intrinsic = WasmIntrinsic;

    fn endianness(&self) -> Endianness {
        Endianness::LittleEndian
    }

    fn address_size(&self) -> usize {
        self.pointer
    }

    fn default_integer_size(&self) -> usize {
        4
    }

    /// Every opcode is byte-aligned, since wasm has no padding in the code section
    fn instruction_alignment(&self) -> usize {
        1
    }

    fn max_instr_len(&self) -> usize {
        insn::MAX_INSTR_LEN
    }

    /// Long `br_table` and `v128.const` encodings would otherwise crowd out the disassembly
    fn opcode_display_len(&self) -> usize {
        8
    }

    fn instruction_info(&self, data: &[u8], addr: u64) -> Option<InstructionInfo> {
        if module::import_stub(addr) {
            let mut info = InstructionInfo::new(module::IMPORT_STRIDE as usize, 0);
            info.add_branch(BranchKind::FunctionReturn);
            return Some(info);
        }

        let decoded = insn::decode(data);
        trace::info(
            addr,
            data.len(),
            decoded.as_ref().map(|i| i.mnemonic()).as_deref(),
        );
        let insn = decoded?;
        let mut info = InstructionInfo::new(insn.len, 0);

        // The core hands this callback no function, so the file has to be inferred and is only
        // trusted when it is unambiguous
        let call = module::lookup_anywhere(addr).and_then(|module| module.call(&insn.op));
        if let Some(target) = call.and_then(|call| call.target) {
            info.add_branch(BranchKind::Call(target));
        }

        match cfg::lookup_anywhere(addr) {
            Some(recovered) => describe(&recovered.terminator, &mut info),
            // With no block stack to resolve labels against, a branch goes somewhere unknown
            // rather than into the instruction after it, which is the one thing that cannot run
            None => match insn.flow() {
                Flow::Trap | Flow::Return | Flow::TailCall => {
                    info.add_branch(BranchKind::FunctionReturn)
                }
                Flow::Branch => info.add_branch(BranchKind::Unresolved),
                Flow::IndirectBranch => info.add_branch(BranchKind::Indirect),
                _ => {}
            },
        }

        Some(info)
    }

    /// The core's own walk asks [`Architecture::instruction_info`] where each instruction goes,
    /// which one instruction cannot say; recovery here has the whole body, so the edges are exact
    fn analyze_basic_blocks(
        &self,
        function: &mut Function,
        context: &mut BasicBlockAnalysisContext,
    ) {
        let view = function.view();
        let start = function.start();
        let id = view_id(&view);

        // An import stub is one synthetic instruction that returns, and recovering it from bytes
        // would read the region as zeros, which decode as `unreachable`
        let arch = *self.as_ref();
        if module::import_stub(start) {
            if let Some(native) = context.create_basic_block(arch, start) {
                native.set_end(start + module::IMPORT_STRIDE);
                native.add_pending_outgoing_edge(&pending_edge(Edge::FunctionReturn, arch));
                context.add_basic_block(native);
            }
            context.finalize();
            return;
        }

        // A function swept out of the middle of a body has to be recovered from the body it sits
        // in, or none of its labels resolve
        let module = module::lookup(id, start);
        let body = module
            .as_ref()
            .and_then(|module| module.body_covering(start).map(|(_, i)| (i.entry, i.end)));

        // A code section lists every body exactly, so an address in the image that is not one of
        // their entries is not a function, whatever the core swept up
        //
        // One block that returns rather than no blocks at all: a function whose IL has no block is
        // reported once per attempt as `Cannot find the source block of mlil`, 1246 warnings across
        // 549 functions on one real module; [`crate::view::drop_invented_functions`] removes the
        // function itself once analysis is done
        if crate::view::invented_at(id, start) {
            if let Some(native) = context.create_basic_block(arch, start) {
                native.set_end(start + 1);
                native.add_pending_outgoing_edge(&pending_edge(Edge::FunctionReturn, arch));
                context.add_basic_block(native);
            }
            context.finalize();
            return;
        }

        let (from, len) = match body {
            Some((entry, end)) => (entry, (end - entry).min(cfg::MAX_BODY_LEN as u64) as usize),
            None => (
                start,
                file_end(&view)
                    .saturating_sub(start)
                    .min(context.max_function_size)
                    .min(cfg::MAX_BODY_LEN as u64) as usize,
            ),
        };

        let code = view.read_vec(from, len);
        let flow = cfg::recover(&code, from, module.as_deref()).restricted_to(start);
        cfg::install(id, &flow);

        let blocks = flow.blocks();
        if blocks.is_empty() {
            tracing::warn!(
                "wasm: nothing decoded at {start:#x}, {} of {len} bytes read from {from:#x}",
                code.len()
            );
            if let Some(native) = context.create_basic_block(arch, start) {
                native.set_end(start + 1);
                native.set_has_invalid_instructions(true);
                context.add_basic_block(native);
            }
            context.finalize();
            return;
        }

        let instruction_data = context.lifter_instruction_data();
        for block in blocks {
            let Some(native) = context.create_basic_block(arch, block.start) else {
                continue;
            };
            native.set_end(block.end);

            let first = flow
                .instructions
                .binary_search_by_key(&block.start, |(at, _)| *at)
                .unwrap_or_else(|index| index);
            for (at, size) in &flow.instructions[first..] {
                if *at >= block.end {
                    break;
                }
                let offset = (at - from) as usize;
                if let Some(bytes) = code.get(offset..offset + size)
                    && let Some(instruction_data) = &instruction_data
                {
                    instruction_data.append(&native, bytes);
                }
            }

            if flow.truncated {
                native.set_has_invalid_instructions(true);
            }
            for edge in &block.edges {
                native.add_pending_outgoing_edge(&pending_edge(*edge, arch));
            }
            context.add_basic_block(native);
        }

        context.finalize();
    }

    fn instruction_text(
        &self,
        data: &[u8],
        _addr: u64,
    ) -> Option<(usize, Vec<InstructionTextToken>)> {
        if module::import_stub(_addr) {
            return Some((
                module::IMPORT_STRIDE as usize,
                vec![InstructionTextToken::new(
                    "<import>",
                    InstructionTextTokenKind::Annotation,
                )],
            ));
        }

        let decoded = insn::decode(data);
        trace::text(_addr, decoded.as_ref().map(|i| i.mnemonic()).as_deref());
        let insn = decoded?;
        let mut tokens = vec![InstructionTextToken::new(
            insn.mnemonic(),
            InstructionTextTokenKind::Instruction,
        )];

        let mut rendered = 0;
        for operand in insn.operands() {
            for (text, kind) in operand_tokens(&operand) {
                tokens.push(if rendered == 0 {
                    InstructionTextToken::new(" ", InstructionTextTokenKind::Text)
                } else {
                    InstructionTextToken::new(", ", InstructionTextTokenKind::OperandSeparator)
                });
                tokens.push(InstructionTextToken::new(text, kind));
                rendered += 1;
            }
        }

        Some((insn.len, tokens))
    }

    fn instruction_llil(
        &self,
        data: &[u8],
        addr: u64,
        il: &LowLevelILMutableFunction,
    ) -> Option<(usize, bool)> {
        if module::import_stub(addr) {
            // The body is somebody else's, and returning keeps the caller's control flow intact
            let address = self.model().link(il);
            il.add_instruction(il.ret(address));
            return Some((module::IMPORT_STRIDE as usize, true));
        }

        let decoded = insn::decode(data);
        trace::llil(addr, decoded.as_ref().map(|i| i.mnemonic()).as_deref());
        let insn = decoded?;

        // Unlike the other callbacks this one can say which file it is lifting, since the IL
        // belongs to a function and the function to a view
        let (recovered, module) = match il.function() {
            Some(function) => {
                let view = view_id(&function.view());
                (cfg::lookup(view, addr), module::lookup(view, addr))
            }
            None => (cfg::lookup_anywhere(addr), module::lookup_anywhere(addr)),
        };

        let resolved = module.as_ref().and_then(|module| module.resolve(&insn.op));
        let width = module
            .as_ref()
            .and_then(|module| moved_width(module, addr, &insn.op));
        // The width comes from the architecture the view chose, and where the globals live from
        // the file, since each is laid out in a place of its own
        let model = Model {
            global_base: module
                .as_ref()
                .map(|module| module.layout.global_base)
                .or_else(|| module::layout_covering(addr).map(|layout| layout.global_base))
                .unwrap_or_default(),
            table_base: module
                .as_ref()
                .map(|module| module.layout.table_base)
                .or_else(|| module::layout_covering(addr).map(|layout| layout.table_base))
                .unwrap_or_default(),
            frame: module
                .as_ref()
                .and_then(|module| frame_at(module, addr))
                .unwrap_or_default(),
            ..self.model()
        };
        lift::lift(
            il,
            model,
            &insn,
            recovered.as_ref(),
            resolved.as_ref(),
            width,
        );
        // The flag reports whether lifting succeeded rather than whether control continues
        Some((insn.len, true))
    }

    fn registers_all(&self) -> Vec<Self::Register> {
        RegKind::ALL
            .map(|kind| WasmRegister::new(kind, self.pointer))
            .to_vec()
    }

    fn registers_full_width(&self) -> Vec<Self::Register> {
        self.registers_all()
    }

    fn register_from_id(&self, id: RegisterId) -> Option<Self::Register> {
        RegKind::from_id(id).map(|kind| WasmRegister::new(kind, self.pointer))
    }

    fn stack_pointer_reg(&self) -> Option<Self::Register> {
        Some(WasmRegister::new(RegKind::Sp, self.pointer))
    }

    fn link_reg(&self) -> Option<Self::Register> {
        Some(WasmRegister::new(RegKind::Lr, self.pointer))
    }

    fn intrinsics(&self) -> Vec<Self::Intrinsic> {
        (0..insn::operator_count() as u32)
            .map(WasmIntrinsic)
            .collect()
    }

    fn intrinsic_from_id(&self, id: IntrinsicId) -> Option<Self::Intrinsic> {
        (id.0 < insn::operator_count() as u32).then_some(WasmIntrinsic(id.0))
    }

    fn can_assemble(&self) -> bool {
        true
    }

    fn assemble(&self, code: &str, _addr: u64) -> Result<Vec<u8>, String> {
        asm::assemble(code)
    }

    /// An instruction can be neutralised whenever its operands can be dropped in the space it
    /// already occupies
    fn convert_to_nop(&self, data: &mut [u8], _addr: u64) -> bool {
        asm::nop_out(data)
    }

    fn is_never_branch_patch_available(&self, data: &[u8], _addr: u64) -> bool {
        asm::can_never_branch(data)
    }

    /// Taking a branch unconditionally means dropping its condition first, and the extra opcode
    /// does not fit in the space the original occupies
    fn is_always_branch_patch_available(&self, _data: &[u8], _addr: u64) -> bool {
        false
    }

    /// There is no branch-unless opcode to swap in, and negating the condition needs a byte the
    /// encoding has no room for
    fn is_invert_branch_patch_available(&self, _data: &[u8], _addr: u64) -> bool {
        false
    }

    /// How many arguments a call takes depends on a type declared elsewhere, so its result cannot
    /// be faked in place
    fn is_skip_and_return_zero_patch_available(&self, _data: &[u8], _addr: u64) -> bool {
        false
    }

    fn is_skip_and_return_value_patch_available(&self, _data: &[u8], _addr: u64) -> bool {
        false
    }

    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

/// The view reaches past the image, so anything sizing a read of the file itself wants this rather
/// than [`BinaryViewBase::end`]
pub(crate) fn file_end(view: &BinaryView) -> u64 {
    match module::layout(view_id(view)) {
        Some(layout) => layout.file_address(layout.image),
        None => view.end(),
    }
}

/// How the function covering `addr` lays its locals out, and whether this is its first instruction
fn frame_at(module: &module::Module, addr: u64) -> Option<lift::Frame> {
    let (function, info) = module.body_covering(addr)?;
    Some(lift::Frame {
        params: info.signature.params.len() as u32,
        locals: module.local_count(function),
        results: info.signature.results.len() as u32,
        result: info.signature.results.first().copied(),
        entry: addr == info.entry,
    })
}

/// Without it the lifter moves every local at the slot width, writing eight bytes into a slot the
/// next operator reads four out of, which the core warns about on every access
fn moved_width(module: &module::Module, addr: u64, op: &Operator) -> Option<usize> {
    let kind = match op {
        Operator::LocalGet { local_index }
        | Operator::LocalSet { local_index }
        | Operator::LocalTee { local_index } => {
            let (function, _) = module.body_covering(addr)?;
            module.local_kind(function, *local_index)?
        }
        Operator::GlobalGet { global_index } | Operator::GlobalSet { global_index } => {
            module.global_kind(*global_index)?
        }
        _ => return None,
    };

    // A `v128` is wider than a slot, and the model has nowhere to put the rest of it
    Some(kind.size().min(SLOT as usize))
}

/// Every raw `.wasm` view starts at zero, so without this two files open at once would share one
/// set of cached answers
pub(crate) fn view_id(view: &BinaryView) -> ViewId {
    view.file().session_id().0 as ViewId
}

/// How many branches an [`InstructionInfo`] can carry, extras being dropped silently
const BRANCH_SLOTS: usize = 3;

/// Only three branches fit in an [`InstructionInfo`], so a `br_table` is reported as one indirect
/// branch and its real edges reach the core through basic block analysis
fn describe(terminator: &Terminator, info: &mut InstructionInfo) {
    match terminator {
        Terminator::Jump(target) => info.add_branch(BranchKind::Unconditional(*target)),
        Terminator::Branch { taken, not_taken } => {
            info.add_branch(BranchKind::True(*taken));
            info.add_branch(BranchKind::False(*not_taken));
        }
        // A table knows exactly where it goes, but only three branches fit here
        Terminator::Table { targets, default } => {
            let mut distinct: Vec<u64> = Vec::new();
            let mut leaves = false;
            for target in targets.iter().chain([default]) {
                match target {
                    Some(to) if !distinct.contains(to) => distinct.push(*to),
                    Some(_) => {}
                    None => leaves = true,
                }
            }

            if distinct.len() + usize::from(leaves) <= BRANCH_SLOTS {
                for to in distinct {
                    info.add_branch(BranchKind::Unconditional(to));
                }
                if leaves {
                    info.add_branch(BranchKind::FunctionReturn);
                }
            } else {
                info.add_branch(BranchKind::Indirect);
            }
        }
        Terminator::Return => info.add_branch(BranchKind::FunctionReturn),
        Terminator::ConditionalReturn { not_taken } => {
            info.add_branch(BranchKind::FunctionReturn);
            info.add_branch(BranchKind::False(*not_taken));
        }
        // A trap continues nowhere, and the `noret` in the IL stops anything downstream reading
        // this as a real return
        Terminator::Halt => info.add_branch(BranchKind::FunctionReturn),
        Terminator::Unresolved => info.add_branch(BranchKind::Unresolved),
    }
}

/// `br_table` and multi-value `select` produce one token per entry, so each stays individually
/// selectable in the UI
fn operand_tokens(operand: &Operand) -> Vec<(String, InstructionTextTokenKind)> {
    match operand {
        Operand::Index { value, .. } => vec![(
            value.to_string(),
            InstructionTextTokenKind::Integer {
                operand: None,
                value: u64::from(*value),
                size: Some(4),
            },
        )],
        Operand::I32(value) => vec![(
            value.to_string(),
            InstructionTextTokenKind::Integer {
                operand: None,
                value: *value as u64,
                size: Some(4),
            },
        )],
        Operand::I64(value) => vec![(
            value.to_string(),
            InstructionTextTokenKind::Integer {
                operand: None,
                value: *value as u64,
                size: Some(8),
            },
        )],
        Operand::F32(value) => vec![(
            insn::wat_f32(*value),
            InstructionTextTokenKind::FloatingPoint {
                value: f64::from(*value),
                size: Some(4),
            },
        )],
        Operand::F64(value) => vec![(
            insn::wat_f64(*value),
            InstructionTextTokenKind::FloatingPoint {
                value: *value,
                size: Some(8),
            },
        )],
        // A token's numeric value is only 64 bits wide, so the text carries the whole vector
        Operand::V128(value) => vec![(
            format!("{value:#034x}"),
            InstructionTextTokenKind::Integer {
                operand: None,
                value: *value as u64,
                size: Some(16),
            },
        )],
        Operand::Lane(lane) => vec![(
            lane.to_string(),
            InstructionTextTokenKind::Integer {
                operand: None,
                value: u64::from(*lane),
                size: Some(1),
            },
        )],
        Operand::Lanes(lanes) => lanes
            .iter()
            .map(|lane| {
                (
                    lane.to_string(),
                    InstructionTextTokenKind::Integer {
                        operand: None,
                        value: u64::from(*lane),
                        size: Some(1),
                    },
                )
            })
            .collect(),
        Operand::MemArg {
            align,
            offset,
            memory,
        } => {
            let mut tokens = Vec::new();
            if *memory != 0 {
                tokens.push((
                    format!("memory={memory}"),
                    InstructionTextTokenKind::Annotation,
                ));
            }
            // The order the text format writes them in, and the only one the assembler takes
            tokens.push((
                format!("offset={offset}"),
                InstructionTextTokenKind::Annotation,
            ));
            // The text format writes the byte count, and a value with no byte count is no
            // alignment at all, so it is shown as the power it claims to be
            tokens.push((
                match 1u64.checked_shl(u32::from(*align)) {
                    Some(bytes) => format!("align={bytes}"),
                    None => format!("align=2^{align}"),
                },
                InstructionTextTokenKind::Annotation,
            ));
            tokens
        }
        // An empty block type is spelled by writing nothing at all
        Operand::BlockType(None) => Vec::new(),
        Operand::BlockType(Some(name)) | Operand::Type(name) => {
            vec![(name.clone(), InstructionTextTokenKind::TypeName)]
        }
        Operand::Types(names) => names
            .iter()
            .map(|name| (name.clone(), InstructionTextTokenKind::TypeName))
            .collect(),
        Operand::Labels(labels) => labels
            .iter()
            .map(|label| {
                (
                    label.to_string(),
                    InstructionTextTokenKind::Integer {
                        operand: None,
                        value: u64::from(*label),
                        size: Some(4),
                    },
                )
            })
            .collect(),
        Operand::Ordering(name) => vec![((*name).to_owned(), InstructionTextTokenKind::Annotation)],
        Operand::Table(summary) => vec![(summary.clone(), InstructionTextTokenKind::Annotation)],
    }
}

/// Two architectures, because a memory64 module has eight byte pointers and the width the core is
/// told has to be the width the lifter computes addresses at
pub fn register() {
    for (name, pointer) in [(NAME, 4usize), (NAME64, 8usize)] {
        let arch = architecture::register_architecture(name, WasmArchitecture::new(pointer));
        let convention = register_calling_convention(arch, name, WasmCallingConvention);
        arch.as_ref().set_default_calling_convention(&convention);
    }
}

/// The architecture a module of this shape belongs to
pub fn name_for(memory64: bool) -> &'static str {
    if memory64 { NAME64 } else { NAME }
}

/// Whether `name` is one of this plugin's architectures
pub fn is_ours(name: &str) -> bool {
    name == NAME || name == NAME64
}

/// Turns a recovered edge into the shape the core wants it
fn pending_edge(edge: Edge, arch: CoreArchitecture) -> PendingBasicBlockEdge {
    let (kind, target, fallthrough) = match edge {
        Edge::Unconditional(target) => (BranchType::UnconditionalBranch, target, false),
        Edge::True(target) => (BranchType::TrueBranch, target, false),
        Edge::False(target) => (BranchType::FalseBranch, target, true),
        Edge::FunctionReturn => (BranchType::FunctionReturn, 0, false),
        Edge::Unresolved => (BranchType::UnresolvedBranch, 0, false),
    };

    PendingBasicBlockEdge::new(kind, target, arch, fallthrough)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(operand: &Operand) -> Vec<String> {
        operand_tokens(operand)
            .into_iter()
            .map(|(text, _)| text)
            .collect()
    }

    #[test]
    fn register_ids_are_distinct_and_round_trip() {
        let mut seen = Vec::new();
        for reg in RegKind::ALL {
            assert!(!seen.contains(&reg.id()), "{reg:?} reuses an id");
            seen.push(reg.id());
            assert_eq!(
                RegKind::ALL.into_iter().find(|r| r.id() == reg.id()),
                Some(reg)
            );
        }
    }

    #[test]
    fn every_operator_has_an_intrinsic_id() {
        assert!(insn::operator_count() > 600);

        for id in 0..insn::operator_count() as u32 {
            let name = insn::operator_name(id).expect("every id names an operator");
            assert!(!name.is_empty());
            assert_eq!(WasmIntrinsic(id).id(), IntrinsicId(id));
        }

        assert!(insn::operator_name(insn::operator_count() as u32).is_none());
    }

    #[test]
    fn register_ids_stay_out_of_the_temporary_range() {
        for reg in RegKind::ALL {
            assert!(!reg.id().is_temporary(), "{reg:?}");
        }
    }

    #[test]
    fn a_register_holding_an_address_is_as_wide_as_one() {
        for kind in [RegKind::Sp, RegKind::Fp, RegKind::Lr] {
            assert_eq!(WasmRegister::new(kind, 4).size(), 4);
            assert_eq!(WasmRegister::new(kind, 8).size(), 8);
        }
        // A result is any wasm value, so it takes a whole slot either way
        assert_eq!(WasmRegister::new(RegKind::Rv, 4).size(), SLOT as usize);
        assert_eq!(WasmRegister::new(RegKind::Rv, 8).size(), SLOT as usize);
    }

    #[test]
    fn scalar_operands_render_as_one_token() {
        assert_eq!(
            render(&Operand::Index {
                field: "local_index",
                value: 7
            }),
            ["7"]
        );
        assert_eq!(render(&Operand::I32(-1)), ["-1"]);
        assert_eq!(render(&Operand::I64(128)), ["128"]);
        assert_eq!(render(&Operand::Lane(3)), ["3"]);
        assert_eq!(render(&Operand::Type("funcref".to_owned())), ["funcref"]);
    }

    #[test]
    fn a_memarg_renders_its_memory_only_when_it_is_not_the_default() {
        assert_eq!(
            render(&Operand::MemArg {
                align: 2,
                offset: 16,
                memory: 0
            }),
            ["offset=16", "align=4"]
        );
        assert_eq!(
            render(&Operand::MemArg {
                align: 0,
                offset: 0,
                memory: 2
            }),
            ["memory=2", "offset=0", "align=1"]
        );
    }

    #[test]
    fn an_impossible_alignment_renders_as_the_power_it_claims() {
        assert_eq!(
            render(&Operand::MemArg {
                align: 96,
                offset: 1,
                memory: 0
            }),
            ["offset=1", "align=2^96"]
        );
        assert_eq!(
            render(&Operand::MemArg {
                align: 63,
                offset: 0,
                memory: 0
            }),
            ["offset=0", "align=9223372036854775808"]
        );
    }

    #[test]
    fn an_empty_block_type_renders_nothing() {
        assert!(render(&Operand::BlockType(None)).is_empty());
        assert_eq!(render(&Operand::BlockType(Some("i32".to_owned()))), ["i32"]);
    }

    #[test]
    fn tables_render_one_token_per_entry() {
        assert_eq!(render(&Operand::Labels(vec![0, 1, 3])), ["0", "1", "3"]);
        assert_eq!(
            render(&Operand::Types(vec!["i32".to_owned(), "i64".to_owned()])),
            ["i32", "i64"]
        );
        assert_eq!(render(&Operand::Lanes([0; 16])).len(), 16);
    }

    #[test]
    fn a_vector_constant_keeps_all_of_its_bits_in_the_text() {
        assert_eq!(
            render(&Operand::V128(u128::MAX)),
            ["0xffffffffffffffffffffffffffffffff"]
        );
    }

    #[test]
    fn every_operand_kind_renders() {
        let operands = [
            Operand::Index {
                field: "mem",
                value: 0,
            },
            Operand::I32(0),
            Operand::I64(0),
            Operand::F32(0.5),
            Operand::F64(0.5),
            Operand::V128(0),
            Operand::Lane(0),
            Operand::Lanes([0; 16]),
            Operand::MemArg {
                align: 1,
                offset: 0,
                memory: 0,
            },
            Operand::BlockType(Some("i32".to_owned())),
            Operand::Labels(vec![0]),
            Operand::Type("any".to_owned()),
            Operand::Types(vec!["i32".to_owned()]),
            Operand::Ordering("seq_cst"),
            Operand::Table("0 catches".to_owned()),
        ];

        for operand in &operands {
            assert!(!render(operand).is_empty(), "{operand:?} rendered nothing");
        }
    }
}
