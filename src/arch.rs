//! The Binary Ninja `Architecture` implementation
//!
//! Glue only: decoding lives in [`crate::insn`], control flow recovery in [`crate::cfg`], and
//! semantics in [`crate::lift`]

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::OnceLock;

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

pub const STACK_REGISTERS: u32 = 256;

pub const ARGUMENT_REGISTERS: u32 = 64;

pub const RESULT_REGISTERS: u32 = 64;

pub const CAUGHT_REGISTERS: u32 = 8;

pub const LOCAL_REGISTERS: u32 = 4096;

const EXCEPTION_BASE: u32 = 2 + 2 * (STACK_REGISTERS + ARGUMENT_REGISTERS);

const CAUGHT_BASE: u32 = EXCEPTION_BASE + 1;

const LOCAL_BASE: u32 = CAUGHT_BASE + CAUGHT_REGISTERS;

const RESULT_BASE: u32 = LOCAL_BASE + 2 * LOCAL_REGISTERS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegKind {
    Sp,
    /// Return address, since wasm keeps its call stack out of reach of the program
    Lr,
    Stack(u32),
    Argument(u32),
    Exception,
    Caught(u32),
    Local(u32),
    Result(u32),
}

/// The width comes along because [`RegisterInfo::size`] is asked without an architecture to ask
/// about, and a register holding an address is as wide as an address is
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WasmRegister {
    pub kind: RegKind,
    pub low: bool,
    /// Width of an address in the architecture this register belongs to
    pub pointer: usize,
}

impl WasmRegister {
    pub fn new(kind: RegKind, pointer: usize) -> Self {
        Self {
            kind,
            low: false,
            pointer,
        }
    }

    pub fn sized(kind: RegKind, width: usize, pointer: usize) -> Self {
        Self {
            kind,
            low: width < SLOT as usize,
            pointer,
        }
    }

    fn from_id(id: RegisterId, pointer: usize) -> Option<Self> {
        let kind = match id.0 {
            0 => RegKind::Sp,
            1 => RegKind::Lr,
            n if n < 2 + 2 * STACK_REGISTERS => RegKind::Stack((n - 2) / 2),
            n if n < EXCEPTION_BASE => RegKind::Argument((n - 2 - 2 * STACK_REGISTERS) / 2),
            EXCEPTION_BASE => RegKind::Exception,
            n if n < LOCAL_BASE => RegKind::Caught(n - CAUGHT_BASE),
            n if n < RESULT_BASE => RegKind::Local((n - LOCAL_BASE) / 2),
            n if n < RESULT_BASE + 2 * RESULT_REGISTERS => RegKind::Result((n - RESULT_BASE) / 2),
            _ => return None,
        };
        let low = match kind {
            RegKind::Stack(_) | RegKind::Argument(_) => id.0 % 2 == 1,
            RegKind::Local(_) => (id.0 - LOCAL_BASE) % 2 == 1,
            RegKind::Result(_) => (id.0 - RESULT_BASE) % 2 == 1,
            _ => false,
        };
        Some(Self { kind, low, pointer })
    }

    fn all(pointer: usize, low: bool) -> impl Iterator<Item = Self> {
        let halved = move |kind| {
            let full = Self::new(kind, pointer);
            std::iter::once(full).chain(low.then_some(Self { low: true, ..full }))
        };
        let operands = (0..STACK_REGISTERS)
            .map(RegKind::Stack)
            .chain((0..ARGUMENT_REGISTERS).map(RegKind::Argument));
        let unpaired =
            std::iter::once(RegKind::Exception).chain((0..CAUGHT_REGISTERS).map(RegKind::Caught));
        let values = (0..LOCAL_REGISTERS)
            .map(RegKind::Local)
            .chain((0..RESULT_REGISTERS).map(RegKind::Result));
        [RegKind::Sp, RegKind::Lr]
            .into_iter()
            .map(move |kind| Self::new(kind, pointer))
            .chain(operands.flat_map(halved))
            .chain(unpaired.map(move |kind| Self::new(kind, pointer)))
            .chain(values.flat_map(halved))
    }
}

impl Register for WasmRegister {
    type InfoType = Self;

    fn name(&self) -> Cow<'_, str> {
        let half = if self.low { "d" } else { "" };
        match self.kind {
            RegKind::Sp => Cow::Borrowed("sp"),
            RegKind::Lr => Cow::Borrowed("lr"),
            RegKind::Stack(n) => Cow::Owned(format!("s{n}{half}")),
            RegKind::Argument(n) => Cow::Owned(format!("a{n}{half}")),
            RegKind::Exception => Cow::Borrowed("exn"),
            RegKind::Caught(n) => Cow::Owned(format!("caught{n}")),
            RegKind::Local(n) => Cow::Owned(format!("l{n}{half}")),
            RegKind::Result(n) => Cow::Owned(format!("r{n}{half}")),
        }
    }

    fn info(&self) -> Self {
        *self
    }

    fn id(&self) -> RegisterId {
        let paired = match self.kind {
            RegKind::Sp => return RegisterId(0),
            RegKind::Lr => return RegisterId(1),
            RegKind::Stack(n) => 2 + 2 * n,
            RegKind::Argument(n) => 2 + 2 * (STACK_REGISTERS + n),
            RegKind::Exception => return RegisterId(EXCEPTION_BASE),
            RegKind::Caught(n) => return RegisterId(CAUGHT_BASE + n),
            RegKind::Local(n) => LOCAL_BASE + 2 * n,
            RegKind::Result(n) => RESULT_BASE + 2 * n,
        };
        RegisterId(paired + u32::from(self.low))
    }
}

impl RegisterInfo for WasmRegister {
    type RegType = Self;

    fn parent(&self) -> Option<Self> {
        self.low.then_some(Self {
            low: false,
            ..*self
        })
    }

    fn size(&self) -> usize {
        match self.kind {
            RegKind::Sp | RegKind::Lr | RegKind::Exception | RegKind::Caught(_) => self.pointer,
            _ if self.low => 4,
            _ => SLOT as usize,
        }
    }

    fn offset(&self) -> usize {
        0
    }

    fn implicit_extend(&self) -> ImplicitRegisterExtend {
        if self.low {
            ImplicitRegisterExtend::ZeroExtendToFullWidth
        } else {
            ImplicitRegisterExtend::NoExtend
        }
    }
}

/// Every operator gets one whether or not [`crate::lift`] models it, so an id read back from a
/// saved database always resolves
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WasmIntrinsic {
    pub id: u32,
    pub pointer: usize,
}

const THROWN_NAME: &str = "thrown";
const EXCEPTION_VALUE_NAME: &str = "exception_value";
const RETHROW_NAME: &str = "rethrow";
const THROW_NAME: &str = "throw";
const SUSPENDED_NAME: &str = "suspended";

pub const THROWN: u32 = insn::stable_id(THROWN_NAME);

pub const EXCEPTION_VALUE: u32 = insn::stable_id(EXCEPTION_VALUE_NAME);

pub const RETHROW: u32 = insn::stable_id(RETHROW_NAME);

pub const SUSPENDED: u32 = insn::stable_id(SUSPENDED_NAME);

const ESCAPES: [u32; 4] = [THROWN, EXCEPTION_VALUE, RETHROW, SUSPENDED];

pub fn throw(values: u32) -> Option<u32> {
    (values <= ARGUMENT_REGISTERS).then(|| insn::stable_id(&format!("{THROW_NAME}/{values}")))
}

fn thrown_values(id: u32) -> Option<u32> {
    static BY_ID: OnceLock<HashMap<u32, u32>> = OnceLock::new();
    BY_ID
        .get_or_init(|| {
            (0..=ARGUMENT_REGISTERS)
                .filter_map(|values| Some((throw(values)?, values)))
                .collect()
        })
        .get(&id)
        .copied()
}

fn intrinsic_ids() -> impl Iterator<Item = u32> {
    insn::operator_ids()
        .chain(ESCAPES)
        .chain((0..=ARGUMENT_REGISTERS).filter_map(throw))
}

impl Intrinsic for WasmIntrinsic {
    fn name(&self) -> Cow<'_, str> {
        match self.id {
            THROWN => Cow::Borrowed(THROWN_NAME),
            EXCEPTION_VALUE => Cow::Borrowed(EXCEPTION_VALUE_NAME),
            RETHROW => Cow::Borrowed(RETHROW_NAME),
            SUSPENDED => Cow::Borrowed(SUSPENDED_NAME),
            id if thrown_values(id).is_some() => Cow::Borrowed(THROW_NAME),
            id => Cow::Owned(insn::operator_name(id).unwrap_or_else(|| format!("op{id}"))),
        }
    }

    fn id(&self) -> IntrinsicId {
        IntrinsicId(self.id)
    }

    /// An operator's operands and immediates are slots, since a value's type belongs to the
    /// operator rather than the operand; the lifter's own intrinsics say what they take
    fn inputs(&self) -> Vec<NameAndType> {
        let exception = || NameAndType::new("exception", Type::int(self.pointer, false).into());
        match self.id {
            THROWN | SUSPENDED => Vec::new(),
            EXCEPTION_VALUE => vec![
                exception(),
                NameAndType::new("index", Type::int(4, false).into()),
            ],
            RETHROW => vec![exception()],
            id if let Some(values) = thrown_values(id) => std::iter::once(NameAndType::new(
                "tag",
                Type::int(self.pointer, false).into(),
            ))
            .chain(
                (0..values).map(|nth| NameAndType::new(format!("value{nth}"), slot_type().into())),
            )
            .collect(),
            id if let Some(conversion) = lift::saturating(id) => {
                vec![NameAndType::new(
                    "value",
                    Type::float(conversion.from).into(),
                )]
            }
            id => {
                let arity = insn::operator_arity(id).unwrap_or_default();
                (0..arity.pops)
                    .map(|index| format!("arg{index}"))
                    .chain(insn::operator_immediates(id))
                    .map(|name| NameAndType::new(name, slot_type().into()))
                    .collect()
            }
        }
    }

    fn outputs(&self) -> Vec<Conf<Ref<Type>>> {
        match self.id {
            THROWN | SUSPENDED => vec![Type::int(self.pointer, false).into()],
            EXCEPTION_VALUE => vec![slot_type().into()],
            RETHROW => Vec::new(),
            id if thrown_values(id).is_some() => Vec::new(),
            id if let Some(conversion) = lift::saturating(id) => {
                vec![Type::int(conversion.to, conversion.signed).into()]
            }
            id => {
                let arity = insn::operator_arity(id).unwrap_or_default();
                (0..arity.pushes).map(|_| slot_type().into()).collect()
            }
        }
    }
}

fn slot_type() -> Ref<Type> {
    Type::int(SLOT as usize, false)
}

struct WasmCallingConvention(CoreCallingConvention);

impl AsRef<CoreCallingConvention> for WasmCallingConvention {
    fn as_ref(&self) -> &CoreCallingConvention {
        &self.0
    }
}

fn register_id(kind: RegKind) -> RegisterId {
    WasmRegister::new(kind, 0).id()
}

fn argument_registers() -> Vec<RegisterId> {
    (0..ARGUMENT_REGISTERS)
        .map(|n| register_id(RegKind::Argument(n)))
        .collect()
}

pub fn local_of(id: RegisterId) -> Option<u32> {
    match WasmRegister::from_id(id, 0)? {
        WasmRegister {
            kind: RegKind::Local(n),
            low: false,
            ..
        } => Some(n),
        _ => None,
    }
}

pub fn is_operand(id: RegisterId) -> bool {
    matches!(
        WasmRegister::from_id(id, 0),
        Some(WasmRegister {
            kind: RegKind::Stack(_),
            ..
        })
    )
}

pub fn result_register(n: u32) -> RegisterId {
    register_id(RegKind::Result(n))
}

pub fn clobbered_registers() -> Vec<RegisterId> {
    argument_registers()
        .into_iter()
        .chain((0..RESULT_REGISTERS).map(result_register))
        .chain([register_id(RegKind::Exception)])
        .collect()
}

impl CallingConvention for WasmCallingConvention {
    fn caller_saved_registers(&self) -> Vec<RegisterId> {
        clobbered_registers()
    }

    fn callee_saved_registers(&self) -> Vec<RegisterId> {
        (0..STACK_REGISTERS)
            .map(RegKind::Stack)
            .chain([RegKind::Lr])
            .chain((0..CAUGHT_REGISTERS).map(RegKind::Caught))
            .map(register_id)
            .collect()
    }

    fn int_arg_registers(&self) -> Vec<RegisterId> {
        argument_registers()
    }

    fn float_arg_registers(&self) -> Vec<RegisterId> {
        argument_registers()
    }

    fn arg_registers_shared_index(&self) -> bool {
        true
    }

    fn reserved_stack_space_for_arg_registers(&self) -> bool {
        false
    }

    fn stack_adjusted_on_return(&self) -> bool {
        false
    }

    fn is_eligible_for_heuristics(&self) -> bool {
        false
    }

    fn return_int_reg(&self) -> Option<RegisterId> {
        Some(result_register(0))
    }

    fn return_hi_int_reg(&self) -> Option<RegisterId> {
        None
    }

    fn return_float_reg(&self) -> Option<RegisterId> {
        self.return_int_reg()
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
                Flow::Branch | Flow::Arm => info.add_branch(BranchKind::Unresolved),
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

        let module = module::lookup(id, start);
        let body_end = module
            .as_ref()
            .and_then(|module| module.body_covering(start).map(|(_, info)| info.end));
        let len = match body_end {
            Some(end) => (end - start).min(cfg::MAX_BODY_LEN as u64) as usize,
            None => file_end(&view)
                .saturating_sub(start)
                .min(context.max_function_size)
                .min(cfg::MAX_BODY_LEN as u64) as usize,
        };

        let code = view.read_vec(start, len);
        let flow = cfg::recover(&code, start, module.as_deref());
        cfg::install(id, &flow);

        let blocks = flow.reachable_blocks();
        if blocks.is_empty() {
            tracing::warn!(
                "wasm: nothing decoded at {start:#x}, {} of {len} bytes read",
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
                let offset = (at - start) as usize;
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
            let view = il.function().map(|function| view_id(&function.view()));
            let results = view
                .and_then(|view| module::import_signature(view, addr))
                .map(|signature| signature.results)
                .unwrap_or_default();
            lift::stub(il, self.pointer, &results);
            return Some((module::IMPORT_STRIDE as usize, true));
        }

        let decoded = insn::decode_any(data);
        trace::llil(addr, decoded.as_ref().map(|i| i.mnemonic()).as_deref());
        let insn = decoded?;

        // Unlike the other callbacks this one can say which file it is lifting, since the IL
        // belongs to a function and the function to a view
        let (recovered, module, height, dispatch) = match il.function() {
            Some(function) => {
                let view = view_id(&function.view());
                (
                    cfg::lookup(view, addr),
                    module::lookup(view, addr),
                    cfg::height(view, addr),
                    cfg::dispatch(view, addr),
                )
            }
            None => (
                cfg::lookup_anywhere(addr),
                module::lookup_anywhere(addr),
                cfg::height_anywhere(addr),
                cfg::dispatch_anywhere(addr),
            ),
        };

        let resolved = module.as_ref().and_then(|module| module.resolve(&insn.op));
        // The width comes from the architecture the view chose, and where the globals live from
        // the file, since each is laid out in a place of its own
        let layout = module
            .as_ref()
            .map(|module| module.layout)
            .or_else(|| module::layout_covering(addr))
            .unwrap_or_default();
        let global = match insn.op {
            Operator::GlobalGet { global_index } | Operator::GlobalSet { global_index } => module
                .as_ref()
                .and_then(|module| module.global_kind(global_index)),
            _ => None,
        };
        let model = Model {
            addr: self.pointer,
            layout,
            stack_pointer: module.as_ref().and_then(|module| {
                let (_, info) = module.body_covering(addr)?;
                info.frame.and(module.stack_pointer)
            }),
            frame: module
                .as_deref()
                .and_then(|module| frame_at(module, addr))
                .unwrap_or_default(),
            height,
            global,
        };
        lift::lift(
            il,
            model,
            &insn,
            recovered.as_ref(),
            resolved.as_ref(),
            dispatch.as_ref(),
        );
        // The flag reports whether lifting succeeded rather than whether control continues
        Some((insn.len, true))
    }

    fn registers_all(&self) -> Vec<Self::Register> {
        WasmRegister::all(self.pointer, true).collect()
    }

    fn registers_full_width(&self) -> Vec<Self::Register> {
        WasmRegister::all(self.pointer, false).collect()
    }

    fn register_from_id(&self, id: RegisterId) -> Option<Self::Register> {
        WasmRegister::from_id(id, self.pointer)
    }

    fn stack_pointer_reg(&self) -> Option<Self::Register> {
        Some(WasmRegister::new(RegKind::Sp, self.pointer))
    }

    fn link_reg(&self) -> Option<Self::Register> {
        Some(WasmRegister::new(RegKind::Lr, self.pointer))
    }

    fn intrinsics(&self) -> Vec<Self::Intrinsic> {
        intrinsic_ids()
            .map(|id| WasmIntrinsic {
                id,
                pointer: self.pointer,
            })
            .collect()
    }

    fn intrinsic_from_id(&self, id: IntrinsicId) -> Option<Self::Intrinsic> {
        (insn::operator_name(id.0).is_some()
            || ESCAPES.contains(&id.0)
            || thrown_values(id.0).is_some())
        .then_some(WasmIntrinsic {
            id: id.0,
            pointer: self.pointer,
        })
    }

    fn can_assemble(&self) -> bool {
        true
    }

    fn assemble(&self, code: &str, _addr: u64) -> Result<Vec<u8>, String> {
        asm::assemble(code)
    }

    /// An instruction can be neutralised whenever its operands can be dropped in the space it
    /// already occupies
    fn convert_to_nop(&self, data: &mut [u8], addr: u64) -> bool {
        if let Some(Operator::LocalSet { local_index }) = insn::decode(data).map(|insn| insn.op)
            && !module::lookup_anywhere(addr)
                .and_then(|module| {
                    let (function, _) = module.body_covering(addr)?;
                    module
                        .local_kinds(function)
                        .get(local_index as usize)
                        .copied()
                })
                .is_some_and(|kind| kind != module::ValueKind::Ref)
        {
            return false;
        }
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
fn frame_at(module: &module::Module, addr: u64) -> Option<lift::Frame<'_>> {
    let (function, info) = module.body_covering(addr)?;
    Some(lift::Frame {
        params: info.signature.params.len() as u32,
        locals: module.local_kinds(function),
        results: &info.signature.results,
        entry: addr == info.entry,
        stack: info.frame.map(|(holder, _)| holder),
        balanced: info.balanced,
    })
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
        Terminator::Table { .. } | Terminator::Suspend { .. } => {
            let edges = terminator.edges();
            if edges.len() <= BRANCH_SLOTS {
                for edge in edges {
                    info.add_branch(branch_kind(edge));
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
        // Nothing falls through a trap or a throw, and the IL ends one in `noret` or in the handler
        // a throw lands in, so nothing downstream reads this as a real return
        Terminator::Halt => info.add_branch(BranchKind::FunctionReturn),
        Terminator::Unresolved => info.add_branch(BranchKind::Unresolved),
    }
}

fn branch_kind(edge: Edge) -> BranchKind {
    match edge {
        Edge::Unconditional(to) => BranchKind::Unconditional(to),
        Edge::True(to) => BranchKind::True(to),
        Edge::False(to) => BranchKind::False(to),
        Edge::FunctionReturn => BranchKind::FunctionReturn,
        Edge::Unresolved => BranchKind::Unresolved,
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
        Operand::Table(summary) if summary.is_empty() => Vec::new(),
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
        let all: Vec<_> = WasmRegister::all(4, true).collect();
        let end = RESULT_BASE + 2 * RESULT_REGISTERS;
        assert_eq!(all.len(), end as usize);
        let mut seen = std::collections::BTreeSet::new();
        for reg in all {
            assert!(seen.insert(reg.id().0), "{reg:?} reuses an id");
            assert_eq!(WasmRegister::from_id(reg.id(), 4), Some(reg));
        }
        assert_eq!(WasmRegister::from_id(RegisterId(end), 4), None);
    }

    #[test]
    fn a_saved_register_id_still_names_the_same_register() {
        let named =
            |id: u32| WasmRegister::from_id(RegisterId(id), 4).map(|reg| reg.name().into_owned());
        assert_eq!(named(3).as_deref(), Some("s0d"));
        assert_eq!(named(EXCEPTION_BASE).as_deref(), Some("exn"));
        assert_eq!(named(EXCEPTION_BASE + 1).as_deref(), Some("caught0"));
        assert_eq!(named(EXCEPTION_BASE + 9).as_deref(), Some("l0"));
        assert_eq!(named(EXCEPTION_BASE + 12).as_deref(), Some("l1d"));
        assert_eq!(named(RESULT_BASE + 1).as_deref(), Some("r0d"));
    }

    #[test]
    fn every_operator_has_an_intrinsic_id_of_its_own() {
        let ids: Vec<u32> = insn::operator_ids().collect();
        assert!(ids.len() > 600);
        for id in &ids {
            let name = insn::operator_name(*id).expect("every id names an operator");
            assert!(!name.is_empty());
            assert_eq!(
                WasmIntrinsic {
                    id: *id,
                    pointer: 4
                }
                .id(),
                IntrinsicId(*id)
            );
        }

        let every: Vec<u32> = intrinsic_ids().collect();
        let distinct: std::collections::BTreeSet<u32> = every.iter().copied().collect();
        assert_eq!(distinct.len(), every.len(), "two intrinsics hash to one id");
        assert_eq!(
            every.len(),
            ids.len() + ESCAPES.len() + ARGUMENT_REGISTERS as usize + 1
        );
        for values in [0, 3, ARGUMENT_REGISTERS] {
            let id = throw(values).expect("within the limit");
            assert_eq!(thrown_values(id), Some(values));
        }
        assert_eq!(throw(ARGUMENT_REGISTERS + 1), None);
    }

    #[test]
    fn an_intrinsic_id_is_the_same_in_every_build() {
        let id = |bytes: &[u8]| insn::decode(bytes).and_then(|insn| insn.operator_id());
        assert_eq!(id(&[0x08, 0x00]), Some(0x10a7_f343), "throw");
        assert_eq!(id(&[0x6a]), Some(0xbdc8_d671), "i32.add");
        assert_eq!(id(&[0x00]), Some(0x2167_5135), "unreachable");
        assert_eq!((THROWN, EXCEPTION_VALUE), (0x0ca2_2853, 0x3c2a_9948));
    }

    #[test]
    fn register_ids_stay_out_of_the_temporary_range() {
        for reg in WasmRegister::all(8, true) {
            assert!(!reg.id().is_temporary(), "{reg:?}");
        }
    }

    #[test]
    fn a_register_holding_an_address_is_as_wide_as_one() {
        for kind in [RegKind::Sp, RegKind::Lr] {
            assert_eq!(WasmRegister::new(kind, 4).size(), 4);
            assert_eq!(WasmRegister::new(kind, 8).size(), 8);
        }
    }

    #[test]
    fn a_value_register_is_a_slot_with_a_zero_extending_low_half() {
        for kind in [RegKind::Stack(3), RegKind::Argument(0), RegKind::Result(2)] {
            let full = WasmRegister::sized(kind, 8, 4);
            let low = WasmRegister::sized(kind, 4, 4);
            assert_eq!(full.size(), SLOT as usize);
            assert_eq!(full.parent(), None);
            assert_eq!(low.size(), 4);
            assert_eq!(low.parent(), Some(full));
            assert_eq!(
                low.implicit_extend(),
                ImplicitRegisterExtend::ZeroExtendToFullWidth
            );
            assert_ne!(low.id(), full.id());
        }
        assert_eq!(WasmRegister::sized(RegKind::Stack(3), 4, 4).name(), "s3d");
        assert_eq!(WasmRegister::sized(RegKind::Argument(1), 8, 4).name(), "a1");
        assert_eq!(WasmRegister::sized(RegKind::Result(0), 4, 4).name(), "r0d");
    }

    #[test]
    fn arguments_share_one_register_list_for_every_type() {
        let registers = argument_registers();
        assert_eq!(registers.len(), ARGUMENT_REGISTERS as usize);
        assert_eq!(registers[0], register_id(RegKind::Argument(0)));
        assert!(registers.iter().all(|id| matches!(
            WasmRegister::from_id(*id, 4).map(|r| (r.kind, r.low)),
            Some((RegKind::Argument(_), false))
        )));
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
            Operand::Table("(catch_all 0)".to_owned()),
        ];

        for operand in &operands {
            assert!(!render(operand).is_empty(), "{operand:?} rendered nothing");
        }
        assert!(
            render(&Operand::Table(String::new())).is_empty(),
            "a try_table with no result and no clauses"
        );
    }
}
