//! Single-instruction decoding on top of `wasmparser`

use std::collections::HashMap;
use std::sync::OnceLock;

use wasmparser::{
    AbstractHeapType, BinaryReader, BlockType, BrTable, Catch, FrameKind, FrameStack, Handle,
    HeapType, Ieee32, Ieee64, MemArg, Operator, Ordering, RefType, ResumeTable, TryTable,
    UnpackedIndex, V128, ValType, VisitOperator, VisitSimdOperator,
};

/// The most the core hands over for one instruction's text and info, so there a `br_table` past
/// around 250 entries is invalid rather than truncated; analysis and lifting read whole bodies
/// and decode it with [`decode_any`]
pub const MAX_INSTR_LEN: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub struct Instruction<'a> {
    pub op: Operator<'a>,
    pub len: usize,
}

impl Instruction<'_> {
    pub fn mnemonic(&self) -> String {
        mnemonic(&self.op).unwrap_or_else(|| UNKNOWN_MNEMONIC.to_owned())
    }

    pub fn operands(&self) -> Vec<Operand> {
        operands(&self.op)
    }

    pub fn flow(&self) -> Flow {
        flow(&self.op)
    }

    /// `None` when the effect depends on a type the module declares elsewhere
    pub fn arity(&self) -> Option<Arity> {
        arity(&self.op)
    }

    pub fn operator_id(&self) -> Option<u32> {
        operator_id(&self.op)
    }
}

const UNKNOWN_MNEMONIC: &str = "(unknown)";

pub fn decode(data: &[u8]) -> Option<Instruction<'_>> {
    decode_any(data).filter(|insn| insn.len <= MAX_INSTR_LEN)
}

/// Uncapped, unlike [`decode`], for the callers that have the whole body: giving up at an
/// oversized `br_table` would cost the rest of it its control flow
pub fn decode_any(data: &[u8]) -> Option<Instruction<'_>> {
    // Some operators are legal only inside a particular block, and disassembly starts at an
    // arbitrary address; between these two frames every operator decodes
    for frame in [FrameKind::If, FrameKind::LegacyTry] {
        let mut reader = BinaryReader::new(data, 0);
        if let Ok(op) = reader.visit_operator(&mut OperatorFactory { frame }) {
            return Some(Instruction {
                op,
                len: usize::try_from(reader.original_position()).ok()?,
            });
        }
    }

    None
}

/// `wasmparser` has an equivalent internally but does not expose it, and its public reader tracks
/// a control stack this decoder cannot supply
struct OperatorFactory {
    frame: FrameKind,
}

impl FrameStack for OperatorFactory {
    fn current_frame(&self) -> Option<FrameKind> {
        Some(self.frame)
    }
}

macro_rules! define_visitors {
    ($( @$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident ($($ann:tt)*) )*) => {
        $(
            fn $visit(&mut self $($(, $arg: $argty)*)?) -> Self::Output {
                Operator::$op $({ $($arg),* })?
            }
        )*
    };
}

impl<'a> VisitOperator<'a> for OperatorFactory {
    type Output = Operator<'a>;

    fn simd_visitor(&mut self) -> Option<&mut dyn VisitSimdOperator<'a, Output = Self::Output>> {
        Some(self)
    }

    wasmparser::for_each_visit_operator!(define_visitors);
}

impl<'a> VisitSimdOperator<'a> for OperatorFactory {
    wasmparser::for_each_visit_simd_operator!(define_visitors);
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Arity {
    pub pops: u32,
    pub pushes: u32,
}

/// Targets are absent on purpose: a label is relative to the enclosing blocks, which one
/// instruction does not carry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Normal,
    Trap,
    BlockStart,
    BlockEnd,
    /// Control reaches an arm along its own edge, never by falling out of the arm above it
    Arm,
    Branch,
    ConditionalBranch,
    IndirectBranch,
    Return,
    Call,
    IndirectCall,
    TailCall,
}

impl Flow {
    pub fn falls_through(self) -> bool {
        !matches!(
            self,
            Flow::Trap
                | Flow::Return
                | Flow::Branch
                | Flow::IndirectBranch
                | Flow::TailCall
                | Flow::Arm
        )
    }
}

pub fn raises(op: &Operator) -> bool {
    matches!(
        op,
        Operator::Call { .. }
            | Operator::CallIndirect { .. }
            | Operator::CallRef { .. }
            | Operator::Suspend { .. }
            | Operator::Resume { .. }
            | Operator::ResumeThrow { .. }
            | Operator::ResumeThrowRef { .. }
            | Operator::Switch { .. }
            | Operator::Throw { .. }
            | Operator::ThrowRef
            | Operator::Rethrow { .. }
    )
}

pub fn conditional_label(op: &Operator) -> Option<u32> {
    match *op {
        Operator::BrIf { relative_depth }
        | Operator::BrOnNull { relative_depth }
        | Operator::BrOnNonNull { relative_depth }
        | Operator::BrOnCast { relative_depth, .. }
        | Operator::BrOnCastFail { relative_depth, .. }
        | Operator::BrOnCastDescEq { relative_depth, .. }
        | Operator::BrOnCastDescEqFail { relative_depth, .. } => Some(relative_depth),
        _ => None,
    }
}

pub fn resume_table<'a>(op: &'a Operator) -> Option<&'a ResumeTable> {
    match op {
        Operator::Resume { resume_table, .. }
        | Operator::ResumeThrow { resume_table, .. }
        | Operator::ResumeThrowRef { resume_table, .. } => Some(resume_table),
        _ => None,
    }
}

pub fn catch_label(catch: &Catch) -> u32 {
    match *catch {
        Catch::One { label, .. }
        | Catch::OneRef { label, .. }
        | Catch::All { label }
        | Catch::AllRef { label } => label,
    }
}

pub fn flow(op: &Operator) -> Flow {
    match op {
        Operator::Unreachable
        | Operator::Throw { .. }
        | Operator::ThrowRef
        | Operator::Rethrow { .. } => Flow::Trap,
        Operator::Block { .. }
        | Operator::Loop { .. }
        | Operator::If { .. }
        | Operator::Try { .. }
        | Operator::TryTable { .. } => Flow::BlockStart,
        Operator::End | Operator::Delegate { .. } => Flow::BlockEnd,
        Operator::Else | Operator::Catch { .. } | Operator::CatchAll => Flow::Arm,
        Operator::Br { .. } => Flow::Branch,
        op if conditional_label(op).is_some() => Flow::ConditionalBranch,
        Operator::BrTable { .. } => Flow::IndirectBranch,
        Operator::Return => Flow::Return,
        Operator::Call { .. } => Flow::Call,
        Operator::CallIndirect { .. } | Operator::CallRef { .. } => Flow::IndirectCall,
        Operator::ReturnCall { .. }
        | Operator::ReturnCallIndirect { .. }
        | Operator::ReturnCallRef { .. } => Flow::TailCall,
        _ => Flow::Normal,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// Named after the `wasmparser` payload field it came from, such as `local_index`
    Index {
        field: &'static str,
        value: u32,
    },
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    V128(u128),
    Lane(u8),
    Lanes([u8; 16]),
    MemArg {
        /// A power-of-two exponent, not a byte count: sweeping data turns up values no real
        /// alignment could have, so the shift belongs to whoever renders it
        align: u8,
        offset: u64,
        memory: u32,
    },
    BlockType(Option<String>),
    /// `br_table` targets, followed by its default
    Labels(Vec<u32>),
    Type(String),
    Types(Vec<String>),
    Ordering(&'static str),
    /// A branch or handler table too structured to render as a number
    Table(String),
}

type ImmediateTable = Vec<Vec<(&'static str, &'static [&'static str])>>;

struct Immediates<'a>(std::marker::PhantomData<&'a ()>);

/// Generated from the operator list `wasmparser` exposes, so new proposals arrive with the crate
macro_rules! define_operator_table {
    ($( @$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident (arity $($arity:tt)*) )*) => {
        /// An operator's stable identity is its name here, hashed by [`stable_id`]
        static VISITORS: &[&str] = &[$(stringify!($visit)),*];

        /// Indexed alongside [`VISITORS`]
        static ARITIES: &[Option<Arity>] = &[$(define_operator_table!(@arity $($arity)*)),*];

        impl<'a> Immediates<'a> {
            fn table() -> ImmediateTable {
                vec![$(vec![$($((stringify!($arg), <$argty as Payload>::INPUTS)),*)?]),*]
            }
        }

        fn visitor_name(op: &Operator) -> Option<&'static str> {
            match op {
                $(Operator::$op { .. } => Some(stringify!($visit)),)*
                _ => None,
            }
        }

        pub fn mnemonic(op: &Operator) -> Option<String> {
            Some(wat_name(visitor_name(op)?))
        }

        fn encoded_operands(op: &Operator) -> Vec<Operand> {
            match op {
                $(
                    Operator::$op $({ $($arg),* })? => vec![
                        $($(Payload::operand($arg, stringify!($arg))),*)?
                    ],
                )*
                _ => Vec::new(),
            }
        }

        pub fn memarg(op: &Operator) -> Option<MemArg> {
            match op {
                $(
                    Operator::$op $({ $($arg),* })? => {
                        None $($(.or_else(|| Payload::memarg($arg)))*)?
                    }
                )*
                _ => None,
            }
        }

        pub fn immediates(op: &Operator) -> Vec<u64> {
            match op {
                $(
                    Operator::$op $({ $($arg),* })? => {
                        let values: Vec<Vec<u64>> = vec![$($(Payload::values($arg)),*)?];
                        values.concat()
                    }
                )*
                _ => Vec::new(),
            }
        }

        /// Every operand an operator carries, in the order the text format writes them
        pub fn operands(op: &Operator) -> Vec<Operand> {
            let mut operands = encoded_operands(op);
            text_format_order(&mut operands);
            reference_type_spelling(op, &mut operands);
            operands
        }

        /// `None` for the operators whose effect depends on a type declared elsewhere
        pub fn arity(op: &Operator) -> Option<Arity> {
            match op {
                $(Operator::$op { .. } => define_operator_table!(@arity $($arity)*),)*
                _ => None,
            }
        }
    };
    (@arity custom) => { None };
    (@arity $pops:literal -> $pushes:literal) => { Some(Arity { pops: $pops, pushes: $pushes }) };
}

wasmparser::for_each_operator!(define_operator_table);

/// A hash of the operator's name rather than its position in `wasmparser`'s list, which moves when
/// a release adds operators, so an intrinsic id saved in a database keeps meaning the same thing
pub fn operator_id(op: &Operator) -> Option<u32> {
    visitor_name(op).map(stable_id)
}

pub fn operator_ids() -> impl Iterator<Item = u32> {
    VISITORS.iter().map(|name| stable_id(name))
}

pub const fn stable_id(name: &str) -> u32 {
    let bytes = name.as_bytes();
    let mut hash: u32 = 0x811c_9dc5;
    let mut at = 0;
    while at < bytes.len() {
        hash = (hash ^ bytes[at] as u32).wrapping_mul(0x0100_0193);
        at += 1;
    }
    hash
}

fn by_id() -> &'static HashMap<u32, usize> {
    static BY_ID: OnceLock<HashMap<u32, usize>> = OnceLock::new();
    BY_ID.get_or_init(|| {
        VISITORS
            .iter()
            .enumerate()
            .map(|(index, name)| (stable_id(name), index))
            .collect()
    })
}

/// The text format writes the space first, the encoding what it selects. Stable, so `memory.copy`
/// keeps its destination before its source
fn text_format_order(operands: &mut [Operand]) {
    let names_a_space = |operand: &Operand| match operand {
        Operand::Index { field, .. } => !matches!(
            *field,
            "data_index" | "elem_index" | "type_index" | "segment" | "seg"
        ),
        _ => true,
    };

    let mut ordered: Vec<Operand> = operands
        .iter()
        .filter(|o| names_a_space(o))
        .cloned()
        .collect();
    ordered.extend(operands.iter().filter(|o| !names_a_space(o)).cloned());
    operands.clone_from_slice(&ordered);
}

/// Nullability is in the opcode for these, and the text format writes it on the operand
fn reference_type_spelling(op: &Operator, operands: &mut [Operand]) {
    let nullable = match op {
        Operator::RefTestNullable { .. }
        | Operator::RefCastNullable { .. }
        | Operator::RefCastDescEqNullable { .. } => "null ",
        Operator::RefTestNonNull { .. }
        | Operator::RefCastNonNull { .. }
        | Operator::RefCastDescEqNonNull { .. } => "",
        _ => return,
    };
    if let Some(Operand::Type(ty)) = operands.first_mut() {
        *ty = format!("(ref {nullable}{ty})");
    }
}

pub fn operator_name(id: u32) -> Option<String> {
    by_id().get(&id).map(|index| wat_name(VISITORS[*index]))
}

pub fn operator_arity(id: u32) -> Option<Arity> {
    ARITIES.get(*by_id().get(&id)?).copied().flatten()
}

pub fn operator_immediates(id: u32) -> Vec<String> {
    static IMMEDIATES: OnceLock<ImmediateTable> = OnceLock::new();
    let Some(index) = by_id().get(&id) else {
        return Vec::new();
    };
    IMMEDIATES.get_or_init(Immediates::table)[*index]
        .iter()
        .flat_map(|(name, inputs)| inputs.iter().map(move |suffix| format!("{name}{suffix}")))
        .collect()
}

/// The two differ only in where the namespace separators fall, so `visit_i32_atomic_rmw8_add` is
/// `i32.atomic.rmw8.add`: a leading token from [`NAMESPACES`], an optional `atomic`, an optional
/// read-modify-write width
pub fn wat_name(visit: &str) -> String {
    let name = visit.strip_prefix("visit_").unwrap_or(visit);
    if let Some(exception) = spelled_differently(name) {
        return exception.to_owned();
    }

    let tokens: Vec<&str> = name.split('_').collect();
    let mut segments = 0;
    if tokens.len() > 1 && NAMESPACES.contains(&tokens[0]) {
        segments = 1;
        if tokens.len() > segments + 1 && tokens[segments] == "atomic" {
            segments += 1;
            if tokens.len() > segments + 1 && RMW_WIDTHS.contains(&tokens[segments]) {
                segments += 1;
            }
        }
    }

    if segments == 0 {
        return name.to_owned();
    }
    let mut out = tokens[..segments].join(".");
    out.push('.');
    out.push_str(&tokens[segments..].join("_"));
    out
}

/// Rust spells a NaN `NaN`, which the format has no token for, and prints every payload the same
pub fn wat_f32(value: f32) -> String {
    let bits = value.to_bits();
    match nan_payload(value.is_nan(), u64::from(bits & 0x007f_ffff), 0x0040_0000) {
        Some(spelling) => format!("{}{spelling}", sign_of(bits >> 31 != 0)),
        None => value.to_string(),
    }
}

pub fn wat_f64(value: f64) -> String {
    let bits = value.to_bits();
    match nan_payload(
        value.is_nan(),
        bits & 0x000f_ffff_ffff_ffff,
        0x0008_0000_0000_0000,
    ) {
        Some(spelling) => format!("{}{spelling}", sign_of(bits >> 63 != 0)),
        None => value.to_string(),
    }
}

/// `nan` on its own is the canonical one, and any other needs its payload written out
fn nan_payload(is_nan: bool, payload: u64, canonical: u64) -> Option<String> {
    if !is_nan {
        return None;
    }
    Some(if payload == canonical {
        "nan".to_owned()
    } else {
        format!("nan:{payload:#x}")
    })
}

fn sign_of(negative: bool) -> &'static str {
    if negative { "-" } else { "" }
}

/// Operators whose visitor name carries more than the text format spells out
fn spelled_differently(name: &str) -> Option<&'static str> {
    Some(match name {
        "ref_test_non_null" | "ref_test_nullable" => "ref.test",
        "ref_cast_non_null" | "ref_cast_nullable" => "ref.cast",
        "ref_cast_desc_eq_non_null" | "ref_cast_desc_eq_nullable" => "ref.cast_desc_eq",
        "typed_select" | "typed_select_multi" => "select",
        _ => return None,
    })
}

const NAMESPACES: &[&str] = &[
    "any", "array", "atomic", "cont", "data", "elem", "extern", "f32", "f32x4", "f64", "f64x2",
    "global", "i16x8", "i31", "i32", "i32x4", "i64", "i64x2", "i8x16", "local", "memory", "ref",
    "struct", "table", "v128",
];

const RMW_WIDTHS: &[&str] = &["rmw", "rmw8", "rmw16", "rmw32"];

trait Payload {
    const INPUTS: &'static [&'static str] = &[];

    fn operand(&self, field: &'static str) -> Operand;

    fn values(&self) -> Vec<u64> {
        Vec::new()
    }

    fn memarg(&self) -> Option<MemArg> {
        None
    }
}

fn halves(value: u128) -> Vec<u64> {
    vec![value as u64, (value >> 64) as u64]
}

impl Payload for u32 {
    const INPUTS: &'static [&'static str] = &[""];

    fn operand(&self, field: &'static str) -> Operand {
        Operand::Index {
            field,
            value: *self,
        }
    }

    fn values(&self) -> Vec<u64> {
        vec![u64::from(*self)]
    }
}

impl Payload for u8 {
    const INPUTS: &'static [&'static str] = &[""];

    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Lane(*self)
    }

    fn values(&self) -> Vec<u64> {
        vec![u64::from(*self)]
    }
}

impl Payload for i32 {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::I32(*self)
    }
}

impl Payload for i64 {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::I64(*self)
    }
}

impl Payload for Ieee32 {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::F32(f32::from_bits(self.bits()))
    }
}

impl Payload for Ieee64 {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::F64(f64::from_bits(self.bits()))
    }
}

impl Payload for V128 {
    const INPUTS: &'static [&'static str] = &["_low", "_high"];

    fn operand(&self, _field: &'static str) -> Operand {
        Operand::V128(u128::from_le_bytes(*self.bytes()))
    }

    fn values(&self) -> Vec<u64> {
        halves(u128::from_le_bytes(*self.bytes()))
    }
}

impl Payload for [u8; 16] {
    const INPUTS: &'static [&'static str] = &["_low", "_high"];

    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Lanes(*self)
    }

    fn values(&self) -> Vec<u64> {
        halves(u128::from_le_bytes(*self))
    }
}

impl Payload for MemArg {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::MemArg {
            align: self.align,
            offset: self.offset,
            memory: self.memory,
        }
    }

    fn memarg(&self) -> Option<MemArg> {
        Some(*self)
    }
}

impl Payload for BlockType {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::BlockType(block_type_text(self))
    }
}

fn block_type_text(ty: &BlockType) -> Option<String> {
    match ty {
        BlockType::Empty => None,
        BlockType::Type(ty) => Some(format!("(result {})", val_type_text(ty))),
        BlockType::FuncType(index) => Some(format!("(type {index})")),
    }
}

pub fn val_type_text(ty: &ValType) -> String {
    match ty {
        ValType::Ref(ty) => ref_type_text(ty),
        ty => ty.to_string(),
    }
}

fn ref_type_text(ty: &RefType) -> String {
    let heap = ty.heap_type();
    if ty.is_nullable()
        && let HeapType::Abstract { shared: false, ty } = heap
    {
        return format!("{}ref", nullable_abstract_name(ty));
    }
    let null = if ty.is_nullable() { "null " } else { "" };
    format!("(ref {null}{})", heap_type_text(&heap))
}

fn nullable_abstract_name(ty: AbstractHeapType) -> &'static str {
    match ty {
        AbstractHeapType::None => "null",
        AbstractHeapType::NoExtern => "nullextern",
        AbstractHeapType::NoFunc => "nullfunc",
        AbstractHeapType::NoExn => "nullexn",
        AbstractHeapType::NoCont => "nullcont",
        ty => abstract_heap_type_name(ty),
    }
}

fn heap_type_text(ty: &HeapType) -> String {
    match ty {
        HeapType::Abstract { shared: false, ty } => abstract_heap_type_name(*ty).to_owned(),
        HeapType::Abstract { shared: true, ty } => {
            format!("(shared {})", abstract_heap_type_name(*ty))
        }
        HeapType::Concrete(index) => type_index_name(index),
        HeapType::Exact(index) => format!("(exact {})", type_index_name(index)),
    }
}

impl Payload for BrTable<'_> {
    fn operand(&self, _field: &'static str) -> Operand {
        let mut labels: Vec<u32> = self.targets().filter_map(Result::ok).collect();
        labels.push(self.default());
        Operand::Labels(labels)
    }
}

impl Payload for ValType {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Type(val_type_text(self))
    }
}

impl Payload for Vec<ValType> {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Types(self.iter().map(val_type_text).collect())
    }
}

impl Payload for RefType {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Type(ref_type_text(self))
    }
}

const CONCRETE_HEAP_TYPE: u64 = 0;
const EXACT_HEAP_TYPE: u64 = 1;
const ABSTRACT_HEAP_TYPE: u64 = 2;
const SHARED_ABSTRACT_HEAP_TYPE: u64 = 3;

impl Payload for HeapType {
    const INPUTS: &'static [&'static str] = &["", "_kind"];

    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Type(heap_type_text(self))
    }

    fn values(&self) -> Vec<u64> {
        match self {
            HeapType::Concrete(index) => vec![type_index_value(index), CONCRETE_HEAP_TYPE],
            HeapType::Exact(index) => vec![type_index_value(index), EXACT_HEAP_TYPE],
            HeapType::Abstract { shared, ty } => vec![
                u64::from(abstract_heap_type_code(*ty)),
                if *shared {
                    SHARED_ABSTRACT_HEAP_TYPE
                } else {
                    ABSTRACT_HEAP_TYPE
                },
            ],
        }
    }
}

fn abstract_heap_type_code(ty: AbstractHeapType) -> u8 {
    match ty {
        AbstractHeapType::Func => 0x70,
        AbstractHeapType::Extern => 0x6f,
        AbstractHeapType::Any => 0x6e,
        AbstractHeapType::Eq => 0x6d,
        AbstractHeapType::I31 => 0x6c,
        AbstractHeapType::Struct => 0x6b,
        AbstractHeapType::Array => 0x6a,
        AbstractHeapType::Exn => 0x69,
        AbstractHeapType::Cont => 0x68,
        AbstractHeapType::None => 0x71,
        AbstractHeapType::NoExtern => 0x72,
        AbstractHeapType::NoFunc => 0x73,
        AbstractHeapType::NoExn => 0x74,
        AbstractHeapType::NoCont => 0x75,
    }
}

fn type_index_value(index: &UnpackedIndex) -> u64 {
    index
        .as_module_index()
        .or_else(|| index.as_rec_group_index())
        .map_or(u64::MAX, u64::from)
}

fn abstract_heap_type_name(ty: AbstractHeapType) -> &'static str {
    match ty {
        AbstractHeapType::Func => "func",
        AbstractHeapType::Extern => "extern",
        AbstractHeapType::Any => "any",
        AbstractHeapType::None => "none",
        AbstractHeapType::NoExtern => "noextern",
        AbstractHeapType::NoFunc => "nofunc",
        AbstractHeapType::Eq => "eq",
        AbstractHeapType::Struct => "struct",
        AbstractHeapType::Array => "array",
        AbstractHeapType::I31 => "i31",
        AbstractHeapType::Exn => "exn",
        AbstractHeapType::NoExn => "noexn",
        AbstractHeapType::Cont => "cont",
        AbstractHeapType::NoCont => "nocont",
    }
}

fn type_index_name(index: &UnpackedIndex) -> String {
    match (index.as_module_index(), index.as_rec_group_index()) {
        (Some(index), _) => index.to_string(),
        (_, Some(index)) => format!("rec={index}"),
        // `wasmparser` grows a third variant when its validator is compiled in, as the
        // conformance harness does
        (None, None) => "type=?".to_owned(),
    }
}

impl Payload for Ordering {
    const INPUTS: &'static [&'static str] = &[""];

    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Ordering(match self {
            Ordering::SeqCst => "seq_cst",
            Ordering::AcqRel => "acq_rel",
        })
    }

    fn values(&self) -> Vec<u64> {
        vec![match self {
            Ordering::SeqCst => 0,
            Ordering::AcqRel => 1,
        }]
    }
}

impl Payload for TryTable {
    fn operand(&self, _field: &'static str) -> Operand {
        let clauses = self.catches.iter().map(|catch| match catch {
            Catch::One { tag, label } => format!("(catch {tag} {label})"),
            Catch::OneRef { tag, label } => format!("(catch_ref {tag} {label})"),
            Catch::All { label } => format!("(catch_all {label})"),
            Catch::AllRef { label } => format!("(catch_all_ref {label})"),
        });
        let parts: Vec<String> = block_type_text(&self.ty)
            .into_iter()
            .chain(clauses)
            .collect();
        Operand::Table(parts.join(" "))
    }
}

impl Payload for ResumeTable {
    fn operand(&self, _field: &'static str) -> Operand {
        let handlers: Vec<String> = self
            .handlers
            .iter()
            .map(|handle| match handle {
                Handle::OnLabel { tag, label } => format!("(on {tag} {label})"),
                Handle::OnSwitch { tag } => format!("(on {tag} switch)"),
            })
            .collect();
        Operand::Table(handlers.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(data: &[u8]) -> Instruction<'_> {
        decode(data).expect("decodes")
    }

    fn name(data: &[u8]) -> String {
        decoded(data).mnemonic()
    }

    #[test]
    fn mvp_mnemonics() {
        assert_eq!(name(&[0x01]), "nop");
        assert_eq!(name(&[0x00]), "unreachable");
        assert_eq!(name(&[0x0b]), "end");
        assert_eq!(name(&[0x6a]), "i32.add");
        assert_eq!(name(&[0x20, 0x00]), "local.get");
        assert_eq!(name(&[0x24, 0x00]), "global.set");
        assert_eq!(name(&[0x2c, 0x00, 0x00]), "i32.load8_s");
        assert_eq!(name(&[0x3f, 0x00]), "memory.size");
        assert_eq!(name(&[0xa7]), "i32.wrap_i64");
        assert_eq!(name(&[0xb2]), "f32.convert_i32_s");
        assert_eq!(name(&[0xbe]), "f32.reinterpret_i32");
        assert_eq!(name(&[0xc0]), "i32.extend8_s");
    }

    #[test]
    fn mnemonics_without_a_namespace() {
        assert_eq!(name(&[0x0c, 0x00]), "br");
        assert_eq!(name(&[0x0d, 0x00]), "br_if");
        assert_eq!(name(&[0x0e, 0x00, 0x00]), "br_table");
        assert_eq!(name(&[0x11, 0x00, 0x00]), "call_indirect");
        assert_eq!(name(&[0x1a]), "drop");
        assert_eq!(name(&[0x1b]), "select");
    }

    #[test]
    fn post_mvp_mnemonics() {
        assert_eq!(name(&[0xfc, 0x00]), "i32.trunc_sat_f32_s");
        assert_eq!(name(&[0xfc, 0x0a, 0x00, 0x00]), "memory.copy");
        assert_eq!(name(&[0xfc, 0x0e, 0x00, 0x00]), "table.copy");
        assert_eq!(name(&[0xd0, 0x70]), "ref.null");
        assert_eq!(name(&[0xd1]), "ref.is_null");
        assert_eq!(name(&[0xfd, 0x0b, 0x00, 0x00]), "v128.store");
        assert_eq!(name(&[0xfd, 0x6e]), "i8x16.add");
        assert_eq!(name(&[0xfd, 0x15, 0x00]), "i8x16.extract_lane_s");
        assert_eq!(name(&[0xfe, 0x00, 0x00, 0x00]), "memory.atomic.notify");
        assert_eq!(name(&[0xfe, 0x1e, 0x00, 0x00]), "i32.atomic.rmw.add");
        assert_eq!(name(&[0xfe, 0x20, 0x00, 0x00]), "i32.atomic.rmw8.add_u");
        assert_eq!(name(&[0xfe, 0x22, 0x00, 0x00]), "i64.atomic.rmw8.add_u");
        assert_eq!(name(&[0xfe, 0x03, 0x00]), "atomic.fence");
    }

    #[test]
    fn mnemonics_that_collapse_variants() {
        assert_eq!(name(&[0x1c, 0x01, 0x7f]), "select");
        assert_eq!(name(&[0xfb, 0x14, 0x6e]), "ref.test");
        assert_eq!(name(&[0xfb, 0x15, 0x6e]), "ref.test");
        assert_eq!(name(&[0xfb, 0x16, 0x6e]), "ref.cast");
        assert_eq!(name(&[0xfb, 0x17, 0x6e]), "ref.cast");
    }

    #[test]
    fn a_float_is_spelled_the_way_the_text_format_spells_it() {
        assert_eq!(wat_f32(1.5), "1.5");
        assert_eq!(wat_f32(-0.0), "-0");
        assert_eq!(wat_f32(f32::INFINITY), "inf");
        assert_eq!(wat_f32(f32::NEG_INFINITY), "-inf");
        assert_eq!(wat_f32(f32::NAN), "nan");
        assert_eq!(wat_f32(f32::from_bits(0x7fc0_0000)), "nan");
        assert_eq!(wat_f32(f32::from_bits(0xffc0_0000)), "-nan");
        assert_eq!(wat_f32(f32::from_bits(0x7f80_0001)), "nan:0x1");
        assert_eq!(wat_f32(f32::from_bits(0x7fa0_0000)), "nan:0x200000");

        assert_eq!(wat_f64(1.5), "1.5");
        assert_eq!(wat_f64(f64::INFINITY), "inf");
        assert_eq!(wat_f64(f64::NAN), "nan");
        assert_eq!(wat_f64(f64::from_bits(0xfff8_0000_0000_0000)), "-nan");
        assert_eq!(wat_f64(f64::from_bits(0x7ff0_0000_0000_0001)), "nan:0x1");
    }

    #[test]
    fn lengths_cover_the_immediates() {
        assert_eq!(decoded(&[0x6a]).len, 1);
        assert_eq!(decoded(&[0x20, 0x83, 0x01]).len, 3);
        assert_eq!(decoded(&[0x43, 0x00, 0x00, 0x80, 0x3f]).len, 5);
        assert_eq!(decoded(&[0x44; 9]).len, 9);
        assert_eq!(decoded(&[0x0e, 0x02, 0x00, 0x01, 0x03]).len, 5);
    }

    #[test]
    fn index_operands_carry_their_field() {
        assert_eq!(
            decoded(&[0x20, 0x83, 0x01]).operands(),
            [Operand::Index {
                field: "local_index",
                value: 131
            }]
        );
        assert_eq!(
            decoded(&[0x10, 0x07]).operands(),
            [Operand::Index {
                field: "function_index",
                value: 7
            }]
        );
        // The bytes carry the type first, the text format the table
        assert_eq!(
            decoded(&[0x11, 0x02, 0x01]).operands(),
            [
                Operand::Index {
                    field: "table_index",
                    value: 1
                },
                Operand::Index {
                    field: "type_index",
                    value: 2
                }
            ]
        );
    }

    /// The orders `wasmprinter` prints for these bytes
    #[test]
    fn two_index_operators_are_written_the_way_the_text_format_writes_them() {
        let fields = |bytes: &[u8]| -> Vec<(&'static str, u32)> {
            decoded(bytes)
                .operands()
                .into_iter()
                .filter_map(|operand| match operand {
                    Operand::Index { field, value } => Some((field, value)),
                    _ => None,
                })
                .collect()
        };

        assert_eq!(
            fields(&[0xfc, 0x08, 0x00, 0x01]),
            [("mem", 1), ("data_index", 0)]
        );
        assert_eq!(
            fields(&[0xfc, 0x0c, 0x00, 0x01]),
            [("table", 1), ("elem_index", 0)]
        );
        assert_eq!(
            fields(&[0xfc, 0x0a, 0x01, 0x00]),
            [("dst_mem", 1), ("src_mem", 0)]
        );
    }

    #[test]
    fn constant_operands() {
        assert_eq!(decoded(&[0x41, 0x7f]).operands(), [Operand::I32(-1)]);
        assert_eq!(decoded(&[0x42, 0x80, 0x01]).operands(), [Operand::I64(128)]);
        assert_eq!(
            decoded(&[0x43, 0x00, 0x00, 0x80, 0x3f]).operands(),
            [Operand::F32(1.0)]
        );
        assert_eq!(
            decoded(&[0x44, 0, 0, 0, 0, 0, 0, 0xf0, 0x3f]).operands(),
            [Operand::F64(1.0)]
        );
    }

    #[test]
    fn memarg_alignment_stays_the_exponent_it_was_encoded_as() {
        assert_eq!(
            decoded(&[0x28, 0x02, 0x10]).operands(),
            [Operand::MemArg {
                align: 2,
                offset: 16,
                memory: 0
            }]
        );

        assert_eq!(
            decoded(&[0x3a, 0x60, 0x01, 0x7f]).operands(),
            [Operand::MemArg {
                align: 32,
                offset: 127,
                memory: 1
            }]
        );
    }

    #[test]
    fn simd_operands() {
        let mut data = vec![0xfd, 0x0c];
        data.extend(1..=16u8);
        assert_eq!(
            decode(&data).unwrap().operands(),
            [Operand::V128(u128::from_le_bytes([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ]))]
        );
        assert_eq!(decoded(&[0xfd, 0x15, 0x03]).operands(), [Operand::Lane(3)]);
    }

    #[test]
    fn block_type_operands() {
        assert_eq!(
            decoded(&[0x02, 0x40]).operands(),
            [Operand::BlockType(None)]
        );
        assert_eq!(
            decoded(&[0x03, 0x7f]).operands(),
            [Operand::BlockType(Some("(result i32)".to_owned()))]
        );
        assert_eq!(
            decoded(&[0x04, 0x07]).operands(),
            [Operand::BlockType(Some("(type 7)".to_owned()))]
        );
    }

    #[test]
    fn br_table_lists_targets_then_default() {
        let insn = decoded(&[0x0e, 0x02, 0x00, 0x01, 0x03]);
        assert_eq!(insn.operands(), [Operand::Labels(vec![0, 1, 3])]);
        assert_eq!(insn.flow(), Flow::IndirectBranch);
    }

    #[test]
    fn types_and_handlers_render_as_the_text_format_writes_them() {
        assert_eq!(
            decoded(&[0x1f, 0x7f, 0x02, 0x00, 0x00, 0x01, 0x02, 0x02]).operands(),
            [Operand::Table(
                "(result i32) (catch 0 1) (catch_all 2)".to_owned()
            )]
        );
        assert_eq!(
            decoded(&[0x1c, 0x01, 0x70]).operands(),
            [Operand::Type("funcref".to_owned())]
        );
        assert_eq!(
            decoded(&[0x02, 0x63, 0x03]).operands(),
            [Operand::BlockType(Some("(result (ref null 3))".to_owned()))]
        );
        assert_eq!(
            decoded(&[0xfb, 0x24, 0x00]).operands(),
            [Operand::Type("(ref null 0)".to_owned())]
        );
        assert_eq!(
            decoded(&[0xfb, 0x23, 0x00]).operands(),
            [Operand::Type("(ref 0)".to_owned())]
        );
    }

    #[test]
    fn control_flow_classification() {
        assert_eq!(decoded(&[0x00]).flow(), Flow::Trap);
        assert_eq!(decoded(&[0x02, 0x40]).flow(), Flow::BlockStart);
        assert_eq!(decoded(&[0x05]).flow(), Flow::Arm);
        assert_eq!(decoded(&[0x0b]).flow(), Flow::BlockEnd);
        assert_eq!(decoded(&[0x0c, 0x00]).flow(), Flow::Branch);
        assert_eq!(decoded(&[0x0d, 0x00]).flow(), Flow::ConditionalBranch);
        assert_eq!(decoded(&[0x0f]).flow(), Flow::Return);
        assert_eq!(decoded(&[0x10, 0x00]).flow(), Flow::Call);
        assert_eq!(decoded(&[0x11, 0x00, 0x00]).flow(), Flow::IndirectCall);
        assert_eq!(decoded(&[0x12, 0x00]).flow(), Flow::TailCall);
        assert_eq!(decoded(&[0x6a]).flow(), Flow::Normal);
    }

    #[test]
    fn only_terminators_stop_the_fallthrough() {
        assert!(Flow::Normal.falls_through());
        assert!(Flow::Call.falls_through());
        assert!(Flow::ConditionalBranch.falls_through());
        assert!(!Flow::Return.falls_through());
        assert!(!Flow::Trap.falls_through());
        assert!(!Flow::Branch.falls_through());
        assert!(!Flow::TailCall.falls_through());
        assert!(
            !Flow::Arm.falls_through(),
            "an arm is jumped over, not run into"
        );
    }

    #[test]
    fn no_throw_falls_through() {
        assert_eq!(decoded(&[0x08, 0x00]).flow(), Flow::Trap, "throw");
        assert_eq!(decoded(&[0x09, 0x00]).flow(), Flow::Trap, "rethrow");
        assert_eq!(decoded(&[0x0a]).flow(), Flow::Trap, "throw_ref");
    }

    #[test]
    fn arities_come_from_the_operator_table() {
        assert_eq!(decoded(&[0x6a]).arity(), Some(Arity { pops: 2, pushes: 1 }));
        assert_eq!(decoded(&[0x45]).arity(), Some(Arity { pops: 1, pushes: 1 }));
        assert_eq!(
            decoded(&[0x41, 0x00]).arity(),
            Some(Arity { pops: 0, pushes: 1 })
        );
        assert_eq!(decoded(&[0x1a]).arity(), Some(Arity { pops: 1, pushes: 0 }));
        assert_eq!(
            decoded(&[0x36, 0x02, 0x00]).arity(),
            Some(Arity { pops: 2, pushes: 0 })
        );
        assert_eq!(decoded(&[0x10, 0x00]).arity(), None, "call needs its type");
        assert_eq!(decoded(&[0x02, 0x40]).arity(), None, "block needs its type");
    }

    #[test]
    fn rejects_invalid() {
        assert!(decode(&[]).is_none());
        assert!(decode(&[0x27]).is_none(), "unassigned opcode");
        assert!(decode(&[0xff]).is_none(), "unassigned opcode");
        assert!(decode(&[0x20]).is_none(), "truncated immediate");
        assert!(decode(&[0x28, 0x02]).is_none(), "truncated memarg");
    }

    #[test]
    fn an_intrinsic_takes_every_immediate_it_declares() {
        let taken = |bytes: &[u8]| {
            let insn = decode(bytes).expect("decodes");
            let id = insn.operator_id().expect("an operator");
            let values = immediates(&insn.op);
            assert_eq!(values.len(), operator_immediates(id).len(), "{bytes:02x?}");
            (operator_immediates(id), values)
        };
        let sixteen: Vec<u8> = (1..=16).collect();
        let halves = vec![0x0807_0605_0403_0201, 0x100f_0e0d_0c0b_0a09];

        let v128_const = [&[0xfd, 0x0c][..], &sixteen].concat();
        assert_eq!(
            taken(&v128_const),
            (
                vec!["value_low".into(), "value_high".into()],
                halves.clone()
            )
        );
        let shuffle = [&[0xfd, 0x0d][..], &sixteen].concat();
        assert_eq!(
            taken(&shuffle),
            (vec!["lanes_low".into(), "lanes_high".into()], halves)
        );
        assert_eq!(
            taken(&[0xfd, 0x1b, 0x03]),
            (vec!["lane".into()], vec![3]),
            "i32x4.extract_lane"
        );
        assert_eq!(
            taken(&[0xfd, 0x54, 0x00, 0x08, 0x05]),
            (vec!["lane".into()], vec![5]),
            "v128.load8_lane, its offset folded into the address"
        );
        assert_eq!(
            taken(&[0xfc, 0x0b, 0x00]),
            (vec!["mem".into()], vec![0]),
            "memory.fill"
        );
        assert_eq!(
            taken(&[0xfe, 0x1e, 0x02, 0x10]),
            (Vec::<String>::new(), Vec::new()),
            "i32.atomic.rmw.add"
        );
        let heap_type = vec!["hty".into(), "hty_kind".into()];
        assert_eq!(
            taken(&[0xfb, 0x14, 0x03]),
            (heap_type.clone(), vec![3, CONCRETE_HEAP_TYPE]),
            "ref.test (ref 3)"
        );
        assert_eq!(
            taken(&[0xfb, 0x15, 0x70]),
            (heap_type, vec![0x70, ABSTRACT_HEAP_TYPE]),
            "ref.test funcref"
        );
        assert_eq!(
            taken(&[0xfe, 0x4f, 0x01, 0x02]).1,
            vec![1, 2],
            "global.atomic.get acq_rel 2"
        );
    }

    #[test]
    fn rejects_oversized_br_table() {
        let entries = MAX_INSTR_LEN + 1;
        let mut data = vec![0x0e];
        let mut count = entries as u32;
        while count >= 0x80 {
            data.push((count as u8 & 0x7f) | 0x80);
            count >>= 7;
        }
        data.push(count as u8);
        data.extend(std::iter::repeat_n(0x00, entries + 1));

        assert!(decode(&data).is_none());
        assert!(
            decode(&data[..3]).is_none(),
            "and not by running out of input"
        );
    }

    #[test]
    fn accepts_a_br_table_that_fits() {
        let entries = 200;
        let mut data = vec![0x0e, 0xc8, 0x01];
        data.extend(std::iter::repeat_n(0x00, entries + 1));

        let insn = decode(&data).expect("200 entries fit");
        assert_eq!(insn.len, data.len());
    }
}
