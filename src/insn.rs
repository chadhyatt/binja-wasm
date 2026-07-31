//! Single-instruction decoding on top of `wasmparser`

use std::collections::HashMap;
use std::sync::OnceLock;

use wasmparser::{
    AbstractHeapType, BinaryReader, BlockType, BrTable, FrameKind, FrameStack, HeapType, Ieee32,
    Ieee64, MemArg, Operator, Ordering, RefType, ResumeTable, TryTable, UnpackedIndex, ValType,
    VisitOperator, VisitSimdOperator, V128,
};

/// The longest instruction the core will take, so a `br_table` past around 250 entries cannot be
/// one instruction to any plugin and decodes as invalid rather than truncating
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

/// Uncapped, unlike [`decode`]: analysis needs only the operator and its length, and giving up at
/// an oversized `br_table` would cost the rest of the body its control flow
pub fn decode_any(data: &[u8]) -> Option<Instruction<'_>> {
    // Some operators are legal only inside a particular block, and disassembly starts at an
    // arbitrary address; between these two frames every operator decodes
    for frame in [FrameKind::If, FrameKind::LegacyTry] {
        let mut reader = BinaryReader::new(data, 0);
        if let Ok(op) = reader.visit_operator(&mut OperatorFactory { frame }) {
            return Some(Instruction {
                op,
                len: reader.original_position(),
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
            Flow::Trap | Flow::Return | Flow::Branch | Flow::IndirectBranch | Flow::TailCall
        )
    }
}

pub fn flow(op: &Operator) -> Flow {
    match op {
        Operator::Unreachable | Operator::Throw { .. } | Operator::ThrowRef => Flow::Trap,
        Operator::Block { .. }
        | Operator::Loop { .. }
        | Operator::If { .. }
        | Operator::Try { .. }
        | Operator::TryTable { .. } => Flow::BlockStart,
        Operator::End | Operator::Delegate { .. } => Flow::BlockEnd,
        Operator::Else | Operator::Catch { .. } | Operator::CatchAll => Flow::Arm,
        Operator::Br { .. } | Operator::Rethrow { .. } => Flow::Branch,
        Operator::BrIf { .. }
        | Operator::BrOnNull { .. }
        | Operator::BrOnNonNull { .. }
        | Operator::BrOnCast { .. }
        | Operator::BrOnCastFail { .. }
        | Operator::BrOnCastDescEq { .. }
        | Operator::BrOnCastDescEqFail { .. } => Flow::ConditionalBranch,
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

/// Generated from the operator list `wasmparser` exposes, so new proposals arrive with the crate
macro_rules! define_operator_table {
    ($( @$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident (arity $($arity:tt)*) )*) => {
        /// A position in this list is an operator's stable identity
        static VISITORS: &[&str] = &[$(stringify!($visit)),*];

        /// Indexed alongside [`VISITORS`]
        static ARITIES: &[Option<Arity>] = &[$(define_operator_table!(@arity $($arity)*)),*];

        fn visitor_name(op: &Operator) -> Option<&'static str> {
            match op {
                $(Operator::$op { .. } => Some(stringify!($visit)),)*
                _ => None,
            }
        }

        pub fn mnemonic(op: &Operator) -> Option<String> {
            Some(wat_name(visitor_name(op)?))
        }

        /// Every operand an operator carries, in encoding order
        pub fn operands(op: &Operator) -> Vec<Operand> {
            match op {
                $(
                    Operator::$op $({ $($arg),* })? => vec![
                        $($(Payload::operand($arg, stringify!($arg))),*)?
                    ],
                )*
                _ => Vec::new(),
            }
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

pub fn operator_count() -> usize {
    VISITORS.len()
}

/// The operator's position in `wasmparser`'s own list, so an intrinsic id saved in a database
/// still means the same thing when the database is reopened
pub fn operator_id(op: &Operator) -> Option<u32> {
    static BY_NAME: OnceLock<HashMap<&'static str, u32>> = OnceLock::new();

    let by_name = BY_NAME.get_or_init(|| {
        VISITORS
            .iter()
            .enumerate()
            .map(|(index, name)| (*name, index as u32))
            .collect()
    });

    by_name.get(visitor_name(op)?).copied()
}

pub fn operator_name(id: u32) -> Option<String> {
    VISITORS.get(id as usize).map(|visit| wat_name(visit))
}

pub fn operator_arity(id: u32) -> Option<Arity> {
    ARITIES.get(id as usize).copied().flatten()
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
    fn operand(&self, field: &'static str) -> Operand;
}

impl Payload for u32 {
    fn operand(&self, field: &'static str) -> Operand {
        Operand::Index {
            field,
            value: *self,
        }
    }
}

impl Payload for u8 {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Lane(*self)
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
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::V128(u128::from_le_bytes(*self.bytes()))
    }
}

impl Payload for [u8; 16] {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Lanes(*self)
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
}

impl Payload for BlockType {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::BlockType(match self {
            BlockType::Empty => None,
            BlockType::Type(ty) => Some(ty.to_string()),
            BlockType::FuncType(index) => Some(format!("type={index}")),
        })
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
        Operand::Type(self.to_string())
    }
}

impl Payload for Vec<ValType> {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Types(self.iter().map(ValType::to_string).collect())
    }
}

impl Payload for RefType {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Type(self.to_string())
    }
}

impl Payload for HeapType {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Type(match self {
            HeapType::Abstract { shared, ty } => {
                let name = abstract_heap_type_name(*ty);
                if *shared {
                    format!("shared {name}")
                } else {
                    name.to_owned()
                }
            }
            HeapType::Concrete(index) => type_index_name(index),
            HeapType::Exact(index) => format!("exact {}", type_index_name(index)),
        })
    }
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
    match index {
        UnpackedIndex::Module(index) => format!("type={index}"),
        UnpackedIndex::RecGroup(index) => format!("rec={index}"),
        // `wasmparser` grows a third variant when its validator is compiled in, as the
        // conformance harness does
        #[allow(unreachable_patterns)]
        _ => "type=?".to_owned(),
    }
}

impl Payload for Ordering {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Ordering(match self {
            Ordering::SeqCst => "seq_cst",
            Ordering::AcqRel => "acq_rel",
        })
    }
}

impl Payload for TryTable {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Table(format!("{} catches", self.catches.len()))
    }
}

impl Payload for ResumeTable {
    fn operand(&self, _field: &'static str) -> Operand {
        Operand::Table(format!("{} handlers", self.handlers.len()))
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
        assert_eq!(
            decoded(&[0x11, 0x02, 0x01]).operands(),
            [
                Operand::Index {
                    field: "type_index",
                    value: 2
                },
                Operand::Index {
                    field: "table_index",
                    value: 1
                }
            ]
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
            [Operand::BlockType(Some("i32".to_owned()))]
        );
        assert_eq!(
            decoded(&[0x04, 0x07]).operands(),
            [Operand::BlockType(Some("type=7".to_owned()))]
        );
    }

    #[test]
    fn br_table_lists_targets_then_default() {
        let insn = decoded(&[0x0e, 0x02, 0x00, 0x01, 0x03]);
        assert_eq!(insn.operands(), [Operand::Labels(vec![0, 1, 3])]);
        assert_eq!(insn.flow(), Flow::IndirectBranch);
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
