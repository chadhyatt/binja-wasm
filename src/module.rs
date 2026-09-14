//! Reading the module around a function
//!
//! An architecture plugin is handed bytes and an address, which cannot say what `call 3` calls or
//! how many operands it takes; the type and code sections can, and sit in the same image

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, LazyLock, RwLock};

use wasmparser::{Operator, Parser, Payload, TypeRef};

use crate::ViewId;
use crate::insn::Arity;

const MAX_LOCALS: usize = 1 << 16;

pub const MAX_IMAGE_LEN: usize = 256 << 20;

/// The region is bytes the view builds, so its size is an allocation. Sixty times the largest
/// table seen here
pub const MAX_TABLE_SLOTS: u64 = 1 << 22;

/// Nothing is stored at one, so the addresses only have to be distinct
pub const IMPORT_STRIDE: u64 = 4;

pub const GLOBAL_STRIDE: u64 = 8;

const PAGE: u64 = 64 << 10;

const WASM32_CEILING: u64 = 1 << 32;

const WASM64_CEILING: u64 = 1 << 48;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Shape {
    pub image: u64,
    /// What the module declares, or what its data reaches
    pub memory: u64,
    pub globals: u64,
    pub imports: u64,
    /// Slots of the first table, which is what `call_indirect` selects from
    pub tables: u64,
    pub memory64: bool,
}

/// Linear memory starts at zero, and that is what makes a pointer work: a string at memory offset
/// `0x1e0d` is passed around as the plain number `0x1e0d`, and mapping memory anywhere else costs
/// it every cross reference
///
/// The bases are per file, so a small module is not buried in space reserved for the largest one
/// anyone might open, and a memory64 module has room wasm32 cannot give it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// One past the last mapped byte of linear memory
    pub memory_end: u64,
    pub file_base: u64,
    pub image: u64,
    pub global_base: u64,
    pub globals: u64,
    pub table_base: u64,
    pub tables: u64,
    pub import_base: u64,
    pub imports: u64,
    pub pointer: usize,
}

impl Layout {
    /// Linear memory is whatever fits below `base`, which is what a module read on its own gets
    pub fn at(base: u64, shape: Shape) -> Self {
        let pointer = if shape.memory64 { 8 } else { 4 };
        let global_base = align_up(base.saturating_add(shape.image));
        let table_base = align_up(global_base.saturating_add(shape.globals * GLOBAL_STRIDE));
        let import_base = align_up(table_base.saturating_add(shape.tables * pointer as u64));
        Self {
            memory_end: shape.memory.min(base),
            file_base: base,
            image: shape.image,
            global_base,
            globals: shape.globals,
            table_base,
            tables: shape.tables,
            import_base,
            imports: shape.imports,
            pointer,
        }
    }

    /// Two files still overlap in memory, which is unavoidable since zero is the only address it
    /// can start at, and harmless since memory holds no code
    pub fn allocate(shape: Shape) -> Self {
        let ceiling = if shape.memory64 {
            WASM64_CEILING
        } else {
            WASM32_CEILING
        };
        // Everything above memory, plus a page of slack per alignment step
        let pointer = if shape.memory64 { 8u64 } else { 4 };
        let span = align_up(shape.image)
            + align_up(shape.globals * GLOBAL_STRIDE)
            + align_up(shape.tables * pointer)
            + align_up(shape.imports * IMPORT_STRIDE)
            + PAGE;
        let memory = shape.memory.min(ceiling.saturating_sub(span));
        Self::at(
            reserve(span, align_up(memory), ceiling),
            Shape { memory, ..shape },
        )
    }

    /// Identity below the window, since a memory offset *is* an address; the clamp keeps a segment
    /// reaching past the end of memory out of the file above it
    pub fn memory_address(&self, offset: u64) -> u64 {
        offset.min(self.memory_end)
    }

    pub fn memory_mapped(&self, offset: u64) -> bool {
        offset < self.memory_end
    }

    pub fn file_address(&self, offset: u64) -> u64 {
        self.file_base.saturating_add(offset)
    }

    /// Everything read out of a module is an address, so anything handing the core a backing range
    /// has to come back through here
    pub fn file_offset(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.file_base)
    }

    /// Globals are not addressable from wasm code, so anywhere outside linear memory will do
    pub fn global_address(&self, index: u32) -> u64 {
        self.global_base + u64::from(index) * GLOBAL_STRIDE
    }

    /// An import has no body, so a place of its own is what a call to it points at
    pub fn import_address(&self, index: u32) -> u64 {
        self.import_base + u64::from(index) * IMPORT_STRIDE
    }

    pub fn import_end(&self) -> u64 {
        self.import_base + self.imports * IMPORT_STRIDE
    }

    /// No load reaches a table, but an address per slot lets the view say what each one holds
    pub fn table_address(&self, slot: u64) -> u64 {
        self.table_base + slot * self.pointer as u64
    }

    pub fn table_end(&self) -> u64 {
        self.table_base + self.tables * self.pointer as u64
    }

    pub fn global_end(&self) -> u64 {
        self.global_base + self.globals * GLOBAL_STRIDE
    }

    pub fn is_import_stub(&self, addr: u64) -> bool {
        (self.import_base..self.import_end()).contains(&addr)
    }

    /// One past the last address the file occupies, which the view's length follows
    pub fn end(&self) -> u64 {
        self.import_end()
            .max(self.global_end())
            .max(self.memory_end)
    }

    /// The core rejects overlapping segments without a word, so this has to hold for every layout
    pub fn is_ordered(&self) -> bool {
        self.memory_end <= self.file_base
            && self.file_base + self.image <= self.global_base
            && self.global_end() <= self.table_base
            && self.table_end() <= self.import_base
    }
}

impl Default for Layout {
    fn default() -> Self {
        Self::at(0, Shape::default())
    }
}

fn align_up(value: u64) -> u64 {
    value.saturating_add(PAGE - 1) & !(PAGE - 1)
}

static LAYOUTS: LazyLock<RwLock<BTreeMap<ViewId, Layout>>> = LazyLock::new(Default::default);

pub fn install_layout(view: ViewId, layout: Layout) {
    let mut layouts = match LAYOUTS.write() {
        Ok(layouts) => layouts,
        Err(poisoned) => poisoned.into_inner(),
    };
    layouts.insert(view, layout);
}

pub fn layout(view: ViewId) -> Option<Layout> {
    let layouts = match LAYOUTS.read() {
        Ok(layouts) => layouts,
        Err(poisoned) => poisoned.into_inner(),
    };
    layouts.get(&view).copied()
}

/// All that the callbacks handed an address and nothing else have to tell a stub from real code,
/// and exact rather than a guess, since two files never share a region above memory
pub fn import_stub(addr: u64) -> bool {
    let layouts = match LAYOUTS.read() {
        Ok(layouts) => layouts,
        Err(poisoned) => poisoned.into_inner(),
    };
    layouts.values().any(|layout| layout.is_import_stub(addr))
}

/// The layout of the file covering `addr` above linear memory, for the same callbacks
pub fn layout_covering(addr: u64) -> Option<Layout> {
    let layouts = match LAYOUTS.read() {
        Ok(layouts) => layouts,
        Err(poisoned) => poisoned.into_inner(),
    };
    layouts
        .values()
        .find(|layout| (layout.file_base..layout.end()).contains(&addr))
        .copied()
}

/// Bumped past every file placed so far and never below `floor`, so the first file opened sits as
/// low as its own memory allows and the overview stays scaled to the file
static NEXT: LazyLock<RwLock<u64>> = LazyLock::new(|| RwLock::new(0));

fn reserve(span: u64, floor: u64, ceiling: u64) -> u64 {
    let mut next = match NEXT.write() {
        Ok(next) => next,
        Err(poisoned) => poisoned.into_inner(),
    };

    let mut base = align_up((*next).max(floor));
    if base.saturating_add(span) > ceiling {
        // Enough files in one session to fill the space; starting over costs the older one the
        // answers worked out from an address alone, which beats wrapping the arithmetic
        tracing::warn!(
            "wasm view: the address space is full at {base:#x}, reusing it from {floor:#x}"
        );
        base = align_up(floor);
    }
    *next = base.saturating_add(span);
    base
}

/// Reading a module walks the whole image and every function in it asks the same questions, so the
/// answers are kept; one with no functions is cached too, so a non-module is read only once
static READ: LazyLock<RwLock<Cache>> = LazyLock::new(Default::default);

type Cache = BTreeMap<(ViewId, u64), (u64, Arc<Module>)>;

pub fn install(view: ViewId, base: u64, end: u64, module: Module) -> Arc<Module> {
    let module = Arc::new(module);
    let mut read = match READ.write() {
        Ok(read) => read,
        // This is a cache, so a poisoned lock is better carried on with than propagated
        Err(poisoned) => poisoned.into_inner(),
    };
    read.insert((view, base), (end, Arc::clone(&module)));
    module
}

pub fn lookup(view: ViewId, addr: u64) -> Option<Arc<Module>> {
    let read = match READ.read() {
        Ok(read) => read,
        Err(poisoned) => poisoned.into_inner(),
    };

    let (_, (end, module)) = read.range((view, u64::MIN)..=(view, addr)).next_back()?;
    (addr < *end).then(|| Arc::clone(module))
}

/// A component contributes one per nested core module, so anything covering all the code in a file
/// walks these
pub fn all(view: ViewId) -> Vec<Arc<Module>> {
    let read = match READ.read() {
        Ok(read) => read,
        Err(poisoned) => poisoned.into_inner(),
    };

    read.range((view, u64::MIN)..=(view, u64::MAX))
        .map(|(_, (_, module))| Arc::clone(module))
        .collect()
}

/// Answers only when every file covering the address agrees, so an ambiguous one gives nothing
/// rather than another file's answer
pub fn lookup_anywhere(addr: u64) -> Option<Arc<Module>> {
    let read = match READ.read() {
        Ok(read) => read,
        Err(poisoned) => poisoned.into_inner(),
    };

    // One entry per open file, so walking them all is walking a handful
    let mut found: Option<&Arc<Module>> = None;
    for ((_, base), (end, module)) in read.iter() {
        if !(*base..*end).contains(&addr) {
            continue;
        }
        match found {
            Some(seen) if !Arc::ptr_eq(seen, module) => return None,
            _ => found = Some(module),
        }
    }
    found.map(Arc::clone)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    I32,
    I64,
    F32,
    F64,
    V128,
    /// Reference types differ only in what they point at, which is a module type
    Ref,
}

impl ValueKind {
    pub fn size(self) -> usize {
        match self {
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
            Self::V128 => 16,
            // References are opaque handles, and wasm32 addresses are four bytes
            Self::Ref => 4,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::V128 => "v128",
            Self::Ref => "ref",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    fn of(ty: &wasmparser::ValType) -> Self {
        use wasmparser::ValType;
        match ty {
            ValType::I32 => Self::I32,
            ValType::I64 => Self::I64,
            ValType::F32 => Self::F32,
            ValType::F64 => Self::F64,
            ValType::V128 => Self::V128,
            ValType::Ref(_) => Self::Ref,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Signature {
    pub params: Vec<ValueKind>,
    pub results: Vec<ValueKind>,
}

impl Signature {
    pub fn arity(&self) -> Arity {
        Arity {
            pops: self.params.len() as u32,
            pushes: self.results.len() as u32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionInfo {
    /// Start of the body, at its locals declaration, which is what a DWARF `DW_AT_low_pc` names
    pub start: u64,
    /// The first instruction, past the locals declaration
    pub entry: u64,
    /// One past the last byte of the body
    pub end: u64,
    pub signature: Signature,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportInfo {
    /// The module it is imported from, such as `env` or `wasi_snapshot_preview1`
    pub module: String,
    pub field: String,
    pub signature: Signature,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolved {
    Call(Call),
    Aggregate(Arity),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    /// `None` only for the indirect forms, which take their callee off the stack
    pub target: Option<u64>,
    /// Includes the operand naming the callee, where there is one
    pub arity: Arity,
    pub params: Vec<ValueKind>,
    pub result: Option<ValueKind>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionSpan {
    pub name: String,
    pub start: u64,
    pub end: u64,
    pub code: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalInfo {
    pub kind: ValueKind,
    pub mutable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataSpan {
    pub start: u64,
    pub end: u64,
    /// Where an active segment with a constant offset is copied to; the others have none
    pub memory_offset: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Module {
    /// `None` for a type that is not a function, which garbage collection makes common
    types: Vec<Option<Signature>>,
    structs: BTreeMap<u32, u32>,
    tags: BTreeMap<u32, u32>,
    /// Indexed by function index, which counts imports before anything defined here
    functions: BTreeMap<u32, FunctionInfo>,
    imports: BTreeMap<u32, ImportInfo>,
    /// In address order, so finding the body covering an address is a search rather than a scan
    bodies: Vec<(u64, u64, u32)>,
    exports: BTreeMap<u32, String>,
    names: BTreeMap<u32, String>,
    locals: BTreeMap<u32, BTreeMap<u32, String>>,
    /// By type index, from the extended name section; a function's own local names come first
    parameter_names: BTreeMap<u32, BTreeMap<u32, String>>,
    /// Type index per function index, imports included
    function_types: BTreeMap<u32, u32>,
    /// Parameters first, per function index; the lifter moves a local at this width, so a slot is
    /// written and read as the same size
    local_kinds: BTreeMap<u32, Vec<ValueKind>>,
    /// What the first table holds, by slot, from the active segments at a constant offset
    elements: BTreeMap<u64, u32>,
    /// Globals, in an index space that counts imported ones first
    globals: BTreeMap<u32, GlobalInfo>,
    global_names: BTreeMap<u32, String>,
    data_names: BTreeMap<u32, String>,
    memory_names: BTreeMap<u32, String>,
    table_names: BTreeMap<u32, String>,
    type_names: BTreeMap<u32, String>,
    tag_names: BTreeMap<u32, String>,
    pub name: Option<String>,
    pub sections: Vec<SectionSpan>,
    /// Initial size of each declared memory, in bytes
    pub memories: Vec<u64>,
    pub memory64: bool,
    /// Declared size of each table, in slots
    pub tables: Vec<u64>,
    pub data: Vec<DataSpan>,
    pub start: Option<u32>,
    /// The whole file for a core module, or the nested range when it came out of a component
    pub base: u64,
    pub end: u64,
    pub layout: Layout,
}

impl Module {
    pub fn shape(&self) -> Shape {
        Shape {
            image: self.end.saturating_sub(self.base),
            memory: self.memories.first().copied().unwrap_or(0).max(
                self.data
                    .iter()
                    .filter_map(|span| {
                        Some(span.memory_offset? + span.end.saturating_sub(span.start))
                    })
                    .max()
                    .unwrap_or(0),
            ),
            globals: self.globals.len() as u64,
            imports: self.imports.len() as u64,
            tables: self.table_slots(),
            memory64: self.memory64,
        }
    }

    /// The table section says how big the table is; an element segment's offset does not
    fn table_slots(&self) -> u64 {
        let declared = self.tables.first().copied().unwrap_or(0);
        let reached = self
            .elements
            .keys()
            .max()
            .map_or(0, |slot| slot.saturating_add(1));
        // Nothing declared means an imported table, whose size is somebody else's
        let slots = if declared == 0 { reached } else { declared };
        slots.min(MAX_TABLE_SLOTS)
    }

    /// Reading a module needs a base and working out the base needs the module, so it is read at
    /// one place and moved to the other rather than read twice
    pub fn place(&mut self, layout: Layout) {
        let shift = layout.file_base.wrapping_sub(self.layout.file_base);
        let move_to = |address: &mut u64| *address = address.wrapping_add(shift);

        for info in self.functions.values_mut() {
            move_to(&mut info.start);
            move_to(&mut info.entry);
            move_to(&mut info.end);
        }
        for (start, end, _) in &mut self.bodies {
            move_to(start);
            move_to(end);
        }
        for section in &mut self.sections {
            move_to(&mut section.start);
            move_to(&mut section.end);
        }
        for span in &mut self.data {
            move_to(&mut span.start);
            move_to(&mut span.end);
        }
        move_to(&mut self.base);
        move_to(&mut self.end);
        self.layout = layout;
    }

    /// An import has no body, so it gets a place outside the image rather than no address at all
    pub fn entry(&self, function: u32) -> Option<u64> {
        self.functions
            .get(&function)
            .map(|info| info.entry)
            .or_else(|| {
                self.imports
                    .contains_key(&function)
                    .then(|| self.layout.import_address(function))
            })
    }

    pub fn body(&self, function: u32) -> Option<&FunctionInfo> {
        self.functions.get(&function)
    }

    pub fn body_covering(&self, addr: u64) -> Option<(u32, &FunctionInfo)> {
        let found = self
            .bodies
            .binary_search_by(|(start, end, _)| {
                if addr < *start {
                    std::cmp::Ordering::Greater
                } else if addr >= *end {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()?;
        let (_, _, index) = self.bodies[found];
        Some((index, self.functions.get(&index)?))
    }

    pub fn arity(&self, function: u32) -> Option<Arity> {
        self.signature(function).map(Signature::arity)
    }

    pub fn signature(&self, function: u32) -> Option<&Signature> {
        self.functions
            .get(&function)
            .map(|info| &info.signature)
            .or_else(|| self.imports.get(&function).map(|info| &info.signature))
    }

    pub fn type_arity(&self, index: u32) -> Option<Arity> {
        self.type_signature(index).map(Signature::arity)
    }

    pub fn type_signature(&self, index: u32) -> Option<&Signature> {
        self.types.get(index as usize)?.as_ref()
    }

    pub fn function(&self, index: u32) -> Option<&FunctionInfo> {
        self.functions.get(&index)
    }

    pub fn functions(&self) -> impl Iterator<Item = (u32, &FunctionInfo)> {
        self.functions.iter().map(|(index, info)| (*index, info))
    }

    pub fn imports(&self) -> impl Iterator<Item = (u32, &ImportInfo)> {
        self.imports.iter().map(|(index, info)| (*index, info))
    }

    pub fn import(&self, index: u32) -> Option<&ImportInfo> {
        self.imports.get(&index)
    }

    /// Failing every name the module gives, the index, since a `.wasm` has no addresses of its own
    pub fn name_of(&self, function: u32) -> String {
        if let Some(name) = self.names.get(&function) {
            return name.clone();
        }
        if let Some(name) = self.exports.get(&function) {
            return name.clone();
        }
        if let Some(import) = self.imports.get(&function) {
            return format!("{}.{}", import.module, import.field);
        }
        format!("func_{function}")
    }

    pub fn is_named(&self, function: u32) -> bool {
        self.names.contains_key(&function)
            || self.exports.contains_key(&function)
            || self.imports.contains_key(&function)
    }

    pub fn export_name(&self, function: u32) -> Option<&str> {
        self.exports.get(&function).map(String::as_str)
    }

    /// From the local name subsection, or failing that for a parameter, the name its function
    /// type gives it
    pub fn local_name(&self, function: u32, local: u32) -> Option<&str> {
        if let Some(name) = self
            .locals
            .get(&function)
            .and_then(|locals| locals.get(&local))
        {
            return Some(name);
        }
        let params = self.signature(function)?.params.len() as u32;
        if local >= params {
            return None;
        }
        let ty = self.function_types.get(&function)?;
        self.parameter_names
            .get(ty)?
            .get(&local)
            .map(String::as_str)
    }

    /// The declared locals, spelled `index: name`; parameters are named through the prototype
    pub fn local_names_past(&self, function: u32, params: u32) -> Vec<String> {
        let Some(locals) = self.locals.get(&function) else {
            return Vec::new();
        };
        locals
            .range(params..)
            .map(|(index, name)| format!("{index}: {name}"))
            .collect()
    }

    pub fn table_entries(&self) -> impl Iterator<Item = (u64, u32)> + '_ {
        self.elements
            .iter()
            .map(|(slot, function)| (*slot, *function))
    }

    pub fn table_entry(&self, slot: u64) -> Option<u32> {
        self.elements.get(&slot).copied()
    }

    /// What a `call_indirect` on that type can reach without trapping
    pub fn table_slots_of_type(&self, type_index: u32) -> Vec<(u64, u32)> {
        let Some(wanted) = self.types.get(type_index as usize).and_then(Option::as_ref) else {
            return Vec::new();
        };
        self.elements
            .iter()
            .filter(|(_, function)| self.signature(**function) == Some(wanted))
            .map(|(slot, function)| (*slot, *function))
            .collect()
    }

    /// Parameters included
    pub fn local_count(&self, function: u32) -> u32 {
        self.local_kinds
            .get(&function)
            .map_or(0, |kinds| kinds.len() as u32)
    }

    pub fn local_kind(&self, function: u32, local: u32) -> Option<ValueKind> {
        self.local_kinds
            .get(&function)?
            .get(local as usize)
            .copied()
    }

    pub fn pointer_width(&self) -> usize {
        if self.memory64 { 8 } else { 4 }
    }

    pub fn global_kind(&self, index: u32) -> Option<ValueKind> {
        self.globals.get(&index).map(|global| global.kind)
    }

    pub fn globals(&self) -> impl Iterator<Item = (u32, &GlobalInfo)> {
        self.globals.iter().map(|(index, info)| (*index, info))
    }

    pub fn global_name(&self, index: u32) -> String {
        self.global_names
            .get(&index)
            .cloned()
            .unwrap_or_else(|| format!("global_{index}"))
    }

    pub fn data_name(&self, index: u32) -> String {
        self.data_names
            .get(&index)
            .cloned()
            .unwrap_or_else(|| format!("data_{index}"))
    }

    pub fn memory_name(&self, index: u32) -> Option<&str> {
        self.memory_names.get(&index).map(String::as_str)
    }

    pub fn table_name(&self, index: u32) -> Option<&str> {
        self.table_names.get(&index).map(String::as_str)
    }

    pub fn tag_name(&self, index: u32) -> Option<&str> {
        self.tag_names.get(&index).map(String::as_str)
    }

    pub fn type_name(&self, index: u32) -> Option<&str> {
        self.type_names.get(&index).map(String::as_str)
    }

    pub fn is_empty(&self) -> bool {
        self.functions.is_empty() && self.imports.is_empty()
    }

    pub fn struct_fields(&self, index: u32) -> Option<u32> {
        self.structs.get(&index).copied()
    }

    /// What a handler for it is entered holding
    pub fn tag_arity(&self, index: u32) -> Option<u32> {
        self.tags.get(&index).copied()
    }

    pub fn resolve(&self, op: &Operator) -> Option<Resolved> {
        if let Some(call) = self.call(op) {
            return Some(Resolved::Call(call));
        }

        // A handler is only reached along a throw edge, so nothing is consumed getting there
        if let Operator::Catch { tag_index } = op {
            let carried = self.tags.get(tag_index).copied()?;
            return Some(Resolved::Aggregate(Arity {
                pops: 0,
                pushes: carried,
            }));
        }

        let (index, extra) = match op {
            Operator::StructNew { struct_type_index } => (*struct_type_index, 0),
            // The descriptor form takes the descriptor on top of the fields
            Operator::StructNewDesc { struct_type_index } => (*struct_type_index, 1),
            _ => return None,
        };

        let fields = self.struct_fields(index)?;
        Some(Resolved::Aggregate(Arity {
            pops: fields.saturating_add(extra),
            pushes: 1,
        }))
    }

    /// The indirect forms take the callee as an operand, so their arity is the signature's plus
    /// the table index or reference they consume
    pub fn call(&self, op: &Operator) -> Option<Call> {
        match op {
            Operator::Call { function_index } | Operator::ReturnCall { function_index } => {
                let signature = self.signature(*function_index)?;
                Some(Call {
                    target: self.entry(*function_index),
                    arity: signature.arity(),
                    params: signature.params.clone(),
                    result: signature.results.first().copied(),
                })
            }
            Operator::CallIndirect { type_index, .. }
            | Operator::ReturnCallIndirect { type_index, .. }
            | Operator::CallRef { type_index }
            | Operator::ReturnCallRef { type_index } => {
                let signature = self.type_signature(*type_index)?;
                let mut arity = signature.arity();
                arity.pops = arity.pops.saturating_add(1);
                Some(Call {
                    target: None,
                    arity,
                    params: signature.params.clone(),
                    result: signature.results.first().copied(),
                })
            }
            _ => None,
        }
    }
}

/// The second half of the version word is the layer, zero for a core module and one for a
/// component, whose index spaces are not its nested modules'
fn is_core_module(image: &[u8]) -> bool {
    image.starts_with(b"\0asm") && image.get(6..8) == Some(&[0, 0])
}

/// A component is a container whose core modules each number their functions, types and globals
/// from zero, so each one is a span of its own
pub fn core_module_spans(image: &[u8]) -> Vec<Range<usize>> {
    if is_core_module(image) {
        return std::iter::once(0..image.len()).collect();
    }
    if !image.starts_with(b"\0asm") {
        return Vec::new();
    }

    // Nested modules at any depth, since `parse_all` descends on its own
    let mut spans = Vec::new();
    for payload in Parser::new(0).parse_all(image) {
        let Ok(Payload::ModuleSection {
            unchecked_range, ..
        }) = payload
        else {
            continue;
        };
        // The parser does not bounds check that range
        if let (Ok(start), Ok(end)) = (
            usize::try_from(unchecked_range.start),
            usize::try_from(unchecked_range.end),
        ) && end <= image.len()
        {
            spans.push(start..end);
        }
    }
    spans
}

/// Reading a component's nested modules as one would resolve calls against another module's
/// signatures, so each is read separately and placed at the bytes it occupies
pub fn parse_all(image: &[u8], base: u64) -> Vec<Module> {
    core_module_spans(image)
        .into_iter()
        .filter_map(|span| {
            let start = span.start as u64;
            let mut module = parse(image.get(span)?, base + start)?;
            // The addresses are the file's already, so `place` has to measure from the file
            module.layout = Layout::at(base, module.shape());
            Some(module)
        })
        .collect()
}

/// `base` is the address `image` starts at, so what comes back is addresses rather than offsets;
/// `None` unless `image` begins with a core module header
pub fn parse(image: &[u8], base: u64) -> Option<Module> {
    if !is_core_module(image) {
        return None;
    }

    let mut signatures: Vec<u32> = Vec::new();
    let mut module = Module {
        base,
        end: base + image.len() as u64,
        ..Module::default()
    };
    let mut next_index = 0u32;
    let mut next_global = 0u32;
    let mut next_tag = 0u32;
    let mut defined = 0u32;

    for payload in Parser::new(0).parse_all(image) {
        // Carrying on past an unreadable section would shift every later index by one and
        // mislabel the rest of the module
        let Ok(payload) = payload else {
            break;
        };

        if let Some(span) = section_span(&payload, base) {
            module.sections.push(span);
        }

        match payload {
            Payload::TypeSection(reader) => {
                for group in reader.into_iter().flatten() {
                    for ty in group.into_types() {
                        if let wasmparser::CompositeInnerType::Struct(fields) =
                            &ty.composite_type.inner
                        {
                            let index = module.types.len() as u32;
                            module.structs.insert(index, fields.fields.len() as u32);
                        }
                        module.types.push(signature_of(&ty));
                    }
                }
            }
            Payload::ImportSection(reader) => {
                for import in reader.into_imports().flatten() {
                    if let TypeRef::Tag(tag) = import.ty {
                        // An imported tag takes an index before any declared one
                        let carried = module
                            .types
                            .get(tag.func_type_idx as usize)
                            .and_then(|ty| ty.as_ref())
                            .map_or(0, |ty| ty.params.len() as u32);
                        module.tags.insert(next_tag, carried);
                        next_tag += 1;
                    }
                    // An imported table takes index 0, ahead of any declared one
                    if let TypeRef::Table(table) = import.ty {
                        module.tables.push(table.initial);
                    }
                    if let TypeRef::Memory(memory) = import.ty {
                        let page = 1u64 << memory.page_size_log2.unwrap_or(16);
                        module.memories.push(memory.initial.saturating_mul(page));
                        module.memory64 |= memory.memory64;
                    }
                    if let TypeRef::Global(global) = import.ty {
                        // An imported global takes an index before any declared one
                        module.globals.insert(
                            next_global,
                            GlobalInfo {
                                kind: ValueKind::of(&global.content_type),
                                mutable: global.mutable,
                            },
                        );
                        next_global += 1;
                    }
                    if let TypeRef::Func(type_index) | TypeRef::FuncExact(type_index) = import.ty {
                        let signature = module
                            .types
                            .get(type_index as usize)
                            .cloned()
                            .flatten()
                            .unwrap_or_default();
                        module.imports.insert(
                            next_index,
                            ImportInfo {
                                module: clean(import.module),
                                field: clean(import.name),
                                signature,
                            },
                        );
                        module.function_types.insert(next_index, type_index);
                        next_index += 1;
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                signatures.extend(reader.into_iter().flatten());
            }
            Payload::TagSection(reader) => {
                for tag in reader.into_iter().flatten() {
                    let carried = module
                        .types
                        .get(tag.func_type_idx as usize)
                        .and_then(|ty| ty.as_ref())
                        .map_or(0, |ty| ty.params.len() as u32);
                    module.tags.insert(next_tag, carried);
                    next_tag += 1;
                }
            }
            Payload::MemorySection(reader) => {
                for memory in reader.into_iter().flatten() {
                    let page = 1u64 << memory.page_size_log2.unwrap_or(16);
                    module.memories.push(memory.initial.saturating_mul(page));
                    module.memory64 |= memory.memory64;
                }
            }
            Payload::TableSection(reader) => {
                for table in reader.into_iter().flatten() {
                    module.tables.push(table.ty.initial);
                }
            }
            // What a `call_indirect` selects from: an index no active segment filled traps rather
            // than calling anything
            Payload::ElementSection(reader) => {
                for element in reader.into_iter().flatten() {
                    let wasmparser::ElementKind::Active {
                        table_index,
                        offset_expr,
                    } = &element.kind
                    else {
                        continue;
                    };
                    // A passive segment is filled at run time, and only the first table is what a
                    // `call_indirect` reaches without an immediate
                    if table_index.unwrap_or(0) != 0 {
                        continue;
                    }
                    let Some(at) = const_value(offset_expr) else {
                        continue;
                    };
                    // A slot naming no function stays empty, and the rest keep their own index
                    let filled: Vec<Option<u32>> = match &element.items {
                        wasmparser::ElementItems::Functions(items) => {
                            items.clone().into_iter().map(|item| item.ok()).collect()
                        }
                        wasmparser::ElementItems::Expressions(_, items) => items
                            .clone()
                            .into_iter()
                            .map(|item| item.ok().and_then(|expr| ref_func_index(&expr)))
                            .collect(),
                    };
                    for (nth, function) in filled.into_iter().enumerate() {
                        if let Some(function) = function {
                            module
                                .elements
                                .insert(at.saturating_add(nth as u64), function);
                        }
                    }
                }
            }
            Payload::GlobalSection(reader) => {
                for global in reader.into_iter().flatten() {
                    module.globals.insert(
                        next_global,
                        GlobalInfo {
                            kind: ValueKind::of(&global.ty.content_type),
                            mutable: global.ty.mutable,
                        },
                    );
                    next_global += 1;
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader.into_iter().flatten() {
                    if export.kind == wasmparser::ExternalKind::Func {
                        module.exports.insert(export.index, clean(export.name));
                    }
                }
            }
            Payload::StartSection { func, .. } => module.start = Some(func),
            Payload::DataSection(reader) => {
                for segment in reader.into_iter().flatten() {
                    module.data.push(DataSpan {
                        start: base + segment.data.as_ptr() as u64 - image.as_ptr() as u64,
                        end: base + segment.range.end,
                        memory_offset: data_offset(&segment),
                    });
                }
            }
            Payload::CustomSection(section) if section.name() == "name" => {
                read_names(section.data(), &mut module);
            }
            Payload::CodeSectionEntry(body) => {
                // The index space counts every body whether or not this one reads, so deriving the
                // counter from how many were kept renames everything after the first bad one
                let ordinal = defined;
                defined += 1;

                // A made up signature would have a call read as taking no arguments
                let Some(&type_index) = signatures.get(ordinal as usize) else {
                    continue;
                };
                let Some(Some(signature)) = module.types.get(type_index as usize).cloned() else {
                    continue;
                };

                // The body opens with a locals declaration; the code starts after it
                let Ok(reader) = body.get_operators_reader() else {
                    continue;
                };
                let index = next_index.saturating_add(ordinal);
                module.function_types.insert(index, type_index);

                // Parameters are locals too, and are numbered before the declared ones
                let mut kinds = signature.params.clone();
                if let Ok(locals) = body.get_locals_reader() {
                    for (count, ty) in locals.into_iter().flatten() {
                        // Nothing bounds the count here, and reserving four billion locals aborts
                        // rather than unwinds
                        let room = MAX_LOCALS.saturating_sub(kinds.len());
                        if room == 0 {
                            break;
                        }
                        let kind = ValueKind::of(&ty);
                        kinds.extend(std::iter::repeat_n(kind, (count as usize).min(room)));
                    }
                }
                module.local_kinds.insert(index, kinds);

                let info = FunctionInfo {
                    start: base + body.range().start,
                    entry: base + reader.original_position(),
                    end: base + body.range().end,
                    signature,
                };

                module.bodies.push((info.start, info.end, index));
                module.functions.insert(index, info);
            }
            Payload::End(_) => break,
            _ => {}
        }
    }

    // A passive segment has no destination of its own, so it comes from the code that copies it,
    // which is why this runs once the code section has been read
    if module.data.iter().any(|span| span.memory_offset.is_none()) {
        for (index, start) in passive_destinations(image, base, &module) {
            if let Some(span) = module.data.get_mut(index as usize) {
                span.memory_offset.get_or_insert(start);
            }
        }
    }

    // Provisional, and right for a module read on its own; a view covering a whole file works out
    // one layout for all of it and calls `place` with that
    module.layout = Layout::at(base, module.shape());
    (!module.is_empty()).then_some(module)
}

/// `Data::range` covers the header too, so the contents come from the slice itself rather than
/// from re-deriving how long that header was
fn data_offset(segment: &wasmparser::Data) -> Option<u64> {
    use wasmparser::DataKind;

    let DataKind::Active {
        memory_index,
        offset_expr,
    } = &segment.kind
    else {
        return None;
    };
    // Only the first memory is mapped, and another one's offsets are a different address space
    if *memory_index != 0 {
        return None;
    }
    const_value(offset_expr)
}

/// A module sharing its memory between threads leaves its segments passive and copies them in from
/// the start function, which is then the only statement of where they go: without it the bytes sit
/// nowhere the program addresses, and every pointer into them lands in blank memory
///
/// Only the start function is read, since decoding every body on the chance one copies a segment
/// would cost a pass over the module per view
fn passive_destinations(image: &[u8], base: u64, module: &Module) -> BTreeMap<u32, u64> {
    let mut found: BTreeMap<u32, Option<u64>> = BTreeMap::new();
    let Some(info) = module.start.and_then(|start| module.function(start)) else {
        return BTreeMap::new();
    };
    let Some(body) = image.get((info.entry - base) as usize..(info.end - base) as usize) else {
        return BTreeMap::new();
    };

    // The three operands are pushed in order but not always together, since a real module leaves
    // the destination on the stack across a `global.set`, so the stack is tracked rather than the
    // run of constants before the operator
    let mut stack: Vec<Option<u64>> = Vec::new();
    let mut at = 0;
    while at < body.len() {
        let Some(insn) = crate::insn::decode_any(&body[at..]) else {
            break;
        };
        at += insn.len;

        if let Operator::MemoryInit { data_index, .. } = insn.op {
            let length = stack.pop().flatten();
            let offset = stack.pop().flatten();
            let dest = stack.pop().flatten();
            found
                .entry(data_index)
                .and_modify(|seen| {
                    // A segment copied to two different places has no one address to be at
                    if *seen != copied_to(module, data_index, dest, offset, length) {
                        *seen = None;
                    }
                })
                .or_insert_with(|| copied_to(module, data_index, dest, offset, length));
            continue;
        }

        match insn.op {
            Operator::I32Const { value } => stack.push(Some(value as u32 as u64)),
            // An operator this layer cannot state the effect of leaves nothing to line the next
            // operands up against
            _ => match insn.arity() {
                Some(arity) if insn.flow().falls_through() => {
                    stack.truncate(stack.len().saturating_sub(arity.pops as usize));
                    stack.resize(stack.len() + arity.pushes as usize, None);
                }
                _ => stack.clear(),
            },
        }
    }

    found
        .into_iter()
        .filter_map(|(index, start)| Some((index, start?)))
        .collect()
}

/// The destination names where the copied part goes, so the segment begins that much earlier
///
/// A copy reaching past the end of its segment counts for nothing, which is the check that stops a
/// mistracked operand stack reading three unrelated constants as a destination
fn copied_to(
    module: &Module,
    index: u32,
    dest: Option<u64>,
    offset: Option<u64>,
    length: Option<u64>,
) -> Option<u64> {
    let span = module.data.get(index as usize)?;
    let (dest, offset, length) = (dest?, offset?, length?);
    (offset.checked_add(length)? <= span.end - span.start).then_some(())?;
    dest.checked_sub(offset)
}

/// The constant an initialiser expression is, for the ones that are a constant at all
fn const_value(expr: &wasmparser::ConstExpr) -> Option<u64> {
    match expr.get_operators_reader().read().ok()? {
        Operator::I32Const { value } => Some(value as u32 as u64),
        Operator::I64Const { value } => Some(value as u64),
        _ => None,
    }
}

/// The other encoding an element segment has for an entry: the expression that produces it
fn ref_func_index(expr: &wasmparser::ConstExpr) -> Option<u32> {
    match expr.get_operators_reader().read().ok()? {
        Operator::RefFunc { function_index } => Some(function_index),
        _ => None,
    }
}

/// The name and extent of whichever section a payload came from
fn section_span(payload: &Payload, base: u64) -> Option<SectionSpan> {
    let (name, range, code) = match payload {
        Payload::TypeSection(reader) => ("type", reader.range(), false),
        Payload::ImportSection(reader) => ("import", reader.range(), false),
        Payload::FunctionSection(reader) => ("function", reader.range(), false),
        Payload::TableSection(reader) => ("table", reader.range(), false),
        Payload::MemorySection(reader) => ("memory", reader.range(), false),
        Payload::TagSection(reader) => ("tag", reader.range(), false),
        Payload::GlobalSection(reader) => ("global", reader.range(), false),
        Payload::ExportSection(reader) => ("export", reader.range(), false),
        Payload::ElementSection(reader) => ("element", reader.range(), false),
        Payload::DataSection(reader) => ("data", reader.range(), false),
        Payload::CodeSectionStart { range, .. } => ("code", range.clone(), true),
        Payload::StartSection { range, .. } => ("start", range.clone(), false),
        Payload::DataCountSection { range, .. } => ("data count", range.clone(), false),
        // A custom section spans its own name as well as its payload, and handing the name bytes
        // to a DWARF reader gets it nothing
        Payload::CustomSection(section) => {
            let data = section.data_offset()..section.range().end;
            return Some(SectionSpan {
                name: clean(section.name()),
                start: base + data.start,
                end: base + data.end,
                code: false,
            });
        }
        _ => return None,
    };

    Some(SectionSpan {
        name: name.to_owned(),
        start: base + range.start,
        end: base + range.end,
        code,
    })
}

/// The section is advisory, so anything unreadable in it is skipped rather than allowed to spoil
/// the rest of the module
fn read_names(data: &[u8], module: &mut Module) {
    use wasmparser::{BinaryReader, IndirectNameMap, Name, NameSectionReader};

    /// Subsection 12 of the extended name section proposal, which `wasmparser` does not know yet
    const PARAMETER_NAMES: u8 = 12;

    let reader = NameSectionReader::new(BinaryReader::new(data, 0));
    for subsection in reader.into_iter().flatten() {
        match subsection {
            Name::Module { name, .. } => module.name = Some(clean(name)),
            Name::Function(names) => {
                for naming in names.into_iter().flatten() {
                    module.names.insert(naming.index, clean(naming.name));
                }
            }
            Name::Local(functions) => collect_indirect(functions, &mut module.locals),
            Name::Global(names) => collect(names, &mut module.global_names),
            Name::Data(names) => collect(names, &mut module.data_names),
            Name::Memory(names) => collect(names, &mut module.memory_names),
            Name::Table(names) => collect(names, &mut module.table_names),
            Name::Type(names) => collect(names, &mut module.type_names),
            Name::Tag(names) => collect(names, &mut module.tag_names),
            Name::Unknown {
                ty: PARAMETER_NAMES,
                data,
                ..
            } => {
                if let Ok(types) = IndirectNameMap::new(BinaryReader::new(data, 0)) {
                    collect_indirect(types, &mut module.parameter_names);
                }
            }
            // A label is a block depth rather than an address, a field is a GC type, and an
            // element segment or a tag's parameters have no address either, so none of them has
            // anywhere to go here
            Name::Label(_) | Name::Field(_) | Name::Element(_) | Name::Unknown { .. } => {}
        }
    }
}

fn collect(names: wasmparser::NameMap, into: &mut BTreeMap<u32, String>) {
    for naming in names.into_iter().flatten() {
        into.insert(naming.index, clean(naming.name));
    }
}

fn collect_indirect(
    names: wasmparser::IndirectNameMap,
    into: &mut BTreeMap<u32, BTreeMap<u32, String>>,
) {
    for group in names.into_iter().flatten() {
        collect(group.names, into.entry(group.index).or_default());
    }
}

/// A module may declare a name of any length, and nothing readable needs more than this
const MAX_NAME_LEN: usize = 512;

/// A WebAssembly name is any sequence of Unicode scalar values, U+0000 included, and the bindings
/// *panic* converting one to a C string, which inside a view callback aborts the process
pub(crate) fn clean(name: &str) -> String {
    name.chars()
        .take(MAX_NAME_LEN)
        .map(|c| if c.is_control() { '_' } else { c })
        .collect()
}

fn signature_of(ty: &wasmparser::SubType) -> Option<Signature> {
    use wasmparser::CompositeInnerType;

    match &ty.composite_type.inner {
        CompositeInnerType::Func(func) => Some(Signature {
            params: func.params().iter().map(ValueKind::of).collect(),
            results: func.results().iter().map(ValueKind::of).collect(),
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(text: &str) -> Vec<u8> {
        wat::parse_str(text).expect("the fixture assembles")
    }

    fn call_to(module: &Module, function: u32) -> Option<Call> {
        module.call(&Operator::Call {
            function_index: function,
        })
    }

    #[test]
    fn a_component_reads_as_one_module_per_nested_module() {
        let image = build(
            r#"(component
                 (core module (func (result i32) i32.const 1))
                 (core module
                   (func (param i32 i32) (result i64) i64.const 2)
                   (func (result f32) f32.const 0)))"#,
        );

        // The whole file is a container, and declares no functions of its own
        assert!(parse(&image, 0).is_none(), "a component is not a module");

        let modules = parse_all(&image, 0);
        assert_eq!(modules.len(), 2, "{:?}", modules.len());
        assert_eq!(modules[0].functions().count(), 1);
        assert_eq!(modules[1].functions().count(), 2);

        // Each numbers its own functions from zero, and their spans do not overlap
        assert_eq!(modules[0].arity(0), Some(Arity { pops: 0, pushes: 1 }));
        assert_eq!(modules[1].arity(0), Some(Arity { pops: 2, pushes: 1 }));
        assert_eq!(modules[1].arity(1), Some(Arity { pops: 0, pushes: 1 }));
        assert!(
            modules[0].end <= modules[1].base,
            "{:#x?} overlaps {:#x?}",
            modules[0].base..modules[0].end,
            modules[1].base..modules[1].end
        );

        // The addresses are the file's, or the view would map a body where the bytes are not
        for module in &modules {
            for (_, info) in module.functions() {
                assert!(
                    (module.base..module.end).contains(&info.entry),
                    "{:#x} is outside {:#x}..{:#x}",
                    info.entry,
                    module.base,
                    module.end
                );
                assert_eq!(
                    &image[info.start as usize..info.end as usize].len(),
                    &((info.end - info.start) as usize)
                );
            }
        }
    }

    #[test]
    fn placing_a_component_leaves_its_nested_modules_where_they_are() {
        let image = build(
            r#"(component
                 (core module (func (result i32) i32.const 1))
                 (core module (func (result i32) i32.const 2) (func nop)))"#,
        );

        let mut modules = parse_all(&image, 0);
        let before: Vec<(u64, u64, Vec<u64>)> = modules
            .iter()
            .map(|module| {
                (
                    module.base,
                    module.end,
                    module.functions().map(|(_, info)| info.entry).collect(),
                )
            })
            .collect();
        assert_eq!(modules.len(), 2);
        assert_ne!(before[0].0, before[1].0, "they start at different offsets");

        let layout = Layout::at(0, Shape::default());
        for module in &mut modules {
            module.place(layout);
        }

        for (module, (base, end, entries)) in modules.iter().zip(before) {
            assert_eq!((module.base, module.end), (base, end));
            let moved: Vec<u64> = module.functions().map(|(_, info)| info.entry).collect();
            assert_eq!(moved, entries);
        }
    }

    #[test]
    fn a_sixty_four_bit_memory_is_recognised() {
        let plain = parse(&build(r#"(module (memory 1) (func nop))"#), 0).expect("parses");
        assert!(!plain.memory64);
        assert_eq!(plain.pointer_width(), 4);

        let wide = parse(&build(r#"(module (memory i64 1) (func nop))"#), 0).expect("parses");
        assert!(wide.memory64, "declared i64 memory");
        assert_eq!(wide.pointer_width(), 8);

        // An imported memory is still this module's, and is numbered before any declared one
        let imported = parse(
            &build(r#"(module (import "env" "m" (memory i64 1)) (func nop))"#),
            0,
        )
        .expect("parses");
        assert!(imported.memory64, "imported i64 memory");
    }

    #[test]
    fn every_local_carries_its_declared_type() {
        let image = build(
            r#"(module
                 (func (param i32 i64) (result i32)
                   (local f64) (local i32 i32)
                   i32.const 1))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.local_kind(0, 0), Some(ValueKind::I32), "param 0");
        assert_eq!(module.local_kind(0, 1), Some(ValueKind::I64), "param 1");
        assert_eq!(module.local_kind(0, 2), Some(ValueKind::F64), "declared");
        assert_eq!(module.local_kind(0, 3), Some(ValueKind::I32));
        assert_eq!(module.local_kind(0, 4), Some(ValueKind::I32));
        assert_eq!(module.local_kind(0, 5), None, "past the end");
        assert_eq!(module.local_kind(1, 0), None, "no such function");
    }

    #[test]
    fn finds_every_defined_function() {
        let image = build(
            r#"(module
                 (func (result i32) i32.const 1)
                 (func (param i32 i32) (result i32) local.get 0)
                 (func))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.functions().count(), 3);
        assert_eq!(module.arity(0), Some(Arity { pops: 0, pushes: 1 }));
        assert_eq!(module.arity(1), Some(Arity { pops: 2, pushes: 1 }));
        assert_eq!(module.arity(2), Some(Arity { pops: 0, pushes: 0 }));
        assert_eq!(module.arity(3), None);
    }

    #[test]
    fn imports_shift_the_index_of_everything_defined() {
        let image = build(
            r#"(module
                 (import "env" "a" (func (param i32)))
                 (import "env" "b" (func (result i64)))
                 (func (result f32) f32.const 0))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(
            module.entry(0),
            Some(module.layout.import_address(0)),
            "an import is called at a place of its own"
        );
        assert_eq!(module.body(0), None, "but it has no body here");
        assert_eq!(module.arity(0), Some(Arity { pops: 1, pushes: 0 }));
        assert_eq!(module.arity(1), Some(Arity { pops: 0, pushes: 1 }));
        assert!(module.entry(2).is_some(), "the defined function is index 2");
        assert_eq!(module.arity(2), Some(Arity { pops: 0, pushes: 1 }));
    }

    #[test]
    fn the_entry_address_points_at_the_first_instruction() {
        let image = build("(module (func (local i32 i64) nop))");
        let module = parse(&image, 0).expect("parses");
        let info = module.function(0).expect("one function");

        assert_eq!(image[info.entry as usize], 0x01, "nop");
        assert_eq!(image[info.end as usize - 1], 0x0b, "end");
    }

    #[test]
    fn addresses_are_relative_to_the_base() {
        let image = build("(module (func nop))");
        let at_zero = parse(&image, 0).expect("parses");
        let shifted = parse(&image, 0x4000).expect("parses");

        assert_eq!(
            shifted.entry(0).unwrap(),
            at_zero.entry(0).unwrap() + 0x4000
        );
    }

    #[test]
    fn multi_value_results_are_counted() {
        let image = build(
            r#"(module
                 (type (func (param i32 i64 f32) (result i32 i64)))
                 (func (type 0) local.get 0 local.get 1))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.arity(0), Some(Arity { pops: 3, pushes: 2 }));
    }

    #[test]
    fn a_direct_call_resolves_to_the_callee() {
        let image = build(
            r#"(module
                 (import "env" "a" (func (param i32) (result i32)))
                 (func (param i32) local.get 0 call 0 drop)
                 (func (result i32) i32.const 1 call 0))"#,
        );
        let module = parse(&image, 0).expect("parses");

        let import = call_to(&module, 0).expect("the import is a call");
        assert_eq!(
            import.target,
            Some(module.layout.import_address(0)),
            "a call to an import goes to the symbol standing in for it"
        );
        assert_eq!(import.arity, Arity { pops: 1, pushes: 1 });

        let defined = call_to(&module, 2).expect("a defined function is a call");
        assert_eq!(defined.target, module.entry(2));
        assert_eq!(defined.arity, Arity { pops: 0, pushes: 1 });

        assert_eq!(call_to(&module, 9), None, "no such function");
    }

    #[test]
    fn an_indirect_call_counts_the_operand_naming_the_callee() {
        let image = build(
            r#"(module
                 (type (func (param i32 i32) (result i64)))
                 (table 1 funcref)
                 (func (param i32) i32.const 0 i32.const 0 i32.const 0 call_indirect (type 0) drop))"#,
        );
        let module = parse(&image, 0).expect("parses");

        let call = module
            .call(&Operator::CallIndirect {
                type_index: 0,
                table_index: 0,
            })
            .expect("call_indirect resolves");
        assert_eq!(call.target, None);
        assert_eq!(call.arity, Arity { pops: 3, pushes: 1 });
    }

    #[test]
    fn a_type_that_is_not_a_function_has_no_signature() {
        let image = build(
            r#"(module
                 (type (struct (field i32)))
                 (type (func (param i32)))
                 (func nop))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.type_arity(0), None, "a struct is not callable");
        assert_eq!(module.type_arity(1), Some(Arity { pops: 1, pushes: 0 }));
        assert_eq!(module.type_arity(9), None);
    }

    #[test]
    fn functions_are_named_the_way_the_module_names_them() {
        let image = build(
            r#"(module
                 (import "env" "puts" (func (param i32)))
                 (func $helper nop)
                 (func (export "run") nop)
                 (func nop))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.name_of(0), "env.puts", "an import names its origin");
        assert_eq!(module.name_of(1), "helper", "from the name section");
        assert_eq!(module.name_of(2), "run", "from the export section");
        assert_eq!(
            module.name_of(3),
            "func_3",
            "nothing to go on but the index"
        );
        assert!(!module.is_named(3));
        assert_eq!(module.export_name(2), Some("run"));
    }

    #[test]
    fn a_parameter_falls_back_to_the_name_its_type_gives_it() {
        // Subsection 12 names parameters 0, 1 and 2 of type 0 x, y and z; the local names cover
        // only parameter 0 of function 1
        let image = build(
            r#"(module
                 (type (func (param i32 i32)))
                 (import "env" "f" (func (type 0)))
                 (func (type 0) (param $a i32) (param i32) (local i64) nop)
                 (@custom "name" "\0c\0c\01\00\03\00\01x\01\01y\02\01z"))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(
            module.local_name(0, 0),
            Some("x"),
            "an import has only its type"
        );
        assert_eq!(module.local_name(0, 1), Some("y"));
        assert_eq!(
            module.local_name(1, 0),
            Some("a"),
            "the local name subsection wins"
        );
        assert_eq!(module.local_name(1, 1), Some("y"), "the type fills the gap");
        assert_eq!(
            module.local_name(1, 2),
            None,
            "a declared local is no parameter, whatever the type says"
        );
        assert!(module.local_names_past(1, 2).is_empty());
    }

    #[test]
    fn a_module_with_no_names_still_names_its_exports() {
        let image = build(r#"(module (func (export "go") nop) (func nop))"#);
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.name_of(0), "go");
        assert_eq!(module.name_of(1), "func_1");
    }

    #[test]
    fn parameters_carry_their_declared_types() {
        let image = build("(module (func (param i32 f64) (result i64) unreachable))");
        let module = parse(&image, 0).expect("parses");
        let signature = module.signature(0).expect("a signature");

        assert_eq!(signature.params, [ValueKind::I32, ValueKind::F64]);
        assert_eq!(signature.results, [ValueKind::I64]);
        assert_eq!(signature.arity(), Arity { pops: 2, pushes: 1 });
    }

    #[test]
    fn sections_are_reported_with_only_the_code_one_marked_code() {
        let image = build(r#"(module (memory 1) (data (i32.const 8) "hi") (func nop))"#);
        let module = parse(&image, 0).expect("parses");

        let named: Vec<&str> = module
            .sections
            .iter()
            .map(|section| section.name.as_str())
            .collect();
        assert!(named.contains(&"type"), "{named:?}");
        assert!(named.contains(&"code"), "{named:?}");
        assert!(named.contains(&"data"), "{named:?}");

        let code: Vec<&str> = module
            .sections
            .iter()
            .filter(|section| section.code)
            .map(|section| section.name.as_str())
            .collect();
        assert_eq!(code, ["code"]);

        // A section outside the image would have the view map something that is not there
        for section in &module.sections {
            assert!(section.start < section.end, "{section:?}");
            assert!(section.end <= image.len() as u64, "{section:?}");
        }
    }

    #[test]
    fn data_segments_are_found_with_their_contents() {
        let image = build(r#"(module (memory 1) (func nop) (data (i32.const 1024) "hello"))"#);
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.data.len(), 1);
        let span = &module.data[0];
        assert_eq!(span.memory_offset, Some(1024));
        assert_eq!(
            &image[span.start as usize..span.end as usize],
            b"hello",
            "the span covers the contents and nothing else"
        );
    }

    #[test]
    fn a_segment_belonging_to_another_memory_is_left_unplaced() {
        let image = build(
            r#"(module (memory 1) (memory 1) (func nop)
                 (data (memory 0) (i32.const 16) "AAAA")
                 (data (memory 1) (i32.const 16) "BBBB"))"#,
        );
        let module = parse(&image, 0).expect("parses");

        let placed: Vec<_> = module
            .data
            .iter()
            .map(|span| {
                (
                    span.memory_offset,
                    &image[span.start as usize..span.end as usize],
                )
            })
            .collect();
        assert_eq!(
            placed,
            [(Some(16), &b"AAAA"[..]), (None, &b"BBBB"[..])],
            "the first memory's segment is placed and the second memory's is not"
        );
    }

    #[test]
    fn a_passive_segment_lands_where_memory_init_puts_it() {
        let image = build(
            r#"(module
                 (memory 1)
                 (data $late "hello")
                 (start $init)
                 (func $init
                   i32.const 1024
                   i32.const 0
                   i32.const 5
                   memory.init $late
                   data.drop $late))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.data.len(), 1);
        assert_eq!(module.data[0].memory_offset, Some(1024));
    }

    #[test]
    fn a_partial_copy_places_the_segment_by_where_its_first_byte_would_go() {
        let image = build(
            r#"(module
                 (memory 1)
                 (data $late "hello")
                 (start $init)
                 (func $init
                   i32.const 1024
                   i32.const 2
                   i32.const 3
                   memory.init $late))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.data[0].memory_offset, Some(1022));
    }

    #[test]
    fn a_segment_copied_twice_is_left_where_it_was() {
        let image = build(
            r#"(module
                 (memory 1)
                 (data $late "hello")
                 (start $init)
                 (func $init
                   i32.const 1024
                   i32.const 0
                   i32.const 5
                   memory.init $late
                   i32.const 2048
                   i32.const 0
                   i32.const 5
                   memory.init $late))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.data[0].memory_offset, None);
    }

    #[test]
    fn a_computed_destination_leaves_the_segment_unplaced() {
        let image = build(
            r#"(module
                 (memory 1)
                 (global $where i32 (i32.const 1024))
                 (data $late "hello")
                 (start $init)
                 (func $init
                   global.get $where
                   i32.const 0
                   i32.const 5
                   memory.init $late))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.data[0].memory_offset, None);
    }

    #[test]
    fn the_start_function_is_the_entry_point() {
        let image = build("(module (func $go nop) (start $go))");
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.start, Some(0));
        assert_eq!(module.entry(0), module.body(0).map(|info| info.entry));
    }

    #[test]
    fn globals_are_indexed_and_named_like_functions() {
        let image = build(
            r#"(module
                 (import "env" "base" (global i32))
                 (global $counter (mut i64) (i64.const 0))
                 (global f32 (f32.const 0))
                 (func nop))"#,
        );
        let module = parse(&image, 0).expect("parses");

        let globals: Vec<_> = module
            .globals()
            .map(|(index, info)| (index, info.kind, info.mutable))
            .collect();
        assert_eq!(
            globals,
            [
                (0, ValueKind::I32, false),
                (1, ValueKind::I64, true),
                (2, ValueKind::F32, false)
            ],
            "the import takes index 0"
        );

        assert_eq!(module.global_name(1), "counter", "from the name section");
        assert_eq!(module.global_name(2), "global_2", "nothing names it");
        let layout = module.layout;
        assert_eq!(layout.global_address(1), layout.global_base + GLOBAL_STRIDE);
    }

    #[test]
    fn struct_construction_resolves_against_the_type_section() {
        let image = build(
            r#"(module
                 (type $point (struct (field i32) (field i32) (field f64)))
                 (type $empty (struct))
                 (func nop))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.struct_fields(0), Some(3));
        assert_eq!(module.struct_fields(1), Some(0));
        assert_eq!(module.struct_fields(7), None, "no such type");

        assert_eq!(
            module.resolve(&Operator::StructNew {
                struct_type_index: 0
            }),
            Some(Resolved::Aggregate(Arity { pops: 3, pushes: 1 }))
        );
        // The descriptor form takes one more, on top of the fields
        assert_eq!(
            module.resolve(&Operator::StructNewDesc {
                struct_type_index: 0
            }),
            Some(Resolved::Aggregate(Arity { pops: 4, pushes: 1 }))
        );
    }

    #[test]
    fn a_data_segment_is_placed_in_memory_as_well_as_in_the_file() {
        let image = build(r#"(module (memory 2) (func nop) (data (i32.const 64) "hi"))"#);
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.memories, [2 * 64 * 1024], "two pages");

        let span = &module.data[0];
        assert_eq!(&image[span.start as usize..span.end as usize], b"hi");
        assert_eq!(span.memory_offset, Some(64));

        // A pointer held in an `i32.const` has to name the byte it points at, with nothing added
        let layout = Layout::allocate(module.shape());
        assert_eq!(layout.memory_address(64), 64);

        // A file offset is mapped above memory, so the two cannot be confused
        assert_eq!(
            layout.file_address(span.start),
            layout.file_base + span.start
        );
        assert!(layout.file_address(span.start) > layout.memory_address(span.start));
    }

    #[test]
    fn no_layout_overlaps_itself() {
        let shapes = [
            Shape::default(),
            Shape {
                image: 1,
                memory: 0,
                globals: 0,
                imports: 0,
                tables: 0,
                memory64: false,
            },
            // The most a wasm32 module can declare, which leaves no room at all
            Shape {
                image: 4 << 20,
                memory: 1 << 32,
                globals: 4096,
                imports: 8192,
                tables: 65536,
                memory64: false,
            },
            // The same past what a wasm32 address can hold, which only memory64 reaches
            Shape {
                image: MAX_IMAGE_LEN as u64,
                memory: 1 << 40,
                globals: 1 << 20,
                imports: 1 << 20,
                tables: 1 << 20,
                memory64: true,
            },
        ];

        for shape in shapes {
            for layout in [Layout::at(1 << 30, shape), Layout::allocate(shape)] {
                assert!(layout.is_ordered(), "{layout:?} overlaps itself");
                assert!(
                    !layout.is_import_stub(layout.global_base),
                    "globals are not code"
                );
                assert!(!layout.is_import_stub(layout.file_base), "nor is the file");
                assert!(!layout.is_import_stub(layout.import_end()));
                assert_eq!(layout.pointer, if shape.memory64 { 8 } else { 4 });
                assert!(
                    layout.end()
                        <= if shape.memory64 {
                            WASM64_CEILING
                        } else {
                            WASM32_CEILING
                        }
                );
            }
        }
    }

    #[test]
    fn two_files_do_not_share_the_regions_above_memory() {
        let small = Layout::allocate(Shape {
            image: 4 << 10,
            memory: 1 << 20,
            globals: 4,
            imports: 4,
            tables: 8,
            memory64: false,
        });
        let large = Layout::allocate(Shape {
            image: 1 << 20,
            memory: 64 << 20,
            globals: 900,
            imports: 900,
            tables: 4096,
            memory64: false,
        });

        assert!(
            small.memory_end <= small.file_base,
            "memory is below the file"
        );
        assert!(large.memory_end <= large.file_base);
        assert!(
            small.end() <= large.file_base || large.end() <= small.file_base,
            "{small:?} and {large:?} overlap"
        );
        for index in [0, 3] {
            assert!(!large.is_import_stub(small.import_address(index)));
            assert!(!small.is_import_stub(large.import_address(index)));
        }
    }

    #[test]
    fn placing_a_module_moves_every_address_it_holds() {
        let image = build(r#"(module (memory 1) (func nop) (data (i32.const 0) "hi"))"#);
        let read = parse(&image, 0).expect("parses");

        let mut moved = read.clone();
        let layout = Layout::allocate(read.shape());
        moved.place(layout);

        let shift = layout.file_base;
        assert_eq!(moved.base, read.base + shift);
        assert_eq!(moved.end, read.end + shift);
        for ((_, before), (_, after)) in read.functions().zip(moved.functions()) {
            assert_eq!(after.start, before.start + shift);
            assert_eq!(after.entry, before.entry + shift);
            assert_eq!(after.end, before.end + shift);
        }
        for (before, after) in read.sections.iter().zip(&moved.sections) {
            assert_eq!(
                (after.start, after.end),
                (before.start + shift, before.end + shift)
            );
        }
        for (before, after) in read.data.iter().zip(&moved.data) {
            assert_eq!(
                (after.start, after.end),
                (before.start + shift, before.end + shift)
            );
            assert_eq!(
                after.memory_offset, before.memory_offset,
                "memory does not move"
            );
        }
    }

    #[test]
    fn the_element_section_says_what_a_table_slot_holds() {
        let image = build(
            r#"(module
                 (type $unary (func (param i32) (result i32)))
                 (type $nullary (func))
                 (func $a (type $unary) local.get 0)
                 (func $b (type $unary) local.get 0)
                 (func $c (type $nullary))
                 (table 8 funcref)
                 (elem (i32.const 2) $a $c $b))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(
            module.table_entry(2),
            Some(0),
            "the segment starts at slot 2"
        );
        assert_eq!(module.table_entry(3), Some(2));
        assert_eq!(module.table_entry(4), Some(1));
        assert_eq!(
            module.table_entry(0),
            None,
            "nothing fills the slots below it"
        );
        assert_eq!(
            module.shape().tables,
            8,
            "sized to the table the module declared, not to the last slot a segment filled"
        );

        // What a `call_indirect` on each type could reach without trapping
        let unary: Vec<_> = module
            .table_slots_of_type(0)
            .into_iter()
            .map(|(slot, _)| slot)
            .collect();
        assert_eq!(unary, [2, 4]);
        assert_eq!(module.table_slots_of_type(1).len(), 1);
        assert!(module.table_slots_of_type(9).is_empty(), "no such type");
    }

    /// What a module with reference types enabled emits
    #[test]
    fn an_element_segment_written_as_expressions_fills_the_same_slots() {
        let image = build(
            r#"(module
                 (type $unary (func (param i32) (result i32)))
                 (func $a (type $unary) local.get 0)
                 (func $b (type $unary) local.get 0)
                 (table 8 funcref)
                 (elem (i32.const 2) funcref (ref.func $a) (ref.null func) (ref.func $b)))"#,
        );
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.table_entry(2), Some(0));
        assert_eq!(
            module.table_entry(3),
            None,
            "a null entry traps rather than naming a function"
        );
        assert_eq!(
            module.table_entry(4),
            Some(1),
            "and the slots after it keep their own index"
        );
        assert_eq!(module.shape().tables, 8, "the whole declared table");
    }

    #[test]
    fn a_table_is_as_big_as_the_table_section_says() {
        let image = build(r#"(module (func $f) (table 4 funcref) (elem (i32.const 1) $f))"#);
        let module = parse(&image, 0).expect("parses");
        assert_eq!(
            module.shape().tables,
            4,
            "the declared size, not the filled one"
        );

        // An engine rejects this one, so the slot it names is unreachable
        let reaching =
            build(r#"(module (func $f) (table 4 funcref) (elem (i32.const 4294967200) $f))"#);
        let module = parse(&reaching, 0).expect("parses");
        assert!(
            module
                .table_entries()
                .any(|(slot, _)| slot > u64::from(u32::MAX) / 2),
            "the segment really does name a slot out there"
        );
        assert_eq!(
            module.shape().tables,
            4,
            "and the region stays the size the table section declared"
        );
    }

    #[test]
    fn a_passive_element_segment_fills_no_slot() {
        let image = build(r#"(module (func $a) (table 4 funcref) (elem funcref (ref.func $a)))"#);
        let module = parse(&image, 0).expect("parses");
        assert_eq!(module.table_entries().count(), 0);
        assert_eq!(
            module.shape().tables,
            4,
            "the table is still there, with every slot trapping"
        );
    }

    #[test]
    fn a_body_is_found_by_the_address_it_covers() {
        let image = build("(module (func (local i32) nop) (func nop) (func (local i64 i64) nop))");
        let module = parse(&image, 0).expect("parses");

        for (index, info) in module.functions() {
            // The locals declaration counts as part of the body, and DWARF points at it
            assert_eq!(
                module.body_covering(info.start).map(|(i, _)| i),
                Some(index)
            );
            assert_eq!(
                module.body_covering(info.entry).map(|(i, _)| i),
                Some(index)
            );
            assert_eq!(
                module.body_covering(info.end - 1).map(|(i, _)| i),
                Some(index)
            );
        }

        let last = module.functions().last().expect("a last body").1;
        assert_eq!(
            module.body_covering(last.end),
            None,
            "one past the last body"
        );
        assert_eq!(module.body_covering(0), None, "the header is not a body");
        assert_eq!(module.body_covering(u64::MAX), None);
    }

    #[test]
    fn names_are_made_safe_to_hand_to_the_core() {
        assert_eq!(clean("plain"), "plain");
        assert_eq!(
            clean("has\0nul"),
            "has_nul",
            "a nul would abort the process"
        );
        assert_eq!(clean("tab\there\nand newline"), "tab_here_and newline");
        assert!(!clean(&"x".repeat(10_000)).is_empty());
        assert_eq!(clean(&"x".repeat(10_000)).chars().count(), MAX_NAME_LEN);

        // Anything printable is left exactly as the module spelled it, including non-ASCII
        assert_eq!(clean("名前"), "名前");
        assert_eq!(
            clean("std::vector<int>::push_back"),
            "std::vector<int>::push_back"
        );
    }

    #[test]
    fn a_handler_receives_what_its_tag_carries() {
        let image = build(
            r#"(module
                 (import "env" "e" (tag (param i32)))
                 (tag (param i64 f32))
                 (tag)
                 (func nop))"#,
        );
        let module = parse(&image, 0).expect("parses");

        let carried = |tag: u32| module.resolve(&Operator::Catch { tag_index: tag });
        assert_eq!(
            carried(0),
            Some(Resolved::Aggregate(Arity { pops: 0, pushes: 1 })),
            "the import takes index 0"
        );
        assert_eq!(
            carried(1),
            Some(Resolved::Aggregate(Arity { pops: 0, pushes: 2 }))
        );
        assert_eq!(
            carried(2),
            Some(Resolved::Aggregate(Arity { pops: 0, pushes: 0 }))
        );
        assert_eq!(carried(9), None, "no such tag");
    }

    #[test]
    fn anything_that_is_not_a_call_resolves_to_nothing() {
        let image = build("(module (func nop))");
        let module = parse(&image, 0).expect("parses");

        assert_eq!(module.call(&Operator::Nop), None);
        assert_eq!(module.call(&Operator::Br { relative_depth: 0 }), None);
    }

    #[test]
    fn refuses_anything_that_is_not_a_module() {
        assert!(parse(b"", 0).is_none());
        assert!(parse(b"\x7fELF\x02\x01\x01", 0).is_none());
        assert!(parse(b"\0asm", 0).is_none(), "a header with no functions");

        let image = build("(module (memory 1))");
        assert!(parse(&image, 0).is_none(), "no code section");
    }

    #[test]
    fn survives_a_truncated_image() {
        let image = build("(module (func nop) (func nop))");
        for cut in 1..image.len() {
            let _ = parse(&image[..cut], 0);
        }
    }

    #[test]
    fn the_cache_only_answers_for_addresses_it_covers() {
        let image = build("(module (func nop))");
        let base = 0x9000_0000;
        let view = 1;
        install(
            view,
            base,
            base + image.len() as u64,
            parse(&image, base).unwrap(),
        );

        assert!(lookup(view, base).is_some());
        assert!(lookup(view, base + image.len() as u64 - 1).is_some());
        assert!(lookup(view, base + image.len() as u64).is_none());
        assert!(lookup(view, base - 1).is_none());
    }

    #[test]
    fn two_files_at_the_same_address_keep_their_own_answers() {
        let first = build("(module (func (param i32 i32) (result i32) local.get 0))");
        let second = build("(module (func nop))");
        let base = 0x1000;

        install(2, base, base + 0x800, parse(&first, base).unwrap());
        install(3, base, base + 0x800, parse(&second, base).unwrap());

        assert_eq!(
            lookup(2, base).unwrap().arity(0),
            Some(Arity { pops: 2, pushes: 1 })
        );
        assert_eq!(
            lookup(3, base).unwrap().arity(0),
            Some(Arity { pops: 0, pushes: 0 })
        );

        // The callback that cannot say which file it is looking at gets nothing rather than a
        // coin flip
        assert!(lookup_anywhere(base).is_none());
    }
}
