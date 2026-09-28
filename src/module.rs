//! Reading the module around a function
//!
//! An architecture plugin is handed bytes and an address, which cannot say what `call 3` calls or
//! how many operands it takes; the type and code sections can, and sit in the same image

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::{Arc, LazyLock, PoisonError, RwLock};

use wasmparser::{MemArg, Operator, Parser, Payload, TypeRef};

use crate::ViewId;
use crate::frame::{Effects, EntryRegion, Save, Saves};
use crate::insn::Arity;

const MAX_LOCALS: usize = 1 << 16;

pub const MAX_IMAGE_LEN: usize = 256 << 20;

/// The region is bytes the view builds, so its size is an allocation. Sixty times the largest
/// table seen here
pub const MAX_TABLE_SLOTS: u64 = 1 << 22;

/// Nothing is stored at one, so the addresses only have to be distinct
pub const IMPORT_STRIDE: u64 = 4;

pub const GLOBAL_STRIDE: u64 = 8;

const STACK_POINTER: &str = "__stack_pointer";

const PAGE: u64 = 64 << 10;

const WASM32_CEILING: u64 = 1 << 32;

const HOST_MEMORY: u64 = i32::MIN.unsigned_abs() as u64 + PAGE;

const WASM64_CEILING: u64 = 1 << 48;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Shape {
    pub image: u64,
    pub memory: u64,
    /// What the module declares, or what its data reaches
    pub reserved: u64,
    pub globals: u64,
    pub imports: u64,
    /// Slots of the first table, which is what `call_indirect` selects from
    pub tables: u64,
    pub tags: u64,
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
    pub tag_base: u64,
    pub tags: u64,
    pub pointer: usize,
}

impl Layout {
    /// Linear memory is whatever fits below `base`, which is what a module read on its own gets
    pub fn at(base: u64, shape: Shape) -> Self {
        let pointer = if shape.memory64 { 8 } else { 4 };
        let global_base = align_up(base.saturating_add(shape.image));
        let table_base = align_up(global_base.saturating_add(shape.globals * GLOBAL_STRIDE));
        let import_base = align_up(table_base.saturating_add(shape.tables * pointer as u64));
        let tag_base = align_up(import_base.saturating_add(shape.imports * IMPORT_STRIDE));
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
            tag_base,
            tags: shape.tags,
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
            + align_up(shape.tags * pointer)
            + PAGE;
        let reserved = shape
            .reserved
            .max(shape.memory)
            .min(ceiling.saturating_sub(span));
        let memory = shape.memory.min(reserved);
        Self::at(
            reserve(span, align_up(reserved), ceiling),
            Shape { memory, ..shape },
        )
    }

    /// Identity below the window, since a memory offset *is* an address; the clamp keeps a segment
    /// reaching past the end of memory out of the file above it
    pub fn memory_address(&self, offset: u64) -> u64 {
        offset.min(self.memory_end)
    }

    pub fn memory_mapped(&self, offset: u64) -> bool {
        offset != 0 && offset < self.memory_end
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

    pub fn tag_address(&self, index: u32) -> u64 {
        self.tag_base + u64::from(index) * self.pointer as u64
    }

    pub fn tag_end(&self) -> u64 {
        self.tag_base + self.tags * self.pointer as u64
    }

    /// An address per slot, which `table.get` and `call_indirect` load through
    pub fn table_address(&self, slot: u64) -> u64 {
        self.table_base
            .saturating_add(slot.saturating_mul(self.pointer as u64))
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
        self.tag_end()
            .max(self.import_end())
            .max(self.global_end())
            .max(self.memory_end)
    }

    /// The core rejects overlapping segments without a word, so this has to hold for every layout
    pub fn is_ordered(&self) -> bool {
        self.memory_end <= self.file_base
            && self.file_base + self.image <= self.global_base
            && self.global_end() <= self.table_base
            && self.table_end() <= self.import_base
            && self.import_end() <= self.tag_base
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

pub fn place_all(modules: &mut [Module], layout: Layout) {
    let (mut imports, mut globals, mut tables, mut tags) = (0u64, 0u64, 0u64, 0u64);
    for module in modules {
        let shape = module.shape();
        module.place(Layout {
            tag_base: layout
                .tag_base
                .saturating_add(tags.saturating_mul(layout.pointer as u64)),
            tags: shape.tags,
            import_base: layout
                .import_base
                .saturating_add(imports.saturating_mul(IMPORT_STRIDE)),
            imports: shape.imports,
            global_base: layout
                .global_base
                .saturating_add(globals.saturating_mul(GLOBAL_STRIDE)),
            globals: shape.globals,
            table_base: layout
                .table_base
                .saturating_add(tables.saturating_mul(layout.pointer as u64)),
            tables: shape.tables,
            ..layout
        });
        imports = imports.saturating_add(shape.imports);
        globals = globals.saturating_add(shape.globals);
        tables = tables.saturating_add(shape.tables);
        tags = tags.saturating_add(shape.tags);
    }
}

pub fn install_layout(view: ViewId, layout: Layout) {
    let mut layouts = LAYOUTS.write().unwrap_or_else(PoisonError::into_inner);
    layouts.insert(view, layout);
    let mut holders = HOLDERS.write().unwrap_or_else(PoisonError::into_inner);
    *holders.entry(view).or_default() += 1;
}

static HOLDERS: LazyLock<RwLock<BTreeMap<ViewId, usize>>> = LazyLock::new(Default::default);

static RESTORED: LazyLock<RwLock<BTreeSet<ViewId>>> = LazyLock::new(Default::default);

pub fn mark_restored(view: ViewId) {
    let mut restored = RESTORED.write().unwrap_or_else(PoisonError::into_inner);
    restored.insert(view);
}

pub fn unmark_restored(view: ViewId) {
    let mut restored = RESTORED.write().unwrap_or_else(PoisonError::into_inner);
    restored.remove(&view);
}

pub fn restored(view: ViewId) -> bool {
    let restored = RESTORED.read().unwrap_or_else(PoisonError::into_inner);
    restored.contains(&view)
}

pub fn release(view: ViewId) -> bool {
    let mut holders = HOLDERS.write().unwrap_or_else(PoisonError::into_inner);
    let Some(count) = holders.get_mut(&view) else {
        return false;
    };
    *count -= 1;
    if *count != 0 {
        return false;
    }
    holders.remove(&view);

    let mut layouts = LAYOUTS.write().unwrap_or_else(PoisonError::into_inner);
    if let Some(layout) = layouts.remove(&view) {
        let mut reserved = RESERVED.write().unwrap_or_else(PoisonError::into_inner);
        reserved.retain(|(start, _)| *start != layout.file_base);
    }
    let mut read = READ.write().unwrap_or_else(PoisonError::into_inner);
    read.retain(|(owner, _), _| *owner != view);
    unmark_restored(view);
    true
}

pub fn layout(view: ViewId) -> Option<Layout> {
    let layouts = LAYOUTS.read().unwrap_or_else(PoisonError::into_inner);
    layouts.get(&view).copied()
}

/// All that the callbacks handed an address and nothing else have to tell a stub from real code,
/// and exact rather than a guess, since two files never share a region above memory
pub fn import_stub(addr: u64) -> bool {
    let layouts = LAYOUTS.read().unwrap_or_else(PoisonError::into_inner);
    layouts.values().any(|layout| layout.is_import_stub(addr))
}

/// The layout of the file covering `addr` above linear memory, for the same callbacks
pub fn layout_covering(addr: u64) -> Option<Layout> {
    let layouts = LAYOUTS.read().unwrap_or_else(PoisonError::into_inner);
    layouts
        .values()
        .find(|layout| (layout.file_base..layout.end()).contains(&addr))
        .copied()
}

/// Never below `floor`, so the first file opened sits as low as its own memory allows and the
/// overview stays scaled to the file
static RESERVED: LazyLock<RwLock<Vec<(u64, u64)>>> = LazyLock::new(Default::default);

fn reserve(span: u64, floor: u64, ceiling: u64) -> u64 {
    let mut reserved = RESERVED.write().unwrap_or_else(PoisonError::into_inner);

    let mut base = align_up(floor);
    for &(start, end) in reserved.iter() {
        if end <= base {
            continue;
        }
        if start >= base.saturating_add(span) {
            break;
        }
        base = align_up(end);
    }
    if base.saturating_add(span) > ceiling {
        // Enough files in one session to fill the space; starting over costs the older one the
        // answers worked out from an address alone, which beats wrapping the arithmetic
        tracing::warn!(
            "wasm view: the address space is full at {base:#x}, reusing it from {floor:#x}"
        );
        base = align_up(floor);
    }
    take(&mut reserved, (base, base.saturating_add(span)));
    base
}

pub fn claim(layout: &Layout) -> bool {
    let mut reserved = RESERVED.write().unwrap_or_else(PoisonError::into_inner);
    let range = (layout.file_base, layout.end());
    if reserved
        .iter()
        .any(|&(start, end)| start < range.1 && range.0 < end)
    {
        return false;
    }
    take(&mut reserved, range);
    true
}

fn take(reserved: &mut Vec<(u64, u64)>, range: (u64, u64)) {
    let at = reserved.partition_point(|taken| *taken <= range);
    reserved.insert(at, range);
}

/// Reading a module walks the whole image and every function in it asks the same questions, so the
/// answers are kept; one with no functions is cached too, so a non-module is read only once
static READ: LazyLock<RwLock<Cache>> = LazyLock::new(Default::default);

type Cache = BTreeMap<(ViewId, u64), (u64, Arc<Module>)>;

pub fn install(view: ViewId, base: u64, end: u64, module: Module) -> Arc<Module> {
    let module = Arc::new(module);
    // This is a cache, so a poisoned lock is better carried on with than propagated
    let mut read = READ.write().unwrap_or_else(PoisonError::into_inner);
    read.insert((view, base), (end, Arc::clone(&module)));
    module
}

pub fn lookup(view: ViewId, addr: u64) -> Option<Arc<Module>> {
    let read = READ.read().unwrap_or_else(PoisonError::into_inner);

    let (_, (end, module)) = read.range((view, u64::MIN)..=(view, addr)).next_back()?;
    (addr < *end).then(|| Arc::clone(module))
}

/// A component contributes one per nested core module, so anything covering all the code in a file
/// walks these
pub fn all(view: ViewId) -> Vec<Arc<Module>> {
    let read = READ.read().unwrap_or_else(PoisonError::into_inner);

    read.range((view, u64::MIN)..=(view, u64::MAX))
        .map(|(_, (_, module))| Arc::clone(module))
        .collect()
}

/// Answers only when every file covering the address agrees, so an ambiguous one gives nothing
/// rather than another file's answer
pub fn import_signature(view: ViewId, addr: u64) -> Option<Signature> {
    all(view).iter().find_map(|module| {
        let layout = module.layout;
        if !layout.is_import_stub(addr) {
            return None;
        }
        let index = u32::try_from((addr - layout.import_base) / IMPORT_STRIDE).ok()?;
        Some(module.import(index)?.signature.clone())
    })
}

pub fn lookup_anywhere(addr: u64) -> Option<Arc<Module>> {
    let read = READ.read().unwrap_or_else(PoisonError::into_inner);

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
    pub frame: Option<(u32, u64)>,
    pub balanced: bool,
    pub returns_first_argument: bool,
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
    Function(u64),
    Switching(Arity),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    /// `None` only for the indirect forms, which take their callee off the stack
    pub target: Option<u64>,
    /// Includes the operand naming the callee, where there is one
    pub arity: Arity,
    pub params: Vec<ValueKind>,
    pub results: Vec<ValueKind>,
    pub returns_argument: bool,
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
    pub init: Option<u64>,
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
    value_types: Vec<Option<wasmparser::FuncType>>,
    structs: BTreeMap<u32, u32>,
    tags: BTreeMap<u32, Option<u32>>,
    tag_results: BTreeMap<u32, u32>,
    tag_aliases: BTreeMap<u32, u32>,
    tag_names: BTreeMap<u32, String>,
    continuations: BTreeMap<u32, u32>,
    last_param_types: BTreeMap<u32, u32>,
    pub stack_pointer: Option<u32>,
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
    address_taken: BTreeSet<u32>,
    /// Globals, in an index space that counts imported ones first
    globals: BTreeMap<u32, GlobalInfo>,
    global_names: BTreeMap<u32, String>,
    data_names: BTreeMap<u32, String>,
    type_names: BTreeMap<u32, String>,
    pub name: Option<String>,
    pub sections: Vec<SectionSpan>,
    /// Initial size of each declared memory, in bytes
    pub memories: Vec<u64>,
    pub memory_extent: u64,
    host_memory: u64,
    pub memory64: bool,
    /// Declared size of each table, in slots
    pub tables: Vec<u64>,
    table_imported: bool,
    table_exported: bool,
    table_written: bool,
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
            memory: self.memory_extent,
            reserved: self
                .memories
                .first()
                .copied()
                .unwrap_or(0)
                .max(self.data_reach())
                .max(self.host_memory),
            globals: self.globals.len() as u64,
            imports: self.imports.len() as u64,
            tables: self.table_slots(),
            tags: self.tags.len() as u64,
            memory64: self.memory64,
        }
    }

    fn address_kind(&self) -> ValueKind {
        if self.memory64 {
            ValueKind::I64
        } else {
            ValueKind::I32
        }
    }

    fn settle_returned_arguments(&mut self, forwarding: &[(u32, Vec<Forward>)]) {
        let mut held = BTreeSet::new();
        let mut dependents: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (index, forwards) in forwarding {
            let passed = forwards.iter().all(|forward| {
                self.signature(forward.callee)
                    .is_some_and(|signature| forward.passes_first(signature.params.len()))
            });
            if passed {
                held.insert(*index);
                for forward in forwards {
                    dependents.entry(forward.callee).or_default().push(*index);
                }
            }
        }
        let returns = |callee: u32| {
            held.contains(&callee)
                || self
                    .functions
                    .get(&callee)
                    .is_some_and(|info| info.returns_first_argument)
        };
        let mut dropped: Vec<u32> = forwarding
            .iter()
            .filter(|(index, forwards)| {
                held.contains(index) && !forwards.iter().all(|forward| returns(forward.callee))
            })
            .map(|(index, _)| *index)
            .collect();
        for index in &dropped {
            held.remove(index);
        }
        while let Some(index) = dropped.pop() {
            for dependent in dependents.remove(&index).unwrap_or_default() {
                if held.remove(&dependent) {
                    dropped.push(dependent);
                }
            }
        }
        for index in held {
            if let Some(info) = self.functions.get_mut(&index) {
                info.returns_first_argument = true;
            }
        }
    }

    fn name_go_registers(&mut self) {
        const REGISTERS: [(&str, ValueKind); 8] = [
            ("SP", ValueKind::I32),
            ("CTXT", ValueKind::I64),
            ("g", ValueKind::I64),
            ("RET0", ValueKind::I64),
            ("RET1", ValueKind::I64),
            ("RET2", ValueKind::I64),
            ("RET3", ValueKind::I64),
            ("PAUSE", ValueKind::I32),
        ];
        let laid_out = (0u32..).zip(REGISTERS).all(|(index, (_, kind))| {
            self.globals
                .get(&index)
                .is_some_and(|global| global.mutable && global.kind == kind)
        });
        if !self.built_by_go() || !laid_out || self.globals.len() != REGISTERS.len() {
            return;
        }
        for (index, (name, _)) in (0u32..).zip(REGISTERS) {
            self.global_names
                .entry(index)
                .or_insert_with(|| name.to_owned());
        }
    }

    fn name_go_parameters(&mut self, resuming: &BTreeSet<u32>) {
        if !self.built_by_go() {
            return;
        }
        let resumable = Signature {
            params: vec![ValueKind::I32],
            results: vec![ValueKind::I32],
        };
        let hosted = Signature {
            params: vec![ValueKind::I32],
            results: Vec::new(),
        };
        let mut named = Vec::new();
        for (index, info) in &self.functions {
            if info.signature == resumable && resuming.contains(index) {
                named.push((*index, "PC_B"));
            }
        }
        for (index, import) in &self.imports {
            if import.module == "gojs" && import.signature == hosted {
                named.push((
                    *index,
                    if import.field == "debug" {
                        "value"
                    } else {
                        "sp"
                    },
                ));
            }
        }
        for (index, name) in named {
            self.locals
                .entry(index)
                .or_default()
                .entry(0)
                .or_insert_with(|| name.to_owned());
        }
    }

    fn built_by_go(&self) -> bool {
        self.sections
            .iter()
            .any(|section| section.name == "go:buildid")
    }

    fn find_stack_pointer(
        &self,
        imported: Option<u32>,
        candidates: &BTreeMap<u32, Candidates>,
    ) -> Option<u32> {
        let usable = |index: &u32| {
            self.globals
                .get(index)
                .is_some_and(|global| global.mutable && global.kind == self.address_kind())
        };
        let named = self
            .global_names
            .iter()
            .find(|(_, name)| *name == STACK_POINTER)
            .map(|(index, _)| *index)
            .or(imported);
        if let Some(named) = named.filter(usable) {
            return Some(named);
        }

        let mut uses: BTreeMap<u32, usize> = BTreeMap::new();
        for found in candidates.values() {
            for save in &found.entry {
                if save.size != 0 {
                    *uses.entry(save.global).or_default() += 1;
                }
            }
        }
        let most = uses.values().copied().max()?;
        let mut leaders = uses.iter().filter(|(_, count)| **count == most);
        let (leader, _) = leaders.next()?;
        (leaders.next().is_none() && usable(leader)).then_some(*leader)
    }

    pub fn table_written(&self) -> bool {
        self.table_written
    }

    pub fn table_sealed(&self) -> bool {
        !self.table_imported && !self.table_exported && !self.table_written
    }

    /// The table section says how big the table is; an element segment's offset does not
    fn table_slots(&self) -> u64 {
        let declared = self.tables.first().copied().unwrap_or(0);
        let reached = self
            .elements
            .keys()
            .max()
            .map_or(0, |slot| slot.saturating_add(1));
        let slots = if self.table_imported {
            declared.max(reached)
        } else {
            declared
        };
        slots.min(MAX_TABLE_SLOTS)
    }

    fn data_reach(&self) -> u64 {
        self.data
            .iter()
            .filter_map(|span| {
                Some(
                    span.memory_offset?
                        .saturating_add(span.end.saturating_sub(span.start)),
                )
            })
            .max()
            .unwrap_or(0)
    }

    fn static_extent(&self, referenced: u64) -> u64 {
        let declared = self.memories.first().copied().unwrap_or(0);
        let named = self
            .globals
            .values()
            .filter(|global| global.kind == self.address_kind())
            .filter_map(|global| global.init)
            .filter(|at| *at <= declared)
            .max()
            .unwrap_or(0);
        self.data_reach().max(named).max(referenced.min(declared))
    }

    fn add_memory(&mut self, memory: &wasmparser::MemoryType, imported: bool) {
        let page = 1u64 << memory.page_size_log2.unwrap_or(16);
        if self.memories.is_empty() {
            self.memory64 = memory.memory64;
            if imported {
                let declared = memory.maximum.map(|pages| pages.saturating_mul(page));
                self.host_memory = declared.unwrap_or(HOST_MEMORY).min(HOST_MEMORY);
            }
        }
        self.memories.push(memory.initial.saturating_mul(page));
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

    pub fn type_value_types(&self, index: u32) -> Option<&wasmparser::FuncType> {
        self.value_types.get(index as usize)?.as_ref()
    }

    pub fn value_types(&self, function: u32) -> Option<&wasmparser::FuncType> {
        self.type_value_types(*self.function_types.get(&function)?)
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

    pub fn export_name(&self, function: u32) -> Option<&str> {
        self.exports.get(&function).map(String::as_str)
    }

    pub fn exports(&self) -> impl Iterator<Item = (u32, &str)> {
        self.exports
            .iter()
            .map(|(function, name)| (*function, name.as_str()))
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

    pub fn declared_local_names(
        &self,
        function: u32,
        params: u32,
    ) -> impl Iterator<Item = (u32, &str)> {
        self.locals
            .get(&function)
            .into_iter()
            .flat_map(move |locals| locals.range(params..))
            .map(|(index, name)| (*index, name.as_str()))
    }

    pub fn table_entries(&self) -> impl Iterator<Item = (u64, u32)> + '_ {
        self.elements
            .iter()
            .map(|(slot, function)| (*slot, *function))
    }

    pub fn table_entry(&self, slot: u64) -> Option<u32> {
        self.elements.get(&slot).copied()
    }

    pub fn address_taken(&self, function: u32) -> bool {
        self.address_taken.contains(&function)
    }

    /// Parameters included
    pub fn local_kinds(&self, function: u32) -> &[ValueKind] {
        self.local_kinds.get(&function).map_or(&[], Vec::as_slice)
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
            .unwrap_or_else(|| format!("data_segment_{index}"))
    }

    pub fn type_name(&self, index: u32) -> Option<&str> {
        self.type_names.get(&index).map(String::as_str)
    }

    pub fn is_empty(&self) -> bool {
        self.functions.is_empty() && self.imports.is_empty()
    }

    pub fn drop_frames(&mut self) {
        for info in self.functions.values_mut() {
            info.frame = None;
            info.balanced = false;
        }
    }

    pub fn struct_fields(&self, index: u32) -> Option<u32> {
        self.structs.get(&index).copied()
    }

    pub fn tags(&self) -> impl Iterator<Item = u32> + '_ {
        self.tags.keys().copied()
    }

    pub fn tag_name(&self, index: u32) -> String {
        self.tag_names
            .get(&index)
            .cloned()
            .unwrap_or_else(|| format!("tag_{index}"))
    }

    /// What a handler for it is entered holding
    pub fn tag_arity(&self, index: u32) -> Option<u32> {
        self.tags.get(&index).copied().flatten()
    }

    pub fn tag_identity(&self, index: u32) -> u32 {
        self.tag_aliases.get(&index).copied().unwrap_or(index)
    }

    fn declare_tag(&mut self, index: u32, type_index: u32) -> Option<(u32, u32)> {
        let shape = self
            .types
            .get(type_index as usize)
            .and_then(|ty| ty.as_ref())
            .map(|ty| (ty.params.len() as u32, ty.results.len() as u32));
        self.tags.insert(index, shape.map(|(params, _)| params));
        if let Some((_, results)) = shape {
            self.tag_results.insert(index, results);
        }
        shape
    }

    fn continuation(&self, index: u32) -> Option<&Signature> {
        let function = *self.continuations.get(&index)?;
        self.types.get(function as usize)?.as_ref()
    }

    fn switching(&self, op: &Operator) -> Option<Arity> {
        let count = |values: &[ValueKind]| values.len() as u32;
        match *op {
            Operator::Suspend { tag_index } => Some(Arity {
                pops: self.tag_arity(tag_index)?,
                pushes: *self.tag_results.get(&tag_index)?,
            }),
            Operator::Resume {
                cont_type_index, ..
            } => {
                let function = self.continuation(cont_type_index)?;
                Some(Arity {
                    pops: count(&function.params).checked_add(1)?,
                    pushes: count(&function.results),
                })
            }
            Operator::ResumeThrow {
                cont_type_index,
                tag_index,
                ..
            } => Some(Arity {
                pops: self.tag_arity(tag_index)?.checked_add(1)?,
                pushes: count(&self.continuation(cont_type_index)?.results),
            }),
            Operator::ResumeThrowRef {
                cont_type_index, ..
            } => Some(Arity {
                pops: 2,
                pushes: count(&self.continuation(cont_type_index)?.results),
            }),
            Operator::Switch {
                cont_type_index, ..
            } => {
                let function = *self.continuations.get(&cont_type_index)?;
                let from = self.types.get(function as usize)?.as_ref()?;
                let to = self.continuation(*self.last_param_types.get(&function)?)?;
                Some(Arity {
                    pops: count(&from.params),
                    pushes: count(&to.params),
                })
            }
            _ => None,
        }
    }

    pub fn resolve(&self, op: &Operator) -> Option<Resolved> {
        if let Some(call) = self.call(op) {
            return Some(Resolved::Call(call));
        }

        if let Operator::RefFunc { function_index } = op {
            return self.entry(*function_index).map(Resolved::Function);
        }

        if let Some(arity) = self.switching(op) {
            return Some(Resolved::Switching(arity));
        }

        if let Operator::ContBind {
            argument_index,
            result_index,
        } = *op
        {
            let bound = self
                .continuation(argument_index)?
                .params
                .len()
                .checked_sub(self.continuation(result_index)?.params.len())?;
            return Some(Resolved::Aggregate(Arity {
                pops: u32::try_from(bound).ok()?.checked_add(1)?,
                pushes: 1,
            }));
        }

        // A handler is only reached along a throw edge, so nothing is consumed getting there
        if let Operator::Catch { tag_index } = op {
            let carried = self.tag_arity(*tag_index)?;
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
                    results: signature.results.clone(),
                    returns_argument: self
                        .functions
                        .get(function_index)
                        .is_some_and(|info| info.returns_first_argument),
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
                    results: signature.results.clone(),
                    returns_argument: false,
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
    let mut imported_tags: BTreeMap<(&str, &str, (u32, u32)), u32> = BTreeMap::new();
    let mut defined = 0u32;
    let mut unreadable = false;
    let mut referenced = 0u64;
    let mut named_stack_pointer = None;
    let mut candidates: BTreeMap<u32, Candidates> = BTreeMap::new();
    let mut effects: BTreeMap<u32, Effects> = BTreeMap::new();
    let mut taken: BTreeSet<u32> = BTreeSet::new();
    let mut imported_globals = 0u32;
    let mut forwarding: Vec<(u32, Vec<Forward>)> = Vec::new();
    let mut switches: BTreeMap<u32, Switch> = BTreeMap::new();
    let mut resuming: BTreeSet<u32> = BTreeSet::new();

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
                    let start = module.types.len() as u32;
                    let absolute = |index: wasmparser::UnpackedIndex| {
                        index.as_module_index().or_else(|| {
                            index
                                .as_rec_group_index()
                                .and_then(|offset| start.checked_add(offset))
                        })
                    };
                    for ty in group.into_types() {
                        let index = module.types.len() as u32;
                        match &ty.composite_type.inner {
                            wasmparser::CompositeInnerType::Struct(fields) => {
                                module.structs.insert(index, fields.fields.len() as u32);
                            }
                            wasmparser::CompositeInnerType::Cont(cont) => {
                                if let Some(function) = absolute(cont.0.unpack()) {
                                    module.continuations.insert(index, function);
                                }
                            }
                            wasmparser::CompositeInnerType::Func(function) => {
                                if let Some(wasmparser::ValType::Ref(last)) =
                                    function.params().last()
                                    && let wasmparser::HeapType::Concrete(target)
                                    | wasmparser::HeapType::Exact(target) = last.heap_type()
                                    && let Some(target) = absolute(target)
                                {
                                    module.last_param_types.insert(index, target);
                                }
                            }
                            _ => {}
                        }
                        let func = func_type_of(&ty);
                        module.types.push(func.as_ref().map(signature_of));
                        module.value_types.push(func);
                    }
                }
            }
            Payload::ImportSection(reader) => {
                let mut imports = Vec::new();
                for import in reader.into_imports() {
                    match import {
                        Ok(import) => imports.push(import),
                        Err(_) => {
                            unreadable = true;
                            break;
                        }
                    }
                }
                for import in imports {
                    if let TypeRef::Tag(tag) = import.ty {
                        // An imported tag takes an index before any declared one
                        let shape = module.declare_tag(next_tag, tag.func_type_idx);
                        module.tag_names.insert(
                            next_tag,
                            format!("{}.{}", clean(import.module), clean(import.name)),
                        );
                        if let Some(shape) = shape {
                            let first = *imported_tags
                                .entry((import.module, import.name, shape))
                                .or_insert(next_tag);
                            if first != next_tag {
                                module.tag_aliases.insert(next_tag, first);
                            }
                        }
                        next_tag += 1;
                    }
                    // An imported table takes index 0, ahead of any declared one
                    if let TypeRef::Table(table) = import.ty {
                        module.table_imported |= module.tables.is_empty();
                        module.tables.push(table.initial);
                    }
                    if let TypeRef::Memory(memory) = import.ty {
                        module.add_memory(&memory, true);
                    }
                    if let TypeRef::Global(global) = import.ty {
                        module.global_names.insert(
                            next_global,
                            format!("{}.{}", clean(import.module), clean(import.name)),
                        );
                        if import.name == STACK_POINTER {
                            named_stack_pointer = Some(next_global);
                        }
                        // An imported global takes an index before any declared one
                        module.globals.insert(
                            next_global,
                            GlobalInfo {
                                kind: ValueKind::of(&global.content_type),
                                mutable: global.mutable,
                                init: None,
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
                imported_globals = next_global;
            }
            Payload::FunctionSection(reader) => {
                signatures.extend(reader.into_iter().flatten());
            }
            Payload::TagSection(reader) => {
                for tag in reader.into_iter().flatten() {
                    module.declare_tag(next_tag, tag.func_type_idx);
                    next_tag += 1;
                }
            }
            Payload::MemorySection(reader) => {
                for memory in reader.into_iter().flatten() {
                    module.add_memory(&memory, false);
                }
            }
            Payload::TableSection(reader) => {
                for table in reader.into_iter().flatten() {
                    module.tables.push(table.ty.initial);
                    if let wasmparser::TableInit::Expr(expr) = &table.init {
                        taken.extend(functions_taken(expr));
                    }
                }
            }
            // What a `call_indirect` selects from: an index no active segment filled traps rather
            // than calling anything
            Payload::ElementSection(reader) => {
                for element in reader.into_iter().flatten() {
                    match &element.items {
                        wasmparser::ElementItems::Functions(items) => {
                            taken.extend(items.clone().into_iter().flatten());
                        }
                        wasmparser::ElementItems::Expressions(_, items) => {
                            for expr in items.clone().into_iter().flatten() {
                                taken.extend(functions_taken(&expr));
                            }
                        }
                    }
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
                        let slot = at.saturating_add(nth as u64);
                        if slot >= MAX_TABLE_SLOTS {
                            break;
                        }
                        match function {
                            Some(function) => module.elements.insert(slot, function),
                            None => module.elements.remove(&slot),
                        };
                    }
                }
            }
            Payload::GlobalSection(reader) => {
                for global in reader.into_iter().flatten() {
                    taken.extend(functions_taken(&global.init_expr));
                    module.globals.insert(
                        next_global,
                        GlobalInfo {
                            kind: ValueKind::of(&global.ty.content_type),
                            mutable: global.ty.mutable,
                            init: const_value(&global.init_expr),
                        },
                    );
                    next_global += 1;
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader.into_iter().flatten() {
                    match export.kind {
                        wasmparser::ExternalKind::Func => {
                            module.exports.insert(export.index, clean(export.name));
                        }
                        wasmparser::ExternalKind::Table => {
                            module.table_exported |= export.index == 0;
                        }
                        wasmparser::ExternalKind::Tag => {
                            module
                                .tag_names
                                .entry(export.index)
                                .or_insert_with(|| clean(export.name));
                        }
                        _ => {}
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
                let scanned = scan(reader.clone());
                referenced = referenced.max(scanned.furthest);
                module.table_written |= scanned.writes_table;
                let index = next_index.saturating_add(ordinal);
                effects.insert(index, scanned.effects);
                if let Some(switch) = scanned.switch {
                    switches.insert(index, switch);
                }
                if scanned.resumes {
                    resuming.insert(index);
                }
                if !scanned.entry.is_empty() || !scanned.matched.is_empty() {
                    candidates.insert(
                        index,
                        Candidates {
                            reader: reader.clone(),
                            entry: scanned.entry,
                            matched: scanned.matched,
                        },
                    );
                }
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

                let hands_back = signature
                    .params
                    .first()
                    .is_some_and(|first| signature.results == [*first]);
                let returns_first_argument = match scanned.returned {
                    Some(forwards) if hands_back && !forwards.is_empty() => {
                        forwarding.push((index, forwards));
                        false
                    }
                    Some(_) => hands_back,
                    None => false,
                };
                let info = FunctionInfo {
                    start: base + body.range().start,
                    entry: base + reader.original_position(),
                    end: base + body.range().end,
                    signature,
                    frame: None,
                    balanced: false,
                    returns_first_argument,
                };

                module.bodies.push((info.start, info.end, index));
                module.functions.insert(index, info);
            }
            Payload::End(_) => break,
            _ => {}
        }
        if unreadable {
            break;
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

    module.settle_returned_arguments(&forwarding);
    module.address_taken = taken;
    module.name_go_registers();
    module.name_go_parameters(&resuming);
    module.stack_pointer = module.find_stack_pointer(named_stack_pointer, &candidates);
    if let Some(pointer) = module.stack_pointer {
        let candidates = candidates
            .into_iter()
            .filter_map(|(index, found)| {
                let mut saves: Vec<Save> = found
                    .entry
                    .into_iter()
                    .chain(found.matched)
                    .filter(|save| save.global == pointer)
                    .collect();
                saves.sort_by_key(|save| save.size == 0);
                saves.dedup();
                (!saves.is_empty()).then_some((index, (found.reader, saves)))
            })
            .collect();
        let kind = |global: u32| {
            module
                .globals
                .get(&global)
                .filter(|info| info.mutable)
                .map(|info| info.kind)
        };
        let aliases = if pointer < imported_globals {
            (0..imported_globals)
                .filter(|global| *global != pointer && kind(*global) == kind(pointer))
                .collect()
        } else {
            BTreeSet::new()
        };
        let frames = crate::frame::frames(
            &module,
            pointer,
            candidates,
            &effects,
            module.address_taken.clone(),
            &aliases,
            asyncify_state(&switches, &effects),
        );
        for (index, (frame, balanced)) in frames {
            if let Some(info) = module.functions.get_mut(&index) {
                info.frame = Some(frame);
                info.balanced = balanced;
            }
        }
    }

    referenced = referenced.max(crate::debug::memory_reach(image, &module));
    module.memory_extent = module.static_extent(referenced);

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

        if let Operator::MemoryInit { data_index, mem } = insn.op {
            let length = stack.pop().flatten();
            let offset = stack.pop().flatten();
            let dest = stack.pop().flatten();
            if mem != 0 || length == Some(0) {
                continue;
            }
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

struct Candidates<'a> {
    reader: wasmparser::OperatorsReader<'a>,
    entry: Vec<Save>,
    matched: Vec<Save>,
}

struct Scan {
    furthest: u64,
    writes_table: bool,
    entry: Vec<Save>,
    matched: Vec<Save>,
    effects: Effects,
    returned: Option<Vec<Forward>>,
    switch: Option<Switch>,
    resumes: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Switch {
    state: u32,
    data: u32,
    to: i32,
    carries_data: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Forward {
    callee: u32,
    pushed: u32,
    first: u64,
}

impl Forward {
    fn passes_first(&self, params: usize) -> bool {
        params
            .checked_sub(1)
            .and_then(|below| u32::try_from(below).ok())
            .is_some_and(|below| below < self.pushed && self.first >> below & 1 == 1)
    }
}

#[derive(Default)]
struct FirstLocalReturned {
    depth: u32,
    broken: bool,
    after_get: bool,
    after_exit: bool,
    ended: bool,
    pushed: u32,
    first: u64,
    called: Option<Forward>,
    forwards: Vec<Forward>,
}

impl FirstLocalReturned {
    fn forward(&self, callee: u32) -> Forward {
        Forward {
            callee,
            pushed: self.pushed,
            first: self.first,
        }
    }

    fn leaves(&mut self, called: Option<Forward>) {
        match called {
            _ if self.after_get => {}
            Some(forward) => self.forwards.push(forward),
            None => self.broken = true,
        }
    }

    fn step(&mut self, op: &Operator) {
        let leaving = |relative_depth: u32| relative_depth == self.depth;
        let called = self.called.take();
        match op {
            Operator::LocalSet { local_index: 0 }
            | Operator::LocalTee { local_index: 0 }
            | Operator::ReturnCallIndirect { .. }
            | Operator::ReturnCallRef { .. } => self.broken = true,
            Operator::ReturnCall { function_index } => {
                self.forwards.push(self.forward(*function_index));
            }
            Operator::Call { function_index } => self.called = Some(self.forward(*function_index)),
            Operator::Return => self.leaves(called),
            Operator::Br { relative_depth } if leaving(*relative_depth) => self.leaves(called),
            op if crate::insn::conditional_label(op).is_some_and(leaving) => {
                self.broken = true;
            }
            Operator::BrTable { targets } => {
                self.broken |= leaving(targets.default())
                    || targets.targets().any(|target| target.map_or(true, leaving));
            }
            Operator::TryTable { try_table } => {
                self.broken |= try_table
                    .catches
                    .iter()
                    .any(|catch| leaving(crate::insn::catch_label(catch)));
                self.depth += 1;
            }
            Operator::Block { .. }
            | Operator::Loop { .. }
            | Operator::If { .. }
            | Operator::Try { .. } => self.depth += 1,
            Operator::End if self.depth == 0 => {
                if !self.after_exit {
                    self.leaves(called);
                }
                self.ended = true;
            }
            Operator::End | Operator::Delegate { .. } => self.depth = self.depth.saturating_sub(1),
            _ => {
                if let Some(table) = crate::insn::resume_table(op) {
                    self.broken |= table.handlers.iter().any(|handle| {
                        matches!(*handle, wasmparser::Handle::OnLabel { label, .. } if leaving(label))
                    });
                }
            }
        }
        self.after_get = matches!(op, Operator::LocalGet { local_index: 0 });
        self.after_exit = !crate::insn::flow(op).falls_through();
        if crate::insn::arity(op).is_some_and(|arity| arity.pops == 0 && arity.pushes == 1) {
            self.first = self.first << 1 | u64::from(self.after_get);
            self.pushed = (self.pushed + 1).min(u64::BITS);
        } else {
            (self.pushed, self.first) = (0, 0);
        }
    }

    fn forwards(self) -> Option<Vec<Forward>> {
        (self.ended && !self.broken).then_some(self.forwards)
    }
}

fn scan(mut reader: wasmparser::OperatorsReader) -> Scan {
    let mut scan = Scan {
        furthest: 0,
        writes_table: false,
        entry: Vec::new(),
        matched: Vec::new(),
        effects: Effects::default(),
        returned: None,
        switch: None,
        resumes: false,
    };
    let mut returned = FirstLocalReturned::default();
    let mut head = Vec::new();
    let mut first_read: Option<Option<bool>> = None;
    let mut constant = None;
    let mut entry = Some(EntryRegion::default());
    let mut saves = Saves::default();
    let mut at = 0;
    let (mut null, mut null_then_count) = (false, false);
    while !reader.eof() {
        let Ok(op) = reader.read() else {
            scan.effects.unread();
            break;
        };
        let grows_null = null_then_count && matches!(op, Operator::TableGrow { .. });
        null_then_count = null
            && matches!(
                op,
                Operator::I32Const { .. }
                    | Operator::I64Const { .. }
                    | Operator::LocalGet { .. }
                    | Operator::GlobalGet { .. }
            );
        null = matches!(op, Operator::RefNull { .. });
        scan.writes_table |= !grows_null
            && matches!(
                op,
                Operator::TableSet { table: 0 }
                    | Operator::TableFill { table: 0 }
                    | Operator::TableGrow { table: 0 }
                    | Operator::TableCopy { dst_table: 0, .. }
                    | Operator::TableInit { table: 0, .. }
                    | Operator::TableAtomicSet { table_index: 0, .. }
                    | Operator::TableAtomicRmwXchg { table_index: 0, .. }
                    | Operator::TableAtomicRmwCmpxchg { table_index: 0, .. }
            );
        if let Some(MemArg {
            offset, memory: 0, ..
        }) = crate::insn::memarg(&op)
        {
            let base = constant
                .filter(|_| crate::insn::arity(&op).is_some_and(|arity| arity.pops == 1))
                .unwrap_or(0);
            scan.furthest = scan
                .furthest
                .max(offset.saturating_add(base).saturating_add(16));
        }
        constant = match op {
            Operator::I32Const { value } => Some(value as u32 as u64),
            Operator::I64Const { value } => Some(value as u64),
            _ => None,
        };
        if let Some(region) = &mut entry
            && !region.step(&op, at, &mut scan.entry)
        {
            entry = None;
        }
        saves.step(&op, at, &mut scan.matched);
        scan.effects.step(&op);
        at += 1;
        returned.step(&op);
        first_read = match first_read {
            None => matches!(op, Operator::LocalGet { local_index: 0 }).then_some(None),
            Some(None) => Some(Some(matches!(op, Operator::BrTable { .. }))),
            decided => decided,
        };
        if head.len() <= SWITCH_LENGTH {
            head.push(op);
        }
    }
    scan.returned = returned.forwards();
    scan.switch = asyncify_switch(&head);
    scan.resumes = matches!(first_read, None | Some(Some(true)));
    scan
}

const SWITCH_LENGTH: usize = 13;

fn asyncify_switch(ops: &[Operator]) -> Option<Switch> {
    let [
        Operator::I32Const { value: to },
        Operator::GlobalSet {
            global_index: state,
        },
        rest @ ..,
    ] = ops
    else {
        return None;
    };
    let (carried, rest) = match rest {
        [
            Operator::LocalGet { local_index: 0 },
            Operator::GlobalSet { global_index },
            rest @ ..,
        ] => (Some(*global_index), rest),
        _ => (None, rest),
    };
    let [
        Operator::GlobalGet { global_index: data },
        position,
        Operator::GlobalGet {
            global_index: again,
        },
        end,
        compare,
        Operator::If {
            blockty: wasmparser::BlockType::Empty,
        },
        Operator::Unreachable,
        Operator::End,
        Operator::End,
    ] = rest
    else {
        return None;
    };
    let bounded = match (position, end, compare) {
        (
            Operator::I32Load { memarg: first },
            Operator::I32Load { memarg: second },
            Operator::I32GtU,
        ) => (first.offset, second.offset) == (0, 4),
        (
            Operator::I64Load { memarg: first },
            Operator::I64Load { memarg: second },
            Operator::I64GtU,
        ) => (first.offset, second.offset) == (0, 8),
        _ => false,
    };
    (bounded && data == again && carried.is_none_or(|carried| carried == *data)).then_some(Switch {
        state: *state,
        data: *data,
        to: *to,
        carries_data: carried.is_some(),
    })
}

fn asyncify_state(
    switches: &BTreeMap<u32, Switch>,
    effects: &BTreeMap<u32, Effects>,
) -> Option<u32> {
    let starts = |to: i32| -> BTreeSet<(u32, u32)> {
        switches
            .values()
            .filter(|switch| switch.carries_data && switch.to == to)
            .map(|switch| (switch.state, switch.data))
            .collect()
    };
    let unwinding = starts(1);
    let rewinding = starts(2);
    let mut both = unwinding.intersection(&rewinding);
    let (Some((state, _)), None) = (both.next(), both.next()) else {
        return None;
    };
    effects
        .iter()
        .filter(|(_, effect)| effect.writes(*state))
        .all(|(index, _)| {
            switches
                .get(index)
                .is_some_and(|switch| switch.state == *state)
        })
        .then_some(*state)
}

/// The constant an initialiser expression is, for the ones that are a constant at all
fn const_value(expr: &wasmparser::ConstExpr) -> Option<u64> {
    let mut stack: Vec<u64> = Vec::new();
    let mut reader = expr.get_operators_reader();
    loop {
        let (width, operator): (u32, fn(u64, u64) -> u64) = match reader.read().ok()? {
            Operator::I32Const { value } => {
                stack.push(value as u32 as u64);
                continue;
            }
            Operator::I64Const { value } => {
                stack.push(value as u64);
                continue;
            }
            Operator::End => return (stack.len() == 1).then(|| stack[0]),
            Operator::I32Add => (32, u64::wrapping_add),
            Operator::I32Sub => (32, u64::wrapping_sub),
            Operator::I32Mul => (32, u64::wrapping_mul),
            Operator::I64Add => (64, u64::wrapping_add),
            Operator::I64Sub => (64, u64::wrapping_sub),
            Operator::I64Mul => (64, u64::wrapping_mul),
            _ => return None,
        };
        let right = stack.pop()?;
        let left = stack.pop()?;
        let value = operator(left, right);
        stack.push(if width == 32 {
            value as u32 as u64
        } else {
            value
        });
    }
}

fn functions_taken(expr: &wasmparser::ConstExpr) -> Vec<u32> {
    expr.get_operators_reader()
        .into_iter()
        .filter_map(|op| match op.ok()? {
            Operator::RefFunc { function_index } => Some(function_index),
            _ => None,
        })
        .collect()
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
    for subsection in reader {
        let Ok(subsection) = subsection else {
            break;
        };
        match subsection {
            Name::Module { name, .. } => {
                module.name = Some(clean(name)).filter(|name| !name.is_empty())
            }
            Name::Function(names) => collect(names, &mut module.names),
            Name::Local(functions) => collect_indirect(functions, &mut module.locals),
            Name::Global(names) => collect(names, &mut module.global_names),
            Name::Data(names) => collect(names, &mut module.data_names),
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
            // element segment has no address either, so none of them has anywhere to go here
            Name::Label(_)
            | Name::Field(_)
            | Name::Element(_)
            | Name::Memory(_)
            | Name::Table(_)
            | Name::Unknown { .. } => {}
        }
    }
}

fn collect(names: wasmparser::NameMap, into: &mut BTreeMap<u32, String>) {
    for naming in names.into_iter().flatten() {
        let name = clean(naming.name);
        if !name.is_empty() {
            into.insert(naming.index, name);
        }
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

fn func_type_of(ty: &wasmparser::SubType) -> Option<wasmparser::FuncType> {
    match &ty.composite_type.inner {
        wasmparser::CompositeInnerType::Func(func) => Some(func.clone()),
        _ => None,
    }
}

fn signature_of(func: &wasmparser::FuncType) -> Signature {
    Signature {
        params: func.params().iter().map(ValueKind::of).collect(),
        results: func.results().iter().map(ValueKind::of).collect(),
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

        let wide = parse(&build(r#"(module (memory i64 1) (func nop))"#), 0).expect("parses");
        assert!(wide.memory64, "declared i64 memory");

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

        assert_eq!(
            module.local_kinds(0),
            [
                ValueKind::I32,
                ValueKind::I64,
                ValueKind::F64,
                ValueKind::I32,
                ValueKind::I32
            ],
            "parameters first, then the declared locals"
        );
        assert!(module.local_kinds(1).is_empty(), "no such function");
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
    fn a_signature_keeps_the_reference_types_its_kinds_leave_out() {
        let module = parse(
            &build(
                "(module (type $t (struct)) \
                   (import \"env\" \"f\" (func (param externref) (result (ref null $t)))) \
                   (func (param i32 (ref $t)) (result funcref) (ref.null func)))",
            ),
            0,
        )
        .expect("parses");
        let texts = |function: u32| {
            let values = module.value_types(function).expect("a function type");
            let text = |types: &[wasmparser::ValType]| {
                types
                    .iter()
                    .map(crate::insn::val_type_text)
                    .collect::<Vec<_>>()
            };
            (text(values.params()), text(values.results()))
        };

        assert_eq!(
            texts(0),
            (
                vec!["externref".to_owned()],
                vec!["(ref null 0)".to_owned()]
            )
        );
        assert_eq!(
            texts(1),
            (
                vec!["i32".to_owned(), "(ref 0)".to_owned()],
                vec!["funcref".to_owned()]
            )
        );
        assert_eq!(
            module
                .signature(1)
                .map(|signature| signature.params.clone()),
            Some(vec![ValueKind::I32, ValueKind::Ref]),
            "the kinds still what a call moves"
        );
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
        assert_eq!(module.declared_local_names(1, 2).count(), 0);
    }

    #[test]
    fn declared_locals_are_named_past_the_parameters() {
        let image =
            build(r#"(module (func $f (param $p i32) (local $acc i32) (local $i i64) nop))"#);
        let module = parse(&image, 0).expect("parses");
        assert_eq!(
            module.declared_local_names(0, 1).collect::<Vec<_>>(),
            [(1, "acc"), (2, "i")]
        );
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
    fn an_unnamed_data_segment_is_not_named_like_an_address() {
        let image = build(
            r#"(module (memory 1)
                 (data $strings (i32.const 16) "abc")
                 (data (i32.const 7) "d")
                 (func nop))"#,
        );
        let module = parse(&image, 0).expect("parses");
        assert_eq!(module.data_name(0), "strings", "from the name section");
        assert_eq!(
            module.data_name(1),
            "data_segment_1",
            "not data_1, which reads as the byte at address 1"
        );
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
    fn a_host_memory_keeps_the_addresses_it_may_have_clear_of_the_file() {
        let shape = |memory: &str| {
            let image = build(&format!("(module {memory} (func nop))"));
            parse(&image, 0).expect("parses").shape()
        };

        let open = shape(r#"(import "env" "memory" (memory 0))"#);
        assert_eq!(
            open.reserved, HOST_MEMORY,
            "a host's memory of any size, short of where a negative number points"
        );
        assert!(Layout::allocate(open).file_base >= HOST_MEMORY);
        assert_eq!(
            shape(r#"(import "env" "memory" (memory 0 16))"#).reserved,
            16 * PAGE,
            "no further than the most it may grow to"
        );
        assert_eq!(
            shape("(memory 0 16)").reserved,
            0,
            "a memory of its own starts as it declares"
        );
    }

    #[test]
    fn no_layout_overlaps_itself() {
        let shapes = [
            Shape::default(),
            Shape {
                image: 1,
                memory: 0,
                reserved: 0,
                globals: 0,
                imports: 0,
                tables: 0,
                tags: 0,
                memory64: false,
            },
            // The most a wasm32 module can declare, which leaves no room at all
            Shape {
                image: 4 << 20,
                memory: 1 << 32,
                reserved: 1 << 32,
                globals: 4096,
                imports: 8192,
                tables: 65536,
                tags: 0,
                memory64: false,
            },
            // The same past what a wasm32 address can hold, which only memory64 reaches
            Shape {
                image: MAX_IMAGE_LEN as u64,
                memory: 1 << 40,
                reserved: 1 << 40,
                globals: 1 << 20,
                imports: 1 << 20,
                tables: 1 << 20,
                tags: 0,
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
    fn the_stack_pointer_is_the_global_functions_take_their_frames_from() {
        let prologue = |global: u32| {
            format!(
                "(func (local i32) (global.get {global}) (i32.const 48) (i32.sub) (local.tee 0) \
                 (global.set {global}) (global.set {global} (i32.add (local.get 0) (i32.const 48))))"
            )
        };
        let stripped = format!(
            "(module (memory 1) (global (mut i32) (i32.const 0)) (global (mut i32) (i32.const 4096)) {} {})",
            prologue(1),
            prologue(1)
        );
        let module = parse(&build(&stripped), 0).expect("parses");
        assert_eq!(module.stack_pointer, Some(1), "no names, only the idiom");
        let (_, info) = module.functions().next().expect("a body");
        assert_eq!(info.frame, Some((0, 48)));

        let named =
            r#"(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) (func))"#;
        assert_eq!(
            parse(&build(named), 0).expect("parses").stack_pointer,
            Some(0)
        );

        let imported =
            r#"(module (import "env" "__stack_pointer" (global (mut i32))) (memory 1) (func))"#;
        let module = parse(&build(imported), 0).expect("parses");
        assert_eq!(module.stack_pointer, Some(0));
        assert_eq!(module.global_name(0), "env.__stack_pointer");

        let counter = "(module (global (mut i32) (i32.const 9)) \
             (func (global.set 0 (i32.sub (global.get 0) (i32.const 1)))))";
        assert_eq!(
            parse(&build(counter), 0).expect("parses").stack_pointer,
            None,
            "a count going down is not a frame being taken"
        );

        let unoptimised = "(module (memory 1) (global (mut i32) (i32.const 4096)) \
             (func (local i32) (local.set 0 (i32.sub (global.get 0) (i32.const 32))) \
               (global.set 0 (local.get 0)) (global.set 0 (i32.add (local.get 0) (i32.const 32)))) \
             (func (local i32 i32) (local.set 1 (i32.sub (global.get 0) (i32.const 16)))))";
        let module = parse(&build(unoptimised), 0).expect("parses");
        assert_eq!(module.stack_pointer, Some(0));
        let frames: Vec<_> = module.functions().map(|(_, info)| info.frame).collect();
        assert_eq!(
            frames,
            [Some((0, 32)), Some((1, 16))],
            "a leaf takes its frame without moving the pointer"
        );

        let spelled = |body: &str| {
            let text = format!(
                "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
                 (func (local i32 i32) {body}))"
            );
            let module = parse(&build(&text), 0).expect("parses");
            let (_, info) = module.functions().next().expect("a body").to_owned();
            info.frame
        };
        assert_eq!(
            spelled("(local.set 0 (i32.add (global.get 0) (i32.const -16)))"),
            Some((0, 16)),
            "an addition of a negative size takes the same frame"
        );
        assert_eq!(
            spelled(
                "(local.set 0 (global.get 0)) \
                 (local.set 1 (i32.sub (local.get 0) (i32.const 48))) (global.set 0 (local.get 1)) \
                 (global.set 0 (local.get 0))"
            ),
            Some((1, 48)),
            "the pointer saved first, the frame taken from the copy"
        );
        assert_eq!(
            spelled(
                "(local.set 0 (i32.and (i32.sub (global.get 0) (i32.const 32)) (i32.const -16)))"
            ),
            None,
            "a realigned frame has no size known ahead of time"
        );

        let late = "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
             (func (local i32) (if (i32.const 1) (then \
               (local.set 0 (i32.sub (global.get 0) (i32.const 16))) \
               (global.set 0 (local.get 0))))))";
        let module = parse(&build(late), 0).expect("parses");
        let (_, info) = module.functions().next().expect("a body");
        assert_eq!(
            info.frame, None,
            "a frame taken on one path only leaves the others where they were"
        );

        let tied = format!(
            "(module (memory 1) (global (mut i32) (i32.const 0)) (global (mut i32) (i32.const 0)) {} {})",
            prologue(0),
            prologue(1)
        );
        assert_eq!(parse(&build(&tied), 0).expect("parses").stack_pointer, None);

        let immutable = r#"(module (global $__stack_pointer i32 (i32.const 4096)) (func))"#;
        assert_eq!(
            parse(&build(immutable), 0).expect("parses").stack_pointer,
            None
        );
    }

    #[test]
    fn a_frame_is_kept_where_every_way_out_gives_the_pointer_back() {
        let framed = |body: &str| {
            let text = format!(
                "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
                 (func $abort) (func $zero (result i32) (i32.const 0)) \
                 (func $fill (param i32) (result i32) (local.get 0)) \
                 (func $other (param i32) (result i32) (i32.add (local.get 0) (i32.const 4))) \
                 (func (param i32) (result i32) (local i32 i32) \
                   (local.set 1 (i32.sub (global.get 0) (i32.const 32))) (global.set 0 (local.get 1)) \
                   {body}))"
            );
            let module = parse(&build(&text), 0).expect("parses");
            module.function(4).expect("a body").frame
        };
        let restore = "(global.set 0 (i32.add (local.get 1) (i32.const 32)))";
        let inner =
            "(local.set 2 (i32.sub (global.get 0) (i32.const 16))) (global.set 0 (local.get 2))";
        let inner_restore = "(global.set 0 (i32.add (local.get 2) (i32.const 16)))";

        assert_eq!(
            framed(&format!(
                "(block (br_if 0 (local.get 0)) {restore} (return (i32.const 1))) {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "an early return that restores the pointer"
        );
        assert_eq!(
            framed(&format!(
                "{inner} (call $abort) {inner_restore} {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "an inlined frame given back before control meets anything"
        );
        assert_eq!(
            framed(&format!(
                "(local.set 2 (global.get 0)) (global.set 0 (i32.sub (global.get 0) (local.get 0))) \
                 (call $abort) (global.set 0 (local.get 2)) {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "an allocation of unknown size given back before control meets anything"
        );
        assert_eq!(
            framed(&format!("{restore} (return_call $zero)")),
            Some((1, 32)),
            "a tail call once the pointer is restored"
        );
        assert_eq!(
            framed(&format!(
                "try call $abort catch_all end {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "a handler resumes at the frame"
        );
        assert_eq!(
            framed(&format!(
                "(block $caught (try_table (catch_all $caught) (call $abort))) {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "a handler label reached at the frame"
        );
        assert_eq!(
            framed(&format!(
                "try call $abort catch_all (global.set 0 (local.get 1)) (local.set 1 (i32.const 7)) \
                 (if (local.get 0) (then (call $abort))) rethrow 0 end {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "a handler that reuses the frame local once it has restored the pointer"
        );
        assert_eq!(
            framed(&format!(
                "loop try call $abort (br_if 1 (local.get 0)) catch_all (global.set 0 (local.get 1)) \
                 (local.set 1 (i32.const 7)) (if (local.get 0) (then (call $abort))) rethrow 0 end end \
                 {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "a loop whose handler reuses the frame local"
        );
        assert_eq!(
            framed(&format!(
                "(local.set 2 (i32.const 0)) (loop (local.set 2 (i32.add (local.get 2) (i32.const 1))) \
                 (br_if 0 (i32.lt_u (local.get 2) (i32.const 10)))) {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "a counter that changes on every iteration"
        );
        assert_eq!(
            framed(&format!(
                "(local.set 1 (call $fill (local.get 1))) {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "the frame handed back by a callee that returns its first argument"
        );
        assert_eq!(
            framed(&format!(
                "(local.set 2 (local.get 1)) (loop (local.set 2 (i32.sub (local.get 2) (i32.const 16))) \
                 (global.set 0 (local.get 2)) (br_if 0 (local.get 0))) {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "a pointer carried around a loop to a deeper frame each time"
        );
        assert_eq!(
            framed(&format!(
                "(local.set 2 (local.get 1)) (loop (global.set 0 (local.get 2)) \
                 (local.set 2 (i32.load (i32.const 8))) (br_if 0 (local.get 0))) {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "a pointer taken from a local the loop overwrites"
        );
        assert_eq!(
            framed(&format!(
                "(block (br_if 0 (local.get 0)) {inner} (br_if 0 (local.get 0)) {inner_restore}) \
                 {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "a path shared with an inlined frame still open"
        );
        assert_eq!(
            framed(&format!(
                "(loop (global.set 0 (i32.sub (global.get 0) (local.get 0))) (br_if 0 (local.get 0))) \
                 {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "an allocation on every iteration"
        );
        assert_eq!(
            framed(&format!(
                "(if (local.get 0) (then (global.set 0 (i32.load (i32.const 8))))) {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "the pointer replaced on one path only"
        );
        assert_eq!(
            framed(&format!(
                "{inner} try call $abort catch_all end {inner_restore} {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "a handler resuming at the frame while an inlined frame is open"
        );
        assert_eq!(
            framed(&format!(
                "(if (i32.eqz (i32.const 5)) (then (return (i32.const 1)))) {restore} (i32.const 0)"
            )),
            Some((1, 32)),
            "a way out a constant condition never takes"
        );
        assert_eq!(
            framed(&format!(
                "(drop (br_if 0 (i32.const 1) (i32.ne (i32.const 3) (i32.const 3)))) {restore} \
                 (i32.const 0)"
            )),
            Some((1, 32)),
            "a branch out a constant condition never takes"
        );

        assert_eq!(
            framed(&format!(
                "(local.set 1 (call $other (local.get 1))) {restore} (i32.const 0)"
            )),
            None,
            "the frame replaced by what some other callee returns"
        );
        assert_eq!(
            framed(&format!(
                "(if (i32.or (local.get 0) (i32.const 2)) (then (return (i32.const 1)))) {restore} \
                 (i32.const 0)"
            )),
            None,
            "a way out a condition known to hold always takes"
        );
        let release = "(global.set 0 (i32.add (global.get 0) (i32.const 32)))";
        assert_eq!(
            framed(&format!(
                "(if (local.get 0) (then {inner})) {release} (i32.const 0)"
            )),
            None,
            "a pointer given back from wherever paths at different depths left it"
        );
        assert_eq!(
            framed(&format!(
                "(loop (global.set 0 (i32.sub (global.get 0) (i32.const 16))) (br_if 0 (local.get 0))) \
                 {release} (i32.const 0)"
            )),
            None,
            "a pointer given back from wherever a loop that goes deeper each time left it"
        );
        assert_eq!(
            framed("(i32.const 0)"),
            None,
            "a return with the frame still taken"
        );
        assert_eq!(
            framed("(return_call $zero)"),
            None,
            "a tail call with the frame still taken"
        );
    }

    #[test]
    fn a_pointer_only_saved_for_the_handlers_is_a_frame_of_nothing() {
        let framed = |body: &str| {
            let text = format!(
                "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
                 (func $abort) \
                 (func (param i32) (result i32) (local i32) \
                   (local.set 1 (global.get 0)) {body}) \
                 (func (result i32) (local i32) (local.set 0 (global.get 0)) \
                   (drop (call 1 (i32.const 0))) (drop (global.get 0)) (i32.const 0)))"
            );
            let module = parse(&build(&text), 0).expect("parses");
            let frame = |index| module.function(index).expect("a body").frame;
            (frame(1), frame(2))
        };
        let balanced = (Some((1, 0)), Some((0, 0)));
        let moving = (Some((1, 0)), None);

        assert_eq!(
            framed(
                "try call $abort catch_all (global.set 0 (local.get 1)) rethrow 0 end (i32.const 0)"
            ),
            balanced,
            "every handler restores the pointer before it goes anywhere"
        );
        assert_eq!(
            framed(
                "(block $caught (try_table (catch_all $caught) (call $abort))) \
                 (global.set 0 (local.get 1)) (i32.const 0)"
            ),
            balanced,
            "a handler label that restores the pointer"
        );
        assert_eq!(
            framed("try call $abort catch_all end (i32.const 0)"),
            moving,
            "a handler that returns with whatever the throw left in the pointer"
        );
        assert_eq!(
            framed(
                "try call $abort catch_all (drop (global.get 0)) (global.set 0 (local.get 1)) end \
                 (i32.const 0)"
            ),
            (None, None),
            "a handler that reads the pointer the throw left"
        );
        assert_eq!(
            framed(
                "(block $out (loop $again try (br_if $out (local.get 0)) call $abort catch_all end \
                   (br $again))) (i32.const 0)"
            ),
            moving,
            "a handler that loops back with the pointer the throw left, then returns"
        );
    }

    #[test]
    fn a_pointer_saved_past_the_first_branch_is_a_frame_once_it_is_the_frame_it_claims() {
        let framed = |body: &str| {
            let text = format!(
                "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
                 (func $abort) \
                 (func (param i32) (result i32) (local i32) {body} (i32.const 0)))"
            );
            let module = parse(&build(&text), 0).expect("parses");
            module.function(1).expect("a body").frame
        };

        assert_eq!(
            framed(
                "(if (local.get 0) (then (local.set 1 (global.get 0)) \
                   try call $abort catch_all (global.set 0 (local.get 1)) rethrow 0 end))"
            ),
            Some((1, 0)),
            "a save only the path that can throw makes"
        );
        assert_eq!(
            framed(
                "(if (local.get 0) (then \
                   (local.set 1 (i32.sub (global.get 0) (i32.const 16))) (global.set 0 (local.get 1)) \
                   (call $abort) (global.set 0 (i32.add (local.get 1) (i32.const 16)))))"
            ),
            Some((1, 16)),
            "a frame only one path allocates"
        );
        assert_eq!(
            framed(
                "(if (local.get 0) (then \
                   (global.set 0 (local.tee 1 (i32.add (global.get 0) (i32.const -64)))) \
                   (call $abort) (global.set 0 (i32.sub (local.get 1) (i32.const -64)))))"
            ),
            Some((1, 64)),
            "a frame allocated by adding a negative size"
        );
        assert_eq!(
            framed(
                "(global.set 0 (i32.sub (global.get 0) (i32.const 16))) (local.set 1 (global.get 0)) \
                 (call $abort) (global.set 0 (i32.add (local.get 1) (i32.const 16)))"
            ),
            Some((1, 16)),
            "a save made once the pointer is moved, which is sixteen bytes into a frame"
        );
        assert_eq!(
            framed(
                "(local.set 1 (i32.const 5)) (local.set 1 (global.get 0)) \
                 try call $abort catch_all (global.set 0 (local.get 1)) rethrow 0 end"
            ),
            Some((1, 0)),
            "a local that held something else before it held the pointer"
        );
    }

    #[test]
    fn a_call_that_can_leave_the_pointer_moved_counts_as_moving_it() {
        let frames = |caller: &str| {
            let text = format!(
                "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
                 (func $restore (param i32) (global.set 0 (local.get 0))) \
                 (func $relay (call $restore (i32.const 64))) \
                 (func (param i32) (result i32) (local i32) {caller}) \
                 (func (result i32) (local i32) (local.set 0 (global.get 0)) \
                   (drop (call 2 (i32.const 0))) (drop (global.get 0)) (i32.const 0)))"
            );
            let module = parse(&build(&text), 0).expect("parses");
            let frame = |index| module.function(index).expect("a body").frame;
            (frame(2), frame(3))
        };
        let frame =
            "(local.set 1 (i32.sub (global.get 0) (i32.const 32))) (global.set 0 (local.get 1))";

        assert_eq!(
            frames(&format!(
                "{frame} (call $relay) (global.set 0 (i32.add (local.get 1) (i32.const 32))) \
                 (i32.const 0)"
            )),
            (Some((1, 32)), Some((0, 0))),
            "a frame restored from its own local after the call"
        );
        assert_eq!(
            frames(&format!(
                "{frame} (call $relay) (drop (global.get 0)) \
                 (global.set 0 (i32.add (local.get 1) (i32.const 32))) (i32.const 0)"
            )),
            (None, None),
            "the pointer read after the call"
        );
        assert_eq!(
            frames(&format!(
                "{frame} (global.set 0 (i32.add (local.get 1) (i32.const 32))) (return_call $relay)"
            )),
            (Some((1, 32)), None),
            "a tail call to it once the frame is given back, which moves it for the caller"
        );
        assert_eq!(
            frames(
                "(local.set 1 (global.get 0)) try (call $relay) catch_all \
                 (global.set 0 (local.get 1)) rethrow 0 end (i32.const 0)"
            ),
            (Some((1, 0)), None),
            "a pointer only saved for the handlers, returned from with the call's pointer"
        );
    }

    #[test]
    fn a_call_through_a_table_or_a_continuation_can_reach_a_mover() {
        let frame = |body: &str, extra: &str| {
            let text = format!(
                "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
                 (type $v (func)) (type $c (cont $v)) (table 1 funcref) \
                 (func $mover (global.set 0 (i32.sub (global.get 0) (i32.const 64)))) \
                 (func $safe) \
                 (func (local i32) \
                   (local.set 0 (i32.sub (global.get 0) (i32.const 16))) \
                   (global.set 0 (local.get 0)) {body}) {extra})"
            );
            let module = parse(&build(&text), 0).expect("parses");
            module.function(2).expect("a body").frame
        };
        let indirect = "(call_indirect (type $v) (i32.const 0)) \
             (global.set 0 (i32.add (global.get 0) (i32.const 16)))";

        assert_eq!(
            frame(indirect, "(elem (i32.const 0) $safe)"),
            Some((0, 16)),
            "a table holding nothing that moves it"
        );
        assert_eq!(
            frame(indirect, "(elem (i32.const 0) $mover)"),
            None,
            "a table holding one that does"
        );
        assert_eq!(
            frame(indirect, "(global funcref (ref.func $mover))"),
            None,
            "one taken as a reference anywhere"
        );
        let wide = "(func $wide (param i32) \
             (global.set 0 (i32.sub (global.get 0) (i32.const 64))))";
        assert_eq!(
            frame(indirect, &format!("{wide} (elem (i32.const 0) $wide)")),
            Some((0, 16)),
            "a table holding one of another type, which the call's type check keeps out"
        );
        assert_eq!(
            frame(
                "(call_ref $v (ref.func $mover)) \
                 (global.set 0 (i32.add (global.get 0) (i32.const 16)))",
                "(elem declare func $mover)"
            ),
            None,
            "a call through a reference to one"
        );
        assert_eq!(
            frame(
                "(global.set 0 (i32.add (local.get 0) (i32.const 16))) \
                 (return_call_indirect (type $v) (i32.const 0))",
                "(elem (i32.const 0) $mover)"
            ),
            Some((0, 16)),
            "a tail call through the table once the frame is given back"
        );
        assert_eq!(
            frame(
                "(call $resumer) (global.set 0 (i32.add (global.get 0) (i32.const 16)))",
                "(elem declare func $safe) \
                 (func $resumer (resume $c (cont.new $c (ref.func $safe))))"
            ),
            None,
            "a callee that runs a continuation"
        );
    }

    #[test]
    fn the_host_gives_the_pointer_back_whatever_it_can_reach() {
        let frame = |call: &str, exported: &str| {
            let export = |what: &str| {
                if exported.contains(what) {
                    format!("(export \"{what}\")")
                } else {
                    String::new()
                }
            };
            let text = format!(
                "(module (import \"env\" \"cb\" (func $cb)) (type $v (func)) (memory 1) \
                 (global $__stack_pointer {} (mut i32) (i32.const 4096)) \
                 (table {} 1 funcref) (elem (i32.const 0) $safe) \
                 (func $mover {} (global.set 0 (i32.sub (global.get 0) (i32.const 64)))) \
                 (func $safe) \
                 (func (param (ref null $v)) (local i32) \
                   (local.set 1 (i32.sub (global.get 0) (i32.const 16))) (global.set 0 (local.get 1)) \
                   {call} (global.set 0 (i32.add (global.get 0) (i32.const 16)))))",
                export("sp"),
                export("table"),
                export("mover"),
            );
            let module = parse(&build(&text), 0).expect("parses");
            module.function(3).expect("a body").frame
        };
        for exported in ["", "mover", "sp", "table mover"] {
            for call in [
                "(call $cb)",
                "(call_indirect (type $v) (i32.const 0))",
                "(call_ref $v (local.get 0))",
            ] {
                assert_eq!(
                    frame(call, exported),
                    Some((1, 16)),
                    "{call} with {exported:?} exported"
                );
            }
        }
    }

    #[test]
    fn another_import_of_the_right_type_may_be_the_imported_pointer() {
        let frame = |alias: &str| {
            let text = format!(
                "(module (import \"env\" \"__stack_pointer\" (global $sp (mut i32))) \
                 (import \"env\" \"other\" (global $alias {alias})) (memory 1) \
                 (func $g (global.set $alias (global.get $alias))) \
                 (func (local i32) \
                   (local.set 0 (i32.sub (global.get $sp) (i32.const 16))) (global.set $sp (local.get 0)) \
                   (call $g) (global.set $sp (i32.add (global.get $sp) (i32.const 16)))))"
            );
            let module = parse(&build(&text), 0).expect("parses");
            module.function(1).expect("a body").frame
        };
        assert_eq!(frame("(mut i32)"), None);
        assert_eq!(
            frame("(mut i64)"),
            Some((0, 16)),
            "a global of another type"
        );
    }

    #[test]
    fn a_function_that_catches_can_return_with_the_pointer_a_throw_left() {
        let frame = |absorber: &str| {
            let text = format!(
                "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) (tag $e) \
                 (func $thrower (local i32) \
                   (local.set 0 (i32.sub (global.get 0) (i32.const 16))) (global.set 0 (local.get 0)) \
                   (throw $e)) \
                 (func $absorber {absorber}) \
                 (func (local i32) \
                   (local.set 0 (i32.sub (global.get 0) (i32.const 16))) (global.set 0 (local.get 0)) \
                   (call $absorber) (global.set 0 (i32.add (global.get 0) (i32.const 16)))))"
            );
            let module = parse(&build(&text), 0).expect("parses");
            module.function(2).expect("a body").frame
        };

        assert_eq!(
            frame("(call $thrower)"),
            Some((0, 16)),
            "the throw passes through"
        );
        assert_eq!(frame("try (call $thrower) catch_all end"), None);
        assert_eq!(
            frame("(block $h (try_table (catch_all $h) (call $thrower)))"),
            None
        );
    }

    #[test]
    fn a_function_asyncify_instruments_gives_the_pointer_back_to_callers_it_never_unwinds() {
        let frames = |switches: &[(&str, i32, bool)]| {
            let check = "(if (i32.gt_u (i32.load (global.get $data)) \
                 (i32.load offset=4 (global.get $data))) (then unreachable))";
            let switches: String = switches
                .iter()
                .map(|(name, to, carries)| {
                    let (param, carry) = if *carries {
                        ("(param i32)", "(global.set $data (local.get 0))")
                    } else {
                        ("", "")
                    };
                    format!(
                        "(func (export \"{name}\") {param} \
                           (global.set $state (i32.const {to})) {carry} {check})"
                    )
                })
                .collect();
            let text = format!(
                "(module (import \"env\" \"sleep\" (func $sleep)) (memory 1) \
                 (global $__stack_pointer (mut i32) (i32.const 4096)) \
                 (global $state (mut i32) (i32.const 0)) (global $data (mut i32) (i32.const 0)) \
                 (func $sleeper (local i32) \
                   (if (i32.eq (global.get $state) (i32.const 2)) \
                     (then (local.set 0 (i32.load (global.get $data))))) \
                   (block $unwind \
                     (if (i32.eqz (global.get $state)) \
                       (then (local.set 0 (i32.sub (global.get $__stack_pointer) (i32.const 32))) \
                             (global.set $__stack_pointer (local.get 0)))) \
                     (call $sleep) \
                     (br_if $unwind (i32.eq (global.get $state) (i32.const 1))) \
                     (global.set $__stack_pointer (i32.add (local.get 0) (i32.const 32))) \
                     (return)) \
                   (i32.store (global.get $data) (local.get 0))) \
                 (func $caller (local i32) \
                   (local.set 0 (i32.sub (global.get $__stack_pointer) (i32.const 16))) \
                   (global.set $__stack_pointer (local.get 0)) \
                   (call $sleeper) \
                   (global.set $__stack_pointer (i32.add (global.get $__stack_pointer) (i32.const 16)))) \
                 (func $contained (local i32) \
                   (if (i32.eq (global.get $state) (i32.const 2)) \
                     (then (drop (i32.load (global.get $data))))) \
                   (if (i32.eqz (global.get $state)) \
                     (then (local.set 0 (i32.sub (global.get $__stack_pointer) (i32.const 16))) \
                           (global.set $__stack_pointer (local.get 0)) \
                           (global.set $__stack_pointer (i32.add (local.get 0) (i32.const 16))) \
                           (return))) \
                   (unreachable)) \
                 {switches})"
            );
            let module = parse(&build(&text), 0).expect("parses");
            [1, 2, 3].map(|index| module.function(index).expect("a body").frame)
        };
        let asyncify = [
            ("asyncify_start_unwind", 1, true),
            ("asyncify_stop_unwind", 0, false),
            ("asyncify_start_rewind", 2, true),
            ("asyncify_stop_rewind", 0, false),
        ];

        assert_eq!(
            frames(&asyncify),
            [None, Some((0, 16)), Some((0, 16))],
            "an instrumented function a rewind enters without taking its frame, a caller outside \
             the instrumentation, and an instrumented function a rewind never takes a frame in"
        );
        assert_eq!(
            frames(&asyncify[..2]),
            [None, None, Some((0, 16))],
            "a state global nothing rewinds with is not known to be Asyncify's"
        );
    }

    #[test]
    fn a_loop_entered_again_after_a_moving_call_keeps_the_frame_until_the_pointer_is_read() {
        let frame = |body: &str| {
            let text = format!(
                "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
                 (func $mover (global.set 0 (i32.sub (global.get 0) (i32.const 64)))) \
                 (func (param i32) (local i32) \
                   (local.set 1 (i32.sub (global.get 0) (i32.const 16))) (global.set 0 (local.get 1)) \
                   {body} (global.set 0 (i32.add (local.get 1) (i32.const 16)))))"
            );
            let module = parse(&build(&text), 0).expect("parses");
            module.function(1).expect("a body").frame
        };

        assert_eq!(
            frame("(loop $again (call $mover) (br_if $again (local.get 0)))"),
            Some((1, 16))
        );
        assert_eq!(
            frame("(loop $again (drop (global.get 0)) (call $mover) (br_if $again (local.get 0)))"),
            None,
            "read on the second time round"
        );
    }

    #[test]
    fn an_atomic_write_to_the_pointer_moves_it() {
        let frames = |body: &str| {
            let text = format!(
                "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
                 (func $shrink (drop (global.atomic.rmw.sub seq_cst 0 (i32.const 64)))) \
                 (func (local i32) \
                   (local.set 0 (i32.sub (global.get 0) (i32.const 16))) \
                   (global.set 0 (local.get 0)) {body}))"
            );
            let module = parse(&build(&text), 0).expect("parses");
            module.function(1).expect("a body").frame
        };
        let restore = "(global.set 0 (i32.add (global.get 0) (i32.const 16)))";

        assert_eq!(frames(restore), Some((0, 16)));
        assert_eq!(
            frames(&format!(
                "(drop (global.atomic.rmw.sub seq_cst 0 (i32.const 64))) {restore}"
            )),
            None,
            "in the function itself"
        );
        assert_eq!(
            frames(&format!("(call $shrink) {restore}")),
            None,
            "in a callee"
        );
    }

    #[test]
    fn a_handler_does_not_trust_a_holder_written_before_something_threw() {
        let frame = |body: &str| {
            let text = format!(
                "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
                 (func $abort) \
                 (func (param i32) (result i32) (local $h i32) \
                   (local.set $h (global.get 0)) {body} (i32.const 0)))"
            );
            let module = parse(&build(&text), 0).expect("parses");
            module.function(1).expect("a body").frame
        };

        assert_eq!(
            frame(
                "(block $caught (try_table (catch_all $caught) \
                   (local.set $h (i32.const 7)) (call $abort) (return (i32.const 1)))) \
                 (global.set 0 (local.get $h))"
            ),
            None,
            "try_table"
        );
        assert_eq!(
            frame(
                "try $outer try $inner (local.set $h (i32.const 7)) (call $abort) \
                 delegate $outer (return (i32.const 1)) \
                 catch_all (global.set 0 (local.get $h)) end"
            ),
            None,
            "delegate"
        );
        assert_eq!(
            frame(
                "try loop $L (call $abort) (local.set $h (i32.const 7)) (br_if $L (local.get 0)) \
                 end (return (i32.const 1)) \
                 catch_all (global.set 0 (local.get $h)) end"
            ),
            None,
            "a loop that throws on its next turn"
        );
        assert_eq!(
            frame(
                "try $outer try $inner (call $abort) catch_all (local.set $h (i32.const 7)) \
                 (call $abort) end (return (i32.const 1)) \
                 catch_all (global.set 0 (local.get $h)) end"
            ),
            None,
            "a handler that throws again"
        );
        assert_eq!(
            frame(
                "try (call $abort) (local.set $h (i32.const 7)) (return (i32.const 1)) \
                 catch_all (global.set 0 (local.get $h)) end"
            ),
            Some((1, 0)),
            "written after the last thing that throws"
        );
    }

    #[test]
    fn a_callee_that_cannot_be_read_to_its_end_can_move_the_pointer() {
        let text = "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
             (func $unread (drop (i32.const 1234567))) \
             (func (local i32) \
               (local.set 0 (i32.sub (global.get 0) (i32.const 16))) (global.set 0 (local.get 0)) \
               (call $unread) (global.set 0 (i32.add (global.get 0) (i32.const 16)))))";
        let mut image = build(text);
        let frame = |image: &[u8]| {
            let module = parse(image, 0).expect("parses");
            module.function(1).expect("a body").frame
        };
        assert_eq!(frame(&image), Some((0, 16)));

        let constant = image
            .windows(5)
            .position(|bytes| bytes == [0x41, 0x87, 0xad, 0xcb, 0x00])
            .expect("the constant is in the body");
        image[constant] = 0xff;
        assert_eq!(frame(&image), None);
    }

    #[test]
    fn a_loop_does_not_carry_its_entry_values_around() {
        let frame = |body: &str| {
            let text = format!(
                "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
                 (func (param $c i32) (local $h i32) \
                   global.get 0 i32.const 16 i32.sub local.tee $h global.set 0 {body}))"
            );
            let module = parse(&build(&text), 0).expect("parses");
            module.function(0).expect("a body").frame
        };

        assert_eq!(
            frame(
                "local.get $h \
                 loop $L (param i32) \
                   global.set 0 global.get 0 i32.const 16 i32.sub \
                   local.get $c i32.const 1 i32.sub local.tee $c br_if $L \
                   drop \
                 end \
                 global.get 0 i32.const 16 i32.add global.set 0"
            ),
            None,
            "a pointer moved further on every turn"
        );
        assert_eq!(
            frame(
                "local.get $c \
                 loop $L (param i32) \
                   i32.const 1 i32.sub local.tee $c local.get $c br_if $L \
                   drop \
                 end \
                 local.get $h i32.const 16 i32.add global.set 0"
            ),
            Some((1, 16)),
            "a count carried around beside it"
        );
    }

    #[test]
    fn a_frame_is_found_however_its_pointer_is_computed() {
        let frame = |body: &str| {
            let text = format!(
                "(module (memory 1) (global $__stack_pointer (mut i32) (i32.const 4096)) \
                 (func $work) \
                 (func (local $entry i32) (local $h i32) {body}))"
            );
            let module = parse(&build(&text), 0).expect("parses");
            module.function(1).expect("a body").frame
        };

        assert_eq!(
            frame(
                "global.get 0 \
                 i32.const -2147483648 i32.add i32.const -2147483648 i32.add \
                 i32.const -16 i32.add local.tee $h global.set 0 \
                 local.get $h i32.const 16 i32.add \
                 i32.const -2147483648 i32.sub i32.const -2147483648 i32.sub global.set 0"
            ),
            Some((1, 16)),
            "arithmetic that wraps at the address width"
        );
        assert_eq!(
            frame(
                "global.get 0 local.set $entry call $work \
                 global.get 0 i32.const 32 i32.sub local.tee $h global.set 0 \
                 local.get $entry global.set 0"
            ),
            Some((1, 32)),
            "an allocation past a save of the pointer as it was"
        );
    }

    #[test]
    fn a_callee_is_known_to_return_its_first_argument_only_when_every_exit_does() {
        let returns = |function: &str| {
            let text = format!("(module (memory 1) {function})");
            let module = parse(&build(&text), 0).expect("parses");
            let returns = module.function(0).expect("a body").returns_first_argument;
            let call = call_to(&module, 0).expect("resolves");
            assert_eq!(call.returns_argument, returns);
            returns
        };
        assert!(returns(
            "(func (param i32 i32 i32) (result i32) \
               (if (i32.eqz (local.get 2)) (then (return (local.get 0)))) \
               (memory.fill (local.get 0) (local.get 1) (local.get 2)) (local.get 0))"
        ));
        assert!(returns(
            "(func (param i32) (result i32) (block (br_if 0 (local.get 0)) (return (local.get 0))) \
               (unreachable))"
        ));
        assert!(!returns(
            "(func (param i32) (result i32) (local.set 0 (i32.const 1)) (local.get 0))"
        ));
        assert!(!returns(
            "(func (param i32 i32) (result i32) (local.get 1))"
        ));
        assert!(!returns(
            "(func (param i32) (result i32) (br_if 0 (local.get 0) (local.get 0)) (local.get 0))"
        ));
        assert!(!returns(
            "(func (param i32) (result i32) (block (result i32) (local.get 0)))"
        ));
        assert!(!returns("(func (param i32) (result i64) (i64.const 0))"));
        assert!(!returns("(func (param i64) (result i32) (i32.const 0))"));
        assert!(!returns("(func (result i32) (i32.const 0))"));
        assert!(!returns(
            "(type $f (func)) (type $c (cont $f)) (tag $e) \
             (func (param (ref null $c)) (result (ref null $c)) \
               (resume $c (on $e 0) (local.get 0)) (local.get 0))"
        ));
    }

    #[test]
    fn a_callee_that_hands_its_first_argument_on_returns_what_the_next_one_does() {
        let returns = |functions: &str| {
            let module = parse(&build(&format!("(module {functions})")), 0).expect("parses");
            let returns = module.function(0).expect("a body").returns_first_argument;
            let call = call_to(&module, 0).expect("resolves");
            assert_eq!(call.returns_argument, returns);
            returns
        };
        let inner = "(func $inner (param i32 i32) (result i32) (local.get 0))";

        assert!(
            returns(&format!(
                "(func (param i32 i32) (result i32) (call $inner (local.get 0) (local.get 1))) {inner}"
            )),
            "a call to one that returns it"
        );
        assert!(
            returns(&format!(
                "(func (param i32) (result i32) (return_call $inner (local.get 0) (i32.const 1))) {inner}"
            )),
            "a tail call"
        );
        assert!(
            returns(&format!(
                "(func (param i32) (result i32) (call $middle (local.get 0))) \
                 (func $middle (param i32) (result i32) (call $inner (local.get 0) (local.get 0))) \
                 {inner}"
            )),
            "through another that hands it on"
        );
        assert!(
            returns(
                "(func $count (param i32 i32) (result i32) \
                   (if (local.get 1) (then (return_call $count (local.get 0) (local.get 1)))) \
                   (local.get 0))"
            ),
            "a recursion that hands it on"
        );

        assert!(
            !returns(&format!(
                "(func (param i32 i32) (result i32) (call $inner (local.get 1) (local.get 0))) {inner}"
            )),
            "a call given something else first"
        );
        assert!(
            !returns(
                "(func (param i32 i32) (result i32) (call $three (i32.load (local.get 1)) \
                   (local.get 0) (local.get 1))) \
                 (func $three (param i32 i32 i32) (result i32) (local.get 0)) (memory 1)"
            ),
            "a call whose first argument is worked out"
        );
        assert!(
            !returns(
                "(func (param i32 i32) (result i32) (call $other (local.get 0) (local.get 1))) \
                 (func $other (param i32 i32) (result i32) (local.get 1))"
            ),
            "a call to one that returns something else"
        );
        assert!(
            !returns(&format!(
                "(func (param i32 i32) (result i32) \
                   (if (local.get 1) (then (return (call $inner (local.get 0) (local.get 1))))) \
                   (local.get 1)) {inner}"
            )),
            "another exit that returns something else"
        );
    }

    #[test]
    fn a_function_reference_resolves_to_the_function() {
        let module = parse(
            &build("(module (func $f) (elem declare func $f) (func (drop (ref.func $f))))"),
            0,
        )
        .expect("parses");
        let entry = module.entry(0).expect("a body");
        assert_eq!(
            module.resolve(&Operator::RefFunc { function_index: 0 }),
            Some(Resolved::Function(entry))
        );
    }

    #[test]
    fn only_a_table_nothing_else_reaches_has_empty_slots_known_to_be_null() {
        let sealed = |text: &str| parse(&build(text), 0).expect("parses").table_sealed();
        assert!(sealed(
            "(module (table 4 funcref) (func) (elem (i32.const 1) 0))"
        ));
        assert!(!sealed(
            r#"(module (import "env" "table" (table 4 funcref)) (func) (elem (i32.const 1) 0))"#
        ));
        assert!(!sealed(
            r#"(module (table (export "table") 4 funcref) (func) (elem (i32.const 1) 0))"#
        ));
        assert!(!sealed(
            "(module (table 4 funcref) (func (table.set 0 (i32.const 0) (ref.func 0))) \
             (elem declare func 0))"
        ));
        assert!(sealed(
            "(module (table 4 funcref) (func (drop (table.grow 0 (ref.null func) (i32.const 1)))))"
        ));
        assert!(!sealed(
            "(module (table 4 funcref) (func (drop (table.grow 0 (ref.func 0) (i32.const 1)))) \
             (elem declare func 0))"
        ));
    }

    #[test]
    fn the_file_goes_above_declared_memory_and_only_what_is_reached_is_mapped() {
        let layout = Layout::allocate(Shape {
            image: 0x1000,
            memory: 0x5000,
            reserved: 0x100_0000,
            ..Shape::default()
        });
        assert!(layout.file_base >= 0x100_0000);
        assert_eq!(layout.memory_end, 0x5000);
    }

    #[test]
    fn a_claimed_placement_is_held_until_its_file_is_released() {
        let layout = Layout::at(
            0x7700_0000,
            Shape {
                image: 0x1000,
                imports: 1,
                ..Shape::default()
            },
        );
        let view = 0x7700_0000;
        assert!(claim(&layout));
        install_layout(view, layout);
        assert!(!claim(&layout), "another file cannot take it");
        assert!(release(view));
        assert!(claim(&layout), "free again once released");
        install_layout(view, layout);
        install_layout(view, layout);
        assert!(
            !release(view),
            "a second view of the same file still holds it"
        );
        assert!(release(view));
    }

    #[test]
    fn two_files_do_not_share_the_regions_above_memory() {
        let small = Layout::allocate(Shape {
            image: 4 << 10,
            memory: 1 << 20,
            reserved: 1 << 20,
            globals: 4,
            imports: 4,
            tables: 8,
            tags: 0,
            memory64: false,
        });
        let large = Layout::allocate(Shape {
            image: 1 << 20,
            memory: 64 << 20,
            reserved: 64 << 20,
            globals: 900,
            imports: 900,
            tables: 4096,
            tags: 0,
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
        assert_eq!(
            module.table_entries().count(),
            0,
            "a slot past what the region can hold is not recorded"
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
    fn a_go_module_names_the_parameters_its_abi_passes() {
        let names = |marked: bool| {
            let marker = if marked {
                r#"(@custom "go:buildid" "id")"#
            } else {
                ""
            };
            let image = build(&format!(
                "(module \
                   (import \"gojs\" \"runtime.wasmExit\" (func (param i32))) \
                   (import \"gojs\" \"debug\" (func (param i32))) \
                   (import \"env\" \"other\" (func (param i32))) \
                   (func (param i32) (result i32) \
                     (block (block (br_table 0 1 (local.get 0)))) (i32.const 0)) \
                   (func (param i32) (result i32) (local.get 0)) \
                   (func (param i32) (result i32) (i32.const 0)) \
                   {marker})"
            ));
            let module = parse(&image, 0).expect("parses");
            (0..6)
                .map(|index| module.local_name(index, 0).map(str::to_owned))
                .collect::<Vec<_>>()
        };
        let named = |name: &str| Some(name.to_owned());

        assert_eq!(
            names(true),
            [
                named("sp"),
                named("value"),
                None,
                named("PC_B"),
                None,
                named("PC_B")
            ],
            "the stack pointer a host function reads its arguments from, the value it logs, \
             nothing for another host, the resume point, nothing for a parameter used for \
             something else, and the resume point of a function with nowhere to resume"
        );
        assert_eq!(names(false), [const { None }; 6], "not without Go's mark");
    }

    #[test]
    fn a_go_module_names_its_registers() {
        let go = |marked: bool, first: &str, named: bool| {
            let registers = format!(
                "(global (mut {first}) ({first}.const 0)) {} (global (mut i32) (i32.const 0))",
                "(global (mut i64) (i64.const 0)) ".repeat(6)
            );
            let marker = if marked {
                r#"(@custom "go:buildid" "id")"#
            } else {
                ""
            };
            let name = if named {
                r#"(@custom "name" "\07\05\01\00\02sp")"#
            } else {
                ""
            };
            let image = build(&format!("(module (func) {registers} {marker} {name})"));
            let module = parse(&image, 0).expect("parses");
            (0..8)
                .map(|index| module.global_name(index))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            go(true, "i32", false),
            ["SP", "CTXT", "g", "RET0", "RET1", "RET2", "RET3", "PAUSE"]
        );
        assert_eq!(
            go(false, "i32", false)[0],
            "global_0",
            "not without Go's mark"
        );
        assert_eq!(
            go(true, "i64", false)[0],
            "global_0",
            "nor with another layout"
        );
        assert_eq!(
            go(true, "i32", true)[..2],
            ["sp", "CTXT"],
            "and a name the module gives wins"
        );
    }

    #[test]
    fn a_function_is_address_taken_wherever_a_reference_to_it_is_made() {
        let image = build(
            r#"(module
                (func $called) (func $slotted) (func $passive) (func $declared) (func $held)
                (func $caller call $called ref.func $declared drop)
                (table 1 funcref) (elem (i32.const 0) $slotted)
                (elem funcref (ref.func $passive)) (elem declare func $declared)
                (global funcref (ref.func $held)))"#,
        );
        let module = parse(&image, 0).expect("parses");
        let taken: Vec<bool> = (0..6).map(|index| module.address_taken(index)).collect();
        assert_eq!(taken, [false, true, true, true, true, false]);
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
    fn switching_stacks_takes_and_gives_what_the_types_say() {
        let image = build(
            r#"(module
                 (type $f0 (func))
                 (type $c0 (cont $f0))
                 (type $f1 (func (param i32) (result i64)))
                 (type $c1 (cont $f1))
                 (type $f2 (func (param i32 f32) (result i64)))
                 (type $c2 (cont $f2))
                 (rec (type $fs (func (param i64 (ref null $cs)))) (type $cs (cont $fs)))
                 (tag $t (param i32 i32) (result f64))
                 (tag $s)
                 (func))"#,
        );
        let module = parse(&image, 0).expect("parses");
        let table = || wasmparser::ResumeTable {
            handlers: Vec::new(),
        };
        let arity = |op: Operator| match module.resolve(&op) {
            Some(Resolved::Switching(arity)) => Some(("switching", arity.pops, arity.pushes)),
            Some(Resolved::Aggregate(arity)) => Some(("aggregate", arity.pops, arity.pushes)),
            _ => None,
        };

        assert_eq!(
            arity(Operator::Suspend { tag_index: 0 }),
            Some(("switching", 2, 1))
        );
        assert_eq!(
            arity(Operator::Resume {
                cont_type_index: 3,
                resume_table: table()
            }),
            Some(("switching", 2, 1))
        );
        assert_eq!(
            arity(Operator::ResumeThrow {
                cont_type_index: 3,
                tag_index: 0,
                resume_table: table()
            }),
            Some(("switching", 3, 1))
        );
        assert_eq!(
            arity(Operator::ResumeThrowRef {
                cont_type_index: 1,
                resume_table: table()
            }),
            Some(("switching", 2, 0))
        );
        assert_eq!(
            arity(Operator::Switch {
                cont_type_index: 7,
                tag_index: 1
            }),
            Some(("switching", 2, 2)),
            "a continuation typed in terms of itself, through its recursion group"
        );
        assert_eq!(
            arity(Operator::ContBind {
                argument_index: 5,
                result_index: 3
            }),
            Some(("aggregate", 2, 1))
        );
        assert_eq!(
            arity(Operator::Resume {
                cont_type_index: 0,
                resume_table: table()
            }),
            None,
            "a function type rather than a continuation"
        );
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

    #[test]
    fn an_extended_constant_offset_is_evaluated() {
        let image = build(
            r#"(module (memory 1) (func nop)
                 (data (i32.add (i32.const 1024) (i32.const 16)) "a")
                 (data (i32.sub (i32.const 1024) (i32.const 16)) "b")
                 (data (i32.add (global.get 0) (i32.const 16)) "c")
                 (global i32 (i32.const 0)))"#,
        );
        let module = parse(&image, 0).expect("parses");
        let offsets: Vec<_> = module.data.iter().map(|span| span.memory_offset).collect();
        assert_eq!(offsets, [Some(1040), Some(1008), None]);
    }

    #[test]
    fn a_later_null_element_empties_the_slot() {
        let image = build(
            r#"(module (table 2 funcref) (func $f)
                 (elem (i32.const 0) $f)
                 (elem (i32.const 0) funcref (ref.null func)))"#,
        );
        let module = parse(&image, 0).expect("parses");
        assert_eq!(module.table_entry(0), None);
    }

    #[test]
    fn the_first_memory_decides_the_pointer_width() {
        let module =
            parse(&build("(module (memory 1) (memory i64 1) (func nop))"), 0).expect("parses");
        assert!(!module.memory64);
        let module =
            parse(&build("(module (memory i64 1) (memory 1) (func nop))"), 0).expect("parses");
        assert!(module.memory64);
    }

    #[test]
    fn only_a_copy_into_the_first_memory_places_a_segment() {
        let image = build(
            r#"(module (memory 1) (memory 1)
                 (data $d "hello") (data $e "world")
                 (func $s
                   (memory.init 1 $d (i32.const 1024) (i32.const 0) (i32.const 5))
                   (memory.init 0 $e (i32.const 2048) (i32.const 0) (i32.const 0)))
                 (start $s))"#,
        );
        let module = parse(&image, 0).expect("parses");
        assert_eq!(module.data[0].memory_offset, None, "copied into memory 1");
        assert_eq!(module.data[1].memory_offset, None, "nothing copied");
    }

    #[test]
    fn an_imported_table_is_as_big_as_its_elements_reach() {
        let image = build(
            r#"(module (import "env" "t" (table 1 funcref)) (func $f)
                 (elem (i32.const 100) $f))"#,
        );
        let module = parse(&image, 0).expect("parses");
        assert_eq!(module.shape().tables, 101);
    }

    #[test]
    fn offsets_at_the_top_of_a_64_bit_space_do_not_overflow() {
        let module = parse(
            &build(r#"(module (memory i64 1) (func nop) (data (i64.const -1) "x"))"#),
            0,
        )
        .expect("parses");
        assert_eq!(module.shape().memory, u64::MAX);

        let module = parse(
            &build(
                r#"(module (table i64 1 funcref) (func $f)
                     (elem (table 0) (i64.const 0x4000000000000000) func $f))"#,
            ),
            0,
        )
        .expect("parses");
        assert_eq!(module.table_entries().count(), 0);
        assert_eq!(module.layout.table_address(u64::MAX), u64::MAX);
    }

    #[test]
    fn an_unreadable_import_stops_the_index_space_rather_than_shifting_it() {
        let mut image = b"\0asm\x01\0\0\0".to_vec();
        image.extend([0x01, 0x04, 0x01, 0x60, 0x00, 0x00]);
        image.extend([0x02, 0x13, 0x03]);
        image.extend([0x01, b'm', 0x01, b'a', 0x00, 0x00]);
        image.extend([0x01, b'm', 0x01, b'b', 0x09, 0x00]);
        image.extend([0x01, b'm', 0x01, b'c', 0x00, 0x00]);
        image.extend([0x03, 0x02, 0x01, 0x00]);
        image.extend([0x0a, 0x04, 0x01, 0x02, 0x00, 0x0b]);
        let module = parse(&image, 0).expect("the first import is still read");
        assert_eq!(module.imports().count(), 1);
        assert_eq!(
            module.functions().count(),
            0,
            "no body is given a shifted index"
        );
    }

    #[test]
    fn nested_modules_each_get_their_own_imports_globals_and_table() {
        let first = build(
            r#"(module (import "env" "a" (func (param i32))) (global i32 (i32.const 1))
                 (table 2 funcref) (func $f) (elem (i32.const 0) $f))"#,
        );
        let second = build(
            r#"(module (import "env" "b" (func (param i64))) (global i64 (i64.const 2))
                 (table 1 funcref) (func $g) (elem (i32.const 0) $g))"#,
        );
        let mut modules = vec![
            parse(&first, 0).expect("parses"),
            parse(&second, 0).expect("parses"),
        ];
        let layout = Layout::at(
            0x10000,
            Shape {
                image: 0x100,
                imports: 2,
                globals: 2,
                tables: 3,
                ..Shape::default()
            },
        );
        place_all(&mut modules, layout);

        let (a, b) = (&modules[0].layout, &modules[1].layout);
        assert_ne!(a.import_address(0), b.import_address(0));
        assert_eq!(b.import_address(0), a.import_end());
        assert_eq!(b.global_address(0), a.global_end());
        assert_eq!(b.table_address(0), a.table_end());
        assert_eq!(b.table_end(), layout.table_end());
        assert!(layout.is_import_stub(b.import_address(0)));
    }

    #[test]
    fn a_file_placed_after_a_large_one_does_not_land_on_an_older_one() {
        let small = |memory64| Shape {
            image: 0x1000,
            memory: 1 << 20,
            reserved: 1 << 20,
            memory64,
            ..Shape::default()
        };
        let first = Layout::allocate(small(false));
        let _wide = Layout::allocate(Shape {
            memory: 8 << 30,
            reserved: 8 << 30,
            ..small(true)
        });
        let third = Layout::allocate(small(false));
        let span = |layout: &Layout| layout.file_base..layout.end();
        assert!(
            span(&third).end <= first.file_base || span(&third).start >= span(&first).end,
            "{first:?} and {third:?} overlap"
        );
    }

    fn with_dwarf(mut image: Vec<u8>, variables: &[(u64, Option<u64>, bool)]) -> Vec<u8> {
        use gimli::write::{Address, AttributeValue, DwarfUnit, EndianVec, Expression, Sections};

        let encoding = gimli::Encoding {
            format: gimli::Format::Dwarf32,
            version: 4,
            address_size: 4,
        };
        let mut dwarf = DwarfUnit::new(encoding);
        let root = dwarf.unit.root();
        let byte = dwarf.unit.add(root, gimli::DW_TAG_base_type);
        dwarf
            .unit
            .get_mut(byte)
            .set(gimli::DW_AT_byte_size, AttributeValue::Udata(1));
        for &(at, count, narrowed) in variables {
            let ty = match count {
                Some(count) => {
                    let array = dwarf.unit.add(root, gimli::DW_TAG_array_type);
                    dwarf
                        .unit
                        .get_mut(array)
                        .set(gimli::DW_AT_type, AttributeValue::UnitRef(byte));
                    let range = dwarf.unit.add(array, gimli::DW_TAG_subrange_type);
                    dwarf
                        .unit
                        .get_mut(range)
                        .set(gimli::DW_AT_count, AttributeValue::Udata(count));
                    array
                }
                None => byte,
            };
            let mut location = Expression::new();
            location.op_addr(Address::Constant(at));
            if narrowed {
                location.op_deref_size(1);
                location.op(gimli::DW_OP_stack_value);
            }
            let variable = dwarf.unit.add(root, gimli::DW_TAG_variable);
            let entry = dwarf.unit.get_mut(variable);
            entry.set(gimli::DW_AT_type, AttributeValue::UnitRef(ty));
            entry.set(gimli::DW_AT_location, AttributeValue::Exprloc(location));
        }

        let mut sections = Sections::new(EndianVec::new(gimli::LittleEndian));
        dwarf.write(&mut sections).expect("writes");
        sections
            .for_each(|id, data| {
                let (name, payload) = (id.name().as_bytes(), data.slice());
                if !payload.is_empty() {
                    let mut body = Vec::new();
                    leb128(&mut body, name.len() as u64);
                    body.extend_from_slice(name);
                    body.extend_from_slice(payload);
                    image.push(0);
                    leb128(&mut image, body.len() as u64);
                    image.extend_from_slice(&body);
                }
                Ok::<(), gimli::write::Error>(())
            })
            .expect("appends");
        image
    }

    fn leb128(out: &mut Vec<u8>, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    #[test]
    fn memory_reaches_the_zeroed_data_dwarf_places_after_the_initialised() {
        let stack_first = r#"(module (memory 4)
             (global $sp (mut i32) (i32.const 0x10000))
             (data (i32.const 0x10000) "hi")
             (func))"#;
        let module = parse(&build(stack_first), 0).expect("parses");
        assert_eq!(
            module.memory_extent, 0x10002,
            "the data and the stack top alone"
        );

        let image = with_dwarf(build(stack_first), &[(0x10010, Some(0x1000), false)]);
        let module = parse(&image, 0).expect("parses");
        assert_eq!(module.memory_extent, 0x11010, "the whole of a zeroed array");

        let image = with_dwarf(build(stack_first), &[(0x12000, None, true)]);
        let module = parse(&image, 0).expect("parses");
        assert_eq!(module.memory_extent, 0x12001, "a global narrowed to a flag");

        let image = with_dwarf(build(stack_first), &[(0x3_0000, Some(0x2_0000), false)]);
        let module = parse(&image, 0).expect("parses");
        assert_eq!(
            module.memory_extent, 0x4_0000,
            "never past what is declared"
        );

        let image = with_dwarf(build(stack_first), &[(0xffff_ffff, None, false)]);
        let module = parse(&image, 0).expect("parses");
        assert_eq!(module.memory_extent, 0x10002, "a tombstone places nothing");
    }

    #[test]
    fn memory_reaches_as_far_as_the_module_names_and_no_further() {
        let image = build(
            r#"(module (memory 256)
                 (global $sp (mut i32) (i32.const 0x9000))
                 (data (i32.const 0x400) "hi")
                 (func (result i32)
                   (i32.store offset=0x7000 (i32.const 0) (i32.const 1))
                   (i32.load (i32.const 0x6000))))"#,
        );
        let module = parse(&image, 0).expect("parses");
        assert_eq!(
            module.memory_extent, 0x9000,
            "the stack top names the furthest byte"
        );

        let beyond = build(
            r#"(module (memory 1)
                 (func (result i32) (i32.load offset=0x7fffffff (i32.const 0))))"#,
        );
        let module = parse(&beyond, 0).expect("parses");
        assert_eq!(module.memory_extent, 0x10000, "never past what is declared");

        let dynamic = build(
            r#"(module (memory 256) (data (i32.const 0x400) "hi")
                 (func (param i32) (result i32) (i32.load offset=8 (local.get 0))))"#,
        );
        let module = parse(&dynamic, 0).expect("parses");
        assert_eq!(module.memory_extent, 0x402, "the data is all it states");
    }
}
