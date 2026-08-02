//! The WebAssembly binary view
//!
//! Maps each section, creates one function per body, and gives each one the name and prototype the
//! module declares
//!
//! `init` never reports failure: a half-built view that returned an error is left in a state the
//! next callback aborts on, so anything missing is left out instead

use std::collections::BTreeMap;

use binaryninja::architecture::{ArchitectureExt, CoreArchitecture};
use binaryninja::binary_view::{
    register_binary_view_event, BinaryView, BinaryViewBase, BinaryViewEventType, BinaryViewExt,
    Result,
};
use binaryninja::confidence::Conf;
use binaryninja::custom_binary_view::{
    register_view_type, BinaryViewType, BinaryViewTypeBase, CustomBinaryView, CustomBinaryViewType,
    CustomView, CustomViewBuilder,
};
use binaryninja::data_buffer::DataBuffer;
use binaryninja::platform::Platform;
use binaryninja::rc::Ref;
use binaryninja::section::{SectionBuilder, Semantics};
use binaryninja::segment::{SegmentBuilder, SegmentFlags};
use binaryninja::settings::Settings;
use binaryninja::symbol::{Binding, Symbol, SymbolType};
use binaryninja::types::{
    FunctionParameter, MemberAccess, MemberScope, QualifiedName, StructureBuilder, Type,
};
use binaryninja::variable::{Variable, VariableSourceType};
use binaryninja::workflow::{activity, Activity, AnalysisContext, Workflow};
use binaryninja::Endianness;

use crate::arch;
use crate::cfg;
use crate::lift;
use crate::module::{self, Module, Signature, ValueKind};
use crate::settings;

pub const NAME: &str = "WASM";

pub struct WasmViewType {
    core: BinaryViewType,
}

impl AsRef<BinaryViewType> for WasmViewType {
    fn as_ref(&self) -> &BinaryViewType {
        &self.core
    }
}

impl BinaryViewTypeBase for WasmViewType {
    fn is_valid_for(&self, data: &BinaryView) -> bool {
        let mut header = [0u8; 8];
        if data.read(&mut header, 0) != header.len() || header[..4] != *b"\0asm" {
            return false;
        }

        // The upper half of the version word is the layer, and a component is claimed too, since
        // the modules nested in it are what this analyses
        let version = u16::from_le_bytes([header[4], header[5]]);
        let layer = u16::from_le_bytes([header[6], header[7]]);
        match layer {
            0 => version == 1,
            1 => true,
            _ => false,
        }
    }

    /// A code section states where every function is, so a sweep has nothing to add and everything
    /// to invent: 245 functions on one real module, each a byte of a length prefix
    ///
    /// Half of what turns it off, [`choose_analysis`] being the other, since this key is not in
    /// every core's load settings schema
    fn load_settings_for_data(&self, data: &BinaryView) -> Option<Ref<Settings>> {
        let settings = self.default_load_settings_for_data(data)?;
        settings::for_load(&settings, WORKFLOW);
        Some(settings)
    }
}

impl CustomBinaryViewType for WasmViewType {
    fn create_custom_view<'builder>(
        &self,
        data: &BinaryView,
        builder: CustomViewBuilder<'builder, Self>,
    ) -> Result<CustomView<'builder>> {
        builder.create::<WasmView>(data, ())
    }
}

pub struct WasmView {
    handle: Ref<BinaryView>,
    /// The core falls back to `read` until the segment map is activated, which is after `init`
    /// returns, and without this every address reads as empty there
    backing: Vec<(u64, u64, Option<u64>)>,
    /// Bytes of the file itself
    image: u64,
    entry: u64,
    layout: module::Layout,
}

impl AsRef<BinaryView> for WasmView {
    fn as_ref(&self) -> &BinaryView {
        &self.handle
    }
}

impl BinaryViewBase for WasmView {
    fn entry_point(&self) -> u64 {
        self.entry
    }

    fn default_endianness(&self) -> Endianness {
        Endianness::LittleEndian
    }

    fn address_size(&self) -> usize {
        self.layout.pointer
    }

    /// The file is the address space, and nothing in it is rebased
    fn relocatable(&self) -> bool {
        false
    }

    /// Ranges the file does not fill, such as the globals, read as zero, which is what they hold
    fn read(&self, buf: &mut [u8], offset: u64) -> usize {
        let Some(parent) = self.handle.parent_view() else {
            return 0;
        };

        let Some((start, end, file)) = self
            .backing
            .iter()
            .find(|(start, end, _)| (*start..*end).contains(&offset))
        else {
            return 0;
        };

        let available = (end - offset).min(buf.len() as u64) as usize;
        match file {
            Some(file) => parent.read(&mut buf[..available], file + (offset - start)),
            None => {
                buf[..available].fill(0);
                available
            }
        }
    }

    fn len(&self) -> u64 {
        self.backing.last().map_or(self.image, |(_, end, _)| *end)
    }
}

unsafe impl CustomBinaryView for WasmView {
    type Args = ();

    fn new(handle: &BinaryView, _args: &Self::Args) -> Result<Self> {
        Ok(Self {
            handle: handle.to_owned(),
            backing: Vec::new(),
            image: 0,
            entry: 0,
            layout: module::Layout::default(),
        })
    }

    fn init(&mut self, _args: Self::Args) -> Result<()> {
        let Some(parent) = self.handle.parent_view() else {
            return Ok(());
        };

        self.image = parent.len().min(module::MAX_IMAGE_LEN as u64);
        let image = parent.read_vec(0, self.image as usize);

        // A component's modules each number their functions from zero, so they are read separately
        // and the one with the most code owns the regions there is only one of
        //
        // Read at zero and moved afterwards, since where the file goes depends on how much memory
        // to leave below it, which is what the modules have to be read to find out
        let mut modules = module::parse_all(&image, 0);
        // The core builds a view of the same file more than once, so the first layout a file gets
        // is the one it keeps: a second leaves the first view's modules at addresses nothing backs
        let view = arch::view_id(&self.handle);
        self.layout = module::layout(view).unwrap_or_else(|| self.place(&modules));
        for module in &mut modules {
            module.place(self.layout);
        }
        module::install_layout(view, self.layout);
        tracing::info!(
            "wasm view: memory 0..{:#x}, file {:#x}..{:#x}, {} globals at {:#x}, {} imports at \
             {:#x}, {} bit pointers",
            self.layout.memory_end,
            self.layout.file_base,
            self.layout.file_address(self.layout.image),
            self.layout.globals,
            self.layout.global_base,
            self.layout.imports,
            self.layout.import_base,
            self.layout.pointer * 8,
        );
        // The core rejects one of two overlapping regions in silence, which reads as a file with
        // no data in it
        if !self.layout.is_ordered() {
            tracing::error!("wasm view: {:?} overlaps itself", self.layout);
        }

        let primary = modules
            .iter()
            .enumerate()
            .max_by_key(|(_, module)| module.functions().count())
            .map(|(at, _)| at);

        // A module with neither functions nor imports is still a module
        let Some(primary) = primary.filter(|_| modules.iter().any(|m| !m.is_empty())) else {
            tracing::warn!("wasm view: no functions or imports found, mapping the file only");
            let file = self.layout.file_base..self.layout.file_address(self.image);
            self.backing.push((file.start, file.end, Some(0)));
            self.handle.add_segment(
                SegmentBuilder::new(file)
                    .parent_backing(0..self.image)
                    .flags(SegmentFlags::new().readable(true).contains_data(true))
                    .is_auto(true),
            );
            return Ok(());
        };

        // Where the file backs each address is worked out either way, since `read` answers from it
        // until the segment map is active; only handing the core the segments a second time has to
        // be skipped, since they land on top of the ones already there
        let place = self.handle.segments().iter().count() == 0;
        let (file_segments, memory_segments, extra_segments) =
            self.map(&modules, &modules[primary], place);
        self.backing.sort_by_key(|(start, _, _)| *start);
        let mapped = file_segments + memory_segments + extra_segments;
        let annotated = self.annotate(&modules, &modules[primary], place);

        // Adding a segment or a section reports nothing, so counting what landed is the only way
        // to notice one the core rejected
        let segments = self.handle.segments().iter().count();
        let sections = self.handle.sections().iter().count();
        if place && (segments != mapped || sections != annotated) {
            tracing::warn!(
                "wasm view: {segments} of {mapped} segments accepted ({file_segments} for the \
                 file, {memory_segments} for memory, {extra_segments} for globals and imports), \
                 and {sections} of {annotated} sections"
            );
        }

        // A memory64 module indexes memory with an `i64`, so its pointers are eight bytes
        let wanted = arch::name_for(self.layout.pointer == 8);
        let Some(arch) = CoreArchitecture::by_name(wanted) else {
            tracing::error!("wasm view: the {wanted} architecture is not registered");
            return Ok(());
        };
        if platform_for(&arch).is_none() {
            tracing::error!("wasm view: the {wanted} architecture has no platform");
            return Ok(());
        }
        self.handle.set_default_arch(&arch);
        self.handle
            .set_default_platform(&platform_for(&arch).expect("checked above"));

        self.recover(&modules, &image);
        self.entry = entry_of(&modules[primary]).unwrap_or(0);

        // Before any function exists, since creating one starts analysis and `analyze_basic_blocks`
        // reads the body's extent from here, without which the walk runs through the functions
        // that follow and the core merges them into one
        let view = arch::view_id(&self.handle);
        let installed: Vec<_> = modules
            .into_iter()
            .map(|module| {
                let (base, end) = (module.base, module.end);
                module::install(view, base, end, module)
            })
            .collect();

        // No functions and no entry point here: creating either starts analysis, analysis reads the
        // view, and a read while `init` is still running aborts the process
        drop(installed);

        Ok(())
    }
}

impl WasmView {
    /// The regions there is only one of come from the module with the most code, but the ones
    /// addressed by index are sized to the largest index space across every module, or a nested
    /// module with more globals would reach past the end
    fn place(&self, modules: &[Module]) -> module::Layout {
        let largest = |shape: fn(&module::Shape) -> u64| {
            modules
                .iter()
                .map(|module| shape(&module.shape()))
                .max()
                .unwrap_or(0)
        };
        let primary = modules
            .iter()
            .max_by_key(|module| module.functions().count())
            .map(Module::shape)
            .unwrap_or_default();

        module::Layout::allocate(module::Shape {
            image: self.image,
            memory: primary.memory,
            globals: largest(|shape| shape.globals),
            imports: largest(|shape| shape.imports),
            tables: largest(|shape| shape.tables),
            // There is one architecture for the view, so one memory64 module anywhere makes every
            // pointer in the file eight bytes
            memory64: modules.iter().any(|module| module.memory64),
        })
    }

    /// Only the code section is executable, or a type section disassembles as a run of `try` and
    /// `br_table`
    fn map(&mut self, modules: &[Module], primary: &Module, place: bool) -> (usize, usize, usize) {
        let mut added = 0usize;
        // Every module's sections, so a component's nested code is executable too
        let spans: Vec<(u64, u64, bool)> = modules
            .iter()
            .flat_map(|module| module.sections.iter())
            .filter(|section| section.start < section.end)
            .map(|section| (section.start, section.end, section.code))
            .collect();
        self.handle.begin_bulk_add_segments();

        // Memory goes first, because what it backs is what the file image leaves out: mapping
        // those bytes in both places made every string appear twice, the file copy unable to carry
        // a reference; it is taken from what memory mapped rather than what the module declares, so
        // a segment memory had no room for keeps its bytes in the file
        let (memory_segments, mut initialisers) = self.map_memory(primary, place);
        initialisers.sort_unstable();

        let covering = covering(
            spans,
            self.layout.file_base,
            self.layout.file_address(self.image),
        );
        let covering = without(covering, &whole_data_section(primary, &initialisers));

        for (start, end, code) in covering {
            let offset = start - self.layout.file_base;
            self.backing.push((start, end, Some(offset)));
            let flags = SegmentFlags::new()
                .readable(true)
                .executable(code)
                .contains_code(code)
                .contains_data(!code);
            added += 1;
            if place {
                self.handle.add_segment(
                    SegmentBuilder::new(start..end)
                        .parent_backing(offset..offset + (end - start))
                        .flags(flags)
                        .is_auto(true),
                );
            }
        }

        let file_segments = added;
        added = 0;

        // Globals are not addressable from wasm code, so the lifter models them at a region of
        // their own and the view maps it to give each one a name
        let globals = modules
            .iter()
            .map(|module| module.globals().count() as u64)
            .max()
            .unwrap_or(0);
        if globals != 0 {
            let span = globals * module::GLOBAL_STRIDE;
            self.backing.push((
                self.layout.global_base,
                self.layout.global_base + span,
                None,
            ));
            added += 1;
            if place {
                self.handle.add_segment(
                    SegmentBuilder::new(
                        self.layout.global_base
                            ..self.layout.global_base + globals * module::GLOBAL_STRIDE,
                    )
                    .flags(
                        SegmentFlags::new()
                            .readable(true)
                            .writable(true)
                            .contains_data(true),
                    )
                    .is_auto(true),
                );
            }
        }

        // The lifter reads the slot an index names, so the region holds the addresses the element
        // section put there; nothing in the file has that shape, since it stores function indices
        // as LEB128, so the bytes are built here
        //
        // An unbacked region does not read as unknown: the core reads it as zeros, so every
        // `call_indirect` whose index folded became a call to address zero, 392 on one module
        if self.layout.tables != 0 {
            self.backing
                .push((self.layout.table_base, self.layout.table_end(), None));
            added += 1;
            if place {
                self.handle.memory_map().add_data_memory_region(
                    "wasm table",
                    self.layout.table_base,
                    &DataBuffer::new(&self.table_image(modules)),
                    Some(SegmentFlags::new().readable(true).contains_data(true)),
                );
            }
        }

        // Calls really do go here, so the region has to be executable, and backed by the file or
        // the core refuses to create a function in it at all; there are no real bytes for a stub,
        // so the backing points at the module header and is never read
        let imports = modules
            .iter()
            .map(|module| module.imports().count() as u64)
            .max()
            .unwrap_or(0);
        let stubs = imports * module::IMPORT_STRIDE;
        if imports != 0 && stubs <= self.image {
            self.backing.push((
                self.layout.import_base,
                self.layout.import_base + stubs,
                Some(0),
            ));
            added += 1;
            if place {
                self.handle.add_segment(
                    SegmentBuilder::new(self.layout.import_base..self.layout.import_base + stubs)
                        .parent_backing(0..stubs)
                        .flags(
                            SegmentFlags::new()
                                .readable(true)
                                .executable(true)
                                .contains_code(true),
                        )
                        .is_auto(true),
                );
            }
        }
        self.handle.end_bulk_add_segments();
        (file_segments, memory_segments, added)
    }

    /// One address per slot; a slot nothing fills stays zero, which is what reaching it would do,
    /// since an uninitialised entry traps rather than calling anything
    fn table_image(&self, modules: &[Module]) -> Vec<u8> {
        let pointer = self.layout.pointer;
        let mut image = vec![0u8; (self.layout.tables as usize).saturating_mul(pointer)];
        let Some(primary) = modules
            .iter()
            .max_by_key(|module| module.functions().count())
        else {
            return image;
        };

        for (slot, function) in primary.table_entries() {
            let Some(entry) = primary.entry(function) else {
                continue;
            };
            let at = (slot as usize).saturating_mul(pointer);
            let Some(room) = at
                .checked_add(pointer)
                .and_then(|end| image.get_mut(at..end))
            else {
                continue;
            };
            room.copy_from_slice(&entry.to_le_bytes()[..pointer]);
        }
        image
    }

    /// The code addresses memory rather than the file, so a string only resolves if its bytes are
    /// readable at the address the load computes
    fn map_memory(&mut self, module: &Module, place: bool) -> (usize, Vec<(u64, u64)>) {
        let mut added = 0usize;
        let mut backed: Vec<(u64, u64)> = Vec::new();
        let mut placed: Vec<(u64, u64, u64)> = module
            .data
            .iter()
            .filter_map(|span| {
                let at = span.memory_offset?;
                let len = span.end.checked_sub(span.start)?;
                // The file is indexed by offset rather than by the address the span was read at
                let file = self.layout.file_offset(span.start)?;
                (len > 0).then_some((at, len, file))
            })
            .collect();
        placed.sort_by_key(|(at, _, _)| *at);

        let declared = module.memories.first().copied().unwrap_or(0);
        let (layout, end) = memory_layout(placed, declared, self.layout.memory_end);
        if end == 0 {
            return (added, backed);
        }

        let writable = SegmentFlags::new()
            .readable(true)
            .writable(true)
            .contains_data(true);

        for (offset, len, file) in layout {
            let span = self.layout.memory_address(offset)..self.layout.memory_address(offset + len);
            let mut segment = SegmentBuilder::new(span.clone())
                .flags(writable)
                .is_auto(true);
            self.backing.push((span.start, span.end, file));
            if let Some(file) = file {
                segment = segment.parent_backing(file..file + len);
                backed.push((
                    self.layout.file_address(file),
                    self.layout.file_address(file + len),
                ));
            }
            added += 1;
            if place {
                self.handle.add_segment(segment);
            }
        }

        if place {
            self.handle.add_section(
                SectionBuilder::new(
                    "memory".to_string(),
                    self.layout.memory_address(0)..self.layout.memory_address(end),
                )
                .semantics(Semantics::ReadWriteData)
                .is_auto(true),
            );
        }

        (added, backed)
    }

    /// Also what puts a compiler's DWARF where the core can see it
    fn annotate(&self, modules: &[Module], primary: &Module, place: bool) -> usize {
        let mut added = 0usize;
        // The core keys sections by name, and two custom sections may share one, so each gets a
        // name of its own rather than replacing the last
        let mut taken: BTreeMap<String, usize> = BTreeMap::new();

        for section in modules.iter().flat_map(|module| module.sections.iter()) {
            if section.start >= section.end {
                continue;
            }

            let wanted = if section.name.is_empty() {
                format!("custom@{:#x}", section.start)
            } else {
                section.name.clone()
            };
            let seen = taken.entry(wanted.clone()).or_default();
            *seen += 1;
            let name = if *seen == 1 {
                wanted
            } else {
                format!("{wanted}.{}", *seen - 1)
            };

            let semantics = if section.code {
                Semantics::ReadOnlyCode
            } else {
                Semantics::ReadOnlyData
            };
            added += 1;
            if place {
                self.handle.add_section(
                    SectionBuilder::new(name, section.start..section.end)
                        .semantics(semantics)
                        .is_auto(true),
                );
            }
        }

        let imports = modules
            .iter()
            .map(|module| module.imports().count() as u64)
            .max()
            .unwrap_or(0);
        if imports != 0 {
            added += 1;
            if place {
                self.handle.add_section(
                    SectionBuilder::new(
                        "extern".to_string(),
                        self.layout.import_base
                            ..self.layout.import_base + imports * module::IMPORT_STRIDE,
                    )
                    .semantics(Semantics::External)
                    .is_auto(true),
                );
            }
        }

        // Giving a data segment a type is what turns the bytes in it into strings the core finds
        for (index, span) in primary.data.iter().enumerate() {
            let len = span.end.saturating_sub(span.start);
            if len == 0 {
                continue;
            }
            // No variable over the whole segment: an array of `len` bytes renders one row per byte
            // and buries the strings the core finds by itself
            let name = primary.data_name(index as u32);
            let at = span
                .memory_offset
                .map_or(span.start, |at| self.layout.memory_address(at));
            self.handle.define_auto_symbol(
                &Symbol::builder(SymbolType::Data, &name, at)
                    .short_name(name)
                    .create(),
            );
        }

        // Named after the function a `call_indirect` reaches through the slot, since the address a
        // function has here is this view's invention and is nowhere in the file
        let table = primary.table_entries().count();
        if table != 0 {
            added += 1;
            self.handle.add_section(
                SectionBuilder::new(
                    "table slots".to_string(),
                    self.layout.table_base..self.layout.table_end(),
                )
                .semantics(Semantics::ReadOnlyData)
                .is_auto(true),
            );
        }
        for (slot, function) in primary.table_entries() {
            let at = self.layout.table_address(slot);
            let name = format!("table[{slot}] {}", primary.name_of(function));
            self.handle.define_auto_symbol(
                &Symbol::builder(SymbolType::Data, &name, at)
                    .short_name(name)
                    .create(),
            );
            if let Some(entry) = primary.entry(function) {
                self.handle.set_comment_at(
                    at,
                    &format!("wasm: call_indirect {slot} goes to {entry:#x}"),
                );
            }
        }

        for (index, global) in primary.globals() {
            let address = self.layout.global_address(index);
            let ty = value_type(global.kind, self.layout.pointer);
            // A `v128` is wider than the slot the region gives it, and would bury the next global
            if global.kind.size() as u64 <= module::GLOBAL_STRIDE {
                self.handle.define_auto_data_var(address, ty.as_ref());
            }

            let name = primary.global_name(index);
            self.handle.define_auto_symbol(
                &Symbol::builder(SymbolType::Data, &name, address)
                    .short_name(name)
                    .create(),
            );
        }

        added
    }

    /// A branch names a label rather than an address, so doing this where every body is known is
    /// what gives `instruction_info` a real answer by the time the core asks
    fn recover(&self, modules: &[Module], image: &[u8]) {
        let view = arch::view_id(&self.handle);
        let mut unreadable = 0usize;
        let mut unbalanced = 0usize;

        // Per module rather than over every body at once, since resolving a call or a block type
        // means asking the module that body belongs to
        for module in modules {
            for (_, info) in module.functions() {
                let body = (info.entry - self.layout.file_base) as usize
                    ..(info.end - self.layout.file_base) as usize;
                let Some(code) = image.get(body) else {
                    unreadable += 1;
                    continue;
                };
                let flow = cfg::recover(code, info.entry, Some(module));
                if flow.underflow {
                    cfg::note_unbalanced(view, info.entry);
                    unbalanced += 1;
                }
                cfg::install(view, &flow);
            }
        }

        if unreadable != 0 {
            tracing::warn!("wasm view: {unreadable} function bodies lay outside the image");
        }
        if unbalanced != 0 {
            tracing::warn!(
                "wasm view: {unbalanced} function bodies take more off the operand stack than they \
                 put on it, so this file is not one a WebAssembly engine would load and nothing it \
                 declares about itself is enforced"
            );
        }
    }
}

type MemorySpan = (u64, u64, Option<u64>);

/// `placed` is `(memory offset, length, file offset)` per data segment, and what comes back is the
/// same triple with no file offset where nothing initialises that stretch
///
/// The core rejects overlapping segments in silence, so a module declaring two that overlap keeps
/// the first and the rest of that stretch stays unbacked
fn memory_layout(
    mut placed: Vec<(u64, u64, u64)>,
    declared: u64,
    window: u64,
) -> (Vec<MemorySpan>, u64) {
    // Anything past the window is dropped rather than wrapped into the regions above it
    placed.retain(|(at, len, _)| at.saturating_add(*len) <= window);
    placed.sort_by_key(|(at, _, _)| *at);

    let end = placed
        .iter()
        .map(|(at, len, _)| at + len)
        .max()
        .unwrap_or(0)
        .max(declared)
        .min(window);
    if end == 0 {
        return (Vec::new(), 0);
    }

    // Giving a segment a length longer than its backing, as an ELF does with `.bss`, halves the
    // count, but the core splits each one back apart, so the view holds the same number either way
    let mut layout = Vec::new();
    let mut at = 0u64;
    for (offset, len, file) in placed {
        if offset < at {
            continue;
        }
        if offset > at {
            layout.push((at, offset - at, None));
        }
        layout.push((offset, len, Some(file)));
        at = offset + len;
    }
    if at < end {
        layout.push((at, end - at, None));
    }

    (layout, end)
}

/// Memory is where a data section is mapped, so mapping it here as well would be the same bytes at
/// two addresses, which is what an ELF avoids with `.data`
///
/// Cut whole rather than a segment at a time, or the framing between one segment's contents and the
/// next is stranded: 234 segments where 23 will do, on a module with 212 data segments
fn whole_data_section(module: &Module, initialisers: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let placed = module
        .data
        .iter()
        .filter(|span| span.start < span.end)
        .count();
    if initialisers.is_empty() || initialisers.len() != placed {
        return initialisers.to_vec();
    }

    let start = initialisers.iter().map(|(start, _)| *start).min();
    let end = initialisers.iter().map(|(_, end)| *end).max();
    match start.zip(end) {
        Some((start, end)) => vec![(start, end)],
        None => initialisers.to_vec(),
    }
}

/// `holes` are ranges shown somewhere else in the address space, so leaving them mapped here would
/// present the same bytes twice
fn without(covering: Vec<(u64, u64, bool)>, holes: &[(u64, u64)]) -> Vec<(u64, u64, bool)> {
    let mut kept = Vec::with_capacity(covering.len());

    for (start, end, code) in covering {
        let mut at = start;
        for (from, to) in holes.iter().filter(|(f, t)| *f < end && *t > start) {
            if *from > at {
                kept.push((at, *from, code));
            }
            at = at.max(*to);
        }
        if at < end {
            kept.push((at, end, code));
        }
    }

    kept
}

/// The core rejects overlapping segments and says nothing about it, so the covering has to be
/// exact, and the bytes between sections are still part of the file: the eight byte header, and
/// every section's own id and length prefix
fn covering(mut spans: Vec<(u64, u64, bool)>, base: u64, image: u64) -> Vec<(u64, u64, bool)> {
    // A truncated file leaves a section past its own bytes, which the core backs without a word
    spans.retain(|(start, end, _)| *start < image && *end > base);
    for (start, end, _) in &mut spans {
        *start = (*start).max(base);
        *end = (*end).min(image);
    }
    spans.sort_by_key(|(start, _, _)| *start);

    let mut covering: Vec<(u64, u64, bool)> = Vec::new();
    let mut at = base;
    for (start, end, code) in spans {
        if start > at {
            covering.push((at, start, false));
        }
        if end > at {
            covering.push((at.max(start), end, code));
            at = end;
        }
    }
    if at < image {
        covering.push((at, image, false));
    }

    covering
}

/// User functions rather than auto ones, since a code section states where every function begins
/// and an auto function the core has no other reason to keep is dropped on reanalysis
fn declare(view: &BinaryView, module: &Module, platform: &Platform) {
    let (mut stubs_failed, mut bodies_failed) = (0usize, 0usize);

    // Naming the address a call to an import goes to is what turns `unimplemented {call 5}` into
    // `env.foo(...)`
    for (index, import) in module.imports() {
        let address = module.layout.import_address(index);
        let name = module.name_of(index);
        // `ImportedFunction` with a global binding is what puts it in the symbol list; `External`
        // is merely undefined here, which shows up nowhere
        view.define_auto_symbol(
            &Symbol::builder(SymbolType::ImportedFunction, &name, address)
                .binding(Binding::Global)
                .short_name(name)
                .create(),
        );

        // A call to an import is a call to code, and the architecture disassembles this address as
        // a stub that returns
        match view.add_user_function_with_platform(address, platform) {
            Some(stub) => {
                if let Some(prototype) = prototype(view, module, index, &import.signature) {
                    stub.set_user_type(&prototype);
                }
                stub.set_user_pure(Conf::new(false, binaryninja::confidence::MAX_CONFIDENCE));
                bind_parameters(&stub, &import.signature);
            }
            None => stubs_failed += 1,
        }
    }

    for (index, info) in module.functions() {
        let name = module.name_of(index);

        // The symbol goes first, since the function created next picks its name up from it; an
        // export is bound globally, which is what the symbol list filters on
        let binding = if module.export_name(index).is_some() {
            Binding::Global
        } else {
            Binding::Local
        };
        view.define_auto_symbol(
            &Symbol::builder(SymbolType::Function, &name, info.entry)
                .binding(binding)
                .short_name(name)
                .create(),
        );

        // Created without a prototype, then given one, since a type the core cannot satisfy is not
        // worth failing the function over
        let Some(function) = view.add_user_function_with_platform(info.entry, platform) else {
            bodies_failed += 1;
            continue;
        };

        // The module's public surface is where a host enters, so analysis starts there
        if module.export_name(index).is_some() {
            view.add_entry_point_with_platform(info.entry, platform);
        }
        // A type section states the signature exactly, so it is not something the core should
        // replace with what it inferred
        if let Some(prototype) = prototype(view, module, index, &info.signature) {
            function.set_user_type(&prototype);
        }
        bind_parameters(&function, &info.signature);

        let mut notes = Vec::new();
        // Before anything else, since it changes what the rest of the note is worth
        if cfg::is_unbalanced(arch::view_id(view), info.entry) {
            notes.push(
                "the operand stack goes below empty in this body, which no module an engine \
                 would load can do, so the signature below is what the file claims rather than \
                 anything enforced"
                    .to_string(),
            );
        }
        if let Some(export) = module.export_name(index) {
            notes.push(format!("exported as \"{export}\""));
        }
        // Parameters are named through the prototype, but the locals past them are slots off `fp`
        // rather than anything the core has a variable for
        let locals = module.local_names_past(index, info.signature.params.len() as u32);
        if !locals.is_empty() {
            notes.push(format!("locals: {}", locals.join(", ")));
        }
        if !notes.is_empty() {
            let notes: Vec<String> = notes.iter().map(|note| format!("wasm: {note}")).collect();
            function.set_comment(&notes.join("\n"));
        }
    }

    // Whether the code section became functions is the first thing anyone needs to know
    let bodies = module.functions().count();
    let imports = module.imports().count();
    tracing::info!(
        "wasm view: {} of {bodies} functions and {} of {imports} import stubs created, \
             {} globals, {} sections",
        bodies - bodies_failed,
        imports - stubs_failed,
        module.globals().count(),
        module.sections.len(),
    );
    if bodies_failed != 0 || stubs_failed != 0 {
        tracing::warn!(
            "wasm view: {bodies_failed} function bodies and {stubs_failed} import stubs \
                 could not be created"
        );
    }
}
/// Binds each parameter to the slot the caller left it in, without which a call site resolves only
/// the first argument and the stores for the rest read as dead
fn bind_parameters(function: &binaryninja::function::Function, signature: &Signature) {
    let params = signature.params.len() as u32;
    if params == 0 {
        return;
    }
    let slots: Vec<Variable> = (0..params).map(parameter_slot).collect();
    function.set_user_parameter_variables(slots, u8::MAX);
}

/// Multi-value returns have no single type to be, so they become a struct registered under the
/// function's own name
fn prototype(
    view: &BinaryView,
    module: &Module,
    index: u32,
    signature: &Signature,
) -> Option<Ref<Type>> {
    let parameters: Vec<FunctionParameter> = signature
        .params
        .iter()
        .enumerate()
        .map(|(nth, kind)| {
            let name = module
                .local_name(index, nth as u32)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("arg{nth}"));
            // Stated even though the core drops it, since where it survives it agrees with the
            // frame the caller built
            let at = parameter_slot(nth as u32);
            FunctionParameter::new(parameter_type(*kind), name, Some(at))
        })
        .collect();

    let returns = match signature.results.as_slice() {
        [] => Type::void(),
        [only] => value_type(*only, module.layout.pointer),
        many => result_struct(view, &module.name_of(index), many, module.layout.pointer),
    };

    Some(Type::function(returns.as_ref(), parameters, false))
}
/// The core lays consecutive stack parameters out by the width of their types, so a four byte `i32`
/// in an eight byte slot puts every parameter after the first somewhere nothing was written
///
/// The name and whether it is a float stay the module's own; only the width is the slot's, which an
/// `i32` really does occupy the low half of
pub(crate) fn parameter_type(kind: ValueKind) -> Ref<Type> {
    let slot = lift::SLOT as usize;
    match kind {
        ValueKind::I32 | ValueKind::I64 => Type::named_int(slot, true, kind.name()),
        ValueKind::F32 | ValueKind::F64 => Type::named_float(slot, kind.name()),
        ValueKind::V128 | ValueKind::Ref => Type::named_int(slot, false, kind.name()),
    }
}

pub(crate) fn parameter_slot(nth: u32) -> Variable {
    Variable::new(
        VariableSourceType::StackVariableSourceType,
        0,
        i64::from(nth) * lift::SLOT as i64,
    )
}

fn result_struct(
    view: &BinaryView,
    function: &str,
    results: &[ValueKind],
    pointer: usize,
) -> Ref<Type> {
    let mut builder = StructureBuilder::new();
    for (nth, kind) in results.iter().enumerate() {
        builder.append(
            &value_type(*kind, pointer),
            &format!("result{nth}"),
            MemberAccess::PublicAccess,
            MemberScope::NoScope,
        );
    }

    let name = QualifiedName::from(format!("{function}_results"));
    let structure = Type::structure(&builder.finalize());
    let registered = view.define_auto_type(name, "wasm", &structure);
    Type::named_type_from_type(registered, &structure)
}
/// `start` is the only thing the format calls an entry point, so failing that the conventional
/// exported names are what a host would call
fn entry_of(module: &Module) -> Option<u64> {
    // The start function may be an import, which has no body to begin execution at
    if let Some(body) = module.start.and_then(|index| module.body(index)) {
        return Some(body.entry);
    }

    for wanted in ["_start", "main", "__main_argc_argv", "_initialize"] {
        for (index, info) in module.functions() {
            if module.export_name(index) == Some(wanted) {
                return Some(info.entry);
            }
        }
    }

    None
}

/// Named as the text format does, so a prototype reads the way its source did rather than in C
/// spellings nothing in the file mentions
pub(crate) fn value_type(kind: ValueKind, pointer: usize) -> Ref<Type> {
    match kind {
        ValueKind::I32 => Type::named_int(4, true, "i32"),
        ValueKind::I64 => Type::named_int(8, true, "i64"),
        ValueKind::F32 => Type::named_float(4, "f32"),
        ValueKind::F64 => Type::named_float(8, "f64"),
        ValueKind::V128 => Type::named_int(16, false, "v128"),
        ValueKind::Ref => Type::named_int(pointer, false, "ref"),
    }
}

/// The architecture's own standalone platform, since a second one built with `Platform::new` has a
/// different identity under the same name and nothing else in the core would know it
fn platform_for(arch: &CoreArchitecture) -> Option<Ref<Platform>> {
    arch.standalone_platform()
        .or_else(|| Platform::by_name(&arch.name()))
}

/// Never during `init`: the bindings hold the view's context in a placeholder state for its
/// duration, so a callback back into the view panics there, and creating a function starts the
/// analysis that would make one
///
/// The segment map is not activated until finalization either, so this is also the first point a
/// function's address validates against anything
fn on_finalized(view: &BinaryView) {
    // The event fires for every view of every file, and a raw view of the same file shares its
    // session id, so the architecture is what tells them apart; not `view_type()`, which does not
    // yet report `WASM` while the view is finalizing
    if !view
        .default_arch()
        .is_some_and(|arch| arch::is_ours(&arch.name()))
    {
        return;
    }

    tracing::info!("wasm view: finalized, declaring functions");
    choose_analysis(view);

    // Every module, since a component's nested ones start past its own header and looking one up
    // by the view's first address would find nothing
    let modules = module::all(arch::view_id(view));
    if modules.is_empty() {
        tracing::warn!("wasm view: finalized with nothing read from the image");
        return;
    }
    if modules.iter().all(|module| module.is_empty()) {
        tracing::warn!("wasm view: finalized with no functions or imports");
        return;
    }

    let Some(arch) = view.default_arch() else {
        tracing::error!("wasm view: finalized with no architecture");
        return;
    };
    let Some(platform) = platform_for(&arch) else {
        tracing::error!(
            "wasm view: the {} architecture has no platform",
            arch.name()
        );
        return;
    };

    for module in &modules {
        declare(view, module, &platform);
    }

    // The entry belongs to whichever module holds the code, the one `init` mapped memory for
    let primary = modules
        .iter()
        .max_by_key(|module| module.functions().count())
        .expect("checked above");
    match entry_of(primary) {
        Some(entry) => view.add_entry_point_with_platform(entry, &platform),
        None => tracing::info!("wasm view: no start function and no conventional export"),
    }
    view.update_analysis();
}

fn on_analysed(view: &BinaryView) {
    // Same gate as `on_finalized`, since this fires for every view of every file
    if !view
        .default_arch()
        .is_some_and(|arch| arch::is_ours(&arch.name()))
    {
        return;
    }
    drop_invented_functions(view);
}

/// A code section states where every function begins, so anything else the core made a function
/// of is not one: 216 past the 719 one real module declares, at length prefixes and declarations
///
/// Not restricted to auto functions, since every function here reports itself as a user function,
/// the core's own included, so one created inside a body by hand does not survive the next pass
fn drop_invented_functions(view: &BinaryView) {
    let id = arch::view_id(view);
    let invented: Vec<_> = view
        .functions()
        .iter()
        .filter(|function| invented_at(id, function.start()))
        .map(|function| function.to_owned())
        .collect();

    for function in &invented {
        view.remove_user_function(function);
        view.remove_auto_function(function, true);
    }
    if !invented.is_empty() {
        tracing::info!(
            "wasm view: dropped {} functions the core invented in the image",
            invented.len()
        );
    }
}

fn note_addresses_inside_strings(view: &BinaryView) {
    let Some(layout) = module::layout(arch::view_id(view)) else {
        return;
    };

    let mut strings: Vec<(u64, u64)> = view
        .strings()
        .iter()
        .filter(|string| layout.memory_mapped(string.start))
        .map(|string| (string.start, string.start + string.length as u64))
        .collect();
    strings.sort_unstable();

    let mut noted = 0usize;
    for variable in view.data_variables().iter() {
        if !variable.auto_discovered || !layout.memory_mapped(variable.address) {
            continue;
        }
        let at = strings.partition_point(|(start, _)| *start <= variable.address);
        let Some(&(start, end)) = at.checked_sub(1).and_then(|index| strings.get(index)) else {
            continue;
        };
        if variable.address <= start || variable.address >= end {
            continue;
        }
        if view
            .comment_at(variable.address)
            .is_some_and(|had| !had.is_empty())
        {
            continue;
        }
        view.set_comment_at(
            variable.address,
            &format!(
                "wasm: {} bytes into the string at {start:#x}",
                variable.address - start
            ),
        );
        noted += 1;
    }

    if noted != 0 {
        tracing::info!("wasm view: noted {noted} data variables that start inside a string");
    }
}

fn choose_analysis(view: &BinaryView) {
    settings::for_view(view, WORKFLOW);
}

/// The core's own workflow ends with `core.module.deleteUnusedAutoFunctions`, which is this same
/// job for a format that cannot say where its functions are, so ours runs next to it
const ROOT: &str = "core.module.metaAnalysis";
const WORKFLOW: &str = "wasm.module.analysis";
const SWEEP: &str = "wasm.module.dropInventedFunctions";

/// The finalization event fires once and the core goes on making functions long after it, minting
/// one at a basic block start as the linear view reaches a part of the file nothing has rendered
///
/// Registering the workflow is half of it, [`WasmViewType::load_settings_for_data`] selecting it
/// for the files this plugin claims being the other
pub fn register_workflow() {
    let config = activity::Config::action(
        SWEEP,
        "Drop functions the module does not declare",
        "A WebAssembly code section states where every function begins, so an address in the image \
         that is not one of their entries is not a function, whatever the core made of it.",
    )
    .eligibility(activity::Eligibility::auto());

    let activity = Activity::new_with_action(config, |context: &AnalysisContext| {
        drop_invented_functions(&context.view());
        note_addresses_inside_strings(&context.view());
    });

    let Some(core) = Workflow::get(ROOT) else {
        tracing::warn!("wasm view: there is no {ROOT} to extend, so the sweep runs once only");
        return;
    };
    let Ok(cloned) = core.clone_to(WORKFLOW).register_activity(&activity) else {
        tracing::warn!("wasm view: {SWEEP} was not accepted, so the sweep runs once only");
        return;
    };

    // Next to the core's own sweep, or failing that wherever an update ends, since which
    // activities a core carries is its business
    let mut placed = Err(cloned);
    for anchor in [
        "core.module.deleteUnusedAutoFunctions",
        "core.module.finishUpdate",
        "core.module.advancedAnalysis",
        "core.module.baseAnalysis",
    ] {
        let Err(workflow) = placed else { break };
        placed = match workflow.insert_after(anchor, [SWEEP]) {
            Ok(workflow) => {
                tracing::debug!("wasm view: {SWEEP} runs after {anchor}");
                Ok(workflow)
            }
            Err(()) => Err(Workflow::get(ROOT)
                .expect("checked above")
                .clone_to(WORKFLOW)
                .register_activity(&activity)
                .expect("accepted above")),
        };
    }
    let Ok(workflow) = placed else {
        tracing::warn!("wasm view: nothing in {ROOT} to hang {SWEEP} on, so it runs once only");
        return;
    };
    if workflow.register().is_err() {
        tracing::warn!("wasm view: {WORKFLOW} was rejected, so the sweep runs once only");
    }
}

pub(crate) fn invented_at(view: crate::ViewId, start: u64) -> bool {
    module::lookup(view, start).is_some_and(|module| {
        module
            .body_covering(start)
            .is_none_or(|(_, info)| info.entry != start)
    })
}

pub fn register() {
    register_workflow();
    register_view_type(NAME, "WebAssembly Module", |core| WasmViewType { core });
    register_binary_view_event(
        BinaryViewEventType::BinaryViewFinalizationEvent,
        on_finalized,
    );
    // After analysis rather than on finalization, since the core has invented nothing yet at the
    // point the functions are created
    register_binary_view_event(
        BinaryViewEventType::BinaryViewInitialAnalysisCompletionEvent,
        on_analysed,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(spans: Vec<(u64, u64, bool)>, image: u64) -> Vec<(u64, u64, bool)> {
        let covering = covering(spans, 0, image);

        let mut at = 0;
        for (start, end, _) in &covering {
            assert_eq!(
                *start, at,
                "a gap or an overlap at {start:#x} in {covering:?}"
            );
            assert!(start < end, "an empty segment in {covering:?}");
            at = *end;
        }
        assert!(at <= image, "{covering:?} runs past the file");
        covering
    }

    const WINDOW: u64 = 1 << 20;

    fn check_memory(placed: Vec<(u64, u64, u64)>, declared: u64) -> Vec<MemorySpan> {
        let (layout, end) = memory_layout(placed, declared, WINDOW);

        let mut at = 0;
        for (offset, len, _) in &layout {
            assert_eq!(*offset, at, "a gap or an overlap in {layout:?}");
            assert!(*len > 0, "an empty segment in {layout:?}");
            at = offset + len;
        }
        assert_eq!(at, end, "{layout:?} does not reach the end of memory");
        layout
    }

    #[test]
    fn data_segments_back_the_memory_they_fill_and_nothing_else() {
        // One page declared, with "hi" landing at offset 64 from file offset 500
        let layout = check_memory(vec![(64, 2, 500)], 65536);
        assert_eq!(
            layout,
            [(0, 64, None), (64, 2, Some(500)), (66, 65470, None)],
            "only the segment's own stretch is backed"
        );
    }

    #[test]
    fn several_data_segments_keep_their_order_and_their_gaps() {
        let layout = check_memory(vec![(100, 10, 900), (8, 4, 800)], 200);
        assert_eq!(
            layout,
            [
                (0, 8, None),
                (8, 4, Some(800)),
                (12, 88, None),
                (100, 10, Some(900)),
                (110, 90, None)
            ]
        );
    }

    #[test]
    fn data_past_the_declared_size_is_still_mapped() {
        let layout = check_memory(vec![(0, 16, 400)], 0);
        assert_eq!(layout, [(0, 16, Some(400))]);
    }

    #[test]
    fn memory_with_nothing_in_it_maps_nothing() {
        assert_eq!(memory_layout(Vec::new(), 0, WINDOW), (Vec::new(), 0));
    }

    #[test]
    fn memory_past_the_window_is_dropped_rather_than_wrapped() {
        let layout = check_memory(vec![(WINDOW, 8, 0)], 0);
        assert!(layout.is_empty(), "a segment outside the window is dropped");

        let (_, end) = memory_layout(Vec::new(), 1u64 << 32, WINDOW);
        assert_eq!(end, WINDOW, "clamped to what can be mapped");
    }

    #[test]
    fn overlapping_data_segments_do_not_produce_overlapping_memory() {
        check_memory(vec![(0, 32, 100), (16, 32, 200)], 64);
        check_memory(vec![(8, 8, 100), (8, 8, 200)], 32);
    }

    #[test]
    fn sections_tile_the_file_with_the_bytes_between_them() {
        // The eight byte header, then a type section, then a code section, then a gap
        let covering = check(vec![(8, 20, false), (24, 60, true)], 64);
        assert_eq!(
            covering,
            [
                (0, 8, false),
                (8, 20, false),
                (20, 24, false),
                (24, 60, true),
                (60, 64, false)
            ],
            "only the code section is code"
        );
    }

    #[test]
    fn a_section_reaching_the_end_leaves_nothing_over() {
        assert_eq!(
            check(vec![(8, 32, true)], 32),
            [(0, 8, false), (8, 32, true)]
        );
    }

    #[test]
    fn unordered_and_overlapping_sections_still_tile() {
        check(vec![(40, 60, false), (8, 20, true)], 64);
        check(vec![(8, 40, true), (20, 30, false)], 64);
        check(vec![(8, 40, true), (8, 40, false)], 40);
        check(vec![(0, 64, true)], 64);
    }

    #[test]
    fn an_empty_or_absent_section_list_still_maps_the_file() {
        assert_eq!(check(vec![], 64), [(0, 64, false)]);
        assert!(covering(vec![], 0, 0).is_empty());
    }

    #[test]
    fn a_section_past_the_image_does_not_map_past_it() {
        assert_eq!(
            covering(vec![(8, 20, true)], 0, 12),
            [(0, 8, false), (8, 12, true)]
        );
        assert_eq!(covering(vec![(20, 40, true)], 0, 12), [(0, 12, false)]);
    }

    #[test]
    fn a_covering_leaves_out_what_is_mapped_elsewhere() {
        let file = vec![(0x10, 0x20, false), (0x20, 0x40, true)];

        // A hole inside one span splits it, and the rest is untouched
        assert_eq!(
            without(file.clone(), &[(0x14, 0x18)]),
            [(0x10, 0x14, false), (0x18, 0x20, false), (0x20, 0x40, true)]
        );

        // A hole covering a span entirely drops it
        assert_eq!(without(file.clone(), &[(0x10, 0x20)]), [(0x20, 0x40, true)]);

        // Two holes in one span, and one that reaches past its end
        assert_eq!(
            without(file.clone(), &[(0x12, 0x14), (0x16, 0x30)]),
            [(0x10, 0x12, false), (0x14, 0x16, false), (0x30, 0x40, true)]
        );

        // Nothing to remove leaves the covering as it was
        assert_eq!(without(file.clone(), &[]), file);
    }

    fn build(text: &str) -> Vec<u8> {
        wat::parse_str(text).expect("the fixture assembles")
    }

    #[test]
    fn a_fully_placed_data_section_leaves_the_file_in_one_piece() {
        let image = build(
            r#"(module (memory 1) (func nop)
                 (data (i32.const 0) "one") (data (i32.const 64) "two"))"#,
        );
        let read = module::parse(&image, 0).expect("parses");

        // Each segment's contents, with the framing between them left over
        let initialisers: Vec<(u64, u64)> = read.data.iter().map(|d| (d.start, d.end)).collect();
        assert_eq!(initialisers.len(), 2);
        let holes = whole_data_section(&read, &initialisers);
        assert_eq!(
            holes,
            [(initialisers[0].0, initialisers[1].1)],
            "one hole over the whole of it"
        );
    }

    #[test]
    fn a_passive_data_segment_keeps_its_bytes_in_the_file() {
        let image = build(
            r#"(module (memory 1) (func nop)
                 (data (i32.const 0) "active") (data "passive"))"#,
        );
        let read = module::parse(&image, 0).expect("parses");
        assert_eq!(read.data.len(), 2);

        // Only the active one reached memory, so the exact holes are kept
        let initialisers = vec![(read.data[0].start, read.data[0].end)];
        assert_eq!(whole_data_section(&read, &initialisers), initialisers);
    }

    #[test]
    fn sections_from_several_modules_still_tile_the_file() {
        let second = [(0x40, 0x50, false), (0x50, 0x68, true)];
        let first = [(0x10, 0x18, false), (0x18, 0x30, true)];

        let covering = check([second.as_slice(), first.as_slice()].concat(), 0x80);
        let code: Vec<_> = covering
            .iter()
            .filter(|(_, _, code)| *code)
            .map(|(start, end, _)| (*start, *end))
            .collect();
        assert_eq!(code, [(0x18, 0x30), (0x50, 0x68)], "{covering:?}");
    }
}
