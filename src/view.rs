//! The WebAssembly binary view
//!
//! Maps each section, creates one function per body, and gives each one the name and prototype the
//! module declares
//!
//! `initialize` never reports failure: a half-built view that returned an error is left in a state the
//! next callback aborts on, so anything missing is left out instead

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use wasmparser::{FuncType, Operator, ValType};

use binaryninja::Endianness;
use binaryninja::architecture::{
    Architecture, ArchitectureExt, CoreArchitecture, CoreRegister, Register, RegisterId,
};
use binaryninja::binary_view::{
    BinaryView, BinaryViewBase, BinaryViewEventType, CustomBinaryView, CustomBinaryViewType,
    register_binary_view_event, register_binary_view_type,
};
use binaryninja::confidence::{Conf, MAX_CONFIDENCE};
use binaryninja::data_buffer::DataBuffer;
use binaryninja::function::{Function, FunctionUpdateType};
use binaryninja::medium_level_il::{
    MediumLevelILFunction, MediumLevelILInstruction, MediumLevelILInstructionKind,
};
use binaryninja::platform::Platform;
use binaryninja::rc::{Array, Ref};
use binaryninja::section::{SectionBuilder, Semantics};
use binaryninja::segment::{SegmentBuilder, SegmentFlags};
use binaryninja::settings::Settings;
use binaryninja::symbol::{Binding, Symbol, SymbolType};
use binaryninja::types::{
    FunctionParameter, MemberAccess, MemberScope, QualifiedName, ReturnValue, StructureBuilder,
    Type, ValueLocation, ValueLocationComponent,
};
use binaryninja::variable::{Variable, VariableSourceType};
use binaryninja::workflow::{Activity, AnalysisContext, Workflow, activity};

use crate::arch;
use crate::cfg;
use crate::debug;
use crate::insn;
use crate::lift;
use crate::module::{self, FunctionInfo, Module, Signature, ValueKind};
use crate::settings;
use crate::upgrade;
use crate::wasi;

pub const NAME: &str = "WASM";

pub struct WasmViewType;

impl CustomBinaryViewType for WasmViewType {
    type CustomBinaryView = WasmView;
    const NAME: &'static str = NAME;
    const LONG_NAME: &'static str = "WebAssembly Module";

    fn create_binary_view(&self, data: &BinaryView) -> Result<WasmView, ()> {
        Ok(WasmView {
            parent: data.to_owned(),
            backing: Vec::new(),
            image: 0,
            entry: 0,
            layout: module::Layout::default(),
            held: None,
        })
    }

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
    /// Half of what turns it off, [`settings::preset`] being the other, since this key is not in
    /// every core's load settings schema
    fn load_settings_for_data(&self, data: &BinaryView) -> Option<Ref<Settings>> {
        settings::load_settings(data)
    }
}

pub struct WasmView {
    parent: Ref<BinaryView>,
    /// The core falls back to `read` until the segment map is activated, which is after `initialize`
    /// returns, and without this every address reads as empty there
    backing: Vec<(u64, u64, Option<u64>)>,
    /// Bytes of the file itself
    image: u64,
    entry: u64,
    layout: module::Layout,
    held: Option<crate::ViewId>,
}

impl Drop for WasmView {
    fn drop(&mut self) {
        if let Some(id) = self.held
            && module::release(id)
        {
            cfg::forget(id);
            debug::forget(id);
        }
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
        let Some((start, end, file)) = self
            .backing
            .iter()
            .find(|(start, end, _)| (*start..*end).contains(&offset))
        else {
            return 0;
        };

        let available = (end - offset).min(buf.len() as u64) as usize;
        match file {
            Some(file) => self
                .parent
                .read(&mut buf[..available], file + (offset - start)),
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

impl CustomBinaryView for WasmView {
    fn initialize(&mut self, view: &BinaryView) -> bool {
        self.image = self.parent.len().min(module::MAX_IMAGE_LEN as u64);
        let image = self.parent.read_vec(0, self.image as usize);

        // A component's modules each number their functions from zero, so they are read separately
        // and the one with the most code owns the regions there is only one of
        //
        // Read at zero and moved afterwards, since where the file goes depends on how much memory
        // to leave below it, which is what the modules have to be read to find out
        let mut modules = module::parse_all(&image, 0);
        if !settings::STACK_FRAMES.opening(&self.parent) {
            for module in &mut modules {
                module.drop_frames();
            }
        }
        // The core builds a view of the same file more than once, so the first layout a file gets
        // is the one it keeps: a second leaves the first view's modules at addresses nothing backs
        let id = arch::view_id(view);
        let known = module::layout(id);
        let restored = known
            .is_none()
            .then(|| self.restore(view, &modules))
            .flatten();
        self.layout = known
            .or(restored)
            .unwrap_or_else(|| module::Layout::allocate(self.shape(&modules)));
        module::place_all(&mut modules, self.layout);
        module::install_layout(id, self.layout);
        if restored.is_some() {
            module::mark_restored(id);
        }
        self.held = Some(id);
        tracing::info!(
            "wasm view: memory below {:#x}, file {:#x}..{:#x}, {} globals at {:#x}, {} imports at \
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
            view.add_segment(
                SegmentBuilder::new(file)
                    .parent_backing(0..self.image)
                    .flags(SegmentFlags::new().readable(true).contains_data(true))
                    .is_auto(true),
            );
            return true;
        };

        // Where the file backs each address is worked out either way, since `read` answers from it
        // until the segment map is active; only handing the core the segments a second time has to
        // be skipped, since they land on top of the ones already there
        let place = view.segments().iter().count() == 0;
        let (file_segments, memory_segments, extra_segments) =
            self.map(view, &modules, &modules[primary], place);
        self.backing.sort_by_key(|(start, _, _)| *start);
        let mapped = file_segments + memory_segments + extra_segments;
        let annotated = self.annotate(view, &modules, &modules[primary], place)
            + usize::from(memory_segments != 0);

        // Adding a segment or a section reports nothing, so counting what landed is the only way
        // to notice one the core rejected
        let segments = view.segments().iter().count();
        let sections = view.sections().iter().count();
        if place && (segments != mapped || sections != annotated) {
            tracing::warn!(
                "wasm view: {segments} of {mapped} segments accepted ({file_segments} for the \
                 file, {memory_segments} for memory, {extra_segments} above the file), \
                 and {sections} of {annotated} sections"
            );
        }

        // A memory64 module indexes memory with an `i64`, so its pointers are eight bytes
        let wanted = arch::name_for(self.layout.pointer == 8);
        let Some(arch) = CoreArchitecture::by_name(wanted) else {
            tracing::error!("wasm view: the {wanted} architecture is not registered");
            return true;
        };
        let Some(platform) = platform_for(&arch) else {
            tracing::error!("wasm view: the {wanted} architecture has no platform");
            return true;
        };
        view.set_default_arch(&arch);
        view.set_default_platform(&platform);

        self.recover(view, &modules, &image);
        self.entry = entry_of(&modules[primary]).unwrap_or(0);

        // Before any function exists, since creating one starts analysis and `analyze_basic_blocks`
        // reads the body's extent from here, without which the walk runs through the functions
        // that follow and the core merges them into one
        let installed: Vec<_> = modules
            .into_iter()
            .map(|module| {
                let (base, end) = (module.base, module.end);
                module::install(id, base, end, module)
            })
            .collect();

        // No functions and no entry point here: creating either starts analysis, analysis reads the
        // view, and a read while `initialize` is still running aborts the process
        drop(installed);

        true
    }
}

impl WasmView {
    fn restore(&self, view: &BinaryView, modules: &[Module]) -> Option<module::Layout> {
        if !view.file().is_database_backed() {
            return None;
        }
        let layout = upgrade::saved_layout(view, modules, self.shape(modules))?;
        if !module::claim(&layout) {
            tracing::warn!(
                "wasm view: the database placed the file at {:#x}, where another open file is, so \
                 its saved analysis will not line up until that file is closed and this one \
                 reopened",
                layout.file_base
            );
            return None;
        }
        Some(layout)
    }

    /// The regions there is only one of come from the module with the most code, but the ones
    /// addressed by index are sized to the largest index space across every module, or a nested
    /// module with more globals would reach past the end
    fn shape(&self, modules: &[Module]) -> module::Shape {
        let total = |shape: fn(&module::Shape) -> u64| {
            modules
                .iter()
                .map(|module| shape(&module.shape()))
                .fold(0u64, u64::saturating_add)
        };
        let primary = modules
            .iter()
            .max_by_key(|module| module.functions().count())
            .map(Module::shape)
            .unwrap_or_default();

        module::Shape {
            image: self.image,
            memory: primary.memory,
            reserved: primary.reserved,
            globals: total(|shape| shape.globals),
            imports: total(|shape| shape.imports),
            tables: total(|shape| shape.tables),
            tags: total(|shape| shape.tags),
            // There is one architecture for the view, so one memory64 module anywhere makes every
            // pointer in the file eight bytes
            memory64: modules.iter().any(|module| module.memory64),
        }
    }

    /// Only the code section is executable, or a type section disassembles as a run of `try` and
    /// `br_table`
    fn map(
        &mut self,
        view: &BinaryView,
        modules: &[Module],
        primary: &Module,
        place: bool,
    ) -> (usize, usize, usize) {
        let mut added = 0usize;
        // Every module's sections, so a component's nested code is executable too
        let spans: Vec<(u64, u64, bool)> = modules
            .iter()
            .flat_map(|module| module.sections.iter())
            .filter(|section| section.start < section.end)
            .map(|section| (section.start, section.end, section.code))
            .collect();
        view.begin_bulk_add_segments();

        // Memory goes first, because what it backs is what the file image leaves out: mapping
        // those bytes in both places made every string appear twice, the file copy unable to carry
        // a reference; it is taken from what memory mapped rather than what the module declares, so
        // a segment memory had no room for keeps its bytes in the file
        let (memory_segments, mut initialisers) = self.map_memory(view, primary, place);
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
                view.add_segment(
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
        let globals = self.layout.globals;
        if globals != 0 {
            let span = globals * module::GLOBAL_STRIDE;
            self.backing.push((
                self.layout.global_base,
                self.layout.global_base + span,
                None,
            ));
            added += 1;
            if place {
                view.add_segment(
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
            let runs = self.table_runs(modules);
            added += runs.len();
            if place {
                let image = self.table_image(modules);
                let pointer = self.layout.pointer;
                for (slots, known) in runs {
                    view.memory_map().add_data_memory_region(
                        &table_name("wasm table", &slots, self.layout.tables),
                        self.layout.table_address(slots.start as u64),
                        &DataBuffer::new(&image[slots.start * pointer..slots.end * pointer]),
                        Some(
                            SegmentFlags::new()
                                .readable(true)
                                .writable(!known)
                                .contains_data(true),
                        ),
                    );
                }
            }
        }

        // Calls really do go here, so the region has to be executable, and backed by the file or
        // the core refuses to create a function in it at all; there are no real bytes for a stub,
        // so the backing points at the module header and is never read
        let imports = self.layout.imports;
        let stubs = imports * module::IMPORT_STRIDE;
        if imports != 0 && stubs <= self.image {
            self.backing.push((
                self.layout.import_base,
                self.layout.import_base + stubs,
                Some(0),
            ));
            added += 1;
            if place {
                view.add_segment(
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
        if self.layout.tags != 0 {
            let tags = self.layout.tag_base..self.layout.tag_end();
            self.backing.push((tags.start, tags.end, None));
            added += 1;
            if place {
                view.add_segment(
                    SegmentBuilder::new(tags)
                        .flags(SegmentFlags::new().readable(true).contains_data(true))
                        .is_auto(true),
                );
            }
        }
        view.end_bulk_add_segments();
        (file_segments, memory_segments, added)
    }

    fn table_runs(&self, modules: &[Module]) -> Vec<(std::ops::Range<usize>, bool)> {
        let known = self.table_known(modules);
        let mut first = 0;
        known
            .chunk_by(|a, b| a == b)
            .map(|run| {
                let slots = first..first + run.len();
                first = slots.end;
                (slots, run[0])
            })
            .collect()
    }

    fn table_known(&self, modules: &[Module]) -> Vec<bool> {
        let mut known = vec![false; self.layout.tables as usize];
        for module in modules {
            let first =
                (module.layout.table_base - self.layout.table_base) as usize / self.layout.pointer;
            let end = first
                .saturating_add(module.layout.tables as usize)
                .min(known.len());
            let Some(own) = known.get_mut(first..end) else {
                continue;
            };
            if module.table_sealed() {
                own.fill(true);
                continue;
            }
            if module.table_written() {
                continue;
            }
            for (slot, _) in module.table_entries() {
                if let Some(filled) = usize::try_from(slot)
                    .ok()
                    .and_then(|slot| own.get_mut(slot))
                {
                    *filled = true;
                }
            }
        }
        known
    }

    /// One address per slot; a slot nothing fills stays zero, which is what reaching it would do,
    /// since an uninitialised entry traps rather than calling anything
    fn table_image(&self, modules: &[Module]) -> Vec<u8> {
        let pointer = self.layout.pointer;
        let mut image = vec![0u8; (self.layout.tables as usize).saturating_mul(pointer)];
        for module in modules {
            for (slot, function) in module.table_entries() {
                let Some(entry) = module
                    .entry(function)
                    .filter(|_| slot < module.layout.tables)
                else {
                    continue;
                };
                let at = module.layout.table_address(slot) - self.layout.table_base;
                let Some(room) = usize::try_from(at)
                    .ok()
                    .and_then(|at| image.get_mut(at..at.checked_add(pointer)?))
                else {
                    continue;
                };
                room.copy_from_slice(&entry.to_le_bytes()[..pointer]);
            }
        }
        image
    }

    /// The code addresses memory rather than the file, so a string only resolves if its bytes are
    /// readable at the address the load computes
    fn map_memory(
        &mut self,
        view: &BinaryView,
        module: &Module,
        place: bool,
    ) -> (usize, Vec<(u64, u64)>) {
        let mut added = 0usize;
        let mut backed: Vec<(u64, u64)> = Vec::new();
        let placed: Vec<(u64, u64, u64)> = module
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

        let end = self.layout.memory_end;
        let layout = memory_layout(placed, end);
        let Some(&(start, _, _)) = layout.first() else {
            return (added, backed);
        };

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
                view.add_segment(segment);
            }
        }

        if place {
            view.add_section(
                SectionBuilder::new(
                    "linear memory".to_string(),
                    self.layout.memory_address(start)..self.layout.memory_address(end),
                )
                .semantics(Semantics::ReadWriteData)
                .is_auto(true),
            );
        }

        (added, backed)
    }

    /// Also what puts a compiler's DWARF where the core can see it
    fn annotate(
        &self,
        view: &BinaryView,
        modules: &[Module],
        primary: &Module,
        place: bool,
    ) -> usize {
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
                view.add_section(
                    SectionBuilder::new(name, section.start..section.end)
                        .semantics(semantics)
                        .is_auto(true),
                );
            }
        }

        let imports = self.layout.imports;
        if imports != 0 {
            added += 1;
            if place {
                view.add_section(
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
            view.define_auto_symbol(
                &Symbol::builder(SymbolType::Data, &name, at)
                    .short_name(name)
                    .create(),
            );
        }

        // Named after the function a `call_indirect` reaches through the slot, since the address a
        // function has here is this view's invention and is nowhere in the file
        let slots: Vec<(&Module, u64, u32)> = modules
            .iter()
            .flat_map(|module| {
                module
                    .table_entries()
                    .filter(|(slot, _)| *slot < module.layout.tables)
                    .map(move |(slot, function)| (module, slot, function))
            })
            .collect();
        if !slots.is_empty() {
            let runs = self.table_runs(modules);
            added += runs.len();
            if place {
                for (run, known) in runs {
                    let start = self.layout.table_address(run.start as u64);
                    let end = self.layout.table_address(run.end as u64);
                    view.add_section(
                        SectionBuilder::new(
                            table_name("table slots", &run, self.layout.tables),
                            start..end,
                        )
                        .semantics(if known {
                            Semantics::ReadOnlyData
                        } else {
                            Semantics::ReadWriteData
                        })
                        .is_auto(true),
                    );
                }
            }
        }
        for (module, slot, function) in slots {
            let at = module.layout.table_address(slot);
            let name = format!("table[{slot}] {}", module.name_of(function));
            view.define_auto_symbol(
                &Symbol::builder(SymbolType::Data, &name, at)
                    .short_name(name)
                    .create(),
            );
            if let Some(entry) = module.entry(function) {
                view.set_comment_at(
                    at,
                    &format!("wasm: call_indirect {slot} goes to {entry:#x}"),
                );
            }
        }

        for module in modules {
            for (index, global) in module.globals() {
                let address = module.layout.global_address(index);
                let ty = value_type(global.kind, self.layout.pointer);
                // A `v128` is wider than the slot the region gives it, and would bury the next global
                if global.kind.size() as u64 <= module::GLOBAL_STRIDE {
                    view.define_auto_data_var(address, ty.as_ref());
                }

                let name = module.global_name(index);
                view.define_auto_symbol(
                    &Symbol::builder(SymbolType::Data, &name, address)
                        .short_name(name)
                        .create(),
                );
            }
            for index in module.tags() {
                let name = module.tag_name(index);
                view.define_auto_symbol(
                    &Symbol::builder(SymbolType::Data, &name, module.layout.tag_address(index))
                        .short_name(name)
                        .create(),
                );
            }
        }

        added
    }

    /// A branch names a label rather than an address, so doing this where every body is known is
    /// what gives `instruction_info` a real answer by the time the core asks
    fn recover(&self, view: &BinaryView, modules: &[Module], image: &[u8]) {
        let id = arch::view_id(view);
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
                    cfg::note_unbalanced(id, info.entry);
                    unbalanced += 1;
                }
                cfg::install(id, &flow);
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
fn memory_layout(mut placed: Vec<(u64, u64, u64)>, end: u64) -> Vec<MemorySpan> {
    // Anything past the end is dropped rather than wrapped into the regions above it
    placed.retain(|(at, len, _)| at.saturating_add(*len) <= end);
    placed.sort_by_key(|(at, _, _)| *at);

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
    if let Some(&(0, len, None)) = layout.first() {
        if len > 1 {
            layout[0] = (1, len - 1, None);
        } else {
            layout.remove(0);
        }
    }
    layout
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
pub(crate) fn type_stub(stub: &Function, wasi: &wasi::Import) {
    stub.set_user_type(&wasi.ty);
    if !wasi.returns {
        stub.set_user_can_return(Conf::new(false, MAX_CONFIDENCE));
    }
}

fn declare(view: &BinaryView, module: &Module, platform: &Platform) {
    let (mut stubs_failed, mut bodies_failed) = (0usize, 0usize);
    let saved = module::restored(arch::view_id(view));

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
            Some(_) if saved => {}
            Some(stub) => {
                match wasi::prototype(view, import) {
                    Some(wasi) => type_stub(&stub, &wasi),
                    None => stub.set_user_type(&prototype(view, module, index, &import.signature)),
                }
                stub.set_user_pure(Conf::new(false, MAX_CONFIDENCE));
                stub.set_user_clobbered_registers(clobbered(platform), MAX_CONFIDENCE);
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
        if saved {
            continue;
        }
        // A type section states the signature exactly, so it is not something the core should
        // replace with what it inferred
        function.set_user_type(&prototype(view, module, index, &info.signature));
        function.set_user_clobbered_registers(clobbered(platform), MAX_CONFIDENCE);
        type_indirect_calls(view, module, &function, info);

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
fn table_name(name: &str, slots: &std::ops::Range<usize>, tables: u64) -> String {
    if slots.start == 0 && slots.end as u64 == tables {
        name.to_string()
    } else {
        format!("{name}[{}..{}]", slots.start, slots.end)
    }
}

fn clobbered(platform: &Platform) -> Vec<CoreRegister> {
    let arch = platform.arch();
    arch::clobbered_registers()
        .into_iter()
        .filter_map(|id| arch.register_from_id(id))
        .collect()
}

fn type_indirect_calls(
    view: &BinaryView,
    module: &Module,
    function: &Function,
    info: &FunctionInfo,
) {
    let code = view.read_vec(info.entry, info.end.saturating_sub(info.entry) as usize);
    let mut offset = 0;
    while let Some(insn) = code.get(offset..).and_then(insn::decode_any) {
        let at = info.entry + offset as u64;
        offset += insn.len;
        let (Operator::CallIndirect { type_index, .. }
        | Operator::ReturnCallIndirect { type_index, .. }
        | Operator::CallRef { type_index }
        | Operator::ReturnCallRef { type_index }) = insn.op
        else {
            continue;
        };
        let Some(signature) = module.type_signature(type_index) else {
            continue;
        };
        let owner = module.type_name(type_index).map_or_else(
            || format!("type_{type_index}"),
            |name| format!("{name}_type"),
        );
        let ty = function_type(
            view,
            &owner,
            signature,
            module.type_value_types(type_index),
            module.layout.pointer,
            |_| None,
        );
        function.set_user_call_type_adjustment(at, Some(&*ty), None);
    }
}

fn name_locals(view: &BinaryView, function: &Function, il: &MediumLevelILFunction) {
    let id = arch::view_id(view);
    let Some(module) = module::lookup(id, function.start()) else {
        return;
    };
    if module::restored(id) {
        debug::load_saved(view);
    }
    let Some((index, info)) = module
        .body_covering(function.start())
        .filter(|(_, info)| info.entry == function.start())
    else {
        return;
    };
    let kinds = module.local_kinds(index);
    let params = info.signature.params.len() as u32;
    let declared: BTreeMap<u32, &str> = module.declared_local_names(index, params).collect();
    let assigned = debug::assigned(id, info.entry);
    if assigned.is_empty() && declared.is_empty() {
        return;
    }
    let described: HashSet<u32> = assigned.values().map(|(local, _)| *local).collect();
    let parameters = function.parameter_variables().contents;
    let current = |variable: &Variable, kind: ValueKind| {
        function
            .variable_type(variable)
            .unwrap_or_else(|| Conf::new(register_type(kind, module.layout.pointer), 0))
    };

    let mut names = Vec::new();
    let mut holders = Vec::new();
    let mut aliases: HashMap<Variable, BTreeSet<usize>> = HashMap::new();
    for variable in il.variables().iter() {
        let Some(local) = register_of(&variable).and_then(arch::local_of) else {
            continue;
        };
        let Some(&kind) = kinds.get(local as usize) else {
            continue;
        };
        if parameters.contains(&variable) || function.is_var_user_defined(&variable) {
            continue;
        }
        let definitions = described
            .contains(&local)
            .then(|| il.variable_definitions(&variable));
        let from_dwarf = definitions
            .as_ref()
            .and_then(|all| assigned_name(all, &assigned));
        if let (Some(_), Some(all)) = (&from_dwarf, &definitions) {
            for operand in aliases_of(il, all, &assigned) {
                aliases.entry(operand).or_default().insert(names.len());
            }
        }
        let (name, ty) = match from_dwarf {
            Some((name, Some(ty))) => (name, Conf::new(ty, MAX_CONFIDENCE)),
            Some((name, None)) => (name, current(&variable, kind)),
            None => {
                let Some(name) = declared.get(&local) else {
                    continue;
                };
                let ty = register_type(kind, module.layout.pointer);
                (name.to_string(), Conf::new(ty, MAX_CONFIDENCE))
            }
        };
        let first = definitions
            .iter()
            .flat_map(|all| all.iter())
            .map(|definition| definition.address)
            .min()
            .unwrap_or(u64::MAX);
        holders.push((name.clone(), storage_of(&variable), first));
        names.push((variable, name, ty));
    }
    holders.extend(
        il.variables()
            .iter()
            .filter(|variable| {
                variable.ty == VariableSourceType::StackVariableSourceType
                    || function.is_var_user_defined(variable)
            })
            .chain(parameters.iter().copied())
            .map(|variable| (function.variable_name(&variable), storage_of(&variable), 0)),
    );
    let renamed = distinguish(&holders);
    let final_name = |nth: usize| {
        let (variable, name, _) = &names[nth];
        renamed
            .get(&(name.clone(), storage_of(variable)))
            .unwrap_or(name)
    };
    for (nth, (variable, _, ty)) in names.iter().enumerate() {
        function.create_auto_var(variable, ty, final_name(nth), false);
    }
    for (operand, fed) in &aliases {
        if let (Some(&nth), 1) = (fed.first(), fed.len())
            && !function.is_var_user_defined(operand)
        {
            function.create_auto_var(operand, &names[nth].2, final_name(nth), false);
        }
    }
}

fn aliases_of(
    il: &MediumLevelILFunction,
    definitions: &Array<MediumLevelILInstruction>,
    assigned: &debug::Assignments,
) -> Vec<Variable> {
    let once = definitions.len() == 1;
    definitions
        .iter()
        .filter(|definition| assigned.contains_key(&definition.address))
        .filter_map(|definition| match &definition.kind {
            MediumLevelILInstructionKind::SetVar(set) => il.instruction_from_expr_index(set.src),
            _ => None,
        })
        .filter_map(|source| match source.kind {
            MediumLevelILInstructionKind::Var(read) => Some(read.src),
            _ => None,
        })
        .filter(|operand| {
            register_of(operand).is_some_and(arch::is_operand)
                && il.variable_definitions(operand).len() == 1
                && (once || il.variable_uses(operand).len() == 1)
        })
        .collect()
}

fn assigned_name(
    definitions: &Array<MediumLevelILInstruction>,
    assigned: &debug::Assignments,
) -> Option<(String, Option<Ref<Type>>)> {
    let lists: Vec<&Vec<debug::Assigned>> = definitions
        .iter()
        .filter_map(|definition| assigned.get(&definition.address))
        .map(|(_, names)| names)
        .collect();
    let complete = lists.len() == definitions.len();
    lists
        .first()?
        .iter()
        .find(|(name, ..)| {
            let held: Vec<&debug::Assigned> = lists
                .iter()
                .filter_map(|list| list.iter().find(|(other, ..)| other == name))
                .collect();
            held.len() == lists.len() && (complete || held.iter().any(|(.., inlined)| !inlined))
        })
        .map(|(name, ty, _)| (name.clone(), ty.clone()))
}

type Storage = (bool, i64);

fn storage_of(variable: &Variable) -> Storage {
    (
        variable.ty == VariableSourceType::StackVariableSourceType,
        variable.storage,
    )
}

fn distinguish(holders: &[(String, Storage, u64)]) -> HashMap<(String, Storage), String> {
    let mut storages: BTreeMap<&str, BTreeMap<Storage, u64>> = BTreeMap::new();
    for (name, storage, first) in holders {
        let earliest = storages
            .entry(name)
            .or_default()
            .entry(*storage)
            .or_insert(*first);
        *earliest = (*earliest).min(*first);
    }
    let mut used: HashSet<String> = storages.keys().map(|name| name.to_string()).collect();
    let mut renamed = HashMap::new();
    for (name, held) in &storages {
        let mut order: Vec<(u64, Storage)> = held
            .iter()
            .map(|(storage, first)| (*first, *storage))
            .collect();
        order.sort_unstable();
        let mut nth = 0;
        for (_, storage) in order.into_iter().skip(1) {
            let suffixed = loop {
                nth += 1;
                let candidate = format!("{name}_{nth}");
                if used.insert(candidate.clone()) {
                    break candidate;
                }
            };
            renamed.insert((name.to_string(), storage), suffixed);
        }
    }
    renamed
}

fn register_of(variable: &Variable) -> Option<RegisterId> {
    if variable.ty != VariableSourceType::RegisterVariableSourceType {
        return None;
    }
    u32::try_from(variable.storage).ok().map(RegisterId)
}

fn prototype(view: &BinaryView, module: &Module, index: u32, signature: &Signature) -> Ref<Type> {
    function_type(
        view,
        &module.name_of(index),
        signature,
        module.value_types(index),
        module.layout.pointer,
        |nth| module.local_name(index, nth).map(str::to_owned),
    )
}

fn function_type(
    view: &BinaryView,
    owner: &str,
    signature: &Signature,
    values: Option<&FuncType>,
    pointer: usize,
    names: impl Fn(u32) -> Option<String>,
) -> Ref<Type> {
    let parameters = signature
        .params
        .iter()
        .enumerate()
        .map(|(nth, kind)| {
            let name = names(nth as u32).unwrap_or_else(|| format!("arg{nth}"));
            let value = values.and_then(|values| values.params().get(nth));
            FunctionParameter::new(slot_type(*kind, value, pointer), name, None)
        })
        .collect();
    Type::function(
        returns(
            view,
            owner,
            &signature.results,
            values.map(FuncType::results),
            pointer,
        ),
        parameters,
        false,
    )
}

pub(crate) fn register_type(kind: ValueKind, pointer: usize) -> Ref<Type> {
    match kind {
        ValueKind::V128 => Type::named_int(lift::SLOT as usize, false, kind.name()),
        kind => value_type(kind, pointer),
    }
}

pub(crate) fn slot_type(kind: ValueKind, value: Option<&ValType>, pointer: usize) -> Ref<Type> {
    match value {
        Some(value @ ValType::Ref(_)) => {
            Type::named_int(pointer, false, &insn::val_type_text(value))
        }
        _ => register_type(kind, pointer),
    }
}

/// Multi-value returns have no single type to be, so they become a struct registered as
/// `{owner}_results`, the owner being a function or, for an indirect call, its type as
/// `{name}_type` or `type_{index}`
pub(crate) fn returns(
    view: &BinaryView,
    owner: &str,
    results: &[ValueKind],
    values: Option<&[ValType]>,
    pointer: usize,
) -> ReturnValue {
    let value = |nth: usize| values.and_then(|values| values.get(nth));
    let [_, _, ..] = results else {
        return results
            .first()
            .map_or_else(Type::void, |only| slot_type(*only, value(0), pointer))
            .into();
    };

    let mut builder = StructureBuilder::new();
    let mut components = Vec::new();
    for (nth, kind) in results.iter().enumerate() {
        let ty = slot_type(*kind, value(nth), pointer);
        let offset = nth as u64 * lift::SLOT;
        builder.insert(
            &ty,
            &format!("result{nth}"),
            offset,
            false,
            MemberAccess::PublicAccess,
            MemberScope::NoScope,
        );
        if nth < arch::RESULT_REGISTERS as usize {
            components.push(ValueLocationComponent {
                variable: Variable::from_register_id(arch::result_register(nth as u32)),
                offset: offset as i64,
                size: Some(ty.width()),
            });
        }
    }

    let name = QualifiedName::from(format!("{owner}_results"));
    let structure = Type::structure(&builder.finalize());
    let registered = view.define_auto_type(name, "wasm", &structure);
    ReturnValue {
        ty: Conf::new(
            Type::named_type_from_type(registered, &structure),
            MAX_CONFIDENCE,
        ),
        location: Some(Conf::new(
            ValueLocation {
                components,
                indirect: false,
                returned_pointer: None,
            },
            MAX_CONFIDENCE,
        )),
    }
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
    settings::preset(view);

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
    if !module::restored(arch::view_id(view)) {
        upgrade::stamp(view);
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
    let Some(arch) = view
        .default_arch()
        .filter(|arch| arch::is_ours(&arch.name()))
    else {
        return;
    };
    if let Some(platform) = platform_for(&arch) {
        let modules = module::all(arch::view_id(view));
        if !module::restored(arch::view_id(view))
            && restore_call_effects(view, &modules, &platform) != 0
        {
            view.update_analysis();
        }
    }
    drop_invented_functions(view);
}

pub(crate) fn name_debug_locals(view: &BinaryView) {
    let id = arch::view_id(view);
    for function in view.functions().iter() {
        if debug::has_assignments(id, function.start()) {
            function.reanalyze(FunctionUpdateType::FullAutoFunctionUpdate);
        }
    }
}

fn restore_call_effects(view: &BinaryView, modules: &[Arc<Module>], platform: &Platform) -> usize {
    let registers = clobbered(platform);
    let wanted: BTreeSet<u32> = registers.iter().map(|reg| reg.id().0).collect();
    let restore = |function: &Function| {
        let current = function.clobbered_registers();
        let held: BTreeSet<u32> = current.contents.iter().map(|reg| reg.id().0).collect();
        let lost = current.confidence < MAX_CONFIDENCE || held != wanted;
        if lost {
            function.set_user_clobbered_registers(registers.iter().cloned(), MAX_CONFIDENCE);
        }
        lost
    };
    let mut reset = 0usize;
    for module in modules {
        for (index, _) in module.imports() {
            let Some(stub) = view.function_at(platform, module.layout.import_address(index)) else {
                continue;
            };
            let pure = stub.is_pure();
            let impure = pure.contents || pure.confidence < MAX_CONFIDENCE;
            if impure {
                stub.set_user_pure(Conf::new(false, MAX_CONFIDENCE));
            }
            reset += usize::from(restore(&stub) | impure);
        }
        for (_, info) in module.functions() {
            if let Some(function) = view.function_at(platform, info.entry) {
                reset += usize::from(restore(&function));
            }
        }
    }
    if reset != 0 {
        tracing::debug!("wasm view: call effects set again on {reset} functions");
    }
    reset
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

/// The core's own workflow ends with `core.module.deleteUnusedAutoFunctions`, which is this same
/// job for a format that cannot say where its functions are, so ours runs next to it
const ROOT: &str = "core.module.metaAnalysis";
pub(crate) const WORKFLOW: &str = "wasm.module.analysis";
const SWEEP: &str = "wasm.module.dropInventedFunctions";
const EFFECTS: &str = "wasm.module.restoreCallEffects";
const DEBUG_INFO: &str = "core.module.loadDebugInfo";
const FUNCTION_ROOT: &str = "core.function.metaAnalysis";
const NAME_LOCALS: &str = "wasm.function.nameLocals";
const MEDIUM_LEVEL: &str = "core.function.generateMediumLevelIL";

fn register_function_workflow() {
    let config = activity::Config::action(
        NAME_LOCALS,
        "Name locals",
        "Names the variables a function's locals split into, from its DWARF locations and the \
         module's name section.",
    )
    .eligibility(activity::Eligibility::auto().run_once(false));
    let activity = Activity::new_with_action(config, |context: &AnalysisContext| {
        let function = context.function();
        if arch::is_ours(&function.arch().name())
            && let Some(il) = context.mlil_function()
        {
            name_locals(&context.view(), &function, &il);
        }
    });
    let registered = Workflow::cloned(FUNCTION_ROOT)
        .ok_or(())
        .and_then(|workflow| workflow.activity_after(&activity, MEDIUM_LEVEL))
        .and_then(|workflow| workflow.register());
    if registered.is_err() {
        tracing::warn!(
            "wasm view: {NAME_LOCALS} could not join {FUNCTION_ROOT}, so locals keep the names \
             analysis gives them"
        );
    }
}

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
    let sweep = Activity::new_with_action(config, |context: &AnalysisContext| {
        drop_invented_functions(&context.view());
        note_addresses_inside_strings(&context.view());
    });

    let config = activity::Config::action(
        EFFECTS,
        "Restore what calls clobber",
        "Sets what each function clobbers back to the argument, result and exception registers, \
         after applying debug info has replaced it with registers inferred from the body.",
    )
    .eligibility(activity::Eligibility::auto());
    let effects = Activity::new_with_action(config, |context: &AnalysisContext| {
        let view = context.view();
        let id = arch::view_id(&view);
        if let Some(platform) = view
            .default_arch()
            .filter(|arch| arch::is_ours(&arch.name()))
            .and_then(|arch| platform_for(&arch))
            && !module::restored(id)
        {
            restore_call_effects(&view, &module::all(id), &platform);
        }
    });

    let once = format!("so {SWEEP} and {EFFECTS} run once, when analysis ends");
    let Some(core) = Workflow::get(ROOT) else {
        tracing::warn!("wasm view: there is no {ROOT} to extend, {once}");
        return;
    };
    let build = |sweep_after: &str, effects_after: Option<&str>| {
        let mut builder = core
            .clone_to(WORKFLOW)
            .register_activity(&sweep)?
            .register_activity(&effects)?
            .insert_after(sweep_after, [SWEEP])?;
        if let Some(anchor) = effects_after {
            builder = builder.insert_after(anchor, [EFFECTS])?;
        }
        Ok::<_, ()>(builder)
    };

    // Next to the core's own sweep, or failing that wherever an update ends, since which
    // activities a core carries is its business
    let anchors = [
        "core.module.deleteUnusedAutoFunctions",
        "core.module.finishUpdate",
        "core.module.advancedAnalysis",
        "core.module.baseAnalysis",
    ];
    let debug_info = core.contains(DEBUG_INFO).then_some(DEBUG_INFO);
    if debug_info.is_none() {
        tracing::warn!(
            "wasm view: no {DEBUG_INFO} in {ROOT}, so what calls clobber is only restored once \
             analysis ends"
        );
    }
    let Some(workflow) = anchors
        .iter()
        .find_map(|anchor| build(anchor, debug_info).ok())
    else {
        tracing::warn!("wasm view: nothing in {ROOT} to hang {SWEEP} on, {once}");
        return;
    };
    if workflow.register().is_err() {
        tracing::warn!("wasm view: {WORKFLOW} was rejected, {once}");
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
    register_function_workflow();
    register_binary_view_type(WasmViewType);
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

    #[test]
    fn a_name_on_several_storages_keeps_its_first_and_numbers_the_rest() {
        let register = |storage: i64| (false, storage);
        let holders = [
            ("n".to_owned(), register(514), 0),
            ("n".to_owned(), register(1026), 40),
            ("i".to_owned(), register(1030), 30),
            ("i".to_owned(), register(1034), 5),
            ("i".to_owned(), register(1030), 10),
            ("i".to_owned(), register(1038), 20),
            ("i_1".to_owned(), register(1042), 50),
            ("total".to_owned(), register(1046), 8),
            ("total".to_owned(), register(1046), 60),
            ("total".to_owned(), (true, 1046), 0),
        ];
        let renamed = distinguish(&holders);
        let named =
            |name: &str, storage: Storage| renamed.get(&(name.to_owned(), storage)).cloned();

        assert_eq!(
            named("n", register(514)),
            None,
            "a parameter keeps its name"
        );
        assert_eq!(named("n", register(1026)), Some("n_1".into()));
        assert_eq!(
            named("i", register(1034)),
            None,
            "the first set keeps the name"
        );
        assert_eq!(
            named("i", register(1030)),
            Some("i_2".into()),
            "numbered past a name already held"
        );
        assert_eq!(named("i", register(1038)), Some("i_3".into()));
        assert_eq!(named("i_1", register(1042)), None);
        assert_eq!(named("total", (true, 1046)), None);
        assert_eq!(
            named("total", register(1046)),
            Some("total_1".into()),
            "one storage however split, apart from a stack slot at the same number"
        );
    }

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

    fn check_memory(placed: Vec<(u64, u64, u64)>, end: u64) -> Vec<MemorySpan> {
        let layout = memory_layout(placed, end);

        let mut at = layout.first().map_or(0, |(offset, _, _)| *offset);
        assert!(at <= 1, "{layout:?} leaves more than address zero unmapped");
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
            [(1, 63, None), (64, 2, Some(500)), (66, 65470, None)],
            "only the segment's own stretch is backed"
        );
    }

    #[test]
    fn several_data_segments_keep_their_order_and_their_gaps() {
        let layout = check_memory(vec![(100, 10, 900), (8, 4, 800)], 200);
        assert_eq!(
            layout,
            [
                (1, 7, None),
                (8, 4, Some(800)),
                (12, 88, None),
                (100, 10, Some(900)),
                (110, 90, None)
            ]
        );
    }

    #[test]
    fn a_null_pointer_is_not_an_address_unless_something_is_there() {
        assert_eq!(check_memory(Vec::new(), 64), [(1, 63, None)]);
        assert_eq!(
            check_memory(vec![(1, 3, 100)], 8),
            [(1, 3, Some(100)), (4, 4, None)]
        );
        assert_eq!(
            check_memory(vec![(0, 2, 100)], 4),
            [(0, 2, Some(100)), (2, 2, None)]
        );
    }

    #[test]
    fn memory_with_nothing_in_it_maps_nothing() {
        assert_eq!(memory_layout(Vec::new(), 0), Vec::new());
        assert_eq!(
            memory_layout(vec![(0, 8, 100)], 0),
            Vec::new(),
            "a segment in no memory"
        );
    }

    #[test]
    fn data_past_the_end_of_memory_is_dropped_rather_than_wrapped() {
        assert_eq!(
            check_memory(vec![(60, 8, 100), (8, 4, 200)], 64),
            [(1, 7, None), (8, 4, Some(200)), (12, 52, None)]
        );
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
