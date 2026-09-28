//! DWARF import, from the module's own custom sections, the file its `external_debug_info` section
//! names, or an external file built from the same code, with split units from the `.dwo` files or
//! `.dwp` package beside it
//!
//! Binary Ninja's own DWARF plugin claims a `.wasm` and fails on it, since its `object` crate is
//! built without `wasm`; `gimli` reads the custom sections directly. An address is relative to the
//! code section payload, and `DW_AT_low_pc` points at a body's locals declaration rather than its
//! first instruction

use std::borrow::Borrow;
use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Read;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, PoisonError, RwLock};

use binaryninja::binary_view::{BinaryView, BinaryViewBase, MetadataStoreFlags};
use binaryninja::confidence::{Conf, MAX_CONFIDENCE};
use binaryninja::debuginfo::{
    CustomDebugInfoParser, DebugFunctionInfo, DebugInfo, DebugInfoParser,
};
use binaryninja::function::Function;
use binaryninja::rc::Ref;
use binaryninja::types::{
    BaseStructure, EnumerationBuilder, FunctionParameter, MemberAccess, MemberScope,
    NamedTypeReference, NamedTypeReferenceClass, QualifiedName, StructureBuilder, StructureType,
    Type, TypeClass,
};
use binaryninja::variable::{NamedVariableWithType, Variable};
use gimli::{AttributeValue, Dwarf, EndianSlice, LittleEndian, SectionId};
use wasmparser::Operator;

use crate::ViewId;
use crate::module::{self, FunctionInfo, Module, SectionSpan, Signature, ValueKind};
use crate::{arch, insn, lift, settings, view, wasi};

pub const NAME: &str = "WASM DWARF";

type Slice<'a> = EndianSlice<'a, LittleEndian>;
type Entry<'a> = gimli::DebuggingInformationEntry<Slice<'a>>;
type Described = (Option<String>, Option<Die>);
type Slot = (Option<String>, Passed);

const RETURN_SLOT: &str = "result";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Die {
    unit: usize,
    offset: gimli::UnitOffset,
}

impl Die {
    fn of(unit: usize, entry: &Entry) -> Self {
        Self {
            unit,
            offset: entry.offset(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Passed {
    Omitted,
    Value(Die),
    Indirect(Die),
    Split(Die),
    Variadic,
    Unknown,
}

#[derive(Clone, Copy)]
enum Element {
    Empty,
    One(Die),
    Many,
    Unknown,
}

struct Tree<'a> {
    dwarfs: Vec<&'a Dwarf<Slice<'a>>>,
    units: Vec<gimli::Unit<Slice<'a>>>,
    files: Vec<usize>,
    blocks: Vec<Range<usize>>,
    skeletons: HashMap<usize, usize>,
    signatures: HashMap<gimli::DebugTypeSignature, Die>,
    scopes: Vec<OnceCell<Scopes>>,
    unread: Option<u64>,
}

impl<'a> Tree<'a> {
    fn read(dwarfs: Vec<&'a Dwarf<Slice<'a>>>) -> Self {
        let mut tree = Self {
            dwarfs,
            units: Vec::new(),
            files: Vec::new(),
            blocks: Vec::new(),
            skeletons: HashMap::new(),
            signatures: HashMap::new(),
            scopes: Vec::new(),
            unread: None,
        };
        let mut unmatched: HashMap<gimli::DwoId, usize> = HashMap::new();
        for file in 0..tree.dwarfs.len() {
            let start = tree.units.len();
            for mut unit in units_of(tree.dwarfs[file], &mut tree.unread) {
                let index = tree.units.len();
                match (file, unit.dwo_id) {
                    (0, Some(id)) => {
                        unmatched.insert(id, index);
                    }
                    (_, Some(id)) => {
                        let Some(skeleton) = unmatched.remove(&id) else {
                            continue;
                        };
                        unit.copy_relocated_attributes(&tree.units[skeleton]);
                        tree.skeletons.insert(index, skeleton);
                    }
                    (_, None) => {}
                }
                if let gimli::UnitType::Type {
                    type_signature,
                    type_offset,
                }
                | gimli::UnitType::SplitType {
                    type_signature,
                    type_offset,
                } = unit.header.type_()
                {
                    tree.signatures.entry(type_signature).or_insert(Die {
                        unit: index,
                        offset: type_offset,
                    });
                }
                tree.units.push(unit);
                tree.files.push(file);
                tree.scopes.push(OnceCell::new());
            }
            tree.blocks.push(start..tree.units.len());
        }
        tree
    }

    fn dwarf(&self, unit: usize) -> &'a Dwarf<Slice<'a>> {
        self.dwarfs[self.files[unit]]
    }

    fn lines_of(&self, unit: usize) -> usize {
        self.skeletons.get(&unit).copied().unwrap_or(unit)
    }

    fn entry(&self, die: Die) -> Option<Entry<'a>> {
        self.units[die.unit].entry(die.offset).ok()
    }

    fn root(&self, unit: usize) -> Option<Entry<'a>> {
        let read = &self.units[unit];
        read.entry(read.root_offset()).ok()
    }

    fn address(&self, unit: usize, entry: &Entry, attr: gimli::DwAt) -> Option<u64> {
        let read = &self.units[unit];
        let address = self
            .dwarf(unit)
            .attr_address(read, entry.attr_value(attr)?)
            .ok()??;
        (!is_tombstone(address, read.encoding().address_size)).then_some(address)
    }

    fn reference(&self, unit: usize, value: AttributeValue<Slice>) -> Option<Die> {
        match value {
            AttributeValue::UnitRef(offset) => Some(Die { unit, offset }),
            AttributeValue::DebugInfoRef(at) => {
                let block = self.blocks[self.files[unit]].clone();
                let nth = self.units[block.clone()]
                    .partition_point(|read| {
                        read.header
                            .debug_info_offset()
                            .is_some_and(|start| start <= at)
                    })
                    .checked_sub(1)?;
                let unit = block.start + nth;
                let offset = at.to_unit_offset(&self.units[unit].header)?;
                Some(Die { unit, offset })
            }
            AttributeValue::DebugTypesRef(signature) => self.signatures.get(&signature).copied(),
            _ => None,
        }
    }

    fn follow(&self, unit: usize, entry: &Entry, attr: gimli::DwAt) -> Option<Die> {
        let target = self.reference(unit, entry.attr_value(attr)?)?;
        let signed = self
            .entry(target)
            .and_then(|skeleton| skeleton.attr_value(gimli::DW_AT_signature))
            .and_then(|signature| self.reference(target.unit, signature));
        Some(signed.unwrap_or(target))
    }

    fn type_of(&self, unit: usize, entry: &Entry) -> Option<Die> {
        self.follow(unit, entry, gimli::DW_AT_type)
    }

    fn string(&self, unit: usize, entry: &Entry<'a>, attr: gimli::DwAt) -> Option<String> {
        self.text(unit, entry.attr_value(attr)?)
    }

    fn file_name(
        &self,
        unit: usize,
        header: &gimli::LineProgramHeader<Slice<'a>>,
        index: u64,
    ) -> Option<String> {
        self.text(unit, header.file(index)?.path_name())
    }

    fn text(&self, unit: usize, value: AttributeValue<Slice<'a>>) -> Option<String> {
        let raw = self
            .dwarf(unit)
            .attr_string(&self.units[unit], value)
            .ok()?;
        // A name out of a debug section is no more trustworthy than one out of the module
        let name = module::clean(std::str::from_utf8(raw.slice()).ok()?);
        (!name.is_empty()).then_some(name)
    }

    fn name(&self, unit: usize, entry: &Entry<'a>) -> Option<String> {
        self.string(unit, entry, gimli::DW_AT_name)
    }

    fn qualified(&self, die: Die, name: &str) -> String {
        self.scopes[die.unit]
            .get_or_init(|| Scopes::of(self, die.unit))
            .qualified(die.offset.0, name)
    }

    fn encoding(&self, unit: usize) -> gimli::Encoding {
        self.units[unit].encoding()
    }
}

fn units_of<'a>(dwarf: &Dwarf<Slice<'a>>, unread: &mut Option<u64>) -> Vec<gimli::Unit<Slice<'a>>> {
    let mut units = Vec::new();
    let mut headers = dwarf.units();
    // `next` stops on the first malformed header rather than reporting it, since a
    // truncated section is a reason to keep what was read
    loop {
        match headers.next() {
            Ok(Some(header)) => units.extend(dwarf.unit(header).ok()),
            Err(gimli::Error::UnknownVersion(version)) => {
                unread.get_or_insert(version);
                break;
            }
            Ok(None) | Err(_) => break,
        }
    }
    let mut types = dwarf.type_units();
    while let Ok(Some(header)) = types.next() {
        units.extend(dwarf.unit(header).ok());
    }
    units
}

struct WasmDwarf;

impl CustomDebugInfoParser for WasmDwarf {
    fn is_valid(&self, view: &BinaryView) -> bool {
        // Only claim a module that carries DWARF or names a file beside it, so an ordinary build
        // is untouched
        let modules = modules_of(view);
        modules.iter().any(|module| carries_dwarf(module))
            || reference(view, &modules).is_some_and(|name| !sibling_files(view, &name).is_empty())
    }

    fn parse_info(
        &self,
        debug_info: &mut DebugInfo,
        view: &BinaryView,
        debug_file: &BinaryView,
        progress: Box<dyn Fn(usize, usize) -> Result<(), ()>>,
    ) -> bool {
        let modules = modules_of(view);
        if modules.is_empty() {
            return false;
        }
        let own = Source::new(file_image(view), &modules);
        let elsewhere = if debug_file.file().session_id() != view.file().session_id() {
            let Some(chosen) = external(debug_file, &own) else {
                return false;
            };
            Some(chosen)
        } else {
            (!own.has_dwarf())
                .then(|| sibling(view, &modules, &own))
                .flatten()
        };
        let carrier = elsewhere.as_ref().unwrap_or(&own);
        if !carrier.has_dwarf() {
            return false;
        }

        let mut tally = Tally::default();
        let mut registered = HashMap::new();
        let mut named = HashMap::new();
        let mut placed = HashSet::new();
        let mut seen = 0usize;
        let sources = settings::SOURCE_LINES.get(view);

        // An address is relative to the code section payload of the module carrying it, so each
        // nested module is loaded against its own
        for (ordinal, module) in modules.iter().enumerate() {
            let Some(base) = code_base(module) else {
                continue;
            };

            let mut imports: HashMap<String, u32> = HashMap::new();
            for (index, import) in module.imports() {
                imports.insert(import.field.clone(), index);
                imports.insert(module.name_of(index), index);
            }

            let load = |id: SectionId| -> Result<Slice, gimli::Error> {
                Ok(EndianSlice::new(
                    carrier.section(ordinal, id.name()),
                    LittleEndian,
                ))
            };

            let Ok(mut dwarf) = Dwarf::load(load) else {
                continue;
            };
            dwarf.populate_abbreviations_cache(gimli::AbbreviationsCacheStrategy::All);
            let images = split_images(view, &dwarf);
            let splits = split_dwarfs(&images, &dwarf);
            let tree = Tree::read(std::iter::once(&dwarf).chain(&splits).collect());
            if let Some(version) = tree.unread {
                tracing::warn!(
                    "wasm DWARF: stopped at a unit of DWARF version {version}, which is not read"
                );
            }

            let mut import = Import {
                tree: &tree,
                module,
                image: &own.image,
                imports: &imports,
                base,
                view,
                debug_info,
                registered: &mut registered,
                assigned: HashMap::new(),
                named: &mut named,
                built: HashMap::new(),
                building: HashSet::new(),
                cut: false,
                placed: &mut placed,
                tally: &mut tally,
                internalised: internalised(&tree),
                renumbered: renumbered(&tree, module, &own.image, base),
            };
            if import.renumbered {
                tracing::warn!(
                    "wasm DWARF: the locals were renumbered after the DWARF was written, as \
                     wasm-opt does, so a local is named only where the code agrees on which one \
                     it became"
                );
            }
            for unit in 0..tree.units.len() {
                seen += 1;
                if progress(seen, seen + 1).is_err() {
                    tracing::info!("wasm DWARF: import cancelled");
                    return false;
                }
                import.walk(unit);
                if sources {
                    import.lines(unit);
                }
            }
        }

        tracing::info!(
            "wasm DWARF: {} functions, {} types, {} data variables, {} named locals, \
             {} import prototypes and {} source lines from {seen} units",
            tally.functions,
            tally.types,
            tally.variables,
            tally.locals,
            tally.stubs,
            tally.lines,
        );
        if tally.functions == 0 && tally.unplaced != 0 {
            tracing::warn!(
                "wasm DWARF: none of the {} functions it describes spans a function body here, \
                 so the code was changed after it was built and its functions and lines are not \
                 applied",
                tally.unplaced,
            );
        } else if tally.unnamed != 0 || tally.unplaced != 0 {
            tracing::debug!(
                "wasm DWARF: {} subprograms with no name anywhere and {} whose range is no \
                 function body",
                tally.unnamed,
                tally.unplaced,
            );
        }
        if tally.unmapped != 0 {
            tracing::warn!(
                "wasm DWARF: {} data variables lie past the linear memory this view maps and are \
                 not named",
                tally.unmapped,
            );
        }
        store_assigned(view);
        view::name_debug_locals(view);
        tally.functions != 0 || tally.types != 0 || tally.variables != 0
    }
}

#[derive(Default)]
struct Tally {
    functions: usize,
    types: usize,
    variables: usize,
    locals: usize,
    stubs: usize,
    lines: usize,
    unnamed: usize,
    unplaced: usize,
    unmapped: usize,
}

struct Import<'a, 'b> {
    tree: &'a Tree<'a>,
    module: &'a Module,
    image: &'a [u8],
    imports: &'a HashMap<String, u32>,
    base: u64,
    view: &'a BinaryView,
    debug_info: &'b DebugInfo,
    registered: &'b mut HashMap<String, Option<u64>>,
    assigned: HashMap<Die, String>,
    named: &'b mut HashMap<String, Ref<Type>>,
    built: HashMap<Die, Option<Ref<Type>>>,
    building: HashSet<Die>,
    cut: bool,
    placed: &'b mut HashSet<u64>,
    tally: &'b mut Tally,
    internalised: bool,
    renumbered: bool,
}

impl<'a> Import<'a, '_> {
    fn walk(&mut self, unit: usize) {
        let tree = self.tree;
        let mut entries = tree.units[unit].entries();
        while let Ok(Some(entry)) = entries.next_dfs() {
            let die = Die::of(unit, entry);
            match entry.tag() {
                gimli::DW_TAG_subprogram => self.subprogram(die, entry),
                gimli::DW_TAG_variable => self.variable(die, entry),
                gimli::DW_TAG_structure_type
                | gimli::DW_TAG_class_type
                | gimli::DW_TAG_union_type
                | gimli::DW_TAG_enumeration_type
                | gimli::DW_TAG_typedef
                    if !is_declaration(entry) && has_name(entry) =>
                {
                    self.build_type(die, 0);
                }
                _ => {}
            }
        }
    }

    fn subprogram(&mut self, die: Die, entry: &Entry<'a>) {
        let Some(low_pc) = self.tree.address(die.unit, entry, gimli::DW_AT_low_pc) else {
            self.stub(die, entry);
            return;
        };

        let Some((index, body)) = body_at(self.module, self.base.wrapping_add(low_pc))
            .filter(|(_, body)| spans(self.tree, die.unit, entry, low_pc, body))
        else {
            self.tally.unplaced += 1;
            return;
        };
        if !self.placed.insert(body.entry) {
            return;
        }

        let declaration = origin(self.tree, die).unwrap_or(die);
        let raw = self
            .linkage_at(die)
            .or_else(|| self.linkage_at(declaration));
        let Some(name) = self.name_at(declaration).or_else(|| raw.clone()) else {
            self.tally.unnamed += 1;
            return;
        };
        let name = self.tree.qualified(declaration, &name);
        let raw = raw.unwrap_or_else(|| name.clone());

        let prototype = self.prototype(die, entry, index, &body.signature);
        let (slots, assigned, frame) = self.locals(die, entry, index, body);
        let named: HashSet<&String> = assigned
            .values()
            .flat_map(|(_, names)| names)
            .map(|(name, ..)| name)
            .collect();
        self.tally.locals += slots.len() + named.len();
        keep_assigned(arch::view_id(self.view), body.entry, assigned);

        let info = DebugFunctionInfo::new(
            Some(name.clone()),
            Some(name),
            Some(raw),
            Some(prototype),
            Some(body.entry),
            None,
            // The components marshalling in these bindings is wrong for a non-empty slice
            Vec::new(),
            slots,
        );
        if self.debug_info.add_function(&info) {
            self.tally.functions += 1;
        }
        self.annotate(body.entry, declaration, frame);
    }

    fn stub(&mut self, die: Die, entry: &Entry<'a>) -> Option<()> {
        if !is_declaration(entry) {
            return None;
        }
        let name = self.tree.name(die.unit, entry)?;
        if self.linkage_at(die).is_some_and(|linkage| linkage != name) {
            return None;
        }
        let index = *self.imports.get(name.as_str())?;
        let import = self
            .module
            .import(index)
            .filter(|import| !wasi::specifies(import))?;
        let address = self.module.layout.import_address(index);
        if !self.placed.insert(address) {
            return None;
        }

        let prototype = self.prototype(die, entry, index, &import.signature);
        let stub = self.view.functions_at(address).iter().next()?.to_owned();
        stub.set_user_type(&prototype);
        stub.set_user_pure(Conf::new(false, MAX_CONFIDENCE));
        self.tally.stubs += 1;
        Some(())
    }

    fn variable(&mut self, die: Die, entry: &Entry<'a>) {
        let Some(location) = self.location(die.unit, entry, gimli::DW_AT_location) else {
            return;
        };
        let (at, stored) = match location {
            Where::Memory(at) if self.module.layout.memory_mapped(at) => {
                (self.module.layout.memory_address(at), None)
            }
            Where::Stored(at, size) if self.module.layout.memory_mapped(at) => (
                self.module.layout.memory_address(at),
                Some(usize::from(size)),
            ),
            Where::Memory(_) | Where::Stored(..) => {
                self.tally.unmapped += 1;
                return;
            }
            Where::Global(index) if index < self.module.globals().count() as u32 => {
                (self.module.layout.global_address(index), None)
            }
            _ => return,
        };

        let (Some(name), Some(ty)) = described(self.tree, die) else {
            return;
        };
        let Some(ty) = self.build_type(ty, 1) else {
            return;
        };
        let ty = match stored {
            Some(size) if ty.width() != size as u64 => Type::int(size, false),
            _ => ty,
        };

        let declaration = origin(self.tree, die).unwrap_or(die);
        let name = self.tree.qualified(declaration, &name);
        self.debug_info.add_data_variable(at, &ty, Some(&name), &[]);
        self.tally.variables += 1;
    }

    fn locals(
        &mut self,
        die: Die,
        entry: &Entry<'a>,
        function: u32,
        body: &FunctionInfo,
    ) -> (Vec<NamedVariableWithType>, Assignments, Vec<String>) {
        let tree = self.tree;
        let params = body.signature.params.len() as u32;
        let code = body_code(self.module, self.image, body);
        let frame_base = match self.frame_base(die.unit, entry, function) {
            Some(Where::Local(local) | Where::PointedByLocal(local)) => body
                .frame
                .filter(|(holder, _)| {
                    *holder == local || code.is_some_and(|code| copies(code, *holder, local))
                })
                .map(|(_, size)| size),
            Some(Where::Global(global) | Where::PointedByGlobal(global))
                if Some(global) == self.module.stack_pointer =>
            {
                body.frame.map(|(_, size)| size)
            }
            _ => None,
        };
        let within = body.start - self.base..body.end - self.base;
        let (held, starts) = held_variables(tree, die, within);

        let mut stacked = Vec::new();
        let outer_first = held.iter().filter(|variable| !variable.inlined);
        for variable in outer_first.chain(held.iter().filter(|variable| variable.inlined)) {
            if let [(None, Where::Frame(at))] = variable.places[..]
                && let (Some(name), ty) = described(tree, variable.die)
            {
                let ty = ty.and_then(|ty| self.build_type(ty, 1));
                stacked.push((at, name, ty, variable.inlined));
            }
        }
        let mut frame = Vec::new();
        let slots = stack_slots(stacked, frame_base, &mut frame);

        let sets = code.map_or_else(BTreeMap::new, |code| {
            let at = body.entry - self.base;
            assignments(
                self.module,
                code,
                at,
                params,
                &held,
                &starts,
                self.renumbered,
            )
        });
        let kinds = self.module.local_kinds(function);
        let mut assigned = HashMap::new();
        for (address, (local, candidates)) in sets {
            let Some(&kind) = kinds.get(local as usize) else {
                continue;
            };
            let mut names: Vec<Assigned> = Vec::new();
            for (nth, pointed) in candidates {
                let (Some(name), ty) = described(tree, held[nth].die) else {
                    continue;
                };
                let inlined = held[nth].inlined;
                if let Some((_, _, known)) = names.iter_mut().find(|(known, ..)| *known == name) {
                    *known &= inlined;
                    continue;
                }
                let passed = ty.map(|ty| {
                    if pointed {
                        Passed::Indirect(ty)
                    } else {
                        Passed::Value(ty)
                    }
                });
                let ty = passed.and_then(|passed| self.carried_as(passed, kind));
                if ty.is_some() || !self.renumbered {
                    names.push((name, ty, inlined));
                }
            }
            if !names.is_empty() {
                assigned.insert(self.base.wrapping_add(address), (local, names));
            }
        }
        (slots, assigned, frame)
    }

    fn annotate(&self, entry: u64, declaration: Die, frame: Vec<String>) {
        let mut notes = Vec::new();
        if let Some(at) = self.declared_at(declaration) {
            notes.push(format!("wasm: declared at {at}"));
        }
        if !frame.is_empty() {
            notes.push(format!("wasm: frame:\n  {}", frame.join("\n  ")));
        }
        if notes.is_empty() {
            return;
        }

        let Some(function) = self.function_at(entry) else {
            return;
        };
        let existing = function.comment();
        if existing.contains("wasm: frame:") || existing.contains("wasm: declared at") {
            return;
        }
        let notes = notes.join("\n");
        function.set_comment(&if existing.is_empty() {
            notes
        } else {
            format!("{existing}\n{notes}")
        });
    }

    fn lines(&mut self, unit: usize) {
        let tree = self.tree;
        let read = &tree.units[unit];
        if let gimli::UnitType::Type { .. } | gimli::UnitType::SplitType { .. } =
            read.header.type_()
        {
            return;
        }
        let Some(program) = read.line_program.clone() else {
            return;
        };
        let mut names: HashMap<u64, String> = HashMap::new();
        let mut sequence: Vec<(u64, u64, u64)> = Vec::new();
        let mut rows = program.rows();

        while let Ok(Some((header, row))) = rows.next_row() {
            let address = self.base.wrapping_add(row.address());
            if row.end_sequence() {
                self.sequence(&sequence, address, &names);
                sequence.clear();
                continue;
            }
            let (Some(line), true) = (row.line(), row.is_stmt()) else {
                continue;
            };
            let (file, line) = (row.file_index(), line.get());
            if sequence
                .last()
                .is_some_and(|&(_, before, at)| (before, at) == (file, line))
            {
                continue;
            }
            names
                .entry(file)
                .or_insert_with(|| tree.file_name(unit, header, file).unwrap_or_default());
            sequence.push((address, file, line));
        }
    }

    fn sequence(&mut self, rows: &[(u64, u64, u64)], end: u64, names: &HashMap<u64, String>) {
        let Some(info) = sequence_body(self.module, rows, end) else {
            return;
        };
        let Some(function) = self.function_at(info.entry) else {
            return;
        };
        for &(address, file, line) in rows {
            let name = names.get(&file).map_or("", String::as_str);
            function.set_comment_at(address.max(info.entry), &format!("{name}:{line}"));
            self.tally.lines += 1;
        }
    }

    fn function_at(&self, entry: u64) -> Option<Ref<Function>> {
        self.view
            .functions_at(entry)
            .iter()
            .next()
            .map(|f| f.clone())
    }

    fn declared_at(&self, declaration: Die) -> Option<String> {
        let entry = self.tree.entry(declaration)?;
        let index = udata(&entry, gimli::DW_AT_decl_file)?;
        let line = udata(&entry, gimli::DW_AT_decl_line)?;
        let unit = self.tree.lines_of(declaration.unit);
        let program = self.tree.units[unit].line_program.as_ref()?;
        let name = self.tree.file_name(unit, program.header(), index)?;
        Some(format!("{name}:{line}"))
    }

    fn prototype(
        &mut self,
        die: Die,
        entry: &Entry<'a>,
        index: u32,
        signature: &Signature,
    ) -> Ref<Type> {
        let low_pc = self.tree.address(die.unit, entry, gimli::DW_AT_low_pc);
        let first = low_pc
            .zip(self.module.function(index))
            .and_then(|(low, info)| low.checked_add(info.entry.checked_sub(info.start)?));
        let fixed = self.module.import(index).is_some() || self.module.address_taken(index);
        let spilled = match self.frame_base(die.unit, entry, index) {
            Some(Where::Local(frame)) => self
                .module
                .function(index)
                .and_then(|info| body_code(self.module, self.image, info))
                .map(|code| spills(code, frame, &signature.params))
                .unwrap_or_default(),
            _ => BTreeMap::new(),
        };
        let (slots, returned) = parameter_slots(
            self.tree,
            die,
            signature,
            &Start { low_pc, first },
            self.internalised,
            fixed,
            &spilled,
        );

        let pointer = self.module.layout.pointer;
        let values = self.module.value_types(index);
        let mut parameters = Vec::with_capacity(signature.params.len());
        for (nth, kind) in signature.params.iter().enumerate() {
            let (name, passed) = slots[nth].clone();
            let name = name
                .or_else(|| self.module.local_name(index, nth as u32).map(str::to_owned))
                .unwrap_or_else(|| format!("arg{nth}"));
            let value = values.and_then(|values| values.params().get(nth));
            let ty = self
                .carried_as(passed, *kind)
                .unwrap_or_else(|| view::slot_type(*kind, value, pointer));
            parameters.push(FunctionParameter::new(ty, name, None));
        }

        let results = values.map(wasmparser::FuncType::results);
        let returns = match signature.results.as_slice() {
            [] => Type::void().into(),
            [only] => match returned {
                Passed::Indirect(_) => None,
                returned => self.carried_as(returned, *only),
            }
            .unwrap_or_else(|| {
                view::slot_type(*only, results.and_then(|results| results.first()), pointer)
            })
            .into(),
            many => view::returns(
                self.view,
                &self.module.name_of(index),
                many,
                results,
                pointer,
            ),
        };

        Type::function(returns, parameters, false)
    }

    fn carried_as(&mut self, passed: Passed, kind: ValueKind) -> Option<Ref<Type>> {
        let pointer = self.module.layout.pointer;
        let address = lift::width(kind, pointer) == pointer && !kind.is_float();
        match passed {
            Passed::Value(at) => self.fitting_type(at, kind),
            Passed::Indirect(at) if address => {
                let target = self.build_type(at, 0)?;
                Some(Type::pointer_of_width(&target, pointer, false, false, None))
            }
            Passed::Variadic if address => Some(Type::pointer_of_width(
                &Type::void(),
                pointer,
                false,
                false,
                None,
            )),
            Passed::Indirect(_)
            | Passed::Variadic
            | Passed::Split(_)
            | Passed::Omitted
            | Passed::Unknown => None,
        }
    }

    fn points_at_code(&self, die: Die, entry: &Entry<'a>) -> bool {
        let tree = self.tree;
        underlying(tree, tree.type_of(die.unit, entry))
            .is_some_and(|(_, target)| target.tag() == gimli::DW_TAG_subroutine_type)
    }

    fn fitting_type(&mut self, die: Die, kind: ValueKind) -> Option<Ref<Type>> {
        let (_, entry) = underlying(self.tree, Some(die))?;
        if is_aggregate(entry.tag()) {
            return None;
        }

        let float = matches!(
            entry.attr_value(gimli::DW_AT_encoding),
            Some(AttributeValue::Encoding(gimli::DW_ATE_float))
        );
        let widened = kind == ValueKind::I32
            && !float
            && matches!(
                entry.tag(),
                gimli::DW_TAG_base_type | gimli::DW_TAG_enumeration_type
            );
        let width = lift::width(kind, self.module.layout.pointer) as u64;
        self.build_type(die, 0).filter(|ty| {
            kind.is_float() == float
                && (ty.width() == width || (widened && (1..width).contains(&ty.width())))
        })
    }

    fn build_type(&mut self, die: Die, depth: usize) -> Option<Ref<Type>> {
        if let Some(known) = self.built.get(&die) {
            return known.clone();
        }
        if depth >= MAX_TYPE_DEPTH || !self.building.insert(die) {
            self.cut = true;
            return None;
        }
        let outer = std::mem::replace(&mut self.cut, false);
        let built = self.construct(die, depth);
        self.building.remove(&die);
        if !self.cut {
            self.built.insert(die, built.clone());
        }
        self.cut |= outer;
        built
    }

    fn construct(&mut self, die: Die, depth: usize) -> Option<Ref<Type>> {
        let entry = self.tree.entry(die)?;
        let pointer = self.module.layout.pointer;

        match entry.tag() {
            gimli::DW_TAG_base_type => base_type(&entry),
            tag if is_pointer_tag(tag) => {
                if self.points_at_code(die, &entry) {
                    return Some(Type::named_int(pointer, false, "funcref"));
                }
                let target = self
                    .named_target(die, &entry)
                    .or_else(|| self.type_at(die.unit, &entry, depth))
                    .unwrap_or_else(Type::void);
                Some(Type::pointer_of_width(
                    target.as_ref(),
                    pointer,
                    false,
                    false,
                    None,
                ))
            }
            gimli::DW_TAG_structure_type => {
                self.aggregate(die, &entry, depth, StructureType::StructStructureType)
            }
            gimli::DW_TAG_class_type => {
                self.aggregate(die, &entry, depth, StructureType::ClassStructureType)
            }
            gimli::DW_TAG_union_type => {
                self.aggregate(die, &entry, depth, StructureType::UnionStructureType)
            }
            gimli::DW_TAG_enumeration_type => self.enumeration(die, &entry),
            gimli::DW_TAG_array_type => self.array(die, &entry, depth),
            gimli::DW_TAG_subroutine_type => self.subroutine(die, &entry, depth),
            gimli::DW_TAG_typedef => self.typedef(die, &entry, depth),
            gimli::DW_TAG_const_type
            | gimli::DW_TAG_volatile_type
            | gimli::DW_TAG_restrict_type
            | gimli::DW_TAG_atomic_type => self.type_at(die.unit, &entry, depth),
            _ => None,
        }
    }

    fn aggregate(
        &mut self,
        die: Die,
        entry: &Entry<'a>,
        depth: usize,
        kind: StructureType,
    ) -> Option<Ref<Type>> {
        let class = match kind {
            StructureType::ClassStructureType => NamedTypeReferenceClass::ClassNamedTypeClass,
            StructureType::UnionStructureType => NamedTypeReferenceClass::UnionNamedTypeClass,
            _ => NamedTypeReferenceClass::StructNamedTypeClass,
        };
        let Some(name) = self.aggregate_name(die, entry) else {
            return self.structure(die, entry, depth, kind);
        };
        if is_declaration(entry) || self.registered.contains_key(&name) {
            return Some(self.reference(class, &name));
        }
        self.registered
            .insert(name.clone(), udata(entry, gimli::DW_AT_byte_size));

        match self.structure(die, entry, depth, kind) {
            Some(body) => Some(self.register(class, &name, body)),
            None => {
                self.registered.remove(&name);
                None
            }
        }
    }

    fn structure(
        &mut self,
        die: Die,
        entry: &Entry<'a>,
        depth: usize,
        kind: StructureType,
    ) -> Option<Ref<Type>> {
        let tree = self.tree;
        let mut builder = StructureBuilder::new();
        builder.structure_type(kind);
        if let Some(size) = udata(entry, gimli::DW_AT_byte_size) {
            builder.width(size);
        }

        let mut bases = Vec::new();
        let mut members = 0usize;
        for child in &children(tree, die) {
            match child.tag() {
                gimli::DW_TAG_member => {}
                gimli::DW_TAG_inheritance => {
                    let Some(at) = member_offset(tree, die.unit, child) else {
                        continue;
                    };
                    let width = tree
                        .type_of(die.unit, child)
                        .and_then(|base| tree.entry(base))
                        .and_then(|base| udata(&base, gimli::DW_AT_byte_size))
                        .unwrap_or(0);
                    if let Some(ty) = self.type_at(die.unit, child, depth) {
                        bases.push((ty, at, width));
                    }
                    continue;
                }
                _ => continue,
            }

            if is_declaration(child) {
                continue;
            }
            let name = tree
                .name(die.unit, child)
                .unwrap_or_else(|| format!("field{members}"));

            let width =
                udata(child, gimli::DW_AT_bit_size).and_then(|bits| u8::try_from(bits).ok());
            let bits = member_bits(tree, die.unit, child);
            let (Some(bits), Some(ty)) = (bits, self.type_at(die.unit, child, depth)) else {
                continue;
            };
            builder.insert_bitwise(
                &ty,
                &name,
                bits,
                width,
                false,
                MemberAccess::PublicAccess,
                MemberScope::NoScope,
            );
            members += 1;
        }

        let bases: Vec<BaseStructure> = bases
            .iter()
            .filter_map(|(ty, at, width)| {
                Some(BaseStructure::new(
                    ty.get_named_type_reference()?,
                    *at,
                    *width,
                ))
            })
            .collect();
        if !bases.is_empty() {
            builder.base_structures(&bases);
        }

        (members != 0 || !bases.is_empty() || builder.current_width() != 0)
            .then(|| Type::structure(&builder.finalize()))
    }

    fn enumeration(&mut self, die: Die, entry: &Entry<'a>) -> Option<Ref<Type>> {
        let tree = self.tree;
        let underlying = underlying(tree, tree.type_of(die.unit, entry)).map(|(_, at)| at);
        let width = udata(entry, gimli::DW_AT_byte_size)
            .or_else(|| {
                underlying
                    .as_ref()
                    .and_then(|at| udata(at, gimli::DW_AT_byte_size))
            })
            .unwrap_or(4);
        let width = NonZeroUsize::new(width as usize)?;
        let class = NamedTypeReferenceClass::EnumNamedTypeClass;
        let name = self.registered_name(die, entry);
        if let Some(name) = &name {
            if is_declaration(entry) || self.registered.contains_key(name) {
                return Some(self.reference(class, name));
            }
            self.registered.insert(name.clone(), None);
        }

        let mut builder = EnumerationBuilder::new();
        for child in &children(tree, die) {
            if child.tag() != gimli::DW_TAG_enumerator {
                continue;
            }
            let (Some(member), Some(value)) = (
                tree.name(die.unit, child),
                constant(child, gimli::DW_AT_const_value),
            ) else {
                continue;
            };
            builder.insert(&member, value);
        }

        let signed = underlying.as_ref().is_none_or(|at| {
            !matches!(
                at.attr_value(gimli::DW_AT_encoding),
                Some(AttributeValue::Encoding(
                    gimli::DW_ATE_unsigned | gimli::DW_ATE_unsigned_char | gimli::DW_ATE_boolean
                ))
            )
        });
        let body = Type::enumeration(&builder.finalize(), width, signed);
        match name {
            Some(name) => Some(self.register(class, &name, body)),
            None => Some(body),
        }
    }

    fn typedef(&mut self, die: Die, entry: &Entry<'a>, depth: usize) -> Option<Ref<Type>> {
        let class = NamedTypeReferenceClass::TypedefNamedTypeClass;
        let Some(declared) = self.registered_name(die, entry) else {
            return self.type_at(die.unit, entry, depth);
        };
        let target = self
            .tree
            .type_of(die.unit, entry)
            .and_then(|at| Some((at, self.tree.entry(at)?)));
        if let Some((at, target)) = target
            && self.registered_name(at, &target).as_ref() == Some(&declared)
        {
            return self.type_at(die.unit, entry, depth);
        }
        let name = self.typedef_name(die, entry)?;
        if self.registered.contains_key(&name) {
            return Some(self.reference(class, &name));
        }
        self.registered
            .insert(name.clone(), byte_size(self.tree, die, 0));

        match self.type_at(die.unit, entry, depth) {
            Some(body) => Some(self.register(class, &name, body)),
            None => {
                self.registered.remove(&name);
                None
            }
        }
    }

    fn array(&mut self, die: Die, entry: &Entry<'a>, depth: usize) -> Option<Ref<Type>> {
        let element = self.type_at(die.unit, entry, depth)?;
        Some(
            dimensions(self.tree, die)
                .iter()
                .rev()
                .fold(element, |inner, count| {
                    Type::array(inner.as_ref(), count.unwrap_or(0))
                }),
        )
    }

    fn subroutine(&mut self, die: Die, entry: &Entry<'a>, depth: usize) -> Option<Ref<Type>> {
        let returns = self
            .type_at(die.unit, entry, depth)
            .unwrap_or_else(Type::void);

        let mut parameters = Vec::new();
        let mut variadic = false;
        for child in &children(self.tree, die) {
            match child.tag() {
                gimli::DW_TAG_unspecified_parameters => variadic = true,
                gimli::DW_TAG_formal_parameter => {
                    let ty = self
                        .type_at(die.unit, child, depth)
                        .unwrap_or_else(Type::void);
                    parameters.push(FunctionParameter::new(
                        ty,
                        format!("arg{}", parameters.len()),
                        None,
                    ));
                }
                _ => continue,
            }
        }

        Some(Type::function(returns.as_ref(), parameters, variadic))
    }

    fn register(
        &mut self,
        class: NamedTypeReferenceClass,
        name: &str,
        body: Ref<Type>,
    ) -> Ref<Type> {
        wasi::give_way(self.view, name);
        self.debug_info.add_type(name, &body, &[]);
        self.tally.types += 1;
        self.named.insert(name.to_owned(), body);
        self.reference(class, name)
    }

    fn reference(&self, class: NamedTypeReferenceClass, name: &str) -> Ref<Type> {
        match self.named.get(name) {
            Some(body) => Type::named_type_from_type(name, body),
            None => Type::named_type(&NamedTypeReference::new(class, name)),
        }
    }

    fn named_target(&mut self, die: Die, entry: &Entry<'a>) -> Option<Ref<Type>> {
        let at = unqualified(self.tree, self.tree.type_of(die.unit, entry))?;
        self.named_at(at)
    }

    fn named_at(&mut self, mut at: Die) -> Option<Ref<Type>> {
        let tree = self.tree;
        let mut target = tree.entry(at)?;
        for _ in 0..MAX_TYPE_DEPTH {
            if target.tag() != gimli::DW_TAG_typedef {
                break;
            }
            let Some((inner, aliased)) = tree
                .type_of(at.unit, &target)
                .and_then(|inner| Some((inner, tree.entry(inner)?)))
            else {
                break;
            };
            let name = self.registered_name(inner, &aliased);
            if name.is_none() || name != self.registered_name(at, &target) {
                break;
            }
            (at, target) = (inner, aliased);
        }
        let (class, name) = match target.tag() {
            gimli::DW_TAG_structure_type => (
                NamedTypeReferenceClass::StructNamedTypeClass,
                self.aggregate_name(at, &target),
            ),
            gimli::DW_TAG_class_type => (
                NamedTypeReferenceClass::ClassNamedTypeClass,
                self.aggregate_name(at, &target),
            ),
            gimli::DW_TAG_union_type => (
                NamedTypeReferenceClass::UnionNamedTypeClass,
                self.aggregate_name(at, &target),
            ),
            gimli::DW_TAG_enumeration_type => (
                NamedTypeReferenceClass::EnumNamedTypeClass,
                self.registered_name(at, &target),
            ),
            gimli::DW_TAG_typedef => (
                NamedTypeReferenceClass::TypedefNamedTypeClass,
                self.typedef_name(at, &target),
            ),
            _ => return None,
        };
        Some(self.reference(class, &name?))
    }

    fn registered_name(&self, die: Die, entry: &Entry<'a>) -> Option<String> {
        let name = self.tree.name(die.unit, entry)?;
        Some(self.tree.qualified(die, &name))
    }

    fn aggregate_name(&mut self, die: Die, entry: &Entry<'a>) -> Option<String> {
        self.distinct_name(die, entry, udata(entry, gimli::DW_AT_byte_size))
    }

    fn typedef_name(&mut self, die: Die, entry: &Entry<'a>) -> Option<String> {
        self.distinct_name(die, entry, byte_size(self.tree, die, 0))
    }

    fn distinct_name(&mut self, die: Die, entry: &Entry<'a>, size: Option<u64>) -> Option<String> {
        if let Some(name) = self.assigned.get(&die) {
            return Some(name.clone());
        }
        let base = self.registered_name(die, entry)?;
        let mut name = base.clone();
        if !is_declaration(entry) && size.is_some() {
            let mut nth = 1;
            while self
                .registered
                .get(&name)
                .is_some_and(|known| known.is_some() && *known != size)
            {
                nth += 1;
                name = format!("{base}_{nth}");
            }
        }
        self.assigned.insert(die, name.clone());
        Some(name)
    }

    fn type_at(&mut self, unit: usize, entry: &Entry<'a>, depth: usize) -> Option<Ref<Type>> {
        let target = self.tree.type_of(unit, entry)?;
        self.build_type(target, depth + 1)
    }

    fn name_at(&self, die: Die) -> Option<String> {
        self.tree.name(die.unit, &self.tree.entry(die)?)
    }

    fn linkage_at(&self, die: Die) -> Option<String> {
        self.tree
            .string(die.unit, &self.tree.entry(die)?, gimli::DW_AT_linkage_name)
    }

    fn location(&self, unit: usize, entry: &Entry<'a>, attr: gimli::DwAt) -> Option<Where> {
        let AttributeValue::Exprloc(expression) = entry.attr_value(attr)? else {
            return None;
        };
        place(self.tree, unit, expression)
    }

    fn frame_base(&self, unit: usize, entry: &Entry<'a>, function: u32) -> Option<Where> {
        let base = self.location(unit, entry, gimli::DW_AT_frame_base)?;
        if !self.renumbered || !matches!(base, Where::Local(_) | Where::PointedByLocal(_)) {
            return Some(base);
        }
        let info = self.module.function(function)?;
        let (holder, size) = info.frame?;
        let code = body_code(self.module, self.image, info)?;
        (size > 0 && prologue_frame(code, self.module.stack_pointer?, holder))
            .then_some(Where::Local(holder))
    }
}

fn spans<'a>(
    tree: &Tree<'a>,
    unit: usize,
    entry: &Entry<'a>,
    low_pc: u64,
    body: &FunctionInfo,
) -> bool {
    let Some(value) = entry.attr_value(gimli::DW_AT_high_pc) else {
        return true;
    };
    let size = match value {
        AttributeValue::Addr(_) | AttributeValue::DebugAddrIndex(_) => tree
            .dwarf(unit)
            .attr_address(&tree.units[unit], value)
            .ok()
            .flatten()
            .and_then(|high| high.checked_sub(low_pc)),
        value => udata_of(value),
    };
    size.is_some() && size == body.end.checked_sub(body.start)
}

fn sequence_body<'m>(
    module: &'m Module,
    rows: &[(u64, u64, u64)],
    end: u64,
) -> Option<&'m FunctionInfo> {
    let &(first, ..) = rows.first()?;
    let (_, info) = module.body_covering(first)?;
    (info.end == end).then_some(info)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Where {
    Memory(u64),
    Stored(u64, u8),
    Local(u32),
    Global(u32),
    PointedByLocal(u32),
    PointedByGlobal(u32),
    Frame(i64),
}

fn place(tree: &Tree, unit: usize, expression: gimli::Expression<Slice>) -> Option<Where> {
    let read = &tree.units[unit];
    let encoding = read.encoding();
    let mut operations = expression.operations(encoding);
    let place = match operations.next().ok()?? {
        gimli::Operation::Address { address } => Where::Memory(address),
        gimli::Operation::AddressIndex { index } => {
            Where::Memory(tree.dwarf(unit).address(read, index).ok()?)
        }
        gimli::Operation::FrameOffset { offset } => Where::Frame(offset),
        gimli::Operation::WasmLocal { index } => Where::Local(index),
        gimli::Operation::WasmGlobal { index } => Where::Global(index),
        _ => return None,
    };
    if let Where::Memory(address) = place
        && is_tombstone(address, encoding.address_size)
    {
        return None;
    }
    match (place, operations.next().ok()?) {
        (Where::Local(index), None) => Some(Where::PointedByLocal(index)),
        (Where::Global(index), None) => Some(Where::PointedByGlobal(index)),
        (_, None) => Some(place),
        (Where::Local(_) | Where::Global(_), Some(gimli::Operation::StackValue)) => {
            operations.next().ok()?.is_none().then_some(place)
        }
        (
            Where::Memory(at),
            Some(gimli::Operation::Deref {
                base_type: gimli::UnitOffset(0),
                size,
                space: false,
            }),
        ) => leaves_unchanged(operations).then_some(Where::Stored(at, size)),
        _ => None,
    }
}

fn origin(tree: &Tree, die: Die) -> Option<Die> {
    declarations(tree, die).get(1..)?.last().map(|(at, _)| *at)
}

const LINKS: [gimli::DwAt; 2] = [gimli::DW_AT_specification, gimli::DW_AT_abstract_origin];

fn declarations<'a>(tree: &Tree<'a>, from: Die) -> Vec<(Die, Entry<'a>)> {
    let mut chain: Vec<(Die, Entry<'a>)> = tree
        .entry(from)
        .map(|entry| (from, entry))
        .into_iter()
        .collect();
    while chain.len() <= MAX_TYPE_DEPTH {
        let Some((at, step)) = chain.last() else {
            break;
        };
        let next = LINKS.into_iter().find_map(|attr| {
            let next = tree.follow(at.unit, step, attr).filter(|next| next != at)?;
            Some((next, tree.entry(next)?))
        });
        match next {
            Some(link) => chain.push(link),
            None => break,
        }
    }
    chain
}

fn children<'a>(tree: &Tree<'a>, die: Die) -> Vec<Entry<'a>> {
    let mut found = Vec::new();
    let Ok(mut entries) = tree.units[die.unit].entries_tree(Some(die.offset)) else {
        return found;
    };
    let Ok(root) = entries.root() else {
        return found;
    };
    let mut nodes = root.children();
    while let Ok(Some(node)) = nodes.next() {
        found.push(node.entry().clone());
    }
    found
}

fn dimensions(tree: &Tree, die: Die) -> Vec<Option<u64>> {
    children(tree, die)
        .iter()
        .filter(|child| child.tag() == gimli::DW_TAG_subrange_type)
        .map(|child| {
            udata(child, gimli::DW_AT_count).or_else(|| {
                udata(child, gimli::DW_AT_upper_bound).and_then(|last| last.checked_add(1))
            })
        })
        .collect()
}

fn byte_size(tree: &Tree, die: Die, depth: usize) -> Option<u64> {
    if depth >= MAX_TYPE_DEPTH {
        return None;
    }
    let entry = tree.entry(die)?;
    if let Some(size) = udata(&entry, gimli::DW_AT_byte_size) {
        return Some(size);
    }
    let address = u64::from(tree.encoding(die.unit).address_size);
    match entry.tag() {
        tag if is_pointer_tag(tag) => Some(address),
        gimli::DW_TAG_ptr_to_member_type if is_aggregate_scalar(tree, die.unit, &entry) => {
            Some(2 * address)
        }
        gimli::DW_TAG_ptr_to_member_type => Some(address),
        gimli::DW_TAG_array_type => {
            let element = byte_size(tree, tree.type_of(die.unit, &entry)?, depth + 1)?;
            dimensions(tree, die)
                .into_iter()
                .try_fold(element, |size, count| size.checked_mul(count.unwrap_or(0)))
        }
        _ => byte_size(tree, tree.type_of(die.unit, &entry)?, depth + 1),
    }
}

pub fn memory_reach(image: &[u8], module: &Module) -> u64 {
    let load = |id: SectionId| -> Result<Slice, gimli::Error> {
        let bytes = section_of(module, id.name())
            .and_then(|(start, end)| {
                let start = usize::try_from(start.checked_sub(module.base)?).ok()?;
                let end = usize::try_from(end.checked_sub(module.base)?).ok()?;
                image.get(start..end)
            })
            .unwrap_or(&[]);
        Ok(EndianSlice::new(bytes, LittleEndian))
    };
    let Ok(dwarf) = Dwarf::load(load) else {
        return 0;
    };

    let tree = Tree::read(vec![&dwarf]);
    let mut reach = 0u64;
    for (unit, read) in tree.units.iter().enumerate() {
        let mut cursor = read.entries();
        while let Ok(Some(entry)) = cursor.next_dfs() {
            if entry.tag() != gimli::DW_TAG_variable {
                continue;
            }
            let Some(AttributeValue::Exprloc(expression)) = entry.attr_value(gimli::DW_AT_location)
            else {
                continue;
            };
            let (at, size) = match place(&tree, unit, expression) {
                Some(Where::Stored(at, size)) => (at, u64::from(size)),
                Some(Where::Memory(at)) => {
                    let size = described(&tree, Die::of(unit, entry))
                        .1
                        .and_then(|ty| byte_size(&tree, ty, 0))
                        .unwrap_or(1);
                    (at, size)
                }
                _ => continue,
            };
            reach = reach.max(at.saturating_add(size.max(1)));
        }
    }
    reach
}

pub type Assigned = (String, Option<Ref<Type>>, bool);

pub type Assignments = HashMap<u64, (u32, Vec<Assigned>)>;

static ASSIGNED: LazyLock<RwLock<HashMap<(ViewId, u64), Assignments>>> =
    LazyLock::new(Default::default);

fn keep_assigned(view: ViewId, entry: u64, assigned: Assignments) {
    let mut kept = ASSIGNED.write().unwrap_or_else(PoisonError::into_inner);
    if assigned.is_empty() {
        kept.remove(&(view, entry));
    } else {
        kept.insert((view, entry), assigned);
    }
}

pub fn has_assignments(view: ViewId, entry: u64) -> bool {
    let kept = ASSIGNED.read().unwrap_or_else(PoisonError::into_inner);
    kept.contains_key(&(view, entry))
}

pub fn assigned(view: ViewId, entry: u64) -> Assignments {
    let kept = ASSIGNED.read().unwrap_or_else(PoisonError::into_inner);
    kept.get(&(view, entry)).cloned().unwrap_or_default()
}

pub fn forget(view: ViewId) {
    let mut kept = ASSIGNED.write().unwrap_or_else(PoisonError::into_inner);
    kept.retain(|(owner, _), _| *owner != view);
    let mut loaded = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
    loaded.remove(&view);
}

static LOADED: LazyLock<Mutex<HashSet<ViewId>>> = LazyLock::new(Default::default);

pub fn load_saved(view: &BinaryView) {
    let mut loaded = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
    if loaded.insert(arch::view_id(view)) {
        restore_assigned(view);
    }
}

const STORED: &str = "binja_wasm.assignments";

const CLASSES: [NamedTypeReferenceClass; 6] = [
    NamedTypeReferenceClass::UnknownNamedTypeClass,
    NamedTypeReferenceClass::TypedefNamedTypeClass,
    NamedTypeReferenceClass::ClassNamedTypeClass,
    NamedTypeReferenceClass::StructNamedTypeClass,
    NamedTypeReferenceClass::UnionNamedTypeClass,
    NamedTypeReferenceClass::EnumNamedTypeClass,
];

fn store_assigned(view: &BinaryView) {
    let id = arch::view_id(view);
    let kept = ASSIGNED.read().unwrap_or_else(PoisonError::into_inner);
    let stored: Vec<serde_json::Map<String, serde_json::Value>> = module::all(id)
        .iter()
        .map(|module| {
            module
                .functions()
                .filter_map(|(index, info)| {
                    let sets = kept
                        .get(&(id, info.entry))?
                        .iter()
                        .map(|(at, (local, names))| {
                            let names: Vec<serde_json::Value> = names
                                .iter()
                                .map(|(name, ty, inlined)| {
                                    let ty = ty.as_deref().and_then(encoded);
                                    serde_json::json!([name, ty, inlined])
                                })
                                .collect();
                            serde_json::json!([at - info.entry, local, names])
                        });
                    Some((index.to_string(), serde_json::Value::Array(sets.collect())))
                })
                .collect()
        })
        .collect();
    if stored.iter().any(|functions| !functions.is_empty())
        && let Ok(text) = serde_json::to_string(&stored)
    {
        view.store_metadata(STORED, text, MetadataStoreFlags::PERSISTENT);
    }
}

fn restore_assigned(view: &BinaryView) {
    let Some(text) = view
        .query_metadata(STORED)
        .and_then(|stored| stored.get_string())
    else {
        return;
    };
    let Ok(stored) =
        serde_json::from_slice::<Vec<serde_json::Map<String, serde_json::Value>>>(text.to_bytes())
    else {
        tracing::warn!("wasm DWARF: the database's local names could not be read");
        return;
    };
    let id = arch::view_id(view);
    for (module, functions) in module::all(id).iter().zip(stored) {
        for (index, sets) in functions {
            let Some(info) = index.parse().ok().and_then(|index| module.function(index)) else {
                continue;
            };
            let assigned = sets
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|set| {
                    let [at, local, names] = set.as_array()?.as_slice() else {
                        return None;
                    };
                    let names = names
                        .as_array()?
                        .iter()
                        .filter_map(|named| {
                            let [name, ty, inlined] = named.as_array()?.as_slice() else {
                                return None;
                            };
                            Some((name.as_str()?.to_owned(), decoded(ty), inlined.as_bool()?))
                        })
                        .collect();
                    let local = u32::try_from(local.as_u64()?).ok()?;
                    Some((info.entry.checked_add(at.as_u64()?)?, (local, names)))
                })
                .collect();
            keep_assigned(id, info.entry, assigned);
        }
    }
}

fn encoded(ty: &Type) -> Option<serde_json::Value> {
    if let Some(named) = ty.get_named_type_reference() {
        let class = CLASSES.iter().position(|class| *class == named.class())?;
        return Some(serde_json::json!({ "named": [class, named.name().items] }));
    }
    let width = ty.width();
    Some(match ty.type_class() {
        TypeClass::VoidTypeClass => serde_json::json!("void"),
        TypeClass::BoolTypeClass => serde_json::json!("bool"),
        TypeClass::IntegerTypeClass => {
            serde_json::json!({ "int": [width, ty.is_signed().contents, ty.to_string()] })
        }
        TypeClass::FloatTypeClass => serde_json::json!({ "float": [width, ty.to_string()] }),
        TypeClass::PointerTypeClass => {
            let target = encoded(&ty.target()?.contents)?;
            serde_json::json!({ "pointer": [width, ty.is_const().contents, target] })
        }
        TypeClass::ArrayTypeClass => {
            let element = encoded(&ty.element_type()?.contents)?;
            serde_json::json!({ "array": [element, ty.count()] })
        }
        _ => return None,
    })
}

fn decoded(value: &serde_json::Value) -> Option<Ref<Type>> {
    match value.as_str() {
        Some("void") => return Some(Type::void()),
        Some("bool") => return Some(Type::bool()),
        Some(_) => return None,
        None => {}
    }
    let (kind, fields) = value.as_object()?.iter().next()?;
    let width = |field: &serde_json::Value| usize::try_from(field.as_u64()?).ok();
    match (kind.as_str(), fields.as_array()?.as_slice()) {
        ("named", [class, items]) => {
            let class = *CLASSES.get(width(class)?)?;
            let items: Vec<String> = serde_json::from_value(items.clone()).ok()?;
            let reference = NamedTypeReference::new(class, QualifiedName::new(items));
            Some(Type::named_type(&reference))
        }
        ("int", [size, signed, name]) => Some(Type::named_int(
            width(size)?,
            signed.as_bool()?,
            name.as_str()?,
        )),
        ("float", [size, name]) => Some(Type::named_float(width(size)?, name.as_str()?)),
        ("pointer", [size, constant, target]) => Some(Type::pointer_of_width(
            &decoded(target)?,
            width(size)?,
            constant.as_bool()?,
            false,
            None,
        )),
        ("array", [element, count]) => Some(Type::array(&decoded(element)?, count.as_u64()?)),
        _ => None,
    }
}

fn leaves_unchanged(mut operations: gimli::OperationIter<Slice>) -> bool {
    let (mut scale, mut offset, mut constant) = (1u64, 0u64, None);
    while let Ok(Some(operation)) = operations.next() {
        match operation {
            gimli::Operation::UnsignedConstant { value } if constant.is_none() => {
                constant = Some(value)
            }
            gimli::Operation::Mul => {
                let Some(by) = constant.take() else {
                    return false;
                };
                scale = scale.wrapping_mul(by);
                offset = offset.wrapping_mul(by);
            }
            gimli::Operation::Plus => {
                let Some(by) = constant.take() else {
                    return false;
                };
                offset = offset.wrapping_add(by);
            }
            gimli::Operation::PlusConstant { value } if constant.is_none() => {
                offset = offset.wrapping_add(value)
            }
            gimli::Operation::StackValue if constant.is_none() => {
                return matches!(operations.next(), Ok(None)) && scale == 1 && offset == 0;
            }
            _ => return false,
        }
    }
    false
}

/// A circular `.debug_info` would otherwise recurse until the stack ran out, which in a callback
/// the core does not guard is a dead process
const MAX_TYPE_DEPTH: usize = 16;

struct Scopes {
    spans: Vec<Span>,
}

struct Span {
    start: usize,
    end: usize,
    name: String,
    parent: Option<usize>,
}

impl Scopes {
    fn of(tree: &Tree, unit: usize) -> Self {
        let mut spans: Vec<Span> = Vec::new();
        let mut open: Vec<(isize, usize)> = Vec::new();
        let mut entries = tree.units[unit].entries();

        while let Ok(Some(entry)) = entries.next_dfs() {
            let level = entry.depth();
            let at = entry.offset().0;
            while let Some(&(depth, index)) = open.last() {
                if depth < level {
                    break;
                }
                spans[index].end = at;
                open.pop();
            }

            if !is_scope(entry.tag()) {
                continue;
            }
            let signed = || {
                let definition = tree.follow(unit, entry, gimli::DW_AT_signature)?;
                tree.name(definition.unit, &tree.entry(definition)?)
            };
            let Some(name) = tree.name(unit, entry).or_else(signed) else {
                continue;
            };
            let parent = open.last().map(|&(_, index)| index);
            spans.push(Span {
                start: at,
                end: usize::MAX,
                name,
                parent,
            });
            open.push((level, spans.len() - 1));
        }

        Self { spans }
    }

    fn qualified(&self, offset: usize, name: &str) -> String {
        let mut parts = vec![name.to_string()];
        let mut at = self.enclosing(offset);
        while let Some(index) = at {
            parts.push(self.spans[index].name.clone());
            at = self.spans[index].parent;
        }
        parts.reverse();
        parts.join("::")
    }

    fn enclosing(&self, offset: usize) -> Option<usize> {
        let mut index = self
            .spans
            .partition_point(|span| span.start <= offset)
            .checked_sub(1)?;
        if self.spans[index].start == offset {
            index = self.spans[index].parent?;
        }
        loop {
            if self.spans[index].end > offset {
                return Some(index);
            }
            index = self.spans[index].parent?;
        }
    }
}

fn is_scope(tag: gimli::DwTag) -> bool {
    matches!(
        tag,
        gimli::DW_TAG_namespace
            | gimli::DW_TAG_class_type
            | gimli::DW_TAG_structure_type
            | gimli::DW_TAG_union_type
            | gimli::DW_TAG_enumeration_type
            | gimli::DW_TAG_subprogram
    )
}

fn base_type(entry: &Entry) -> Option<Ref<Type>> {
    let size = udata(entry, gimli::DW_AT_byte_size)? as usize;
    let encoding = match entry.attr_value(gimli::DW_AT_encoding)? {
        AttributeValue::Encoding(encoding) => encoding,
        _ => return None,
    };
    if encoding == gimli::DW_ATE_complex_float {
        let part = size / 2;
        return (part != 0 && part * 2 == size && part <= 16)
            .then(|| Type::array(&Type::float(part), 2));
    }
    if size == 0 || size > 16 {
        return None;
    }
    Some(match encoding {
        gimli::DW_ATE_float => Type::float(size),
        gimli::DW_ATE_boolean => Type::bool(),
        gimli::DW_ATE_signed_char => Type::char(),
        gimli::DW_ATE_unsigned | gimli::DW_ATE_unsigned_char | gimli::DW_ATE_UTF => {
            Type::int(size, false)
        }
        _ => Type::int(size, true),
    })
}

fn unqualified(tree: &Tree, die: Option<Die>) -> Option<Die> {
    let mut at = die?;
    for _ in 0..MAX_TYPE_DEPTH {
        let entry = tree.entry(at)?;
        match entry.tag() {
            gimli::DW_TAG_const_type
            | gimli::DW_TAG_volatile_type
            | gimli::DW_TAG_restrict_type
            | gimli::DW_TAG_atomic_type => at = tree.type_of(at.unit, &entry)?,
            _ => return Some(at),
        }
    }
    None
}

fn stripped(tree: &Tree, die: Option<Die>) -> Option<Die> {
    let mut at = die?;
    for _ in 0..MAX_TYPE_DEPTH {
        at = unqualified(tree, Some(at))?;
        let entry = tree.entry(at)?;
        if entry.tag() != gimli::DW_TAG_typedef {
            return Some(at);
        }
        at = tree.type_of(at.unit, &entry)?;
    }
    None
}

fn underlying<'a>(tree: &Tree<'a>, die: Option<Die>) -> Option<(Die, Entry<'a>)> {
    let at = stripped(tree, die)?;
    Some((at, tree.entry(at)?))
}

fn described(tree: &Tree, die: Die) -> Described {
    let chain = declarations(tree, die);
    let name = chain
        .iter()
        .find_map(|(at, entry)| tree.name(at.unit, entry));
    let ty = chain
        .iter()
        .find_map(|(at, entry)| tree.type_of(at.unit, entry));
    (name, ty)
}

fn parameters(tree: &Tree, die: Die) -> (Vec<Die>, bool) {
    let found = children(tree, die);
    let variadic = found
        .iter()
        .any(|child| child.tag() == gimli::DW_TAG_unspecified_parameters);
    let formals = found
        .iter()
        .filter(|child| child.tag() == gimli::DW_TAG_formal_parameter)
        .map(|child| Die::of(die.unit, child))
        .collect();
    (formals, variadic)
}

fn returned(tree: &Tree, unit: usize, ty: Option<Die>) -> Passed {
    if speaks_c(tree, unit) {
        passed(tree, ty)
    } else {
        plain(tree, ty)
    }
}

fn internalised(tree: &Tree) -> bool {
    tree.units.iter().any(|read| {
        let mut entries = read.entries();
        while let Ok(Some(entry)) = entries.next_dfs() {
            let rewritten_outside = entry.tag() == gimli::DW_TAG_subprogram
                && entry.attr_value(gimli::DW_AT_external).is_some()
                && entry.attr_value(gimli::DW_AT_calling_convention)
                    == Some(AttributeValue::CallingConvention(gimli::DW_CC_nocall));
            if rewritten_outside || declared_elsewhere(entry) {
                return true;
            }
        }
        false
    })
}

fn declared_elsewhere(die: &Entry) -> bool {
    LINKS.into_iter().any(|attr| {
        die.attr_value(attr)
            .is_some_and(|value| !matches!(value, AttributeValue::UnitRef(_)))
    })
}

fn parameter_slots(
    tree: &Tree,
    die: Die,
    signature: &Signature,
    start: &Start,
    internalised: bool,
    fixed: bool,
    spilled: &BTreeMap<i64, (usize, u64)>,
) -> (Vec<Slot>, Passed) {
    let chain = declarations(tree, die);
    let declaration = chain.last().map_or(die, |(at, _)| *at);
    let (mut formals, mut variadic) = parameters(tree, die);
    if formals.is_empty() && declaration != die {
        (formals, variadic) = parameters(tree, declaration);
    }
    let marked = |attr: gimli::DwAt, value: AttributeValue<Slice>| {
        chain
            .iter()
            .any(|(_, entry)| entry.attr_value(attr) == Some(value))
    };
    let carries = |attr: gimli::DwAt| {
        chain
            .iter()
            .any(|(_, entry)| entry.attr_value(attr).is_some())
    };
    let foreign = chain.iter().any(|(_, entry)| declared_elsewhere(entry));
    let cloned = chain.iter().any(|(at, entry)| {
        entry
            .attr_value(gimli::DW_AT_linkage_name)
            .and_then(|attr| {
                tree.dwarf(at.unit)
                    .attr_string(&tree.units[at.unit], attr)
                    .ok()
            })
            .is_some_and(|name| name.slice().contains(&b'.'))
    });
    let producer = producer(tree, die.unit);
    let optimised = marked(gimli::DW_AT_APPLE_optimized, AttributeValue::Flag(true))
        || carries(gimli::DW_AT_call_all_calls)
        || carries(gimli::DW_AT_GNU_all_call_sites)
        || !matches!(producer, Producer::Clang(_));
    let marks_rewrites = matches!(producer, Producer::Clang(Some(major)) if major >= 23);
    let nocall = marked(
        gimli::DW_AT_calling_convention,
        AttributeValue::CallingConvention(gimli::DW_CC_nocall),
    );
    let rewritten = cloned
        || nocall
        || (!fixed
            && (foreign
                || (optimised
                    && !carries(gimli::DW_AT_virtuality)
                    && (internalised || !carries(gimli::DW_AT_external))
                    && (carries(gimli::DW_AT_specification) || !marks_rewrites))));
    let settled = chain
        .last()
        .is_some_and(|(_, entry)| LINKS.iter().all(|attr| entry.attr_value(*attr).is_none()));
    let typed = chain
        .last()
        .is_some_and(|(_, entry)| entry.attr_value(gimli::DW_AT_type).is_some());
    let result_type = chain
        .last()
        .and_then(|(at, entry)| tree.type_of(at.unit, entry));
    let c = speaks_c(tree, die.unit);
    let shape = Declared {
        formals: formals
            .iter()
            .map(|&formal| {
                let (name, ty) = described(tree, formal);
                let mut declared = Formal {
                    name,
                    ty,
                    passed: argument(tree, die.unit, ty),
                    placed: None,
                    unheld: Vec::new(),
                    partial: false,
                    in_frame: in_frame(tree, formal),
                    constant: false,
                };
                if let Some(entered) = entry_place(tree, formal, ty, start, spilled) {
                    declared.constant = entered == Entered::Constant;
                    declared.unheld = declared.unheld(tree, &entered);
                    if let Some((slot, held, partial)) = declared.entered(tree, entered) {
                        declared.placed = Some((slot, held));
                        declared.partial = partial;
                    }
                }
                declared
            })
            .collect(),
        variadic,
        returned: if settled {
            returned(tree, die.unit, result_type)
        } else {
            Passed::Unknown
        },
        returns: typed || !settled,
        result_type: result_type.filter(|_| settled),
        promotes: c
            && !is_cplusplus(tree, die.unit)
            && !marked(gimli::DW_AT_prototyped, AttributeValue::Flag(true)),
        c,
        address: match tree.encoding(die.unit).address_size {
            8 => ValueKind::I64,
            _ => ValueKind::I32,
        },
        rewritten,
        nocall,
        signature,
    };
    let understood = c
        || shape
            .formals
            .iter()
            .all(|formal| matches!(plain(tree, formal.ty), Passed::Value(_)));
    let partial = shape.formals.iter().any(|formal| formal.partial);
    let slots =
        if cloned || partial || !(c || by_value(tree, die.unit)) || (!rewritten && !understood) {
            located(&shape)
        } else {
            fitted(tree, &shape)
        };
    (slots, shape.returned)
}

enum Producer {
    Clang(Option<u32>),
    Other,
}

fn producer(tree: &Tree, unit: usize) -> Producer {
    let Some(text) = tree
        .root(unit)
        .and_then(|root| tree.string(unit, &root, gimli::DW_AT_producer))
    else {
        return Producer::Other;
    };
    let Some(at) = text.find("clang version ") else {
        return Producer::Other;
    };
    let major = text[at + "clang version ".len()..]
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .and_then(|major| major.parse::<u32>().ok());
    Producer::Clang(major.filter(|_| !text[..at].ends_with("Apple ")))
}

fn argument(tree: &Tree, unit: usize, ty: Option<Die>) -> Passed {
    match passed(tree, ty) {
        Passed::Indirect(at)
            if !is_cplusplus(tree, unit)
                && underlying(tree, Some(at))
                    .is_some_and(|(_, entry)| entry.tag() == gimli::DW_TAG_union_type) =>
        {
            Passed::Unknown
        }
        passed => passed,
    }
}

fn in_frame(tree: &Tree, formal: Die) -> bool {
    matches!(
        tree.entry(formal)
            .and_then(|entry| entry.attr_value(gimli::DW_AT_location)),
        Some(AttributeValue::Exprloc(expression))
            if matches!(place(tree, formal.unit, expression), Some(Where::Frame(_)))
    )
}

struct Start {
    low_pc: Option<u64>,
    first: Option<u64>,
}

impl Start {
    fn covers(&self, at: u64) -> bool {
        self.low_pc
            .is_some_and(|low| (low..=self.first.unwrap_or(low)).contains(&at))
    }
}

struct Declared<'a> {
    formals: Vec<Formal>,
    variadic: bool,
    returned: Passed,
    returns: bool,
    result_type: Option<Die>,
    promotes: bool,
    c: bool,
    address: ValueKind,
    rewritten: bool,
    nocall: bool,
    signature: &'a Signature,
}

struct Formal {
    name: Option<String>,
    ty: Option<Die>,
    passed: Passed,
    placed: Option<(usize, Vec<Slot>)>,
    unheld: Vec<(usize, Slot, u64)>,
    partial: bool,
    in_frame: bool,
    constant: bool,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Entered {
    Whole(usize, bool),
    Pieces(usize, Vec<(u64, u64, Part)>),
    Constant,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Part {
    Held,
    Unheld,
    Constant,
}

impl Declared<'_> {
    fn result_first(&self) -> (bool, bool) {
        let unreturned = self.signature.results.is_empty();
        match self.returned {
            Passed::Indirect(_) | Passed::Split(_) if unreturned => {
                (!self.rewritten, self.rewritten)
            }
            Passed::Unknown if unreturned && self.returns => (false, true),
            _ => (false, false),
        }
    }

    fn result(&self) -> Slot {
        let passed = match (self.returned, self.result_type) {
            (Passed::Split(ty) | Passed::Indirect(ty), _) | (_, Some(ty)) => Passed::Indirect(ty),
            (returned, None) => returned,
        };
        (Some(RETURN_SLOT.to_owned()), passed)
    }
}

impl Formal {
    fn at(&self, indirect: bool) -> Slot {
        let passed = match (self.passed, self.ty) {
            (Passed::Value(_), _) if !indirect => self.passed,
            (Passed::Indirect(_), _) if indirect => self.passed,
            (_, Some(ty)) if indirect => Passed::Indirect(ty),
            (_, Some(ty)) => Passed::Value(ty),
            (_, None) => Passed::Unknown,
        };
        (self.name.clone(), passed)
    }

    fn unheld(&self, tree: &Tree, entered: &Entered) -> Vec<(usize, Slot, u64)> {
        let Entered::Pieces(first, parts) = entered else {
            return Vec::new();
        };
        let total: u64 = parts.iter().map(|&(_, length, _)| length).sum();
        let rest = stripped(tree, self.ty)
            .and_then(|ty| byte_size(tree, ty, 0))
            .and_then(|size| size.checked_sub(total))
            .filter(|rest| *rest != 0)
            .map(|rest| (total, rest, Part::Unheld));
        let passed: Vec<(u64, u64, Part)> = parts
            .iter()
            .chain(&rest)
            .copied()
            .filter(|&(.., part)| part != Part::Constant)
            .collect();
        let held: Vec<usize> = (0..passed.len())
            .filter(|nth| passed[*nth].2 == Part::Held)
            .collect();
        let (Some(&low), Some(&high)) = (held.first(), held.last()) else {
            return Vec::new();
        };
        if high - low + 1 != held.len() || matches!(self.passed, Passed::Split(_)) {
            return Vec::new();
        }
        let mut found = Vec::new();
        for (nth, &(offset, length, part)) in passed.iter().enumerate() {
            if part == Part::Held {
                continue;
            }
            let (Some(at), Some((path, ty))) = (
                (first + nth).checked_sub(low),
                member_at(tree, self.ty, offset, length, 0),
            ) else {
                return Vec::new();
            };
            let name = self
                .name
                .as_ref()
                .map(|name| format!("{name}.{}", path.join(".")));
            found.push((at, (name, Passed::Value(ty)), length));
        }
        found
    }

    fn entered(&self, tree: &Tree, entered: Entered) -> Option<(usize, Vec<Slot>, bool)> {
        let split = matches!(self.passed, Passed::Split(_));
        let (first, parts) = match entered {
            Entered::Whole(slot, indirect) if !split => {
                return Some((slot, vec![self.at(indirect)], false));
            }
            Entered::Whole(..) | Entered::Constant => return None,
            Entered::Pieces(first, parts) => (first, parts),
        };
        let size = byte_size(tree, stripped(tree, self.ty)?, 0)?;
        let total = parts
            .iter()
            .try_fold(0u64, |total, &(_, length, _)| total.checked_add(length))?;
        let padded = parts.iter().all(|&(offset, length, part)| {
            part != Part::Unheld || !overlaps(tree, self.ty, offset, length, 0)
        });
        if total > size {
            return None;
        }
        let partial = total < size || !padded;
        let held: Vec<(u64, u64)> = parts
            .iter()
            .filter(|&&(.., part)| part == Part::Held)
            .map(|&(offset, length, _)| (offset, length))
            .collect();
        if split {
            let half = size / 2;
            let aligned = held
                .iter()
                .all(|&(offset, length)| length == half && (offset == 0 || offset == half));
            let start = first.checked_sub(usize::from(held.first()?.0 == half))?;
            return aligned.then(|| (start, self.halves().to_vec(), false));
        }
        let placed = match held[..] {
            [(0, length)] if length == size => vec![self.at(false)],
            _ => held
                .iter()
                .map(
                    |&(offset, length)| match member_at(tree, self.ty, offset, length, 0) {
                        Some((path, ty)) => (
                            self.name
                                .as_ref()
                                .map(|name| format!("{name}.{}", path.join("."))),
                            Passed::Value(ty),
                        ),
                        None => (None, Passed::Unknown),
                    },
                )
                .collect(),
        };
        Some((first, placed, partial))
    }

    fn halves(&self) -> [Slot; 2] {
        ["low", "high"].map(|half| {
            let named = self.name.as_ref().map(|name| format!("{name}_{half}"));
            (named, Passed::Unknown)
        })
    }
}

fn is_struct(tag: gimli::DwTag) -> bool {
    matches!(tag, gimli::DW_TAG_structure_type | gimli::DW_TAG_class_type)
}

fn member_at(
    tree: &Tree,
    ty: Option<Die>,
    offset: u64,
    length: u64,
    depth: usize,
) -> Option<(Vec<String>, Die)> {
    let (record, entry) = underlying(tree, ty)?;
    let end = offset.checked_add(length)?;
    if depth >= MAX_TYPE_DEPTH || !is_struct(entry.tag()) {
        return None;
    }
    let mut found = None;
    for child in &children(tree, record) {
        match child.tag() {
            gimli::DW_TAG_member if !is_declaration(child) => {}
            gimli::DW_TAG_inheritance | gimli::DW_TAG_variant_part => return None,
            _ => continue,
        }
        let (start, size) = member_storage(tree, record.unit, child)?;
        let stop = start.checked_add(size)?;
        if end <= start || stop <= offset {
            continue;
        }
        if found.is_some() || is_bitfield(child) || offset < start || end > stop {
            return None;
        }
        let member = tree.type_of(record.unit, child)?;
        let name = tree.name(record.unit, child)?;
        let exact = start == offset && size == length;
        found = match member_at(tree, Some(member), offset - start, length, depth + 1) {
            Some((_, leaf)) if exact => Some((vec![name], leaf)),
            Some((mut path, leaf)) => {
                path.insert(0, name);
                Some((path, leaf))
            }
            None if exact => Some((vec![name], member)),
            None => return None,
        };
    }
    found
}

fn overlaps(tree: &Tree, ty: Option<Die>, offset: u64, length: u64, depth: usize) -> bool {
    let Some((record, entry)) = underlying(tree, ty) else {
        return true;
    };
    let Some(end) = offset.checked_add(length) else {
        return true;
    };
    if depth >= MAX_TYPE_DEPTH || !is_struct(entry.tag()) {
        return true;
    }
    children(tree, record)
        .iter()
        .any(|child| match child.tag() {
            gimli::DW_TAG_member if !is_declaration(child) => {
                let Some((start, size)) = member_storage(tree, record.unit, child) else {
                    return true;
                };
                let stop = start.saturating_add(size);
                start < end
                    && offset < stop
                    && (is_bitfield(child) || {
                        let from = offset.max(start);
                        let member = tree.type_of(record.unit, child);
                        overlaps(tree, member, from - start, end.min(stop) - from, depth + 1)
                    })
            }
            gimli::DW_TAG_inheritance | gimli::DW_TAG_variant_part => true,
            _ => false,
        })
}

fn is_pointer(tree: &Tree, ty: Option<Die>) -> bool {
    underlying(tree, ty).is_some_and(|(_, entry)| is_pointer_tag(entry.tag()))
}

fn is_pointer_tag(tag: gimli::DwTag) -> bool {
    matches!(
        tag,
        gimli::DW_TAG_pointer_type
            | gimli::DW_TAG_reference_type
            | gimli::DW_TAG_rvalue_reference_type
    )
}

fn fitted(tree: &Tree, shape: &Declared) -> Vec<Slot> {
    let rewritten = shape.rewritten;
    let kinds = &shape.signature.params;
    let anchors = anchors(shape);
    let (certain, possible) = shape.result_first();
    let buffered = shape.variadic && !rewritten;
    let first = usize::from(certain);
    let Some(end) = kinds.len().checked_sub(usize::from(buffered)) else {
        return located(shape);
    };
    if first > end
        || !anchors
            .windows(2)
            .all(|pair| pair[0].1 + pair[0].2.len() <= pair[1].1)
        || anchors.first().is_some_and(|&(_, slot, _)| slot < first)
        || anchors
            .last()
            .is_some_and(|&(_, slot, held)| slot + held.len() > end)
        || (certain && kinds[0] != shape.address)
        || (buffered && kinds[end] != shape.address)
    {
        return located(shape);
    }

    let mut slots = vec![(None, Passed::Unknown); kinds.len()];
    if certain {
        slots[0] = shape.result();
    }
    if buffered {
        slots[end] = (None, Passed::Variadic);
    }
    let mut from = (0, first);
    for gap in 0..=anchors.len() {
        let until = anchors
            .get(gap)
            .map_or((shape.formals.len(), end), |&(nth, slot, _)| (nth, slot));
        let items: Vec<Item> = (gap == 0 && possible)
            .then_some(Item::Result)
            .into_iter()
            .chain(
                shape.formals[from.0..until.0]
                    .iter()
                    .map(|formal| Item::Formal(formal, width(tree, shape, formal))),
            )
            .chain((gap == anchors.len() && shape.variadic && rewritten).then_some(Item::Buffer))
            .collect();
        let Some(filled) = fill(tree, shape, &items, from.1..until.1) else {
            return located(shape);
        };
        for (at, slot) in filled {
            slots[at] = match slot {
                Filled::Result => shape.result(),
                Filled::Buffer => (None, Passed::Variadic),
                Filled::Whole(formal) => (formal.name.clone(), formal.passed),
                Filled::Half(formal, half) => formal.halves()[half].clone(),
            };
        }
        if let Some(&(nth, slot, held)) = anchors.get(gap) {
            slots[slot..slot + held.len()].clone_from_slice(held);
            from = (nth + 1, slot + held.len());
        }
    }
    slots
}

#[derive(Clone, Copy)]
enum Width {
    Exactly(usize),
    Removable(usize),
    Unsure,
    Any,
}

fn width(tree: &Tree, shape: &Declared, formal: &Formal) -> Width {
    let rewritten = shape.rewritten;
    if shape.nocall && formal.constant {
        return Width::Exactly(0);
    }
    let pointing = is_pointer(tree, formal.ty)
        || matches!(formal.passed, Passed::Value(ty) if is_pointer(tree, Some(ty)));
    if !shape.c {
        let scalar = matches!(plain(tree, formal.ty), Passed::Value(_));
        return match (rewritten, scalar) {
            (false, _) => Width::Exactly(1),
            (true, true) if !pointing || formal.in_frame => Width::Removable(1),
            (true, _) => Width::Any,
        };
    }
    let vector = underlying(tree, formal.ty).is_some_and(|(_, entry)| is_vector(&entry));
    match formal.passed {
        Passed::Omitted => Width::Exactly(0),
        Passed::Split(_) if rewritten => Width::Removable(2),
        Passed::Split(_) => Width::Exactly(2),
        Passed::Unknown if !rewritten => Width::Unsure,
        _ if !rewritten => Width::Exactly(1),
        _ if vector => Width::Any,
        Passed::Value(_) if !pointing => Width::Removable(1),
        _ if formal.in_frame => Width::Removable(1),
        _ => Width::Any,
    }
}

enum Item<'a> {
    Result,
    Formal(&'a Formal, Width),
    Buffer,
}

enum Filled<'a> {
    Result,
    Buffer,
    Whole(&'a Formal),
    Half(&'a Formal, usize),
}

const MAX_GAP: usize = 64;

fn fill<'a>(
    tree: &Tree,
    shape: &Declared,
    items: &[Item<'a>],
    range: Range<usize>,
) -> Option<Vec<(usize, Filled<'a>)>> {
    let span = range.len();
    if span > MAX_GAP {
        return Some(Vec::new());
    }
    let kinds = &shape.signature.params[range.clone()];
    let fits = |item: &Item, taken: &[ValueKind]| match item {
        _ if taken.is_empty() => true,
        Item::Result | Item::Buffer => taken == [shape.address],
        Item::Formal(formal, Width::Any) => promotes_to(tree, shape, formal, taken),
        Item::Formal(_, Width::Unsure) => {
            taken.len() == 1 || taken.iter().all(|kind| *kind == ValueKind::I64)
        }
        Item::Formal(formal, _) => match formal.passed {
            Passed::Split(_) => taken.iter().all(|kind| *kind == ValueKind::I64),
            passed => taken.iter().all(|kind| agrees(tree, shape, passed, *kind)),
        },
    };
    let options = |item: &Item, at: usize| -> Vec<usize> {
        let room = span - at;
        let widths = match item {
            Item::Result | Item::Buffer => vec![0, 1],
            Item::Formal(_, Width::Exactly(width)) => vec![*width],
            Item::Formal(_, Width::Removable(width)) => vec![0, *width],
            Item::Formal(_, Width::Unsure) => vec![0, 1, 2],
            Item::Formal(_, Width::Any) => (0..=room).collect(),
        };
        widths
            .into_iter()
            .filter(|width| *width <= room && fits(item, &kinds[at..at + width]))
            .collect()
    };
    let mut ways = vec![vec![0u8; span + 1]; items.len() + 1];
    ways[items.len()][span] = 1;
    for (nth, item) in items.iter().enumerate().rev() {
        for at in 0..=span {
            ways[nth][at] = options(item, at).into_iter().fold(0u8, |total, width| {
                total.saturating_add(ways[nth + 1][at + width]).min(2)
            });
        }
    }
    match ways[0][0] {
        0 => return None,
        1 => {}
        _ => return Some(Vec::new()),
    }
    let mut filled = Vec::new();
    let mut at = 0;
    for (nth, item) in items.iter().enumerate() {
        let width = options(item, at)
            .into_iter()
            .find(|width| ways[nth + 1][at + width] != 0)?;
        let start = range.start + at;
        match (item, width) {
            (_, 0) => {}
            (Item::Formal(formal, Width::Any), 1)
                if only_the_pointer(tree, shape, formal, kinds[at]) =>
            {
                filled.push((start, Filled::Whole(formal)))
            }
            (Item::Formal(_, Width::Any), _) => {}
            (Item::Result, _) => filled.push((start, Filled::Result)),
            (Item::Buffer, _) => filled.push((start, Filled::Buffer)),
            (Item::Formal(formal, _), 2) => {
                filled.push((start, Filled::Half(formal, 0)));
                filled.push((start + 1, Filled::Half(formal, 1)));
            }
            (Item::Formal(formal, _), _) => filled.push((start, Filled::Whole(formal))),
        }
        at += width;
    }
    Some(filled)
}

fn promotes_to(tree: &Tree, shape: &Declared, formal: &Formal, taken: &[ValueKind]) -> bool {
    taken == [shape.address]
        || promoted_leaves(tree, shape, formal).is_none_or(|leaves| {
            let mut rest = leaves.iter();
            taken.iter().all(|kind| rest.any(|leaf| leaf == kind))
        })
}

fn only_the_pointer(tree: &Tree, shape: &Declared, formal: &Formal, kind: ValueKind) -> bool {
    kind == shape.address
        && promoted_leaves(tree, shape, formal).is_some_and(|leaves| !leaves.contains(&kind))
}

fn promoted_leaves(tree: &Tree, shape: &Declared, formal: &Formal) -> Option<Vec<ValueKind>> {
    let value = match formal.passed {
        Passed::Value(ty) => Some(ty),
        _ => None,
    };
    let pointer = [formal.ty, value]
        .into_iter()
        .flatten()
        .find(|ty| is_pointer(tree, Some(*ty)));
    let promoted = match pointer {
        Some(pointer) => {
            underlying(tree, Some(pointer)).and_then(|(at, entry)| tree.type_of(at.unit, &entry))
        }
        None => formal.ty,
    };
    leaves(tree, promoted, shape.address)
}

const MAX_LEAVES: usize = 64;

fn leaves(tree: &Tree, ty: Option<Die>, address: ValueKind) -> Option<Vec<ValueKind>> {
    let mut found = Vec::new();
    gather_leaves(tree, ty?, address, 0, &mut found)?;
    Some(found)
}

fn gather_leaves(
    tree: &Tree,
    ty: Die,
    address: ValueKind,
    depth: usize,
    found: &mut Vec<ValueKind>,
) -> Option<()> {
    let (at, entry) = underlying(tree, Some(ty))?;
    if depth >= MAX_TYPE_DEPTH || found.len() > MAX_LEAVES {
        return None;
    }
    match entry.tag() {
        tag if is_pointer_tag(tag) => found.push(address),
        gimli::DW_TAG_base_type | gimli::DW_TAG_enumeration_type => {
            let kinds: &[ValueKind] = match (
                entry.attr_value(gimli::DW_AT_encoding),
                byte_size(tree, at, 0)?,
            ) {
                (Some(AttributeValue::Encoding(gimli::DW_ATE_complex_float)), _) => return None,
                (Some(AttributeValue::Encoding(gimli::DW_ATE_float)), 4) => &[ValueKind::F32],
                (Some(AttributeValue::Encoding(gimli::DW_ATE_float)), 8) => &[ValueKind::F64],
                (Some(AttributeValue::Encoding(gimli::DW_ATE_float)), _) => return None,
                (_, 1..=4) => &[ValueKind::I32],
                (_, 8) => &[ValueKind::I64],
                (_, 16) => &[ValueKind::I64, ValueKind::I64],
                _ => return None,
            };
            found.extend_from_slice(kinds);
        }
        tag if is_struct(tag) => {
            let mut members = Vec::new();
            for child in &children(tree, at) {
                match child.tag() {
                    gimli::DW_TAG_member if !is_declaration(child) => {
                        if is_bitfield(child) {
                            return None;
                        }
                        let (start, _) = member_storage(tree, at.unit, child)?;
                        members.push((start, tree.type_of(at.unit, child)?));
                    }
                    gimli::DW_TAG_inheritance | gimli::DW_TAG_variant_part => return None,
                    _ => {}
                }
            }
            members.sort_by_key(|&(start, _)| start);
            for (_, member) in members {
                gather_leaves(tree, member, address, depth + 1, found)?;
            }
        }
        gimli::DW_TAG_array_type if !is_vector(&entry) => {
            let element = tree.type_of(at.unit, &entry)?;
            let each = byte_size(tree, stripped(tree, Some(element))?, 0)?;
            let count = byte_size(tree, at, 0)?.checked_div(each)?;
            for _ in 0..count.min(MAX_LEAVES as u64 + 1) {
                gather_leaves(tree, element, address, depth + 1, found)?;
            }
        }
        _ => return None,
    }
    (found.len() <= MAX_LEAVES).then_some(())
}

fn agrees(tree: &Tree, shape: &Declared, passed: Passed, kind: ValueKind) -> bool {
    let ty = match passed {
        Passed::Value(ty) => ty,
        Passed::Indirect(_) | Passed::Variadic => return kind == shape.address,
        Passed::Omitted | Passed::Split(_) | Passed::Unknown => return true,
    };
    let Some((at, entry)) = underlying(tree, Some(ty)) else {
        return true;
    };
    let size = byte_size(tree, at, 0);
    if is_vector(&entry) {
        return kind == ValueKind::V128 && size.is_some_and(|size| size <= 16);
    }
    let float = entry.attr_value(gimli::DW_AT_encoding)
        == Some(AttributeValue::Encoding(gimli::DW_ATE_float));
    match (float, size) {
        (true, Some(4)) if shape.promotes => matches!(kind, ValueKind::F32 | ValueKind::F64),
        (true, Some(4)) => kind == ValueKind::F32,
        (true, Some(8)) => kind == ValueKind::F64,
        (true, _) => true,
        (false, Some(1..=4)) => kind == ValueKind::I32,
        (false, Some(8)) => kind == ValueKind::I64,
        (false, _) => kind == shape.address,
    }
}

fn anchors<'a>(shape: &'a Declared) -> Vec<(usize, usize, &'a [Slot])> {
    let params = shape.signature.params.len();
    shape
        .formals
        .iter()
        .enumerate()
        .filter_map(|(nth, formal)| {
            let (slot, held) = formal.placed.as_ref()?;
            (!held.is_empty() && slot + held.len() <= params).then_some((nth, *slot, &held[..]))
        })
        .collect()
}

fn located(shape: &Declared) -> Vec<Slot> {
    let params = shape.signature.params.len();
    let mut slots = vec![(None, Passed::Unknown); params];
    let mut claims = vec![0usize; params];
    for (_, slot, held) in anchors(shape) {
        for (at, placed) in (slot..).zip(held) {
            claims[at] += 1;
            slots[at] = placed.clone();
        }
    }
    for (slot, claimed) in slots.iter_mut().zip(&claims) {
        if *claimed > 1 {
            *slot = (None, Passed::Unknown);
        }
    }
    let span = |nth: Option<usize>| {
        let (slot, held) = shape.formals.get(nth?)?.placed.as_ref()?;
        Some((*slot, slot + held.len()))
    };
    let result = {
        let (certain, possible) = shape.result_first();
        certain || possible
    };
    for (nth, formal) in shape.formals.iter().enumerate() {
        let (Some((start, _)), Some(first), Some(last)) = (
            span(Some(nth)),
            formal.unheld.iter().map(|(at, ..)| *at).min(),
            formal.unheld.iter().map(|(at, ..)| *at).max(),
        ) else {
            continue;
        };
        let before = first > start || span(nth.checked_sub(1)).is_some_and(|(_, at)| at == first);
        let ends = !shape.variadic
            && last + 1 == params
            && shape.formals[nth + 1..]
                .iter()
                .all(|later| later.constant && shape.nocall)
            && (0..first).all(|at| claims[at] > 0 || (at == 0 && result));
        let after =
            last < start || ends || span(Some(nth + 1)).is_some_and(|(at, _)| at == last + 1);
        let fits = formal.unheld.iter().all(|(at, _, length)| {
            claims.get(*at) == Some(&0)
                && shape
                    .signature
                    .params
                    .get(*at)
                    .map(|kind| kind.size() as u64)
                    == Some(*length)
        });
        if before && after && fits {
            for (at, filled, _) in &formal.unheld {
                slots[*at] = filled.clone();
            }
        }
    }
    slots
}

fn entry_place(
    tree: &Tree,
    formal: Die,
    ty: Option<Die>,
    start: &Start,
    spilled: &BTreeMap<i64, (usize, u64)>,
) -> Option<Entered> {
    let unit = formal.unit;
    let slot = |index: u32| usize::try_from(index).ok();
    let argument = |expression: gimli::Expression<Slice>| match place(tree, unit, expression) {
        Some(Where::Local(index)) => Some(Entered::Whole(slot(index)?, false)),
        Some(Where::PointedByLocal(index)) => Some(Entered::Whole(slot(index)?, true)),
        Some(Where::Frame(offset)) => {
            let size = byte_size(tree, stripped(tree, ty)?, 0)?;
            from_frame(spilled, offset, size)
        }
        Some(_) => None,
        None if is_constant(tree, unit, expression) => Some(Entered::Constant),
        None => match truncated(tree, unit, expression) {
            Some(index) => Some(Entered::Whole(slot(index)?, false)),
            None => pieces(tree, unit, expression),
        },
    };
    let entry = tree.entry(formal)?;
    if entry.attr_value(gimli::DW_AT_const_value).is_some() {
        return Some(Entered::Constant);
    }
    match entry.attr_value(gimli::DW_AT_location)? {
        AttributeValue::Exprloc(expression) => argument(expression),
        value => {
            let offset = tree
                .dwarf(unit)
                .attr_locations_offset(&tree.units[unit], value)
                .ok()??;
            let entered = |range: &Range<u64>| {
                start.covers(range.start) || start.low_pc.is_some_and(|low| range.contains(&low))
            };
            let held: BTreeSet<_> = location_list(tree, unit, offset)
                .into_iter()
                .filter(|(range, _)| range.start <= range.end && entered(range))
                .map(|(_, data)| argument(data))
                .collect();
            let mut held = held.into_iter();
            match (held.next(), held.next()) {
                (Some(found), None) => found,
                _ => None,
            }
        }
    }
}

fn pieces(tree: &Tree, unit: usize, expression: gimli::Expression<Slice>) -> Option<Entered> {
    let mut operations = expression.operations(tree.encoding(unit));
    let (mut parts, mut locals, mut offset) = (Vec::new(), Vec::new(), 0u64);
    while let Some(operation) = operations.next().ok()? {
        let (part, piece) = match operation {
            gimli::Operation::WasmLocal { .. }
            | gimli::Operation::UnsignedConstant { .. }
            | gimli::Operation::SignedConstant { .. } => {
                let Some(gimli::Operation::StackValue) = operations.next().ok()? else {
                    return None;
                };
                let part = match operation {
                    gimli::Operation::WasmLocal { index } => {
                        locals.push(usize::try_from(index).ok()?);
                        Part::Held
                    }
                    _ => Part::Constant,
                };
                (part, operations.next().ok()??)
            }
            piece => (Part::Unheld, piece),
        };
        let gimli::Operation::Piece {
            size_in_bits,
            bit_offset: None,
        } = piece
        else {
            return None;
        };
        if size_in_bits % 8 != 0 {
            return None;
        }
        parts.push((offset, size_in_bits / 8, part));
        offset = offset.checked_add(size_in_bits / 8)?;
    }
    if !parts.is_empty() && parts.iter().all(|&(.., part)| part == Part::Constant) {
        return Some(Entered::Constant);
    }
    let first = *locals.first()?;
    let consecutive = locals
        .iter()
        .enumerate()
        .all(|(nth, local)| *local == first + nth);
    consecutive.then_some(Entered::Pieces(first, parts))
}

fn spills(code: &[u8], frame: u32, params: &[ValueKind]) -> BTreeMap<i64, (usize, u64)> {
    let mut found: BTreeMap<i64, Option<(usize, u64)>> = BTreeMap::new();
    let mut written = HashSet::new();
    let mut read: [Option<u32>; 2] = [None, None];
    let mut at = 0;
    while let Some(insn) = code.get(at..).and_then(insn::decode_any) {
        at += insn.len;
        if insn::flow(&insn.op) != insn::Flow::Normal {
            break;
        }
        let stored = match insn.op {
            Operator::I32Store { memarg } => Some((memarg, ValueKind::I32, 4)),
            Operator::I32Store16 { memarg } => Some((memarg, ValueKind::I32, 2)),
            Operator::I32Store8 { memarg } => Some((memarg, ValueKind::I32, 1)),
            Operator::I64Store { memarg } => Some((memarg, ValueKind::I64, 8)),
            Operator::F32Store { memarg } => Some((memarg, ValueKind::F32, 4)),
            Operator::F64Store { memarg } => Some((memarg, ValueKind::F64, 8)),
            Operator::V128Store { memarg } => Some((memarg, ValueKind::V128, 16)),
            Operator::LocalSet { local_index } | Operator::LocalTee { local_index } => {
                written.insert(local_index);
                None
            }
            _ => None,
        };
        if let (Some((memarg, kind, width)), [Some(base), Some(local)]) = (stored, read)
            && base == frame
            && memarg.memory == 0
            && !written.contains(&local)
            && let Ok(held) = usize::try_from(local)
            && params.get(held) == Some(&kind)
            && let Ok(offset) = i64::try_from(memarg.offset)
        {
            found
                .entry(offset)
                .and_modify(|spill| *spill = None)
                .or_insert(Some((held, width)));
        }
        let got = match insn.op {
            Operator::LocalGet { local_index } => Some(local_index),
            _ => None,
        };
        read = [read[1], got];
    }
    found
        .into_iter()
        .filter_map(|(offset, spill)| Some((offset, spill?)))
        .collect()
}

fn body_code<'i>(module: &Module, image: &'i [u8], body: &FunctionInfo) -> Option<&'i [u8]> {
    let from = usize::try_from(module.layout.file_offset(body.entry)?).ok()?;
    let to = usize::try_from(module.layout.file_offset(body.end)?).ok()?;
    image.get(from..to)
}

fn prologue_frame(code: &[u8], pointer: u32, holder: u32) -> bool {
    let mut previous = None;
    let mut at = 0;
    while let Some(insn) = code.get(at..).and_then(insn::decode_any) {
        at += insn.len;
        if let Operator::GlobalSet { global_index } = insn.op
            && global_index == pointer
        {
            return matches!(
                previous,
                Some(Operator::LocalTee { local_index } | Operator::LocalGet { local_index })
                    if local_index == holder
            );
        }
        previous = Some(insn.op);
    }
    false
}

fn copies(code: &[u8], holder: u32, copy: u32) -> bool {
    let (mut previous, mut held) = (None, 0usize);
    let mut at = 0;
    while let Some(insn) = code.get(at..).and_then(insn::decode_any) {
        at += insn.len;
        match insn.op {
            Operator::LocalSet { local_index } | Operator::LocalTee { local_index }
                if local_index == copy =>
            {
                return held == 1
                    && matches!(previous, Some(Operator::LocalGet { local_index }) if local_index == holder);
            }
            Operator::LocalSet { local_index } | Operator::LocalTee { local_index }
                if local_index == holder =>
            {
                held += 1;
            }
            _ => {}
        }
        previous = Some(insn.op);
    }
    false
}

fn renumbered(tree: &Tree, module: &Module, image: &[u8], base: u64) -> bool {
    let (mut agreeing, mut disagreeing) = (0usize, 0usize);
    let mut seen = HashSet::new();
    for (unit, read) in tree.units.iter().enumerate() {
        let mut entries = read.entries();
        while let Ok(Some(entry)) = entries.next_dfs() {
            if entry.tag() != gimli::DW_TAG_subprogram {
                continue;
            }
            let Some(low_pc) = tree.address(unit, entry, gimli::DW_AT_low_pc) else {
                continue;
            };
            let Some((index, body)) = body_at(module, base.wrapping_add(low_pc))
                .filter(|(_, body)| spans(tree, unit, entry, low_pc, body))
            else {
                continue;
            };
            let Some(code) = body_code(module, image, body).filter(|_| seen.insert(body.entry))
            else {
                continue;
            };
            let framed =
                body.frame
                    .zip(module.stack_pointer)
                    .filter(|((holder, size), pointer)| {
                        *size > 0 && prologue_frame(code, *pointer, *holder)
                    });
            if let Some(((holder, _), _)) = framed
                && let Some(AttributeValue::Exprloc(expression)) =
                    entry.attr_value(gimli::DW_AT_frame_base)
                && let Some(Where::Local(local)) = place(tree, unit, expression)
                && local != holder
                && !copies(code, holder, local)
            {
                return true;
            }
            let locals = module.local_kinds(index).len();
            let params = body.signature.params.len();
            let set = locals_set(code, body.entry - base);
            let within = body.start - base..body.end - base;
            let (held, _) = held_variables(tree, Die::of(unit, entry), within);
            for (range, at) in held.iter().flat_map(|variable| &variable.places) {
                let (Where::Local(local) | Where::PointedByLocal(local)) = *at else {
                    continue;
                };
                let local = local as usize;
                if local >= locals {
                    return true;
                }
                let Some(start) = range
                    .as_ref()
                    .map(|range| range.start)
                    .filter(|_| local >= params)
                else {
                    continue;
                };
                match set.get(&start) {
                    Some(&written) if written as usize == local => agreeing += 1,
                    Some(_) => disagreeing += 1,
                    None => {}
                }
            }
        }
    }
    let names = module
        .sections
        .iter()
        .position(|section| section.name == "name");
    let dwarf = module
        .sections
        .iter()
        .position(|section| section.name.starts_with(".debug_"));
    let rewritten = names.zip(dwarf).is_some_and(|(names, dwarf)| names < dwarf);
    disagreeing > agreeing || (rewritten && agreeing == 0)
}

fn locals_set(code: &[u8], address: u64) -> HashMap<u64, u32> {
    let mut set = HashMap::new();
    let mut at = 0;
    while let Some(insn) = code.get(at..).and_then(insn::decode_any) {
        at += insn.len;
        if let Operator::LocalSet { local_index } | Operator::LocalTee { local_index } = insn.op {
            set.insert(address + at as u64, local_index);
        }
    }
    set
}

struct Held {
    die: Die,
    inlined: bool,
    scope: Vec<Range<u64>>,
    places: Vec<(Option<Range<u64>>, Where)>,
}

fn held_variables<'a>(
    tree: &Tree<'a>,
    subprogram: Die,
    within: Range<u64>,
) -> (Vec<Held>, HashSet<u64>) {
    let unit = subprogram.unit;
    let (dwarf, read) = (tree.dwarf(unit), &tree.units[unit]);
    let covered = |entry: &Entry<'a>| {
        let mut found = Vec::new();
        if let Ok(mut ranges) = dwarf.die_ranges(read, entry) {
            while let Ok(Some(range)) = ranges.next() {
                if within.start <= range.begin && range.begin < range.end && range.end <= within.end
                {
                    found.push(range.begin..range.end);
                }
            }
        }
        found
    };

    let (mut held, mut starts) = (Vec::new(), HashSet::new());
    let Ok(mut cursor) = read.entries_at_offset(subprogram.offset) else {
        return (held, starts);
    };
    let whole = match cursor.next_dfs() {
        Ok(Some(root)) => covered(root),
        _ => return (held, starts),
    };
    let mut scopes = vec![(0, whole)];
    let mut inlined = isize::MAX;
    while let Ok(Some(child)) = cursor.next_dfs() {
        let level = child.depth();
        if level <= 0 {
            break;
        }
        if level <= inlined {
            inlined = isize::MAX;
        }
        scopes.retain(|(depth, _)| *depth < level);
        match child.tag() {
            gimli::DW_TAG_lexical_block | gimli::DW_TAG_inlined_subroutine => {
                let ranges = covered(child);
                starts.extend(ranges.iter().map(|range| range.start));
                scopes.push((level, ranges));
                if child.tag() == gimli::DW_TAG_inlined_subroutine {
                    inlined = inlined.min(level);
                }
            }
            gimli::DW_TAG_variable | gimli::DW_TAG_formal_parameter => {
                let places = placements(tree, unit, child);
                if !places.is_empty() {
                    held.push(Held {
                        die: Die::of(unit, child),
                        inlined: level > inlined,
                        scope: scopes
                            .last()
                            .map(|(_, ranges)| ranges.clone())
                            .unwrap_or_default(),
                        places,
                    });
                }
            }
            _ => {}
        }
    }
    (held, starts)
}

fn placements<'a>(
    tree: &Tree<'a>,
    unit: usize,
    entry: &Entry<'a>,
) -> Vec<(Option<Range<u64>>, Where)> {
    match entry.attr_value(gimli::DW_AT_location) {
        Some(AttributeValue::Exprloc(expression)) => place(tree, unit, expression)
            .map(|at| (None, at))
            .into_iter()
            .collect(),
        Some(value) => tree
            .dwarf(unit)
            .attr_locations_offset(&tree.units[unit], value)
            .ok()
            .flatten()
            .map(|offset| location_list(tree, unit, offset))
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(range, expression)| Some((Some(range), place(tree, unit, expression)?)))
            .collect(),
        None => Vec::new(),
    }
}

fn assignments(
    module: &Module,
    code: &[u8],
    at: u64,
    params: u32,
    held: &[Held],
    starts: &HashSet<u64>,
    renumbered: bool,
) -> BTreeMap<u64, (u32, Vec<(usize, bool)>)> {
    let mut insns = Vec::new();
    let mut offset = 0usize;
    while let Some(insn) = code.get(offset..).and_then(insn::decode_any) {
        let len = insn.len;
        insns.push((at + offset as u64, insn));
        offset += len;
    }
    let nth_at: HashMap<u64, usize> = insns
        .iter()
        .enumerate()
        .map(|(nth, (start, _))| (*start, nth))
        .chain(std::iter::once((at + offset as u64, insns.len())))
        .collect();
    let written = |nth: usize| match insns[nth].1.op {
        Operator::LocalSet { local_index } | Operator::LocalTee { local_index } => {
            Some(local_index)
        }
        _ => None,
    };
    let straight = |nth: usize| {
        matches!(
            insns[nth].1.flow(),
            insn::Flow::Normal | insn::Flow::Call | insn::Flow::IndirectCall
        )
    };
    let reaching = |mut nth: usize, local: u32| {
        while nth > 0 {
            nth -= 1;
            if written(nth) == Some(local) {
                return Some(nth);
            }
            if !straight(nth) {
                return None;
            }
        }
        None
    };
    let crowded = |set: usize| {
        let (mut need, mut nth) = (1i64, set);
        while nth > 0 {
            nth -= 1;
            if written(nth).is_some() {
                return true;
            }
            let insn = &insns[nth].1;
            let Some(net) = lift::stack_effect(insn, module.resolve(&insn.op).as_ref())
                .filter(|_| straight(nth))
            else {
                return false;
            };
            need += net;
            if need <= 0 {
                return nth > 0 && written(nth - 1).is_some();
            }
        }
        false
    };

    let mut votes: HashMap<u32, BTreeMap<u32, BTreeSet<usize>>> = HashMap::new();
    if renumbered {
        for (range, at) in held.iter().flat_map(|variable| &variable.places) {
            let (Some(range), Where::Local(old) | Where::PointedByLocal(old)) = (range, *at) else {
                continue;
            };
            let Some(&from) = nth_at.get(&range.start) else {
                continue;
            };
            let opening = insns
                .get(from)
                .is_some_and(|(_, insn)| insn.flow() == insn::Flow::BlockStart);
            if starts.contains(&range.start) || opening {
                continue;
            }
            if let Some(set) = from.checked_sub(1)
                && let Some(new) = written(set)
                && !crowded(set)
            {
                let sets = votes.entry(old).or_default().entry(new).or_default();
                sets.insert(set);
            }
        }
    }
    let mut spans: BTreeMap<u32, Vec<&Range<u64>>> = BTreeMap::new();
    for (range, at) in held.iter().flat_map(|variable| &variable.places) {
        if let (Some(range), Where::Local(old) | Where::PointedByLocal(old)) = (range, *at) {
            spans.entry(old).or_default().push(range);
        }
    }
    let mut chosen: HashMap<u32, u32> = votes
        .iter()
        .filter_map(|(&old, voted)| {
            let [(&new, sets)] = voted.iter().collect::<Vec<_>>()[..] else {
                return None;
            };
            let reached: HashSet<u64> = spans
                .get(&old)?
                .iter()
                .map(|range| range.start)
                .filter(|start| {
                    nth_at
                        .get(start)
                        .is_some_and(|&from| reaching(from, new).is_some())
                })
                .collect();
            (old >= params && new >= params && (sets.len() >= 2 || reached.len() >= 2))
                .then_some((old, new))
        })
        .collect();
    let mut sharing: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (&old, &new) in &chosen {
        sharing.entry(new).or_default().push(old);
    }
    let overlap = |one: u32, other: u32| {
        spans[&one].iter().any(|a| {
            spans[&other]
                .iter()
                .any(|b| a.start < b.end && b.start < a.end)
        })
    };
    for olds in sharing.values() {
        for &one in olds {
            if olds
                .iter()
                .any(|&other| other != one && overlap(one, other))
            {
                chosen.remove(&one);
            }
        }
    }
    let mapped = |old: u32| {
        if !renumbered || old < params {
            Some(old)
        } else {
            chosen.get(&old).copied()
        }
    };

    let mut found: BTreeMap<usize, Vec<(bool, bool, usize, bool)>> = BTreeMap::new();
    for (nth, variable) in held.iter().enumerate() {
        for (range, at) in &variable.places {
            let (Where::Local(old) | Where::PointedByLocal(old)) = *at else {
                continue;
            };
            let Some(local) = mapped(old) else {
                continue;
            };
            let pointed = matches!(at, Where::PointedByLocal(_));
            match range {
                Some(range) => {
                    if let Some(&from) = nth_at.get(&range.start)
                        && let Some(set) = reaching(from, local)
                    {
                        let later = set + 1 != from;
                        let candidate = (later, variable.inlined, nth, pointed);
                        found.entry(set).or_default().push(candidate);
                    }
                }
                None if !renumbered => {
                    for set in (0..insns.len()).filter(|set| written(*set) == Some(local)) {
                        let (start, insn) = &insns[set];
                        let after = start + insn.len as u64;
                        if variable.scope.iter().any(|scope| scope.contains(&after)) {
                            let candidate = (true, variable.inlined, nth, pointed);
                            found.entry(set).or_default().push(candidate);
                        }
                    }
                }
                None => {}
            }
        }
    }
    found
        .into_iter()
        .filter_map(|(set, mut candidates)| {
            candidates.sort_unstable();
            let mut seen = HashSet::new();
            let ordered = candidates
                .into_iter()
                .filter(|(_, _, nth, _)| seen.insert(*nth))
                .map(|(_, _, nth, pointed)| (nth, pointed))
                .collect();
            Some((insns[set].0, (written(set)?, ordered)))
        })
        .collect()
}

fn from_frame(spilled: &BTreeMap<i64, (usize, u64)>, offset: i64, size: u64) -> Option<Entered> {
    let end = offset.checked_add(i64::try_from(size).ok()?)?;
    let relative = |at: i64| u64::try_from(at - offset).ok();
    let (mut parts, mut locals, mut at) = (Vec::new(), Vec::new(), offset);
    for (&start, &(local, width)) in spilled.range(offset..end) {
        let gap = u64::try_from(start.checked_sub(at)?).ok()?;
        if gap != 0 {
            parts.push((relative(at)?, gap, Part::Unheld));
        }
        parts.push((relative(start)?, width, Part::Held));
        locals.push(local);
        at = start.checked_add(i64::try_from(width).ok()?)?;
    }
    if at > end {
        return None;
    }
    if at < end {
        parts.push((relative(at)?, u64::try_from(end - at).ok()?, Part::Unheld));
    }
    let first = *locals.first()?;
    let consecutive = locals
        .iter()
        .enumerate()
        .all(|(nth, local)| *local == first + nth);
    consecutive.then_some(Entered::Pieces(first, parts))
}

fn is_constant(tree: &Tree, unit: usize, expression: gimli::Expression<Slice>) -> bool {
    let mut operations = expression.operations(tree.encoding(unit));
    let mut next = || operations.next();
    matches!(
        (next(), next(), next()),
        (
            Ok(Some(
                gimli::Operation::UnsignedConstant { .. } | gimli::Operation::SignedConstant { .. }
            )),
            Ok(Some(gimli::Operation::StackValue)),
            Ok(None),
        )
    )
}

fn truncated(tree: &Tree, unit: usize, expression: gimli::Expression<Slice>) -> Option<u32> {
    let mut operations = expression.operations(tree.encoding(unit));
    let mut next = || operations.next().ok();
    match (next()?, next()?, next()?, next()?, next()?) {
        (
            Some(gimli::Operation::WasmLocal { index }),
            Some(gimli::Operation::UnsignedConstant { value: mask }),
            Some(gimli::Operation::And),
            Some(gimli::Operation::StackValue),
            None,
        ) if mask != 0 && mask & mask.wrapping_add(1) == 0 => Some(index),
        _ => None,
    }
}

fn location_list<'a>(
    tree: &Tree<'a>,
    unit: usize,
    offset: gimli::LocationListsOffset,
) -> Vec<(Range<u64>, gimli::Expression<Slice<'a>>)> {
    let (dwarf, unit) = (tree.dwarf(unit), &tree.units[unit]);
    let size = unit.encoding().address_size;
    let address = |index| dwarf.address(unit, index).ok();
    let mut base = Some(unit.low_pc);
    let mut found = Vec::new();
    let Ok(mut raw) = dwarf.raw_locations(unit, offset) else {
        return found;
    };
    while let Ok(Some(entry)) = raw.next() {
        let (begin, end, data) = match entry {
            gimli::RawLocListEntry::BaseAddress { addr } => {
                base = Some(addr);
                continue;
            }
            gimli::RawLocListEntry::BaseAddressx { addr } => {
                base = address(addr);
                continue;
            }
            gimli::RawLocListEntry::AddressOrOffsetPair { begin, end, data }
            | gimli::RawLocListEntry::OffsetPair { begin, end, data } => {
                let base = base.filter(|base| !is_tombstone(*base, size));
                let at = |offset: u64| base?.checked_add(offset);
                (at(begin), at(end), data)
            }
            gimli::RawLocListEntry::StartxEndx { begin, end, data } => {
                (address(begin), address(end), data)
            }
            gimli::RawLocListEntry::StartxLength {
                begin,
                length,
                data,
            } => {
                let begin = address(begin);
                (
                    begin,
                    begin.and_then(|begin| begin.checked_add(length)),
                    data,
                )
            }
            gimli::RawLocListEntry::StartEnd { begin, end, data } => (Some(begin), Some(end), data),
            gimli::RawLocListEntry::StartLength {
                begin,
                length,
                data,
            } => (Some(begin), begin.checked_add(length), data),
            gimli::RawLocListEntry::DefaultLocation { data } => (Some(0), Some(u64::MAX), data),
        };
        if let (Some(begin), Some(end)) = (begin, end)
            && !is_tombstone(begin, size)
        {
            found.push((begin..end, data));
        }
    }
    found
}

fn passed(tree: &Tree, ty: Option<Die>) -> Passed {
    let Some(ty) = ty else {
        return Passed::Unknown;
    };
    let Some((record, entry)) = underlying(tree, Some(ty)) else {
        return Passed::Unknown;
    };
    match entry.tag() {
        gimli::DW_TAG_array_type if is_vector(&entry) => return Passed::Value(ty),
        gimli::DW_TAG_array_type => return Passed::Unknown,
        _ if is_aggregate_scalar(tree, record.unit, &entry) => return Passed::Indirect(ty),
        tag if !is_record(tag) => return scalar(tree, ty),
        _ => {}
    }
    if is_declaration(&entry) {
        return Passed::Unknown;
    }
    let cplusplus = is_cplusplus(tree, record.unit);
    match entry.attr_value(gimli::DW_AT_calling_convention) {
        Some(AttributeValue::CallingConvention(gimli::DW_CC_pass_by_reference)) => {
            return Passed::Indirect(ty);
        }
        None if cplusplus => return Passed::Unknown,
        _ => {}
    }
    let mut budget = MAX_CLASSIFIED;
    match element(tree, record, 0, cplusplus, &mut budget) {
        Element::Empty => Passed::Omitted,
        Element::One(element) => scalar(tree, element),
        Element::Many => Passed::Indirect(ty),
        Element::Unknown => Passed::Unknown,
    }
}

fn scalar(tree: &Tree, ty: Die) -> Passed {
    let wide = underlying(tree, Some(ty)).is_some_and(|(_, entry)| {
        entry.tag() == gimli::DW_TAG_base_type && udata(&entry, gimli::DW_AT_byte_size) == Some(16)
    });
    if wide {
        Passed::Split(ty)
    } else {
        Passed::Value(ty)
    }
}

fn is_vector(entry: &Entry) -> bool {
    entry.attr_value(gimli::DW_AT_GNU_vector).is_some()
}

fn is_aggregate_scalar(tree: &Tree, unit: usize, entry: &Entry) -> bool {
    match entry.tag() {
        gimli::DW_TAG_base_type => {
            entry.attr_value(gimli::DW_AT_encoding)
                == Some(AttributeValue::Encoding(gimli::DW_ATE_complex_float))
        }
        gimli::DW_TAG_ptr_to_member_type => underlying(tree, tree.type_of(unit, entry))
            .is_some_and(|(_, target)| target.tag() == gimli::DW_TAG_subroutine_type),
        _ => false,
    }
}

fn plain(tree: &Tree, ty: Option<Die>) -> Passed {
    let Some((at, entry)) = underlying(tree, ty) else {
        return Passed::Unknown;
    };
    match ty {
        Some(ty)
            if !is_aggregate(entry.tag())
                && !is_aggregate_scalar(tree, at.unit, &entry)
                && byte_size(tree, at, 0).is_some_and(|size| (1..16).contains(&size)) =>
        {
            Passed::Value(ty)
        }
        _ => Passed::Unknown,
    }
}

fn by_value(tree: &Tree, unit: usize) -> bool {
    language(tree, unit) == Some(gimli::DW_LANG_Rust)
}

fn language(tree: &Tree, unit: usize) -> Option<gimli::DwLang> {
    match tree.root(unit)?.attr_value(gimli::DW_AT_language)? {
        AttributeValue::Language(language) => Some(language),
        _ => None,
    }
}

fn is_cplusplus(tree: &Tree, unit: usize) -> bool {
    matches!(
        language(tree, unit),
        Some(
            gimli::DW_LANG_C_plus_plus
                | gimli::DW_LANG_C_plus_plus_03
                | gimli::DW_LANG_C_plus_plus_11
                | gimli::DW_LANG_C_plus_plus_14
                | gimli::DW_LANG_C_plus_plus_17
                | gimli::DW_LANG_C_plus_plus_20
                | gimli::DW_LANG_ObjC_plus_plus
        )
    )
}

fn speaks_c(tree: &Tree, unit: usize) -> bool {
    is_cplusplus(tree, unit)
        || matches!(
            language(tree, unit),
            Some(
                gimli::DW_LANG_C89
                    | gimli::DW_LANG_C
                    | gimli::DW_LANG_C99
                    | gimli::DW_LANG_C11
                    | gimli::DW_LANG_C17
                    | gimli::DW_LANG_ObjC
            )
        )
}

const MAX_CLASSIFIED: usize = 4096;

fn element(tree: &Tree, record: Die, depth: usize, cplusplus: bool, budget: &mut usize) -> Element {
    let Some(entry) = tree.entry(record) else {
        return Element::Unknown;
    };
    if depth >= MAX_TYPE_DEPTH || *budget == 0 || is_declaration(&entry) {
        return Element::Unknown;
    }
    *budget -= 1;
    let unit = record.unit;
    let members = children(tree, record);
    let record_size = udata(&entry, gimli::DW_AT_byte_size);
    let bases: Vec<Option<Element>> = members
        .iter()
        .map(|child| {
            (child.tag() == gimli::DW_TAG_inheritance).then(|| {
                stripped(tree, tree.type_of(unit, child)).map_or(Element::Unknown, |base| {
                    element(tree, base, depth + 1, cplusplus, budget)
                })
            })
        })
        .collect();
    let spans: Vec<Option<(u64, u64)>> = members
        .iter()
        .zip(&bases)
        .map(|(child, base)| match (child.tag(), base) {
            (gimli::DW_TAG_member, _) if !is_declaration(child) => {
                member_storage(tree, unit, child)
            }
            (gimli::DW_TAG_inheritance, Some(base)) if !matches!(base, Element::Empty) => {
                let at = member_offset(tree, unit, child)?;
                let base = stripped(tree, tree.type_of(unit, child))?;
                Some((at, byte_size(tree, base, 0)?))
            }
            _ => None,
        })
        .collect();
    let union = entry.tag() == gimli::DW_TAG_union_type;
    let mut starts: Vec<(u64, u64, usize)> = spans
        .iter()
        .enumerate()
        .filter_map(|(nth, span)| span.map(|(start, len)| (start, start.saturating_add(len), nth)))
        .collect();
    starts.sort_unstable();
    let mut widest: Vec<((u64, usize), u64)> = Vec::with_capacity(starts.len());
    for &(_, end, nth) in &starts {
        let next = match widest.last() {
            Some(&((top, holder), second)) if end <= top => ((top, holder), second.max(end)),
            Some(&((top, _), _)) => ((end, nth), top),
            None => ((end, nth), 0),
        };
        widest.push(next);
    }
    let shares_storage = |nth: usize| {
        !union
            && spans[nth].is_some_and(|(at, _)| {
                let started = starts.partition_point(|&(start, _, _)| start <= at);
                record_size.is_some_and(|size| at >= size)
                    || started.checked_sub(1).is_some_and(|last| {
                        let ((top, holder), second) = widest[last];
                        (if holder == nth { second } else { top }) > at
                    })
            })
    };
    let mut found = None;
    let mut unknown = false;
    let mut unknown_elements = 0usize;
    for (nth, child) in members.iter().enumerate() {
        let (classified, counts) = match (child.tag(), &bases[nth]) {
            (gimli::DW_TAG_inheritance, Some(base)) => (*base, false),
            (gimli::DW_TAG_member, _) => {
                if is_declaration(child) || (!has_name(child) && is_bitfield(child)) {
                    continue;
                }
                let unique = cplusplus && !shares_storage(nth);
                let classified = match tree.type_of(unit, child) {
                    Some(member) => field(tree, member, depth, cplusplus, unique, budget),
                    None => Element::Unknown,
                };
                (classified, unique)
            }
            _ => continue,
        };
        let inner = match classified {
            Element::Empty => continue,
            Element::Unknown if counts => {
                unknown_elements += 1;
                if found.is_some() || unknown_elements > 1 {
                    return Element::Many;
                }
                continue;
            }
            Element::Unknown => {
                unknown = true;
                continue;
            }
            Element::Many => return Element::Many,
            Element::One(inner) => inner,
        };
        if found.is_some() || unknown_elements > 0 {
            return Element::Many;
        }
        found = Some(inner);
    }
    if unknown || unknown_elements > 0 {
        return Element::Unknown;
    }
    let Some(found) = found else {
        return Element::Empty;
    };
    match (byte_size(tree, found, 0), record_size) {
        (Some(element), Some(record)) if element == record => Element::One(found),
        (Some(_), Some(_)) => Element::Many,
        _ => Element::Unknown,
    }
}

fn field(
    tree: &Tree,
    member: Die,
    depth: usize,
    cplusplus: bool,
    unique: bool,
    budget: &mut usize,
) -> Element {
    let Some((at, entry)) = underlying(tree, Some(member)) else {
        return Element::Unknown;
    };
    if depth >= MAX_TYPE_DEPTH {
        return Element::Unknown;
    }
    match entry.tag() {
        gimli::DW_TAG_array_type if is_vector(&entry) => Element::One(member),
        gimli::DW_TAG_array_type => {
            let Some(inner) = tree.type_of(at.unit, &entry) else {
                return Element::Unknown;
            };
            let mut single = true;
            for count in dimensions(tree, at) {
                match count {
                    None => return Element::Many,
                    Some(0) => return Element::Empty,
                    Some(1) => {}
                    Some(_) => single = false,
                }
            }
            match field(tree, inner, depth + 1, cplusplus, true, budget) {
                Element::Empty => Element::Empty,
                Element::Unknown => Element::Unknown,
                found if single => found,
                _ => Element::Many,
            }
        }
        tag if is_record(tag) => match element(tree, at, depth + 1, cplusplus, budget) {
            Element::Empty if cplusplus && unique => Element::Many,
            found => found,
        },
        _ if is_aggregate_scalar(tree, at.unit, &entry) => Element::Many,
        _ => Element::One(member),
    }
}

fn is_record(tag: gimli::DwTag) -> bool {
    matches!(
        tag,
        gimli::DW_TAG_structure_type | gimli::DW_TAG_class_type | gimli::DW_TAG_union_type
    )
}

fn is_bitfield(entry: &Entry) -> bool {
    entry.attr_value(gimli::DW_AT_bit_size).is_some()
}

fn is_aggregate(tag: gimli::DwTag) -> bool {
    is_record(tag) || tag == gimli::DW_TAG_array_type
}

fn has_name(entry: &Entry) -> bool {
    entry.attr_value(gimli::DW_AT_name).is_some()
}

fn is_declaration(entry: &Entry) -> bool {
    matches!(
        entry.attr_value(gimli::DW_AT_declaration),
        Some(AttributeValue::Flag(true))
    )
}

fn is_tombstone(address: u64, size: u8) -> bool {
    let bits = u32::from(size) * 8;
    if bits == 0 || bits > 64 {
        return false;
    }
    let mask = u64::MAX >> (64 - bits);
    address & mask >= mask - 1
}

fn udata(entry: &Entry, attr: gimli::DwAt) -> Option<u64> {
    let value = entry.attr_value(attr)?;
    match value {
        AttributeValue::Udata(value) => Some(value),
        AttributeValue::Sdata(value) => u64::try_from(value).ok(),
        AttributeValue::FileIndex(value) => Some(value),
        AttributeValue::Data1(value) => Some(u64::from(value)),
        AttributeValue::Data2(value) => Some(u64::from(value)),
        AttributeValue::Data4(value) => Some(u64::from(value)),
        AttributeValue::Data8(value) => Some(value),
        _ => None,
    }
}

fn member_bits(tree: &Tree, unit: usize, member: &Entry) -> Option<u64> {
    udata(member, gimli::DW_AT_data_bit_offset).or_else(|| {
        let at = member_offset(tree, unit, member)?.checked_mul(8)?;
        let Some(from_top) = udata(member, gimli::DW_AT_bit_offset) else {
            return Some(at);
        };
        let width = udata(member, gimli::DW_AT_bit_size)?;
        let storage = udata(member, gimli::DW_AT_byte_size)
            .or_else(|| byte_size(tree, tree.type_of(unit, member)?, 0))?;
        at.checked_add(storage.checked_mul(8)?)?
            .checked_sub(from_top.checked_add(width)?)
    })
}

fn member_storage(tree: &Tree, unit: usize, member: &Entry) -> Option<(u64, u64)> {
    let bits = member_bits(tree, unit, member)?;
    let length = match udata(member, gimli::DW_AT_bit_size) {
        Some(width) => (bits % 8).checked_add(width)?.div_ceil(8),
        None => byte_size(tree, tree.type_of(unit, member)?, 0)?,
    };
    Some((bits / 8, length))
}

fn member_offset(tree: &Tree, unit: usize, entry: &Entry) -> Option<u64> {
    let Some(value) = entry.attr_value(gimli::DW_AT_data_member_location) else {
        return Some(0);
    };
    let AttributeValue::Exprloc(expression) = value else {
        return udata(entry, gimli::DW_AT_data_member_location);
    };
    let mut operations = expression.operations(tree.encoding(unit));
    match (operations.next().ok()?, operations.next().ok()?) {
        (Some(gimli::Operation::PlusConstant { value }), None) => Some(value),
        _ => None,
    }
}

fn constant(entry: &Entry, attr: gimli::DwAt) -> Option<u64> {
    match entry.attr_value(attr)? {
        AttributeValue::Sdata(value) => Some(value as u64),
        other => udata_of(other),
    }
}

fn udata_of(value: AttributeValue<Slice>) -> Option<u64> {
    match value {
        AttributeValue::Udata(value) => Some(value),
        AttributeValue::Data1(value) => Some(u64::from(value)),
        AttributeValue::Data2(value) => Some(u64::from(value)),
        AttributeValue::Data4(value) => Some(u64::from(value)),
        AttributeValue::Data8(value) => Some(value),
        _ => None,
    }
}

fn frame_note(at: i64, name: &str, ty: Option<&Ref<Type>>) -> String {
    let ty = ty.map_or_else(|| "?".to_string(), |ty| ty.to_string());
    let sign = if at < 0 { "-" } else { "+" };
    format!("{sign}{:#x} {name} ({ty})", at.unsigned_abs())
}

fn stack_slots(
    found: Vec<(i64, String, Option<Ref<Type>>, bool)>,
    frame: Option<u64>,
    listed: &mut Vec<String>,
) -> Vec<NamedVariableWithType> {
    let size = frame.and_then(|size| i64::try_from(size).ok());
    let mut placed: Vec<(i64, i64, String, Ref<Type>, bool)> = Vec::new();
    for (at, name, ty, inlined) in found {
        let end = ty
            .as_ref()
            .and_then(|ty| at.checked_add(i64::try_from(ty.width()).ok()?));
        match (size, ty, end) {
            (Some(size), Some(ty), Some(end)) if at >= 0 && end <= size => {
                if !placed
                    .iter()
                    .any(|(start, _, named, ..)| *start == at && *named == name)
                {
                    placed.push((at, end, name, ty, inlined));
                }
            }
            (_, ty, _) if !inlined => listed.push(frame_note(at, &name, ty.as_ref())),
            _ => {}
        }
    }
    let spans: Vec<(i64, i64, bool)> = placed
        .iter()
        .map(|(start, end, .., inlined)| (*start, *end, *inlined))
        .collect();

    let mut slots = Vec::new();
    for ((at, _, name, ty, inlined), kept) in placed.into_iter().zip(unclashed(&spans)) {
        match size {
            Some(size) if kept => slots.push(NamedVariableWithType::new(
                Variable::from_stack_offset(at - size),
                Conf::new(ty, MAX_CONFIDENCE),
                name,
                false,
            )),
            _ if !inlined => listed.push(frame_note(at, &name, Some(&ty))),
            _ => {}
        }
    }
    slots
}

fn unclashed(spans: &[(i64, i64, bool)]) -> Vec<bool> {
    spans
        .iter()
        .enumerate()
        .map(|(nth, &(start, end, inlined))| {
            !spans.iter().enumerate().any(|(other, &(from, to, below))| {
                other != nth && from < end && start < to && (inlined || !below)
            })
        })
        .collect()
}

fn body_at(module: &Module, address: u64) -> Option<(u32, &FunctionInfo)> {
    module
        .body_covering(address)
        .filter(|(_, info)| info.start == address)
}

fn code_section(module: &Module) -> Option<&SectionSpan> {
    module.sections.iter().find(|section| section.code)
}

/// Where a DWARF address of zero points
fn code_base(module: &Module) -> Option<u64> {
    code_section(module).map(|section| section.start)
}

fn section_of(module: &Module, name: &str) -> Option<(u64, u64)> {
    module
        .sections
        .iter()
        .find(|section| section.name == name)
        .map(|section| (section.start, section.end))
}

fn carries_dwarf(module: &Module) -> bool {
    section_of(module, SectionId::DebugInfo.name()).is_some_and(|(start, end)| end > start)
}

struct Source {
    image: Vec<u8>,
    modules: Vec<Sections>,
}

struct Sections {
    custom: HashMap<String, Range<usize>>,
    code: Option<Range<usize>>,
}

impl Source {
    fn new(image: Vec<u8>, modules: &[impl Borrow<Module>]) -> Self {
        let modules = modules
            .iter()
            .map(|module| {
                let module = module.borrow();
                let offsets = |span: &SectionSpan| -> Option<Range<usize>> {
                    let start = usize::try_from(module.layout.file_offset(span.start)?).ok()?;
                    let end = usize::try_from(module.layout.file_offset(span.end)?).ok()?;
                    Some(start..end)
                };
                let mut custom = HashMap::new();
                for span in module.sections.iter().filter(|span| !span.code) {
                    if let Some(range) = offsets(span) {
                        custom.entry(span.name.clone()).or_insert(range);
                    }
                }
                Sections {
                    custom,
                    code: code_section(module).and_then(offsets),
                }
            })
            .collect();
        Self { image, modules }
    }

    fn parsed(image: Vec<u8>) -> Self {
        let modules = module::parse_all(&image, 0);
        Self::new(image, &modules)
    }

    fn section(&self, module: usize, name: &str) -> &[u8] {
        self.modules
            .get(module)
            .and_then(|sections| sections.custom.get(name))
            .and_then(|range| self.image.get(range.clone()))
            .unwrap_or(&[])
    }

    fn has_dwarf(&self) -> bool {
        (0..self.modules.len())
            .any(|module| !self.section(module, SectionId::DebugInfo.name()).is_empty())
    }

    fn code(&self, module: usize) -> Option<&[u8]> {
        self.image.get(self.modules.get(module)?.code.clone()?)
    }

    fn same_code(&self, other: &Source) -> bool {
        self.modules.len() == other.modules.len()
            && (0..self.modules.len()).all(|module| self.code(module) == other.code(module))
    }
}

const EXTERNAL_DEBUG_INFO: &str = "external_debug_info";

fn referenced_file(payload: &[u8]) -> Option<String> {
    let mut reader = wasmparser::BinaryReader::new(payload, 0);
    reader.read_string().ok().map(str::to_owned)
}

fn reference(view: &BinaryView, modules: &[Arc<Module>]) -> Option<String> {
    let raw = view.parent_view()?;
    modules.iter().find_map(|module| {
        let (start, end) = section_of(module, EXTERNAL_DEBUG_INFO)?;
        let offset = module.layout.file_offset(start)?;
        let len = usize::try_from(end.checked_sub(start)?).ok()?;
        referenced_file(&raw.read_vec(offset, len))
    })
}

fn matching(found: Source, own: &Source, path: &Path) -> Option<Source> {
    if !found.has_dwarf() {
        return None;
    }
    if !found.same_code(own) {
        tracing::warn!(
            "wasm DWARF: {} does not hold the same code as this module, so its debug information \
             is not applied",
            path.display()
        );
        return None;
    }
    tracing::info!(
        "wasm DWARF: reading debug information from {}",
        path.display()
    );
    Some(found)
}

fn external(debug_file: &BinaryView, own: &Source) -> Option<Source> {
    let path = debug_file.file().file_path();
    let found = Source::parsed(file_image(debug_file));
    if !found.has_dwarf() {
        tracing::warn!("wasm DWARF: {} carries no DWARF", path.display());
        return None;
    }
    matching(found, own, &path)
}

fn sibling(view: &BinaryView, modules: &[Arc<Module>], own: &Source) -> Option<Source> {
    let reference = reference(view, modules)?;
    sibling_files(view, &reference)
        .into_iter()
        .find_map(|path| matching(Source::parsed(read_file(&path)?), own, &path))
}

fn skeletons(dwarf: &Dwarf<Slice>) -> Vec<(gimli::DwoId, Option<String>)> {
    let mut found = Vec::new();
    let mut headers = dwarf.units();
    while let Ok(Some(header)) = headers.next() {
        let Ok(unit) = dwarf.unit(header) else {
            continue;
        };
        let Some(id) = unit.dwo_id else {
            continue;
        };
        let name = unit
            .dwo_name()
            .ok()
            .flatten()
            .and_then(|value| dwarf.attr_string(&unit, value).ok())
            .and_then(|raw| std::str::from_utf8(raw.slice()).ok().map(str::to_owned));
        found.push((id, name));
    }
    found
}

fn split_images(view: &BinaryView, dwarf: &Dwarf<Slice>) -> Vec<Vec<u8>> {
    let mut wanted = skeletons(dwarf);
    if wanted.is_empty() {
        return Vec::new();
    }
    let mut images = Vec::new();
    let file = view.file();
    let path = file
        .original_file_path()
        .unwrap_or_else(|| file.file_path());
    let package = path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| sibling_files(view, &format!("{name}.dwp")))
        .unwrap_or_default()
        .into_iter()
        .find_map(|path| Some((read_file(&path)?, path)));
    if let Some((image, path)) = package {
        if let Some(index) = custom_sections(&image)
            .get(SectionId::DebugCuIndex.name())
            .and_then(|data| gimli::DebugCuIndex::new(data, LittleEndian).index().ok())
        {
            wanted.retain(|(id, _)| index.find(id.0).is_none());
        }
        tracing::info!("wasm DWARF: reading split units from {}", path.display());
        images.push(image);
    }

    let mut missing = Vec::new();
    let mut named = HashSet::new();
    for (_, name) in wanted {
        let Some(name) = name else {
            missing.push("an unnamed file".to_owned());
            continue;
        };
        if !named.insert(name.clone()) {
            continue;
        }
        match sibling_files(view, &name)
            .into_iter()
            .find_map(|path| Some((read_file(&path)?, path)))
        {
            Some((image, path)) => {
                tracing::info!("wasm DWARF: reading split units from {}", path.display());
                images.push(image);
            }
            None => missing.push(name),
        }
    }
    if let Some(first) = missing.first() {
        tracing::warn!(
            "wasm DWARF: {} compilation units keep their DWARF in files that were not found, \
             {first} among them",
            missing.len()
        );
    }
    images
}

fn split_dwarfs<'a>(images: &'a [Vec<u8>], parent: &Dwarf<Slice<'a>>) -> Vec<Dwarf<Slice<'a>>> {
    let mut dwarfs = Vec::new();
    for image in images {
        let sections = custom_sections(image);
        let section = |id: SectionId| -> Result<Slice<'a>, gimli::Error> {
            let data = id
                .dwo_name()
                .and_then(|name| sections.get(name))
                .copied()
                .unwrap_or(&[]);
            Ok(EndianSlice::new(data, LittleEndian))
        };
        if sections.contains_key(SectionId::DebugCuIndex.name()) {
            let empty = EndianSlice::new(&[], LittleEndian);
            let Ok(package) = gimli::DwarfPackage::load(section, empty) else {
                continue;
            };
            for row in 1..=package.cu_index.unit_count() {
                dwarfs.extend(package.cu_sections(row, parent).ok());
            }
            for row in 1..=package.tu_index.unit_count() {
                dwarfs.extend(package.tu_sections(row, parent).ok());
            }
        } else if let Ok(mut dwo) = Dwarf::load(section) {
            dwo.make_dwo(parent);
            dwarfs.push(dwo);
        }
    }
    dwarfs
}

fn custom_sections(image: &[u8]) -> HashMap<&str, &[u8]> {
    let mut sections = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(image) {
        match payload {
            Ok(wasmparser::Payload::CustomSection(section)) => {
                sections.entry(section.name()).or_insert(section.data());
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    sections
}

pub(crate) fn sibling_files(view: &BinaryView, reference: &str) -> Vec<PathBuf> {
    if !settings::sibling_debug_files(view) {
        return Vec::new();
    }
    let file = view.file();
    let opened = file.file_path();
    let original = file.original_file_path().unwrap_or_else(|| opened.clone());
    let beside: Vec<&Path> = [original.parent(), opened.parent()]
        .into_iter()
        .flatten()
        .collect();
    candidate_files(reference, &beside, &settings::debug_directories(view))
}

fn candidate_files(reference: &str, beside: &[&Path], directories: &[String]) -> Vec<PathBuf> {
    let url = reference.contains("://");
    let reference = if url {
        reference.split(['?', '#']).next().unwrap_or(reference)
    } else {
        reference
    };
    let name = reference.rsplit(['/', '\\']).next().unwrap_or(reference);
    let relative = Path::new(reference);
    let plain = !url
        && relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)));
    let mut components = Path::new(name).components();
    let single = matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    );

    let mut candidates: Vec<PathBuf> = Vec::new();
    for directory in beside {
        if plain {
            candidates.push(directory.join(relative));
        }
        if single {
            candidates.push(directory.join(name));
        }
    }
    if single {
        candidates.extend(
            directories
                .iter()
                .map(|directory| Path::new(directory).join(name)),
        );
    }

    let mut seen = HashSet::new();
    candidates.retain(|path| {
        seen.insert(path.clone())
            && std::fs::metadata(path)
                .is_ok_and(|meta| meta.is_file() && meta.len() <= module::MAX_IMAGE_LEN as u64)
    });
    candidates
}

pub(crate) fn read_file(path: &Path) -> Option<Vec<u8>> {
    let limit = module::MAX_IMAGE_LEN as u64;
    let mut image = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .take(limit + 1)
        .read_to_end(&mut image)
        .ok()?;
    (image.len() as u64 <= limit).then_some(image)
}

pub(crate) fn file_image(view: &BinaryView) -> Vec<u8> {
    // From the parent rather than through this view, which has holes where a data segment's bytes
    // are shown in linear memory instead and stops reading at the first of them
    let raw = view.parent_view().unwrap_or_else(|| view.to_owned());
    let len = raw.len().min(module::MAX_IMAGE_LEN as u64) as usize;
    raw.read_vec(0, len)
}

pub(crate) fn modules_of(view: &BinaryView) -> Vec<Arc<Module>> {
    if view.parent_view().is_none() {
        return Vec::new();
    }
    module::all(crate::arch::view_id(view))
}

pub fn register() {
    DebugInfoParser::register(NAME, WasmDwarf);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The names gimli asks for are the ones a wasm custom section is called, which is why reading
    /// them directly works
    #[test]
    fn dwarf_sections_are_named_the_way_wasm_spells_them() {
        assert_eq!(SectionId::DebugInfo.name(), ".debug_info");
        assert_eq!(SectionId::DebugAbbrev.name(), ".debug_abbrev");
        assert_eq!(SectionId::DebugStr.name(), ".debug_str");
        assert_eq!(SectionId::DebugLine.name(), ".debug_line");
    }

    #[test]
    fn debug_information_is_taken_only_from_a_file_with_the_same_code() {
        let assemble = |text: &str| wat::parse_str(text).expect("assembles");
        let stripped = Source::parsed(assemble(
            r#"(module (func nop) (@custom "external_debug_info" "\0fargs.debug.wasm"))"#,
        ));
        let split = Source::parsed(assemble(
            r#"(module (func nop) (@custom ".debug_info" "\01\02\03"))"#,
        ));
        let other = Source::parsed(assemble(
            r#"(module (func unreachable) (@custom ".debug_info" "\01\02\03"))"#,
        ));

        assert!(!stripped.has_dwarf());
        assert_eq!(
            referenced_file(stripped.section(0, EXTERNAL_DEBUG_INFO)).as_deref(),
            Some("args.debug.wasm")
        );
        assert!(split.has_dwarf());
        assert_eq!(split.section(0, ".debug_info"), [1, 2, 3]);
        assert_eq!(referenced_file(split.section(0, EXTERNAL_DEBUG_INFO)), None);

        let found = matching(split, &stripped, Path::new("args.debug.wasm"));
        assert!(found.is_some(), "the same body, with DWARF");
        assert!(
            matching(other, &stripped, Path::new("other.wasm")).is_none(),
            "a body that differs"
        );
        let plain = Source::parsed(assemble("(module (func nop))"));
        assert!(
            matching(plain, &stripped, Path::new("plain.wasm")).is_none(),
            "no DWARF to take"
        );
    }

    #[test]
    fn a_referenced_file_is_only_looked_for_where_it_is_expected() {
        let root = std::env::temp_dir().join(format!("binja-wasm-sibling-{}", std::process::id()));
        let (beside, elsewhere) = (root.join("beside"), root.join("elsewhere"));
        std::fs::create_dir_all(beside.join("sub/dir.wasm")).expect("directories");
        std::fs::create_dir_all(&elsewhere).expect("directories");
        for file in [
            beside.join("a.debug.wasm"),
            beside.join("sub/b.debug.wasm"),
            elsewhere.join("c.debug.wasm"),
        ] {
            std::fs::write(file, b"\0asm").expect("written");
        }
        let found = |reference: &str, directories: &[String]| {
            candidate_files(reference, &[beside.as_path()], directories)
        };

        assert_eq!(found("a.debug.wasm", &[]), [beside.join("a.debug.wasm")]);
        assert_eq!(
            found("sub/b.debug.wasm", &[]),
            [beside.join("sub/b.debug.wasm")]
        );
        assert_eq!(
            found("https://host/sub/a.debug.wasm?v=1#top", &[]),
            [beside.join("a.debug.wasm")],
            "a URL is only its file name"
        );
        let absolute = elsewhere.join("c.debug.wasm");
        assert!(
            found(&absolute.to_string_lossy(), &[]).is_empty(),
            "an absolute path is not followed"
        );
        assert!(
            found("../elsewhere/c.debug.wasm", &[]).is_empty(),
            "nor one that climbs out"
        );
        assert_eq!(
            found(
                "../elsewhere/c.debug.wasm",
                &[elsewhere.to_string_lossy().into_owned()]
            ),
            [absolute],
            "though its name is looked for in the debug directories"
        );
        assert!(found("sub/dir.wasm", &[]).is_empty(), "a directory");
        assert!(found("zero", &["/dev".to_owned()]).is_empty(), "a device");

        std::fs::remove_dir_all(&root).expect("removed");
    }

    #[test]
    fn a_module_without_dwarf_offers_nothing() {
        let image = wat::parse_str("(module (func nop))").expect("assembles");
        let module = module::parse(&image, 0).expect("parses");

        assert_eq!(section_of(&module, SectionId::DebugInfo.name()), None);
        assert!(!carries_dwarf(&module));
        assert!(code_base(&module).is_some(), "but it does have code");
    }

    /// A DWARF address is relative to the code section rather than the file
    #[test]
    fn addresses_are_relative_to_the_code_section() {
        let image = wat::parse_str("(module (func nop) (func nop))").expect("assembles");
        let module = module::parse(&image, 0).expect("parses");
        let base = code_base(&module).expect("a code section");

        assert!(base > 0, "the code section never starts at the file's head");
        for (_, info) in module.functions() {
            assert!(info.start >= base);
            assert!(body_at(&module, info.start).is_some());
            assert!(body_at(&module, info.start + 1).is_none());
        }
    }

    #[test]
    fn debug_information_describes_a_body_only_where_it_spans_it() {
        let image = wat::parse_str("(module (func nop nop) (func nop))").expect("assembles");
        let module = module::parse(&image, 0).expect("parses");
        let bodies: Vec<&FunctionInfo> = module.functions().map(|(_, info)| info).collect();
        let (first, second) = (bodies[0], bodies[1]);
        let size = first.end - first.start;

        let spanning = |high: Option<Written>| {
            let mut types = Types::new();
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            if let Some(high) = high {
                types.set(function, gimli::DW_AT_high_pc, high);
            }
            types.inspect(|tree| {
                let probe = Types::probe(tree);
                let entry = tree.entry(probe).expect("an entry");
                spans(tree, probe.unit, &entry, 0x10, first)
            })
        };
        let at = |address| Written::Address(gimli::write::Address::Constant(address));
        assert!(spanning(Some(Written::Udata(size))));
        assert!(
            spanning(Some(at(0x10 + size))),
            "an end given as an address"
        );
        assert!(
            !spanning(Some(Written::Udata(size - 1))),
            "the copy of an inline function a linker discarded, pointed at the one it kept"
        );
        assert!(!spanning(Some(at(0x10 + size + 1))));
        assert!(spanning(None), "no end says nothing against it");

        let rows = |address: u64| [(address, 1, 1)];
        assert!(sequence_body(&module, &rows(first.entry), first.end).is_some());
        assert!(
            sequence_body(&module, &rows(first.entry), first.end - 1).is_none(),
            "a discarded copy's lines, ending inside the body kept in its place"
        );
        assert!(sequence_body(&module, &rows(first.entry), second.end).is_none());
        assert!(sequence_body(&module, &[], first.end).is_none());
    }

    #[test]
    fn a_narrowed_global_is_named_only_when_its_flag_is_its_value() {
        let encoding = gimli::Encoding {
            format: gimli::Format::Dwarf32,
            version: 5,
            address_size: 4,
        };
        let after_deref = |bytes: &[u8]| {
            leaves_unchanged(
                gimli::Expression(Slice::new(bytes, LittleEndian)).operations(encoding),
            )
        };
        let (lit0, lit1, lit3, lit5) = (0x30, 0x31, 0x33, 0x35);
        let (mul, plus, plus_uconst, stack_value) = (0x1e, 0x22, 0x23, 0x9f);

        assert!(after_deref(&[lit1, mul, lit0, plus, stack_value]));
        assert!(after_deref(&[stack_value]));
        assert!(after_deref(&[plus_uconst, 0, stack_value]));

        assert!(
            !after_deref(&[lit5, mul, lit3, plus, stack_value]),
            "a flag for 3 or 8"
        );
        assert!(
            !after_deref(&[lit1, mul]),
            "a pointer to the value, not the value"
        );
        assert!(!after_deref(&[stack_value, lit0]));
        assert!(!after_deref(&[mul, stack_value]));
        assert!(!after_deref(&[lit1, stack_value]));
    }

    use gimli::write::{
        AttributeValue as Written, DebugInfoRef, EndianVec, LineProgram, Sections, UnitEntryId,
        UnitId,
    };

    struct Types {
        dwarf: gimli::write::Dwarf,
        unit: UnitId,
        signed: Vec<(u64, &'static str)>,
        fixed: bool,
        spilled: BTreeMap<i64, (usize, u64)>,
    }

    const ENCODING: gimli::Encoding = gimli::Encoding {
        format: gimli::Format::Dwarf32,
        version: 5,
        address_size: 4,
    };

    impl Types {
        fn new() -> Self {
            let mut dwarf = gimli::write::Dwarf::new();
            let unit = dwarf
                .units
                .add(gimli::write::Unit::new(ENCODING, LineProgram::none()));
            let mut types = Self {
                dwarf,
                unit,
                signed: Vec::new(),
                fixed: false,
                spilled: BTreeMap::new(),
            };
            let root = types.root();
            types.set(
                root,
                gimli::DW_AT_producer,
                Written::String("clang version 24.0.0".into()),
            );
            types
        }

        fn another(&mut self) -> UnitId {
            let root = self.root();
            let inherited: Vec<(gimli::DwAt, Written)> =
                [gimli::DW_AT_producer, gimli::DW_AT_language]
                    .into_iter()
                    .filter_map(|attr| Some((attr, self.unit().get(root).get(attr)?.clone())))
                    .collect();
            let previous = self.unit;
            self.unit = self
                .dwarf
                .units
                .add(gimli::write::Unit::new(ENCODING, LineProgram::none()));
            let root = self.root();
            for (attr, value) in inherited {
                self.set(root, attr, value);
            }
            previous
        }

        fn unit(&mut self) -> &mut gimli::write::Unit {
            self.dwarf.units.get_mut(self.unit)
        }

        fn root(&mut self) -> UnitEntryId {
            self.unit().root()
        }

        fn across(unit: UnitId, entry: UnitEntryId) -> Written {
            Written::DebugInfoRef(DebugInfoRef::Entry(unit, entry))
        }

        fn add(&mut self, parent: Option<UnitEntryId>, tag: gimli::DwTag) -> UnitEntryId {
            let parent = parent.unwrap_or_else(|| self.root());
            self.unit().add(parent, tag)
        }

        fn set(&mut self, entry: UnitEntryId, attr: gimli::DwAt, value: Written) -> UnitEntryId {
            self.unit().get_mut(entry).set(attr, value);
            entry
        }

        fn sign(&mut self, skeleton: UnitEntryId, signature: u64, definition: &'static str) {
            self.set(skeleton, gimli::DW_AT_declaration, Written::Flag(true));
            self.set(
                skeleton,
                gimli::DW_AT_signature,
                Written::DebugTypesRef(gimli::DebugTypeSignature(signature)),
            );
            self.signed.push((signature, definition));
        }

        fn base(&mut self, name: &str, size: u64, encoding: gimli::DwAte) -> UnitEntryId {
            let ty = self.add(None, gimli::DW_TAG_base_type);
            self.set(ty, gimli::DW_AT_name, Written::String(name.into()));
            self.set(ty, gimli::DW_AT_byte_size, Written::Udata(size));
            self.set(ty, gimli::DW_AT_encoding, Written::Encoding(encoding))
        }

        fn pointer(&mut self, to: UnitEntryId) -> UnitEntryId {
            let ty = self.add(None, gimli::DW_TAG_pointer_type);
            self.set(ty, gimli::DW_AT_type, Written::UnitRef(to))
        }

        fn record(&mut self, tag: gimli::DwTag, size: u64, members: &[UnitEntryId]) -> UnitEntryId {
            let record = self.add(None, tag);
            self.set(record, gimli::DW_AT_byte_size, Written::Udata(size));
            self.set(
                record,
                gimli::DW_AT_calling_convention,
                Written::CallingConvention(gimli::DW_CC_pass_by_value),
            );
            for (nth, ty) in members.iter().enumerate() {
                let member = self.add(Some(record), gimli::DW_TAG_member);
                self.set(
                    member,
                    gimli::DW_AT_name,
                    Written::String(format!("m{nth}").into()),
                );
                self.set(member, gimli::DW_AT_type, Written::UnitRef(*ty));
            }
            record
        }

        fn structure(&mut self, size: u64, members: &[UnitEntryId]) -> UnitEntryId {
            self.record(gimli::DW_TAG_structure_type, size, members)
        }

        fn laid_out(&mut self, size: u64, members: &[(UnitEntryId, u64)]) -> UnitEntryId {
            let types: Vec<UnitEntryId> = members.iter().map(|&(ty, _)| ty).collect();
            let record = self.structure(size, &types);
            let placed: Vec<UnitEntryId> = self.unit().get(record).children().copied().collect();
            for (member, &(_, offset)) in placed.into_iter().zip(members) {
                self.set(
                    member,
                    gimli::DW_AT_data_member_location,
                    Written::Udata(offset),
                );
            }
            record
        }

        fn array(&mut self, of: UnitEntryId, counts: &[Option<u64>]) -> UnitEntryId {
            let array = self.add(None, gimli::DW_TAG_array_type);
            self.set(array, gimli::DW_AT_type, Written::UnitRef(of));
            for count in counts {
                let range = self.add(Some(array), gimli::DW_TAG_subrange_type);
                if let Some(count) = count {
                    self.set(range, gimli::DW_AT_count, Written::Udata(*count));
                }
            }
            array
        }

        fn language(mut self, language: gimli::DwLang) -> Self {
            let root = self.root();
            self.set(root, gimli::DW_AT_language, Written::Language(language));
            self
        }

        fn named(&mut self, entry: UnitEntryId, name: &str) -> UnitEntryId {
            self.set(entry, gimli::DW_AT_name, Written::String(name.into()))
        }

        fn formal(
            &mut self,
            function: UnitEntryId,
            name: &str,
            ty: UnitEntryId,
            location: Option<Written>,
        ) -> UnitEntryId {
            let formal = self.add(Some(function), gimli::DW_TAG_formal_parameter);
            self.named(formal, name);
            self.set(formal, gimli::DW_AT_type, Written::UnitRef(ty));
            match location {
                Some(location) => self.set(formal, gimli::DW_AT_location, location),
                None => formal,
            }
        }

        fn in_local(local: u32) -> gimli::write::Expression {
            let mut expression = gimli::write::Expression::new();
            expression.op_wasm_local(local);
            expression.op(gimli::DW_OP_stack_value);
            expression
        }

        fn sections(mut self) -> Sections<EndianVec<LittleEndian>> {
            let mut sections = Sections::new(EndianVec::new(LittleEndian));
            self.dwarf.write(&mut sections).expect("writes");
            sections
        }

        fn load(sections: &Sections<EndianVec<LittleEndian>>) -> Dwarf<Slice<'_>> {
            Dwarf::load(|id| {
                Ok::<_, gimli::Error>(Slice::new(
                    sections.get(id).map_or(&[][..], |section| section.slice()),
                    LittleEndian,
                ))
            })
            .expect("loads")
        }

        fn inspect<T>(mut self, check: impl FnOnce(&Tree) -> T) -> T {
            let signed = std::mem::take(&mut self.signed);
            let sections = self.sections();
            let dwarf = Self::load(&sections);
            let mut tree = Tree::read(vec![&dwarf]);
            for (signature, definition) in signed {
                let definition = Self::find(&tree, definition);
                tree.signatures
                    .insert(gimli::DebugTypeSignature(signature), definition);
            }
            check(&tree)
        }

        fn search(tree: &Tree, name: &str) -> Option<Die> {
            for (unit, read) in tree.units.iter().enumerate() {
                let mut entries = read.entries();
                while let Ok(Some(entry)) = entries.next_dfs() {
                    let die = Die::of(unit, entry);
                    if !is_declaration(entry) && described(tree, die).0.as_deref() == Some(name) {
                        return Some(die);
                    }
                }
            }
            None
        }

        fn find(tree: &Tree, name: &str) -> Die {
            Self::search(tree, name).unwrap_or_else(|| panic!("no {name}"))
        }

        fn probe(tree: &Tree) -> Die {
            Self::find(tree, "probe")
        }

        fn type_of(tree: &Tree, die: Die) -> Option<Die> {
            tree.type_of(die.unit, &tree.entry(die).expect("an entry"))
        }

        fn describe(tree: &Tree, passed: Passed) -> String {
            let named = |at: Die| {
                let entry = tree.entry(at).expect("an entry");
                tree.name(at.unit, &entry)
                    .unwrap_or_else(|| entry.tag().to_string())
            };
            match passed {
                Passed::Omitted => "omitted".into(),
                Passed::Indirect(at) => format!("by pointer to {}", named(at)),
                Passed::Split(_) => "in halves".into(),
                Passed::Variadic => "the variadic arguments".into(),
                Passed::Unknown => "unknown".into(),
                Passed::Value(at) => format!("as {}", named(at)),
            }
        }

        fn passes(mut self, ty: UnitEntryId) -> String {
            let probe = self.add(None, gimli::DW_TAG_variable);
            self.named(probe, "probe");
            self.set(probe, gimli::DW_AT_type, Written::UnitRef(ty));
            self.inspect(
                |tree| match passed(tree, Self::type_of(tree, Self::probe(tree))) {
                    Passed::Indirect(_) => "by pointer".into(),
                    passed => Self::describe(tree, passed),
                },
            )
        }

        fn returns(self) -> String {
            self.inspect(|tree| {
                let probe = Self::probe(tree);
                Self::describe(tree, returned(tree, probe.unit, Self::type_of(tree, probe)))
            })
        }

        fn slots(self, params: usize, results: usize, entry_pc: Option<u64>) -> Vec<String> {
            let signature = Signature {
                params: vec![ValueKind::I32; params],
                results: vec![ValueKind::I32; results],
            };
            self.slots_of(&signature, entry_pc)
        }

        fn slots_of(self, signature: &Signature, entry_pc: Option<u64>) -> Vec<String> {
            let (fixed, spilled) = (self.fixed, self.spilled.clone());
            self.inspect(|tree| {
                let start = Start {
                    low_pc: entry_pc,
                    first: entry_pc,
                };
                parameter_slots(
                    tree,
                    Self::probe(tree),
                    signature,
                    &start,
                    internalised(tree),
                    fixed,
                    &spilled,
                )
                .0
                .into_iter()
                .map(|(name, passed)| {
                    format!(
                        "{}: {}",
                        name.as_deref().unwrap_or("?"),
                        Self::describe(tree, passed)
                    )
                })
                .collect()
            })
        }
    }

    #[test]
    fn a_parameter_is_placed_where_its_function_is_entered_holding_it() {
        let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_14);
        let long = types.base("long long", 8, gimli::DW_ATE_signed);
        let big = types.structure(16, &[long, long]);
        let maker = types.structure(1, &[]);
        let this = types.pointer(maker);
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let make = types.add(None, gimli::DW_TAG_subprogram);
        types.named(make, "probe");
        types.set(make, gimli::DW_AT_type, Written::UnitRef(big));
        types.formal(make, "this", this, None);
        types.formal(make, "x", int, Some(Written::Exprloc(Types::in_local(1))));
        assert_eq!(
            types.slots(2, 0, None),
            ["?: unknown", "x: as int"],
            "a method whose unused this was removed, leaving no slot for the returned record"
        );

        let mut types = Types::new().language(gimli::DW_LANG_C99);
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let pointer = types.pointer(int);
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.named(function, "probe");
        types.formal(
            function,
            "p",
            pointer,
            Some(Written::Exprloc(Types::in_local(1))),
        );
        types.formal(function, "q", int, None);
        assert_eq!(
            types.slots(2, 0, None),
            ["?: unknown", "p: as DW_TAG_pointer_type"],
            "a slot the first parameter was not entered in"
        );

        let mut types = Types::new().language(gimli::DW_LANG_C99);
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.named(function, "probe");
        types.formal(
            function,
            "a",
            int,
            Some(Written::Exprloc(Types::in_local(0))),
        );
        types.formal(function, "b", int, None);
        assert_eq!(types.slots(2, 0, None), ["a: as int", "b: as int"]);

        let mut types = Types::new().language(gimli::DW_LANG_Rust);
        let byte = types.base("u8", 1, gimli::DW_ATE_unsigned);
        let rgb = types.structure(3, &[byte, byte, byte]);
        let int = types.base("i32", 4, gimli::DW_ATE_signed);
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.named(function, "probe");
        types.formal(function, "c", rgb, None);
        types.formal(
            function,
            "n",
            int,
            Some(Written::Exprloc(Types::in_local(1))),
        );
        assert_eq!(
            types.slots(2, 0, None),
            ["?: unknown", "n: as i32"],
            "a language whose calls are not C's"
        );

        let scalars = |language: gimli::DwLang| {
            let mut types = Types::new().language(language);
            let int = types.base("integer", 4, gimli::DW_ATE_signed);
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            types.formal(function, "n", int, None);
            types.slots(1, 0, None)
        };
        assert_eq!(scalars(gimli::DW_LANG_Rust), ["n: as integer"]);
        assert_eq!(
            scalars(gimli::DW_LANG_Fortran95),
            ["?: unknown"],
            "one that may pass even a scalar by reference"
        );

        let located = |entry_pc: Option<u64>, ranges: &[(u64, u64, u32)]| {
            let mut types = Types::new().language(gimli::DW_LANG_C99);
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            types.set(
                function,
                gimli::DW_AT_calling_convention,
                Written::CallingConvention(gimli::DW_CC_nocall),
            );
            let list = ranges
                .iter()
                .map(|&(begin, end, local)| gimli::write::Location::StartEnd {
                    begin: gimli::write::Address::Constant(begin),
                    end: gimli::write::Address::Constant(end),
                    data: Types::in_local(local),
                })
                .collect();
            let list = types.unit().locations.add(gimli::write::LocationList(list));
            types.formal(function, "a", int, Some(Written::LocationListRef(list)));
            types.formal(function, "b", int, None);
            types.slots(3, 0, entry_pc)
        };
        assert_eq!(
            located(Some(0x10), &[(0x10, 0x20, 1), (0x20, 0x40, 0)]),
            ["?: unknown", "a: as int", "?: unknown"],
            "where the list has it at the entry"
        );
        assert_eq!(
            located(Some(0x10), &[(0x10, 0x10, 1)]),
            ["?: unknown", "a: as int", "?: unknown"],
            "even where the entry's range was left empty"
        );
        assert_eq!(
            located(Some(0x10), &[(0x16, 0x40, 1)]),
            ["?: unknown", "?: unknown", "?: unknown"],
            "a range starting later says where a value went, not which parameter it was"
        );
        assert_eq!(
            located(Some(0x10), &[(0x16, 0x16, 1)]),
            ["?: unknown", "?: unknown", "?: unknown"]
        );
        assert_eq!(
            located(None, &[(0x10, 0x20, 1)]),
            ["?: unknown", "?: unknown", "?: unknown"]
        );
    }

    #[test]
    fn a_flag_is_placed_where_the_local_it_is_masked_from_is() {
        let placed = |mask: u64| {
            let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_14);
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let pair = types.laid_out(8, &[(int, 0), (int, 4)]);
            let pointer = types.pointer(pair);
            let flag = types.base("bool", 1, gimli::DW_ATE_boolean);
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            types.set(
                function,
                gimli::DW_AT_calling_convention,
                Written::CallingConvention(gimli::DW_CC_nocall),
            );
            types.formal(function, "p", pointer, None);
            let mut masked = gimli::write::Expression::new();
            masked.op_wasm_local(1);
            masked.op_constu(mask);
            masked.op(gimli::DW_OP_and);
            masked.op(gimli::DW_OP_stack_value);
            types.formal(function, "on", flag, Some(Written::Exprloc(masked)));
            types.slots(2, 0, None)
        };
        assert_eq!(placed(1), ["?: unknown", "on: as bool"]);
        assert_eq!(
            placed(6),
            ["?: unknown", "?: unknown"],
            "a mask that keeps more than the low bits is a computation, not the local"
        );
    }

    #[test]
    fn a_parameter_optimised_away_is_one_that_had_nowhere_to_be() {
        let removed = |nocall: bool, formals: &[(&str, bool)], params: usize| {
            let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_14);
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let long = types.base("long long", 8, gimli::DW_ATE_signed);
            let big = types.structure(16, &[long, long]);
            types.named(big, "Big");
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            types.set(function, gimli::DW_AT_type, Written::UnitRef(big));
            if nocall {
                types.set(
                    function,
                    gimli::DW_AT_calling_convention,
                    Written::CallingConvention(gimli::DW_CC_nocall),
                );
            }
            for (nth, (name, located)) in formals.iter().enumerate() {
                let location = located.then(|| Written::Exprloc(Types::in_local(nth as u32)));
                types.formal(function, name, int, location);
            }
            types.slots(params, 0, None)
        };
        assert_eq!(
            removed(true, &[("unused", false), ("x", true)], 2),
            ["?: unknown", "x: as int"],
            "a result nothing writes can lose its slot too, so either could be the one missing"
        );
        assert_eq!(
            removed(true, &[("a", false), ("b", false), ("x", true)], 3),
            ["?: unknown", "?: unknown", "x: as int"]
        );
        let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_14);
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let long = types.base("long long", 8, gimli::DW_ATE_signed);
        let big = types.structure(16, &[long, long]);
        types.named(big, "Big");
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.named(function, "probe");
        types.set(function, gimli::DW_AT_type, Written::UnitRef(big));
        types.set(
            function,
            gimli::DW_AT_calling_convention,
            Written::CallingConvention(gimli::DW_CC_nocall),
        );
        types.formal(
            function,
            "x",
            int,
            Some(Written::Exprloc(Types::in_local(1))),
        );
        assert_eq!(
            types.slots(2, 0, None),
            ["result: by pointer to Big", "x: as int"],
            "though a slot only the result can fill is the result"
        );
        assert_eq!(
            removed(false, &[("unused", false), ("x", true)], 2),
            ["?: unknown", "x: as int"],
            "nothing says an external function's signature changed, so only x's location counts"
        );
    }

    #[test]
    fn variadic_arguments_arrive_as_one_last_parameter() {
        let variadic = |big: bool, params: usize| {
            let mut types = Types::new().language(gimli::DW_LANG_C11);
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            if big {
                let long = types.base("long long", 8, gimli::DW_ATE_signed);
                let big = types.structure(16, &[long, long]);
                types.named(big, "Big");
                types.set(function, gimli::DW_AT_type, Written::UnitRef(big));
            }
            types.formal(function, "n", int, None);
            types.add(Some(function), gimli::DW_TAG_unspecified_parameters);
            types.slots(params, 0, None)
        };
        assert_eq!(
            variadic(false, 2),
            ["n: as int", "?: the variadic arguments"]
        );
        assert_eq!(
            variadic(true, 3),
            [
                "result: by pointer to Big",
                "n: as int",
                "?: the variadic arguments"
            ]
        );
        assert_eq!(
            variadic(false, 3),
            ["?: unknown", "?: unknown", "?: unknown"],
            "one parameter more than the buffer accounts for"
        );
    }

    #[test]
    fn a_parameter_is_only_placed_on_a_value_of_its_kind() {
        let lanes = |params: &[ValueKind]| {
            let mut types = Types::new().language(gimli::DW_LANG_C11);
            let float = types.base("float", 4, gimli::DW_ATE_float);
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let vector = types.array(float, &[Some(2)]);
            types.set(vector, gimli::DW_AT_GNU_vector, Written::Flag(true));
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            types.formal(function, "v", vector, None);
            types.formal(function, "n", int, None);
            let signature = Signature {
                params: params.to_vec(),
                results: Vec::new(),
            };
            types.slots_of(&signature, None)
        };
        assert_eq!(
            lanes(&[ValueKind::V128, ValueKind::I32]),
            ["v: as DW_TAG_array_type", "n: as int"]
        );
        assert_eq!(
            lanes(&[ValueKind::F32, ValueKind::F32]),
            ["?: unknown", "?: unknown"],
            "without SIMD the vector is one value per lane, so n is no parameter at all"
        );

        let mut types = Types::new().language(gimli::DW_LANG_C11);
        let double = types.base("double", 8, gimli::DW_ATE_float);
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.named(function, "probe");
        types.formal(function, "d", double, None);
        let signature = Signature {
            params: vec![ValueKind::I64],
            results: Vec::new(),
        };
        assert_eq!(types.slots_of(&signature, None), ["?: unknown"]);

        let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_14);
        let double = types.base("double", 8, gimli::DW_ATE_float);
        let holder = types.structure(8, &[double]);
        let member = types.add(None, gimli::DW_TAG_ptr_to_member_type);
        types.set(member, gimli::DW_AT_type, Written::UnitRef(double));
        types.set(
            member,
            gimli::DW_AT_containing_type,
            Written::UnitRef(holder),
        );
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.named(function, "probe");
        types.formal(function, "m", member, None);
        assert_eq!(
            types.slots(1, 0, None),
            ["m: as DW_TAG_ptr_to_member_type"],
            "a pointer to a data member is an offset however wide the member"
        );
    }

    #[test]
    fn a_signature_the_optimiser_may_have_changed_unannounced_is_not_read_by_position() {
        let built =
            |producer: &str, marker: Option<gimli::DwAt>, external: bool, linkage: Option<&str>| {
                let mut types = Types::new().language(gimli::DW_LANG_C11);
                let root = types.root();
                types.set(
                    root,
                    gimli::DW_AT_producer,
                    Written::String(producer.into()),
                );
                let int = types.base("int", 4, gimli::DW_ATE_signed);
                let pointer = types.pointer(int);
                let function = types.add(None, gimli::DW_TAG_subprogram);
                types.named(function, "probe");
                if let Some(marker) = marker {
                    types.set(function, marker, Written::Flag(true));
                }
                if external {
                    types.set(function, gimli::DW_AT_external, Written::Flag(true));
                }
                if let Some(linkage) = linkage {
                    types.set(
                        function,
                        gimli::DW_AT_linkage_name,
                        Written::String(linkage.into()),
                    );
                }
                types.formal(function, "p", pointer, None);
                types.formal(
                    function,
                    "n",
                    int,
                    Some(Written::Exprloc(Types::in_local(1))),
                );
                types
            };
        let optimised = Some(gimli::DW_AT_APPLE_optimized);
        let probe = |producer: &str| built(producer, optimised, false, None).slots(2, 0, None);
        let named = ["p: as DW_TAG_pointer_type", "n: as int"];
        let unnamed = ["?: unknown", "n: as int"];
        assert_eq!(
            probe("clang version 24.0.0"),
            named,
            "a compiler that marks every signature it changes"
        );
        assert_eq!(
            probe("clang version 22.1.8"),
            unnamed,
            "one that promotes arguments without saying so"
        );
        assert_eq!(probe("Apple clang version 24.0.0"), unnamed);
        assert_eq!(
            built("clang version 22.1.8", None, false, None).slots(2, 0, None),
            named,
            "an unoptimised function keeps the signature it was written with"
        );
        assert_eq!(
            built(
                "clang version 22.1.8",
                Some(gimli::DW_AT_call_all_calls),
                false,
                None
            )
            .slots(2, 0, None),
            unnamed,
            "which a compiler tuned for GDB tells apart only by describing every call"
        );
        assert_eq!(
            built("clang LLVM (rustc version 1.98.1)", None, false, None).slots(2, 0, None),
            unnamed,
            "and one that never says whether it optimised cannot tell apart at all"
        );
        assert_eq!(
            built("clang version 22.1.8", optimised, true, None).slots(2, 0, None),
            named,
            "a function callers elsewhere may call keeps its signature"
        );
        let mut referenced = built("clang version 22.1.8", optimised, false, None);
        referenced.fixed = true;
        assert_eq!(
            referenced.slots(2, 0, None),
            named,
            "as does one whose address is taken"
        );
        let mut resumed = built(
            "clang version 24.0.0",
            optimised,
            true,
            Some("_Z5probePii.resume"),
        );
        resumed.fixed = true;
        assert_eq!(
            resumed.slots(2, 0, None),
            unnamed,
            "unless it is a copy the optimiser made, which may take something else entirely"
        );
    }

    #[test]
    fn a_c_union_may_be_passed_as_its_first_member() {
        let union = |language: gimli::DwLang| {
            let mut types = Types::new().language(language);
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let float = types.base("float", 4, gimli::DW_ATE_float);
            let passed = types.record(gimli::DW_TAG_union_type, 4, &[int, float]);
            types.named(passed, "U");
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            types.formal(function, "u", passed, None);
            types.slots(1, 0, None)
        };
        assert_eq!(union(gimli::DW_LANG_C_plus_plus_14), ["u: by pointer to U"]);
        assert_eq!(
            union(gimli::DW_LANG_C11),
            ["u: unknown"],
            "a transparent union is passed as its first member, and nothing says which one this is"
        );
    }

    #[test]
    fn a_rust_record_may_come_back_through_a_leading_pointer() {
        let returning = |leading: bool| {
            let mut types = Types::new().language(gimli::DW_LANG_Rust);
            let root = types.root();
            types.set(
                root,
                gimli::DW_AT_producer,
                Written::String("clang LLVM (rustc version 1.98.1)".into()),
            );
            let int = types.base("u32", 4, gimli::DW_ATE_unsigned);
            let rect = types.structure(16, &[int, int, int, int]);
            types.named(rect, "Rect");
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            types.set(function, gimli::DW_AT_type, Written::UnitRef(rect));
            if leading {
                types.formal(function, "scale", int, None);
            }
            types.formal(
                function,
                "x",
                int,
                Some(Written::Exprloc(Types::in_local(1))),
            );
            types.slots(2, 0, None)
        };
        assert_eq!(
            returning(false),
            ["result: by pointer to Rect", "x: as u32"]
        );
        assert_eq!(
            returning(true),
            ["?: unknown", "x: as u32"],
            "or through the slot of a parameter the optimiser removed"
        );
    }

    #[test]
    fn a_parameter_entered_in_pieces_is_placed_member_by_member() {
        let pieces = |narrow: bool, parts: &[(Option<u32>, u64)]| {
            let mut types = Types::new().language(gimli::DW_LANG_Rust);
            let root = types.root();
            types.set(
                root,
                gimli::DW_AT_producer,
                Written::String("clang LLVM (rustc version 1.98.1)".into()),
            );
            let byte = types.base("u8", 1, gimli::DW_ATE_unsigned);
            let length = types.base("usize", 4, gimli::DW_ATE_unsigned);
            let first = if narrow { byte } else { types.pointer(byte) };
            let pair = types.laid_out(8, &[(first, 0), (length, 4)]);
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            let mut location = gimli::write::Expression::new();
            for &(local, size) in parts {
                if let Some(local) = local {
                    location.op_wasm_local(local);
                    location.op(gimli::DW_OP_stack_value);
                }
                location.op_piece(size);
            }
            types.formal(function, "s", pair, Some(Written::Exprloc(location)));
            types.slots(3, 0, None)
        };
        let unplaced = ["?: unknown", "?: unknown", "?: unknown"];
        assert_eq!(
            pieces(false, &[(Some(1), 4), (Some(2), 4)]),
            [
                "?: unknown",
                "s.m0: as DW_TAG_pointer_type",
                "s.m1: as usize"
            ]
        );
        assert_eq!(
            pieces(true, &[(Some(1), 1), (None, 3), (Some(2), 4)]),
            ["?: unknown", "s.m0: as u8", "s.m1: as usize"],
            "padding between them is described as nowhere"
        );
        assert_eq!(
            pieces(false, &[(Some(1), 4), (Some(0), 4)]),
            unplaced,
            "pieces from locals out of order are not the parameters it arrived in"
        );
        let partly = ["?: unknown", "s.m0: as DW_TAG_pointer_type", "?: unknown"];
        assert_eq!(
            pieces(false, &[(Some(1), 4)]),
            partly,
            "a record only partly described places that part, and nothing is fitted around a \
             parameter that may take the next slot too"
        );
        assert_eq!(
            pieces(false, &[(Some(1), 4), (None, 4)]),
            partly,
            "as does a member whose value is nowhere"
        );
    }

    #[test]
    fn an_inlined_variable_names_a_frame_slot_only_where_nothing_else_is() {
        assert_eq!(
            unclashed(&[(0, 8, false), (4, 8, true), (8, 16, true), (16, 20, true)]),
            [true, false, true, true],
            "the function's own variable keeps a slot an inlined one overlaps"
        );
        assert_eq!(
            unclashed(&[(0, 8, true), (4, 12, true)]),
            [false, false],
            "two inlined variables in one slot leave it unnamed"
        );
        assert_eq!(
            unclashed(&[(0, 8, false), (4, 12, false)]),
            [false, false],
            "as do two of the function's own"
        );
    }

    #[test]
    fn a_member_left_out_of_a_pair_takes_the_slot_its_neighbours_leave_it() {
        let placed = |after: Option<u32>| {
            let mut types = Types::new().language(gimli::DW_LANG_Rust);
            let root = types.root();
            types.set(
                root,
                gimli::DW_AT_producer,
                Written::String("clang LLVM (rustc version 1.98.1)".into()),
            );
            let byte = types.base("u8", 1, gimli::DW_ATE_unsigned);
            let length = types.base("usize", 4, gimli::DW_ATE_unsigned);
            let pointer = types.pointer(byte);
            let pair = types.laid_out(8, &[(pointer, 0), (length, 4)]);
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            let mut location = gimli::write::Expression::new();
            location.op_wasm_local(0);
            location.op(gimli::DW_OP_stack_value);
            location.op_piece(4);
            types.formal(function, "s", pair, Some(Written::Exprloc(location)));
            let at = after.map(|local| Written::Exprloc(Types::in_local(local)));
            types.formal(function, "after", length, at);
            types.slots(3, 0, None)
        };

        assert_eq!(
            placed(Some(2)),
            [
                "s.m0: as DW_TAG_pointer_type",
                "s.m1: as usize",
                "after: as usize"
            ],
            "the next parameter one slot on leaves that slot to the rest of the pair"
        );
        assert_eq!(
            placed(Some(1)),
            [
                "s.m0: as DW_TAG_pointer_type",
                "after: as usize",
                "?: unknown"
            ],
            "the rest was removed where the next parameter follows at once"
        );
        assert_eq!(
            placed(None),
            ["s.m0: as DW_TAG_pointer_type", "?: unknown", "?: unknown"],
            "nothing says where the rest went without the next parameter placed"
        );
    }

    fn rust_function(types: &mut Types, rewritten: bool) -> UnitEntryId {
        let root = types.root();
        types.set(
            root,
            gimli::DW_AT_producer,
            Written::String("clang LLVM (rustc version 1.98.1)".into()),
        );
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.named(function, "probe");
        if rewritten {
            types.set(
                function,
                gimli::DW_AT_calling_convention,
                Written::CallingConvention(gimli::DW_CC_nocall),
            );
        }
        function
    }

    fn constant_location(value: u64) -> Written {
        let mut expression = gimli::write::Expression::new();
        expression.op_constu(value);
        expression.op(gimli::DW_OP_stack_value);
        Written::Exprloc(expression)
    }

    #[test]
    fn a_parameter_constant_at_entry_left_a_rewritten_signature() {
        let slots = |nocall: bool, external: bool, constant: Written, params: usize| {
            let mut types = Types::new().language(gimli::DW_LANG_Rust);
            let function = rust_function(&mut types, nocall);
            if external {
                types.set(function, gimli::DW_AT_external, Written::Flag(true));
            }
            let flag = types.base("bool", 1, gimli::DW_ATE_boolean);
            let byte = types.base("u8", 1, gimli::DW_ATE_unsigned);
            let character = types.base("char", 4, gimli::DW_ATE_UTF);
            let short = types.base("i16", 2, gimli::DW_ATE_signed);
            types.formal(
                function,
                "x",
                flag,
                Some(Written::Exprloc(Types::in_local(0))),
            );
            types.formal(function, "y", byte, None);
            let z = types.formal(function, "z", character, None);
            match constant {
                Written::Exprloc(_) => types.set(z, gimli::DW_AT_location, constant),
                _ => types.set(z, gimli::DW_AT_const_value, constant),
            };
            types.formal(function, "w", short, None);
            types.slots(params, 1, None)
        };

        let dropped = ["x: as bool", "y: as u8", "w: as i16"];
        assert_eq!(slots(true, false, Written::Udata(120), 3), dropped);
        assert_eq!(
            slots(true, false, constant_location(120), 3),
            dropped,
            "a location that is the constant says the same"
        );
        assert_eq!(
            slots(false, false, Written::Udata(120), 3),
            ["x: as bool", "?: unknown", "?: unknown"],
            "a signature only perhaps rewritten may have lost any of them"
        );
        assert_eq!(
            slots(false, true, Written::Udata(120), 4),
            ["x: as bool", "y: as u8", "z: as char", "w: as i16"],
            "a signature nothing rewrote still passes it"
        );
    }

    #[test]
    fn a_constant_half_of_a_pair_takes_no_slot() {
        let mut types = Types::new().language(gimli::DW_LANG_Rust);
        let function = rust_function(&mut types, true);
        let code = types.base("u32", 4, gimli::DW_ATE_unsigned);
        let byte = types.base("u8", 1, gimli::DW_ATE_unsigned);
        let length = types.base("usize", 4, gimli::DW_ATE_unsigned);
        let pointer = types.pointer(byte);
        let slice = types.laid_out(8, &[(pointer, 0), (length, 4)]);
        types.formal(
            function,
            "code",
            code,
            Some(Written::Exprloc(Types::in_local(0))),
        );
        let mut location = gimli::write::Expression::new();
        location.op_wasm_local(1);
        location.op(gimli::DW_OP_stack_value);
        location.op_piece(4);
        location.op_constu(4);
        location.op(gimli::DW_OP_stack_value);
        location.op_piece(4);
        types.formal(function, "dst", slice, Some(Written::Exprloc(location)));

        assert_eq!(
            types.slots(2, 1, None),
            ["code: as u32", "dst.m0: as DW_TAG_pointer_type"]
        );
    }

    #[test]
    fn a_promoted_record_fills_slots_its_fields_could() {
        let slots = |params: Vec<ValueKind>| {
            let mut types = Types::new().language(gimli::DW_LANG_Rust);
            let function = rust_function(&mut types, true);
            let long = types.base("u64", 8, gimli::DW_ATE_unsigned);
            let rect = types.laid_out(32, &[(long, 0), (long, 8), (long, 16), (long, 24)]);
            types.named(rect, "Rect");
            types.set(function, gimli::DW_AT_type, Written::UnitRef(rect));
            types.formal(function, "r", rect, None);
            let by = types.formal(function, "by", long, None);
            types.set(by, gimli::DW_AT_location, constant_location(5));
            let signature = Signature {
                params,
                results: Vec::new(),
            };
            types.slots_of(&signature, None)
        };

        assert_eq!(
            slots(vec![ValueKind::I32, ValueKind::I32]),
            ["result: by pointer to Rect", "r: by pointer to Rect"],
            "two addresses are the result and the record, since its fields are all wider"
        );
        assert_eq!(
            slots(vec![ValueKind::I32, ValueKind::I64, ValueKind::I64]),
            ["result: by pointer to Rect", "?: unknown", "?: unknown"],
            "fields it may have been promoted to are left unnamed"
        );
    }

    #[test]
    fn the_end_of_the_parameters_bounds_the_rest_of_a_pair() {
        let slots = |returned: bool| {
            let mut types = Types::new().language(gimli::DW_LANG_Rust);
            let function = rust_function(&mut types, true);
            let byte = types.base("u8", 1, gimli::DW_ATE_unsigned);
            let length = types.base("usize", 4, gimli::DW_ATE_unsigned);
            let pointer = types.pointer(byte);
            let slice = types.laid_out(8, &[(pointer, 0), (length, 4)]);
            if returned {
                let list = types.structure(12, &[pointer, length, length]);
                types.set(function, gimli::DW_AT_type, Written::UnitRef(list));
            }
            let level = types.formal(function, "level", byte, None);
            types.set(level, gimli::DW_AT_location, constant_location(2));
            let mut location = gimli::write::Expression::new();
            location.op_wasm_local(1);
            location.op(gimli::DW_OP_stack_value);
            location.op_piece(4);
            types.formal(function, "items", slice, Some(Written::Exprloc(location)));
            types.slots(3, 0, None)
        };

        assert_eq!(
            slots(true),
            [
                "?: unknown",
                "items.m0: as DW_TAG_pointer_type",
                "items.m1: as usize"
            ],
            "the result's pointer accounts for the first slot"
        );
        assert_eq!(
            slots(false),
            [
                "?: unknown",
                "items.m0: as DW_TAG_pointer_type",
                "?: unknown"
            ],
            "an unexplained first slot could as well be the one after it"
        );
    }

    #[test]
    fn a_wide_integer_described_by_one_half_still_takes_both() {
        let described = |parts: &[(Option<u32>, u64)]| {
            let mut types = Types::new().language(gimli::DW_LANG_C11);
            let wide = types.base("__int128", 16, gimli::DW_ATE_signed);
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let long = types.base("long long", 8, gimli::DW_ATE_signed);
            let record = types.laid_out(24, &[(long, 0), (long, 8), (int, 16)]);
            let pointer = types.pointer(record);
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            types.set(
                function,
                gimli::DW_AT_calling_convention,
                Written::CallingConvention(gimli::DW_CC_nocall),
            );
            let mut location = gimli::write::Expression::new();
            for &(local, size) in parts {
                if let Some(local) = local {
                    location.op_wasm_local(local);
                    location.op(gimli::DW_OP_stack_value);
                }
                location.op_piece(size);
            }
            types.formal(function, "w", wide, Some(Written::Exprloc(location)));
            types.formal(function, "after", pointer, None);
            let signature = Signature {
                params: vec![ValueKind::I64, ValueKind::I64, ValueKind::I32],
                results: vec![ValueKind::I32],
            };
            types.slots_of(&signature, None)
        };
        let halves = ["w_low: unknown", "w_high: unknown", "?: unknown"];
        assert_eq!(described(&[(Some(0), 8)]), halves);
        assert_eq!(described(&[(None, 8), (Some(1), 8)]), halves);
        assert_eq!(
            described(&[]),
            ["?: unknown", "?: unknown", "?: unknown"],
            "without either half, a pointer that may have become anything leaves it open"
        );
    }

    #[test]
    fn a_frame_is_the_stack_pointer_only_where_the_prologue_stores_it() {
        let frame = |body: &str, holder: u32| {
            let image = wat::parse_str(format!(
                "(module (global $sp (mut i32) (i32.const 1024)) (func (local i32 i32) {body}))"
            ))
            .expect("assembles");
            let module = module::parse(&image, 0).expect("parses");
            let (_, info) = module.functions().next().expect("a function");
            prologue_frame(body_code(&module, &image, info).expect("mapped"), 0, holder)
        };
        let taken = "global.get $sp i32.const 16 i32.sub local.tee 0 global.set $sp";

        assert!(frame(taken, 0));
        assert!(
            frame(
                "global.get $sp i32.const 16 i32.sub local.set 0 local.get 0 global.set $sp",
                0
            ),
            "stored from the local"
        );
        assert!(!frame(taken, 1), "another local");
        assert!(
            !frame(
                "global.get $sp i32.const 16 i32.sub local.tee 0 i32.const -64 i32.and \
                 global.set $sp",
                0
            ),
            "a frame realigned below the one the local keeps"
        );
        assert!(
            !frame("global.get $sp i32.const 16 i32.sub local.set 0", 0),
            "a pointer never moved"
        );
    }

    #[test]
    fn the_locals_are_renumbered_where_the_code_disagrees_with_the_dwarf() {
        let renumbering =
            |text: &str, rewritten: bool, describe: &dyn Fn(&mut Types, UnitEntryId)| {
                let mut image = wat::parse_str(text).expect("assembles");
                if rewritten {
                    image.extend([0, 12, 11]);
                    image.extend(b".debug_info");
                }
                let module = module::parse(&image, 0).expect("parses");
                let base = code_base(&module).expect("a code section");
                let (_, body) = module.functions().next().expect("a body");
                let mut types = Types::new();
                let function = types.add(None, gimli::DW_TAG_subprogram);
                let (low, high) = (body.start - base, body.end - base);
                types.set(
                    function,
                    gimli::DW_AT_low_pc,
                    Written::Address(gimli::write::Address::Constant(low)),
                );
                types.set(function, gimli::DW_AT_high_pc, Written::Udata(high - low));
                describe(&mut types, function);
                types.inspect(|tree| renumbered(tree, &module, &image, base))
            };
        let expression = |local: u32| {
            let mut expression = gimli::write::Expression::new();
            expression.op_wasm_local(local);
            expression.op(gimli::DW_OP_stack_value);
            expression
        };

        let sets = "(module (func $f (param i32) (result i32) (local i32 i32) \
             (local.set 1 (i32.const 5)) (local.set 2 (i32.const 7)) \
             (i32.add (local.get 1) (local.get 2))))";
        let module = module::parse(&wat::parse_str(sets).expect("assembles"), 0).expect("parses");
        let (_, body) = module.functions().next().expect("a body");
        let image = wat::parse_str(sets).expect("assembles");
        let base = code_base(&module).expect("a code section");
        let after: HashMap<u32, u64> = locals_set(
            body_code(&module, &image, body).expect("mapped"),
            body.entry - base,
        )
        .into_iter()
        .map(|(address, local)| (local, address))
        .collect();
        let end = body.end - base;
        let placed = |claims: &'static [(u32, u32)]| {
            let after = after.clone();
            move |types: &mut Types, function: UnitEntryId| {
                for (set, claimed) in claims {
                    let list = types.unit().locations.add(gimli::write::LocationList(vec![
                        gimli::write::Location::StartEnd {
                            begin: gimli::write::Address::Constant(after[set]),
                            end: gimli::write::Address::Constant(end),
                            data: expression(*claimed),
                        },
                    ]));
                    let variable = types.add(Some(function), gimli::DW_TAG_variable);
                    types.set(
                        variable,
                        gimli::DW_AT_location,
                        Written::LocationListRef(list),
                    );
                }
            }
        };

        assert!(!renumbering(sets, false, &placed(&[(1, 1), (2, 2)])));
        assert!(
            renumbering(sets, false, &placed(&[(1, 2), (2, 1)])),
            "two locals swapped"
        );
        assert!(
            renumbering(sets, false, &placed(&[(1, 1), (2, 7)])),
            "a local the function does not have"
        );
        assert!(
            !renumbering(sets, false, &placed(&[])),
            "nothing to tell, in a module as its linker wrote it"
        );
        assert!(
            renumbering(sets, true, &placed(&[])),
            "nothing to tell, in a module written again after its DWARF"
        );
        assert!(
            !renumbering(sets, true, &placed(&[(1, 1), (2, 2)])),
            "a module written again with its locals where they were"
        );

        let framed = "(module (global $__stack_pointer (mut i32) (i32.const 4096)) \
             (func $f (local i32 i32) \
               global.get 0 i32.const 16 i32.sub local.tee 0 global.set 0 \
               (local.set 1 (i32.const 3)) \
               local.get 0 i32.const 16 i32.add global.set 0))";
        let based = |local: u32| {
            move |types: &mut Types, function: UnitEntryId| {
                types.set(
                    function,
                    gimli::DW_AT_frame_base,
                    Written::Exprloc(expression(local)),
                );
            }
        };
        assert!(!renumbering(framed, false, &based(0)));
        assert!(
            renumbering(framed, false, &based(1)),
            "a frame base in a local the prologue does not store"
        );
        let copied = "(module (global $__stack_pointer (mut i32) (i32.const 4096)) \
             (func $f (local i32 i32) \
               global.get 0 i32.const 16 i32.sub local.tee 0 global.set 0 \
               local.get 0 local.set 1 \
               local.get 1 i32.const 16 i32.add global.set 0))";
        assert!(
            !renumbering(copied, false, &based(1)),
            "a frame base copied from the local the prologue stores, as a function with a \
             variable length array keeps it"
        );
    }

    type Placed = (&'static str, bool, Vec<(Option<Range<usize>>, u32)>);

    fn assigned_in(
        body: &str,
        renumbered: bool,
        block: Option<Range<usize>>,
        variables: &[Placed],
    ) -> Vec<(usize, Vec<String>)> {
        let image = wat::parse_str(format!("(module (func {body}))")).expect("assembles");
        let module = module::parse(&image, 0).expect("parses");
        let base = code_base(&module).expect("a code section");
        let (_, info) = module.functions().next().expect("a body");
        let code = body_code(&module, &image, info).expect("mapped");
        let at = info.entry - base;
        let mut starts = Vec::new();
        let mut offset = 0;
        while let Some(insn) = code.get(offset..).and_then(insn::decode_any) {
            starts.push(at + offset as u64);
            offset += insn.len;
        }
        starts.push(at + offset as u64);
        let address = |nth: usize| Written::Address(gimli::write::Address::Constant(starts[nth]));

        let mut types = Types::new();
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.named(function, "f");
        let (low, high) = (info.start - base, info.end - base);
        types.set(
            function,
            gimli::DW_AT_low_pc,
            Written::Address(gimli::write::Address::Constant(low)),
        );
        types.set(function, gimli::DW_AT_high_pc, Written::Udata(high - low));
        let lexical = block.map(|block| {
            let lexical = types.add(Some(function), gimli::DW_TAG_lexical_block);
            types.set(lexical, gimli::DW_AT_low_pc, address(block.start));
            let size = starts[block.end] - starts[block.start];
            types.set(lexical, gimli::DW_AT_high_pc, Written::Udata(size))
        });
        for (name, inside, places) in variables {
            let parent = if *inside { lexical } else { Some(function) };
            let variable = types.add(parent, gimli::DW_TAG_variable);
            types.named(variable, name);
            let location = match places[..] {
                [(None, local)] => Written::Exprloc(Types::in_local(local)),
                _ => {
                    let entries = places.iter().map(|(range, local)| {
                        let range = range.clone().expect("a range");
                        gimli::write::Location::StartEnd {
                            begin: gimli::write::Address::Constant(starts[range.start]),
                            end: gimli::write::Address::Constant(starts[range.end]),
                            data: Types::in_local(*local),
                        }
                    });
                    let list = gimli::write::LocationList(entries.collect());
                    Written::LocationListRef(types.unit().locations.add(list))
                }
            };
            types.set(variable, gimli::DW_AT_location, location);
        }

        let params = info.signature.params.len() as u32;
        types.inspect(|tree| {
            let (held, scopes) = held_variables(tree, Types::find(tree, "f"), low..high);
            assignments(&module, code, at, params, &held, &scopes, renumbered)
                .into_iter()
                .map(|(address, (_, candidates))| {
                    let nth = starts.iter().position(|start| *start == address);
                    let names = candidates
                        .iter()
                        .map(|(nth, _)| described(tree, held[*nth].die).0.expect("a name"))
                        .collect();
                    (nth.expect("an instruction"), names)
                })
                .collect()
        })
    }

    #[test]
    fn a_variable_is_assigned_where_its_range_starts_after_its_local_is_set() {
        let body = "(param i32) (local i32 i32) \
             i32.const 1 local.set 1 \
             i32.const 2 local.set 2 \
             local.get 1 local.get 2 i32.add local.set 1 \
             block local.get 0 local.set 2 end \
             local.get 2 drop";
        let variables: Vec<Placed> = vec![
            ("a", false, vec![(Some(2..4), 1)]),
            ("b", false, vec![(Some(4..8), 2)]),
            ("delayed", false, vec![(Some(4..8), 1)]),
            ("merged", false, vec![(Some(12..14), 1)]),
            ("scoped", true, vec![(None, 2)]),
        ];
        let named = |names: &[&str]| names.iter().map(|name| name.to_string()).collect();

        assert_eq!(
            assigned_in(body, false, Some(8..12), &variables),
            [
                (1, named(&["a", "delayed"])),
                (3, named(&["b"])),
                (10, named(&["scoped"]))
            ],
            "a range right after the set first, one starting later in the same straight run \
             next, none past a block's end, and a single location only within its scope"
        );
    }

    #[test]
    fn a_renumbered_local_is_matched_only_where_two_places_agree() {
        let body = "(param i32) (local i32 i32 i32) \
             i32.const 1 local.set 1 local.get 0 drop \
             i32.const 2 local.set 1 local.get 1 drop \
             i32.const 3 local.set 2 local.get 2 drop \
             i32.const 4 local.set 3 i32.const 5 local.set 2 local.get 3 drop";
        let twice = ("twice", false, vec![(Some(2..4), 7), (Some(6..8), 7)]);
        let named = |names: &[&str]| names.iter().map(|name| name.to_string()).collect();

        assert_eq!(
            assigned_in(
                body,
                true,
                None,
                &[
                    twice.clone(),
                    ("once", false, vec![(Some(10..12), 8)]),
                    ("reached", false, vec![(Some(10..11), 9), (Some(11..12), 9)]),
                    (
                        "crowded",
                        false,
                        vec![(Some(16..18), 10), (Some(17..18), 10)]
                    ),
                ]
            ),
            [
                (1, named(&["twice"])),
                (5, named(&["twice"])),
                (9, named(&["reached"]))
            ],
            "one set alone is no match, a second range the same local reaches is, and a set \
             whose value is worked out right after another set's is none"
        );
        assert_eq!(
            assigned_in(
                body,
                true,
                None,
                &[
                    twice,
                    ("alongside", false, vec![(Some(2..4), 11), (Some(6..8), 11)])
                ]
            ),
            [],
            "two old locals live at once cannot have become one"
        );
    }

    #[test]
    fn a_local_copies_the_frame_only_where_it_is_first_set_from_it() {
        let copies_frame = |body: &str| {
            let image = wat::parse_str(format!("(module (func (local i32 i32) {body}))"))
                .expect("assembles");
            let module = module::parse(&image, 0).expect("parses");
            let (_, info) = module.functions().next().expect("a function");
            copies(body_code(&module, &image, info).expect("mapped"), 0, 1)
        };
        let frame = "(local.set 0 (i32.const 64))";

        assert!(copies_frame(&format!(
            "{frame} (local.set 1 (local.get 0))"
        )));
        assert!(
            !copies_frame(&format!(
                "{frame} (local.set 0 (i32.const 8)) (local.set 1 (local.get 0))"
            )),
            "after the frame local is written again"
        );
        assert!(
            !copies_frame(&format!(
                "{frame} (local.set 1 (i32.const 3)) (local.set 1 (local.get 0))"
            )),
            "first set from something else"
        );
        assert!(
            !copies_frame("(local.set 1 (local.get 0))"),
            "before the frame local holds anything"
        );
    }

    #[test]
    fn a_parameter_is_found_where_the_prologue_spills_it() {
        let image = wat::parse_str(
            r#"(module
                (global $sp (mut i32) (i32.const 1024))
                (func (param i32 i32 i64 i32) (local i32)
                    global.get $sp i32.const 64 i32.sub local.tee 4 global.set $sp
                    local.get 4 local.get 0 i32.store offset=8
                    local.get 4 local.get 1 i32.store8 offset=12
                    local.get 4 local.get 2 i64.store offset=16
                    local.get 4 local.get 1 i32.store offset=24
                    local.get 4 local.get 0 i32.store offset=24
                    i32.const 5 local.set 3
                    local.get 4 local.get 3 i32.store offset=32
                    local.get 4 local.get 2 i32.wrap_i64 i32.store offset=36
                    local.get 2 local.get 0 i32.store offset=40
                    block end
                    local.get 4 local.get 1 i32.store offset=48))"#,
        )
        .expect("assembles");
        let module = module::parse(&image, 0).expect("parses");
        let (_, info) = module.functions().next().expect("a function");
        let (from, to) = (
            module.layout.file_offset(info.entry).expect("mapped") as usize,
            module.layout.file_offset(info.end).expect("mapped") as usize,
        );
        let found = spills(&image[from..to], 4, &info.signature.params);
        assert_eq!(
            found.into_iter().collect::<Vec<_>>(),
            [(8, (0, 4)), (12, (1, 1)), (16, (2, 8))],
            "not a slot written twice, a parameter already overwritten, a value computed from \
             one, a store through anything but the frame, or anything past the first block"
        );

        let spilled = |spilled: &[(i64, (usize, u64))]| {
            let mut types = Types::new().language(gimli::DW_LANG_Rust);
            let root = types.root();
            types.set(
                root,
                gimli::DW_AT_producer,
                Written::String("clang LLVM (rustc version 1.98.1)".into()),
            );
            let usize_type = types.base("usize", 4, gimli::DW_ATE_unsigned);
            let flag = types.base("bool", 1, gimli::DW_ATE_boolean);
            let layout = types.laid_out(8, &[(usize_type, 0), (usize_type, 4)]);
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            for (name, ty, at) in [
                ("cap", usize_type, 44),
                ("table_layout", layout, 48),
                ("on", flag, 56),
            ] {
                let mut location = gimli::write::Expression::new();
                location.op_fbreg(at);
                types.formal(function, name, ty, Some(Written::Exprloc(location)));
            }
            types.spilled = spilled.iter().copied().collect();
            types.slots(4, 0, None)
        };
        assert_eq!(
            spilled(&[(44, (0, 4)), (48, (1, 4)), (52, (2, 4)), (56, (3, 1))]),
            [
                "cap: as usize",
                "table_layout.m0: as usize",
                "table_layout.m1: as usize",
                "on: as bool"
            ]
        );
        let wide: BTreeMap<i64, (usize, u64)> = [(56, (3, 4))].into_iter().collect();
        assert!(
            from_frame(&wide, 56, 1).is_none(),
            "a spill wider than the parameter is not it"
        );
        let apart: BTreeMap<i64, (usize, u64)> = [(48, (1, 4)), (52, (3, 4))].into_iter().collect();
        assert!(
            from_frame(&apart, 48, 8).is_none(),
            "nor are halves spilled from parameters that are not neighbours"
        );
    }

    #[test]
    fn a_cplusplus_record_is_only_classified_where_it_says_how_it_is_passed() {
        let record = |says: bool| {
            let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_14);
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let holder = types.structure(4, &[int]);
            if !says {
                types
                    .unit()
                    .get_mut(holder)
                    .delete(gimli::DW_AT_calling_convention);
            }
            types.passes(holder)
        };
        assert_eq!(record(true), "as int");
        assert_eq!(
            record(false),
            "unknown",
            "since only the attribute tells a trivially copyable one from one passed by reference"
        );
    }

    #[test]
    fn a_float_declared_without_a_prototype_arrives_as_a_double() {
        let probe = |prototyped: bool, kind: ValueKind| {
            let mut types = Types::new().language(gimli::DW_LANG_C11);
            let float = types.base("float", 4, gimli::DW_ATE_float);
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            if prototyped {
                types.set(function, gimli::DW_AT_prototyped, Written::Flag(true));
            }
            types.formal(function, "b", float, None);
            let signature = Signature {
                params: vec![kind],
                results: Vec::new(),
            };
            types.slots_of(&signature, None)
        };
        assert_eq!(probe(false, ValueKind::F64), ["b: as float"]);
        assert_eq!(probe(true, ValueKind::F64), ["?: unknown"]);
        assert_eq!(
            probe(false, ValueKind::F32),
            ["b: as float"],
            "a producer that never says a function is prototyped"
        );
    }

    #[test]
    fn a_record_known_only_by_name_is_returned_through_the_one_parameter_left_over() {
        let returning = |declared: bool, params: usize| {
            let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_14);
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            if declared {
                let text = types.add(None, gimli::DW_TAG_structure_type);
                types.named(text, "Text");
                types.set(text, gimli::DW_AT_declaration, Written::Flag(true));
                types.set(function, gimli::DW_AT_type, Written::UnitRef(text));
            }
            types.formal(function, "x", int, None);
            types.slots(params, 0, None)
        };
        assert_eq!(
            returning(true, 2),
            ["result: by pointer to Text", "x: as int"]
        );
        assert_eq!(returning(true, 1), ["x: as int"]);
        assert_eq!(
            returning(false, 2),
            ["?: unknown", "?: unknown"],
            "a function returning nothing has no result to be given room for"
        );
    }

    #[test]
    fn a_type_that_contains_itself_is_classified_in_bounded_time() {
        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let array = types.array(int, &[Some(1)]);
        types.set(array, gimli::DW_AT_type, Written::UnitRef(array));
        let holder = types.structure(4, &[array]);
        assert_eq!(types.passes(holder), "unknown");

        let mut types = Types::new();
        let record = types.structure(12, &[]);
        for nth in 0..4 {
            let member = types.add(Some(record), gimli::DW_TAG_member);
            types.named(member, &format!("m{nth}"));
            types.set(member, gimli::DW_AT_type, Written::UnitRef(record));
        }
        assert_eq!(types.passes(record), "unknown");
    }

    #[test]
    fn a_language_whose_calls_are_not_cs_is_only_trusted_with_a_scalar_result() {
        let mut types = Types::new().language(gimli::DW_LANG_Rust);
        let int = types.base("i32", 4, gimli::DW_ATE_signed);
        let wrapper = types.structure(4, &[int]);
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.named(function, "probe");
        types.set(function, gimli::DW_AT_type, Written::UnitRef(wrapper));
        assert_eq!(types.returns(), "unknown");

        let mut types = Types::new().language(gimli::DW_LANG_Rust);
        let int = types.base("i32", 4, gimli::DW_ATE_signed);
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.named(function, "probe");
        types.set(function, gimli::DW_AT_type, Written::UnitRef(int));
        assert_eq!(types.returns(), "as i32");
    }

    #[test]
    fn a_record_is_passed_the_way_the_webassembly_c_abi_passes_it() {
        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        assert_eq!(types.passes(int), "as int");

        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let pointer = types.pointer(int);
        let wrapper = types.structure(4, &[pointer]);
        assert_eq!(
            types.passes(wrapper),
            "as DW_TAG_pointer_type",
            "a pointer wrapper"
        );

        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let position = types.structure(8, &[int, int]);
        let location = types.structure(16, &[position, position]);
        assert_eq!(types.passes(location), "by pointer");

        let mut types = Types::new();
        let empty = types.structure(1, &[]);
        assert_eq!(types.passes(empty), "omitted");

        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let pointer = types.pointer(int);
        let owner = types.structure(4, &[pointer]);
        types.set(
            owner,
            gimli::DW_AT_calling_convention,
            Written::CallingConvention(gimli::DW_CC_pass_by_reference),
        );
        assert_eq!(types.passes(owner), "by pointer", "not trivially copyable");

        let mut types = Types::new();
        let float = types.base("float", 4, gimli::DW_ATE_float);
        let inner = types.structure(4, &[float]);
        let outer = types.structure(4, &[inner]);
        let empty = types.structure(1, &[]);
        let base = types.add(Some(outer), gimli::DW_TAG_inheritance);
        types.set(base, gimli::DW_AT_type, Written::UnitRef(empty));
        assert_eq!(
            types.passes(outer),
            "as float",
            "nested, past an empty base"
        );

        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let one = types.array(int, &[Some(1)]);
        let array = types.structure(4, &[one]);
        assert_eq!(types.passes(array), "as int", "a one element array");

        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let one = types.array(int, &[Some(1), Some(1)]);
        let array = types.structure(4, &[one]);
        assert_eq!(
            types.passes(array),
            "as int",
            "one element in each dimension"
        );

        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let none = types.array(int, &[Some(0)]);
        let trailing = types.structure(4, &[int, none]);
        assert_eq!(types.passes(trailing), "as int", "a zero length array");

        let mut types = Types::new();
        let float = types.base("float", 4, gimli::DW_ATE_float);
        let vector = types.array(float, &[Some(4)]);
        types.set(vector, gimli::DW_AT_GNU_vector, Written::Flag(true));
        let wrapped = types.structure(16, &[vector]);
        assert_eq!(
            types.passes(wrapped),
            "as DW_TAG_array_type",
            "a vector is one value"
        );

        let mut types = Types::new();
        let wide = types.base("__int128", 16, gimli::DW_ATE_signed);
        assert_eq!(types.passes(wide), "in halves");
        let mut types = Types::new();
        let wide = types.base("long double", 16, gimli::DW_ATE_float);
        let wrapped = types.structure(16, &[wide]);
        assert_eq!(types.passes(wrapped), "in halves", "inside a record");

        let mut types = Types::new();
        let complex = types.base("complex", 8, gimli::DW_ATE_complex_float);
        assert_eq!(types.passes(complex), "by pointer", "a complex number");

        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let tagged = types.structure(4, &[int]);
        let shared = types.add(Some(tagged), gimli::DW_TAG_member);
        types.set(shared, gimli::DW_AT_type, Written::UnitRef(int));
        types.set(shared, gimli::DW_AT_declaration, Written::Flag(true));
        let padding = types.add(Some(tagged), gimli::DW_TAG_member);
        types.set(padding, gimli::DW_AT_type, Written::UnitRef(int));
        types.set(padding, gimli::DW_AT_bit_size, Written::Udata(0));
        let alias = types.add(None, gimli::DW_TAG_typedef);
        types.set(alias, gimli::DW_AT_type, Written::UnitRef(tagged));
        assert_eq!(
            types.passes(alias),
            "as int",
            "past a typedef, a static member and an unnamed bitfield"
        );
    }

    #[test]
    fn a_record_that_is_more_than_its_one_value_is_passed_by_pointer() {
        let mut types = Types::new();
        let byte = types.base("char", 1, gimli::DW_ATE_signed_char);
        let padded = types.structure(4, &[byte]);
        assert_eq!(
            types.passes(padded),
            "by pointer",
            "padding past the element"
        );

        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let empty = types.structure(1, &[]);
        let beside = types.structure(8, &[empty, int]);
        assert_eq!(
            types.passes(beside),
            "by pointer",
            "an empty member still takes room"
        );

        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let empty = types.structure(1, &[]);
        let overlapping = types.structure(4, &[empty, int]);
        assert_eq!(
            types.passes(overlapping),
            "as int",
            "an empty member that takes no room"
        );

        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let float = types.base("float", 4, gimli::DW_ATE_float);
        let either = types.record(gimli::DW_TAG_union_type, 4, &[int, float]);
        assert_eq!(types.passes(either), "by pointer");

        let mut types = Types::new();
        let declared = types.add(None, gimli::DW_TAG_structure_type);
        types.set(declared, gimli::DW_AT_declaration, Written::Flag(true));
        assert_eq!(types.passes(declared), "unknown", "nothing to classify");

        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let flexible = types.array(int, &[None]);
        let trailing = types.structure(4, &[int, flexible]);
        assert_eq!(types.passes(trailing), "by pointer", "a flexible array");

        let mut types = Types::new();
        let complex = types.base("complex", 8, gimli::DW_ATE_complex_float);
        let wrapped = types.structure(8, &[complex]);
        assert_eq!(types.passes(wrapped), "by pointer", "a complex member");

        let mut types = Types::new();
        let float = types.base("float", 4, gimli::DW_ATE_float);
        let pair = types.array(float, &[Some(2)]);
        let wrapped = types.structure(8, &[pair]);
        assert_eq!(types.passes(wrapped), "by pointer", "a two element array");

        let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_14);
        let empty = types.structure(1, &[]);
        let holder = types.structure(1, &[empty]);
        assert_eq!(
            types.passes(holder),
            "by pointer",
            "an empty member is still a member in C++"
        );
        let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_14);
        let empty = types.structure(1, &[]);
        assert_eq!(
            types.passes(empty),
            "omitted",
            "though an empty record is not"
        );

        let no_unique_address = |value_at: u64, size: u64| {
            let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_20);
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let empty = types.structure(1, &[]);
            let holder = types.laid_out(size, &[(empty, 0), (int, value_at)]);
            types.passes(holder)
        };
        assert_eq!(
            no_unique_address(0, 4),
            "as int",
            "an empty member sharing the value's storage is [[no_unique_address]]"
        );
        assert_eq!(
            no_unique_address(4, 8),
            "by pointer",
            "one with storage of its own is a member"
        );

        let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_20);
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let empty = types.structure(1, &[]);
        let either = types.record(gimli::DW_TAG_union_type, 4, &[empty, int]);
        assert_eq!(
            types.passes(either),
            "by pointer",
            "every member of a union shares its storage"
        );

        let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_20);
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let empty = types.structure(1, &[]);
        let bits = types.structure(4, &[empty, int]);
        let field = types.unit().get(bits).children().nth(1).copied();
        let field = field.expect("the bitfield");
        types.set(field, gimli::DW_AT_bit_size, Written::Udata(3));
        types.set(field, gimli::DW_AT_data_bit_offset, Written::Udata(8));
        assert_eq!(
            types.passes(bits),
            "by pointer",
            "a bitfield past the empty member"
        );

        let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_20);
        let empty = types.structure(1, &[]);
        let holder = types.structure(1, &[empty]);
        let method = types.add(Some(holder), gimli::DW_TAG_subprogram);
        types.named(method, "get");
        assert_eq!(
            types.passes(holder),
            "by pointer",
            "a method takes no storage for the member to share"
        );

        let mut types = Types::new();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let holder = types.structure(4, &[int]);
        let untyped = types.add(Some(holder), gimli::DW_TAG_member);
        types.named(untyped, "elsewhere");
        assert_eq!(
            types.passes(holder),
            "unknown",
            "a member whose type is not given"
        );
    }

    #[test]
    fn a_parameter_is_only_trusted_where_nothing_contradicts_it() {
        let mut types = Types::new().language(gimli::DW_LANG_C99);
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let pair = types.structure(8, &[int, int]);
        types.named(pair, "pair");
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.named(function, "probe");
        types.formal(
            function,
            "p",
            pair,
            Some(Written::Exprloc(Types::in_local(0))),
        );
        assert_eq!(
            types.slots(1, 0, None),
            ["p: as pair"],
            "entered holding the value itself rather than its address"
        );

        let method_built = |optimised: bool, external: bool, internalised: bool| {
            let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_14);
            if internalised {
                let rewritten = types.add(None, gimli::DW_TAG_subprogram);
                types.named(rewritten, "elsewhere");
                types.set(rewritten, gimli::DW_AT_external, Written::Flag(true));
                types.set(
                    rewritten,
                    gimli::DW_AT_calling_convention,
                    Written::CallingConvention(gimli::DW_CC_nocall),
                );
            }
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let holder = types.structure(4, &[int]);
            types.named(holder, "Holder");
            let this = types.pointer(holder);
            let declaration = types.add(Some(holder), gimli::DW_TAG_subprogram);
            types.named(declaration, "val");
            types.set(declaration, gimli::DW_AT_declaration, Written::Flag(true));
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            types.set(
                function,
                gimli::DW_AT_specification,
                Written::UnitRef(declaration),
            );
            if optimised {
                types.set(function, gimli::DW_AT_APPLE_optimized, Written::Flag(true));
            }
            if external {
                types.set(function, gimli::DW_AT_external, Written::Flag(true));
            }
            types.formal(function, "this", this, None);
            types.formal(
                function,
                "x",
                int,
                Some(Written::Exprloc(Types::in_local(1))),
            );
            types.slots(2, 0, None)
        };
        assert_eq!(
            method_built(true, false, false),
            ["?: unknown", "x: as int"],
            "an optimised method's pointer with nowhere to be may be a field it was replaced by"
        );
        assert_eq!(
            method_built(false, false, false),
            ["this: as DW_TAG_pointer_type", "x: as int"],
            "an unoptimised method keeps the signature it was written with"
        );
        assert_eq!(
            method_built(true, true, false),
            ["this: as DW_TAG_pointer_type", "x: as int"],
            "so does one callers elsewhere may call"
        );
        assert_eq!(
            method_built(true, true, true),
            ["?: unknown", "x: as int"],
            "unless another function seen from outside had its signature changed, which only \
             happens once the whole program is optimised together"
        );

        let spilled = |kept: bool| {
            let mut types = Types::new().language(gimli::DW_LANG_C11);
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let pointer = types.pointer(int);
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            types.set(
                function,
                gimli::DW_AT_calling_convention,
                Written::CallingConvention(gimli::DW_CC_nocall),
            );
            let mut frame = gimli::write::Expression::new();
            frame.op_fbreg(8);
            types.formal(
                function,
                "p",
                pointer,
                kept.then_some(Written::Exprloc(frame)),
            );
            types.formal(
                function,
                "x",
                int,
                Some(Written::Exprloc(Types::in_local(1))),
            );
            types.slots(2, 0, None)
        };
        assert_eq!(
            spilled(true),
            ["p: as DW_TAG_pointer_type", "x: as int"],
            "a pointer kept in the frame was never replaced by what it points at"
        );
        assert_eq!(spilled(false), ["?: unknown", "x: as int"]);

        let mut types = Types::new().language(gimli::DW_LANG_Rust);
        let int = types.base("i32", 4, gimli::DW_ATE_signed);
        let byte = types.base("u8", 1, gimli::DW_ATE_unsigned);
        let pointer = types.pointer(byte);
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.named(function, "probe");
        types.formal(function, "a", int, None);
        types.formal(function, "p", pointer, None);
        assert_eq!(
            types.slots(2, 0, None),
            ["a: as i32", "p: as DW_TAG_pointer_type"],
            "scalars pass as themselves whatever the language"
        );
    }

    #[test]
    fn what_another_unit_declares_is_read_from_there() {
        let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_14);
        let long = types.base("long long", 8, gimli::DW_ATE_signed);
        let big = types.structure(16, &[long, long]);
        types.named(big, "Big");
        let flag = types.base("bool", 1, gimli::DW_ATE_boolean);
        let maker = types.structure(1, &[]);
        let this = types.pointer(maker);
        let declaration = types.add(Some(maker), gimli::DW_TAG_subprogram);
        types.named(declaration, "probe");
        types.set(declaration, gimli::DW_AT_declaration, Written::Flag(true));
        types.set(declaration, gimli::DW_AT_type, Written::UnitRef(big));
        let first = types.another();
        let function = types.add(None, gimli::DW_TAG_subprogram);
        types.set(
            function,
            gimli::DW_AT_specification,
            Types::across(first, declaration),
        );
        for (local, name, ty) in [(1, "this", this), (2, "flag", flag)] {
            let formal = types.add(Some(function), gimli::DW_TAG_formal_parameter);
            types.named(formal, name);
            types.set(formal, gimli::DW_AT_type, Types::across(first, ty));
            types.set(
                formal,
                gimli::DW_AT_location,
                Written::Exprloc(Types::in_local(local)),
            );
        }
        assert_eq!(
            types.slots(3, 0, None),
            [
                "result: by pointer to Big",
                "this: as DW_TAG_pointer_type",
                "flag: as bool"
            ]
        );
    }

    #[test]
    fn a_type_kept_in_a_type_unit_is_found_by_its_signature() {
        let mut types = Types::new().language(gimli::DW_LANG_C_plus_plus_14);
        let space = types.add(None, gimli::DW_TAG_namespace);
        types.named(space, "ns");
        let skeleton = types.add(Some(space), gimli::DW_TAG_structure_type);
        types.sign(skeleton, 0x5eed, "S");
        let declaration = types.add(Some(skeleton), gimli::DW_TAG_subprogram);
        types.named(declaration, "get");
        types.set(declaration, gimli::DW_AT_declaration, Written::Flag(true));
        let definition = types.add(None, gimli::DW_TAG_subprogram);
        types.set(
            definition,
            gimli::DW_AT_specification,
            Written::UnitRef(declaration),
        );
        let probe = types.add(None, gimli::DW_TAG_variable);
        types.named(probe, "probe");
        types.set(probe, gimli::DW_AT_type, Written::UnitRef(skeleton));
        types.another();
        let int = types.base("int", 4, gimli::DW_ATE_signed);
        let defined = types.structure(4, &[int]);
        types.named(defined, "S");

        let (method, probe) = types.inspect(|tree| {
            let get = Types::find(tree, "get");
            let declared = origin(tree, get).expect("a declaration");
            let probe = Types::probe(tree);
            (
                tree.qualified(declared, "get"),
                Types::describe(tree, passed(tree, Types::type_of(tree, probe))),
            )
        });
        assert_eq!(
            method, "ns::S::get",
            "a skeleton is named by its definition"
        );
        assert_eq!(probe, "as int", "and stands for it");
    }

    #[test]
    fn a_split_unit_is_read_only_with_the_skeleton_it_belongs_to() {
        let unit = |id: u64, function: Option<&str>| {
            let mut types = Types::new().language(gimli::DW_LANG_C11);
            let root = types.root();
            types.set(root, gimli::DW_AT_GNU_dwo_id, Written::Data8(id));
            if let Some(function) = function {
                let function_entry = types.add(None, gimli::DW_TAG_subprogram);
                types.named(function_entry, function);
            }
            types.sections()
        };
        let files = [
            unit(7, None),
            unit(9, Some("stale")),
            unit(7, Some("kept")),
            unit(7, Some("again")),
        ];
        let dwarfs: Vec<Dwarf<Slice>> = files.iter().map(Types::load).collect();
        let tree = Tree::read(dwarfs.iter().collect());
        let kept = Types::find(&tree, "kept");
        assert_eq!(tree.lines_of(kept.unit), 0, "its lines are the skeleton's");
        assert!(
            Types::search(&tree, "stale").is_none(),
            "one whose skeleton is not in the module has addresses from somewhere else"
        );
        assert!(
            Types::search(&tree, "again").is_none(),
            "and a skeleton takes one unit"
        );
    }

    #[test]
    fn a_split_file_is_read_from_its_custom_sections() {
        let unit = |function: Option<&str>| {
            let mut types = Types::new().language(gimli::DW_LANG_C11);
            let root = types.root();
            types.set(root, gimli::DW_AT_GNU_dwo_id, Written::Data8(7));
            if let Some(function) = function {
                let function_entry = types.add(None, gimli::DW_TAG_subprogram);
                types.named(function_entry, function);
            }
            types.sections()
        };
        let mut custom = String::new();
        unit(Some("kept"))
            .for_each(|id, data| {
                if let Some(name) = id.dwo_name().filter(|_| !data.slice().is_empty()) {
                    let bytes: String = data.slice().iter().map(|b| format!("\\{b:02x}")).collect();
                    custom.push_str(&format!(r#"(@custom "{name}" "{bytes}")"#));
                }
                Ok::<_, gimli::Error>(())
            })
            .expect("written");
        let image = wat::parse_str(format!("(module {custom})")).expect("assembles");

        let skeleton = unit(None);
        let main = Types::load(&skeleton);
        let images = [image];
        let splits = split_dwarfs(&images, &main);
        let tree = Tree::read(std::iter::once(&main).chain(&splits).collect());
        assert_eq!(tree.lines_of(Types::find(&tree, "kept").unit), 0);
    }

    #[test]
    fn a_function_whose_declaration_cannot_be_read_may_return_anything() {
        let unread = |params: usize| {
            let mut types = Types::new().language(gimli::DW_LANG_C11);
            let int = types.base("int", 4, gimli::DW_ATE_signed);
            let function = types.add(None, gimli::DW_TAG_subprogram);
            types.named(function, "probe");
            types.set(
                function,
                gimli::DW_AT_specification,
                Written::DebugInfoRefSup(gimli::DebugInfoOffset(0)),
            );
            types.formal(function, "a", int, None);
            types.formal(function, "b", int, None);
            types.slots(params, 0, None)
        };
        assert_eq!(
            unread(2),
            ["?: unknown", "?: unknown"],
            "a record returned through a leading pointer would leave room for only one of them"
        );
        assert_eq!(unread(3), ["result: unknown", "a: as int", "b: as int"]);
    }

    #[test]
    fn a_tombstoned_address_is_recognised_at_either_width() {
        assert!(is_tombstone(0xffff_ffff, 4));
        assert!(is_tombstone(0xffff_fffe, 4));
        assert!(is_tombstone(u64::MAX, 8));
        assert!(is_tombstone(u64::MAX - 1, 8));

        assert!(!is_tombstone(0, 4));
        assert!(!is_tombstone(0xffff_fffd, 4));
        assert!(!is_tombstone(0xffff_ffff, 8), "a real address at wasm64");
    }
}
