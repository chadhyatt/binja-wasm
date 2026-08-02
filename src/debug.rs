//! Importing the DWARF a compiler left in the module's custom sections
//!
//! Binary Ninja's own DWARF plugin claims a `.wasm` and then fails on it: it ignores the view's
//! sections, re-reads the raw file, and hands it to the `object` crate, which is built without its
//! `wasm` feature; reading the custom sections directly is all `gimli` ever needed
//!
//! Two things decide whether any of it lines up: an address is relative to the code section
//! payload, and `DW_AT_low_pc` points at a body's locals declaration rather than its first
//! instruction

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;

use binaryninja::binary_view::{BinaryView, BinaryViewBase, BinaryViewExt};
use binaryninja::confidence::{Conf, MAX_CONFIDENCE};
use binaryninja::debuginfo::{
    CustomDebugInfoParser, DebugFunctionInfo, DebugInfo, DebugInfoParser,
};
use binaryninja::function::Function;
use binaryninja::rc::Ref;
use binaryninja::types::{
    BaseStructure, EnumerationBuilder, FunctionParameter, MemberAccess, MemberScope,
    NamedTypeReference, NamedTypeReferenceClass, StructureBuilder, StructureType, Type, TypeClass,
};
use binaryninja::variable::{NamedVariableWithType, Variable, VariableSourceType};
use gimli::{AttributeValue, Dwarf, EndianSlice, LittleEndian, Reader, SectionId};

use crate::module::{self, FunctionInfo, Module, Signature, ValueKind};
use crate::{lift, settings, view};

pub const NAME: &str = "WASM DWARF";

type Slice<'a> = EndianSlice<'a, LittleEndian>;
type Entry<'a, 'u> = gimli::DebuggingInformationEntry<'a, 'u, Slice<'a>>;

struct WasmDwarf;

impl CustomDebugInfoParser for WasmDwarf {
    fn is_valid(&self, view: &BinaryView) -> bool {
        // Only claim a module that carries DWARF, so an ordinary build is untouched
        read(view).is_some_and(|(_, modules)| {
            modules
                .iter()
                .any(|module| section_of(module, SectionId::DebugInfo).is_some())
        })
    }

    fn parse_info(
        &self,
        debug_info: &mut DebugInfo,
        view: &BinaryView,
        _debug_file: &BinaryView,
        progress: Box<dyn Fn(usize, usize) -> Result<(), ()>>,
    ) -> bool {
        let Some((image, modules)) = read(view) else {
            return false;
        };

        let mut tally = Tally::default();
        let mut registered = HashSet::new();
        let mut placed = HashSet::new();
        let mut seen = 0usize;
        let sources = settings::source_lines();

        // An address is relative to the code section payload of the module carrying it, so each
        // nested module is loaded against its own
        for module in &modules {
            let Some(base) = code_base(module) else {
                continue;
            };

            let mut imports: HashMap<String, u32> = HashMap::new();
            for (index, import) in module.imports() {
                imports.insert(import.field.clone(), index);
                imports.insert(module.name_of(index), index);
            }

            let bodies: Vec<(u64, u64, u64)> = module
                .functions()
                .map(|(_, info)| (info.start, info.end, info.entry))
                .collect();

            let load = |id: SectionId| -> Result<Slice, gimli::Error> {
                // Section spans are addresses; `image` is indexed by file offset
                let bytes = section_of(module, id)
                    .and_then(|(start, end)| {
                        let start = module.layout.file_offset(start)? as usize;
                        let end = module.layout.file_offset(end)? as usize;
                        image.get(start..end)
                    })
                    .unwrap_or(&[]);
                Ok(EndianSlice::new(bytes, LittleEndian))
            };

            let Ok(dwarf) = Dwarf::load(load) else {
                continue;
            };

            let mut units = dwarf.units();
            // `next` stops on the first malformed header rather than reporting it, since a
            // truncated section is a reason to keep what was read
            while let Ok(Some(header)) = units.next() {
                seen += 1;
                if progress(seen, seen + 1).is_err() {
                    break;
                }

                let Ok(unit) = dwarf.unit(header) else {
                    continue;
                };
                let mut import = Import {
                    dwarf: &dwarf,
                    unit: &unit,
                    scopes: Scopes::of(&dwarf, &unit),
                    module,
                    imports: &imports,
                    bodies: &bodies,
                    base,
                    view,
                    debug_info,
                    registered: &mut registered,
                    placed: &mut placed,
                    tally: &mut tally,
                };
                import.walk();
                if sources {
                    import.lines();
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
        if tally.unnamed != 0 || tally.unplaced != 0 {
            tracing::debug!(
                "wasm DWARF: {} subprograms with no name anywhere and {} whose address names \
                 no body",
                tally.unnamed,
                tally.unplaced,
            );
        }
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
}

struct Import<'a, 'b> {
    dwarf: &'a Dwarf<Slice<'a>>,
    unit: &'a gimli::Unit<Slice<'a>>,
    scopes: Scopes,
    module: &'a Module,
    imports: &'a HashMap<String, u32>,
    bodies: &'a [(u64, u64, u64)],
    base: u64,
    view: &'a BinaryView,
    debug_info: &'b DebugInfo,
    registered: &'b mut HashSet<String>,
    placed: &'b mut HashSet<u64>,
    tally: &'b mut Tally,
}

impl Import<'_, '_> {
    fn walk(&mut self) {
        let unit = self.unit;
        let mut entries = unit.entries();
        while let Ok(Some((_, entry))) = entries.next_dfs() {
            match entry.tag() {
                gimli::DW_TAG_subprogram => self.subprogram(entry),
                gimli::DW_TAG_variable => self.variable(entry),
                gimli::DW_TAG_structure_type
                | gimli::DW_TAG_class_type
                | gimli::DW_TAG_union_type
                | gimli::DW_TAG_enumeration_type
                | gimli::DW_TAG_typedef
                    if !is_declaration(entry) && has_name(entry) =>
                {
                    let offset = entry.offset();
                    self.build_type(offset, 0);
                }
                _ => {}
            }
        }
    }

    fn subprogram(&mut self, entry: &Entry) {
        let Some(low_pc) = self.address(entry, gimli::DW_AT_low_pc) else {
            self.stub(entry);
            return;
        };

        let Some((index, body)) = body_at(self.module, self.base.wrapping_add(low_pc)) else {
            self.tally.unplaced += 1;
            return;
        };
        if !self.placed.insert(body.entry) {
            return;
        }

        let declaration = self.origin(entry).unwrap_or_else(|| entry.offset());
        let raw = self
            .linkage_at(entry.offset())
            .or_else(|| self.linkage_at(declaration));
        let Some(name) = self.name_at(declaration).or_else(|| raw.clone()) else {
            self.tally.unnamed += 1;
            return;
        };
        let name = self.scopes.qualified(declaration.0, &name);
        let raw = raw.unwrap_or_else(|| name.clone());

        let prototype = self.prototype(entry, declaration, index, &body.signature);
        let params = body.signature.params.len() as u32;
        let (slots, frame) = self.locals(entry, params);
        self.tally.locals += slots.len();

        let info = DebugFunctionInfo::new(
            Some(name.clone()),
            Some(name),
            Some(raw),
            prototype,
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

    fn stub(&mut self, entry: &Entry) -> Option<()> {
        if !is_declaration(entry) {
            return None;
        }
        let name = name_of(self.dwarf, self.unit, entry)?;
        let index = *self.imports.get(name.as_str())?;
        let import = self.module.import(index)?;
        let address = self.module.layout.import_address(index);
        if !self.placed.insert(address) {
            return None;
        }

        let prototype = self.prototype(entry, entry.offset(), index, &import.signature)?;
        let stub = self.view.functions_at(address).iter().next()?.to_owned();
        stub.set_user_type(&prototype);
        stub.set_user_pure(Conf::new(false, MAX_CONFIDENCE));
        self.tally.stubs += 1;
        Some(())
    }

    fn variable(&mut self, entry: &Entry) {
        let size = self.unit.encoding().address_size;
        let Some(described) = location(entry, size) else {
            return;
        };
        let at = match described {
            Where::Memory(at) if self.module.layout.memory_mapped(at) => {
                self.module.layout.memory_address(at)
            }
            Where::Global(index) if index < self.module.globals().count() as u32 => {
                self.module.layout.global_address(index)
            }
            _ => return,
        };

        let declaration = self.origin(entry).unwrap_or_else(|| entry.offset());
        let Some(name) = self.name_at(declaration) else {
            return;
        };
        let Some(ty) = self.type_at(declaration, 0) else {
            return;
        };

        let name = self.scopes.qualified(declaration.0, &name);
        self.debug_info.add_data_variable(at, &ty, Some(&name), &[]);
        self.tally.variables += 1;
    }

    fn locals(&mut self, entry: &Entry, params: u32) -> (Vec<NamedVariableWithType>, Vec<String>) {
        let unit = self.unit;
        let size = unit.encoding().address_size;
        let (mut slots, mut frame) = (Vec::new(), Vec::new());

        let Ok(mut cursor) = unit.entries_at_offset(entry.offset()) else {
            return (slots, frame);
        };
        if cursor.next_dfs().is_err() {
            return (slots, frame);
        }

        let mut level = 0isize;
        let mut inlined = isize::MAX;
        while let Ok(Some((delta, child))) = cursor.next_dfs() {
            level += delta;
            if level <= 0 {
                break;
            }
            if level <= inlined {
                inlined = isize::MAX;
            }
            if child.tag() == gimli::DW_TAG_inlined_subroutine {
                inlined = level;
                continue;
            }
            if level > inlined
                || !matches!(
                    child.tag(),
                    gimli::DW_TAG_variable | gimli::DW_TAG_formal_parameter
                )
            {
                continue;
            }

            let (Some(name), Some(at)) = (name_of(self.dwarf, unit, child), location(child, size))
            else {
                continue;
            };
            let offset = child.offset();
            match at {
                Where::Local(index) if index >= params => {
                    let Some(ty) = self.type_at(offset, 0) else {
                        continue;
                    };
                    slots.push(NamedVariableWithType::new(
                        Variable::new(
                            VariableSourceType::StackVariableSourceType,
                            0,
                            lift::local_slot(params, index),
                        ),
                        Conf::new(ty, MAX_CONFIDENCE),
                        name,
                        false,
                    ));
                }
                Where::Frame(at) => {
                    let ty = self.type_at(offset, 0);
                    let ty = ty.map_or_else(|| "?".to_string(), |ty| ty.to_string());
                    let sign = if at < 0 { "-" } else { "+" };
                    frame.push(format!("{sign}{:#x} {name} ({ty})", at.unsigned_abs()));
                }
                _ => {}
            }
        }
        (slots, frame)
    }

    fn annotate(&self, entry: u64, declaration: gimli::UnitOffset, frame: Vec<String>) {
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

    fn lines(&mut self) {
        let Some(program) = self.unit.line_program.clone() else {
            return;
        };
        let mut names: HashMap<u64, String> = HashMap::new();
        let mut current: Option<(u64, u64, Ref<Function>)> = None;
        let mut last: Option<(u64, u64)> = None;
        let mut rows = program.rows();

        while let Ok(Some((header, row))) = rows.next_row() {
            if row.end_sequence() {
                last = None;
                continue;
            }
            let (Some(line), true) = (row.line(), row.is_stmt()) else {
                continue;
            };
            let at = (row.file_index(), line.get());
            if last == Some(at) {
                continue;
            }
            last = Some(at);

            let address = self.base.wrapping_add(row.address());
            if !current
                .as_ref()
                .is_some_and(|(start, end, _)| (*start..*end).contains(&address))
            {
                current = self
                    .body_covering(address)
                    .and_then(|(start, end, entry)| Some((start, end, self.function_at(entry)?)));
            }
            let Some((_, _, function)) = &current else {
                continue;
            };

            let name = names.entry(at.0).or_insert_with(|| {
                header
                    .file(at.0)
                    .and_then(|file| self.dwarf.attr_string(self.unit, file.path_name()).ok())
                    .and_then(|raw| std::str::from_utf8(raw.slice()).ok().map(module::clean))
                    .unwrap_or_default()
            });
            function.set_comment_at(address, &format!("{name}:{}", at.1));
            self.tally.lines += 1;
        }
    }

    fn body_covering(&self, address: u64) -> Option<(u64, u64, u64)> {
        let at = self
            .bodies
            .partition_point(|(start, _, _)| *start <= address)
            .checked_sub(1)?;
        let (start, end, entry) = self.bodies[at];
        (address < end).then_some((start, end, entry))
    }

    fn function_at(&self, entry: u64) -> Option<Ref<Function>> {
        self.view
            .functions_at(entry)
            .iter()
            .next()
            .map(|f| f.clone())
    }

    fn declared_at(&self, declaration: gimli::UnitOffset) -> Option<String> {
        let entry = self.unit.entry(declaration).ok()?;
        let index = udata(&entry, gimli::DW_AT_decl_file)?;
        let line = udata(&entry, gimli::DW_AT_decl_line)?;
        let file = self.unit.line_program.as_ref()?.header().file(index)?;
        let raw = self.dwarf.attr_string(self.unit, file.path_name()).ok()?;
        let name = module::clean(std::str::from_utf8(raw.slice()).ok()?);
        (!name.is_empty()).then(|| format!("{name}:{line}"))
    }

    fn prototype(
        &mut self,
        entry: &Entry,
        declaration: gimli::UnitOffset,
        index: u32,
        signature: &Signature,
    ) -> Option<Ref<Type>> {
        let mut described = self.parameters(entry);
        if self.returns_aggregate(declaration)
            && signature.results.is_empty()
            && signature.params.len() == described.len() + 1
        {
            described.insert(0, (None, None));
        }
        let described = (described.len() == signature.params.len()).then_some(described);

        let mut parameters = Vec::with_capacity(signature.params.len());
        for (nth, kind) in signature.params.iter().enumerate() {
            let (name, ty) = match &described {
                Some(described) => described[nth].clone(),
                None => (None, None),
            };
            let name = name
                .or_else(|| self.module.local_name(index, nth as u32).map(str::to_owned))
                .unwrap_or_else(|| format!("arg{nth}"));
            let ty = self
                .slot_wide(ty, *kind)
                .unwrap_or_else(|| view::parameter_type(*kind));
            let at = view::parameter_slot(nth as u32);
            parameters.push(FunctionParameter::new(ty, name, Some(at)));
        }

        let pointer = self.module.layout.pointer;
        let returns = match signature.results.as_slice() {
            [] => Type::void(),
            [only] => self
                .type_at(declaration, 0)
                .filter(|ty| ty.width() as usize == only.size())
                .unwrap_or_else(|| view::value_type(*only, pointer)),
            _ => return None,
        };

        Some(Type::function(returns.as_ref(), parameters, false))
    }

    fn refers_to(&self, offset: gimli::UnitOffset) -> Option<gimli::DwTag> {
        let entry = self.unit.entry(offset).ok()?;
        let at = self.stripped(type_ref(&entry))?;
        Some(self.unit.entry(at).ok()?.tag())
    }

    fn points_at_code(&self, offset: gimli::UnitOffset) -> bool {
        self.refers_to(offset) == Some(gimli::DW_TAG_subroutine_type)
    }

    fn returns_aggregate(&self, declaration: gimli::UnitOffset) -> bool {
        self.refers_to(declaration).is_some_and(is_aggregate)
    }

    fn slot_wide(
        &mut self,
        offset: Option<gimli::UnitOffset>,
        kind: ValueKind,
    ) -> Option<Ref<Type>> {
        let offset = offset?;
        let at = self.stripped(Some(offset))?;
        if is_aggregate(self.unit.entry(at).ok()?.tag()) {
            return None;
        }

        let float = matches!(kind, ValueKind::F32 | ValueKind::F64);
        self.build_type(offset, 0).filter(|ty| {
            ty.width() == crate::lift::SLOT
                && float == (ty.type_class() == TypeClass::FloatTypeClass)
        })
    }

    fn parameters(&self, entry: &Entry) -> Vec<(Option<String>, Option<gimli::UnitOffset>)> {
        let unit = self.unit;
        let mut found = Vec::new();
        let Ok(mut cursor) = unit.entries_at_offset(entry.offset()) else {
            return found;
        };
        if cursor.next_dfs().is_err() {
            return found;
        }

        let mut level = 0isize;
        while let Ok(Some((delta, child))) = cursor.next_dfs() {
            level += delta;
            if level <= 0 {
                break;
            }
            if level > 1 || child.tag() != gimli::DW_TAG_formal_parameter {
                continue;
            }
            found.push((name_of(self.dwarf, unit, child), type_ref(child)));
        }
        found
    }

    fn build_type(&mut self, offset: gimli::UnitOffset, depth: usize) -> Option<Ref<Type>> {
        if depth >= MAX_TYPE_DEPTH {
            return None;
        }
        let entry = self.unit.entry(offset).ok()?;
        let pointer = self.module.layout.pointer;

        match entry.tag() {
            gimli::DW_TAG_base_type => base_type(&entry),
            gimli::DW_TAG_pointer_type
            | gimli::DW_TAG_reference_type
            | gimli::DW_TAG_rvalue_reference_type => {
                if self.points_at_code(offset) {
                    return Some(Type::named_int(pointer, false, "funcref"));
                }
                let target = self.type_at(offset, depth).unwrap_or_else(Type::void);
                Some(Type::pointer_of_width(
                    target.as_ref(),
                    pointer,
                    false,
                    false,
                    None,
                ))
            }
            gimli::DW_TAG_structure_type => {
                self.aggregate(&entry, offset, depth, StructureType::StructStructureType)
            }
            gimli::DW_TAG_class_type => {
                self.aggregate(&entry, offset, depth, StructureType::ClassStructureType)
            }
            gimli::DW_TAG_union_type => {
                self.aggregate(&entry, offset, depth, StructureType::UnionStructureType)
            }
            gimli::DW_TAG_enumeration_type => self.enumeration(&entry, offset),
            gimli::DW_TAG_array_type => self.array(offset, depth),
            gimli::DW_TAG_subroutine_type => self.subroutine(offset, depth),
            gimli::DW_TAG_typedef => self.typedef(&entry, offset, depth),
            gimli::DW_TAG_const_type
            | gimli::DW_TAG_volatile_type
            | gimli::DW_TAG_restrict_type
            | gimli::DW_TAG_atomic_type => self.type_at(offset, depth),
            _ => None,
        }
    }

    fn aggregate(
        &mut self,
        entry: &Entry,
        offset: gimli::UnitOffset,
        depth: usize,
        kind: StructureType,
    ) -> Option<Ref<Type>> {
        let class = match kind {
            StructureType::ClassStructureType => NamedTypeReferenceClass::ClassNamedTypeClass,
            StructureType::UnionStructureType => NamedTypeReferenceClass::UnionNamedTypeClass,
            _ => NamedTypeReferenceClass::StructNamedTypeClass,
        };
        let Some(name) = self.registered_name(entry, offset) else {
            return self.structure(offset, depth, kind);
        };

        if !self.registered.insert(name.clone()) {
            return Some(reference(class, &name));
        }
        if is_declaration(entry) {
            self.registered.remove(&name);
            return Some(reference(class, &name));
        }

        match self.structure(offset, depth, kind) {
            Some(body) => {
                self.debug_info.add_type(&name, &body, &[]);
                self.tally.types += 1;
                Some(reference(class, &name))
            }
            None => {
                self.registered.remove(&name);
                None
            }
        }
    }

    fn structure(
        &mut self,
        offset: gimli::UnitOffset,
        depth: usize,
        kind: StructureType,
    ) -> Option<Ref<Type>> {
        let unit = self.unit;
        let mut builder = StructureBuilder::new();
        builder.structure_type(kind);
        if let Some(size) = udata(&unit.entry(offset).ok()?, gimli::DW_AT_byte_size) {
            builder.width(size);
        }

        let mut bases = Vec::new();
        let mut members = 0usize;
        let mut cursor = unit.entries_at_offset(offset).ok()?;
        cursor.next_dfs().ok()?;

        let mut level = 0isize;
        while let Ok(Some((delta, child))) = cursor.next_dfs() {
            level += delta;
            if level <= 0 {
                break;
            }
            if level > 1 {
                continue;
            }

            match child.tag() {
                gimli::DW_TAG_member => {}
                gimli::DW_TAG_inheritance => {
                    let at = udata(child, gimli::DW_AT_data_member_location).unwrap_or(0);
                    let width = type_ref(child)
                        .and_then(|base| unit.entry(base).ok())
                        .and_then(|base| udata(&base, gimli::DW_AT_byte_size))
                        .unwrap_or(0);
                    if let Some(ty) = self.type_at(child.offset(), depth) {
                        bases.push((ty, at, width));
                    }
                    continue;
                }
                _ => continue,
            }

            if is_declaration(child) {
                continue;
            }
            let name =
                name_of(self.dwarf, unit, child).unwrap_or_else(|| format!("field{members}"));

            let bits = udata(child, gimli::DW_AT_data_bit_offset)
                .or_else(|| udata(child, gimli::DW_AT_data_member_location).map(|at| at * 8));
            let width =
                udata(child, gimli::DW_AT_bit_size).and_then(|bits| u8::try_from(bits).ok());
            let Some(ty) = self.type_at(child.offset(), depth) else {
                continue;
            };
            builder.insert_bitwise(
                &ty,
                &name,
                bits.unwrap_or(0),
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

    fn enumeration(&mut self, entry: &Entry, offset: gimli::UnitOffset) -> Option<Ref<Type>> {
        let underlying = self
            .stripped(type_ref(entry))
            .and_then(|at| self.unit.entry(at).ok());
        let width = udata(entry, gimli::DW_AT_byte_size)
            .or_else(|| {
                underlying
                    .as_ref()
                    .and_then(|at| udata(at, gimli::DW_AT_byte_size))
            })
            .unwrap_or(4);
        let width = NonZeroUsize::new(width as usize)?;
        let name = self.registered_name(entry, offset);
        if let Some(name) = &name {
            if !self.registered.insert(name.clone()) {
                return Some(reference(NamedTypeReferenceClass::EnumNamedTypeClass, name));
            }
        }

        let unit = self.unit;
        let mut builder = EnumerationBuilder::new();
        let mut cursor = unit.entries_at_offset(offset).ok()?;
        cursor.next_dfs().ok()?;

        let mut level = 0isize;
        while let Ok(Some((delta, child))) = cursor.next_dfs() {
            level += delta;
            if level <= 0 {
                break;
            }
            if level > 1 || child.tag() != gimli::DW_TAG_enumerator {
                continue;
            }
            let (Some(member), Some(value)) = (
                name_of(self.dwarf, unit, child),
                constant(child, gimli::DW_AT_const_value),
            ) else {
                continue;
            };
            builder.insert(&member, value);
        }

        let signed = underlying.as_ref().is_none_or(|at| {
            !matches!(
                at.attr_value(gimli::DW_AT_encoding),
                Ok(Some(AttributeValue::Encoding(
                    gimli::DW_ATE_unsigned | gimli::DW_ATE_unsigned_char | gimli::DW_ATE_boolean
                )))
            )
        });
        let body = Type::enumeration(&builder.finalize(), width, signed);
        match name {
            Some(name) => {
                self.debug_info.add_type(&name, &body, &[]);
                self.tally.types += 1;
                Some(reference(
                    NamedTypeReferenceClass::EnumNamedTypeClass,
                    &name,
                ))
            }
            None => Some(body),
        }
    }

    fn typedef(
        &mut self,
        entry: &Entry,
        offset: gimli::UnitOffset,
        depth: usize,
    ) -> Option<Ref<Type>> {
        let class = NamedTypeReferenceClass::TypedefNamedTypeClass;
        let Some(name) = self.registered_name(entry, offset) else {
            return self.type_at(offset, depth);
        };
        if !self.registered.insert(name.clone()) {
            return Some(reference(class, &name));
        }

        match self.type_at(offset, depth) {
            Some(body) => {
                self.debug_info.add_type(&name, &body, &[]);
                self.tally.types += 1;
                Some(reference(class, &name))
            }
            None => {
                self.registered.remove(&name);
                None
            }
        }
    }

    fn array(&mut self, offset: gimli::UnitOffset, depth: usize) -> Option<Ref<Type>> {
        let element = self.type_at(offset, depth)?;

        let unit = self.unit;
        let mut counts = Vec::new();
        let mut cursor = unit.entries_at_offset(offset).ok()?;
        cursor.next_dfs().ok()?;

        let mut level = 0isize;
        while let Ok(Some((delta, child))) = cursor.next_dfs() {
            level += delta;
            if level <= 0 {
                break;
            }
            if level > 1 || child.tag() != gimli::DW_TAG_subrange_type {
                continue;
            }
            let count = udata(child, gimli::DW_AT_count)
                .or_else(|| udata(child, gimli::DW_AT_upper_bound).map(|last| last + 1))
                .unwrap_or(0);
            counts.push(count);
        }

        Some(
            counts
                .iter()
                .rev()
                .fold(element, |inner, count| Type::array(inner.as_ref(), *count)),
        )
    }

    fn subroutine(&mut self, offset: gimli::UnitOffset, depth: usize) -> Option<Ref<Type>> {
        let returns = self.type_at(offset, depth).unwrap_or_else(Type::void);

        let unit = self.unit;
        let mut parameters = Vec::new();
        let mut variadic = false;
        let mut cursor = unit.entries_at_offset(offset).ok()?;
        cursor.next_dfs().ok()?;

        let mut level = 0isize;
        while let Ok(Some((delta, child))) = cursor.next_dfs() {
            level += delta;
            if level <= 0 {
                break;
            }
            if level > 1 {
                continue;
            }
            match child.tag() {
                gimli::DW_TAG_unspecified_parameters => variadic = true,
                gimli::DW_TAG_formal_parameter => {
                    let ty = self
                        .type_at(child.offset(), depth)
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

    fn registered_name(&self, entry: &Entry, offset: gimli::UnitOffset) -> Option<String> {
        let name = name_of(self.dwarf, self.unit, entry)?;
        Some(self.scopes.qualified(offset.0, &name))
    }

    fn type_at(&mut self, offset: gimli::UnitOffset, depth: usize) -> Option<Ref<Type>> {
        let entry = self.unit.entry(offset).ok()?;
        let AttributeValue::UnitRef(target) = entry.attr_value(gimli::DW_AT_type).ok()?? else {
            return None;
        };
        self.build_type(target, depth + 1)
    }

    fn stripped(&self, offset: Option<gimli::UnitOffset>) -> Option<gimli::UnitOffset> {
        let mut at = offset?;
        for _ in 0..MAX_TYPE_DEPTH {
            let entry = self.unit.entry(at).ok()?;
            match entry.tag() {
                gimli::DW_TAG_typedef
                | gimli::DW_TAG_const_type
                | gimli::DW_TAG_volatile_type
                | gimli::DW_TAG_restrict_type
                | gimli::DW_TAG_atomic_type => at = type_ref(&entry)?,
                _ => return Some(at),
            }
        }
        None
    }

    fn name_at(&self, offset: gimli::UnitOffset) -> Option<String> {
        let entry = self.unit.entry(offset).ok()?;
        name_of(self.dwarf, self.unit, &entry)
    }

    fn linkage_at(&self, offset: gimli::UnitOffset) -> Option<String> {
        let entry = self.unit.entry(offset).ok()?;
        let attr = entry.attr_value(gimli::DW_AT_linkage_name).ok()??;
        let raw = self.dwarf.attr_string(self.unit, attr).ok()?;
        let name = module::clean(std::str::from_utf8(raw.slice()).ok()?);
        (!name.is_empty()).then_some(name)
    }

    fn origin(&self, entry: &Entry) -> Option<gimli::UnitOffset> {
        let start = entry.offset();
        let mut at = start;
        for _ in 0..MAX_TYPE_DEPTH {
            let step = self.unit.entry(at).ok()?;
            let next = [gimli::DW_AT_specification, gimli::DW_AT_abstract_origin]
                .into_iter()
                .find_map(|attr| match step.attr_value(attr) {
                    Ok(Some(AttributeValue::UnitRef(offset))) => Some(offset),
                    _ => None,
                });
            match next {
                Some(offset) if offset != at => at = offset,
                _ => return (at != start).then_some(at),
            }
        }
        Some(at)
    }

    fn address(&self, entry: &Entry, attr: gimli::DwAt) -> Option<u64> {
        let AttributeValue::Addr(address) = entry.attr_value(attr).ok()?? else {
            return None;
        };
        (!is_tombstone(address, self.unit.encoding().address_size)).then_some(address)
    }
}

enum Where {
    Memory(u64),
    Local(u32),
    Global(u32),
    Frame(i64),
}

fn location(entry: &Entry, size: u8) -> Option<Where> {
    let AttributeValue::Exprloc(expression) = entry.attr_value(gimli::DW_AT_location).ok()?? else {
        return None;
    };

    let mut operations = expression.0;
    match gimli::DwOp(operations.read_u8().ok()?) {
        gimli::DW_OP_addr => {
            let address = operations.read_address(size).ok()?;
            (!is_tombstone(address, size)).then_some(Where::Memory(address))
        }
        gimli::DW_OP_fbreg => Some(Where::Frame(operations.read_sleb128().ok()?)),
        gimli::DW_OP_WASM_location => match operations.read_uleb128().ok()? {
            0 => Some(Where::Local(
                u32::try_from(operations.read_uleb128().ok()?).ok()?,
            )),
            1 => Some(Where::Global(
                u32::try_from(operations.read_uleb128().ok()?).ok()?,
            )),
            3 => Some(Where::Global(operations.read_u32().ok()?)),
            _ => None,
        },
        _ => None,
    }
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
    fn of(dwarf: &Dwarf<Slice>, unit: &gimli::Unit<Slice>) -> Self {
        let mut spans: Vec<Span> = Vec::new();
        let mut open: Vec<(isize, usize)> = Vec::new();
        let mut level = 0isize;
        let mut entries = unit.entries();

        while let Ok(Some((delta, entry))) = entries.next_dfs() {
            level += delta;
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
            let Some(name) = name_of(dwarf, unit, entry) else {
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
    if size == 0 || size > 16 {
        return None;
    }
    let encoding = match entry.attr_value(gimli::DW_AT_encoding).ok()?? {
        AttributeValue::Encoding(encoding) => encoding,
        _ => return None,
    };
    Some(match encoding {
        gimli::DW_ATE_float => Type::float(size),
        gimli::DW_ATE_boolean => Type::bool(),
        gimli::DW_ATE_signed_char => Type::char(),
        gimli::DW_ATE_unsigned | gimli::DW_ATE_unsigned_char => Type::int(size, false),
        _ => Type::int(size, true),
    })
}

fn reference(class: NamedTypeReferenceClass, name: &str) -> Ref<Type> {
    Type::named_type(&NamedTypeReference::new(class, name))
}

fn type_ref(entry: &Entry) -> Option<gimli::UnitOffset> {
    match entry.attr_value(gimli::DW_AT_type) {
        Ok(Some(AttributeValue::UnitRef(offset))) => Some(offset),
        _ => None,
    }
}

fn is_aggregate(tag: gimli::DwTag) -> bool {
    matches!(
        tag,
        gimli::DW_TAG_structure_type
            | gimli::DW_TAG_class_type
            | gimli::DW_TAG_union_type
            | gimli::DW_TAG_array_type
    )
}

fn has_name(entry: &Entry) -> bool {
    matches!(entry.attr_value(gimli::DW_AT_name), Ok(Some(_)))
}

fn is_declaration(entry: &Entry) -> bool {
    matches!(
        entry.attr_value(gimli::DW_AT_declaration),
        Ok(Some(AttributeValue::Flag(true)))
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
    let value = entry.attr_value(attr).ok()??;
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

fn constant(entry: &Entry, attr: gimli::DwAt) -> Option<u64> {
    match entry.attr_value(attr).ok()?? {
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

fn name_of(dwarf: &Dwarf<Slice>, unit: &gimli::Unit<Slice>, entry: &Entry) -> Option<String> {
    let attr = entry.attr_value(gimli::DW_AT_name).ok()??;
    let raw = dwarf.attr_string(unit, attr).ok()?;
    // A name out of a debug section is no more trustworthy than one out of the module
    let name = module::clean(std::str::from_utf8(raw.slice()).ok()?);
    (!name.is_empty()).then_some(name)
}

fn body_at(module: &Module, address: u64) -> Option<(u32, &FunctionInfo)> {
    module.functions().find(|(_, info)| info.start == address)
}

/// Where a DWARF address of zero points
fn code_base(module: &Module) -> Option<u64> {
    module
        .sections
        .iter()
        .find(|section| section.code)
        .map(|section| section.start)
}

fn section_of(module: &Module, id: SectionId) -> Option<(u64, u64)> {
    let wanted = id.name();
    module
        .sections
        .iter()
        .find(|section| section.name == wanted)
        .map(|section| (section.start, section.end))
}

fn read(view: &BinaryView) -> Option<(Vec<u8>, Vec<Module>)> {
    // From the parent rather than through this view, which has holes where a data segment's bytes
    // are shown in linear memory instead and stops reading at the first of them
    let parent = view.parent_view()?;
    let len = parent.len().min(module::MAX_IMAGE_LEN as u64) as usize;
    let image = parent.read_vec(0, len);

    // Placed where the view placed the file, so every address handed to the core is the one it
    // will look up
    let layout = module::layout(crate::arch::view_id(view))?;
    let mut modules = module::parse_all(&image, 0);
    for module in &mut modules {
        module.place(layout);
    }
    (!modules.is_empty()).then_some((image, modules))
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
    fn a_module_without_dwarf_offers_nothing() {
        let image = wat::parse_str("(module (func nop))").expect("assembles");
        let module = module::parse(&image, 0).expect("parses");

        assert_eq!(section_of(&module, SectionId::DebugInfo), None);
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
