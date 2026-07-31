//! Importing the DWARF a compiler left in the module's custom sections
//!
//! Binary Ninja's own DWARF plugin claims a `.wasm` and then fails on it: it ignores the view's
//! sections, re-reads the raw file, and hands it to the `object` crate, which is built without its
//! `wasm` feature; reading the custom sections directly is all `gimli` ever needed
//!
//! Two things decide whether any of it lines up: an address is relative to the code section
//! payload, and `DW_AT_low_pc` points at a body's locals declaration rather than its first
//! instruction

use binaryninja::binary_view::{BinaryView, BinaryViewBase, BinaryViewExt};
use binaryninja::debuginfo::{
    CustomDebugInfoParser, DebugFunctionInfo, DebugInfo, DebugInfoParser,
};
use binaryninja::rc::Ref;
use binaryninja::types::{
    FunctionParameter, MemberAccess, MemberScope, StructureBuilder, StructureType, Type,
};
use gimli::{AttributeValue, Dwarf, EndianSlice, LittleEndian, Reader, SectionId};

use crate::module::{self, FunctionInfo, Module};

pub const NAME: &str = "WASM DWARF";

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

        let mut found = 0usize;
        let mut seen = 0usize;

        // An address is relative to the code section payload of the module carrying it, so each
        // nested module is loaded against its own
        for module in &modules {
            let Some(base) = code_base(module) else {
                continue;
            };

            let load = |id: SectionId| -> Result<EndianSlice<LittleEndian>, gimli::Error> {
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
                found += subprograms(module.layout, &dwarf, &unit, module, base, debug_info);
                found += variables(module.layout, &dwarf, &unit, debug_info);
            }
        }

        tracing::debug!("imported {found} functions from wasm DWARF");
        found != 0
    }
}

fn subprograms(
    layout: module::Layout,
    dwarf: &Dwarf<EndianSlice<LittleEndian>>,
    unit: &gimli::Unit<EndianSlice<LittleEndian>>,
    module: &Module,
    base: u64,
    debug_info: &mut DebugInfo,
) -> usize {
    let mut added = 0usize;
    let mut entries = unit.entries();

    while let Ok(Some((_, entry))) = entries.next_dfs() {
        if entry.tag() != gimli::DW_TAG_subprogram {
            continue;
        }

        let Ok(Some(low_pc)) = entry.attr_value(gimli::DW_AT_low_pc) else {
            continue;
        };
        let AttributeValue::Addr(low_pc) = low_pc else {
            continue;
        };

        let Some(name) = name_of(dwarf, unit, entry) else {
            continue;
        };

        // A subprogram address lands on the locals declaration rather than the first instruction
        let address = base.wrapping_add(low_pc);
        let Some(body) = body_at(module, address) else {
            continue;
        };

        let info = DebugFunctionInfo::new(
            Some(name.clone()),
            Some(name.clone()),
            Some(name),
            prototype(layout, dwarf, unit, entry),
            Some(body.entry),
            None,
            Vec::new(),
            // The components marshalling in these bindings is wrong for a non-empty slice
            Vec::new(),
        );
        if debug_info.add_function(&info) {
            added += 1;
        }
    }

    added
}

/// A C global lives in linear memory rather than in the file, so its DWARF address is a memory
/// offset and only means anything because the view maps memory at a base of its own
fn variables(
    layout: module::Layout,
    dwarf: &Dwarf<EndianSlice<LittleEndian>>,
    unit: &gimli::Unit<EndianSlice<LittleEndian>>,
    debug_info: &mut DebugInfo,
) -> usize {
    let mut added = 0usize;
    let mut entries = unit.entries();

    while let Ok(Some((_, entry))) = entries.next_dfs() {
        if entry.tag() != gimli::DW_TAG_variable {
            continue;
        }

        let Some(at) = static_address(entry) else {
            continue;
        };
        let Some(name) = name_of(dwarf, unit, entry) else {
            continue;
        };
        let Some(ty) = referenced_type(layout, dwarf, unit, entry, gimli::DW_AT_type, 0) else {
            continue;
        };

        if !layout.memory_mapped(at) {
            continue;
        }
        debug_info.add_data_variable(layout.memory_address(at), &ty, Some(&name), &[]);
        added += 1;
    }

    added
}

/// A local is described by `DW_OP_WASM_location` or a frame offset and has no address to give, so
/// anything that is not a plain `DW_OP_addr` is left alone
fn static_address(
    entry: &gimli::DebuggingInformationEntry<EndianSlice<LittleEndian>>,
) -> Option<u64> {
    let AttributeValue::Exprloc(expression) = entry.attr_value(gimli::DW_AT_location).ok()?? else {
        return None;
    };

    let mut operations = expression.0;
    let opcode = operations.read_u8().ok()?;
    (opcode == gimli::constants::DW_OP_addr.0)
        .then(|| operations.read_u32().ok().map(u64::from))
        .flatten()
}

/// A circular `.debug_info` would otherwise recurse until the stack ran out, which in a callback
/// the core does not guard is a dead process
const MAX_TYPE_DEPTH: usize = 16;

fn prototype(
    layout: module::Layout,
    dwarf: &Dwarf<EndianSlice<LittleEndian>>,
    unit: &gimli::Unit<EndianSlice<LittleEndian>>,
    entry: &gimli::DebuggingInformationEntry<EndianSlice<LittleEndian>>,
) -> Option<Ref<Type>> {
    let returns = referenced_type(layout, dwarf, unit, entry, gimli::DW_AT_type, 0)
        .unwrap_or_else(Type::void);

    // Parameters are the subprogram's own children, so the walk stops at the next sibling
    let mut parameters = Vec::new();
    let mut cursor = unit.entries_at_offset(entry.offset()).ok()?;
    cursor.next_dfs().ok()?;

    let mut level = 0isize;
    while let Ok(Some((delta, child))) = cursor.next_dfs() {
        // An inlined callee is a child too, and its parameters are not this function's
        level += delta;
        if level <= 0 {
            break;
        }
        if level > 1 || child.tag() != gimli::DW_TAG_formal_parameter {
            continue;
        }

        let ty = referenced_type(layout, dwarf, unit, child, gimli::DW_AT_type, 0)
            .unwrap_or_else(|| Type::int(4, true));
        let name =
            name_of(dwarf, unit, child).unwrap_or_else(|| format!("arg{}", parameters.len()));
        parameters.push(FunctionParameter::new(ty, name, None));
    }

    Some(Type::function(returns.as_ref(), parameters, false))
}

fn referenced_type(
    layout: module::Layout,
    dwarf: &Dwarf<EndianSlice<LittleEndian>>,
    unit: &gimli::Unit<EndianSlice<LittleEndian>>,
    entry: &gimli::DebuggingInformationEntry<EndianSlice<LittleEndian>>,
    attr: gimli::DwAt,
    depth: usize,
) -> Option<Ref<Type>> {
    if depth >= MAX_TYPE_DEPTH {
        return None;
    }
    let AttributeValue::UnitRef(offset) = entry.attr_value(attr).ok()?? else {
        return None;
    };
    build_type(layout, dwarf, unit, offset, depth + 1)
}

/// Only the shapes that carry across, since anything else becomes its own width or nothing rather
/// than a guess
fn build_type(
    layout: module::Layout,
    dwarf: &Dwarf<EndianSlice<LittleEndian>>,
    unit: &gimli::Unit<EndianSlice<LittleEndian>>,
    offset: gimli::UnitOffset,
    depth: usize,
) -> Option<Ref<Type>> {
    if depth >= MAX_TYPE_DEPTH {
        return None;
    }
    let entry = unit.entry(offset).ok()?;

    match entry.tag() {
        gimli::DW_TAG_base_type => {
            let size = match entry.attr_value(gimli::DW_AT_byte_size).ok()?? {
                AttributeValue::Udata(size) => size as usize,
                _ => return None,
            };
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
                gimli::DW_ATE_unsigned | gimli::DW_ATE_unsigned_char => Type::int(size, false),
                _ => Type::int(size, true),
            })
        }
        // Sized from the module's own layout rather than from the architecture, which would make
        // the importer untestable without a core
        gimli::DW_TAG_pointer_type => {
            let target = referenced_type(layout, dwarf, unit, &entry, gimli::DW_AT_type, depth)
                .unwrap_or_else(Type::void);
            Some(Type::pointer_of_width(
                target.as_ref(),
                layout.pointer,
                false,
                false,
                None,
            ))
        }
        gimli::DW_TAG_structure_type => structure(layout, dwarf, unit, offset, depth, false),
        gimli::DW_TAG_union_type => structure(layout, dwarf, unit, offset, depth, true),
        // A name for the same thing underneath, so it resolves to what it aliases
        gimli::DW_TAG_typedef
        | gimli::DW_TAG_const_type
        | gimli::DW_TAG_volatile_type
        | gimli::DW_TAG_restrict_type => {
            referenced_type(layout, dwarf, unit, &entry, gimli::DW_AT_type, depth)
        }
        _ => None,
    }
}

/// Members are the type's own children, so the walk stops at the next sibling
fn structure(
    layout: module::Layout,
    dwarf: &Dwarf<EndianSlice<LittleEndian>>,
    unit: &gimli::Unit<EndianSlice<LittleEndian>>,
    offset: gimli::UnitOffset,
    depth: usize,
    union: bool,
) -> Option<Ref<Type>> {
    let mut builder = StructureBuilder::new();
    if union {
        builder.structure_type(StructureType::UnionStructureType);
    }
    let mut cursor = unit.entries_at_offset(offset).ok()?;
    cursor.next_dfs().ok()?;

    let mut members = 0usize;
    let mut level = 0isize;
    while let Ok(Some((delta, child))) = cursor.next_dfs() {
        // `next_dfs` reports the step, not the level, so a nested type's members are not these
        level += delta;
        if level <= 0 {
            break;
        }
        if level > 1 || child.tag() != gimli::DW_TAG_member {
            continue;
        }

        let Some(ty) = referenced_type(layout, dwarf, unit, child, gimli::DW_AT_type, depth) else {
            continue;
        };
        let at = match child.attr_value(gimli::DW_AT_data_member_location) {
            Ok(Some(AttributeValue::Udata(at))) => at,
            // A union's members all start at the beginning, and say so by saying nothing
            _ => 0,
        };
        let name = name_of(dwarf, unit, child).unwrap_or_else(|| format!("field{members}"));

        builder.insert(
            &ty,
            &name,
            at,
            false,
            MemberAccess::PublicAccess,
            MemberScope::NoScope,
        );
        members += 1;
    }

    (members != 0).then(|| Type::structure(&builder.finalize()))
}

fn name_of(
    dwarf: &Dwarf<EndianSlice<LittleEndian>>,
    unit: &gimli::Unit<EndianSlice<LittleEndian>>,
    entry: &gimli::DebuggingInformationEntry<EndianSlice<LittleEndian>>,
) -> Option<String> {
    let attr = entry.attr_value(gimli::DW_AT_name).ok()??;
    let raw = dwarf.attr_string(unit, attr).ok()?;
    // A name out of a debug section is no more trustworthy than one out of the module
    let name = module::clean(std::str::from_utf8(raw.slice()).ok()?);
    (!name.is_empty()).then_some(name)
}

fn body_at(module: &Module, address: u64) -> Option<&FunctionInfo> {
    module
        .functions()
        .map(|(_, info)| info)
        .find(|info| info.start == address)
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
}
