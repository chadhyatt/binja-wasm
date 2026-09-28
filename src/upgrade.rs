//! Stamping saved databases and placing a reopened one where its snapshot put the file

use std::collections::{HashMap, HashSet};

use binaryninja::binary_view::{BinaryView, MetadataStoreFlags};
use binaryninja::metadata::Metadata;
use binaryninja::rc::Ref;

use crate::module::{Layout, Module, Shape};
use crate::view;

const KEY: &str = "binja_wasm";

const MODEL: u64 = 1;

const MODEL_FIELD: &str = "model";

const VERSION_FIELD: &str = "version";

const LOWEST_FIELD: &str = "lowest";

pub fn saved_layout(view: &BinaryView, modules: &[Module], shape: Shape) -> Option<Layout> {
    let data = view.file().database()?.current_snapshot()?.read_data();
    let keys: HashSet<String> = data.keys().iter().map(|key| key.to_string()).collect();
    let read = |key: &str| -> Option<serde_json::Value> {
        let key = format!("{}/{key}", view::NAME);
        let bytes = data.value(keys.get(&key)?)?;
        serde_json::from_slice(bytes.get_data()).ok()
    };

    let stamp = read("value_store").and_then(|store| store.get("value")?.get(KEY).cloned());
    let field = |name: &str| {
        stamp
            .as_ref()?
            .get("value")?
            .get(name)?
            .get("value")
            .cloned()
    };
    let (Some(model), Some(lowest)) = (
        field(MODEL_FIELD).and_then(|value| value.as_u64()),
        field(LOWEST_FIELD).and_then(|value| value.as_u64()),
    ) else {
        tracing::warn!(
            "wasm view: this database carries no {KEY} stamp this build can read, so its saved \
             analysis is not placed; open the .wasm again to analyse it afresh"
        );
        return None;
    };
    if model != MODEL {
        let version = field(VERSION_FIELD)
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_else(|| "an unknown version".to_owned());
        tracing::warn!(
            "wasm view: this database was saved by {KEY} {version} under lifting model {model}, \
             and this build lifts model {MODEL}; open the .wasm again for analysis that matches \
             this build"
        );
    }

    let saved: Vec<u64> = read("functions")
        .as_ref()
        .and_then(|functions| functions.as_array())
        .into_iter()
        .flatten()
        .filter_map(|function| function.get("addr")?.as_u64())
        .map(|address| address.saturating_add(lowest))
        .collect();
    let entries: Vec<u64> = modules
        .iter()
        .flat_map(|module| module.functions().map(|(_, info)| info.entry))
        .collect();
    let mapped = read("memory_map").and_then(|map| mapped_from(&map, lowest));
    let placed = placement(saved, &entries, mapped, shape);
    if placed.is_none() {
        tracing::warn!(
            "wasm view: the database does not show where the file was placed, so analysis saved \
             at other addresses will not line up"
        );
    }
    placed
}

fn placement(
    mut saved: Vec<u64>,
    entries: &[u64],
    mapped: Option<u64>,
    shape: Shape,
) -> Option<Layout> {
    saved.sort_unstable();
    let first = *entries.iter().min()?;
    saved
        .iter()
        .take(SAVED_CANDIDATES)
        .filter_map(|lowest| lowest.checked_sub(first))
        .map(|file_base| {
            let memory = mapped.unwrap_or(shape.memory).min(file_base);
            Layout::at(file_base, Shape { memory, ..shape })
        })
        .map(|layout| {
            let expected: Vec<u64> = entries
                .iter()
                .map(|entry| layout.file_address(*entry))
                .chain(
                    (0..shape.imports)
                        .filter_map(|nth| Some(layout.import_address(u32::try_from(nth).ok()?))),
                )
                .collect();
            let found = expected
                .iter()
                .filter(|at| saved.binary_search(at).is_ok())
                .count();
            (layout, found, expected.len())
        })
        .max_by_key(|(_, found, _)| *found)
        .filter(|(_, found, expected)| found * 2 > *expected)
        .map(|(layout, _, _)| layout)
}

const SAVED_CANDIDATES: usize = 8;

fn mapped_from(map: &serde_json::Value, lowest: u64) -> Option<u64> {
    let mut regions: Vec<(u64, u64)> = map
        .get("memory_regions")?
        .as_array()?
        .iter()
        .filter_map(|region| {
            Some((
                region.get("start")?.as_u64()?.saturating_add(lowest),
                region.get("end")?.as_u64()?.saturating_add(lowest),
            ))
        })
        .collect();
    regions.sort_unstable();
    let first = lowest.min(1);
    let mut end = first;
    for (start, last) in regions {
        if start > end {
            break;
        }
        end = end.max(last.saturating_add(1));
    }
    Some(if end == first { 0 } else { end })
}

pub fn stamp(view: &BinaryView) {
    let lowest = view
        .segments()
        .iter()
        .map(|segment| segment.address_range().start)
        .min()
        .unwrap_or(0);
    let stamp: HashMap<&str, Ref<Metadata>> = HashMap::from([
        (MODEL_FIELD, MODEL.into()),
        (VERSION_FIELD, env!("CARGO_PKG_VERSION").into()),
        (LOWEST_FIELD, lowest.into()),
    ]);
    view.store_metadata(KEY, stamp, MetadataStoreFlags::PERSISTENT);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> Shape {
        Shape {
            image: 0x2000,
            memory: 0x400,
            reserved: 0x10000,
            imports: 2,
            ..Shape::default()
        }
    }

    #[test]
    fn a_saved_file_is_found_where_its_functions_were() {
        let entries = [0x120, 0x180, 0x1f0];
        let saved_at = Layout::at(0x60000, shape());
        let saved: Vec<u64> = entries
            .iter()
            .map(|entry| saved_at.file_address(*entry))
            .chain([saved_at.import_address(0), saved_at.import_address(1)])
            .chain([0x800])
            .collect();

        let layout = placement(saved, &entries, Some(0x20000), shape()).expect("placed");
        assert_eq!(layout.file_base, 0x60000, "past a function made in memory");
        assert_eq!(layout.import_base, saved_at.import_base);
        assert_eq!(layout.memory_end, 0x20000, "memory as far as it was mapped");
    }

    #[test]
    fn saved_functions_that_fit_no_placement_give_none() {
        let entries = [0x120, 0x180, 0x1f0];
        assert!(placement(vec![0x5000, 0x9000, 0x12345], &entries, None, shape()).is_none());
        assert!(placement(Vec::new(), &entries, None, shape()).is_none());
        assert!(placement(vec![0x60120], &[], None, shape()).is_none());
    }

    #[test]
    fn mapped_memory_is_the_run_of_regions_from_the_lowest_address() {
        let map = serde_json::json!({"memory_regions": [
            {"start": 0x41c, "end": 0xfffe},
            {"start": 0, "end": 0x3fe},
            {"start": 0x3ff, "end": 0x41b},
            {"start": 0x1ffff, "end": 0x20ffe},
        ]});
        assert_eq!(
            mapped_from(&map, 1),
            Some(0x10000),
            "regions saved relative to the first mapped address, one"
        );
        assert_eq!(
            mapped_from(
                &serde_json::json!({"memory_regions": [{"start": 0, "end": 0xff}]}),
                0
            ),
            Some(0x100),
            "data placed at address zero"
        );

        let unmapped = serde_json::json!({"memory_regions": [{"start": 0x20000, "end": 0x20fff}]});
        assert_eq!(mapped_from(&unmapped, 1), Some(0));
        assert_eq!(
            mapped_from(
                &serde_json::json!({"memory_regions": [{"start": 0, "end": 0xff}]}),
                0x60000
            ),
            Some(0),
            "a file with no memory, saved relative to where the file was"
        );
        assert_eq!(mapped_from(&serde_json::json!({}), 1), None);
    }
}
