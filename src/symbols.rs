//! Function names from a Unity `NAME.symbols.json` (an object from function index to name) or an
//! Emscripten symbol map (`index:name` lines), or else for a build that minified its import and
//! export names, from the tables in the Emscripten JavaScript that loads it

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use binaryninja::binary_view::{BinaryView, BinaryViewBase};
use binaryninja::debuginfo::{
    CustomDebugInfoParser, DebugFunctionInfo, DebugInfo, DebugInfoParser,
};
use binaryninja::demangle::demangle_llvm;
use binaryninja::symbol::{Binding, Symbol, SymbolType};

use crate::module::{self, Module, Signature, ValueKind};
use crate::{debug, settings, view, wasi};

pub const NAME: &str = "WASM Symbol Map";

const SNIFFED: u64 = 64;

const PROGRESS_STRIDE: usize = 4096;

struct SymbolMap;

impl CustomDebugInfoParser for SymbolMap {
    fn is_valid(&self, view: &BinaryView) -> bool {
        if debug::modules_of(view).is_empty() {
            return looks_like_map(&view.read_vec(0, SNIFFED.min(view.len()) as usize));
        }
        settings::SYMBOL_MAPS.get(view)
            && !(sibling_maps(view).is_empty() && sibling_glue(view).is_empty())
    }

    fn parse_info(
        &self,
        debug_info: &mut DebugInfo,
        view: &BinaryView,
        debug_file: &BinaryView,
        progress: Box<dyn Fn(usize, usize) -> Result<(), ()>>,
    ) -> bool {
        let modules = debug::modules_of(view);
        let Some(module) = modules
            .iter()
            .max_by_key(|module| module.functions().count())
        else {
            return false;
        };
        let names = if debug_file.file().session_id() != view.file().session_id() {
            let path = debug_file.file().file_path();
            match parse(&debug::file_image(debug_file)) {
                Some(names) => fitting(names, module, &path),
                None => {
                    tracing::warn!("wasm symbol map: {} is not one", path.display());
                    None
                }
            }
        } else {
            sibling_maps(view)
                .into_iter()
                .find_map(|path| fitting(parse(&debug::read_file(&path)?)?, module, &path))
                .or_else(|| {
                    sibling_glue(view).into_iter().find_map(|path| {
                        let text = String::from_utf8(debug::read_file(&path)?).ok()?;
                        glue_names(&text, module)
                    })
                })
        };
        let Some(names) = names else {
            return false;
        };

        let mut named = 0usize;
        for (nth, (index, name)) in names.iter().enumerate() {
            if nth % PROGRESS_STRIDE == 0 && progress(nth, names.len()).is_err() {
                tracing::info!("wasm symbol map: import cancelled");
                return false;
            }
            let Some(info) = module.function(*index) else {
                if let Some(import) = module.import(*index) {
                    let address = module.layout.import_address(*index);
                    define(view, SymbolType::ImportedFunction, name, address);
                    if !wasi::specifies(import)
                        && let Some(wasi) = wasi::implemented_as(view, name, &import.signature)
                        && let Some(stub) = view.functions_at(address).iter().next()
                    {
                        view::type_stub(&stub, &wasi);
                    }
                    named += 1;
                }
                continue;
            };
            if module.export_name(*index).is_some() {
                define(view, SymbolType::Function, name, info.entry);
                named += 1;
                continue;
            }
            let (short, full) = display_names(name);
            let function = DebugFunctionInfo::new(
                Some(short),
                Some(full),
                Some(name.clone()),
                None,
                Some(info.entry),
                None,
                Vec::new(),
                Vec::new(),
            );
            named += usize::from(debug_info.add_function(&function));
        }
        tracing::info!("wasm symbol map: {named} functions named");
        named != 0
    }
}

fn define(view: &BinaryView, kind: SymbolType, name: &str, address: u64) {
    let (short, full) = display_names(name);
    view.define_user_symbol(
        &Symbol::builder(kind, name, address)
            .binding(Binding::Global)
            .short_name(short)
            .full_name(full)
            .create(),
    );
}

fn sibling_maps(view: &BinaryView) -> Vec<PathBuf> {
    siblings(view, |name, stem| {
        vec![
            format!("{stem}.symbols.json"),
            format!("{stem}.js.symbols"),
            format!("{stem}.symbols"),
            format!("{name}.symbols"),
        ]
    })
}

fn sibling_glue(view: &BinaryView) -> Vec<PathBuf> {
    siblings(view, |_, stem| {
        ["js", "mjs", "cjs"]
            .iter()
            .map(|extension| format!("{stem}.{extension}"))
            .collect()
    })
}

fn siblings(view: &BinaryView, expected: impl Fn(&str, &str) -> Vec<String>) -> Vec<PathBuf> {
    let file = view.file();
    let path = file
        .original_file_path()
        .unwrap_or_else(|| file.file_path());
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return Vec::new();
    };
    let stem = Path::new(name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(name);
    let mut expected = expected(name, stem);
    expected.dedup();
    expected
        .iter()
        .flat_map(|expected| debug::sibling_files(view, expected))
        .collect()
}

const IMPORT_OBJECTS: [&str; 2] = ["wasmImports", "asmLibraryArg"];

const EXPORT_ACCESSES: [&str; 4] = ["wasmExports[", "[\"asm\"][", "['asm'][", ".asm["];

fn glue_names(text: &str, module: &Module) -> Option<BTreeMap<u32, String>> {
    let mut names = BTreeMap::new();
    let imported = IMPORT_OBJECTS
        .iter()
        .find_map(|object| object_entries(text, object))
        .filter(|entries| minified(entries));
    if let Some(imported) = imported {
        let owners = import_owners(text);
        let fields: HashMap<&str, &str> = imported
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        for (index, import) in module.imports() {
            if owners.contains(&import.module) {
                names.insert(index, symbol(fields.get(import.field.as_str())?));
            }
        }
    }
    let exported = export_entries(text);
    let assigned: Vec<(String, String)> = exported
        .iter()
        .filter_map(|(key, target)| Some((key.clone(), target.clone()?)))
        .collect();
    if minified(&assigned) {
        let targets: HashMap<&str, Option<&str>> = exported
            .iter()
            .map(|(key, target)| (key.as_str(), target.as_deref()))
            .collect();
        for (index, key) in module.exports() {
            if let Some(target) = targets.get(key)? {
                names.insert(index, symbol(target));
            }
        }
    }
    let consistent = names.iter().all(|(index, name)| {
        let Some(letters) = name.strip_prefix("dynCall_") else {
            return true;
        };
        let signature = module
            .function(*index)
            .map(|info| &info.signature)
            .or_else(|| module.import(*index).map(|import| &import.signature));
        signature.is_some_and(|signature| dynamic_call(letters) == Some(signature.clone()))
    });
    (consistent && !names.is_empty()).then_some(names)
}

fn dynamic_call(letters: &str) -> Option<Signature> {
    let kind = |letter| match letter {
        'i' => Some(ValueKind::I32),
        'j' => Some(ValueKind::I64),
        'f' => Some(ValueKind::F32),
        'd' => Some(ValueKind::F64),
        _ => None,
    };
    let mut letters = letters.chars();
    let results = match letters.next()? {
        'v' => Vec::new(),
        result => vec![kind(result)?],
    };
    let params = std::iter::once(Some(ValueKind::I32))
        .chain(letters.map(kind))
        .collect::<Option<Vec<_>>>()?;
    Some(Signature { params, results })
}

fn symbol(identifier: &str) -> String {
    module::clean(identifier.strip_prefix('_').unwrap_or(identifier))
}

fn minified(entries: &[(String, String)]) -> bool {
    let renamed = entries
        .iter()
        .filter(|(key, value)| *key != symbol(value) && key != value)
        .count();
    renamed * 2 > entries.len()
}

fn is_identifier(text: &str) -> bool {
    !text.is_empty()
        && !text.starts_with(|c: char| c.is_ascii_digit())
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}

fn unquoted(text: &str) -> &str {
    let text = text.trim();
    text.strip_prefix(['"', '\''])
        .and_then(|inner| inner.strip_suffix(['"', '\'']))
        .unwrap_or(text)
}

fn object_entries(text: &str, object: &str) -> Option<Vec<(String, String)>> {
    text.match_indices(object).find_map(|(at, _)| {
        let before = text[..at].chars().next_back();
        if before.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$' || c == '.') {
            return None;
        }
        let rest = text[at + object.len()..].trim_start();
        let rest = rest.strip_prefix('=')?;
        let body = rest.trim_start().strip_prefix('{')?;
        let body = &body[..body.find('}')?];
        let entries: Vec<(String, String)> = without_comments(body)
            .split(',')
            .filter_map(|entry| {
                let (key, value) = entry.split_once(':')?;
                let (key, value) = (unquoted(key), value.trim());
                (is_identifier(key) && is_identifier(value))
                    .then(|| (key.to_owned(), value.to_owned()))
            })
            .collect();
        (!entries.is_empty()).then_some(entries)
    })
}

fn without_comments(text: &str) -> String {
    let mut kept = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("/*") {
        kept.push_str(&rest[..at]);
        rest = rest[at..]
            .find("*/")
            .map_or("", |end| &rest[at + end + 2..]);
    }
    kept.push_str(rest);
    kept.lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .collect::<Vec<_>>()
        .join("\n")
}

fn import_owners(text: &str) -> Vec<String> {
    let mut owners = Vec::new();
    for object in IMPORT_OBJECTS {
        for (at, _) in text.match_indices(object) {
            let after = text[at + object.len()..].chars().next();
            if after.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$') {
                continue;
            }
            let Some(head) = text[..at].trim_end().strip_suffix(':') else {
                continue;
            };
            let start = head.rfind(['{', ',', '\n']).map_or(0, |start| start + 1);
            let key = unquoted(&head[start..]);
            if !key.is_empty() && !owners.iter().any(|owner| owner == key) {
                owners.push(key.to_owned());
            }
        }
    }
    owners
}

fn export_entries(text: &str) -> Vec<(String, Option<String>)> {
    let mut entries: Vec<(String, Option<String>)> = Vec::new();
    for access in EXPORT_ACCESSES {
        for (at, _) in text.match_indices(access) {
            let inside = &text[at + access.len()..];
            let Some(close) = inside.find(']') else {
                continue;
            };
            let key = unquoted(&inside[..close]);
            if !is_identifier(key) {
                continue;
            }
            let start = text[..at]
                .rfind([';', '{', '}', '(', '\n', ','])
                .map_or(0, |start| start + 1);
            let target = text[start..at]
                .split_once('=')
                .map(|(target, _)| target.trim().trim_start_matches("var ").trim())
                .filter(|target| is_identifier(target))
                .map(str::to_owned);
            match entries.iter_mut().find(|(known, _)| known == key) {
                Some((_, known)) => *known = known.take().or(target),
                None => entries.push((key.to_owned(), target)),
            }
        }
    }
    entries
}

fn looks_like_map(head: &[u8]) -> bool {
    let head = String::from_utf8_lossy(head);
    let head = head.trim_start();
    let first = match head.strip_prefix('{') {
        Some(object) => object.trim_start().strip_prefix('"'),
        None => Some(head),
    };
    first.is_some_and(|first| {
        let digits = first.bytes().take_while(u8::is_ascii_digit).count();
        digits != 0
            && first[digits..]
                .trim_start_matches('"')
                .trim_start()
                .starts_with(':')
    })
}

fn parse(bytes: &[u8]) -> Option<BTreeMap<u32, String>> {
    let text = std::str::from_utf8(bytes).ok()?;
    let entries: Option<BTreeMap<u32, String>> = if text.trim_start().starts_with('{') {
        let object: HashMap<String, String> = serde_json::from_str(text).ok()?;
        object
            .into_iter()
            .map(|(index, name)| Some((index.parse().ok()?, module::clean(&name))))
            .collect()
    } else {
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let (index, name) = line.split_once(':')?;
                Some((index.trim().parse().ok()?, module::clean(name.trim_end())))
            })
            .collect()
    };
    entries.filter(|entries| !entries.is_empty())
}

fn fitting(
    names: BTreeMap<u32, String>,
    module: &Module,
    path: &Path,
) -> Option<BTreeMap<u32, String>> {
    match describes(&names, module) {
        Ok(()) => Some(names),
        Err(reason) => {
            tracing::warn!(
                "wasm symbol map: {} is not applied, since {reason}",
                path.display()
            );
            None
        }
    }
}

fn describes(names: &BTreeMap<u32, String>, module: &Module) -> Result<(), String> {
    let count = module.imports().count() + module.functions().count();
    if let Some((&last, _)) = names.last_key_value()
        && last as usize >= count
    {
        return Err(format!(
            "it names function {last} and the module has {count}"
        ));
    }
    if names.len() != count {
        return Err(format!(
            "it names {} of the module's {count} functions",
            names.len()
        ));
    }
    Ok(())
}

fn display_names(name: &str) -> (String, String) {
    if name.starts_with("_Z")
        && let Some(demangled) = demangle_llvm(name, true)
    {
        let qualified = module::clean(&demangled.name.to_string());
        return (qualified.clone(), qualified);
    }
    (without_parameters(name).to_owned(), name.to_owned())
}

fn without_parameters(signature: &str) -> &str {
    let mut end = signature.trim_end();
    while let Some(stripped) = [" const", " volatile", " &&", " &"]
        .iter()
        .find_map(|qualifier| end.strip_suffix(qualifier))
    {
        end = stripped.trim_end();
    }
    if !end.ends_with(')') {
        return signature;
    }
    let mut depth = 0usize;
    for (at, character) in end.char_indices().rev() {
        match character {
            ')' => depth += 1,
            '(' => {
                depth -= 1;
                if depth == 0 {
                    let head = end[..at].trim_end();
                    return if head.is_empty() { signature } else { head };
                }
            }
            _ => {}
        }
    }
    signature
}

pub fn register() {
    DebugInfoParser::register(NAME, SymbolMap);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_formats_read_as_the_same_map() {
        let unity = parse(br#"{ "0": "exit", "2": "Unity::Component::SendMessageAny(MessageIdentifier const&, MessageData&)" }"#)
            .expect("a map");
        let emscripten = parse(
            b"0:exit\n2:Unity::Component::SendMessageAny(MessageIdentifier const&, MessageData&)\n",
        )
        .expect("a map");
        assert_eq!(unity, emscripten);
        assert_eq!(unity.len(), 2);

        assert!(looks_like_map(
            br#"{
 "0": "exit","#
        ));
        assert!(looks_like_map(b"12:main\n"));
        assert!(!looks_like_map(b"\0asm\x01\0\0\0"));
        assert!(!looks_like_map(br#"{"name": "x"}"#));
        assert_eq!(parse(b"{}"), None, "nothing named");
        assert_eq!(parse(br#"{"x": "y"}"#), None, "not indices");
        assert_eq!(parse(b"\0asm"), None);
    }

    #[test]
    fn a_map_names_only_the_module_it_was_written_for() {
        let module = module::parse(
            &wat::parse_str(r#"(module (import "env" "exit" (func (param i32))) (func) (func))"#)
                .expect("assembles"),
            0,
        )
        .expect("parses");
        let map = |pairs: &[(u32, &str)]| -> BTreeMap<u32, String> {
            pairs
                .iter()
                .map(|(index, name)| (*index, name.to_string()))
                .collect()
        };

        assert!(describes(&map(&[(0, "exit"), (1, "main"), (2, "helper")]), &module).is_ok());
        assert!(
            describes(&map(&[(1, "main"), (2, "helper")]), &module).is_err(),
            "one from a build with fewer functions"
        );
        assert!(
            describes(
                &map(&[(0, "exit"), (1, "main"), (3, "past_the_end")]),
                &module
            )
            .is_err(),
            "a function the module does not have"
        );
        assert!(
            describes(&map(&[(0, "abort"), (1, "main"), (2, "helper")]), &module).is_ok(),
            "an import under the name the build gave it rather than the minified one"
        );
    }

    #[test]
    fn minified_imports_and_exports_take_the_names_their_glue_gives_them() {
        let module = module::parse(
            &wat::parse_str(
                r#"(module
                    (import "a" "a" (func (param i32) (result i32)))
                    (import "a" "b" (func (param i32)))
                    (func (export "e") (param i32) (result i32) local.get 0)
                    (func (export "j") (param i32 i32 i32))
                    (func (export "d"))
                    (memory (export "c") 1))"#,
            )
            .expect("assembles"),
            0,
        )
        .expect("parses");
        let glue = |exports: &str| {
            format!(
                r#"function getWasmImports(){{var imports={{a:wasmImports}};return imports}}
                function initRuntime(){{runtimeInitialized=true;wasmExports["d"]()}}
                function assignWasmExports(wasmExports){{{exports}
                memory=wasmMemory=wasmExports["c"]}}
                var wasmImports={{a:_emscripten_resize_heap,b:_emscripten_sleep}};"#
            )
        };
        let minified = glue(
            r#"_caller=Module["_caller"]=wasmExports["e"];dynCall_vii=dynCalls["vii"]=wasmExports["j"];"#,
        );
        let names: Vec<(u32, String)> = glue_names(&minified, &module)
            .expect("names")
            .into_iter()
            .collect();
        assert_eq!(
            names,
            [
                (0, "emscripten_resize_heap".into()),
                (1, "emscripten_sleep".into()),
                (2, "caller".into()),
                (3, "dynCall_vii".into()),
            ],
            "the constructors the glue only calls keep their export name"
        );

        assert_eq!(
            glue_names(
                &glue(r#"_caller=wasmExports["e"];dynCall_iii=dynCalls["iii"]=wasmExports["j"];"#),
                &module
            ),
            None,
            "a dynamic call whose letters are not its function's signature is another build's"
        );
        assert_eq!(
            glue_names(&glue(r#"_caller=wasmExports["e"];"#), &module),
            None,
            "nor is glue that leaves out an export"
        );

        let unminified = module::parse(
            &wat::parse_str(
                r#"(module (import "env" "emscripten_sleep" (func (param i32)))
                    (func (export "caller") (param i32) (result i32) local.get 0))"#,
            )
            .expect("assembles"),
            0,
        )
        .expect("parses");
        let plain = "var imports = {\n  'env': wasmImports,\n};\n\
            _caller = Module['_caller'] = wasmExports['caller'];\n\
            var wasmImports = {\n  /** @export */\n  emscripten_sleep: _emscripten_sleep\n};";
        assert_eq!(
            glue_names(plain, &unminified),
            None,
            "names the module already carries are left alone"
        );
    }

    #[test]
    fn a_signature_is_shown_by_its_name() {
        assert_eq!(
            without_parameters(
                "Unity::Component::SendMessageAny(MessageIdentifier const&, MessageData&)"
            ),
            "Unity::Component::SendMessageAny"
        );
        assert_eq!(
            without_parameters("RuntimeInvoker_Static(void (*)(), MethodInfo const*, void*)"),
            "RuntimeInvoker_Static"
        );
        assert_eq!(
            without_parameters("Foo::operator()(int) const"),
            "Foo::operator()"
        );
        assert_eq!(without_parameters("exit"), "exit");
        assert_eq!(without_parameters("(anonymous)"), "(anonymous)");
        assert_eq!(
            without_parameters("18315121a9f14c7140ec55ec386d5f0f"),
            "18315121a9f14c7140ec55ec386d5f0f"
        );
    }
}
