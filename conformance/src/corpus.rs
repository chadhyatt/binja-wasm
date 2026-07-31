//! Finding corpus files and getting the binary modules out of them

use std::path::{Path, PathBuf};

/// Every file under `root` that could hold a module, in a stable order
pub fn files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk(root, &mut out);
    out.sort();
    out
}

fn walk(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_file() {
        // A named file still has to look like a corpus file, or a mistyped path sweeps nothing
        // and calls it a pass
        if holds_a_module(path) {
            out.push(path.to_path_buf());
        }
        return;
    }

    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else if holds_a_module(&path) {
            out.push(path);
        }
    }
}

fn holds_a_module(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("wast" | "wasm" | "wat")
    )
}

/// A `.wast` script holds many, deliberately malformed ones included, and those are kept, since a
/// disassembler meets broken input too
pub fn modules_in(path: &Path) -> Result<Vec<Vec<u8>>, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;

    match path.extension().and_then(|ext| ext.to_str()) {
        Some("wasm") => Ok(vec![bytes]),
        Some("wat") => wat::parse_bytes(&bytes)
            .map(|module| vec![module.into_owned()])
            .map_err(|error| error.to_string()),
        _ => script_modules(&bytes),
    }
}

fn script_modules(bytes: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;

    // The lexer flags confusable characters, which is a lint about source readability and no
    // reason to skip the modules inside
    let mut lexer = wast::lexer::Lexer::new(text);
    lexer.allow_confusing_unicode(true);

    let buffer = wast::parser::ParseBuffer::new_with_lexer(lexer).map_err(|e| e.to_string())?;
    let script = wast::parser::parse::<wast::Wast>(&buffer).map_err(|error| error.to_string())?;

    Ok(script
        .directives
        .into_iter()
        .filter_map(|directive| match directive {
            wast::WastDirective::Module(mut module)
            | wast::WastDirective::ModuleDefinition(mut module)
            | wast::WastDirective::AssertMalformed { mut module, .. }
            | wast::WastDirective::AssertInvalidCustom { mut module, .. }
            | wast::WastDirective::AssertInvalid { mut module, .. } => module.encode().ok(),
            // Valid wasm that only fails to link, which this harness never does
            wast::WastDirective::AssertUnlinkable { mut module, .. } => module.encode().ok(),
            _ => None,
        })
        .collect())
}
