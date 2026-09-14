//! Finding corpus files and getting the binary modules out of them

use std::path::{Path, PathBuf};

use wast::lexer::{Lexer, TokenKind};

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

pub struct Contents {
    pub modules: Vec<Vec<u8>>,
    pub forms: usize,
    pub rejected: Vec<String>,
}

impl Contents {
    fn one(module: Vec<u8>) -> Self {
        Contents {
            modules: vec![module],
            forms: 1,
            rejected: Vec::new(),
        }
    }

    pub fn hole(&self) -> Option<String> {
        let first = self.rejected.first()?;
        Some(format!(
            "{} of {} forms unreadable, first at {first}",
            self.rejected.len(),
            self.forms
        ))
    }
}

/// A `.wast` script holds many, deliberately malformed ones included, and those are kept, since a
/// disassembler meets broken input too
pub fn contents_of(path: &Path) -> Result<Contents, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;

    match path.extension().and_then(|ext| ext.to_str()) {
        Some("wasm") => Ok(Contents::one(bytes)),
        Some("wat") => wat::parse_bytes(&bytes)
            .map(|module| Contents::one(module.into_owned()))
            .map_err(|error| error.to_string()),
        _ => script(&bytes),
    }
}

/// Read one top-level form at a time, so a form the `wast` crate cannot parse costs only itself
/// rather than every module after it
fn script(bytes: &[u8]) -> Result<Contents, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let forms = forms(text)?;

    let mut contents = Contents {
        modules: Vec::new(),
        forms: forms.len(),
        rejected: Vec::new(),
    };
    for form in forms {
        match script_modules(form.text) {
            Ok(modules) => contents.modules.extend(modules),
            Err(error) => {
                contents
                    .rejected
                    .push(format!("line {}: {}", form.line, error.message()))
            }
        }
    }
    Ok(contents)
}

struct Form<'a> {
    text: &'a str,
    line: usize,
}

/// The top-level forms of a script, found with the crate's own lexer so a paren inside a string
/// or comment cannot throw the count off
fn forms(text: &str) -> Result<Vec<Form<'_>>, String> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;

    for token in lexer(text).iter(0) {
        let token = token.map_err(|error| error.message())?;
        match token.kind {
            TokenKind::LParen => {
                if depth == 0 {
                    start = token.offset;
                }
                depth += 1;
            }
            TokenKind::RParen if depth == 0 => {
                return Err(format!(
                    "line {}: unmatched `)`",
                    line_at(text, token.offset)
                ));
            }
            TokenKind::RParen => {
                depth -= 1;
                if depth == 0 {
                    let end = token.offset + token.len as usize;
                    out.push(Form {
                        text: &text[start..end],
                        line: line_at(text, start),
                    });
                }
            }
            TokenKind::Whitespace | TokenKind::LineComment | TokenKind::BlockComment => {}
            _ if depth == 0 => {
                return Err(format!(
                    "line {}: a token outside any form",
                    line_at(text, token.offset)
                ));
            }
            _ => {}
        }
    }

    if depth != 0 {
        return Err(format!("line {}: unclosed form", line_at(text, start)));
    }
    Ok(out)
}

fn line_at(text: &str, offset: usize) -> usize {
    text[..offset].matches('\n').count() + 1
}

/// The lexer flags confusable characters, which is a lint about source readability and no reason
/// to skip the modules inside
fn lexer(text: &str) -> Lexer<'_> {
    let mut lexer = Lexer::new(text);
    lexer.allow_confusing_unicode(true);
    lexer
}

fn script_modules(text: &str) -> Result<Vec<Vec<u8>>, wast::Error> {
    let buffer = wast::parser::ParseBuffer::new_with_lexer(lexer(text))?;
    let script = wast::parser::parse::<wast::Wast>(&buffer)?;

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
