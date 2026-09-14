//! Checks this crate's decoding, module reading, control flow recovery and stack model against
//! real WebAssembly
//!
//! Not a unit test: it wants a corpus on disk and it is slow
//!
//! Parsing is `wasmparser`'s job and is not under test; everything built on top of it is, above all
//! that the stack effect the lifter applies matches what a real validator computes, since moving
//! the stack pointer by the wrong amount corrupts every instruction after it

mod check;
mod corpus;
mod report;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use report::Report;

const DEFAULT_CORPUS: &str = "extern/testsuite";

const USAGE: &str = "\
usage: conformance [OPTIONS] [CORPUS]

  CORPUS             directory to sweep, recursively (default: extern/testsuite)

  --filter SUBSTR    only visit files whose path contains SUBSTR
  --all              print every failure rather than the first 20
  --quiet            print the summary only
  -h, --help         this
";

fn main() -> ExitCode {
    let options = match Options::parse() {
        Ok(Some(options)) => options,
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("{message}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    if !options.corpus.exists() {
        eprintln!("no corpus at {}", options.corpus.display());
        if options.corpus == Path::new(DEFAULT_CORPUS) {
            eprintln!("run `just testsuite` to check out the official suite");
        }
        return ExitCode::FAILURE;
    }

    let mut files = corpus::files(&options.corpus);
    if let Some(filter) = &options.filter {
        files.retain(|path| path.to_string_lossy().contains(filter.as_str()));
    }
    if files.is_empty() {
        eprintln!(
            "no .wast, .wat or .wasm files under {}",
            options.corpus.display()
        );
        return ExitCode::FAILURE;
    }

    let mut report = Report::new(options.limit());
    for file in &files {
        report.files += 1;
        let hole = match corpus::contents_of(file) {
            Ok(contents) => {
                for module in &contents.modules {
                    check::module(file, module, &mut report);
                }
                contents.hole()
            }
            Err(why) => Some(why),
        };
        // Anything the harness cannot read is a hole in the corpus rather than a pass, unless it
        // is one of the handful known to be unreadable
        match (hole, known_unreadable(file)) {
            (Some(why), Some(reason)) => report
                .expected_unreadable
                .push(format!("{} ({reason}): {why}", file.display())),
            (Some(why), None) => report.unreadable.push(format!("{}: {why}", file.display())),
            (None, Some(_)) => report.stale_expectations.push(file.display().to_string()),
            (None, None) => {}
        }
    }

    report.print(options.quiet, options.all);
    if report.passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Forms the `wast` crate cannot parse: it dropped the folded syntax of the legacy exception
/// handling proposal, so `cfg.rs` covers those operators from bytes instead, and it is behind the
/// text format on a couple of newer tests
///
/// Listed rather than ignored, so a hole can neither open nor close quietly
const KNOWN_UNREADABLE: [(&str, &str); 6] = [
    ("legacy/rethrow.wast", "legacy exception handling syntax"),
    ("legacy/throw.wast", "legacy exception handling syntax"),
    ("legacy/try_catch.wast", "legacy exception handling syntax"),
    (
        "legacy/try_delegate.wast",
        "legacy exception handling syntax",
    ),
    (
        "proposals/extended-name-section/custom/name_annot.wast",
        "field name annotations",
    ),
    ("type-subtyping.wast", "multiple supertypes"),
];

fn known_unreadable(path: &Path) -> Option<&'static str> {
    let path = path.to_string_lossy().replace('\\', "/");
    KNOWN_UNREADABLE
        .iter()
        .find(|(suffix, _)| path.ends_with(suffix))
        .map(|(_, reason)| *reason)
}

struct Options {
    corpus: PathBuf,
    filter: Option<String>,
    all: bool,
    quiet: bool,
}

impl Options {
    fn parse() -> Result<Option<Self>, String> {
        let mut options = Options {
            corpus: PathBuf::from(DEFAULT_CORPUS),
            filter: None,
            all: false,
            quiet: false,
        };
        let mut corpus_given = false;
        let mut args = std::env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => return Ok(None),
                "--all" => options.all = true,
                "--quiet" => options.quiet = true,
                "--filter" => {
                    let substring = args.next().ok_or("--filter wants a substring after it")?;
                    if substring.starts_with('-') {
                        return Err(format!("--filter wants a substring, not {substring}"));
                    }
                    options.filter = Some(substring);
                }
                other if other.starts_with('-') => return Err(format!("unknown option {other}")),
                other if corpus_given => return Err(format!("unexpected argument {other}")),
                other => {
                    options.corpus = PathBuf::from(other);
                    corpus_given = true;
                }
            }
        }

        Ok(Some(options))
    }

    fn limit(&self) -> usize {
        if self.all { usize::MAX } else { 20 }
    }
}
