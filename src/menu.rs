//! Menu bar commands (Plugins > WASM > ...)

use std::path::Path;

use binaryninja::background_task::BackgroundTask;
use binaryninja::binary_view::BinaryView;
use binaryninja::command::{Command, register_command};
use binaryninja::debuginfo::{DebugInfo, DebugInfoParser};
use binaryninja::file_metadata::FileMetadata;
use binaryninja::interaction::{self, MessageBoxButtonSet, MessageBoxIcon};
use binaryninja::rc::Ref;

use crate::{debug, module, symbols};

struct Import {
    parser: &'static str,
    prompt: &'static str,
    filter: &'static str,
}

impl Command for Import {
    fn action(&self, view: &BinaryView) {
        let Some(path) = interaction::get_open_filename_input(self.prompt, self.filter) else {
            return;
        };
        let (view, parser) = (view.to_owned(), self.parser);
        std::thread::spawn(move || import(&view, parser, &path));
    }

    fn valid(&self, view: &BinaryView) -> bool {
        !debug::modules_of(view).is_empty()
    }
}

fn import(view: &BinaryView, parser: &str, path: &Path) {
    let task = BackgroundTask::new(&format!("Importing {}", path.display()), false);
    let _running = task.enter();
    let Some(info) = read(view, parser, path) else {
        interaction::show_message_box(
            "Nothing imported",
            &format!(
                "{} has nothing that applies to this module; the log says why.",
                path.display()
            ),
            MessageBoxButtonSet::OKButtonSet,
            MessageBoxIcon::ErrorIcon,
        );
        return;
    };
    module::unmark_restored(crate::arch::view_id(view));
    view.apply_debug_info(&info);
    view.update_analysis();
}

fn read(view: &BinaryView, parser: &str, path: &Path) -> Option<Ref<DebugInfo>> {
    let Ok(parser) = DebugInfoParser::from_name(parser) else {
        tracing::error!("wasm: no debug info parser is registered as {parser}");
        return None;
    };
    let Ok(file) = BinaryView::from_path(&FileMetadata::new(), path) else {
        tracing::warn!("wasm: {} could not be opened", path.display());
        return None;
    };
    let info = parser.parse_debug_info(view, &file, None);
    file.file().close();
    info
}

pub fn register() {
    register_command(
        "WASM\\Import DWARF",
        "Import DWARF file (e.g. from emcc's -gseparate-dwarf)",
        Import {
            parser: debug::NAME,
            prompt: "DWARF for this module",
            filter: "*.wasm",
        },
    );
    register_command(
        "WASM\\Import Symbol Map",
        "Import symbol names from a Unity .symbols.json or an Emscripten symbol map",
        Import {
            parser: symbols::NAME,
            prompt: "Symbol map for this module",
            filter: "*.json *.symbols",
        },
    );
}
