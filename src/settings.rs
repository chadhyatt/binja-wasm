//! Plugin settings and presets

use binaryninja::binary_view::BinaryView;
use binaryninja::rc::Ref;
use binaryninja::settings::{QueryOptions, Settings, SettingsScope};
use binaryninja::workflow::Workflow;
use serde_json::json;

use crate::view;

pub struct Flag {
    key: &'static str,
    title: &'static str,
    description: &'static str,
    default: bool,
}

pub const SOURCE_LINES: Flag = Flag {
    key: "wasm.dwarf.sourceLines",
    title: "Comment Source Lines from DWARF",
    description: "Comment each address the DWARF line table names with the source file and line it \
                  came from. A large module carries hundreds of thousands of these.",
    default: true,
};

pub const STACK_FRAMES: Flag = Flag {
    key: "wasm.analysis.stackFrames",
    title: "Recover Stack Frames",
    description: "Treat the stack pointer global as the stack pointer in each function that \
                  provably restores it before returning, so the shadow stack in linear memory \
                  becomes stack variables. When off, every function reads and writes the global \
                  as memory. Takes effect when the file is opened.",
    default: true,
};

pub const SYMBOL_MAPS: Flag = Flag {
    key: "wasm.debugInfo.symbolMaps",
    title: "Load Sibling Symbol Maps",
    description: "Name functions from the symbol map a Unity (NAME.symbols.json) or Emscripten \
                  (NAME.js.symbols) build writes beside the module, or else name minified \
                  imports and exports from the Emscripten JavaScript (NAME.js) beside it, when \
                  sibling debug files are loaded.",
    default: true,
};

const FLAGS: [&Flag; 3] = [&SOURCE_LINES, &STACK_FRAMES, &SYMBOL_MAPS];

enum Value {
    Bool(bool),
    Workflow(&'static str),
}

struct Preset {
    key: &'static str,
    value: Value,
}

const PRESETS: [Preset; 2] = [
    Preset {
        key: "analysis.linearSweep.autorun",
        value: Value::Bool(false),
    },
    Preset {
        key: "analysis.workflows.moduleWorkflow",
        value: Value::Workflow(view::WORKFLOW),
    },
];

pub fn register() {
    let settings = Settings::global();
    if !settings.register_group("wasm", "WebAssembly") {
        tracing::warn!("wasm: the WebAssembly settings group was not registered");
    }
    for flag in FLAGS {
        let schema = json!({
            "title": flag.title,
            "type": "boolean",
            "default": flag.default,
            "description": flag.description,
        });
        if !settings.register_setting_json(flag.key, &schema.to_string()) {
            tracing::warn!("wasm: the {} setting was not registered", flag.key);
        }
    }
}

fn boolean(key: &str, absent: bool, view: &BinaryView) -> bool {
    let settings = Settings::global();
    if !settings.contains(key) {
        return absent;
    }
    settings.get_bool_with_opts(key, &mut QueryOptions::new_with_view(view))
}

pub fn sibling_debug_files(view: &BinaryView) -> bool {
    boolean("analysis.debugInfo.loadSiblingDebugFiles", false, view)
}

pub fn debug_directories(view: &BinaryView) -> Vec<String> {
    let key = "analysis.debugInfo.debugDirectories";
    if !boolean("analysis.debugInfo.enableDebugDirectories", false, view)
        || !Settings::global().contains(key)
    {
        return Vec::new();
    }
    Settings::global()
        .get_string_list_with_opts(key, &mut QueryOptions::new_with_view(view))
        .iter()
        .map(|directory| directory.to_string())
        .collect()
}

impl Flag {
    pub fn get(&self, view: &BinaryView) -> bool {
        boolean(self.key, self.default, view)
    }

    pub fn opening(&self, data: &BinaryView) -> bool {
        if let Some(load) = data.load_settings(view::NAME)
            && load.contains(self.key)
        {
            let mut chosen = QueryOptions::new_with_view(data);
            let value = load.get_bool_with_opts(self.key, &mut chosen);
            if chosen.scope == SettingsScope::SettingsResourceScope {
                return value;
            }
        }
        self.get(data)
    }
}

impl Preset {
    fn applies(&self, settings: &Settings) -> bool {
        settings.contains(self.key)
            && match self.value {
                Value::Bool(_) => true,
                Value::Workflow(name) => Workflow::get(name).is_some(),
            }
    }

    fn write(&self, settings: &Settings, options: &QueryOptions) {
        match self.value {
            Value::Bool(value) => settings.set_bool_with_opts(self.key, value, options),
            Value::Workflow(name) => settings.set_string_with_opts(self.key, name, options),
        }
    }
}

pub fn load_settings(data: &BinaryView) -> Option<Ref<Settings>> {
    let settings = Settings::new_with_id(&format!(
        "{}.loadSettings.{}",
        view::NAME,
        crate::arch::view_id(data)
    ));
    if !settings.deserialize_schema_with_scope(
        &Settings::global().serialize_schema(),
        SettingsScope::SettingsResourceScope,
    ) {
        tracing::error!("wasm view: could not initialize load settings");
        return None;
    }
    settings.set_resource_id(view::NAME);
    let options =
        QueryOptions::new_with_view(data).with_scope(SettingsScope::SettingsResourceScope);
    for preset in PRESETS.iter().filter(|preset| preset.applies(&settings)) {
        preset.write(&settings, &options);
    }
    Some(settings)
}

pub fn preset(view: &BinaryView) {
    let settings = Settings::global();
    for preset in PRESETS.iter().filter(|preset| preset.applies(&settings)) {
        let mut chosen = QueryOptions::new_with_view(view);
        settings.get_json_with_opts(preset.key, &mut chosen);
        if chosen.scope == SettingsScope::SettingsResourceScope {
            continue;
        }
        let options =
            QueryOptions::new_with_view(view).with_scope(SettingsScope::SettingsResourceScope);
        preset.write(&settings, &options);
    }
}
