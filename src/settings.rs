use binaryninja::binary_view::BinaryView;
use binaryninja::settings::{QueryOptions, Settings, SettingsScope};
use binaryninja::workflow::Workflow;

pub fn register() {
    let settings = Settings::global();
    settings.register_group("wasm", "WebAssembly");
    settings.register_setting_json(
        "wasm.dwarf.sourceLines",
        r#"{
            "title": "Comment source lines from DWARF",
            "type": "boolean",
            "default": true,
            "description": "Comment each address the DWARF line table names with the source file and line it came from. A large module carries hundreds of thousands of these.",
            "ignore": ["SettingsProjectScope", "SettingsResourceScope"]
        }"#,
    );
}

pub fn source_lines() -> bool {
    let settings = Settings::global();
    let key = "wasm.dwarf.sourceLines";
    !settings.contains(key) || settings.get_bool(key)
}

pub fn for_load(settings: &Settings, workflow: &str) {
    analysis(settings, &QueryOptions::new(), workflow);
}

pub fn for_view(view: &BinaryView, workflow: &str) {
    let options =
        QueryOptions::new_with_view(view).with_scope(SettingsScope::SettingsResourceScope);
    analysis(&Settings::global(), &options, workflow);
}

fn analysis(settings: &Settings, options: &QueryOptions, workflow: &str) {
    let sweep = "analysis.linearSweep.autorun";
    let module = "analysis.workflows.moduleWorkflow";

    if settings.contains(sweep) {
        settings.set_bool_with_opts(sweep, false, options);
    }
    if settings.contains(module) && Workflow::get(workflow).is_some() {
        settings.set_string_with_opts(module, workflow, options);
    }
}
