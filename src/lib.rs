pub mod arch;
pub mod asm;
pub mod cfg;
pub mod debug;
pub mod frame;
pub mod insn;
pub mod lift;
pub mod menu;
pub mod module;
pub mod settings;
pub mod symbols;
pub mod upgrade;
pub mod view;
pub mod wasi;

/// Identifies the open file a cached answer came from
pub type ViewId = u64;

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn CorePluginInit() -> bool {
    binaryninja::tracing_init!("binja-wasm");

    if binaryninja::architecture::CoreArchitecture::by_name(arch::NAME).is_some() {
        tracing::error!("binja-wasm is already loaded from another file, so this copy is not");
        return false;
    }

    settings::register();
    arch::register();
    view::register();
    debug::register();
    symbols::register();
    menu::register();

    true
}
