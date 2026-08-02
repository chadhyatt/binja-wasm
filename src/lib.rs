pub mod arch;
pub mod asm;
pub mod cfg;
pub mod debug;
pub mod insn;
pub mod lift;
pub mod module;
pub mod settings;
pub mod view;

/// Identifies the open file a cached answer came from
pub type ViewId = u64;

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn CorePluginInit() -> bool {
    binaryninja::tracing_init!("binja-wasm");

    settings::register();
    arch::register();
    view::register();
    debug::register();

    true
}
