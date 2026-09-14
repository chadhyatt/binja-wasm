#[path = "scripts/link.rs"]
mod link;

fn main() {
    link::binaryninja_core();

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let crate_name = std::env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME not set");
        let lib_name = crate_name.replace('-', "_");
        println!(
            "cargo::rustc-link-arg-cdylib=-Wl,-install_name,@rpath/lib{}.dylib",
            lib_name
        );
    }
}
