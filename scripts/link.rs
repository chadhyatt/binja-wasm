pub fn binaryninja_core() {
    let link_path = std::env::var_os("DEP_BINARYNINJACORE_PATH")
        .expect("DEP_BINARYNINJACORE_PATH not specified");

    println!("cargo::rustc-link-lib=dylib=binaryninjacore");
    println!("cargo::rustc-link-search={}", link_path.display());

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS not set");
    if matches!(target_os.as_str(), "linux" | "macos") {
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", link_path.display());
    }
}
