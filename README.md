# binja-wasm

A WebAssembly architecture plugin for [Binary Ninja](https://binary.ninja/).

### Features

- BNIL lifting
- Control flow and call graph recovery
- Debug info from the `name` section and DWARF
- Memory64 support
- Component model support
- Text format assembly and patching
- Conformance tested against the official WebAssembly test suite

## Install

Prebuilt binaries are provided for each release on the
[releases page](https://github.com/chadhyatt/binja-wasm/releases). Extract and drop the plugin library for your platform into
your user plugins directory:
- Linux: `~/.binaryninja/plugins`
- macOS: `~/Library/Application Support/Binary Ninja/plugins`
- Windows: `%APPDATA%\Binary Ninja\plugins`

### Building

Prerequisites:
- Binary Ninja installed and licensed
- Rust
- Clang

```sh
git clone --depth 1 https://github.com/chadhyatt/binja-wasm.git && cd binja-wasm
cargo build --release
```

Copy the resulting library out of `target/release` into your plugins directory. Optionally, if you have [Just](https://just.systems/) installed you can use `just install` to automatically build and install the plugin.

## License

MIT License - See [LICENSE](LICENSE)
