#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from versions import ROOT, plugin_metadata

DIST = ROOT / "dist"

TARGETS = [
    {
        "name": "linux-x86_64",
        "triple": "x86_64-unknown-linux-gnu",
        "library": "libbinja_wasm.so",
        "archive": "gztar",
    },
    {
        "name": "linux-arm64",
        "triple": "aarch64-unknown-linux-gnu",
        "library": "libbinja_wasm.so",
        "archive": "gztar",
        "tools": ["aarch64-linux-gnu-gcc", "aarch64-linux-gnu-g++"],
        "env": {
            "CC_aarch64_unknown_linux_gnu": "aarch64-linux-gnu-gcc",
            "CXX_aarch64_unknown_linux_gnu": "aarch64-linux-gnu-g++",
            "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER": "aarch64-linux-gnu-gcc",
        },
    },
    {
        "name": "macos-x86_64",
        "triple": "x86_64-apple-darwin",
        "library": "libbinja_wasm.dylib",
        "archive": "gztar",
    },
    {
        "name": "macos-arm64",
        "triple": "aarch64-apple-darwin",
        "library": "libbinja_wasm.dylib",
        "archive": "gztar",
    },
    {
        "name": "windows-x86_64",
        "triple": "x86_64-pc-windows-msvc",
        "library": "binja_wasm.dll",
        "archive": "zip",
    },
]

EXTRA_FILES = ["plugin.json", "LICENSE", "THIRDPARTY"]


def run(command, env=None):
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, **(env or {})})
    if result.returncode != 0:
        sys.exit(result.returncode)


def build(target, ref):
    name, triple = target["name"], target["triple"]
    env = target.get("env", {})

    print(f"> building {name} ({triple})")
    run(["rustup", "target", "add", triple])
    run(
        ["cargo", "build", "--release", "--locked", "--no-default-features", "--target", triple],
        env,
    )

    built = ROOT / "target" / triple / "release" / target["library"]
    if not built.is_file():
        sys.exit(f"{triple} built nothing at {built}")

    staged = DIST / name
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True)
    shutil.copy2(built, staged / target["library"])
    for extra in EXTRA_FILES:
        shutil.copy2(ROOT / extra, staged / extra)

    archive = shutil.make_archive(
        str(DIST / f"binja-wasm-{ref}-{name}"),
        format=target["archive"],
        root_dir=staged,
        base_dir=".",
    )
    shutil.rmtree(staged)
    print(f"> packaged {Path(archive).name}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref", default="", help="what to call the archives (default: the crate version)"
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="NAME",
        help="build only this target, repeatable (default: every target)",
    )
    args = parser.parse_args()

    version, api = plugin_metadata(check=True)
    ref = (args.ref or version).replace("/", "_")

    wanted = TARGETS
    if args.target:
        known = {target["name"] for target in TARGETS}
        unknown = sorted(set(args.target) - known)
        if unknown:
            sys.exit(f"no such target: {', '.join(unknown)}, pick from {', '.join(sorted(known))}")
        wanted = [target for target in TARGETS if target["name"] in args.target]

    missing = {
        tool
        for target in wanted
        for tool in target.get("tools", [])
        if shutil.which(tool) is None
    }
    if missing:
        sys.exit(f"missing cross toolchain: {', '.join(sorted(missing))}")

    print(f"binja-wasm {ref} against Binary Ninja {api}\n")
    DIST.mkdir(exist_ok=True)
    for target in wanted:
        build(target, ref)

    print("dist/")
    for archive in sorted(DIST.iterdir()):
        print(f"  {archive.name}  {archive.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()
