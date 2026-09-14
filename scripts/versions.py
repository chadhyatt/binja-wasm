import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def versions():
    with (ROOT / "Cargo.toml").open("rb") as manifest:
        workspace = tomllib.load(manifest)["workspace"]
    dependencies = workspace["dependencies"]
    api = dependencies["binaryninja"]
    core = dependencies["binaryninjacore-sys"]
    if (api.get("git"), api.get("tag")) != (core.get("git"), core.get("tag")):
        raise ValueError("Binary Ninja dependencies must use the same Git repository and tag")
    match = re.fullmatch(r"stable/(\d+\.\d+\.\d+)", api.get("tag", ""))
    if match is None:
        raise ValueError("Binary Ninja dependencies require a stable/MAJOR.MINOR.BUILD tag")
    return workspace["package"]["version"], match[1]


def plugin_metadata(check=False):
    version, api = versions()
    path = ROOT / "plugin.json"
    plugin = json.loads(path.read_text())
    expected = {
        "version": version,
        "minimumbinaryninjaversion": int(api.rsplit(".", 1)[1]),
    }
    if any(plugin.get(key) != value for key, value in expected.items()):
        if check:
            raise ValueError("plugin.json versions are stale; run just update-versions")
        plugin.update(expected)
        path.write_text(json.dumps(plugin, indent=2) + "\n")
    return version, api


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="check plugin.json against Cargo.toml")
    action.add_argument("--write", action="store_true", help="update plugin.json from Cargo.toml")
    action.add_argument("--api", action="store_true", help="print the Binary Ninja API version")
    args = parser.parse_args()
    try:
        if args.api:
            print(versions()[1])
        else:
            version, api = plugin_metadata(check=args.check)
            print(f"{version} (BN {api})")
    except (OSError, ValueError, KeyError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
