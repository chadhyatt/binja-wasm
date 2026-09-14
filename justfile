set dotenv-load := true
set positional-arguments := true
set windows-powershell := true

lib_name := if os() == "macos" { "libbinja_wasm.dylib" } else if os() == "windows" { "binja_wasm.dll" } else { "libbinja_wasm.so" }

default:
    @just --list

build *ARGS:
    cargo build --locked {{ ARGS }}

build-stub *ARGS:
    cargo build --locked --no-default-features {{ ARGS }}

test *ARGS:
    cargo test --locked {{ ARGS }}

fmt:
    cargo fmt --all

lint:
    python3 scripts/versions.py --check
    cargo fmt --all --check
    cargo clippy --locked --workspace --all-targets -- -D warnings

check: lint test

ci: check build-stub

# Takes a corpus directory, --filter SUBSTR, --all and --quiet. See `conformance --help`.
# Sweep the test suite, checking decoding, module reading, control flow and the stack model.
conformance *ARGS:
    cargo run --locked --release -p conformance -- {{ ARGS }}

# The binaryninja crate is not on docs.rs, so build the docs locally.
doc *ARGS:
    cargo doc --no-deps -p binaryninja --open {{ ARGS }}

# Build the release artifacts into dist/, one archive per platform. Takes --ref, --target, --locked.
[unix]
dist *ARGS:
    python3 scripts/dist.py {{ ARGS }}

# Update plugin.json from the workspace version and API tags.
update-versions:
    python3 scripts/versions.py --write
    cargo update --workspace --quiet

[unix]
install: build
    #!/usr/bin/env bash
    set -euo pipefail
    dir="$(just _plugins-dir)"
    mkdir -p "$dir"
    ln -sf "$PWD/target/debug/{{ lib_name }}" "$dir/{{ lib_name }}"
    echo "linked $dir/{{ lib_name }} -> $PWD/target/debug/{{ lib_name }}"

[windows]
install: build
    @$dir = (just _plugins-dir).Trim(); \
     New-Item -ItemType Directory -Force -Path $dir | Out-Null; \
     $source = Join-Path $PWD 'target\debug\{{ lib_name }}'; \
     $target = Join-Path $dir '{{ lib_name }}'; \
     Remove-Item -Force -ErrorAction Ignore $target; \
     $link = New-Item -ItemType SymbolicLink -Path $target -Target $source -ErrorAction SilentlyContinue; \
     if ($link) { Write-Output "linked $target -> $source" } \
     else { Copy-Item -Force $source $target; Write-Output "copied to $target, re-run after a rebuild" }

[unix]
install-release: (build "--release")
    #!/usr/bin/env bash
    set -euo pipefail
    dir="$(just _plugins-dir)"
    mkdir -p "$dir"
    cp "target/release/{{ lib_name }}" "$dir/{{ lib_name }}"
    echo "installed $dir/{{ lib_name }}"

[windows]
install-release: (build "--release")
    @$dir = (just _plugins-dir).Trim(); \
     New-Item -ItemType Directory -Force -Path $dir | Out-Null; \
     Copy-Item -Force 'target\release\{{ lib_name }}' (Join-Path $dir '{{ lib_name }}'); \
     Write-Output "installed $dir\{{ lib_name }}"

[unix]
uninstall:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -f "$(just _plugins-dir)/{{ lib_name }}"

[windows]
uninstall:
    @Remove-Item -Force -ErrorAction Ignore (Join-Path (just _plugins-dir).Trim() '{{ lib_name }}')

# Launch Binary Ninja with the plugin installed and debug logs on stderr.
[unix]
run *ARGS: install
    #!/usr/bin/env bash
    set -euo pipefail
    exec "$(just _binja)" -e -d -n "$@"

[windows]
run *ARGS: install
    @$binja = (just _binja).Trim(); & $binja -e -d -n {{ ARGS }}

[unix]
setup:
    #!/usr/bin/env bash
    set -euo pipefail

    ask() {
        local answer
        read -r -p "$1 [$2]: " answer </dev/tty || answer=""
        printf '%s' "${answer:-$2}"
    }

    if [ "$(uname -s)" = Darwin ]; then
        support="$HOME/Library/Application Support/Binary Ninja"
        core=libbinaryninjacore.dylib
    else
        support="$HOME/.binaryninja"
        core=libbinaryninjacore.so.1
    fi

    default_dir=""
    [ -f "$support/lastrun" ] && default_dir="$(head -n 1 "$support/lastrun" | tr -d '\r')"

    if [ -f .env ]; then
        read -r -p ".env already exists. Overwrite? [y/N]: " reply </dev/tty || reply=""
        case "$reply" in
            [Yy]*) ;;
            *) echo "keeping existing .env"; exit 0 ;;
        esac
    fi

    binja_dir="$(ask 'Binary Ninja install dir' "$default_dir")"
    plugins_dir="$(ask 'Plugin install dir' "$support/plugins")"

    [ -f "$binja_dir/$core" ] ||
        echo "warning: no $core in '$binja_dir', builds will fail until this is corrected" >&2

    printf 'BINARYNINJADIR=%s\nBN_PLUGINS_DIR=%s\n' "$binja_dir" "$plugins_dir" > .env
    mkdir -p junk "$plugins_dir"

    printf '\nwrote .env:\n'
    cat .env
    printf '\nnext: just install\n'

[windows]
setup:
    @$support = "$env:APPDATA\Binary Ninja"; \
     if (Test-Path .env) { \
         if ((Read-Host '.env already exists. Overwrite? [y/N]') -notmatch '^[Yy]') { \
             Write-Output 'keeping existing .env'; exit 0 \
         } \
     }; \
     $default = ''; \
     if (Test-Path "$support\lastrun") { $default = (Get-Content "$support\lastrun" -First 1).Trim() }; \
     $binja = Read-Host "Binary Ninja install dir [$default]"; if (-not $binja) { $binja = $default }; \
     $plugins = Read-Host "Plugin install dir [$support\plugins]"; if (-not $plugins) { $plugins = "$support\plugins" }; \
     if (-not (Test-Path (Join-Path $binja 'binaryninjacore.dll'))) { \
         Write-Warning "no binaryninjacore.dll in '$binja', builds will fail until this is corrected" \
     }; \
     [IO.File]::WriteAllText('.env', "BINARYNINJADIR=$binja`nBN_PLUGINS_DIR=$plugins`n"); \
     New-Item -ItemType Directory -Force -Path junk, $plugins | Out-Null; \
     Write-Output ''; Write-Output 'wrote .env:'; Get-Content .env; Write-Output ''; Write-Output 'next: just install'

[unix]
_plugins-dir:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "${BN_PLUGINS_DIR:-}" ]; then
        printf '%s' "$BN_PLUGINS_DIR"
    elif [ "$(uname -s)" = Darwin ]; then
        printf '%s' "$HOME/Library/Application Support/Binary Ninja/plugins"
    else
        printf '%s' "$HOME/.binaryninja/plugins"
    fi

[windows]
_plugins-dir:
    @if ($env:BN_PLUGINS_DIR) { Write-Output $env:BN_PLUGINS_DIR } \
     else { Write-Output "$env:APPDATA\Binary Ninja\plugins" }

[unix]
_binja:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="${BINARYNINJADIR:-}"
    if [ -z "$dir" ]; then
        for lastrun in "$HOME/.binaryninja/lastrun" \
                       "$HOME/Library/Application Support/Binary Ninja/lastrun"; do
            [ -f "$lastrun" ] && { dir="$(head -n 1 "$lastrun" | tr -d '\r')"; break; }
        done
    fi
    [ -n "$dir" ] || { echo "no Binary Ninja install found, run 'just setup'" >&2; exit 1; }
    printf '%s' "$dir/binaryninja"

[windows]
_binja:
    @$dir = $env:BINARYNINJADIR; \
     $lastrun = "$env:APPDATA\Binary Ninja\lastrun"; \
     if (-not $dir -and (Test-Path $lastrun)) { $dir = (Get-Content $lastrun -First 1).Trim() }; \
     if (-not $dir) { Write-Error "no Binary Ninja install found, run 'just setup'"; exit 1 }; \
     Write-Output (Join-Path $dir 'binaryninja.exe')
