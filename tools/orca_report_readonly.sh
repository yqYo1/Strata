#!/usr/bin/env bash
# Launch one fresh, bounded Orca report worker. Main-authored; repository LICENSE applies.
# Usage: bash tools/orca_report_readonly.sh /absolute/private/report-directory
# This is prepared source, not proof of effective worker/tool isolation. Main must
# qualify actual product-write denials and Task-bound lifecycle IPC before research.
set -euo pipefail
umask 077

refuse() { printf 'report-only: %s\n' "$*" >&2; exit 1; }
[[ $# == 1 && "$1" == /* ]] || refuse 'one absolute report directory is required'
for report_command in bwrap timeout realpath stat id codex python3; do
    command -v "$report_command" >/dev/null || refuse "missing command: $report_command"
done
report_dir="$(realpath -e -- "$1")"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
codex_store_home="$(realpath -e -- "${CODEX_HOME:-$HOME/.codex}")"
[[ -d "$report_dir" && ! -L "$1" ]] || refuse 'report directory must exist without a final symlink'
[[ "$(stat -c %u -- "$report_dir")" == "$(id -u)" ]] || refuse 'report directory is not owned by this user'
[[ "$(stat -c %a -- "$report_dir")" == 700 ]] || refuse 'report directory must have mode 0700'
case "$report_dir/" in
    "$repo_root/"*|"$codex_store_home/"*) refuse 'report directory must be outside the product and Codex home' ;;
esac
case "$repo_root/" in
    "$report_dir/"*) refuse 'report directory must not contain the product' ;;
esac
case "$codex_store_home/" in
    "$report_dir/"*) refuse 'report directory must not contain the Codex home' ;;
esac
codex_executable="$(realpath -e -- "$(command -v codex)")"
codex_auth="$(realpath -e -- "$codex_store_home/auth.json")"
[[ -f "$codex_auth" && -r "$codex_auth" ]] || refuse 'readable file-based Codex authentication is required'

# These are all inside the named report directory. Refuse reuse rather than
# resuming an old agent or sharing its SQLite state, plugins or MCP configuration.
report_codex_state="$report_dir/.codex-state"
report_tmp="$report_dir/.tmp"
[[ ! -e "$report_codex_state" && ! -e "$report_tmp" ]] || refuse 'fresh report directory state is required'
mkdir -- "$report_codex_state" "$report_tmp"
python3 - "$report_codex_state/config.toml" "$report_dir" <<'PY'
import json, sys
from pathlib import Path
config = '''model = "gpt-6-luna"
model_reasoning_effort = "medium"
sandbox_mode = "workspace-write"
approval_policy = "never"

[sandbox_workspace_write]
network_access = true

'''
config += '[projects.' + json.dumps(sys.argv[2], ensure_ascii=False) + ']\ntrust_level = "trusted"\n'
with Path(sys.argv[1]).open('x') as stream:
    stream.write(config)
PY

# Keep HOME/CODEX_HOME unchanged. Mount private state over the existing Codex
# home in this process's namespace, while retaining read-only auth and runtime.
# The outer root is read-only; /dev and /proc are private namespace mounts.
# Host-readable files and host networking remain visible: this is a local
# filesystem write barrier, not a confidentiality or network-isolation claim.
report_mounts=(
    --die-with-parent --unshare-pid
    --ro-bind / /
    --dev /dev --proc /proc
    --bind "$report_dir" "$report_dir"
    --bind "$report_codex_state" "$codex_store_home"
    --ro-bind "$codex_auth" "$codex_store_home/auth.json"
)
if [[ -d "$codex_store_home/packages" ]]; then
    report_mounts+=(--ro-bind "$codex_store_home/packages" "$codex_store_home/packages")
fi
case "$codex_executable" in
    "$codex_store_home/"*)
        [[ "$codex_executable" == "$codex_store_home/packages/"* ]] ||
            refuse 'Codex executable inside the overlaid home needs a supported read-only runtime path'
        ;;
esac

# A fresh no-daemon CLI avoids the shared app-server's tool executor. The real
# worker must still prove its tool boundary and Task/Dispatch/terminal identity;
# a launcher exit or an idle prompt does not establish task/process closure.
# No /dev/dri is mounted. Temporary output stays in the report directory without
# hiding host /tmp sockets required for Orca lifecycle IPC.
exec env TMPDIR="$report_tmp" timeout --foreground --kill-after=5s 480s \
    bwrap "${report_mounts[@]}" --chdir "$report_dir" \
    "$codex_executable" --no-daemon --no-alt-screen --cd "$report_dir" \
    --sandbox workspace-write --ask-for-approval never \
    --model gpt-6-luna --config 'model_reasoning_effort="medium"' \
    --config sandbox_workspace_write.network_access=true
