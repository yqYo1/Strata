#!/usr/bin/env python3
"""Generate review artifacts only. Does not apply source, build, or run fixtures."""
from pathlib import Path
import difflib
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
inputs = [
    "src/program/generate.cpp", "include/strata/program/prompt_checkpoint.hpp",
    "src/prefill/layer_major.inc", "src/core/conversation_state.cpp",
    "src/prefill/prefill.cpp", "include/strata/prefill/prefill.hpp",
    "bench/results/2026-10-15-pr6-recovery-audit/research-index.json",
    "bench/results/2026-10-15-pr6-recovery-audit/research-agent-cache-settings-round608-v1.txt",
    "AGENTS.md", "docs/ARTIFACT_RETENTION.md",
]

def change(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"source anchor count differs: {old[:100]!r}")
    return text.replace(old, new, 1)

header_path = "include/strata/program/prompt_checkpoint.hpp"
header = (ROOT / header_path).read_text()
header = change(header, "// Niko's opt-in trailing-effort turn stays outside the reusable checkpoint.", '''// The next interval is measured from the checkpoint actually taken, rather
// than from a global grid. Saturation preserves the monotone position contract.
inline int64_t checkpoint_next(int64_t position, int64_t interval) {
    if (position < 0 || interval < 0)
        throw std::invalid_argument("invalid prompt checkpoint interval position");
    const auto limit = std::numeric_limits<int64_t>::max();
    return interval == 0 || interval > limit - position ? limit : position + interval;
}

// First chunk end reaching `next`, capped at the existing outer boundary.
// Use the actual chunk after lend/relayout. Avoid rounding with addition or
// multiplication, which can overflow for large CLI intervals/positions.
inline int64_t checkpoint_segment_end(int64_t begin, int64_t end,
                                      int64_t next, int64_t chunk) {
    if (begin < 0 || end <= begin || next < 0 || chunk <= 0)
        throw std::invalid_argument("invalid prompt checkpoint segment boundary");
    const int64_t remaining = end - begin;
    const int64_t distance = next > begin ? next - begin : 1;
    if (distance >= remaining) return end;
    const int64_t remainder = distance % chunk;
    const int64_t padding = remainder == 0 ? 0 : chunk - remainder;
    if (padding >= remaining - distance) return end;
    return begin + distance + padding; // each partial sum is bounded by end
}

// Niko's opt-in trailing-effort turn stays outside the reusable checkpoint.''')

source_path = "src/program/generate.cpp"
source = (ROOT / source_path).read_text()
source = change(source,
    "        int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;",
    '''        // Layer-major final-layer chunks can arrive after earlier layers have
        // read the whole segment. Save periodic state only after a full run,
        // its MTP callbacks/expert restore, and the prompt-slot refill complete.
        const char* lm_checkpoint_mode = std::getenv("STRATA_PREFILL_LAYER_MAJOR");
        const bool coherent_periodic = !multi_gpu && o.prompt_cache > 0 && o.prompt_cache_every > 0 &&
            lm_checkpoint_mode && std::strcmp(lm_checkpoint_mode, "1") == 0;
        int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;''')
source = change(source,
    "            if (o.prompt_cache_every > 0 && done >= pp_next_check) {",
    "            if (!coherent_periodic && o.prompt_cache_every > 0 && done >= pp_next_check) {")
source = change(source,
    "                pp_next_check = done + o.prompt_cache_every;",
    "                pp_next_check = strata::program::checkpoint_next(done, o.prompt_cache_every);")
source = change(source,
    "                    part_next[i] = done + o.prompt_cache_every;",
    "                    part_next[i] = strata::program::checkpoint_next(done, o.prompt_cache_every);")
source = change(source,
    "            pp_next_check = reread_to > 0 ? INT64_MAX : resume + o.prompt_cache_every;",
    "            pp_next_check = reread_to > 0 ? INT64_MAX : strata::program::checkpoint_next(resume, o.prompt_cache_every);")
old = '''                const bool sp_ok = win ? read_windows(at, to, err) : run_prompt_with_mtp_experts(
                    sp, mtp, native_pack && o.spec == 4 && !o.mtp.empty() && !multi_gpu &&
                    o.layer_split.empty() && o.peer_device < 0 && !peer.valid() &&
                    std::all_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(), [](auto slots) { return slots <= 0; }),
                    ids.data() + at, to - at, at, err);'''
new = '''                bool boundary_failed = false;
                auto run_batched = [&](int64_t a, int64_t b) {
                    return run_prompt_with_mtp_experts(
                        sp, mtp, native_pack && o.spec == 4 && !o.mtp.empty() && !multi_gpu &&
                        o.layer_split.empty() && o.peer_device < 0 && !peer.valid() &&
                        std::all_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(), [](auto slots) { return slots <= 0; }),
                        ids.data() + a, b - a, a, err);
                };
                auto read_batched = [&]() {
                    // Keep the cold, interval-zero and reread diagnostic paths
                    // as one run per existing root/turn/reread boundary.
                    if (!coherent_periodic || reread_to > 0) return run_batched(at, to);
                    for (int64_t q = at; q < to;) {
                        if (stop_req.load()) { err = "cancelled"; return false; }
                        // The outer loop already lent the first segment. Decide
                        // windows on the original outer length, never a newly
                        // shortened periodic segment; all these reads batched.
                        if (q != at && !lend(to - q, err)) {
                            err = "lending the prompt path its slots failed: " + err;
                            boundary_failed = true; return false;
                        }
                        const int64_t end = strata::program::checkpoint_segment_end(q, to, pp_next_check, sp.chunk());
                        if (!run_batched(q, end)) return false;
                        if (!refill(err)) {
                            err = "refilling a lent slot failed: " + err;
                            boundary_failed = true; return false;
                        }
                        if (end >= pp_next_check) {
                            if (!checkpoint_at(end)) {
                                err = "saving a conversation checkpoint failed";
                                boundary_failed = true; return false;
                            }
                            pp_next_check = strata::program::checkpoint_next(end, o.prompt_cache_every);
                        }
                        q = end;
                    }
                    return true;
                };
                const bool sp_ok = win ? read_windows(at, to, err) : read_batched();'''
source = change(source, old, new)
source = change(source,
    "                if (!sp_ok) {\n                    if (!stop_req.load()) {",
    "                if (!sp_ok) {\n                    if (boundary_failed || !stop_req.load()) {")

patch = []
for relative, updated in [(source_path, source), (header_path, header)]:
    path = OUT / "proposed" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated)
    original = (ROOT / relative).read_text()
    patch.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                    fromfile="a/" + relative, tofile="b/" + relative))
(OUT / "source.diff").write_text("".join(patch))

def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

ledger = {
    "status": "proposal_only_not_applied_not_built_not_tested",
    "source_revision": "Main reports fixed 92acf71e; Git was prohibited and not queried",
    "origin": "Niko1221/Strata personal SYCL fork; no CUDA/HIP implementation import",
    "inputs": {relative: identity(ROOT / relative) for relative in inputs},
    "outputs": {str(path.relative_to(OUT)): identity(path) for path in sorted(OUT.rglob("*"))
                if path.is_file() and path.name != "sha256-ledger.json"},
    "commands_executed": ["read-only source/report reads", "python3 generate_proposal.py (generation and hashes only)"],
    "forbidden_actions_performed": [],
}
(OUT / "sha256-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
print(json.dumps({"output": str(OUT), "status": ledger["status"],
                  "patch": identity(OUT / "source.diff")}, indent=2))
