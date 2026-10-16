#!/usr/bin/env python3
"""Generate proposal copies/diff/hashes only. Never apply, compile or run tests."""
from pathlib import Path
import difflib
import hashlib
import json
ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PIN = {
 'src/program/generate.cpp':'d4048082aae24f4afc6c841309b0d0f63f2303085880d92da1cc7ebe4e83e80c',
 'src/prefill/prefill.cpp':'a01c39cc269f84ab85129d8c106d8adb5227e33234fa5a6d7ae6537ab2f68371',
 'include/strata/prefill/prefill.hpp':'2a2fc2930ce4e24dde9c67fd65592887241d8c62659031613e7f47365763e81c',
}
def sha(b): return hashlib.sha256(b).hexdigest()
def replace(s, old, new):
    if s.count(old) != 1: raise RuntimeError('anchor count != 1: '+old[:90])
    return s.replace(old,new)
old = {}
for name, pin in PIN.items():
    raw = (ROOT/name).read_bytes()
    if sha(raw) != pin: raise RuntimeError('source changed: '+name)
    old[name] = raw.decode()
new = dict(old)
name = 'include/strata/prefill/prefill.hpp'
new[name] = replace(new[name], '    /// Checked before every chunk:', '''    // Internal layer-major capture hook: run_impl calls this INLINE after its
    // checked compute-queue drain and before another chunk can advance the layer.
    // Copy is checked and blocking through retirement. Neither it nor any source
    // pointer may be saved/deferred by the callback. Existing callbacks retain
    // their original chunk/stage meanings. Ignored on all other run paths.
    using LayerCheckpointCopy = std::function<bool(void*, const void*, size_t, std::string&)>;
    std::function<bool(int64_t layer, int64_t done, const LayerCheckpointCopy&, std::string&)>
        on_layer_checkpoint;

    /// Checked before every chunk:''')
name = 'src/prefill/prefill.cpp'
new[name] = replace(new[name], '        if (on_chunk || on_stage_chunk) {',
    '        if (on_chunk || on_stage_chunk || (m.layer_major_active && on_layer_checkpoint)) {')
new[name] = replace(new[name], '''            const auto toc2 = Clock::now();
            if (on_stage_chunk''', '''            const auto toc2 = Clock::now();
            if (m.layer_major_active && on_layer_checkpoint) {
                if (LB < 0 || LE != LB + 1 || LE > g.n_layers) {
                    err = "prefill layer checkpoint: not a single owning layer"; return false;
                }
                const LayerCheckpointCopy capture_copy = [&](void* dst, const void* src, size_t bytes,
                                                              std::string& error) {
                    if (!bytes) return true;
                    if (!dst || !src) { error = "prefill layer checkpoint: missing copy buffer"; return false; }
                    const auto submitted = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, m.cs);
                    // Drain even after a submission error. Never free the caller's
                    // pending owners when queue retirement cannot be established.
                    const auto retired = cudaStreamSynchronize(m.cs);
                    if (retired != cudaSuccess) {
                        std::fprintf(stderr, "prefill layer checkpoint: cannot retire copy queue (%d); "
                                             "stopping with host owners live\\n", int(retired));
                        std::fflush(stderr); std::abort();
                    }
                    if (submitted != cudaSuccess) {
                        error = "prefill layer checkpoint: copy submission failed"; return false;
                    }
                    return true;
                };
                if (!on_layer_checkpoint(LB, p0 + T, capture_copy, err)) return false;
            }
            if (on_stage_chunk''')
name = 'src/program/generate.cpp'
new[name] = replace(new[name], '#include "strata/program/prompt_checkpoint.hpp"',
    '#include "strata/program/prompt_checkpoint.hpp"\n#include "strata/program/layerwise_checkpoint.hpp"')
new[name] = replace(new[name], '        int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;', '''        // Gate with the existing exact LM request; no new environment toggle.
        // Caller segmentation remains root/turn/reread only. No periodic reruns.
        const char* checkpoint_lm = std::getenv("STRATA_PREFILL_LAYER_MAJOR");
#if defined(STRATA_ENABLE_SYCL)
        const bool layerwise_periodic = checkpoint_lm && std::strcmp(checkpoint_lm, "1") == 0 &&
            !multi_gpu && o.layer_split.empty() && o.peer_device < 0 && !peer.valid() &&
            o.prompt_cache > 0 && o.prompt_cache_every > 0;
#else
        const bool layerwise_periodic = false;
        (void) checkpoint_lm;
#endif
        strata::program::LayerwiseCheckpoints layerwise;
        int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;''')
new[name] = replace(new[name], '''            if (o.prompt_cache_every > 0 && done >= pp_next_check) {''', '''            if (layerwise.active()) {
                // Target chunk and its MTP draft callback succeeded above. Merely
                // mark pending completion; published checks remain untouched.
                if (!layerwise.prompt(done, e)) return false;
                if (done >= pp_next_check)
                    pp_next_check = strata::program::layerwise_next(done, o.prompt_cache_every);
                return true;
            }
            if (o.prompt_cache_every > 0 && done >= pp_next_check) {''')
new[name] = replace(new[name], '                pp_next_check = done + o.prompt_cache_every;', '''                pp_next_check = layerwise_periodic
                    ? strata::program::layerwise_next(done, o.prompt_cache_every) : done + o.prompt_cache_every;''')
new[name] = replace(new[name], '            pp_next_check = reread_to > 0 ? INT64_MAX : resume + o.prompt_cache_every;', '''            pp_next_check = reread_to > 0 ? INT64_MAX : (layerwise_periodic
                ? strata::program::layerwise_next(resume, o.prompt_cache_every) : resume + o.prompt_cache_every);''')
new[name] = replace(new[name], '''                const auto tsp = Clock::now();
                const bool sp_ok = win ?''', '''                const bool capture = layerwise_periodic && !win && to - at > sp.chunk();
                if (capture && !layerwise.begin(checks, check_clock, o.prompt_cache, cur, req_imgs, ss, g,
                        at, to, sp.chunk(), pp_next_check, o.prompt_cache_every,
                        to == turn_at || to == root_at, err)) {
                    layerwise.discard();
                    std::printf("ERR preparing layer checkpoints: %s\\n", err.c_str());
                    return 1;
                }
                if (capture) {
                    const uint64_t run_id = layerwise.identity();
                    sp.on_layer_checkpoint = [&, run_id](int64_t l, int64_t done,
                        const strata::prefill::Prefill::LayerCheckpointCopy& copy, std::string& e) {
                        return layerwise.layer(run_id, l, done, ss, g, copy, e);
                    };
                }
                const auto tsp = Clock::now();
                const bool sp_ok = win ?''')
new[name] = replace(new[name], '''                if (trace) {
                    std::fprintf(stderr, "strata trace: read %lld tokens (%s) in %.1f ms\\n",''', '''                if (capture) sp.on_layer_checkpoint = {};
                if (trace) {
                    std::fprintf(stderr, "strata trace: read %lld tokens (%s) in %.1f ms\\n",''')
new[name] = replace(new[name], '''                if (!sp_ok) {
                    if (!stop_req.load()) {''', '''                if (!sp_ok) {
                    layerwise.discard();
                    // A simultaneous cancellation must not swallow copy/MTP errors.
                    if (!stop_req.load() || (capture && err != "cancelled")) {''')
new[name] = replace(new[name], '''                at = to;
                if ((to == turn_at || to == root_at) && !checkpoint_at(to)) {''', '''                if (capture) {
                    // run_prompt_with_mtp_experts has returned AFTER reattachment.
                    // Refill before atomic publication, including root/turn events.
                    if (!refill(err)) {
                        layerwise.discard();
                        std::printf("ERR refilling a lent slot failed: %s\\n", err.c_str()); return 1;
                    }
                    if (stop_req.load()) {
                        layerwise.discard(); cancelled = true; break;
                    }
                    if (!layerwise.commit(checks, check_clock, err)) {
                        layerwise.discard();
                        std::printf("ERR publishing layer checkpoints: %s\\n", err.c_str()); return 1;
                    }
                }
                at = to;
                if (!capture && (to == turn_at || to == root_at) && !checkpoint_at(to)) {''')
new['include/strata/program/layerwise_checkpoint.hpp'] = (OUT/'layerwise_checkpoint.hpp').read_text()
diffs=[]
for name,s in new.items():
    path = OUT/'proposed'/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s)
    diffs.extend(difflib.unified_diff(old.get(name,'').splitlines(True),s.splitlines(True),
        fromfile='a/'+name if name in old else '/dev/null',tofile='b/'+name))
(OUT/'source.diff').write_text(''.join(diffs))
ledger = {'status':'PROPOSAL_ONLY; product/build/test/numerical/runtime/performance NOT_RUN',
 'source_revision':'92acf71e as stated by Main; Git not queried',
 'inputs':{n:{'bytes':len(s.encode()),'sha256':sha(s.encode())} for n,s in old.items()},
 'outputs':{n:{'bytes':len(s.encode()),'sha256':sha(s.encode())} for n,s in new.items()},
 'diff_sha256':sha((OUT/'source.diff').read_bytes()),
 'commands_executed':['read-only source/report reads','python3 generate_proposal.py (generation/hash only)'],
 'forbidden_actions_performed':[]}
ledger['evidence_inputs'] = {}
for name in ('AGENTS.md','docs/ARTIFACT_RETENTION.md','.claude/skills/strata-upstream-sync/SKILL.md',
 'bench/results/2026-10-15-pr6-recovery-audit/research-index.json',
 'bench/results/2026-10-15-pr6-recovery-audit/main-review-round608.ja.md',
 'bench/results/2026-10-15-pr6-recovery-audit/research-agent-cache-settings-round608-v1.txt',
 'bench/results/2026-10-15-pr6-recovery-audit/research-checkpoint-cost-round609-v1.txt',
 'bench/results/2026-10-15-pr6-recovery-audit/layer-major-checkpoint-coherence-v1/implementation.ja.md',
 'bench/results/2026-10-15-pr6-recovery-audit/layer-major-checkpoint-coherence-v1/source.diff',
 'src/prefill/layer_major.inc','src/core/conversation_state.cpp'):
    raw = (ROOT/name).read_bytes()
    ledger['evidence_inputs'][name] = {'bytes':len(raw),'sha256':sha(raw)}
ledger['artifacts'] = {}
for path in sorted(OUT.rglob('*')):
    if path.is_file() and path.name != 'sha256-ledger.json':
        raw = path.read_bytes()
        ledger['artifacts'][str(path.relative_to(OUT))] = {'bytes':len(raw),'sha256':sha(raw)}
ledger['retention'] = {'owner':'Main','consumer':'source review/integration and later CPU/native validation',
    'budget_bytes':2097152,'next_review':'Main source review or integration/rejection','deletions':[]}
ledger['validation'] = dict.fromkeys(('product_apply','build','CPU_fixture','GPU','RESTORE',
    'numerical','performance','physical_256K'),'NOT_RUN')
(OUT/'sha256-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
