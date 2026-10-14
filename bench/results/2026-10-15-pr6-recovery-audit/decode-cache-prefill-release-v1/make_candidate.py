"""Materialize an unapplied candidate from the immutable base/ snapshot only.

This script does not build, test, invoke a driver, or write outside its directory.
"""
from pathlib import Path
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
sources = {str(p.relative_to(HERE / 'base')): p.read_text()
           for p in (HERE / 'base').rglob('*') if p.is_file()}
original = sources.copy()


def replace(path, before, after):
    if sources[path].count(before) != 1:
        raise ValueError(f'nonunique source context: {path}: {before[:100]!r}')
    sources[path] = sources[path].replace(before, after, 1)


p = 'include/strata/core/expert_cache.hpp'
replace(p, '    void close();', '''    void close();
#ifdef STRATA_ENABLE_SYCL
    /// Serialized issuer only, after ALL external readers/graphs have retired.
    /// On failure the caller must fail-stop before any owner is destroyed.
    bool close_checked(std::string& err);
    /// Reopen a retired, empty object without submitting initialization work.
    bool reopen_sized_uninitialized(const std::vector<int64_t>& slot_bytes,
                                    int64_t n_layers, int64_t n_expert, int64_t saved_blob_bytes,
                                    std::string& err);
    int64_t allocation_blob_bytes() const { return blob_; }
#endif''')
p = 'src/core/expert_cache.cpp'
replace(p, 'void ExpertCache::close() {', '''#ifdef STRATA_ENABLE_SYCL
bool ExpertCache::close_checked(std::string& err) {
    // The outer lease has checked reader retirement. Keep failed owners intact;
    // do not let the ordinary unchecked destructor retry after a driver error.
    if (blocking_staging_) {
        const auto e = cudaFreeHost(blocking_staging_);
        if (e != cudaSuccess) { err = "decode cache: free staging: " + std::string(cudaGetErrorString(e)); return false; }
        blocking_staging_ = nullptr; blocking_staging_bytes_ = 0;
    }
    if (base_) {
        const auto e = cudaFree(base_);
        if (e != cudaSuccess) { err = "decode cache: free arena: " + std::string(cudaGetErrorString(e)); return false; }
        base_ = nullptr;
    }
    close(); // pointers are null: metadata only, no unchecked device operation
    return true;
}

bool ExpertCache::reopen_sized_uninitialized(const std::vector<int64_t>& sizes,
                                             int64_t layers, int64_t experts, int64_t saved_blob,
                                             std::string& err) {
    if (valid() || slots_ || !off_.empty() || sizes.empty() || saved_blob <= 0 ||
        uint64_t(saved_blob) > uint64_t(INT64_MAX) - 255) {
        err = "decode cache: reopen requires a retired empty object and fixed slot sizes"; return false;
    }
    std::vector<uint64_t> offsets(sizes.size() + 1, 0);
    const uint64_t largest_slot = (uint64_t(saved_blob) + 255) / 256 * 256;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] <= 0 || uint64_t(sizes[i]) > uint64_t(INT64_MAX) - 255) {
            err = "decode cache: invalid fixed slot size"; return false;
        }
        const uint64_t bytes = (uint64_t(sizes[i]) + 255) / 256 * 256;
        if (offsets[i] > uint64_t(INT64_MAX) - bytes) {
            err = "decode cache: fixed slot size overflow"; return false;
        }
        offsets[i + 1] = offsets[i] + bytes;
        if (bytes > largest_slot) { err = "decode cache: slot exceeds original blob layout"; return false; }
    }
    // Match open_sized's byte allocation and slot layout, without a memset.
    if (!open_uninitialized(int64_t(offsets.back()), layers, experts, 1, err)) return false;
    slots_ = int64_t(sizes.size()); blob_ = saved_blob; off_ = std::move(offsets);
    // Preserve normal bounded blocking-fill support, with checked retirement
    // of the one-byte bootstrap slab. On failure the outer lease exits before
    // freeing either the original slab or the new raw allocation.
    if (size_t(saved_blob) > blocking_staging_bytes_) {
        void* next = nullptr;
        if (cudaHostAlloc(&next, size_t(saved_blob), cudaHostAllocDefault) != cudaSuccess) {
            err = "decode cache: restored blocking staging allocation failed"; return false;
        }
        if (blocking_staging_ && cudaFreeHost(blocking_staging_) != cudaSuccess) {
            err = "decode cache: bootstrap staging retirement failed"; return false;
        }
        blocking_staging_ = static_cast<uint8_t*>(next); blocking_staging_bytes_ = size_t(saved_blob);
    }
    layer_next_.assign(size_t(layers), 0);
    for (int64_t l = 0; l < layers; ++l) {
        int64_t lo = 0, hi = 0; layer_slot_range(l, lo, hi); layer_next_[size_t(l)] = int32_t(lo);
    }
    return true;
}
#endif

void ExpertCache::close() {''')

p = 'include/strata/core/expert_source.hpp'
replace(p, '    std::shared_ptr<NativeHitGraphs> native_hit_graphs;', '''    std::shared_ptr<NativeHitGraphs> native_hit_graphs;
    bool decode_cache_ready = true; // false only inside the opt-in serialized prompt lease''')
replace(p, 'bool expert_hit_prepare_graphs(ExpertDispatch& d, void* stream, std::string& err);', '''bool expert_hit_prepare_graphs(ExpertDispatch& d, void* stream, std::string& err,
                               bool failstop_on_error = false);
/// Checked drain and destruction, including the mapped plan. Serialized issuer only.
bool expert_hit_retire_graphs(ExpertDispatch& d, void* stream, std::string& err);''')
p = 'src/core/expert_source.cpp'
replace(p, 'bool expert_hit_prepare_graphs(ExpertDispatch& d, void* stream, std::string& err) {', '''bool expert_hit_retire_graphs(ExpertDispatch& d, void* stream, std::string& err) {
    if (stream == nullptr || cudaStreamSynchronize((cudaStream_t) stream) != cudaSuccess) {
        err = "decode cache: native hit queue retirement failed"; return false;
    }
    auto& p = d.native_hit_graphs;
    if (!p) return true;
    if (p.use_count() != 1) { err = "decode cache: native hit graph has another owner"; return false; }
    for (auto& exec : p->owned) {
        if (exec && cudaGraphExecDestroy(exec) != cudaSuccess) {
            err = "decode cache: native hit executable retirement failed"; return false;
        }
        exec = nullptr;
    }
    p->owned.clear(); p->layers.clear();
    if (p->host_plan && cudaFreeHost(p->host_plan) != cudaSuccess) {
        err = "decode cache: native hit host plan retirement failed"; return false;
    }
    p->host_plan = nullptr; p->ptr = nullptr; p->meta = nullptr; p->dst = nullptr;
    p.reset();
    return true;
}

bool expert_hit_prepare_graphs(ExpertDispatch& d, void* stream, std::string& err,
                               bool failstop_on_error) {''')
replace(p, '    auto graphs = std::make_shared<NativeHitGraphs>();', '''    auto graphs = std::make_shared<NativeHitGraphs>();
    struct CaptureLease {
        bool strict, complete = false;
        ~CaptureLease() noexcept {
            if (strict && !complete) {
                std::fprintf(stderr, "decode cache: native hit recapture failed; owners retained until exit\\n");
                std::fflush(stderr); std::_Exit(EXIT_FAILURE);
            }
        }
    } capture_lease{failstop_on_error};''')
replace(p, '    d.native_hit_graphs = std::move(graphs);\n    return true;', '''    d.native_hit_graphs = std::move(graphs);
    capture_lease.complete = true;
    return true;''')
# Refuse every decode callback before it can consult a map or cache pointer.
for prefix in [
    'void expert_pool_dispatch(void* user,',
    'void expert_pool_dispatch_multi(ExpertDispatch& d,',
    'void expert_hit_run(void* user,',
]:
    at = sources[p].index(prefix)
    failed = sources[p].index('    if (d.failed) return;', at)
    guard = '''#ifdef STRATA_ENABLE_SYCL
    if (!d.decode_cache_ready) {
        d.failed = true; d.fail = "decode cache: prompt lease has not restored decode readiness";
        d.fail_layer = d.layers; return;
    }
#endif
'''
    sources[p] = sources[p][:failed] + guard + sources[p][failed:]

p = 'include/strata/core/verify.hpp'
replace(p, '    bool wait_commit(std::string& err);', '''    bool wait_commit(std::string& err);
#ifdef STRATA_ENABLE_SYCL
    /// Only the serialized request issuer may call these between completed windows.
    /// Retains arena/final_R/streams/events and cache-independent commit graphs.
    bool retire_cache_graphs(std::string& err);
    bool restore_cache_hits_and_recapture(const VerifyHits& hits, std::string& err);
#endif''')
replace(p, '    bool overlap_gpu_ = true, work_launched_ = false;', '''    bool overlap_gpu_ = true, work_launched_ = false;
    bool cache_hits_detached_ = false, cache_graph_recapture_ = false;
    bool retired_cache_windows_[9] = {};''')
p = 'src/core/verify.cpp'
replace(p, 'bool Verifier::init(const WeightTable& wt,', '''#ifdef STRATA_ENABLE_SYCL
bool Verifier::retire_cache_graphs(std::string& err) {
    const OnDevice on_device(device_);
    const auto phase = diag_window_.read().phase;
    if (!cs_ || next_ || released_.load() || cache_hits_detached_ ||
        (phase != VerifyPhase::inactive && phase != VerifyPhase::completed)) {
        err = "decode cache: verifier is not at a healthy single-stage idle boundary"; return false;
    }
    if (!wait_commit(err) || cudaStreamSynchronize(cs_) != cudaSuccess ||
        (copy_ && cudaStreamSynchronize(copy_) != cudaSuccess)) {
        if (err.empty()) err = "decode cache: verifier queue retirement failed";
        return false;
    }
    for (int t = 0; t < 9; ++t) {
        retired_cache_windows_[t] = exec_[t] || !segments_[t].empty() || !work_[t].empty();
        auto destroy = [&](cudaGraphExec_t& e) {
            if (e && cudaGraphExecDestroy(e) != cudaSuccess) return false;
            e = nullptr; return true;
        };
        if (!destroy(exec_[t])) { err = "decode cache: verifier executable retirement failed"; return false; }
        for (auto& e : segments_[t])
            if (!destroy(e)) { err = "decode cache: verifier segment retirement failed"; return false; }
        for (auto& e : work_[t])
            if (!destroy(e)) { err = "decode cache: verifier work retirement failed"; return false; }
        segments_[t].clear(); work_[t].clear();
    }
    // SYCL's device_plan_ is disabled at init. Preserve the commit-only graph,
    // arena and final_R borrowed by normal/coupled MTP graphs.
    if (device_plan_ || slot_off_d_) { err = "decode cache: unsupported verifier device plan"; return false; }
    // The completed host/device plan can still contain old slot addresses.
    // No window reader remains; clear both copies before retiring the cache.
    std::memset(h_plan_, 0, size_t(2 * plan_i32_) * sizeof(int32_t));
    if (cudaMemsetAsync(plan_, 0, size_t(2 * (plan_i32_ + 16)) * sizeof(int32_t), cs_) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "decode cache: old verifier plan pointer retirement failed"; return false;
    }
    hits_ = {}; work_launched_ = false; work_status_ = cudaSuccess;
    last_t_ = 0; cache_hits_detached_ = true;
    return true;
}

bool Verifier::restore_cache_hits_and_recapture(const VerifyHits& hits, std::string& err) {
    const OnDevice on_device(device_);
    if (!cache_hits_detached_ || !hits.d_res || !hits.cache_base || hits.n_slots <= 0 || hits.blob <= 0) {
        err = "decode cache: verifier restore requires detached graphs and complete new hits"; return false;
    }
    hits_ = hits;
    cache_graph_recapture_ = true;
    for (int t = 1; t <= max_t_; ++t)
        if (retired_cache_windows_[t] && !capture(t, err)) return false;
    cache_graph_recapture_ = false; cache_hits_detached_ = false;
    for (auto& t : retired_cache_windows_) t = false;
    return true;
}
#endif

bool Verifier::init(const WeightTable& wt,''')
replace(p, '    if (!record_window(T, cs_, err) || !end_segment(T, err)) {', '''    if (!record_window(T, cs_, err) || !end_segment(T, err)) {
        if (cache_graph_recapture_) {
            std::fprintf(stderr, "decode cache: verifier recapture failed; owners retained until exit\\n");
            std::fflush(stderr); std::_Exit(EXIT_FAILURE);
        }''')
replace(p, '    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }', '''#ifdef STRATA_ENABLE_SYCL
    if (cache_hits_detached_) { err = "verify: decode cache prompt lease is detached"; return false; }
#endif
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }''')

p = 'include/strata/prefill/prefill.hpp'
replace(p, '    int64_t chunk() const;', '''    int64_t chunk() const;
#ifdef STRATA_ENABLE_SYCL
    /// Own-buffer, long layer-major prompt only. These never close the cache;
    /// the caller owns checked graph retirement and a complete restore lease.
    bool detach_decode_cache(int64_t tokens, std::string& err);
    bool rebind_decode_cache(const core::ExpertCache* cache, const int32_t* host_res, std::string& err);
#endif''')
p = 'src/prefill/prefill.cpp'
replace(p, '    bool borrowed = false;', '''    bool borrowed = false;
#ifdef STRATA_ENABLE_SYCL
    bool decode_cache_detached = false;
#endif''')
p = 'src/prefill/layer_major.inc'
replace(p, '// Included inside strata::prefill after the current chunk implementation\'s\n// helpers. This is orchestration only; run_impl owns GDN/QSA/PLE/expert math.\n', '''// Included inside strata::prefill after the current chunk implementation's
// helpers. This is orchestration only; run_impl owns GDN/QSA/PLE/expert math.
#ifdef STRATA_ENABLE_SYCL
bool Prefill::detach_decode_cache(int64_t tokens, std::string& err) {
    const char* layer_major = std::getenv("STRATA_PREFILL_LAYER_MAJOR");
    if (!impl_ || !impl_->g || !impl_->ss || !layer_major || std::strcmp(layer_major, "1") || tokens <= impl_->T ||
        impl_->borrowed || impl_->decode_cache_detached || impl_->layer_major_active ||
        stage_lb_ != 0 || stage_le_ != impl_->g->n_layers || next_ || hand_in_ || embd_rows ||
        impl_->pp || !impl_->compact_moe_fp16 || !impl_->src || !impl_->cache || !impl_->host_res ||
        impl_->copy != impl_->cs || !impl_->stager ||
        impl_->stager->active.load(std::memory_order_acquire)) {
        err = "decode cache: prefill requires an idle long layer-major segment and owned buffers"; return false;
    }
    auto& m = *impl_;
    const uint64_t head = m.stager->head.load(std::memory_order_acquire);
    if ((head & 0xffff) < ((head >> 16) & 0xffff)) {
        err = "decode cache: prefill still has claimable host staging jobs"; return false;
    }
    const core::OnDevice on_device(m.device);
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = "decode cache: prefill entry queue retirement failed"; return false;
    }
    // A closed cache object remains alive, but neither it nor its old map may
    // reach ordinary run_impl before layer-major installs the temporary cache.
    m.cache = nullptr; m.host_res = nullptr; m.decode_cache_detached = true;
    return true;
}

bool Prefill::rebind_decode_cache(const core::ExpertCache* cache, const int32_t* res, std::string& err) {
    if (!impl_ || !impl_->decode_cache_detached || impl_->layer_major_active ||
        !cache || !cache->valid() || !res) {
        err = "decode cache: prefill rebind requires a complete restored cache"; return false;
    }
    const core::OnDevice on_device(impl_->device);
    if (cudaStreamSynchronize(impl_->cs) != cudaSuccess) {
        err = "decode cache: prefill exit queue retirement failed"; return false;
    }
    impl_->cache = cache; impl_->host_res = res; impl_->decode_cache_detached = false;
    return true;
}
#endif
''')
replace(p, '    const char* requested = std::getenv("STRATA_PREFILL_LAYER_MAJOR");', '''    const char* requested = std::getenv("STRATA_PREFILL_LAYER_MAJOR");
#ifdef STRATA_ENABLE_SYCL
    if (impl_ && impl_->decode_cache_detached &&
        (!requested || std::strcmp(requested, "1") || n <= impl_->T)) {
        err = "decode cache: detached prefill cannot fall through to the ordinary chunk path"; return false;
    }
#endif''')

# The main wrapper is isolated in a header, keeping integration with the newer
# MTP/selected-head changes to two call sites and one serialized local lambda.
p = 'src/program/generate.cpp'
replace(p, '#include "strata/program/prefill_logits.hpp"', '''#include "strata/program/prefill_logits.hpp"
#include "strata/program/decode_cache_prefill.hpp"''')
replace(p, '    strata::core::set_coupled_draft(o.coupled_draft);', '''#ifdef STRATA_ENABLE_SYCL
    const char* release_decode_env = std::getenv("STRATA_PREFILL_RELEASE_DECODE_CACHE");
    const bool release_decode_cache = release_decode_env && std::strcmp(release_decode_env, "1") == 0;
    if (release_decode_cache) {
        const char* lm = std::getenv("STRATA_PREFILL_LAYER_MAJOR");
        const char* hit = std::getenv("STRATA_SYCL_NATIVE_HIT_GRAPH");
        const bool remote = o.expert_cache_remote[0] || o.expert_cache_remote[1] || o.expert_cache_remote[2];
        if (!lm || std::strcmp(lm, "1") || o.expert_cache <= 0 || o.expert_profile.empty() || o.spec < 2 ||
            o.native_preset.empty() || !o.no_prefill_borrow || o.prefill_chunk <= 0 || o.prefill_auto ||
            o.no_pool || o.no_capture || !o.no_token_graph || o.expert_cache_cpu_order ||
            !o.layer_split.empty() || !o.split_device.empty() || o.peer_device >= 0 || remote || o.vision ||
            (o.adapt_every > 0 && o.adapt_swaps > 0) || o.gpu_stages || o.graph_only || o.gpu_only_full ||
            (hit && std::strcmp(hit, "0") == 0)) {
            std::fprintf(stderr, "strata generate: decode-cache prefill release requires layer-major=1, "
                                 "native single GPU, explicit fixed --expert-cache/profile/--prefill, "
                                 "--no-prefill-borrow, --spec >=2, native hit graphs, no peers/remotes/split/adaptive/vision\\n");
            return 2;
        }
    }
#endif
    strata::core::set_coupled_draft(o.coupled_draft);''')
replace(p, '    // ================================ plan v0.3 P8: THE PERSISTENT ENGINE (--serve) ================================', '''    auto run_prompt_prefill = [&](strata::prefill::Prefill& pf, strata::core::Verifier* verifier,
                                  const int64_t* ids, int64_t count, int64_t pos, std::string& why) -> bool {
#ifdef STRATA_ENABLE_SYCL
        // Default OFF: no new profile scan, allocation, queue wait or driver query.
        if (release_decode_cache && count > pf.chunk()) {
            if (drive.d.lookahead || drive.d.remote_count || !drive.d.native_hit_graphs ||
                host_res.empty() || !d_res || !srcp || auto_cache || !native_pack || multi_gpu || tgraph.captured) {
                why = "decode cache: incomplete fixed-profile single-GPU hit setup (disable STRATA_LOOKAHEAD)";
                return false;
            }
            return strata::program::decode_cache_prefill::run(
                pf, verifier, xcache, drive.d, thits, host_res, d_res, *srcp, profile,
                g.n_layers, g.n_expert, ids, count, pos, main_stream, why);
        }
#endif
        return pf.run(ids, count, pos, why);
    };

    // ================================ plan v0.3 P8: THE PERSISTENT ENGINE (--serve) ================================''')
# Main integrated the MTP-only wrapper during candidate authoring. Preserve its
# ownership/error/telemetry body and pass the actual prompt call as a template
# closure: MTP lease outside, decode-cache lease inside. No std::function alloc.
replace(p, 'bool run_prompt_with_mtp_experts(strata::prefill::Prefill& prompt, strata::core::MtpDrafter& mtp,',
        'template<class RunPrompt>\nbool run_prompt_with_mtp_experts(strata::prefill::Prefill& prompt, strata::core::MtpDrafter& mtp,')
replace(p, '                                 std::string& err) {\n    const char* release',
        '                                 std::string& err, RunPrompt&& run_prompt) {\n    (void) tokens; (void) pos0; // The closure owns the actual prompt invocation.\n    const char* release')
replace(p, '        return prompt.run(tokens, n, pos0, err);',
        '        return run_prompt();')
replace(p, '    if (prompt.chunk() <= 0 || n <= prompt.chunk()) return prompt.run(tokens, n, pos0, err);',
        '    if (prompt.chunk() <= 0 || n <= prompt.chunk()) return run_prompt();')
replace(p, '        ok = prompt.run(tokens, n, pos0, err);', '        ok = run_prompt();')
replace(p, '                    ids.data() + at, to - at, at, err);', '''                    ids.data() + at, to - at, at, err,
                    [&] { return run_prompt_prefill(sp, &ver, ids.data() + at, to - at, at, err); });''')
replace(p, '                o.tokens.data(), n_batched, 0, err)) {', '''                o.tokens.data(), n_batched, 0, err,
                [&] { return run_prompt_prefill(prefill, nullptr, o.tokens.data(), n_batched, 0, err); })) {''')

# Additional files are authored separately and merged into the diff below.
for p in (HERE / 'new').rglob('*'):
    if p.is_file(): sources[str(p.relative_to(HERE / 'new'))] = p.read_text()
hashes = {}
patch = []
for p, text in sources.items():
    dst = HERE / 'proposed' / p
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    hashes[p] = hashlib.sha256(text.encode()).hexdigest()
    old = original.get(p, '')
    patch.extend(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
                 fromfile='a/' + p if p in original else '/dev/null', tofile='b/' + p, n=5))
(HERE / 'source.diff').write_text(''.join(patch))
(HERE / 'proposed-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(f'Materialized {len(sources)} proposed files and incremental source.diff only within {HERE}')
