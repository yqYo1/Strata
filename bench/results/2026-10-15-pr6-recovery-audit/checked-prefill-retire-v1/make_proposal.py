from pathlib import Path
import difflib, hashlib, json
root=Path.cwd(); out=root/'bench/results/2026-10-15-pr6-recovery-audit/checked-prefill-retire-v1'
changes={}; originals={}
def edit(p, old, new):
 s=changes.get(p)
 if s is None: s=(root/p).read_text(); originals[p]=s
 assert old in s,p
 changes[p]=s.replace(old,new,1)
edit('include/strata/sycl/runtime.hpp','  void *data() const { return data_; }','  // Opt-in explicit release: caller retains this owner until successful return.\n  // On a free exception the caller must fail-stop; never retry an unknown free.\n  void release_checked();\n  void *data() const { return data_; }')
edit('src/sycl/runtime.cpp','void Allocation::release() noexcept {','void Allocation::release_checked() {\n  if (!data_) return;\n  runtime_->wait(); // may throw: owner and pointer are still intact\n  sycl::free(data_, runtime_->context()); // may throw: caller must fail-stop\n  data_ = nullptr;\n  bytes_ = 0;\n}\n\nvoid Allocation::release() noexcept {')
edit('include/strata/sycl/compat/cuda_runtime.h','cudaError_t cudaFreeHost(void *);','cudaError_t cudaFreeHost(void *);\n// Scratch retirement only. Success reports exact registry allocation bytes.\n// Failure preserves registry ownership; do not retry or unwind after failure.\ncudaError_t strataFreeChecked(void *, bool host, size_t *released_bytes);')
edit('src/sycl/compat.cpp','cudaError_t cudaFree(void *p) { return release(p, MemoryKind::Device); }','''cudaError_t strataFreeChecked(void *p, bool host, size_t *released_bytes) {
  if (released_bytes) *released_bytes = 0;
  if (!p) return cudaSuccess;
  return attempt([&] {
    require(released_bytes, "checked free needs byte receipt");
    wait_all(); // no erase/move before all completion checks succeed
    std::lock_guard lock(registry_mutex);
    auto it = allocations.find(p);
    require(it != allocations.end() && it->second.kind ==
        (host ? MemoryKind::Host : MemoryKind::Device), "checked free owner/kind mismatch");
    const size_t bytes = it->second.allocation->size();
    it->second.allocation->release_checked();
    allocations.erase(it); // destructor sees null data and performs no wait/free
    *released_bytes = bytes;
  });
}
cudaError_t cudaFree(void *p) { return release(p, MemoryKind::Device); }''')
edit('include/strata/prefill/prefill.hpp','    void reset();','''    void reset();

    // Serial caller only; all callbacks and cache-tail refill/publication must
    // have finished. Success leaves a fresh empty Impl, preserving callbacks,
    // stage configuration and statistics. Any uncertainty exits without unwind.
    void retire_scratch_checked(const char* phase) noexcept;
    bool scratch_ready() const;''')
edit('src/prefill/prefill.cpp','Prefill::~Prefill() { release(); }','''Prefill::~Prefill() {
#if defined(STRATA_ENABLE_SYCL)
    const char* flag = std::getenv("STRATA_PREFILL_RETIRE_SCRATCH");
    if (flag && std::strcmp(flag, "1") == 0) {
        retire_scratch_checked("owner-teardown"); return;
    }
#endif
    release();
}

bool Prefill::scratch_ready() const { return impl_ && impl_->g && impl_->R; }

void Prefill::retire_scratch_checked(const char* phase) noexcept {
    auto fatal = [&](const char* operation) noexcept {
        std::fprintf(stderr, "strata prefill scratch: phase=%s failed=%s; retaining remaining owners; no unwind/retry\\n",
                     phase, operation);
        std::fflush(stderr); std::_Exit(EXIT_FAILURE);
    };
#if !defined(STRATA_ENABLE_SYCL)
    fatal("requires-SYCL");
#else
    try {
        if (!impl_ || !impl_->g) return;
        Impl& m = *impl_;
        if (m.pp || next_ || hand_in_ || stage_lb_ != 0 ||
            stage_le_ != m.g->n_layers || m.copy != m.cs || core::peer_portable() ||
            m.layer_major_active || m.layer_major_audit || m.layer26_phase_audit)
            fatal("unsupported-owner-topology");
        // Allocate replacement before the first irreversible release.
        auto fresh = std::make_unique<Impl>();
        // No run-local issuer/future survives the caller's completed run. Join
        // persistent host stager threads before checking every device queue.
        if (m.stager) {
            if (m.stager->buf.size() != size_t(m.stager->kRing) ||
                m.stager->pinned.size() != size_t(m.stager->kRing) ||
                m.stager->dma_done.size() != size_t(m.stager->kRing))
                fatal("partial-stager-initialization");
            m.stager->finish();
            { std::lock_guard<std::mutex> lock(m.stager->mu); m.stager->quit = true; }
            m.stager->cv.notify_all();
            for (auto& t : m.stager->threads) if (t.joinable()) t.join();
            m.stager->threads.clear();
        }
        if (cudaDeviceSynchronize() != cudaSuccess) fatal("global-drain");
        size_t free0 = 0, total0 = 0;
        if (cudaMemGetInfo(&free0, &total0) != cudaSuccess || free0 > total0)
            fatal("before-memory-query");
        uint64_t device_bytes = 0, host_bytes = 0, device_allocations = 0;
        auto event = [&](cudaEvent_t& p) {
            if (!p) return;
            if (cudaEventDestroy(p) != cudaSuccess) fatal("event-destroy");
            p = nullptr;
        };
        auto memory = [&](auto*& p, bool host) {
            if (!p) return;
            size_t bytes = 0;
            if (strataFreeChecked(p, host, &bytes) != cudaSuccess) fatal("checked-free");
            p = nullptr; // only after known success; no destructor can retry
            if (host) host_bytes += bytes;
            else { device_bytes += bytes; ++device_allocations; }
        };
        // Free all GPU allocations through the checked registry path. Borrowed
        // cache spans never appear in owned and are never freed here.
        for (auto& p : m.owned) memory(p, false);
        m.owned.clear();
        for (int i = 0; i < RING_MAX; ++i) { event(m.copied[i]); event(m.used[i]); }
        for (int b = 0; b < 2; ++b) {
            memory(m.hand[b], true); event(m.ple_copied[b]);
            if (m.ple_pageable[b].empty()) memory(m.ple_emb_host[b], true);
        }
        memory(m.grp_host, true);
        if (m.stager) {
            auto& s = *m.stager;
            for (int i = 0; i < s.kRing; ++i) {
                event(s.dma_done[i]);
                if (s.pinned[i]) memory(s.buf[i], true);
            }
        }
        // SYCL Gemm uses init_external; mmq::Context's destructor is default.
        // Stager's destructor now sees no threads, events or pinned buffers.
        impl_.swap(fresh);
        fresh.reset();
        size_t free1 = 0, total1 = 0;
        if (cudaMemGetInfo(&free1, &total1) != cudaSuccess || free1 > total1 || total1 != total0)
            fatal("after-memory-query");
        std::fprintf(stderr, "strata prefill scratch: phase=%s device_freed_bytes=%llu host_freed_bytes=%llu "
                     "device_allocations=%llu free_before=%llu free_after=%llu total=%llu\\n", phase,
                     (unsigned long long)device_bytes, (unsigned long long)host_bytes,
                     (unsigned long long)device_allocations, (unsigned long long)free0,
                     (unsigned long long)free1, (unsigned long long)total1);
        std::fflush(stderr);
    } catch (...) { fatal("exception"); }
#endif
}''')
# checked startup/cancel reset: preserve default, retire is safe even for partial init.
edit('src/prefill/prefill.cpp','void Prefill::reset() {\n    release();','''void Prefill::reset() {
#if defined(STRATA_ENABLE_SYCL)
    const char* flag = std::getenv("STRATA_PREFILL_RETIRE_SCRATCH");
    if (flag && std::strcmp(flag, "1") == 0) {
        retire_scratch_checked("explicit-reset"); stats_ = PrefillStats{}; return;
    }
#endif
    release();''')
# Global runtime opt-in admission after parsed options and actual multi-GPU decisions,
# before Prefill startup. It deliberately admits only static cache ownership.
edit('src/program/generate.cpp','    if (o.serve) {\n        if (o.spec < 2','''    const char* retire_scratch_flag = std::getenv("STRATA_PREFILL_RETIRE_SCRATCH");
    const bool retire_scratch = retire_scratch_flag && std::strcmp(retire_scratch_flag, "1") == 0;
    if (retire_scratch) {
#if !defined(STRATA_ENABLE_SYCL)
        std::fprintf(stderr, "strata: scratch retirement requires SYCL\\n"); return 2;
#else
        if (multi_gpu || !o.layer_split.empty() || o.peer_device >= 0 || peer.valid() ||
            std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(), [](auto n) { return n > 0; }) ||
            (o.adapt_every > 0 && o.adapt_swaps > 0)) {
            std::fprintf(stderr, "strata: scratch retirement requires single GPU, no split/peer/remote, static cache\\n");
            return 2;
        }
#endif
    }
    if (o.serve) {
        if (o.spec < 2''')
# Lazy init must precede lend early return (own buffers) and any relayout (borrow).
edit('src/program/generate.cpp','''            auto lend = [&](int64_t tokens, std::string& e) -> bool {
                if (pf_parts.empty()) return true;''','''            auto lend = [&](int64_t tokens, std::string& e) -> bool {
                if (retire_scratch && !sp.scratch_ready()) {
                    // Keep configured maximum geometry for later segments of
                    // this same prompt, then apply the exact request loan below.
                    // RESTORE/read_windows never calls lend and allocates nothing.
                    try {
                        if (!sp.init(wt, g, ss, srcp, &xcache, host_res.data(), o.prefill_chunk,
                                     main_cs, e, borrow, borrow_bytes)) {
                            std::fprintf(stderr, "strata: scratch recreate failed: %s\\n", e.c_str());
                            std::fflush(stderr); std::_Exit(EXIT_FAILURE);
                        }
                    } catch (...) { std::_Exit(EXIT_FAILURE); }
                    // Fresh init used the original maximum loan geometry.
                    for (auto& p : pf_parts) if (p.sp == &sp) p.first_now = p.first;
                    mem_mark("recreated serve prompt scratch");
                }
                if (pf_parts.empty()) return true;''')
edit('src/program/generate.cpp','''                if (!sp_ok) {
                    if (!stop_req.load()) {''','''                if (!sp_ok) {
                    // Failed/cancelled run may leave queued readers. No refill
                    // can overwrite borrowed scratch before a checked drain.
                    if (retire_scratch && cudaDeviceSynchronize() != cudaSuccess) {
                        std::fprintf(stderr, "strata: failed prompt drain; retaining owners\\n");
                        std::fflush(stderr); std::_Exit(EXIT_FAILURE);
                    }
                    if (!stop_req.load()) {''')
edit('src/program/generate.cpp','''            tr("prompt done (slots refilled)");''','''            if (retire_scratch) {
                // Whole prompt loop completed, including all chunk callbacks,
                // checkpoints and final refill. A cancelled prompt is teardown,
                // never represented as successful whole-prompt retirement.
                for (const auto& p : pf_parts) if (!p.lent.empty()) std::_Exit(EXIT_FAILURE);
                // res_upload's historical default ignores its copy status.
                // Opt-in must publish and check the actual residency table.
                if (d_res && cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t),
                                        cudaMemcpyHostToDevice) != cudaSuccess) std::_Exit(EXIT_FAILURE);
                sp.retire_scratch_checked(cancelled ? "cancelled-prompt-teardown" : "whole-prompt-before-decode");
            }
            tr("prompt done (slots refilled)");''')
edit('src/program/generate.cpp','''        prefill.reset();
        mem_mark("releasing the standalone prompt buffers");''','''        if (retire_scratch) {
            if (d_res && cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t),
                                    cudaMemcpyHostToDevice) != cudaSuccess) std::_Exit(EXIT_FAILURE);
            prefill.retire_scratch_checked("standalone-whole-prompt-before-decode");
        } else prefill.reset();
        mem_mark("releasing the standalone prompt buffers");''')
manifest=[]; diff=[]
for p,s in changes.items():
 dest=out/'source'/p; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(s)
 diff += difflib.unified_diff(originals[p].splitlines(True),s.splitlines(True),fromfile='a/'+p,tofile='b/'+p)
 manifest.append({'path':p,'base_sha256':hashlib.sha256(originals[p].encode()).hexdigest(),'candidate_sha256':hashlib.sha256(s.encode()).hexdigest()})
(out/'source.diff').write_text(''.join(diff)); (out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
