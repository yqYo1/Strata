// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/prefill.cpp - see include/strata/prefill/prefill.hpp.
#include "strata/prefill/prefill.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/device.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/phase_serve_cycle.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/kernels.hpp"

#include "strata/core/gpu.hpp"

#include "strata/prefill/publication.hpp"
#include "strata/prefill/compact_moe.hpp"
#include "strata/prefill/workspace_layout.hpp"
#include "strata/prefill/layer_major.hpp"
#include "strata/prefill/issuer_timing.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strata::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
// plan v0.3 P6: staging holds the largest blob of the pack (a native pack's blobs differ per layer)
inline int64_t MAXBLOB() { return (int64_t) strata::kernels::cpu::expert_layout().max_blob; }
constexpr int STAGE = 8;           // host->device expert staging ring (chunks below STREAM_ALL_MIN)
// Step 3: from this chunk size on, every non-resident expert of every layer streams in a fixed order through a
// RING_MAX-slot ring (nearly all 512 are routed at such a chunk), so the copy engine keeps working through the
// attention halves instead of waiting for each layer's routing.
constexpr int RING_MAX = 1024;          // the arrays; the ring itself is ring_slots()
// STRATA_TEST_PAGEABLE=1 (upstream cfe09ab5) makes every pinned host allocation with a pageable fallback in
// Prefill::init fail, so the fallbacks (what a container's memlock limit forces) run on a PC that can pin - for ASan /
// MALLOC_CHECK_=3 runs.
inline bool force_pageable() {
    static const bool v = [] { const char* e = std::getenv("STRATA_TEST_PAGEABLE"); return e != nullptr && e[0] == '1'; }();
    return v;
}
// A host table the GPU reads or writes every layer (the routing ids, the row maps, the group bounds): USM host
// memory that a kernel reads and writes in place (to_device / from_device: copy_i32_from_mapped), so these small copies
// never queue behind the streamed experts on the copy engine (upstream 1b3bb8f8).  A copy from or to pageable memory
// went through that engine: on the RTX 4070 the GPU waited 1.25 s of a 4K prompt at the routing syncs.  The host
// rewrites a table only after the next layer's sync, when the kernel that read it is done.  Pageable, with copies, when
// the allocation fails or STRATA_TEST_PAGEABLE=1.  resize keeps no contents.
template <class T> class HostVec {
public:
    HostVec() = default;
    HostVec(const HostVec&) = delete;
    HostVec& operator=(const HostVec&) = delete;
    ~HostVec() { release(); }
    void resize(size_t n) {
        if (n > cap_) {
            release();
            if (!force_pageable() && strata::gpu::alloc_host(&pinned_, n * sizeof(T))) cap_ = n;
            else { pinned_ = nullptr; fallback_.resize(n); cap_ = n; }
        }
        n_ = n;
    }
    T* data() { return pinned_ != nullptr ? pinned_ : fallback_.data(); }
    size_t size() const { return n_; }
    T& operator[](size_t i) { return data()[i]; }
    void from_device(const T* src, strata::gpu::Stream s) {   // the first size() values, queued on `s`
        if (pinned_ != nullptr) strata::kernels::copy_i32_from_mapped(pinned_, src, (int64_t) n_, s);
        else publication::require(strata::gpu::copy_async(fallback_.data(), src, n_ * sizeof(T), s), "HostVec from_device", strata::gpu::last_error);
    }
    void to_device(T* dst, strata::gpu::Stream s) {
        if (pinned_ != nullptr) strata::kernels::copy_i32_from_mapped(dst, pinned_, (int64_t) n_, s);
        else publication::require(strata::gpu::copy_async(dst, fallback_.data(), n_ * sizeof(T), s), "HostVec to_device", strata::gpu::last_error);
    }
private:
    static_assert(sizeof(T) == 4, "copied as int32");
    void release() {
        if (pinned_ != nullptr) strata::gpu::free(pinned_);
        pinned_ = nullptr;
        fallback_.clear();
        cap_ = n_ = 0;
    }
    T* pinned_ = nullptr;
    std::vector<T> fallback_;
    size_t cap_ = 0, n_ = 0;
};

constexpr int64_t STREAM_ALL_MIN = 2048;
// STRATA_PREFILL_CPU_SHARE (set_cpu_pool, upstream 978d3558, opt-in there): a chunk below STREAM_ALL_MIN hands the
// decode CPU pool - idle while a prompt is read - the non-resident experts few of its tokens route to, instead of
// streaming them.  `auto` (the default here): the share measured (each layer's CPU time per expert and the GPU's per
// streamed expert, events around its expert work read after the next layer's routing sync, as running means): the
// next layers hand the CPU g / (c + g), where both sides end together, so a GPU that streams faster than the CPU
// computes gives it little.  x: a fixed x.  0: every expert on the GPU.  The CPU rounds differently from the GPU, so
// an answer can differ slightly from the GPU-only one.
inline double cpu_share_env() {   // -1: measured
    static const double v = [] {
        const char* e = std::getenv("STRATA_PREFILL_CPU_SHARE");
        if (e == nullptr || std::strcmp(e, "auto") == 0) return -1.0;
        return std::clamp(std::strtod(e, nullptr), 0.0, 1.0);
    }();
    return v;
}
inline bool cpu_share_on() {
    // CPU experts and direct H2D copies can read different spans of one host-USM arena allocation.
    // Without qualified concurrent access the whole allocation must take turns, even for read-only spans.
    // Keep small/tail chunks on the GPU under the serialized protocol until that source overlap is qualified.
    return cpu_share_env() != 0.0 && strata::gpu::host_handoff_concurrent();
}
double g_pinned_share = 1.0;
int g_ring_override = 0;
// The streamed ring: 384 slots when (nearly) every streamed expert is DMA'd from pinned RAM - measured on Q2_0,
// 8192-token chunks: 96 slots 1153 tok/s, 384 1294 (the next layer's experts arrive during its attention half) -
// and 96 when a large share goes through host copies (IQ3_S on 64 GB, a third unpinned: 96 slots 1216, 256 1070 -
// the host copies are the limit and the bigger ring only takes cache slots).  STRATA_PREFILL_RING overrides.
// On a card with little memory the ring takes at most an eighth of it (384 of IQ3_S's 2.7 MB blobs is 1 GiB, 3% of
// a 32 GB card), so the prompt path's buffers do not crowd out the chunk.  From 8192-token chunks on, up to 512 as
// long as the ring stays within kRingBigBytes (RTX 4070, IQ2_XS's 1.5 MB blobs, a 32K prompt in 8192-token chunks: 384
// slots 1296 tok/s, 512 1364, 768 1376; IQ3_XXS's 2.3 MB blobs, 64K: 384 1043, 512 1034, the ring's slots taken from
// the cache's resident experts).
constexpr uint64_t kRingBigBytes = 800ull << 20;
inline int ring_slots(size_t T) {
    const char* v = std::getenv("STRATA_PREFILL_RING");
    const uint64_t blob_b = (uint64_t) std::max<int64_t>(MAXBLOB(), 1);
    const int pinned = (int64_t) T >= 8192 ? (int) std::clamp<uint64_t>(kRingBigBytes / blob_b, 384, 512) : 384;
    int r = v ? (int) std::strtol(v, nullptr, 10)
              : (g_ring_override > 0 ? g_ring_override : g_pinned_share >= 0.9 ? pinned : 96);
    if (!v) {
        static strata::core::PerDevice<uint64_t> per_device;   // the current device's memory, read once for each GPU
        const int dev = strata::core::current_device();
        const uint64_t mem = per_device.get(strata::core::Runtime::at(dev).device(),
                                            [dev] { return strata::core::device_info(dev).total_bytes; });
        const uint64_t blob = (uint64_t) std::max<int64_t>(MAXBLOB(), 1);
        r = (int) std::min<uint64_t>((uint64_t) r, mem / 8 / blob);
    }
    const int big = std::clamp<int>(r, 16, RING_MAX);
    return (int64_t) T >= STREAM_ALL_MIN ? big : STAGE;
}
// The BF16-weight projections (hyper-connection, SSM alpha/beta, indexer, router, shared gate, PLE key/value) take
// BF16 activations here and FP32 ones in decode.  STRATA_PREFILL_BF16X2 adds each activation's BF16 remainder as a
// second GEMM (Y = W.hi + W.lo, ~16 mantissa bits; upstream 61638c1, 4e0592b): a router that picks its top 10 from
// the x decode would.  2 = all but the hyper-connection's; 1 = the hyper-connection's too (its activations are 10240
// wide and its up projection writes as much: slower); 0 (the default: it changes the prompt path's numbers) = off.
inline int bf16x2_mode() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_PREFILL_BF16X2");
        return e != nullptr ? (int) std::strtol(e, nullptr, 10) : 0;
    }();
    return v;
}
inline bool bf16x2() { return bf16x2_mode() != 0; }
inline bool bf16x2_hc() { return bf16x2_mode() == 1; }
// Experts dequantized (FP16) before one grouped product: G of them take G x 9.4 MiB (--prefill-experts)
int g_expert_group = 16;
constexpr int64_t GU_ELEMS = 1280ll * 2560, D_ELEMS = 2560ll * 640;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Either cudaMalloc (owned, freed with the object) or a bump allocation from a borrowed region; with no base and
// no region it only counts, which is how `bytes_needed` sizes the region.
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    uint64_t failed_bytes = 0;          ///< the take that could not be allocated; the failure message names it
    bool count_only = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T* take(size_t n, bool& ok) {
        const uint64_t bytes = workspace::take_bytes(n, sizeof(T));
        uint64_t offset = 0, next = used;
        if (!workspace::advance(bytes, next, base != nullptr && !count_only, cap, offset)) {
            ok = false; failed_bytes = bytes; return nullptr;
        }
        if (count_only) { used = next; return nullptr; }
        if (base != nullptr) {
            used = next;
            return (T*) (base + offset);
        }
        void* p = nullptr;
        if (!strata::gpu::alloc_device(&p, bytes)) { ok = false; failed_bytes = bytes; return nullptr; }
        owned->push_back(p);
        used = next;
        return (T*) p;
    }
};

// upstream 4a9b9041 (#796): what a failed carve had and wanted, for the "do not fit" error
std::string fit_failure(const Alloc& o) {
    size_t fb = 0, tb = 0;
    const bool known = strata::gpu::mem_info(&fb, &tb);
    return " (" + (known ? std::to_string(fb >> 20) + " of " + std::to_string(tb >> 20) + " MiB free at the failure; "
                         : std::string()) +
           std::to_string(o.used >> 20) + " MiB taken, the failing buffer wanted " + std::to_string(o.failed_bytes >> 20) +
           " MiB)";
}

}  // namespace

// Step 4 of the prompt-speed plan: the experts the arena could not pin (a third of the streamed ones on IQ3_S) are
// copied into pinned buffers by these threads, ahead of the launches.  Copied in line by the launching thread they
// left the GPU without queued work while each ~2 MB memcpy ran (~15 s of a 32K prompt on IQ3_S).  Job j - a layer's
// j-th unpinned expert, in launch order - lands in host buffer j % kRing, which is free again once the DMA of job
// j - kRing (recorded by the launching thread, `issued`) is done.
bool stager_sleep() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_STAGER_SLEEP");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}
struct Stager {
    // D-5: the pinned ring's depth (STRATA_STAGER_RING, default 16) - how far the host copies can run ahead of the
    // DMAs of the unpinned experts' blobs
    int kRing = 16;
    // `from` set: a transient source (the GGUF read in place) copies the blob of (l, e) itself (copy_blob), since its
    // blob() pointers do not outlast a chunk's queue
    struct Job { const uint8_t* src; size_t bytes; core::ExpertSource* from = nullptr; int32_t l = 0, e = 0; };
    std::atomic<bool> failed{false}, cancelled{false};
    publication::Failure failure;
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<strata::gpu::Event*> dma_done;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one - the expert pool's issue #29 lesson)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = 0;

    bool init(size_t blob_bytes, int nthreads) {
        if (const char* v = std::getenv("STRATA_STAGER_RING")) kRing = (int) std::clamp<long>(std::strtol(v, nullptr, 10), 2, 256);
        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = (char) (!force_pageable() && strata::gpu::alloc_host((void**) &buf[i], blob_bytes));
            if (!pinned[i]) {
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            if (!strata::gpu::event_create(&dma_done[i])) return false;
        }
        device = strata::core::current_device();
        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] { work(); });
        return true;
    }
    ~Stager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) strata::gpu::event_destroy(dma_done[i]);
            if (buf[i] && pinned[i]) strata::gpu::free(buf[i]);
        }
    }
    void work() {
        const strata::core::OnDevice on_device(device);   // the stage's GPU, on this worker thread too
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                const int j = claim(seen);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                const int b = j % kRing;
                if (j >= kRing)   // job j - kRing's DMA from this buffer is queued: sleep until it is (upstream #1057:
                                  // with --mmap-experts the 32 spinning stagers starved the launching thread)
                    for (int v = issued.load(std::memory_order_acquire); v <= j - kRing;
                         v = issued.load(std::memory_order_acquire)) {
                        if (cancelled.load(std::memory_order_acquire)) break;
                        if (stager_sleep()) issued.wait(v, std::memory_order_acquire);   // STRATA_STAGER_SLEEP=0: spin
                        else std::this_thread::yield();
                    }
                if (cancelled.load(std::memory_order_acquire)) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                try {
                // and done - for a generation's first kRing jobs that is the previous generation's last DMA from
                // the buffer, which nothing else waits for when a chunk ends without a sync (no MTP) or the DMA
                // was a ring entry the routing skipped (an event never recorded returns at once)
                publication::require(strata::gpu::event_sync(dma_done[b]), "strata::gpu::event_sync(dma_done[b])", strata::gpu::last_error);
                const Job& jb = jobs[(size_t) j];
                if (jb.from != nullptr) {
                    if (!jb.from->copy_blob(jb.l, jb.e, buf[b])) failed.store(true, std::memory_order_release);
                } else {
                    std::memcpy(buf[b], jb.src, jb.bytes);
                }
                } catch (...) { failure.capture(); failed.store(true, std::memory_order_release); cancel(); }
                ready[(size_t) j].store(1, std::memory_order_release);
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    int claim(uint32_t g) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            if (head.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return j;
        }
    }
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js) {
        // A worker exception makes this instance terminal. Reconstruct only
        // after the owning Prefill has drained both queues and released it.
        failure.rethrow();
        if (js.empty()) return;
        cancelled.store(false);
        failed.store(false);
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) ready[i].store(0, std::memory_order_relaxed);
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    /// Job j's bytes, in a pinned buffer (waits for the copy).
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) {
            if (cancelled.load(std::memory_order_acquire)) { failure.rethrow(); throw std::runtime_error("stager aborted"); }
            std::this_thread::yield();
        }
        failure.rethrow();
        if (failed.load(std::memory_order_acquire)) throw std::runtime_error("stager copy or event failed");
        return buf[j % kRing];   // a failed copy_blob leaves `failed` set: the chunk ends in an error
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    void issued_one(int j, strata::gpu::Stream copy) {
        publication::require(strata::gpu::event_record(dma_done[j % kRing], copy), "strata::gpu::event_record(dma_done[j % kRing], copy)", strata::gpu::last_error);
        publication::publish_staged(issued, j + 1);
    }
    void cancel() {
        cancelled.store(true, std::memory_order_release);
        issued.store(1 << 30, std::memory_order_release);
        issued.notify_all();
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
        issued.notify_all();
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};

struct Prefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::ExpertSource* src = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
    int64_t T = 0, T_max = 0;
    float* residual_gpu = nullptr;
    const float* residual_host = nullptr;
    layer_major::Layout residual_layout;
    bool layer_major_active() const { return residual_layout.tokens != 0; }
    bool borrowed = false;
    bool compact_moe = false;   // experimental T-row FP16 fallback scratch; no MMQ layers
    strata::gpu::Stream cs = nullptr, copy = nullptr;
    Gemm gemm;
    std::vector<void*> owned;
    // chunk buffers
    float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float* mixed = nullptr;
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    uint16_t *xn16_lo = nullptr, *lo16_lo = nullptr, *mixed_bf_lo = nullptr;   // bf16x2(): the BF16 GEMMs' low parts
    float* bo = nullptr;
    // GDN
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    // QSA
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr, *attn = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* steps_dev = nullptr;
    std::vector<int32_t> steps_host;
    int32_t* sel_ids = nullptr;
    float* sel_scores = nullptr;          // [sel_batch, max_blocks]
    int64_t sel_batch = 256, max_blocks = 0;
    float* attn_scratch = nullptr;
    int64_t attn_batch = 32, cap = 0;
    // MoE
    float *logits = nullptr, *w = nullptr, *GU = nullptr, *Dm = nullptr, *sgate = nullptr, *sup = nullptr,
          *shared = nullptr, *sg = nullptr;
    int32_t *ids = nullptr, *slot_dev = nullptr, *src_dev = nullptr;
    uint16_t *Xs = nullptr, *Hh = nullptr, *sh_h = nullptr;
    // step 2b (MMQ): the activations quantized per layer, H in FP32 and its group's quantized rows, the identity
    // row map, the group bounds, the group buffers of gathered experts
    void *Xq = nullptr, *Hq = nullptr;
    float* H = nullptr;
    int32_t *ids_identity = nullptr, *bounds_dev = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    uint8_t* mring = nullptr;                 // MMQ_RING slots of MMQ_SLOT() bytes (chunks below STREAM_ALL_MIN)
    strata::gpu::Event* mgrp_used[2] = {};    // a group of ring slots read by its products, by the group's parity
    bool mgrp_live[2] = {};
    HostVec<int32_t> bounds_host;
    std::unique_ptr<mmq::Context> mmq_ctx;
    HostVec<int32_t> ids_host, slot_host, src_host;
    std::vector<int32_t> cnt, off;
    uint16_t* dq_gu = nullptr;               // G experts' gate/up, GU_ELEMS apart, and down, D_ELEMS apart
    uint16_t* dq_d = nullptr;
    int32_t* xb_dev = nullptr;               // the experts' first rows in `order`, and the total (grouped products)
    int32_t* compact_xb_dev = nullptr;        // persistent device bounds, reused only in compute queue order
    HostVec<int32_t> xb_host;
    uint8_t* stage_dev[RING_MAX] = {};
    int ring = STAGE;                        // the slots of this layout's ring (ring_slots)
    std::unique_ptr<Stager> stager;          // the unpinned experts' host copies (step 4)
    strata::gpu::Event *copied[RING_MAX] = {}, *used[RING_MAX] = {};
    int used_of[RING_MAX] = {};              // the slot whose `used` event releases this one (a gathered group's last)
    bool stage_live[RING_MAX] = {};
    // PLE
    float* ple_emb = nullptr;
    std::vector<float> ple_pageable[2];      // the fallback when no more RAM can be pinned
    float* ple_emb_host[2] = {};             // pinned, double-buffered: the next chunk's rows are read while this
    strata::gpu::Event* ple_copied[2] = {};          // one runs; the event marks that buffer's upload done
    std::vector<uint32_t> ple_rows[2];
    float* ple_norm = nullptr;
    uint8_t* region = nullptr;               // the attention/MoE scratch region (idle while the PLE block runs)
    uint64_t region_bytes = 0;
    PrefillStats* stats = nullptr;
    // KV streaming: one layer's whole K/V, staged from the host copy per layer and chunk (identity layout)
    strata::kernels::KvHostPools stage;
    int32_t* ident_table = nullptr;
    // set_cpu_pool: a small chunk's CPU experts.  The layer's MoE input (T x N, from `mixed`) and the rows the pool
    // writes (the tail of Dm, in Dm's row order), both in host memory; the pool's activations and jobs
    float *cpu_x = nullptr, *cpu_rows = nullptr;
    size_t cpu_x_n = 0, cpu_rows_n = 0;
    std::vector<uint8_t> cpu_nact;
    std::vector<strata::kernels::cpu::ActQ> cpu_actq;   // a Q2_0 layer's activations (the pool's Q2_0 kernels read ActQ)
    std::vector<strata::kernels::cpu::ExpertJobMulti> cpu_jobs;
    // the measured share: running means of the CPU's ms per expert and the GPU's per streamed expert
    double cpu_c_ms = 0, cpu_g_ms = 0, cpu_share_now = 0.5;
    strata::gpu::Event* cpu_ev[2] = {};   // the GPU's expert work of the last CPU-sharing layer
    bool cpu_pend = false;
    double pend_cpu_ms = 0;
    int64_t pend_n_cpu = 0, pend_n_gpu = 0;
    // layer split: the device, and the hand-off to the next stage (two pinned chunk buffers, used in turn)
    int device = -1;
    float* hand[2] = {};
    // C-4: the chunk's token ids on the device, for one batched embedding gather
    int32_t* tok_dev = nullptr;
    std::vector<int32_t> tok_host;
};

namespace {
// the staging pool of a streamed session: every page of one layer (same sequence in init and bytes_needed)
// STRATA_KV_STAGE_OWN (A/B only): the staging pool gets its own allocation instead of borrowed expert slots, so a
// streamed run lends the prompt path exactly the slots a resident one does (a lent expert runs on the CPU, which
// rounds differently: without this an A/B compares two expert placements as well as two KV placements)
bool stage_own() { static const bool v = std::getenv("STRATA_KV_STAGE_OWN") != nullptr; return v; }
void take_stage(Alloc& o_borrowed, const core::SessionState& ss, const strata::kernels::QsaShapes& s,
                strata::kernels::KvHostPools& st, bool& ok) {
    const core::QsaState& q0 = ss.qsa_states[ss.qsa_primary()];
    if (q0.kv_mode != 1) return;
    if (stage_own() && o_borrowed.count_only) return;
    Alloc own;
    own.owned = o_borrowed.owned;
    Alloc& o = stage_own() ? own : o_borrowed;
    const size_t rows = (size_t) q0.n_pages * s.n_head_kv * s.page_size;
    if (q0.kv_hybrid) {   // K8V4: the three runs of kKvHybrid; pools_of() then reads as the hybrid pools (mode 3)
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_q4) {
        st.k_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
        st.v_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_int8) {
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.v_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
    } else {
        st.k_pool = o.take<uint16_t>(rows * s.head_dim, ok);
        st.v_pool = o.take<uint16_t>(rows * s.head_dim, ok);
    }
}
strata::kernels::QsaAttnPools pools_of(const strata::kernels::KvHostPools& h, const int32_t* table) {
    strata::kernels::QsaAttnPools p;
    p.k_pool = h.k_pool; p.v_pool = h.v_pool; p.k_q = h.k_q; p.v_q = h.v_q; p.k_scale = h.k_scale; p.v_scale = h.v_scale;
    p.k_q4 = h.k_q4; p.v_q4 = h.v_q4;
    p.page_table = table;
    return p;
}
}  // namespace

Prefill::Prefill() : impl_(new Impl) {}
Prefill::~Prefill() { release(); }
void Prefill::reset() { release(); impl_ = std::make_unique<Impl>(); stats_ = {}; }
void Prefill::release() {
    if (!impl_) return;
    if (impl_->stager) impl_->stager->cancel();
    if (impl_->cs) publication::cleanup_require(strata::gpu::stream_sync(impl_->cs), "release/catch cs drain", strata::gpu::last_error);
    if (impl_->copy) publication::cleanup_require(strata::gpu::stream_sync(impl_->copy), "release/catch copy drain", strata::gpu::last_error);
    impl_->stager.reset();
    for (float* p : {impl_->cpu_x, impl_->cpu_rows})
        if (p) strata::gpu::free(p);
    for (strata::gpu::Event* e : impl_->cpu_ev)
        if (e) strata::gpu::event_destroy(e);
    for (int i = 0; i < RING_MAX; ++i) {
        if (impl_->copied[i]) strata::gpu::event_destroy(impl_->copied[i]);
        if (impl_->used[i]) strata::gpu::event_destroy(impl_->used[i]);
    }
    for (int b = 0; b < 2; ++b) {
        if (impl_->mgrp_used[b]) strata::gpu::event_destroy(impl_->mgrp_used[b]);
        if (impl_->hand[b]) strata::gpu::free(impl_->hand[b]);
        if (impl_->ple_copied[b]) strata::gpu::event_destroy(impl_->ple_copied[b]);
        if (impl_->ple_emb_host[b] && impl_->ple_pageable[b].empty()) strata::gpu::free(impl_->ple_emb_host[b]);
    }
    if (impl_->copy) strata::gpu::stream_destroy(impl_->copy);
    for (void* p : impl_->owned) strata::gpu::free(p);
}

namespace {
constexpr int64_t GEMM_SCRATCH = 32ll << 20;        // FP16 elements for the largest dequantized dense weight

// THE ATTENTION HALF AND THE MoE HALF SHARE THEIR BUFFERS.  A layer runs its attention (GDN or QSA), writes it back
// into the residual, and only then its MoE, so the three sets of scratch are never live at once: one region the size
// of the largest holds them all.  That is ~260 KB of the ~680 KB a prompt token cost - which is what lets a chunk
// grow (every expert is streamed once per chunk, so a bigger chunk streams fewer bytes per token).  The sizes are
// counted with the same `take` sequence `init` uses; a mismatch makes `init` fail with "do not fit", never overlap.
// Frozen process-wide opt-in: sizing and every owned/borrowed carve see the same value.
void validate_phase_capacity(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk) {
    // Narrow bounds cover every T*stride, route index and launch extent in this
    // newly admitted fixed-capacity path; they are model geometry, not GPU IDs.
    if (sizeof(size_t) < 8 || (chunk != 12288 && chunk != 4096) || g.n_embd != N || g.hc != HC || g.hc_lr != LR ||
        g.n_layers != 48 || g.n_expert != 512 || g.n_ff != 640 || g.qsa_interval != 4 ||
        g.ssm_state_size != 128 || g.ssm_k_heads != 16 || g.ssm_v_heads != HV ||
        g.ssm_d_conv != 4 || g.ssm_conv_channels != C || g.ssm_value_dim != ZV ||
        g.n_head != 24 || g.n_head_kv != 2 || g.head_dim != 256 ||
        g.idx_q_heads != 4 || g.idx_key_dim != 128 || ss.k != K || ss.max_cells != 262144 ||
        !ss.qsa_states || ss.qsa_alloc != 12 || ss.qsa_ord0 != 0 || stage_own())
        throw std::runtime_error("prefill: bounded phase capacity requires bounded stock geometry and counted KV stage");
    for (int64_t i = 0; i < ss.qsa_alloc; ++i) {
        const auto& q = ss.qsa_states[i];
        if (q.max_cells != 262144 || q.kv_mode != 1 || !q.kv_int8 || q.kv_q4 || q.kv_hybrid ||
            q.n_pages != 65536 || q.n_slots != 8192)
            throw std::runtime_error("prefill: bounded phase capacity requires full mode1/int8 KV and resident32768");
    }
}
bool scratch_alias_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PREFILL_SCRATCH_ALIAS");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}
bool gated_alias_enabled() {
    static const bool on = workspace::exact_one(std::getenv("STRATA_PREFILL_GATED_ALIAS"));
    return on;
}
void validate_gated_alias() {
    if (gated_alias_enabled() && !scratch_alias_enabled())
        throw std::runtime_error("prefill: STRATA_PREFILL_GATED_ALIAS requires STRATA_PREFILL_SCRATCH_ALIAS=1");
}
static_assert(C >= ZV, "GDN y must fit in the retired qkv plane");
uint64_t gdn_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<float>(T * 2 * HV, ok); a.take<float>(T * HV, ok);
    a.take<float>(T * HV, ok); a.take<float>(T * C, ok);
    if (!scratch_alias_enabled()) a.take<float>(T * ZV, ok);
    a.take<uint16_t>(T * ZV, ok);
    return a.used;
}
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const strata::kernels::QsaShapes& s) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * 512, ok); a.take<float>(T * 512, ok); a.take<float>(T * 12288, ok); a.take<float>(T * ZV, ok);
    a.take<float>(T * 128, ok); a.take<float>(T * 512, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    a.take<int32_t>(T * (size_t) cap, ok);
    a.take<float>((size_t) sel_batch * (size_t) max_blocks, ok);
    a.take<float>((size_t) attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);
    return a.used;
}
// Step 2b: which layers' experts go through MMQ (a native pack's layer whose two weight types it covers; a Strata
// pack's experts keep the FP16 path), whether any layer keeps the FP16 path (a type MMQ lacks), and the largest
// gate/up and down matrices a group buffer slot holds.  STRATA_PREFILL_MMQ=0: the FP16 path everywhere (the A/B).
constexpr int MMQ_GROUP = 16;                  // experts per MMQ launch
// Below STREAM_ALL_MIN the MMQ layers' experts are multiplied where their blobs land: a ring of two groups' slots in
// one allocation (a group's slots MMQ_SLOT apart), the streamed ones copied there from the host and the resident ones
// from the cache, so that a group's products read its experts in place.  A group's slots are released together, when
// its products are done (a gather into separate group buffers was three copies an expert, and the RTX 4070's prompt
// path waited on their launches: 980 ms of 4170).  The streamed walk (STREAM_ALL_MIN on) still gathers.
constexpr int MMQ_RING = 2 * MMQ_GROUP;
inline size_t MMQ_SLOT() { return ((size_t) MAXBLOB() + 255) & ~(size_t) 255; }
struct MmqPlan {
    bool any = false, fallback = true;
    std::vector<char> layer;                   // per layer: MMQ
    size_t gu_max = 0, d_max = 0;
};
const MmqPlan& mmq_plan() {
    static const MmqPlan plan = [] {
        MmqPlan p;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const char* env = std::getenv("STRATA_PREFILL_MMQ");
        const bool on = mmq::built() && lay.native && (env == nullptr || std::atoi(env) != 0);
        const int64_t layers = lay.native ? (int64_t) lay.fmt.size() : lay.n_layers;
        p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
        p.fallback = !on || layers <= 0;
        for (int64_t l = 0; on && l < layers; ++l) {
            const int gt = lay.fmt[(size_t) l].gu_type, dt = lay.fmt[(size_t) l].d_type;
            // and gate's rows right before up's (the products read them as one matrix)
            if (!mmq::supported(gt) || !mmq::supported(dt) || lay.fmt[(size_t) l].up_off != mmq::matrix_bytes(gt, 640, N)) {
                p.fallback = true;
                continue;
            }
            p.layer[(size_t) l] = 1;
            p.any = true;
            p.gu_max = std::max(p.gu_max, mmq::matrix_bytes(gt, 1280, N));
            p.d_max = std::max(p.d_max, mmq::matrix_bytes(dt, N, 640));
        }
        return p;
    }();
    return plan;
}
bool compact_moe_requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PREFILL_COMPACT_MOE_SCRATCH");
        return v != nullptr && std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}
bool compact_moe_enabled() {
    static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr;
    const MmqPlan& mp = mmq_plan();
    return compact_moe_requested() && !mp.any && mp.fallback && !dbg;
}
uint64_t moe_set_bytes(size_t T, int64_t n_expert) {
    const MmqPlan& mp = mmq_plan();
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * n_expert, ok); a.take<float>(T * K, ok); a.take<int32_t>(T * K, ok); a.take<int32_t>(T * K, ok);
    a.take<int32_t>(T * K, ok);
    const size_t scratch_rows = compact_moe_enabled() ? T : T * K;
    if (mp.fallback) a.take<uint16_t>(scratch_rows * N, ok);
    a.take<float>(scratch_rows * 1280, ok);
    if (mp.fallback) a.take<uint16_t>(scratch_rows * 640, ok);
    a.take<float>(T * K * N, ok); a.take<float>(T * 640, ok);
    a.take<float>(T * 640, ok); a.take<uint16_t>(T * 640, ok); a.take<float>(T * N, ok); a.take<float>(T, ok);
    if (mp.any) {
        a.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), N), ok);
        a.take<float>(T * K * 640, ok);
        a.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), 640), ok);
    }
    return a.used;
}
}

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes) {
    Impl& m = *impl_;
    m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
    if (layer_major::requested()) {
        if (!layer_major::frozen_prefix().valid || layer_major_reserve_ == 0 || borrow || next_ ||
            stage_lb_ != 0 || (stage_le_ >= 0 && stage_le_ != g.n_layers) || !src || !cache ||
            (cache->phase_mode() && (!core::phase_cache_requested() || !cache->phase_owner()->prefill_leased())) ||
            (core::phase_cache_requested() && !cache->phase_mode()) || embd_rows || on_stage_chunk ||
            !ss.qsa_states || !ss.gdn_state || ss.max_cells <= 0 || ss.max_cells > 262144 || ss.qsa_ord0 != 0 || ss.gdn_ord0 != 0 ||
            g.qsa_interval <= 0 || g.n_layers % g.qsa_interval != 0 ||
            ss.layer_lo != 0 || ss.layer_hi != g.n_layers ||
            ss.qsa_alloc != g.n_layers / g.qsa_interval || ss.gdn_alloc + ss.qsa_alloc != g.n_layers ||
            chunk <= 0 || g.n_expert <= 0 || g.n_expert > 512 || g.n_layers <= 0 || g.n_layers > 512 ||
            std::getenv("STRATA_PREFILL_DUMP_R") || std::getenv("STRATA_PREFILL_DUMP_R_ALL") ||
            std::getenv("STRATA_DBG_NAN") || std::getenv("STRATA_STATE_HASH_GDN")) {
            err = "prefill layer-major prototype: require explicit bounded R_GPU/reserve, full single GPU, "
                  "fresh resident/streamed KV, owned workspace, source/cache and no checkpoint/draft callbacks";
            return false;
        }
        for (int64_t i = 0; i < ss.qsa_alloc; ++i)
            if ((ss.qsa_states[i].kv_mode != 0 && ss.qsa_states[i].kv_mode != 1) ||
                ss.qsa_states[i].kv_mode != ss.qsa_states[ss.qsa_primary()].kv_mode ||
                ss.qsa_states[i].n_pages != ss.qsa_states[ss.qsa_primary()].n_pages ||
                ss.qsa_states[i].max_cells != ss.max_cells) {
                err = "prefill layer-major: unsupported/mixed KV staging geometry"; return false;
            }
        if (!core::Runtime::get().stream(stream).has_property<sycl::property::queue::in_order>()) {
            err = "prefill layer-major: require in-order Runtime compute queue"; return false;
        }
    }
    if ((chunk == 12288 || chunk == 4096) && core::phase_cache_requested()) {
        try { validate_phase_capacity(g, ss, chunk); }
        catch (const std::exception& exception) { err = exception.what(); return false; }
    }
    m.T = chunk; m.cs = stream; m.stats = &stats_;
    try {
        validate_gated_alias();
        if (gated_alias_enabled()) {
            if (chunk <= 0) throw std::runtime_error("prefill: gated alias needs a positive chunk");
            workspace::take_bytes(workspace::product(static_cast<size_t>(chunk), D), sizeof(float));
        }
    }
    catch (const std::exception& exception) { err = exception.what(); return false; }
    if (gated_alias_enabled() &&
        !core::Runtime::get().stream(stream).has_property<sycl::property::queue::in_order>()) {
        err = "prefill: STRATA_PREFILL_GATED_ALIAS requires an in-order compute queue";
        return false;
    }
    if (scratch_alias_enabled() &&
        !core::Runtime::get().stream(stream).has_property<sycl::property::queue::in_order>()) {
        err = "prefill: STRATA_PREFILL_SCRATCH_ALIAS requires an in-order compute queue";
        return false;
    }
    // Reject unsupported opt-ins before allocating anything: MMQ GU stays route-sized, and the existing NaN
    // diagnostic needs all GU rows at once. A compact run must use the ordered queue contract of Runtime.
    m.compact_moe = compact_moe_enabled();
    if (compact_moe_requested() && !m.compact_moe) {
        err = "prefill: STRATA_PREFILL_COMPACT_MOE_SCRATCH requires pure FP16 fallback (no MMQ) and no STRATA_DBG_NAN";
        return false;
    }
    if (m.compact_moe && (g.n_expert <= 0 || g.n_expert > std::numeric_limits<int32_t>::max() ||
        chunk <= 0 || chunk > std::numeric_limits<int32_t>::max() / K ||
        (uint64_t) chunk > std::numeric_limits<size_t>::max() / (K * N * sizeof(float)))) {
        err = "prefill: compact MoE chunk exceeds checked route/allocation bounds";
        return false;
    }
    if (m.compact_moe && !core::Runtime::get().stream(stream).has_property<sycl::property::queue::in_order>()) {
        err = "prefill: compact MoE scratch requires an in-order compute queue";
        return false;
    }
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    m.device = strata::core::current_device();
    if (stage_le_ < 0) stage_le_ = g.n_layers;
    if (stage_lb_ < 0 || stage_lb_ >= stage_le_ || stage_le_ > g.n_layers || (stage_le_ < g.n_layers) != (next_ != nullptr)) {
        err = "prefill: the stage's layer range is wrong";
        return false;
    }
    for (int b = 0; next_ != nullptr && b < 2; ++b)
        if (!m.hand[b] && !strata::gpu::alloc_host((void**) &m.hand[b], (size_t) chunk * D * 4)) {
            err = "prefill: the layer split's hand-off buffers";
            return false;
        }
    if (m.tok_dev == nullptr) {
        if (!strata::gpu::alloc_device((void**) &m.tok_dev, (size_t) chunk * sizeof(int32_t))) {
            err = "prefill: the token id buffer";
            return false;
        }
        m.owned.push_back(m.tok_dev);
        m.tok_host.resize((size_t) chunk);
    }
    m.copy = strata::gpu::copy_stream_create();
    if (!m.copy) { err = "prefill: copy stream"; return false; }
    const size_t T = (size_t) chunk;
    m.T_max = chunk;
    m.borrowed = borrow != nullptr;
    bool ok = true;
    // one-time: events, the stager, the host buffers (for the largest chunk), the identity page table
    for (int i = 0; i < RING_MAX; ++i) {
        if (!strata::gpu::event_create(&m.copied[i])) ok = false;
        if (!strata::gpu::event_create(&m.used[i])) ok = false;
    }
    for (int p = 0; p < 2; ++p)
        if (!m.mgrp_used[p] && !strata::gpu::event_create(&m.mgrp_used[p])) ok = false;
    if (!m.stager) {
        m.stager = std::make_unique<Stager>();
        const int hw = (int) std::thread::hardware_concurrency();
        const char* stv = std::getenv("STRATA_STAGER_THREADS");   // D-5: the host copy threads of unpinned blobs
        // The GGUF read in place (the low-RAM mode): most of a chunk's blobs are page faults on the SSD, so the copies
        // need many reads in flight (upstream 361537b: 32 threads and a 128-deep ring, measured on an RTX 5070 / NVMe
        // PC).  Here as many threads as the CPU has, up to 32, and a ring 4 x as deep; every other source keeps
        // the defaults.
        bool files = false;
        for (int64_t l = 0; src != nullptr && !files && l < g.n_layers; ++l)
            for (int64_t e = 0; !files && e < g.n_expert; ++e) files = src->transient(l, e);
        const int threads = stv ? std::clamp((int) std::strtol(stv, nullptr, 10), 1, 32)
                          : files ? std::clamp(hw, 4, 32) : std::max(2, std::min(4, hw / 4));
        if (files && std::getenv("STRATA_STAGER_RING") == nullptr) m.stager->kRing = 4 * threads;
        if (!m.stager->init((size_t) MAXBLOB(), threads)) ok = false;
    }
    m.steps_host.resize(T * strata::kernels::kStepCount);
    m.ids_host.resize(T * K); m.slot_host.resize(T * K); m.src_host.resize(T * K); m.cnt.resize(m.g->n_expert); m.off.resize(m.g->n_expert + 1);
    for (int b = 0; b < 2; ++b) {
        if (!m.ple_emb_host[b] &&
            (force_pageable() || !strata::gpu::alloc_host((void**) &m.ple_emb_host[b], (size_t) T * N * 4))) {
            m.ple_pageable[b].resize(T * N);          // pageable: the upload is staged before it returns
            m.ple_emb_host[b] = m.ple_pageable[b].data();
        }
        if (!m.ple_copied[b] && !strata::gpu::event_create(&m.ple_copied[b]))
            ok = false;
        m.ple_rows[b].resize(T * strata::kernels::PLE_N_HEADS);
    }
    if (ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {   // KV streaming: the staging pool's identity page table
        const int64_t pages = ss.qsa_states[ss.qsa_primary()].n_pages;
        std::vector<int32_t> ident((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) ident[(size_t) i] = (int32_t) i;
        if (!strata::gpu::alloc_device((void**) &m.ident_table, ident.size() * 4) ||
            !strata::gpu::copy(m.ident_table, ident.data(), ident.size() * 4))
            ok = false;
        else
            m.owned.push_back(m.ident_table);
    }
    if (!ok) { err = "prefill: host buffers or events for a chunk of " + std::to_string(chunk) + " tokens"; return false; }
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    {
        uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
        if (!ok) { err = "prefill: GEMM scratch does not fit"; return false; }
        m.gemm.init_external(stream, gs, GEMM_SCRATCH);
    }
    if (!carve(T, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit" +
              fit_failure(o);
        return false;
    }
    return true;
}

// Every device buffer of a chunk of T tokens, from the Alloc `alloc` (after the GEMM scratch and workspace): `init`
// once, and `relayout` for a request's own chunk.  The order is `bytes_needed`'s.
bool Prefill::carve(size_t T, void* alloc) {
    Impl& m = *impl_;
    Alloc& o = *static_cast<Alloc*>(alloc);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    bool ok = true;
    // xn: the row scales only (F-1, gr_norm_rs: the FP32 normalized rows are not kept; a T x D buffer was 335 MB of
    // the borrowed cache slots at an 8192-token chunk)
    m.emb = o.take<float>(T * N, ok); m.R = o.take<float>(T * D, ok); m.xn = o.take<float>(T * HC, ok);
    m.xn16 = o.take<uint16_t>(T * D, ok); m.lo = o.take<float>(T * LR, ok); m.lo16 = o.take<uint16_t>(T * LR, ok);
    m.gated = gated_alias_enabled() ? nullptr : o.take<float>(workspace::product(T, D), ok);
    m.inj = o.take<float>(T * HC, ok);
    m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
    m.mixed_h = o.take<uint16_t>(T * N, ok);
    // emb's last reader is gr_broadcast on cs (or no embedding at all with hand_in).
    // Layer 0's bo last reader is gr_write[_norm_rs/_cvec_norm_rs], before ple_land.
    // Layer 1's batched/per-token PLE reads finish on cs before its first bo write.
    // Stages starting later never read ple_emb; stages ending before layer 1 land
    // it after their last bo read. Image rows, BF16x2 and MMQ do not alter these
    // boundaries. Skip take entirely: owned must contain only the one allocation.
    m.bo = scratch_alias_enabled() ? m.emb : o.take<float>(T * N, ok);
    if (bf16x2_hc()) { m.xn16_lo = o.take<uint16_t>(T * D, ok); m.lo16_lo = o.take<uint16_t>(T * LR, ok); }
    if (bf16x2()) m.mixed_bf_lo = o.take<uint16_t>(T * N, ok);
    m.steps_dev = o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    m.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    m.max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    {
        // HC gate production/mix precedes each block's first writer on the same ordered queue.
        // PLE/draft keep their earlier/local views; R, xn16 and injection remain separate.
        const auto layout = workspace::region_layout(gdn_set_bytes(T),
            qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch, m.attn_batch, s),
            moe_set_bytes(T, m.g->n_expert), workspace::product(T, D), gated_alias_enabled());
        const uint64_t region = layout.region_extent;
        uint8_t* base = o.take<uint8_t>(workspace::extent_size(region), ok);
        if (!ok || base == nullptr) return false;
        if (gated_alias_enabled()) m.gated = workspace::gate_view(base, layout);
        m.region = base;
        // PLE/draft batching retains the original block region extent even if gated enlarges its backing.
        m.region_bytes = layout.block_max;
        Alloc a;
        a.base = base; a.cap = region; a.owned = &m.owned;
        m.qkv = a.take<float>(T * C, ok); m.z = a.take<float>(T * ZV, ok); m.ab = a.take<float>(T * 2 * HV, ok);
        m.gate = a.take<float>(T * HV, ok); m.beta = a.take<float>(T * HV, ok); m.hbuf = a.take<float>(T * C, ok);
        m.y = scratch_alias_enabled() ? m.qkv : a.take<float>(T * ZV, ok);
        m.y_h = a.take<uint16_t>(T * ZV, ok);
        Alloc b;
        b.base = base; b.cap = region; b.owned = &m.owned;
        m.Kc = b.take<float>(T * 512, ok); m.Vc = b.take<float>(T * 512, ok); m.Qf = b.take<float>(T * 12288, ok);
        m.q = b.take<float>(T * ZV, ok); m.idx_raw = b.take<float>(T * 128, ok); m.q_idx = b.take<float>(T * 512, ok);
        m.attn = b.take<float>(T * ZV, ok); m.attn_h = b.take<uint16_t>(T * ZV, ok);
        m.sel_ids = b.take<int32_t>(T * (size_t) m.cap, ok);
        m.sel_scores = b.take<float>((size_t) m.sel_batch * (size_t) m.max_blocks, ok);
        m.attn_scratch = b.take<float>((size_t) m.attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);
        Alloc c;
        c.base = base; c.cap = region; c.owned = &m.owned;
        m.logits = c.take<float>(T * m.g->n_expert, ok); m.w = c.take<float>(T * K, ok); m.ids = c.take<int32_t>(T * K, ok);
        m.slot_dev = c.take<int32_t>(T * K, ok); m.src_dev = c.take<int32_t>(T * K, ok);
        const MmqPlan& mp = mmq_plan();
        const size_t scratch_rows = m.compact_moe ? T : T * K;
        m.Xs = mp.fallback ? c.take<uint16_t>(scratch_rows * N, ok) : nullptr;
        m.GU = c.take<float>(scratch_rows * 1280, ok);
        m.Hh = mp.fallback ? c.take<uint16_t>(scratch_rows * 640, ok) : nullptr;
        m.Dm = c.take<float>(T * K * N, ok);
        m.sgate = c.take<float>(T * 640, ok); m.sup = c.take<float>(T * 640, ok); m.sh_h = c.take<uint16_t>(T * 640, ok);
        m.shared = c.take<float>(T * N, ok); m.sg = c.take<float>(T, ok);
        if (mp.any) {
            m.Xq = c.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), N), ok);
            m.H = c.take<float>(T * K * 640, ok);
            m.Hq = c.take<uint8_t>(mmq::q8_bytes((int64_t) (T * K), 640), ok);
        }
        if (base == nullptr) ok = false;
    }
    // the FP16 experts' group (157 MB at 16): only where some layer keeps the FP16 path
    m.dq_gu = mmq_plan().fallback ? o.take<uint16_t>((size_t) (g_expert_group * GU_ELEMS), ok) : nullptr;
    m.dq_d = mmq_plan().fallback ? o.take<uint16_t>((size_t) (g_expert_group * D_ELEMS), ok) : nullptr;
    m.xb_dev = o.take<int32_t>((size_t) m.g->n_expert + 1, ok);
    m.compact_xb_dev = m.compact_moe ? o.take<int32_t>((size_t) m.g->n_expert + 1, ok) : nullptr;
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        m.ids_identity = o.take<int32_t>(T * K, ok);
        m.bounds_dev = o.take<int32_t>((size_t) (2 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2)), ok);
        m.grp_gu = m.grp_d = m.mring = nullptr;
        if ((int64_t) T >= STREAM_ALL_MIN) {
            m.grp_gu = o.take<uint8_t>(MMQ_GROUP * mp.gu_max, ok);
            m.grp_d = o.take<uint8_t>(MMQ_GROUP * mp.d_max, ok);
        } else {
            m.mring = o.take<uint8_t>(MMQ_RING * MMQ_SLOT(), ok);
        }
        m.mgrp_live[0] = m.mgrp_live[1] = false;
        // (written at every run's start, not here: when serving, these are live expert-cache slots until a request
        // lends them - a write now would corrupt a resident expert)
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
    }
    // Private layer-major cache holds every active-layer expert. Retain host
    // Stager/events and MMQ gather buffers, but no normal device miss ring.
    // run_impl checks actual full residency/slot capacity before processing.
    m.ring = layer_major::device_ring_slots(ring_slots(T), layer_major::requested());
    if (o.base == nullptr && m.ring > 0) {
        // OWNED buffers: the ring in ONE allocation (upstream 4a9b9041).  384 separate 2.7 MiB allocations took
        // 1,556 MiB on an RTX 4070 (each rounded up to its 2 MiB pages), one of the same total 1,038.  A borrowed
        // region keeps its per-slot layout (and so its price, `bytes_needed`: the loans' slot counts do not move).
        uint8_t* ring_base = o.take<uint8_t>((size_t) m.ring * MMQ_SLOT(), ok);   // slots 256-byte aligned, as alone
        for (int i = 0; ok && i < m.ring; ++i) {
            m.stage_dev[i] = ring_base + (size_t) i * MMQ_SLOT();
            m.stage_live[i] = false;                    // a new buffer: nothing of an earlier layout to wait for
            m.used_of[i] = i;
        }
    } else {
        for (int i = 0; i < m.ring; ++i) {
            m.stage_dev[i] = o.take<uint8_t>((size_t) MAXBLOB(), ok);
            m.stage_live[i] = false;                    // a new buffer: nothing of an earlier layout to wait for
            m.used_of[i] = i;
        }
    }
    m.ple_emb = scratch_alias_enabled() ? m.emb : o.take<float>(T * N, ok);
    m.ple_norm = o.take<float>((size_t) strata::kernels::NG_HC_DIM, ok);
    take_stage(o, ss, s, m.stage, ok);
    m.T = (int64_t) T;
    if (ok && scratch_alias_enabled() && std::getenv("STRATA_TRACE")) {
        const uint64_t gd = gdn_set_bytes(T);
        const uint64_t qs = qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch, m.attn_batch, s);
        const uint64_t mo = moe_set_bytes(T, g.n_expert);
        std::fprintf(stderr, "strata prefill scratch alias: T=%zu borrowed=%d requested=%llu "
                     "gdn=%llu qsa=%llu moe=%llu region=%llu winner=%s "
                     "emb=%p bo=%p ple_emb=%p qkv=%p y=%p\n", T, (int) m.borrowed,
                     (unsigned long long) o.used, (unsigned long long) gd, (unsigned long long) qs,
                     (unsigned long long) mo, (unsigned long long) m.region_bytes,
                     gd == m.region_bytes ? "gdn" : qs == m.region_bytes ? "qsa" : "moe",
                     (void*) m.emb, (void*) m.bo, (void*) m.ple_emb, (void*) m.qkv, (void*) m.y);
    }
    if (ok && scratch_alias_enabled() && std::getenv("STRATA_TRACE")) {
        const auto layout = workspace::region_layout(gdn_set_bytes(T),
            qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch, m.attn_batch, s),
            moe_set_bytes(T, g.n_expert), workspace::product(T, D), gated_alias_enabled());
        std::fprintf(stderr, "strata prefill gated alias: enabled=%d T=%zu borrowed=%d requested=%llu "
                     "block_max=%llu block_winner=%s gate_extent=%llu region_extent=%llu "
                     "allocated_region=%llu region_winner=%s saved_requested=%llu "
                     "gated=%p region=%p pointer_equal=%d\n", (int) gated_alias_enabled(), T, (int) m.borrowed,
                     (unsigned long long) o.used, (unsigned long long) layout.block_max, layout.block_winner,
                     (unsigned long long) layout.gate_extent, (unsigned long long) layout.region_extent,
                     (unsigned long long) layout.allocated_region, layout.region_winner,
                     (unsigned long long) layout.saved_requested, (void*) m.gated, (void*) m.region,
                     (int) ((void*) m.gated == (void*) m.region));
    }
    return ok;
}

bool Prefill::relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err) {
    Impl& m = *impl_;
    if (!m.borrowed || borrow == nullptr || chunk <= 0 || chunk > m.T_max) {
        err = "prefill: relayout needs borrowed buffers and a chunk of at most " + std::to_string(m.T_max);
        return false;
    }
    if (!strata::gpu::stream_sync(m.cs) || !strata::gpu::stream_sync(m.copy)) {
        err = "prefill: relayout: the stream failed";
        return false;
    }
    bool ok = true;
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    if (ok) m.gemm.rebind(gs, GEMM_SCRATCH);
    if (!ok || !carve((size_t) chunk, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit" +
              fit_failure(o);
        return false;
    }
    return true;
}

int64_t Prefill::chunk() const { return impl_->T; }

bool Prefill::draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                       std::string& err) {
    Impl& m = *impl_;
    static const bool off = [] {
        const char* v = std::getenv("STRATA_MTP_BATCH");
        return v != nullptr && v[0] == '0';
    }();
    // A ring (KV streaming: the drafter's window, page p in slot p % n_slots over a host copy) takes the same appends
    // with its own page table and host copy, as a streamed main layer does; the cells written are those the window can
    // still reach (r0 below), which the ring holds, so no two of them share a slot.  STRATA_MTP_BATCH_RING=0: the
    // drafter's own pass for a ring (the A/B).
    static const bool ring_ok = [] {
        const char* v = std::getenv("STRATA_MTP_BATCH_RING");
        return v == nullptr || v[0] != '0';
    }();
    core::QsaState& st = mtp.kv_state_rw();
    if (off || n <= 0 || m.g == nullptr || m.region == nullptr || (st.kv_mode != 0 && !(st.kv_mode == 2 && ring_ok)) ||
        mtp.device() != m.device)
        return false;
    const core::OnDevice on_device(m.device);
    const auto t0 = Clock::now();
    const core::ModelGeometry& g = *m.g;
    const int64_t Nn = g.n_embd, HCN = g.hc * g.n_embd, KV = g.n_head_kv * g.head_dim;
    if (Nn != N || g.hc != HC || g.hc_lr != LR) return false;   // the hyper-connection kernels' fixed shapes
    constexpr int kQ8_0 = 8;   // GGML_TYPE_Q8_0
    const float* w_ne = mtp.tensor_f32("pre_fc_norm_embedding.weight");
    const void* w_fe = mtp.tensor_q8("fc_embedding.weight");
    const float* w_nh = mtp.tensor_f32("pre_fc_norm_hidden.weight");
    const void* w_fh = mtp.tensor_q8("fc_hidden.weight");
    const float* w_hn = mtp.tensor_f32("attn_hyper_connection.hc_norm.weight");
    const uint16_t* w_dn = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_down.weight");
    const uint16_t* w_up = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_up.weight");
    const void* w_k = mtp.tensor_q8("self_attn.k_proj.weight");
    const void* w_v = mtp.tensor_q8("self_attn.v_proj.weight");
    const float* w_kn = mtp.tensor_f32("self_attn.k_norm.weight");
    if (!w_ne || !w_fe || !w_nh || !w_fh || !w_hn || !w_dn || !w_up || !w_k || !w_v || !w_kn) return false;
    const core::NativeEmbed* nemb = core::native_embed();
    const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
    if (!nemb && (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
                  (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
        return false;
    // the cells the drafter's window can still reach
    const int64_t r0 = std::max<int64_t>(0, mtp.first_needed() - cell0);
    if (r0 >= n) return true;
    // per row: emb/e2 (N), en16 (N half), hn/h2/Rm/gated (HCN), hn16/xn16 (HCN half), lo (LR) + lo16, rs, mixed
    // (N) + mixed_h, K and V (KV each), the token id; up to 256 bytes of alignment for each of the 17 buffers
    const uint64_t per_row = 4 * (2 * Nn + 4 * HCN + LR + HC + Nn + 2 * KV + 1) + 2 * (Nn + 2 * HCN + LR + Nn);
    constexpr uint64_t kAlignBytes = (uint64_t) 17 * 256;
    if (m.region_bytes < kAlignBytes) return false;
    int64_t B = std::min<int64_t>(n - r0, (int64_t) ((m.region_bytes - kAlignBytes) / per_row) & ~(int64_t) 63);
    if (st.kv_mode == 2)   // a ring: one batch's cells must not share a slot (a batch can straddle one page more)
        B = std::min<int64_t>(B, ((st.n_slots - 1) * strata::kernels::qsa_real_shapes().page_size) & ~(int64_t) 63);
    if (B < 64) return false;
    uint8_t* q = m.region;
    auto carve = [&](size_t bytes) { void* p = q; q += (bytes + 255) & ~(size_t) 255; return p; };
    auto* emb = (float*) carve((size_t) B * Nn * 4);
    auto* e2 = (float*) carve((size_t) B * Nn * 4);
    auto* en16 = (uint16_t*) carve((size_t) B * Nn * 2);
    auto* hn = (float*) carve((size_t) B * HCN * 4);
    auto* h2 = (float*) carve((size_t) B * HCN * 4);
    auto* Rm = (float*) carve((size_t) B * HCN * 4);
    auto* gated = (float*) carve((size_t) B * HCN * 4);
    auto* hn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    auto* xn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    auto* lo = (float*) carve((size_t) B * LR * 4);
    auto* lo16 = (uint16_t*) carve((size_t) B * LR * 2);
    auto* rs = (float*) carve((size_t) B * HC * 4);
    auto* mixed = (float*) carve((size_t) B * Nn * 4);
    auto* mixed_h = (uint16_t*) carve((size_t) B * Nn * 2);
    auto* Kc = (float*) carve((size_t) B * KV * 4);
    auto* Vc = (float*) carve((size_t) B * KV * 4);
    auto* tok = (int32_t*) carve((size_t) B * 4);
    if ((uint64_t) (q - m.region) > m.region_bytes) return false;
    static const bool timing = std::getenv("STRATA_DRAFT_TIMING") != nullptr;   // debug: where this pass's time goes
    if (timing) strata::gpu::stream_sync(m.cs);
    const auto ti0 = Clock::now();
    if (!mtp.idle(err)) return false;   // the drafter's own stream (its graph uploads) before this writes its K/V
    const double ms_idle = ms_since(ti0);
    const auto tl0 = Clock::now();
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    std::vector<int32_t> tk((size_t) B);
    for (int64_t b0 = r0; b0 < n; b0 += B) {
        const int64_t nb = std::min(B, n - b0), c0 = cell0 + b0;
        for (int64_t i = 0; i < nb; ++i) tk[(size_t) i] = next_tokens[b0 + i];
        if (!strata::gpu::copy_async(tok, tk.data(), (size_t) nb * 4, m.cs)) {
            err = "prefill: the draft tokens' upload failed";
            return false;
        }
        // the input branches: the next token's embedding, and this cell's final residual rows
        if (nemb) {
            nemb->gather_dev(tok, nb, emb, m.cs);
        } else {
            const auto* codes = (const uint8_t*) wemb->data;
            const auto* scales = (const float*) (codes + wemb->codes_bytes);
            const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                   : nullptr;
            strata::kernels::embedding_gather_dev(codes, scales, offsets, tok, (int) nb, wemb->ne0, wemb->code_bits,
                                                  wemb->code_bias, wemb->group_elems,
                                                  (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                  (uint64_t) (wemb->ne0 / wemb->group_elems), emb, m.cs);
        }
        rms_rows(emb, w_ne, nb, Nn, Nn, EPS, m.cs);
        to_f16(emb, en16, nb * Nn, m.cs);
        m.gemm.native(en16, kQ8_0, w_fe, e2, nb, Nn, Nn);
        if (!strata::gpu::copy_async(hn, R_rows + (size_t) b0 * HCN, (size_t) nb * HCN * 4, m.cs)) {
            err = "prefill: the draft rows' copy failed";
            return false;
        }
        if (mtp.hnorm_per_stream())   // --mtp-hnorm stream: as the drafter's own pass (mtp.cpp)
            strata::kernels::native_qsa_rms_norm_grouped(hn, w_nh, hn, (int) Nn, (int) g.hc, (int) (nb * g.hc), EPS, m.cs);
        else
            rms_rows(hn, w_nh, nb, HCN, HCN, EPS, m.cs);
        to_f16(hn, hn16, nb * HCN, m.cs);
        m.gemm.native(hn16, kQ8_0, w_fh, h2, nb * g.hc, Nn, Nn);   // every stream through fc_hidden
        strata::kernels::add_streams_broadcast(h2, e2, Rm, Nn, (int) g.hc, (int) nb, m.cs);
        // the attention hyper-connection's read (its mixed input only: this pass writes nothing back)
        gr_norm_rs(Rm, w_hn, EPS, rs, xn16, nb, m.cs);
        m.gemm.bf16(xn16, w_dn, lo, nb, LR, HCN);
        gr_silu(lo, lo16, nb, m.cs);
        m.gemm.bf16(lo16, w_up, gated, nb, HCN, LR);
        gr_mix_r(Rm, rs, w_hn, gated, mixed, nullptr, nb, m.cs, mixed_h);
        // K and V into the drafter's cache, as the prompt path's QSA layers append theirs
        m.gemm.native(mixed_h, kQ8_0, w_k, Kc, nb, KV, Nn);
        m.gemm.native(mixed_h, kQ8_0, w_v, Vc, nb, KV, Nn);
        rms_rows(Kc, w_kn, nb * g.n_head_kv, g.head_dim, g.head_dim, EPS, m.cs);
        rope(Kc, nb, g.n_head_kv, g.head_dim, KV, c0, strata::kernels::rope_scaling(), m.cs);
        if (st.kv_rot) {   // rotated as the drafter's own pass stores them (mtp.cpp)
            strata::kernels::fwht256_inplace_cuda(Kc, nb * g.n_head_kv, m.cs);
            strata::kernels::fwht256_inplace_cuda(Vc, nb * g.n_head_kv, m.cs);
        }
        if (st.kv_q4) {
            strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, c0, nb, Kc, Vc, s, m.cs, &st.host);
        } else {
            kv_append(Kc, Vc, nb, c0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs, &st.host);
        }
    }
    if (!strata::gpu::stream_sync(m.cs)) {
        err = std::string("prefill: the draft layer's K/V: ") + strata::gpu::last_error();
        return false;
    }
    mtp.ms_prefill += ms_since(t0);
    if (timing)
        std::fprintf(stderr, "strata draft kv: %lld cells from %lld (first needed %lld), batch %lld: drafter idle %.1f ms, "
                     "the batches %.1f ms, all %.1f ms\n", (long long) n, (long long) cell0,
                     (long long) cell0 + (long long) r0, (long long) B, ms_idle, ms_since(tl0), ms_since(t0));
    return true;
}
void Prefill::set_ring_override(int slots) { g_ring_override = std::clamp(slots,0,RING_MAX); }
void Prefill::set_pinned_share(double share) { g_pinned_share = share; }
void Prefill::set_expert_group(int g) { g_expert_group = std::clamp(g, 1, 64); }
int Prefill::expert_group() { return g_expert_group; }
double Prefill::pinned_share() { return g_pinned_share; }

uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk) {
    // the same allocation sequence as `init`, counted
    validate_gated_alias();
    if (gated_alias_enabled()) {
        if (chunk <= 0) throw std::runtime_error("prefill: gated alias needs a positive chunk");
        workspace::take_bytes(workspace::product(static_cast<size_t>(chunk), D), sizeof(float));
    }
    if ((chunk == 12288 || chunk == 4096) && core::phase_cache_requested()) validate_phase_capacity(g, ss, chunk);
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    f(T * N); f(T * D); f(T * HC); o.take<uint16_t>(T * D, ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
    if (!gated_alias_enabled()) f(workspace::product(T, D));
    f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok);
    if (!scratch_alias_enabled()) f(T * N);
    if (bf16x2_hc()) { o.take<uint16_t>(T * D, ok); o.take<uint16_t>(T * LR, ok); }
    if (bf16x2()) o.take<uint16_t>(T * N, ok);
    o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    const int64_t max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    const auto layout = workspace::region_layout(gdn_set_bytes(T), qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
        moe_set_bytes(T, g.n_expert), workspace::product(T, D), gated_alias_enabled());
    o.take<uint8_t>(workspace::extent_size(layout.region_extent), ok);
    if (mmq_plan().fallback) {
        o.take<uint16_t>((size_t) (g_expert_group * GU_ELEMS), ok);
        o.take<uint16_t>((size_t) (g_expert_group * D_ELEMS), ok);
    }
    o.take<int32_t>((size_t) g.n_expert + 1, ok);
    if (compact_moe_enabled()) o.take<int32_t>((size_t) g.n_expert + 1, ok);
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        o.take<int32_t>(T * K, ok);
        o.take<int32_t>((size_t) (2 * (g.n_expert + g.n_expert / MMQ_GROUP + 2)), ok);
        if ((int64_t) T >= STREAM_ALL_MIN) {
            o.take<uint8_t>(MMQ_GROUP * mp.gu_max, ok);
            o.take<uint8_t>(MMQ_GROUP * mp.d_max, ok);
        } else {
            o.take<uint8_t>(MMQ_RING * MMQ_SLOT(), ok);
        }
    }
    for (int i = 0; i < layer_major::device_ring_slots(ring_slots(T), layer_major::requested()); ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    if (!scratch_alias_enabled()) f(T * N);
    f((size_t) strata::kernels::NG_HC_DIM);
    strata::kernels::KvHostPools stage;
    take_stage(o, ss, s, stage, ok);
    if (!ok || o.used > std::numeric_limits<uint64_t>::max() - (8u << 20))
        throw std::overflow_error("prefill: workspace count overflow");
    return o.used + (8u << 20);   // alignment slack
}

PhaseCapacityEstimate Prefill::phase_capacity_estimate(const core::WeightTable& wt,
        const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk) {
    validate_phase_capacity(g, ss, chunk);
    const size_t T = static_cast<size_t>(chunk);
    const auto shapes = strata::kernels::qsa_real_shapes();
    PhaseCapacityEstimate result;
    result.workspace = bytes_needed(g, ss, chunk);
    result.tokens = T * sizeof(int32_t);
    result.identity = static_cast<uint64_t>(ss.qsa_states[ss.qsa_primary()].n_pages) * sizeof(int32_t);
    Alloc stage; stage.count_only = true; bool ok = true;
    strata::kernels::KvHostPools pools;
    take_stage(stage, ss, shapes, pools, ok);
    result.stage_in_workspace = stage.used;
    result.gdn = gdn_set_bytes(T);
    result.qsa = qsa_set_bytes(T, strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, shapes),
                             262144 / shapes.idx_block + 2, 256, 32, shapes);
    result.moe = moe_set_bytes(T, g.n_expert);
    // Lane dispatch may fail at runtime: retain the chooser's full conditional
    // peak unless explicit controls exclude the branch. Actual run T <= capacity.
    const char* keyhead = std::getenv("STRATA_GDN_KEYHEAD");
    if (!std::getenv("STRATA_GDN_REC_HEADS") && (!keyhead || std::strcmp(keyhead, "0") != 0))
        result.chooser = T * HV * 128 * sizeof(float) + 128ull * HV * 128 * sizeof(float);
    const char* bf16_tc = std::getenv("STRATA_BF16_TC");
    if ((!bf16_tc || std::strcmp(bf16_tc, "0") != 0) && strata::kernels::f16_only_matrix_engines()) {
        // Bound all BF16 image shapes, including unused tensors: requested bound,
        // not proof of backend/provider physical peak. No model-specific head size.
        for (const auto& entry : wt.all()) {
            const auto& w = entry.second;
            if (w.kind != core::WeightKind::Bf16InF32 || !w.data || w.ne1 <= 1) continue;
            if (w.ne0 <= 0 || w.ne0 > 16777216 || w.ne1 > 16777216 ||
                static_cast<uint64_t>(w.ne1) > std::numeric_limits<uint64_t>::max() / 2 / static_cast<uint64_t>(w.ne0))
                throw std::runtime_error("prefill: unbounded lazy GEMM shape");
            result.lazy_gemm = std::max(result.lazy_gemm, static_cast<uint64_t>(w.ne1) * static_cast<uint64_t>(w.ne0) * 2);
        }
        if (result.lazy_gemm > std::numeric_limits<uint64_t>::max() - (32ull << 20))
            throw std::overflow_error("prefill: lazy GEMM peak overflow");
        result.lazy_gemm += 32ull << 20; // Gemm's fixed kXSliceElems input-image cap
    }
    return result;
}

namespace {

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (run with --native)"; return false; }
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy);
    return true;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy = 0, const uint16_t* X_lo = nullptr) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    gm.bf16(X, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy);
    if (X_lo) gm.bf16(X_lo, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy, true);
    return true;
}

}  // namespace

namespace {
// STRATA_PREFILL_TIMING=1: the prompt path's GPU time by phase.  Events are recorded on the compute stream in order;
// the time between two consecutive marks is charged to the phase of the first, so a gap where the GPU waits (for the
// host's expert grouping, or for an expert's copy) lands on the phase that was waiting.  Events are reused: the marks
// are folded at every MoE layer's host sync, after which all of them have completed.
enum PfPhase { kPfStart, kPfHc, kPfGdn, kPfQsa, kPfQsaIdx, kPfQsaSel, kPfQsaAttn, kPfRouter, kPfHostGroup, kPfGather,
               kPfWaitCopy, kPfDequant, kPfGemmGU, kPfGemmD, kPfCombine, kPfPle, kPfKvStage, kPfGdnConv, kPfGdnRec, kPfGdnOut,
               kPfCount };
const char* const kPfNames[kPfCount] = {"embed+steps", "hc read", "gdn", "qsa proj", "qsa indexer", "qsa select",
                                        "qsa attn", "router+shared", "host grouping", "gather", "wait copy", "dequant",
                                        "gemm gate/up", "gemm down", "combine", "ple", "kv stage", "gdn conv+gates",
                                        "gdn recurrence", "gdn out proj"};
// Successful expert-blob H2D requests, not PCIe-exclusive traffic or DMA service.
struct PfExpertTransfers {
    uint64_t calls = 0, bytes = 0;
    void add(uint64_t n, uint64_t extent) {
        if (n > std::numeric_limits<uint64_t>::max() - calls ||
            extent > std::numeric_limits<uint64_t>::max() - bytes)
            throw std::runtime_error("prefill expert H2D counters overflow");
        calls += n;
        bytes += extent;
    }
};
// Consumer-owned fields: never written or read by the copy issuer.
struct PfRouteCensus {
    bool routing_observed = false, mmq_enabled = false, plan_fallback = false;
    bool mmq_direct = false, group_gather = false;
    uint64_t routed_rows = 0, expected_rows = 0, active_experts = 0;
    uint64_t active_nonresident = 0, zero_nonresident = 0;
    uint64_t group_accepted = 0, group_declined = 0, individual_gather = 0;
    uint64_t native_dequant_gu = 0, native_dequant_down = 0;
    static void add(uint64_t& counter, uint64_t value = 1) {
        if (value > std::numeric_limits<uint64_t>::max() - counter)
            throw std::runtime_error("prefill route census counter overflow");
        counter += value;
    }
};
struct PfTransferChunk {
    int64_t offset, pos, tokens;
    bool stream_all, threaded;
    std::vector<PfExpertTransfers> layers;
    std::vector<PfRouteCensus> census;
};
struct PfTimer {
    bool on = std::getenv("STRATA_PREFILL_TIMING") != nullptr;
    std::vector<strata::gpu::Event*> ev;
    std::vector<int> ph;
    size_t used = 0;
    double ms[kPfCount] = {};
    // Host wall spent in the timer's CPU API calls and local bookkeeping, not GPU service.
    double mark_host_ms = 0, fold_host_ms = 0;
    size_t marker_calls = 0, new_events = 0, fold_calls = 0, completed_intervals = 0;
    void mark(int phase, strata::gpu::Stream s) {
        if (!on) return;
        const auto t_call = Clock::now();
        ++marker_calls;
        if (used == ev.size()) {
            strata::gpu::Event* e = nullptr;
            publication::require(strata::gpu::event_create(&e, true), "timer event_create", strata::gpu::last_error);
            ++new_events;
            ev.push_back(e);
            ph.push_back(0);
        }
        ph[used] = phase;
        publication::require(strata::gpu::event_record(ev[used], s), "strata::gpu::event_record(ev[used], s)", strata::gpu::last_error);
        ++used;
        mark_host_ms += ms_since(t_call);
    }
    // every recorded mark has completed (the stream was synchronized): charge the gaps, keep the last mark
    void fold() {
        if (!on) return;
        const auto t_call = Clock::now();
        ++fold_calls;
        if (used < 2) { fold_host_ms += ms_since(t_call); return; }
        for (size_t i = 0; i + 1 < used; ++i) {
            float t = 0.0f;
            publication::require(strata::gpu::event_elapsed_ms(&t, ev[i], ev[i + 1]),
                                 "prefill phase timer interval", strata::gpu::last_error);
            ++completed_intervals;
            ms[ph[i]] += t;
        }
        std::swap(ev[0], ev[used - 1]);
        std::swap(ph[0], ph[used - 1]);
        used = 1;
        fold_host_ms += ms_since(t_call);
    }
    ~PfTimer() {
        for (strata::gpu::Event* e : ev) strata::gpu::event_destroy(e);
    }
};
}  // namespace

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    if (!impl_ || !impl_->ss || !impl_->g || impl_->T <= 0) { err = "prefill: not initialized"; return false; }
    if (n == 0) return run_impl(tokens, n, pos0, err);
    return layer_major::requested() ? run_layer_major(tokens, n, pos0, err) : run_impl(tokens, n, pos0, err);
}

bool Prefill::run_layer_major(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) try {
    err.clear();
    Impl& m = *impl_;
    const auto& g = *m.g;
    const core::OnDevice on_device(m.device);
    if (!tokens || stage_lb_ != 0 || stage_le_ != g.n_layers || next_ || hand_in_ || m.borrowed ||
        !m.src || !m.cache ||
        (m.cache->phase_mode() && (!core::phase_cache_requested() || !m.cache->phase_owner()->prefill_leased())) ||
        (core::phase_cache_requested() && !m.cache->phase_mode()) ||
        on_stage_chunk || embd_rows || m.layer_major_active() || !layer_major::frozen_prefix().valid ||
        m.ss->ple_prev[0] != -1 || m.ss->ple_prev[1] != -1 || m.ss->layer_lo != 0 || m.ss->layer_hi != g.n_layers) {
        err = "layer-major: require fresh full single-GPU request without checkpoint/draft state";
        return false;
    }
    for (int64_t i = 0; i < m.ss->qsa_alloc; ++i)
        if (m.ss->qsa_states[i].kv_mode != 0 && m.ss->qsa_states[i].kv_mode != 1) {
            err = "layer-major: unsupported KV staging mode"; return false;
        }
    if (layer_major_full_context_probe_ &&
        !layer_major::full_context_span(n, pos0, m.ss->max_cells, bool(on_chunk))) {
        err = "layer-major full-context probe: exact physical span and completion callback required";
        return false;
    }
    const auto plan = layer_major::plan(n, pos0, m.ss->max_cells, m.T, D,
        layer_major::frozen_prefix().tokens, g.n_expert, g.n_layers, MAXBLOB(), layer_major_reserve_);
    // Pre-existing workspace/model/head/session/ordinary expert cache are charged
    // in these fresh samples. New bridge/cache + explicit reserve must also fit.
    publication::cleanup_require(strata::gpu::stream_sync(m.cs), "layer-major admission compute drain", strata::gpu::last_error);
    publication::cleanup_require(strata::gpu::stream_sync(m.copy), "layer-major admission copy drain", strata::gpu::last_error);
    if (m.stager) m.stager->finish();
    size_t free = 0, total = 0;
    core::detail::HostMemory memory;
    if (!strata::gpu::mem_info(&free, &total) || plan.gpu_need > uint64_t(free) ||
        !core::detail::host_available_memory(memory) || plan.host_need > memory.available) {
        err = "layer-major: fresh VRAM/RAM bridge + layer-cache + explicit reserve admission failed";
        return false;
    }
    const auto& layout = strata::kernels::cpu::expert_layout();
    for (int64_t layer = 0; layer < g.n_layers; ++layer)
        if (layout.blob_bytes(layer) == 0 || layout.blob_bytes(layer) > uint64_t(MAXBLOB())) {
            err = "layer-major: source layout blob extent exceeds slot"; return false;
        }
    std::fprintf(stderr, "strata layer-major: ADMITTED mode=fresh-resident-or-streamed tokens=%lld chunks=%llu requested_gpu_rows=%llu "
        "gpu_rows=%llu host_rows=%llu gpu_residual_bytes=%llu host_residual_bytes=%llu layer_cache_bytes=%llu "
        "reserve=%llu fresh_vram=%zu fresh_ram=%llu preexisting_workspace_head_state_charged=1\n",
        (long long) n, (unsigned long long) plan.chunks,
        (unsigned long long) layer_major::frozen_prefix().tokens, (unsigned long long) plan.gpu_tokens,
        (unsigned long long) (plan.tokens - plan.gpu_tokens), (unsigned long long) plan.gpu_bytes,
        (unsigned long long) plan.host_bytes, (unsigned long long) plan.cache_bytes,
        (unsigned long long) layer_major_reserve_, free, (unsigned long long) memory.available);
    // Declare every owner before the restore guard: its destructor drains before
    // bridge, residency, staged source and layer-cache owners are destroyed.
    core::ExpertCache layer_cache;
    auto free_rows = [](float* pointer) { strata::gpu::free(pointer); };
    std::unique_ptr<float, decltype(free_rows)> device_rows(nullptr, free_rows);
    std::vector<float> rows;
    std::vector<uint8_t> staged;
    std::vector<int32_t> residency;
    const auto started = Clock::now();
    struct Restore {
        Prefill& p; Impl& m;
        int64_t lb, le; const core::ExpertCache* cache; const int32_t* residency;
        decltype(on_chunk) chunk; decltype(on_stage_chunk) stage;
        Clock::time_point started;
        ~Restore() {
            layer_major::retire([&] {
                if (m.stager) m.stager->cancel();
                const bool compute = strata::gpu::stream_sync(m.cs);
                const bool copy = strata::gpu::stream_sync(m.copy);
                if (m.stager) m.stager->finish();
                return compute && copy;
            }, [&] {
                p.stage_lb_ = lb; p.stage_le_ = le;
                p.on_chunk = std::move(chunk); p.on_stage_chunk = std::move(stage);
                m.cache = cache; m.host_res = residency;
                m.residual_gpu = nullptr; m.residual_host = nullptr;
                m.residual_layout = {};
                p.stats_.ms_total += ms_since(started);
            }, [] {
                publication::cleanup_require(false, "layer-major restore drain", strata::gpu::last_error);
            });
        }
    } restore{*this, m, stage_lb_, stage_le_, m.cache, m.host_res, on_chunk, on_stage_chunk, started};
    if (plan.host_bytes / sizeof(float) > rows.max_size() || uint64_t(MAXBLOB()) > staged.max_size() ||
        layer_major::product(uint64_t(g.n_layers), uint64_t(g.n_expert)) > residency.max_size()) {
        err = "layer-major: host owner vector extent exceeds max_size"; return false;
    }
    if (!layer_cache.open_sized(std::vector<int64_t>((size_t) g.n_expert, MAXBLOB()), 1, g.n_expert, err)) return false;
    if (plan.gpu_bytes && !strata::gpu::alloc_device((void**) &m.residual_gpu, (size_t) plan.gpu_bytes)) {
        err = "layer-major: requested GPU prefix allocation failed (no fallback)"; return false;
    }
    device_rows.reset(m.residual_gpu);
    rows.resize((size_t) (plan.host_bytes / sizeof(float)));
    staged.resize((size_t) MAXBLOB());
    residency.assign((size_t) (g.n_layers * g.n_expert), -1);
    m.residual_host = rows.data(); m.residual_layout = plan;
    m.cache = &layer_cache; m.host_res = residency.data();
    const int32_t prev[2] = {m.ss->ple_prev[0], m.ss->ple_prev[1]};
    for (int64_t layer = 0; layer < g.n_layers; ++layer) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        publication::cleanup_require(strata::gpu::stream_sync(m.cs), "layer-major layer reuse drain", strata::gpu::last_error);
        const size_t bytes = (size_t) layout.blob_bytes(layer);
        uint64_t uploaded = 0;
        core::progress_at("layer-major: uploading one expert layer", layer);
        const auto load = Clock::now();
        for (int32_t expert = 0; expert < g.n_expert; ++expert) {
            if (should_stop && should_stop()) { err = "cancelled"; return false; }
            const uint8_t* source = nullptr;
            if (m.src->transient(layer, expert)) {
                if (!m.src->copy_blob(layer, expert, staged.data())) { err = "layer-major: source copy_blob failed"; return false; }
                source = staged.data();
            } else source = m.src->blob(layer, expert);
            void* destination = const_cast<uint8_t*>(layer_cache.device_slot(expert));
            if (!source || !destination) { err = "layer-major: null source/slot"; return false; }
            // blob is stable until the next blob call; synchronous checked copy
            // completes before any source/staging buffer can be reused.
            publication::cleanup_require(strata::gpu::copy(destination, source, bytes),
                                         "layer-major expert copy", strata::gpu::last_error);
            uploaded += bytes;
            ++stats_.experts_streamed;
            if (!m.src->transient(layer, expert) && m.src->pinned(layer, expert)) ++stats_.experts_dma;
        }
        stats_.ms_experts_host += ms_since(load);
        std::fill(residency.begin(), residency.end(), -1);
        for (int32_t expert = 0; expert < g.n_expert; ++expert)
            residency[(size_t) (layer * g.n_expert + expert)] = expert;
        stage_lb_ = layer; stage_le_ = layer + 1;
        m.ss->ple_prev[0] = prev[0]; m.ss->ple_prev[1] = prev[1];
        on_chunk = [&](const float* residual, int64_t count, int64_t position, std::string& error) {
            if (layer_major::final_layer(layer + 1, g.n_layers))
                return !restore.chunk || restore.chunk(residual, count, position, error); // output rows only, no checkpoint
            bool gpu = false; uint64_t offset = 0;
            if (position < pos0 || !layer_major::chunk_store(plan, uint64_t(position - pos0), uint64_t(count), gpu, offset)) {
                error = "layer-major: residual download span crosses store boundary"; return false;
            }
            float* output = gpu ? device_rows.get() + offset : rows.data() + offset;
            publication::cleanup_require(strata::gpu::copy(output, residual, (size_t) count * D * sizeof(float)),
                                         "layer-major residual copy", strata::gpu::last_error);
            return true;
        };
        core::progress_at("layer-major: computing all chunks for layer", layer);
        if (!run_impl(tokens, n, pos0, err)) return false;
        if (std::getenv("STRATA_TRACE")) {
            std::fprintf(stderr, "strata layer-major: layer=%lld role=expert-layer-reuse slots=%lld "
                "uploaded_bytes=%llu computed_chunks=%llu gpu_rows=%llu host_rows=%llu completion=checked\n",
                (long long) layer, (long long) g.n_expert, (unsigned long long) uploaded,
                (unsigned long long) plan.chunks, (unsigned long long) plan.gpu_tokens,
                (unsigned long long) (plan.tokens - plan.gpu_tokens));
            std::fflush(stderr);
        }
    }
    return true;
} catch (const std::exception& exception) {
    err = std::string("layer-major: ") + exception.what(); return false;
} catch (...) {
    err = "layer-major: unknown source/allocation failure"; return false;
}

bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) try {
    err.clear();
    Impl& m = *impl_;
    core::SessionState& ss = *m.ss;
    if (!publication::valid_span(n, pos0, ss.max_cells)) {
        err = "prefill: input span exceeds the physical context capacity";
        return false;
    }
    if (n == 0) return true;
    if (tokens == nullptr) { err = "prefill: missing input tokens"; return false; }
    const core::OnDevice on_device(m.device);
    core::ServiceScope service_scope(core::Runtime::get().service_events());
    const core::ModelGeometry& g = *m.g;
    const auto t_start = Clock::now();
    const int64_t LB = stage_lb_, LE = stage_le_;
    // the next stage reads chunk c on a thread while this one reads chunk c + 1 (declared first: an early return
    // waits for it before anything it reads goes away)
    std::string next_err;
    std::future<bool> next_run;
    int hand_buf = 0;
    double host_sync_ms = 0, host_chunk_ms = 0, host_setup_ms = 0;   // STRATA_PREFILL_TIMING: the host's share
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    PfTimer pt;
    // Independent of phase timing; presence (including 0) enables this optional ledger.
    const bool transfer_counts = std::getenv("STRATA_PREFILL_TRANSFER_COUNTS") != nullptr;
    std::vector<PfTransferChunk> transfer_chunks;
    const strata::gpu::Stream cs = m.cs;
    // the MMQ row table lives in the borrowed cache slots, which the refill after a prompt overwrites with experts:
    // write it again for every prompt (a layout is reused as long as the chunk and the slots are the same)
    if (m.ids_identity != nullptr) mmq::iota(m.ids_identity, m.T * K, m.cs);
    // The PLE rows of a chunk are read from the model file on the host (an SSD read per missed row): the chunk
    // after this one is read on a thread while the GPU runs this one, into the other of two buffers.  The rows
    // depend only on the tokens (the two before a position name its n-grams), so this is the same data.
    const bool ple_on = ss.ple.ready() && LB <= 1 && 1 < LE;
    // the PLE block batched over the chunk: the pinned postops and a BF16 or GGUF-native key (else token by token);
    // STRATA_PLE_BATCH=0 keeps the per-token block (the A/B)
    static const bool ple_batch_env = [] {
        const char* v = std::getenv("STRATA_PLE_BATCH");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const bool ple_batch = ple_on && ple_batch_env && strata::kernels::ple_native_postops_enabled() &&
                           (ss.ple.w.key_bf16 != nullptr || ss.ple.w.key_native_data != nullptr) &&
                           m.region_bytes / ((uint64_t) (3 * strata::kernels::NG_HC_DIM + N + 4) * 4 + (uint64_t) N * 2 + 4096) >= 64;
    const int32_t prev0[2] = {prev[0], prev[1]};
    auto ple_gather = [&m, &ss, tokens, n, prev0](int64_t c0, int buf, std::string& e) -> bool {
        const int64_t T = std::min(m.T, n - c0);
        auto at = [&](int64_t i) { return i < 2 ? prev0[i] : (int32_t) tokens[i - 2]; };   // prev0, then the tokens
        int32_t pv[2] = {at(c0), at(c0 + 1)};
        for (int64_t t = 0; t < T; ++t) {
            const int32_t tok = (int32_t) tokens[c0 + t];
            strata::kernels::ngram_rows(&tok, pv, 1, ss.ple.consts,
                                        m.ple_rows[buf].data() + t * strata::kernels::PLE_N_HEADS);
            pv[0] = pv[1];
            pv[1] = tok;
        }
        return ss.ple.table->gather_batch(m.ple_rows[buf].data(), (size_t) T, m.ple_emb_host[buf], e);
    };
    std::string ple_next_err;
    std::future<bool> ple_next;             // declared after everything it reads: an early return waits for it
    int ple_buf = 0;

    for (int64_t c0 = 0; c0 < n; c0 += m.T) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        if (std::getenv("STRATA_TRACE")) { std::fprintf(stderr, "strata trace: prompt chunk %lld of %lld\n", (long long) c0, (long long) n); std::fflush(stderr); }
        const int64_t T = std::min(m.T, n - c0), p0 = pos0 + c0;
        if (m.layer_major_active() && (!m.cache || !m.host_res || m.ring != 0 || LE != LB + 1 ||
            !layer_major::layer_ready(g.n_expert, m.cache->slots(),
                uint64_t(strata::kernels::cpu::expert_layout().blob_bytes(LB)),
                m.host_res + (size_t) LB * g.n_expert, m.cache->slot_offsets()))) {
            err = "layer-major: active layer is not fully resident in actual private cache capacity"; return false;
        }
        core::progress_at("reading the prompt (batched): preparing the chunk from token", p0);
        if (!m.layer_major_active() || LE == g.n_layers) ++stats_.chunks;
        pt.mark(kPfStart, cs);
        const auto tsetup = Clock::now();
        // ---- embeddings, broadcast to the four streams - or, in a later stage of a layer split, the rows the
        // previous stage handed on
        if (m.layer_major_active() && LB > 0) {
            bool gpu = false; uint64_t offset = 0;
            if (!layer_major::chunk_store(m.residual_layout, uint64_t(c0), uint64_t(T), gpu, offset)) {
                err = "layer-major residual upload span crosses store boundary"; return false;
            }
            const float* input = gpu ? m.residual_gpu + offset : m.residual_host + offset;
            publication::require(strata::gpu::copy_async(m.R, input, (size_t) T * D * 4, m.cs),
                                 "layer-major residual upload", strata::gpu::last_error);
        } else if (hand_in_ != nullptr) {
            if (!strata::gpu::copy_async(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4, m.cs)) {
                err = "prefill: the layer split's hand-off upload failed";
                return false;
            }
        }
        // C-4: the whole chunk's rows in one gather (the same per-element arithmetic as the per-token path, so the
        // same bits); a chunk with picture rows, or a token outside the table, takes the per-token path
        bool batched = hand_in_ == nullptr && !(m.layer_major_active() && LB > 0) && m.tok_dev != nullptr;
        const core::NativeEmbed* nemb = core::native_embed();
        const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
        if (batched && nemb == nullptr &&
            (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
             (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
            batched = false;
        for (int64_t t = 0; batched && t < T; ++t) {
            const int64_t tok = tokens[c0 + t];
            if ((embd_rows && embd_rows[p0 + t]) || tok < 0 || (wemb && tok >= wemb->ne1)) batched = false;
            else m.tok_host[(size_t) t] = (int32_t) tok;
        }
        if (batched) {
            if (!strata::gpu::copy_async(m.tok_dev, m.tok_host.data(), (size_t) T * sizeof(int32_t), m.cs)) {
                err = "prefill: the token id upload failed";
                return false;
            }
            if (nemb) {
                nemb->gather_dev(m.tok_dev, T, m.emb, m.cs);
            } else {
                const auto* codes = (const uint8_t*) wemb->data;
                const auto* scales = (const float*) (codes + wemb->codes_bytes);
                const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                       : nullptr;
                strata::kernels::embedding_gather_dev(codes, scales, offsets, m.tok_dev, (int) T, wemb->ne0,
                                                      wemb->code_bits, wemb->code_bias, wemb->group_elems,
                                                      (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                      (uint64_t) (wemb->ne0 / wemb->group_elems), m.emb, m.cs);
            }
        }
        for (int64_t t = 0; hand_in_ == nullptr && !(m.layer_major_active() && LB > 0) && !batched && t < T; ++t) {
            const float* row = embd_rows ? embd_rows[p0 + t] : nullptr;
            if (row) {
                if (!strata::gpu::copy_async(m.emb + t * N, row, (size_t) N * 4, m.cs)) {
                    err = "prefill: the image embedding upload failed";
                    return false;
                }
            } else if (!core::embed_row(*m.wt, g, tokens[c0 + t], m.emb + t * N, m.cs, err)) {
                return false;
            }
        }
        if (hand_in_ == nullptr && !(m.layer_major_active() && LB > 0)) gr_broadcast(m.emb, m.R, T, m.cs);
        // ---- the PLE rows of the whole chunk, one batched SSD request on a thread (see ple_gather).  A later chunk's
        // were read ahead during the previous chunk; the first chunk's are read beside layer 0 - they are needed from
        // layer 1 on, and gathering them here first left the GPU idle for the whole read (upstream; RTX 4070, 997
        // tokens: 42 ms)
        if (ple_on && !ple_next.valid())
            ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c0, b = ple_buf] {
                return ple_gather(c0, b, ple_next_err);
            });
        bool ple_pending = ple_on;
        // the chunk's rows onto the device just before layer 1 reads them, and the next chunk's gather started
        auto ple_land = [&]() -> bool {
            if (!ple_pending) return true;
            ple_pending = false;
            const auto tp = Clock::now();
            if (!ple_next.get()) {
                err = ple_next_err;
                return false;
            }
            publication::require(strata::gpu::copy_async(m.ple_emb, m.ple_emb_host[ple_buf], (size_t) T * N * 4, m.cs), "copy_async", strata::gpu::last_error);
            publication::require(strata::gpu::event_record(m.ple_copied[ple_buf], m.cs), "strata::gpu::event_record(m.ple_copied[ple_buf], m.cs)", strata::gpu::last_error);
            if (c0 + m.T < n) {
                publication::require(strata::gpu::event_sync(m.ple_copied[ple_buf ^ 1]), "strata::gpu::event_sync(m.ple_copied[ple_buf ^ 1])", strata::gpu::last_error);   // the other buffer's upload (a chunk ago) is done
                ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c1 = c0 + m.T, b = ple_buf ^ 1] {
                    return ple_gather(c1, b, ple_next_err);
                });
            }
            ple_buf ^= 1;
            stats_.ms_ple += ms_since(tp);
            return true;
        };
        for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
        // ---- the QSA step records of every position in the chunk
        for (int64_t t = 0; t < T; ++t) strata::kernels::qsa_step_fill(m.steps_host.data() + t * strata::kernels::kStepCount, p0 + t, s);
        publication::require(strata::gpu::copy_async(m.steps_dev, m.steps_host.data(), (size_t) T * strata::kernels::kStepCount * 4, m.cs), "copy_async", strata::gpu::last_error);

        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < LB; ++l) (core::is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        // step 3: this chunk's stream - every non-resident expert of every layer, layer by layer in id order (entry
        // k lands in ring slot k % ring); a copy is issued once the entry `ring` before it is consumed (its slot's
        // `used` event recorded), so the copy stream never waits on an event that is not queued yet
        const strata::kernels::cpu::ExpertLayout& lay0 = strata::kernels::cpu::expert_layout();
        const bool stream_all = m.ring > STAGE && T >= STREAM_ALL_MIN && m.src != nullptr;
        // Before the issuer/its join guard: a threaded issuer owns these writes,
        // and the consumer reads/moves them only after join and checked drains.
        IssuerTiming issuer_timing(IssuerTiming::requested(), (size_t) (LE - LB));
        const auto issuer_streamed_begin = stats_.experts_streamed;
        std::vector<PfExpertTransfers> chunk_transfers;
        // Also before the join guard, with exclusive consumer ownership. Disabled
        // counts allocate no census storage and perform no additional routing scan.
        std::vector<PfRouteCensus> chunk_census;
        if (transfer_counts) {
            chunk_transfers.resize((size_t) (LE - LB));
            chunk_census.resize((size_t) (LE - LB));
            const MmqPlan& plan = mmq_plan();
            for (int64_t l = LB; l < LE; ++l) {
                auto& c = chunk_census[(size_t) (l - LB)];
                c.mmq_enabled = plan.any && (size_t) l < plan.layer.size() && plan.layer[(size_t) l];
                c.plan_fallback = plan.fallback;
            }
        }
        struct StreamEntry { int32_t l, e; const uint8_t* blob; int job; };
        std::vector<StreamEntry> seq;
        std::vector<size_t> seq_start;
        size_t issued = 0, consumed = 0;
        if (stream_all) {
            seq_start.assign((size_t) g.n_layers + 1, 0);
            std::vector<Stager::Job> js;
            for (int64_t l = LB; l < LE; ++l) {
                seq_start[(size_t) l] = seq.size();
                for (int32_t e = 0; e < m.g->n_expert; ++e) {
                    if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                    const bool tr = m.src->transient(l, e);   // copied by the source in the stager (never pinned)
                    const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
                    if (!tr && !b) { err = "prefill: expert source has no blob"; return false; }
                    int job = -1;
                    if (tr || !m.src->pinned(l, e)) {
                        job = (int) js.size();
                        js.push_back({b, (size_t) lay0.blob_bytes(l), tr ? m.src : nullptr, (int32_t) l, e});
                    }
                    seq.push_back({(int32_t) l, e, b, job});
                }
            }
            for (int64_t l = LE; l <= g.n_layers; ++l) seq_start[(size_t) l] = seq.size();
            m.stager->start(std::move(js));
        }
        struct StagerDone {
            Stager* st;
            strata::gpu::Stream cs, copy;
            void complete() {
                if (!st) return;
                st->cancel();
                publication::cleanup_require(strata::gpu::stream_sync(cs), "stager compute drain", strata::gpu::last_error);
                publication::cleanup_require(strata::gpu::stream_sync(copy), "stager copy drain", strata::gpu::last_error);
                st->finish();
                st->failure.rethrow();
                st = nullptr;
            }
            ~StagerDone() {
                if (st) {
                    st->cancel();
                    publication::cleanup_require(strata::gpu::stream_sync(cs), "unwind cs drain", strata::gpu::last_error);
                    publication::cleanup_require(strata::gpu::stream_sync(copy), "unwind copy drain", strata::gpu::last_error);
                    st->finish();
                }
            }
        } chunk_stager_done{stream_all ? m.stager.get() : nullptr, m.cs, m.copy};
        auto issue_until = [&](size_t limit) {
            limit = std::min(limit, seq.size());
            while (issued < limit) {
                const StreamEntry& en = seq[issued];
                const int sl = (int) (issued % (size_t) m.ring);
                const auto th = Clock::now();
                const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                const auto timing_kind = en.job < 0 ? IssuerTiming::PinnedDirect :
                    (en.blob ? IssuerTiming::UnpinnedStaged : IssuerTiming::TransientStaged);
                // the slot's last reader, the copy and its mark in one submission (copy_async_after)
                const strata::gpu::Event* after = m.stage_live[sl] ? m.used[m.used_of[sl]] : nullptr;
                if (en.job < 0) {
                    publication::require(issuer_timing.measure((size_t) (en.l - LB), timing_kind, false, [&] { return strata::gpu::copy_async_after(m.stage_dev[sl], en.blob, bytes, m.copy, after, m.copied[sl]); }), "strata::gpu::copy_async_after(m.stage_dev[sl], en.blob, bytes, m.copy, after, m.copied[sl])", strata::gpu::last_error);
                    if (transfer_counts && bytes != 0) chunk_transfers[(size_t) (en.l - LB)].add(1, bytes);
                    ++stats_.experts_dma;
                } else {
                    const uint8_t* hb = issuer_timing.measure((size_t) (en.l - LB), timing_kind, true, [&] { return m.stager->wait(en.job); });
                    publication::require(issuer_timing.measure((size_t) (en.l - LB), timing_kind, false, [&] { return strata::gpu::copy_async_after(m.stage_dev[sl], hb, bytes, m.copy, after, m.copied[sl]); }), "strata::gpu::copy_async_after(m.stage_dev[sl], hb, bytes, m.copy, after, m.copied[sl])", strata::gpu::last_error);
                    if (transfer_counts && bytes != 0) chunk_transfers[(size_t) (en.l - LB)].add(1, bytes);
                    m.stager->issued_one(en.job, m.copy);
                }
                issuer_timing.entry((size_t) (en.l - LB), timing_kind, bytes);
                m.stage_live[sl] = true;
                stats_.ms_experts_host += ms_since(th);
                ++stats_.experts_streamed;
                ++issued;
            }
        };
        // D-5: the stream is issued by its own host thread, so the thread launching the layers' kernels never waits
        // behind a host copy of an unpinned blob (that wait left the GPU idle: the 'wait copy' / 'dequant' time of the
        // i-quant prompts).  The same copies in the same order into the same slots, and a slot is refilled only once
        // the compute stream has recorded that it is done with it: the same results.  STRATA_PREFILL_ISSUER=0: inline.
        static const bool issuer_on = [] {
            const char* v = std::getenv("STRATA_PREFILL_ISSUER");
            return v == nullptr || std::strtol(v, nullptr, 10) != 0;
        }();
        std::atomic<size_t> a_issued{0}, a_consumed{0};
        std::atomic<bool> a_stop{false};
        publication::Failure issuer_failure;
        double iss_ms = 0;
        int64_t iss_streamed = 0, iss_dma = 0;
        std::thread issuer;
        struct IssuerJoin {
            std::atomic<bool>* stop;
            std::thread* t;
            Stager* st;
            strata::gpu::Stream cs, copy;
            ~IssuerJoin() {
                publication::stop_join(*stop, *t, [&] { st->cancel(); }, [&] {
                    // No producer can submit now. Drain both queues before stager buffers retire.
                    publication::cleanup_require(strata::gpu::stream_sync(cs), "unwind cs drain", strata::gpu::last_error);
                    publication::cleanup_require(strata::gpu::stream_sync(copy), "unwind copy drain", strata::gpu::last_error);
                });
            }
        } issuer_join{&a_stop, &issuer, m.stager.get(), m.cs, m.copy};
        const bool threaded_issue = stream_all && issuer_on;
        if (threaded_issue) {
            issuer = std::thread([&] {
                try {
                const core::OnDevice od(m.device);
                for (size_t idx = 0; idx < seq.size(); ++idx) {
                    while (idx >= a_consumed.load(std::memory_order_acquire) + (size_t) m.ring) {
                        if (a_stop.load(std::memory_order_acquire)) return;
                        std::this_thread::yield();
                    }
                    if (a_stop.load(std::memory_order_acquire)) return;
                    const StreamEntry& en = seq[idx];
                    const int sl = (int) (idx % (size_t) m.ring);
                    const auto th = Clock::now();
                    const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                    const auto timing_kind = en.job < 0 ? IssuerTiming::PinnedDirect :
                        (en.blob ? IssuerTiming::UnpinnedStaged : IssuerTiming::TransientStaged);
                    // the slot's last reader, the copy and its mark in one submission (copy_async_after)
                    const strata::gpu::Event* after = m.stage_live[sl] ? m.used[m.used_of[sl]] : nullptr;
                    if (en.job < 0) {
                        publication::require(issuer_timing.measure((size_t) (en.l - LB), timing_kind, false, [&] { return strata::gpu::copy_async_after(m.stage_dev[sl], en.blob, bytes, m.copy, after, m.copied[sl]); }), "strata::gpu::copy_async_after(m.stage_dev[sl], en.blob, bytes, m.copy, after, m.copied[sl])", strata::gpu::last_error);
                        if (transfer_counts && bytes != 0) chunk_transfers[(size_t) (en.l - LB)].add(1, bytes);
                        ++iss_dma;
                    } else {
                        const uint8_t* hb = issuer_timing.measure((size_t) (en.l - LB), timing_kind, true, [&] { return m.stager->wait(en.job); });
                        publication::require(issuer_timing.measure((size_t) (en.l - LB), timing_kind, false, [&] { return strata::gpu::copy_async_after(m.stage_dev[sl], hb, bytes, m.copy, after, m.copied[sl]); }), "strata::gpu::copy_async_after(m.stage_dev[sl], hb, bytes, m.copy, after, m.copied[sl])", strata::gpu::last_error);
                        if (transfer_counts && bytes != 0) chunk_transfers[(size_t) (en.l - LB)].add(1, bytes);
                        m.stager->issued_one(en.job, m.copy);
                    }
                    issuer_timing.entry((size_t) (en.l - LB), timing_kind, bytes);
                    m.stage_live[sl] = true;
                    iss_ms += ms_since(th);
                    ++iss_streamed;
                    a_issued.store(idx + 1, std::memory_order_release);
                }
                } catch (...) {
                    issuer_failure.capture();
                    a_stop.store(true, std::memory_order_release);
                    m.stager->cancel();
                }
            });
        } else if (stream_all) {
            issue_until((size_t) m.ring);   // layer 0's first experts, behind the embedding and the PLE
        }
        // the consumer's side: entry k's copy is on the copy stream (the thread issued it), then k is given back
        auto wait_issued = [&](size_t k) {
            if (!threaded_issue) return;
            publication::wait_issued(a_issued, a_stop, issuer_failure, k);
        };
        auto give_back = [&](size_t upto) {
            issuer_failure.rethrow();
            if (threaded_issue) a_consumed.store(upto, std::memory_order_release);
            else issue_until(upto + (size_t) m.ring);
        };
        host_setup_ms += ms_since(tsetup);
        bool normed = false;   // F-2: the previous half's write already normed R for this half (rs, xn16)
        for (int64_t l = LB; l < LE; ++l) {
            core::progress_beat();   // the serve watchdog: a prompt chunk of 8192 tokens is still moving
            core::progress_at("reading the prompt (batched): layer", l, p0);   // a stall names layer and chunk
            const core::LayerView v(*m.wt, l);
            if (l == std::max<int64_t>(LB, 1) && !ple_land()) return false;   // the PLE rows, read from layer 1 on
            // ---- the PLE block at layer 1, token by token (its conv reads the previous tokens' rows)
            if (l == 1 && ple_on && ple_batch) {
                // the whole chunk at once, in sub-batches carved from the idle scratch region: the key and value
                // projections as GEMMs (a token at a time they re-read ~52 MB of BF16 key per token on the IQ
                // files), the rest with the per-token kernels' arithmetic (native_ple_postops_batch)
                pt.mark(kPfPle, cs);
                const auto tp = Clock::now();
                const strata::kernels::PleWeights& pw = ss.ple.w;
                constexpr int64_t HD = strata::kernels::NG_HC_DIM;
                const uint64_t per_token = (uint64_t) (3 * HD + N + 4) * 4 + (uint64_t) N * (bf16x2() ? 4 : 2) + 4096;
                const int64_t SB = std::min<int64_t>(T, (int64_t) (m.region_bytes / per_token));
                for (int64_t s0 = 0; s0 < T; s0 += SB) {
                    const int64_t nb = std::min(SB, T - s0);
                    uint8_t* q = m.region;
                    auto carve_f = [&](size_t n) { float* p = (float*) q; q += (n * 4 + 255) & ~(size_t) 255; return p; };
                    float* key = carve_f((size_t) nb * HD);
                    float* qn = carve_f((size_t) nb * HD);
                    float* gated = carve_f((size_t) nb * HD);
                    float* val = carve_f((size_t) nb * N);
                    float* gate = carve_f((size_t) nb * 4);
                    uint16_t* e16 = (uint16_t*) carve_f((size_t) nb * N / 2);
                    uint16_t* e16_lo = bf16x2() ? (uint16_t*) carve_f((size_t) nb * N / 2) : nullptr;
                    const float* emb = m.ple_emb + s0 * N;
                    if (pw.key_bf16 != nullptr) {
                        to_bf16(emb, e16, nb * N, m.cs, e16_lo);
                        m.gemm.bf16(e16, pw.key_bf16, key, nb, HD, N);
                        if (e16_lo) m.gemm.bf16(e16_lo, pw.key_bf16, key, nb, HD, N, 0, true);
                    } else {
                        to_f16(emb, e16, nb * N, m.cs);
                        m.gemm.native(e16, pw.key_native_type, pw.key_native_data, key, nb, HD, N);
                        to_bf16(emb, e16, nb * N, m.cs, e16_lo);
                    }
                    m.gemm.bf16(e16, pw.value_bf16, val, nb, N, N);
                    if (e16_lo) m.gemm.bf16(e16_lo, pw.value_bf16, val, nb, N, N, 0, true);
                    try {
                        strata::kernels::native_ple_postops_batch(key, m.R + s0 * D, val, ss.ple.hist, pw, qn, gated,
                                                                  gate, (int) nb, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                }
                stats_.ms_ple += ms_since(tp);
            } else if (l == 1 && ple_on) {
                pt.mark(kPfPle, cs);
                const auto tp = Clock::now();
                for (int64_t t = 0; t < T; ++t) {
                    strata::kernels::PleOut po;
                    po.normalized = m.ple_norm;
                    po.result = m.R + t * D;
                    try {
                        strata::kernels::ple_block(m.ple_emb + t * N, m.R + t * D, ss.ple.hist, ss.ple.w, po,
                                                   ss.ple.scratch, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                    strata::kernels::ple_history_advance(ss.ple.hist, m.ple_norm, m.cs);
                }
                stats_.ms_ple += ms_since(tp);
            }
            for (int half = 0; half < 2; ++half) {
                // ---- the hyper-connection read of this half
                const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                  su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                      *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                if (!wn || !wd || !wu || !wi) return false;
                pt.mark(kPfHc, cs);
                // F-1: the row scales into the start of xn (the FP32 normalized rows are not written; gr_mix_r
                // recomputes them from R)
                if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs, m.xn16_lo);
                normed = false;
                if (!bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err, 0, m.xn16_lo)) return false;
                gr_silu(m.lo, m.lo16, T, m.cs, m.lo16_lo);
                if (!bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err, 0, m.lo16_lo)) return false;
                if (!bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err, 0, m.xn16_lo)) return false;
                // Both HC halves: up GEMM -> full gate read in mix -> first GDN/QSA/MoE writer.
                // Gemm's oneMath/native products and mix use this same in-order cs. No later gate reader.
                gr_mix_r(m.R, m.xn, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h,
                         m.mixed_bf_lo);

                if (half == 0 && !core::is_qsa_layer(g, l)) {
                    // ======================= GDN =======================
                    const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                          *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                          *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                          *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                          *wsa = need(v, "ssm_a", err);
                    if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                    pt.mark(kPfGdn, cs);
                    float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                    pt.mark(kPfGdnConv, cs);   // "gdn" is the projections in; the rest on their own lines
                    gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);
                    // Conv is qkv's last reader. The in-order cs completes it before
                    // recurrence materializes FP32 y in qkv's T*C >= T*ZV span.
                    // Keep raw y: the recurrence's output normalizer reads it to
                    // produce y_h; neither its math nor its completion changes.
                    pt.mark(kPfGdnRec, cs);
                    gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, m.cs);
                    pt.mark(kPfGdnOut, cs);
                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;
                    ++gdn_index;
                } else if (half == 0) {
                    // ======================= QSA =======================
                    const core::QsaState& st = ss.qsa_states[qsa_index];
                    const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                                          *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                                          *wik = need(v, "indexer.k_proj.weight", err),
                                          *wiq = need(v, "indexer.q_proj.weight", err),
                                          *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                          *wiqn = need(v, "indexer.q_norm.weight", err),
                                          *wikn = need(v, "indexer.k_norm.weight", err);
                    if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                    pt.mark(kPfQsa, cs);
                    if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
                    if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
                    if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err, 0, m.mixed_bf_lo)) return false;
                    if (!bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err, 0, m.mixed_bf_lo)) return false;
                    rms_rows(m.Kc, (const float*) wkn->data, T * 2, 256, 256, EPS, m.cs);
                    rope(m.Kc, T, 2, 256, 512, p0, strata::kernels::rope_scaling(), m.cs);
                    // KV streaming: this layer's cells [0, p0) come in from the host copy to the staging pool, and the
                    // chunk's cells go to the host copy, the staging pool, and the VRAM slots of resident blocks
                    const bool staged = st.kv_mode == 1;
                    if (staged) {
                        pt.mark(kPfKvStage, cs);
                        strata::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), st.host,
                                                            core::qsa_kv_format(st),
                                                            (p0 + s.page_size - 1) / s.page_size, s, m.cs);
                        pt.mark(kPfQsa, cs);
                    }
                    if (st.kv_hybrid) {   // K8V4: K INT8 unrotated, V rotated Q4_0 (only V and the output rotate)
                        // streamed: each half writes its part of the host copy and of the staging pool
                        strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        const strata::kernels::KvHostPools hk = strata::kernels::kv_hybrid_k_half(st.host),
                                                           hv = strata::kernels::kv_hybrid_v_half(st.host),
                                                           sk = strata::kernels::kv_hybrid_k_half(m.stage),
                                                           sv = strata::kernels::kv_hybrid_v_half(m.stage);
                        const bool mirror = st.host.present();
                        kv_append(m.Kc, m.Kc, T, p0, st.page_table, s.page_size, nullptr, nullptr,
                                  st.k_q, st.k_q, st.k_scale, st.k_scale, m.cs, mirror ? &hk : nullptr,
                                  staged ? &sk : nullptr);
                        strata::kernels::kv_append_q4(st.v_q4, st.v_q4, st.page_table, p0, T, m.Vc, m.Vc, s, m.cs,
                                                      mirror ? &hv : nullptr, staged ? &sv : nullptr);
                    } else {
                        if (st.kv_rot) {   // rotated K and V (kv_q4.hpp), the queries below too, the output back
                            strata::kernels::fwht256_inplace_cuda(m.Kc, T * 2, m.cs);
                            strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        }
                        if (st.kv_q4)
                            strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.Kc, m.Vc, s, m.cs,
                                                          &st.host, staged ? &m.stage : nullptr);
                        else
                            kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs,
                                      &st.host, staged ? &m.stage : nullptr);
                    }
                    split_q(m.Qf, m.q, T, m.cs);
                    rms_rows(m.q, (const float*) wqn->data, T * 24, 256, 256, EPS, m.cs);
                    rope(m.q, T, 24, 256, 6144, p0, strata::kernels::rope_scaling(), m.cs);
                    if (st.kv_rot) strata::kernels::fwht256_inplace_cuda(m.q, T * 24, m.cs);
                    rms_rows(m.q_idx, (const float*) wiqn->data, T * 4, 128, 128, EPS, m.cs);
                    rope(m.q_idx, T, 4, 128, 512, p0, strata::kernels::rope_scaling(), m.cs);
                    // the indexer appends, token by token; then scores + selection for many queries at once:
                    // a query reads completed blocks (final once completed) and `dead` for its own tail block
                    const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    pt.mark(kPfQsaIdx, cs);
                    // C-2: the chunk's appends in three launches instead of one per token (the same end state:
                    // the queries below read it only after the whole chunk is appended). STRATA_INDEXER_PER_TOKEN=1: the old
                    try {
                        static const bool per_token = std::getenv("STRATA_INDEXER_PER_TOKEN") != nullptr;
                        if (!per_token) {
                            strata::kernels::native_qsa_indexer_append_batch(m.idx_raw, T, p0, 0, (const float*) wikn->data,
                                                                             EPS, ib, s, st.max_cells,
                                                                             strata::kernels::rope_scaling(), m.cs);
                        }
                        for (int64_t t = 0; per_token && t < T; ++t) {
                            const int32_t* step_t = m.steps_dev + t * strata::kernels::kStepCount;
                            strata::kernels::native_qsa_indexer_append(m.idx_raw + t * 128, step_t + strata::kernels::kStepPos, 0,
                                                                       (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                                       strata::kernels::rope_scaling(), m.cs);
                        }
                    } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                    pt.mark(kPfQsaSel, cs);
                    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                        const int64_t nb = std::min(m.sel_batch, T - t0);
                        const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                        // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                        const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                strata::kernels::kStepNBid)] + 1;
                        // the scores on tensor cores (3xTF32: FP32-level, not bitwise); STRATA_SELECT_OLD=1: the warp kernel
                        static const bool old_sel = std::getenv("STRATA_SELECT_OLD") != nullptr;
                        if (old_sel || !strata::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,
                                                                             steps0, nb, m.max_blocks, s, m.sel_scores,
                                                                             m.cs, active))
                            strata::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs, active);
                        strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                        m.sel_ids + t0 * m.cap, m.cs);
                    }
                    // STRATA_SEL_OVERLAP (debug, D-1's question): how much do neighbouring queries' selections share?
                    // Per tile of 16 queries: the union of their selected cells against the sum of their widths.
                    if (static const bool ovl = std::getenv("STRATA_SEL_OVERLAP") != nullptr; ovl && qsa_index == 0) {
                        std::vector<int32_t> ids((size_t) (T * m.cap));
                        publication::require(strata::gpu::copy_async(ids.data(), m.sel_ids, ids.size() * 4, m.cs), "copy_async", strata::gpu::last_error);
                        publication::cleanup_require(strata::gpu::stream_sync(m.cs), "strata::gpu::stream_sync(m.cs)", strata::gpu::last_error);
                        double sum_w = 0, sum_u = 0;
                        for (int64_t t0 = 0; t0 + 16 <= T; t0 += 16) {
                            std::vector<int32_t> u;
                            for (int64_t t = t0; t < t0 + 16; ++t) {
                                const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                                sum_w += (double) w;
                                u.insert(u.end(), ids.begin() + t * m.cap, ids.begin() + t * m.cap + w);
                            }
                            std::sort(u.begin(), u.end());
                            sum_u += (double) (std::unique(u.begin(), u.end()) - u.begin());
                        }
                        std::fprintf(stderr, "strata prefill: selection overlap at %lld: 16-query tiles read %.1f%% of the "
                                             "cells one query at a time does\n", (long long) p0, sum_w > 0 ? 100.0 * sum_u / sum_w : 0.0);
                    }
                    // STRATA_IDX_FP16_CHECK: would FP16 pooled indexer keys select the same cells? (the KV-streaming
                    // design's last question). Every query is selected again from the pooled keys and `dead` rounded
                    // to fp16 (exactly what an fp16 store reads back); the agreement with the fp32 selection is
                    // printed cumulatively after each chunk's last QSA layer. Debug: syncs per layer.
                    if (static const bool f16chk = std::getenv("STRATA_IDX_FP16_CHECK") != nullptr; f16chk) {
                        static float *pooled16 = nullptr, *dead16 = nullptr;
                        static int32_t* ids16 = nullptr;
                        static double shared = 0, cells = 0;
                        static long long queries = 0, same = 0, sel_queries = 0;
                        const int64_t rows = st.idx_pooled_rows;
                        if (pooled16 == nullptr &&
                            (!strata::gpu::alloc_device((void**) &pooled16, (size_t) rows * s.idx_dim * 4) ||
                             !strata::gpu::alloc_device((void**) &dead16, (size_t) s.idx_dim * 4) ||
                             !strata::gpu::alloc_device((void**) &ids16, (size_t) (m.T * m.cap) * 4))) {
                            err = "STRATA_IDX_FP16_CHECK: no room for its buffers";
                            return false;
                        }
                        round_f16(st.idx_pooled, pooled16, rows * s.idx_dim, m.cs);
                        round_f16(st.idx_dead, dead16, s.idx_dim, m.cs);
                        for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                            const int64_t nb = std::min(m.sel_batch, T - t0);
                            const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                            strata::kernels::qsa_block_scores(pooled16, dead16, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs);
                            strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                            ids16 + t0 * m.cap, m.cs);
                        }
                        std::vector<int32_t> a((size_t) (T * m.cap)), b((size_t) (T * m.cap));
                        publication::require(strata::gpu::copy_async(a.data(), m.sel_ids, a.size() * 4, m.cs), "copy_async", strata::gpu::last_error);
                        publication::require(strata::gpu::copy_async(b.data(), ids16, b.size() * 4, m.cs), "copy_async", strata::gpu::last_error);
                        publication::cleanup_require(strata::gpu::stream_sync(m.cs), "strata::gpu::stream_sync(m.cs)", strata::gpu::last_error);
                        for (int64_t t = 0; t < T; ++t) {
                            const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                            const int32_t *x = a.data() + t * m.cap, *y = b.data() + t * m.cap;
                            int64_t i = 0, j = 0, c = 0;
                            while (i < w && j < w) {
                                if (x[i] == y[j]) { ++c; ++i; ++j; } else if (x[i] < y[j]) ++i; else ++j;
                            }
                            ++queries;
                            same += c == w;
                            if (p0 + t + 1 > m.cap) { ++sel_queries; shared += (double) c; cells += (double) w; }
                        }
                        if (qsa_index + 1 == g.n_qsa_layers())
                            std::fprintf(stderr, "strata prefill: FP16 indexer keys: %lld of %lld selections identical; "
                                                 "where the selection is sparse, %.4f%% of cells shared (%lld queries)\n",
                                         same, queries, cells > 0 ? 100.0 * shared / cells : 100.0, sel_queries);
                    }
                    // STRATA_QSA_DUMP=<file>: append every QSA layer's selected cells for the prompt's last
                    // STRATA_QSA_DUMP_LAST (4096) positions - records of int32 {qsa layer, pos0, T, cap} + T*cap cells,
                    // for tools/qsa_locality.py (how local the sparse attention's reads are: the KV-streaming question)
                    if (static const char* dump = std::getenv("STRATA_QSA_DUMP"); dump != nullptr) {
                        static const long long last = std::getenv("STRATA_QSA_DUMP_LAST")
                                                          ? std::atoll(std::getenv("STRATA_QSA_DUMP_LAST")) : 4096;
                        if (p0 + T > pos0 + n - last) {
                            std::vector<int32_t> h((size_t) (T * m.cap));
                            publication::require(strata::gpu::copy_async(h.data(), m.sel_ids, h.size() * 4, m.cs), "copy_async", strata::gpu::last_error);
                            publication::cleanup_require(strata::gpu::stream_sync(m.cs), "strata::gpu::stream_sync(m.cs)", strata::gpu::last_error);
                            if (std::FILE* f = std::fopen(dump, "ab")) {
                                const int32_t hdr[4] = {(int32_t) qsa_index, (int32_t) p0, (int32_t) T, (int32_t) m.cap};
                                std::fwrite(hdr, 4, 4, f);
                                std::fwrite(h.data(), 4, h.size(), f);
                                std::fclose(f);
                            }
                        }
                    }
                    const strata::kernels::QsaAttnPools pools = staged ? pools_of(m.stage, m.ident_table)
                                                                       : core::qsa_attn_pools(st);
                    pt.mark(kPfQsaAttn, cs);
                    // perf-review D-1: the whole chunk on tensor cores, one block per (query, KV head), FP32-level
                    // accuracy but not bitwise (qsa_prompt_attn.hpp). The pools it has no kernel for on this device,
                    // or STRATA_PROMPT_ATTN_OLD=1: the decode kernel, 32 queries at a time
                    static const bool old_attn = std::getenv("STRATA_PROMPT_ATTN_OLD") != nullptr;
                    if (old_attn || !strata::kernels::qsa_prompt_attn_batch(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s,
                                                                            m.attn, T, m.cs))
                        for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
                            const int64_t nb = std::min(m.attn_batch, T - t0);
                            strata::kernels::qsa_decode_attn_batch(m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
                                                                   m.steps_dev + t0 * strata::kernels::kStepCount, m.cap,
                                                                   s, m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
                        }
                    if (st.kv_rot || st.kv_hybrid) strata::kernels::fwht256_inplace_cuda(m.attn, T * 24, m.cs);
                    pt.mark(kPfQsa, cs);
                    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs);
                    if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err)) return false;
                    ++qsa_index;
                } else {
                    // ======================= MoE =======================
                    const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                                          *wsu = need(v, "ffn_up_shexp.weight", err),
                                          *wsd = need(v, "ffn_down_shexp.weight", err);
                    if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
                    pt.mark(kPfRouter, cs);
                    if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err, 0, m.mixed_bf_lo)) return false;
                    route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
                    // the shared expert and its scalar gate
                    if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                    if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
                    swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
                    if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
                    if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
                    m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
                    if (m.mixed_bf_lo) m.gemm.bf16(m.mixed_bf_lo, (const uint16_t*) wgi->data, m.sg, T, 1, N, 0, true);
                    // group the (token, k) pairs by expert on the host
                    pt.mark(kPfHostGroup, cs);
                    m.ids_host.resize((size_t) T * K);
                    m.ids_host.from_device(m.ids, m.cs);
                    const strata::kernels::cpu::ExpertLayout& lay_c = strata::kernels::cpu::expert_layout();
                    const bool cpu_maybe = cpu_pool_ != nullptr && cpu_share_on() && !stream_all && m.src != nullptr &&
                                           lay_c.native && !lay_c.fmt.empty();
                    if (cpu_maybe) {   // the MoE input, for the CPU's experts
                        const size_t want = (size_t) T * N;
                        if (m.cpu_x_n < want) {
                            if (m.cpu_x) strata::gpu::free(m.cpu_x);
                            m.cpu_x_n = 0;
                            if (!strata::gpu::alloc_host(&m.cpu_x, want * sizeof(float))) {
                                err = "prefill: cannot allocate the CPU experts' activations";
                                return false;
                            }
                            m.cpu_x_n = want;
                        }
                        publication::require(strata::gpu::copy_async(m.cpu_x, m.mixed, want * sizeof(float), m.cs), "copy_async", strata::gpu::last_error);
                    }
                    publication::cleanup_require(strata::gpu::stream_sync(m.cs), "strata::gpu::stream_sync(m.cs)", strata::gpu::last_error);
                    pt.fold();
                    if (m.cpu_pend) {   // auto: the last CPU-sharing layer's GPU time is final now
                        m.cpu_pend = false;
                        float g_ms = 0;
                        if (strata::gpu::event_elapsed_ms(&g_ms, m.cpu_ev[0], m.cpu_ev[1])) {
                            constexpr double a = 0.25;
                            const double c = m.pend_cpu_ms / (double) m.pend_n_cpu, gm = g_ms / (double) m.pend_n_gpu;
                            m.cpu_c_ms = m.cpu_c_ms > 0 ? m.cpu_c_ms + a * (c - m.cpu_c_ms) : c;
                            m.cpu_g_ms = m.cpu_g_ms > 0 ? m.cpu_g_ms + a * (gm - m.cpu_g_ms) : gm;
                            m.cpu_share_now = std::clamp(m.cpu_g_ms / (m.cpu_c_ms + m.cpu_g_ms), 0.05, 0.9);
                        }
                    }
                    std::fill(m.cnt.begin(), m.cnt.end(), 0);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = m.ids_host[(size_t) i];
                        if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }
                        ++m.cnt[(size_t) e];
                    }
                    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
                    const bool use_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
                    PfRouteCensus* census = transfer_counts ? &chunk_census[(size_t) (l - LB)] : nullptr;
                    if (census) {
                        census->routing_observed = true;
                        census->mmq_enabled = use_mmq;
                        census->plan_fallback = mmq_plan().fallback;
                        if (T < 0 || K <= 0 || (uint64_t) T > std::numeric_limits<uint64_t>::max() / (uint64_t) K)
                            throw std::runtime_error("prefill route census expected rows overflow");
                        census->expected_rows = (uint64_t) T * (uint64_t) K;
                        for (int32_t e = 0; e < m.g->n_expert; ++e) {
                            const int32_t rows = m.cnt[(size_t) e];
                            if (rows < 0) throw std::runtime_error("prefill route census negative rows");
                            PfRouteCensus::add(census->routed_rows, (uint64_t) rows);
                            const bool resident = m.host_res && m.cache &&
                                                  m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                            if (rows > 0) {
                                PfRouteCensus::add(census->active_experts);
                                if (!resident) PfRouteCensus::add(census->active_nonresident);
                            } else if (!resident) {
                                PfRouteCensus::add(census->zero_nonresident);
                            }
                        }
                        if (census->routed_rows != census->expected_rows)
                            throw std::runtime_error("prefill route census row total mismatch");
                    }
                    if (m.compact_moe) {
                        int64_t sum = 0;
                        for (const int32_t count : m.cnt) {
                            if (count < 0 || count > T || count > T * K - sum)
                                throw std::runtime_error("prefill compact MoE invalid routed expert count");
                            sum += count;
                        }
                        if (sum != T * K) throw std::runtime_error("prefill compact MoE incomplete routing coverage");
                    }
                    // set_cpu_pool: the experts the CPU computes this layer (their rows last: [T * K - rows_cpu, T * K))
                    std::vector<char> on_cpu;
                    int64_t rows_cpu = 0, n_cpu = 0, n_stream = 0;
                    if (cpu_maybe) {
                        std::vector<std::pair<int32_t, int32_t>> cand;   // (tokens, expert)
                        for (int32_t e = 0; e < m.g->n_expert; ++e) {
                            const int32_t c = m.cnt[(size_t) e];
                            if (c == 0 || (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0)) continue;
                            ++n_stream;
                            if (c <= strata::kernels::cpu::MAXT && m.src->pinned(l, e)) cand.emplace_back(c, e);
                        }
                        std::sort(cand.begin(), cand.end());
                        const double share = cpu_share_env() >= 0.0 ? cpu_share_env() : m.cpu_share_now;
                        const size_t take = std::min(cand.size(), (size_t) std::llround(share * (double) n_stream));
                        if (take > 0) {
                            on_cpu.assign((size_t) m.g->n_expert, 0);
                            for (size_t i = 0; i < take; ++i) {
                                on_cpu[(size_t) cand[i].second] = 1;
                                rows_cpu += cand[i].first;
                            }
                            n_cpu = (int64_t) take;
                        }
                    }
                    auto gpu_side = [&](int32_t e) { return on_cpu.empty() || !on_cpu[(size_t) e]; };
                    // The experts, in id order: resident ones from VRAM, the others through the staging ring.  F8:
                    // first, out of that order, the resident experts of at most kIqGemmTileRows rows, whose products
                    // iq_gemm_grouped_f16 takes straight from their GGUF blocks in the cache (dequantized in local
                    // memory to the same FP16 values: the same bits as dequantizing and multiplying, without writing
                    // 9.4 MiB an expert; more rows than one work-group's would decode the weights again per group).
                    // The rows are laid out in this order, so the others' groups stay contiguous.
                    // STRATA_PREFILL_FUSED=0: every expert dequantized.
                    static const bool fused_on = [] {
                        const char* v = std::getenv("STRATA_PREFILL_FUSED");
                        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                    }();
                    const strata::kernels::cpu::NativeFmt* ffmt = lay.native ? &lay.fmt[(size_t) l] : nullptr;
                    const bool fuse_res = fused_on && !use_mmq && ffmt != nullptr && m.host_res && m.cache &&
                                          strata::kernels::xmx_available(strata::kernels::XmxType::f16) &&
                                          ffmt->up_off == (size_t) ffmt->n_ff * ffmt->gu_row &&
                                          strata::kernels::iq_gemm_grouped_ok(ffmt->gu_type, 2 * ffmt->n_ff, ffmt->n_embd) &&
                                          strata::kernels::iq_gemm_grouped_ok(ffmt->d_type, ffmt->n_embd, ffmt->n_ff);
                    auto fusable = [&](int32_t e) {
                        return fuse_res && m.cnt[(size_t) e] <= strata::kernels::kIqGemmTileRows &&
                               m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                    };
                    std::vector<int32_t> order;
                    for (int32_t e = 0; e < m.g->n_expert; ++e)
                        if (m.cnt[(size_t) e] > 0 && gpu_side(e) && fusable(e)) order.push_back(e);
                    const size_t nf = order.size();
                    for (int32_t e = 0; e < m.g->n_expert; ++e)
                        if (m.cnt[(size_t) e] > 0 && gpu_side(e) && !fusable(e)) order.push_back(e);
                    {
                        int32_t at = 0;
                        for (const int32_t e : order) { m.off[(size_t) e] = at; at += m.cnt[(size_t) e]; }
                        for (int32_t e = 0; e < m.g->n_expert && !on_cpu.empty(); ++e)   // the CPU's block, last
                            if (on_cpu[(size_t) e]) { m.off[(size_t) e] = at; at += m.cnt[(size_t) e]; }
                        m.off[(size_t) m.g->n_expert] = at;
                    }
                    const int64_t rows_gpu = T * K - rows_cpu;
                    if (m.compact_moe && (rows_gpu < 0 || rows_gpu > T * K))
                        throw std::runtime_error("prefill compact MoE GPU row total outside routes");
                    std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = m.ids_host[(size_t) i];
                        const int32_t p = fill[(size_t) e]++;
                        m.slot_host[(size_t) i] = p;
                        m.src_host[(size_t) p] = (int32_t) (i / K);
                    }
                    m.slot_host.to_device(m.slot_dev, m.cs);
                    m.src_host.to_device(m.src_dev, m.cs);
                    // set_cpu_pool: the CPU's experts on a thread while this one streams and runs the GPU's; their
                    // unweighted rows land in Dm's row order, so the combine weights them as any other row.  The
                    // blobs are taken here: the thread calls no ExpertSource (upstream 667f2eca)
                    std::future<bool> cpu_fut;
                    double cpu_ms = 0;
                    if (rows_cpu > 0) {
                        const size_t want = (size_t) rows_cpu * N;
                        if (m.cpu_rows_n < want) {
                            if (m.cpu_rows) strata::gpu::free(m.cpu_rows);
                            m.cpu_rows_n = 0;
                            const size_t cap_rows = (size_t) rows_cpu * 2;   // headroom: few reallocations
                            if (!strata::gpu::alloc_host(&m.cpu_rows, cap_rows * N * sizeof(float))) {
                                err = "prefill: cannot allocate the CPU experts' rows";
                                return false;
                            }
                            m.cpu_rows_n = cap_rows * N;
                        }
                        const strata::kernels::cpu::NativeFmt& cf = lay.fmt[(size_t) l];
                        const bool q2 = cf.gu_type == 42;   // a native Q2_0 layer: the pool's Q2_0 kernels read ActQ
                        constexpr size_t AB = strata::kernels::cpu::kNativeActBytes;
                        m.cpu_jobs.clear();
                        for (int32_t e = 0; e < m.g->n_expert; ++e) {
                            if (!on_cpu[(size_t) e]) continue;
                            strata::kernels::cpu::ExpertJobMulti j;
                            j.blob = m.src->blob(l, e);
                            if (j.blob == nullptr) { err = "prefill: a CPU expert has no blob"; return false; }
                            j.nt = m.cnt[(size_t) e];
                            for (int r = 0; r < j.nt; ++r)   // the activations are set on the thread
                                j.out[r] = m.cpu_rows + (size_t) ((int64_t) m.off[(size_t) e] + r - rows_gpu) * N;
                            m.cpu_jobs.push_back(j);
                        }
                        if (cpu_share_env() < 0.0) {
                            if (m.cpu_ev[0] == nullptr &&
                                (!strata::gpu::event_create(&m.cpu_ev[0], true) || !strata::gpu::event_create(&m.cpu_ev[1], true))) {
                                err = "prefill: CPU share events";
                                return false;
                            }
                            publication::require(strata::gpu::event_record(m.cpu_ev[0], m.cs), "strata::gpu::event_record(m.cpu_ev[0], m.cs)", strata::gpu::last_error);
                        }
                        cpu_fut = std::async(std::launch::async, [&m, &cf, &cpu_ms, q2, T, rows_gpu, pool = cpu_pool_]() -> bool {
                            const auto t0 = std::chrono::steady_clock::now();
                            if (q2 && m.cpu_actq.size() < (size_t) T) m.cpu_actq.resize((size_t) T);
                            if (!q2 && m.cpu_nact.size() < (size_t) T * AB) m.cpu_nact.resize((size_t) T * AB);
                            std::vector<char> need((size_t) T, 0);   // the tokens the CPU's experts read
                            for (int64_t p = rows_gpu; p < T * K; ++p) need[(size_t) m.src_host[(size_t) p]] = 1;
                            for (int64_t t = 0; t < T; ++t) {
                                if (!need[(size_t) t]) continue;
                                if (q2) strata::kernels::cpu::act_quant_any(m.cpu_x + (size_t) t * N, (int) N, m.cpu_actq[(size_t) t]);
                                else strata::kernels::cpu::native_quant_act(cf, m.cpu_x + (size_t) t * N, m.cpu_nact.data() + (size_t) t * AB);
                            }
                            for (auto& j : m.cpu_jobs)
                                for (int r = 0; r < j.nt; ++r) {
                                    const int64_t p = rows_gpu + (int64_t) (j.out[r] - m.cpu_rows) / N;
                                    const int64_t tok = m.src_host[(size_t) p];
                                    if (q2) j.act[r] = &m.cpu_actq[(size_t) tok];
                                    else j.nact[r] = m.cpu_nact.data() + (size_t) tok * AB;
                                }
                            pool->run_split_multi_native(cf, m.cpu_jobs.data(), (int) m.cpu_jobs.size());
                            cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                            return true;
                        });
                    }
                    struct CpuJoin {   // an early return waits for the thread (it reads this layer's tables)
                        std::future<bool>& f;
                        ~CpuJoin() { if (f.valid()) f.wait(); }
                    } cpu_join{cpu_fut};
                    const int mmq_gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42;
                    const int mmq_dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
                    const size_t mmq_gub = use_mmq ? mmq::matrix_bytes(mmq_gt, 1280, N) : 0;
                    const size_t mmq_db = use_mmq ? mmq::matrix_bytes(mmq_dt, N, 640) : 0;
                    pt.mark(kPfGather, cs);
                    if (use_mmq) {
                        // step 2b: the layer's activations as q8_1 rows in expert order, straight from `mixed`
                        mmq::quantize(m.mixed, m.src_dev, m.Xq, mmq_gt, N, N, T * K, m.cs);
                        // each group's rows: absolute bounds (gate/up reads the layer's rows), relative ones (down
                        // reads the group's own quantized H)
                        const size_t n = order.size(), ng = (n + MMQ_GROUP - 1) / MMQ_GROUP;
                        m.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                        for (size_t j = 0; j < n; ++j) m.bounds_host[j] = m.off[(size_t) order[j]];
                        m.bounds_host[n] = (int32_t) rows_gpu;
                        for (size_t g = 0; g < ng; ++g)
                            for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                m.bounds_host[n + 1 + g * (MMQ_GROUP + 1) + i] =
                                    m.bounds_host[std::min(n, g * MMQ_GROUP + i)] - m.bounds_host[g * MMQ_GROUP];
                        m.bounds_host.to_device(m.bounds_dev, m.cs);
                    } else {
                        if (!m.compact_moe) gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
                        // the grouped products' row bounds, `order`'s first rows and the total
                        m.xb_host.resize(order.size() + 1);
                        for (size_t j = 0; j < order.size(); ++j) m.xb_host[j] = m.off[(size_t) order[j]];
                        m.xb_host[order.size()] = (int32_t) rows_gpu;
                        m.xb_host.to_device(m.xb_dev, m.cs);
                        // F8: the fusable resident experts, kIqGemmGroup a launch
                        std::vector<const void*> fgu, fd;
                        for (size_t j0 = 0; j0 < nf; j0 += (size_t) strata::kernels::kIqGemmGroup) {
                            const size_t j1 = std::min(nf, j0 + (size_t) strata::kernels::kIqGemmGroup);
                            if (m.compact_moe) {
                                const auto groups = compact_moe_groups(m.xb_host.data(), order.data(), order.size(),
                                                                       m.cnt.data(), m.cnt.size(), j0, j1, T, rows_gpu);
                                for (const auto& group : groups) {
                                    fgu.clear(); fd.clear();
                                    for (size_t j = group.first; j < group.last; ++j) {
                                        const int32_t e = order[j];
                                        if (m.cnt[(size_t) e] > strata::kernels::kIqGemmTileRows)
                                            throw std::runtime_error("prefill compact MoE resident IQ row bound");
                                        const uint8_t* b = m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]);
                                        fgu.push_back(b); fd.push_back(b + ffmt->down_off);
                                    }
                                    const int G = (int) (group.last - group.first);
                                    // xb_dev is immutable for this layer. Each translation follows the previous
                                    // Down reader on m.cs, so no asynchronous kernel ever reads overwritten bounds.
                                    compact_moe_bounds(m.xb_dev + group.first, m.compact_xb_dev, G, group.row0, m.cs);
                                    gather_rows16(m.mixed_h, m.src_dev + group.row0, m.Xs, group.rows, N, m.cs);
                                    pt.mark(kPfGemmGU, cs);
                                    strata::kernels::iq_gemm_grouped_f16(ffmt->gu_type, m.Xs, fgu.data(), ffmt->n_ff,
                                                                         m.GU, 1280, m.compact_xb_dev, G, group.max_rows,
                                                                         1280, N, m.cs);
                                    swiglu_interleaved(m.GU, m.Hh, group.rows, m.cs);
                                    pt.mark(kPfGemmD, cs);
                                    strata::kernels::iq_gemm_grouped_f16(ffmt->d_type, m.Hh, fd.data(), 0,
                                                                         m.Dm + (int64_t) group.row0 * N, N,
                                                                         m.compact_xb_dev, G, group.max_rows, N, 640, m.cs);
                                    stats_.experts_resident += G;
                                }
                                continue;
                            }
                            fgu.clear();
                            fd.clear();
                            int64_t maxr = 0;
                            for (size_t j = j0; j < j1; ++j) {
                                const int32_t e = order[j];
                                const uint8_t* b = m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]);
                                fgu.push_back(b);
                                fd.push_back(b + ffmt->down_off);
                                maxr = std::max<int64_t>(maxr, m.cnt[(size_t) e]);
                            }
                            const int G = (int) (j1 - j0);
                            const int64_t r0 = m.xb_host[j0], nr = m.xb_host[j1] - r0;
                            pt.mark(kPfGemmGU, cs);
                            strata::kernels::iq_gemm_grouped_f16(ffmt->gu_type, m.Xs, fgu.data(), ffmt->n_ff, m.GU, 1280,
                                                                 m.xb_dev + j0, G, maxr, 1280, N, m.cs);
                            swiglu_interleaved(m.GU + r0 * 1280, m.Hh + r0 * 640, nr, m.cs);
                            pt.mark(kPfGemmD, cs);
                            strata::kernels::iq_gemm_grouped_f16(ffmt->d_type, m.Hh, fd.data(), 0, m.Dm, N,
                                                                 m.xb_dev + j0, G, maxr, N, 640, m.cs);
                            stats_.experts_resident += G;
                        }
                    }
                    // Products consume a COMPLETE logical dequant group. Compact mode only partitions its rows;
                    // the existing q slots, native_flush, blob readers, and ring release events keep their order.
                    auto fallback_products = [&](size_t j0, size_t j1) {
                        if (m.compact_moe) {
                            if (j0 > j1 || j1 - j0 > (size_t) g_expert_group)
                                throw std::runtime_error("prefill compact MoE dequant window bounds");
                            const auto groups = compact_moe_groups(m.xb_host.data(), order.data(), order.size(),
                                                                   m.cnt.data(), m.cnt.size(), j0, j1, T, rows_gpu);
                            for (const auto& group : groups) {
                                const int G = (int) (group.last - group.first);
                                const size_t q0 = group.first - j0;
                                if (q0 > (size_t) g_expert_group || (size_t) G > (size_t) g_expert_group - q0)
                                    throw std::runtime_error("prefill compact MoE dequant subgroup outside slots");
                                const uint16_t* gu = m.dq_gu + q0 * GU_ELEMS;
                                const uint16_t* dn = m.dq_d + q0 * D_ELEMS;
                                float* dst = m.Dm + (int64_t) group.row0 * N;
                                compact_moe_bounds(m.xb_dev + group.first, m.compact_xb_dev, G, group.row0, m.cs);
                                gather_rows16(m.mixed_h, m.src_dev + group.row0, m.Xs, group.rows, N, m.cs);
                                pt.mark(kPfGemmGU, cs);
                                if (!Gemm::f16_groups(m.Xs, gu, GU_ELEMS, m.GU, group.bounds.data(), G, 1280, N, m.cs))
                                    strata::kernels::xmx_gemm_grouped(m.Xs, gu, GU_ELEMS, m.GU, m.compact_xb_dev, G,
                                                                      group.max_rows, 1280, N, m.cs);
                                swiglu_interleaved(m.GU, m.Hh, group.rows, m.cs);
                                pt.mark(kPfGemmD, cs);
                                if (!Gemm::f16_groups(m.Hh, dn, D_ELEMS, dst, group.bounds.data(), G, N, 640, m.cs))
                                    strata::kernels::xmx_gemm_grouped(m.Hh, dn, D_ELEMS, dst, m.compact_xb_dev, G,
                                                                      group.max_rows, N, 640, m.cs);
                            }
                            return;
                        }
                        int64_t maxr = 0;
                        for (size_t i = j0; i < j1; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                        const int G = (int) (j1 - j0);
                        const int64_t r0 = m.xb_host[j0], nr = m.xb_host[j1] - r0;
                        pt.mark(kPfGemmGU, cs);
                        if (!Gemm::f16_groups(m.Xs, m.dq_gu, GU_ELEMS, m.GU, m.xb_host.data() + j0, G, 1280, N, m.cs))
                            strata::kernels::xmx_gemm_grouped(m.Xs, m.dq_gu, GU_ELEMS, m.GU, m.xb_dev + j0, G, maxr,
                                                              1280, N, m.cs);
                        swiglu_interleaved(m.GU + r0 * 1280, m.Hh + r0 * 640, nr, m.cs);
                        pt.mark(kPfGemmD, cs);
                        if (!Gemm::f16_groups(m.Hh, m.dq_d, D_ELEMS, m.Dm, m.xb_host.data() + j0, G, N, 640, m.cs))
                            strata::kernels::xmx_gemm_grouped(m.Hh, m.dq_d, D_ELEMS, m.Dm, m.xb_dev + j0, G, maxr, N,
                                                              640, m.cs);
                    };
                    // Optional native dequant batching, OFF unless explicitly requested.
                    const bool native_batch = [&] {
                        const char* v = std::getenv("STRATA_PREFILL_NATIVE_DEQUANT_BATCH");
                        return v && std::strtol(v, nullptr, 10) != 0 && lay.native && !use_mmq;
                    }();
                    if (native_batch) {
                    if (m.g->n_expert <= 0 || m.g->n_expert > std::numeric_limits<int32_t>::max())
                        throw std::runtime_error("prefill native batch expert count outside int32");
                    const int32_t nd_expert_end = (int32_t) m.g->n_expert;
                    const uint8_t* nd_blob[16]{};
                    int nd_slot[16]{};
                    int nd_n = 0, nd_q0 = 0;
                    size_t nd_consumed = consumed;
                    uint64_t nd_ga = 0, nd_gd = 0, nd_da = 0, nd_dd = 0;
                    uint64_t nd_ge = 0, nd_de = 0, nd_gfe = 0, nd_dfe = 0, nd_flushes = 0, nd_reuse = 0, nd_released = 0;
                    auto native_flush = [&] {
                        if (!native_batch || nd_n == 0) return;
                        const auto& f = lay.fmt[(size_t) l];
                        const size_t limit = std::numeric_limits<size_t>::max();
                        // NativeFmt supplies real row payloads, not padding or an inferred common type.
                        if (f.n_ff <= 0 || f.n_embd <= 0 || f.gu_row > limit / (size_t) f.n_ff ||
                            f.d_row > limit / (size_t) f.n_embd)
                            throw std::runtime_error("prefill native batch invalid source geometry");
                        const size_t gb = f.gu_row * (size_t) f.n_ff, db = f.d_row * (size_t) f.n_embd;
                        const size_t blob_bytes = (size_t) lay.blob_bytes(l);
                        if (f.bytes > blob_bytes || gb > f.up_off || f.up_off > f.down_off ||
                            gb > f.down_off - f.up_off || f.down_off > f.bytes || db > f.bytes - f.down_off ||
                            nd_q0 < 0 || nd_n > 16 || nd_q0 > g_expert_group - nd_n ||
                            (size_t) g_expert_group > limit / (size_t) GU_ELEMS / sizeof(uint16_t) ||
                            (size_t) g_expert_group > limit / (size_t) D_ELEMS / sizeof(uint16_t) ||
                            (size_t) f.n_ff > (size_t) D_ELEMS / (size_t) f.n_embd ||
                            (size_t) f.n_ff * (size_t) f.n_embd != (size_t) D_ELEMS ||
                            (size_t) GU_ELEMS != 2 * (size_t) D_ELEMS)
                            throw std::runtime_error("prefill native batch source/destination bounds");
                        strata::kernels::IqDequantGroup ng;
                        ng.n = nd_n;
                        for (int i = 0; i < nd_n; ++i) {
                            ng.gate[i] = nd_blob[i]; ng.up[i] = nd_blob[i] + f.up_off;
                            ng.down[i] = nd_blob[i] + f.down_off;
                            ng.gate_bytes[i] = gb; ng.up_bytes[i] = gb; ng.down_bytes[i] = db;
                        }
                        // Rebasing the destination makes API windows <=16 while preserving q in groups <=64.
                        uint16_t* gu = m.dq_gu + (size_t) nd_q0 * GU_ELEMS;
                        uint16_t* dn = m.dq_d + (size_t) nd_q0 * D_ELEMS;
                        const size_t gu_cap = (size_t) (g_expert_group - nd_q0) * GU_ELEMS * sizeof(uint16_t);
                        const size_t dn_cap = (size_t) (g_expert_group - nd_q0) * D_ELEMS * sizeof(uint16_t);
                        const bool ga = strata::kernels::iq_dequant_gu_group_f16(f.gu_type, ng, f.n_ff, f.n_embd,
                                                                                 gu, GU_ELEMS, gu_cap, m.cs);
                        if (transfer_counts) { PfRouteCensus::add(ga ? nd_ga : nd_gd); PfRouteCensus::add(ga ? nd_ge : nd_gfe, nd_n); }
                        if (!ga) for (int i = 0; i < nd_n; ++i) {
                            strata::kernels::iq_dequant_gu_f16(f.gu_type, nd_blob[i], nd_blob[i] + f.up_off,
                                                               f.n_ff, f.n_embd, gu + (size_t) i * GU_ELEMS, m.cs);
                            if (census) PfRouteCensus::add(census->native_dequant_gu);
                        }
                        const bool da = strata::kernels::iq_dequant_down_group_f16(f.d_type, ng, f.n_ff, f.n_embd,
                                                                                   dn, D_ELEMS, dn_cap, m.cs);
                        if (transfer_counts) { PfRouteCensus::add(da ? nd_da : nd_dd); PfRouteCensus::add(da ? nd_de : nd_dfe, nd_n); }
                        if (!da) for (int i = 0; i < nd_n; ++i) {
                            strata::kernels::iq_dequant_f16(f.d_type, nd_blob[i] + f.down_off, f.n_embd * f.n_ff,
                                                            dn + (size_t) i * D_ELEMS, m.cs);
                            if (census) PfRouteCensus::add(census->native_dequant_down);
                        }
                        // Checked markers follow BOTH readers. No slot or producer prefix is published early.
                        for (int i = 0; i < nd_n; ++i) if (nd_slot[i] >= 0) {
                            const int sl = nd_slot[i];
                            publication::require(strata::gpu::event_record(m.used[sl], m.cs), "strata::gpu::event_record", strata::gpu::last_error);
                            m.used_of[sl] = sl;
                            if (transfer_counts) PfRouteCensus::add(nd_released);
                        }
                        nd_n = 0;
                        if (transfer_counts) PfRouteCensus::add(nd_flushes);
                        if (nd_consumed > consumed) { consumed = nd_consumed; give_back(consumed); }
                    };
                    // Stage ahead: the copy stream moves blobs host -> device while the compute stream works.
                    int stage_next = 0;
                    std::vector<int> stage_of(order.size(), -1);
                    // the unpinned ones are copied to pinned buffers by the stager's threads, in this order
                    std::vector<int> job_of(order.size(), -1);
                    if (!stream_all) {
                        std::vector<Stager::Job> js;
                        for (size_t j = 0; j < order.size(); ++j) {
                            const int32_t e = order[j];
                            if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                            const bool tr = m.src->transient(l, e);   // copied by the source (never pinned)
                            if (!tr && m.src->pinned(l, e)) continue;
                            const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
                            if (!tr && !b) { err = "prefill: expert source has no blob"; return false; }
                            job_of[j] = (int) js.size();
                            js.push_back({b, (size_t) lay.blob_bytes(l), tr ? m.src : nullptr, (int32_t) l, e});
                        }
                        m.stager->start(std::move(js));
                    }
                    StagerDone stager_done{stream_all ? nullptr : m.stager.get(), m.cs, m.copy};
                    // MMQ in place (MMQ_RING): expert j's blob in ring slot j % MMQ_RING, its group's slots released
                    // together; else the STAGE ring, a slot at a time
                    const bool mmq_direct = use_mmq && m.mring != nullptr;
                    auto slot_ptr = [&](int sl) { return mmq_direct ? m.mring + (size_t) sl * MMQ_SLOT() : m.stage_dev[sl]; };
                    auto stage_one = [&](size_t j) -> bool {
                        const int32_t e = order[j];
                        const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                        if (resident) return true;
                        int sl = (int) (j % MMQ_RING);
                        if (!mmq_direct) {
                            sl = stage_next;
                            stage_next = (stage_next + 1) % STAGE;
                        }
                        const int par = (int) (j / MMQ_GROUP % 2);
                        auto wait_slot = [&] {
                            if (mmq_direct) {
                                if (m.mgrp_live[par]) publication::require(strata::gpu::stream_wait_event(m.copy, m.mgrp_used[par]), "strata::gpu::stream_wait_event", strata::gpu::last_error);
                            } else if (m.stage_live[sl]) {
                                publication::require(strata::gpu::stream_wait_event(m.copy, m.used[m.used_of[sl]]), "strata::gpu::stream_wait_event(m.copy, m.used[m.used_of[sl]])", strata::gpu::last_error);
                            }
                        };
                        const auto th = Clock::now();
                        const bool tr = m.src->transient(l, e);
                        const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
                        if (!tr && !b) { err = "prefill: expert source has no blob"; return false; }
                        const size_t bytes = (size_t) lay.blob_bytes(l);
                        const bool direct = !tr && m.src->pinned(l, e);
                        const auto timing_kind = direct ? IssuerTiming::PinnedDirect :
                            (tr ? IssuerTiming::TransientStaged : IssuerTiming::UnpinnedStaged);
                        if (direct) {
                            // DMA straight from the page-locked arena: the copy stream only waits for the slot
                            wait_slot();
                            publication::require(issuer_timing.measure((size_t) (l - LB), timing_kind, false, [&] { return strata::gpu::copy_async(slot_ptr(sl), b, bytes, m.copy); }), "copy_async", strata::gpu::last_error);
                            if (transfer_counts && bytes != 0) chunk_transfers[(size_t) (l - LB)].add(1, bytes);
                            ++stats_.experts_dma;
                        } else {
                            // copied to a pinned buffer by the stager (waits only if it is behind), then DMA
                            const uint8_t* hb = issuer_timing.measure((size_t) (l - LB), timing_kind, true, [&] { return m.stager->wait(job_of[j]); });
                            if (m.stager->failed.exchange(false)) { err = "prefill: the expert source could not copy a blob"; return false; }
                            wait_slot();
                            publication::require(issuer_timing.measure((size_t) (l - LB), timing_kind, false, [&] { return strata::gpu::copy_async(slot_ptr(sl), hb, bytes, m.copy); }), "copy_async", strata::gpu::last_error);
                            if (transfer_counts && bytes != 0) chunk_transfers[(size_t) (l - LB)].add(1, bytes);
                            m.stager->issued_one(job_of[j], m.copy);
                        }
                        publication::require(strata::gpu::event_record(m.copied[sl], m.copy), "strata::gpu::event_record(m.copied[sl], m.copy)", strata::gpu::last_error);
                        if (!mmq_direct) m.stage_live[sl] = true;
                        stage_of[j] = sl;
                        issuer_timing.entry((size_t) (l - LB), timing_kind, bytes);
                        stats_.ms_experts_host += ms_since(th);
                        ++stats_.experts_streamed;
                        return true;
                    };
                    // The streamed walk gathers an MMQ group in one launch, after one wait on its last streamed copy
                    // (the copy queue is in order), and releases its ring slots with one event (upstream b3096357,
                    // e8de49eb): a wait, a gather and a mark an expert were three submissions each, and the RTX 4070's
                    // copy engine stood idle behind them.  Its slots go back to the issuer only then; a skipped ring
                    // entry first flushes the open group (`flush`), so at most a group's entries are held back.
                    // STRATA_PREFILL_GROUP_GATHER=0: one wait, gather and mark an expert.
                    static const bool group_env = [] {
                        const char* v = std::getenv("STRATA_PREFILL_GROUP_GATHER");
                        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                    }();
                    const bool group_gather = group_env && stream_all && use_mmq && !mmq_direct;
                    if (census) { census->mmq_direct = mmq_direct; census->group_gather = group_gather; }
                    mmq::GatherGroup gg;
                    int gg_slots[MMQ_GROUP];
                    int gg_nslots = 0;            // ring slots gathered by the next flush
                    size_t gg_consumed = consumed;   // what the issuer may have back after the next flush
                    auto flush = [&]() {
                        if (gg.n > gg.first) {
                            const auto& f = lay.fmt[(size_t) l];
                            if (gg_nslots > 0) {
                                pt.mark(kPfWaitCopy, cs);
                                publication::require(strata::gpu::stream_wait_event(m.cs, m.copied[gg_slots[gg_nslots - 1]]), "strata::gpu::stream_wait_event(m.cs, m.copied[gg_slots[gg_nslots - 1]])", strata::gpu::last_error);
                                pt.mark(kPfDequant, cs);
                            }
                            const bool accepted = mmq::gather_native_group(gg, f.up_off, mmq_gub / 2, f.down_off, mmq_db,
                                                                           m.grp_gu, mmq_gub, m.grp_d, mmq_db, m.cs);
                            if (census) PfRouteCensus::add(accepted ? census->group_accepted : census->group_declined);
                            if (!accepted) {
                                for (int i = gg.first; i < gg.n; ++i) {   // not 16-byte aligned: one at a time
                                    const uint8_t* b = gg.blob[i];
                                    mmq::gather_native(b, b + f.up_off, mmq_gub / 2, b + f.down_off, mmq_db,
                                                       m.grp_gu + i * mmq_gub, m.grp_d + i * mmq_db, m.cs);
                                    if (census) PfRouteCensus::add(census->individual_gather);
                                }
                            }
                            if (gg_nslots > 0) {
                                const int rel = gg_slots[gg_nslots - 1];
                                publication::require(strata::gpu::event_record(m.used[rel], m.cs), "strata::gpu::event_record(m.used[rel], m.cs)", strata::gpu::last_error);
                                for (int i = 0; i < gg_nslots; ++i) m.used_of[gg_slots[i]] = rel;
                            }
                            gg_nslots = 0;
                            gg.first = gg.n;
                        }
                        if (gg_consumed > consumed) {
                            consumed = gg_consumed;
                            give_back(consumed);
                        }
                    };
                    // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
                    // expert) is released once the blob is read
                    auto compute = [&](size_t j, const uint8_t* blob_dev, int slot) -> bool {
                        pt.mark(kPfDequant, cs);
                        if (use_mmq) {
                            const size_t q = j % MMQ_GROUP;
                            const auto& f = lay.fmt[(size_t) l];
                            if (mmq_direct) {
                                // a resident expert into its ring slot (the slot's last group is done: same stream)
                                if (slot < 0)
                                    publication::require(strata::gpu::copy_async(slot_ptr((int) (j % MMQ_RING)), blob_dev,
                                                            (size_t) lay.blob_bytes(l), m.cs), "copy_async", strata::gpu::last_error);
                            } else if (group_gather) {
                                gg.blob[q] = blob_dev;
                                gg.n = (int) q + 1;
                                if (slot >= 0) gg_slots[gg_nslots++] = slot;
                                if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
                                flush();
                                gg = mmq::GatherGroup{};
                            } else {
                                // gather the expert into its group slot (its GGUF blocks; mmq_plan takes native packs)
                                mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off,
                                                   mmq_db, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                                if (census) PfRouteCensus::add(census->individual_gather);
                                if (slot >= 0) { publication::require(strata::gpu::event_record(m.used[slot], m.cs), "strata::gpu::event_record", strata::gpu::last_error); m.used_of[slot] = slot; }
                            }
                            if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
                            // the group's products: gate/up, swiglu, the group's H to int8, down
                            const size_t j0 = j - q, g = j0 / MMQ_GROUP, n = order.size();
                            const int ngx = (int) (q + 1);
                            const int64_t r0 = m.bounds_host[j0], nr = m.bounds_host[j + 1] - r0;
                            int64_t maxr = 0;
                            for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                            pt.mark(kPfGemmGU, cs);
                            const uint8_t* ring0 = mmq_direct ? slot_ptr((int) (j0 % MMQ_RING)) : nullptr;
                            mmq::Product gu;
                            gu.w = mmq_direct ? ring0 : m.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N;
                            gu.expert_bytes = mmq_direct ? MMQ_SLOT() : mmq_gub;
                            gu.n = ngx; gu.xq = m.Xq; gu.bounds = m.bounds_dev + j0; gu.ids = m.ids_identity;
                            gu.total_rows = T * K; gu.max_rows = maxr; gu.dst = m.GU; gu.ld_dst = 1280;
                            m.mmq_ctx->run(gu, m.cs);
                            mmq::swiglu(m.GU + r0 * 1280, m.H + r0 * 640, nr, 640, !lay.native, m.cs);
                            pt.mark(kPfGemmD, cs);
                            mmq::quantize(m.H + r0 * 640, nullptr, m.Hq, mmq_dt, 640, 640, nr, m.cs);
                            mmq::Product dn;
                            dn.w = mmq_direct ? ring0 + f.down_off : m.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640;
                            dn.expert_bytes = mmq_direct ? MMQ_SLOT() : mmq_db;
                            dn.n = ngx; dn.xq = m.Hq; dn.bounds = m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1);
                            dn.ids = m.ids_identity; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = m.Dm + r0 * N;
                            dn.ld_dst = N;
                            m.mmq_ctx->run(dn, m.cs);
                            if (mmq_direct) {
                                const int par = (int) (g % 2);
                                publication::require(strata::gpu::event_record(m.mgrp_used[par], m.cs), "strata::gpu::event_record(m.mgrp_used[par], m.cs)", strata::gpu::last_error);
                                m.mgrp_live[par] = true;
                            }
                            return true;
                        }
                        // dequantize into the group's slot q; the group's products once it is full (or the last)
                        const int G = g_expert_group, q = (int) ((j - nf) % (size_t) G);   // groups from nf on
                        uint16_t *gu16 = m.dq_gu + q * GU_ELEMS, *d16 = m.dq_d + q * D_ELEMS;
                        if (native_batch) {
                            if (nd_n == 16) native_flush();
                            if (nd_n == 0) nd_q0 = q;
                            if (q != nd_q0 + nd_n) throw std::runtime_error("prefill native batch noncontiguous window");
                            nd_blob[nd_n] = blob_dev; nd_slot[nd_n] = slot; ++nd_n;
                        } else if (lay.native) {
                            // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas
                            const auto& f = lay.fmt[(size_t) l];
                            strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                                               gu16, m.cs);
                            if (census) PfRouteCensus::add(census->native_dequant_gu);
                            strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, d16, m.cs);
                            if (census) PfRouteCensus::add(census->native_dequant_down);
                        } else {
                            blob_dequant_f16(blob_dev, gu16, d16, m.cs);
                        }
                        if (!native_batch && slot >= 0) { publication::require(strata::gpu::event_record(m.used[slot], m.cs), "strata::gpu::event_record", strata::gpu::last_error); m.used_of[slot] = slot; }
                        if (native_batch && (nd_n == 16 || (stream_all && nd_consumed > consumed && nd_consumed - consumed >= (size_t) m.ring))) native_flush();
                        if (q + 1 < G && j + 1 < order.size()) return true;
                        if (native_batch) native_flush();   // weights must be expanded before the unchanged GEMMs
                        const size_t j0 = j - (size_t) q;
                        fallback_products(j0, j + 1);
                        return true;
                    };
                    if (!stream_all) {
                        size_t staged = nf;
                        const size_t lookahead = mmq_direct ? (size_t) MMQ_GROUP : (size_t) STAGE - 1;
                        for (size_t j = nf; j < order.size(); ++j) {
                            while (staged < order.size() && staged <= j + lookahead) {
                                if (native_batch) {
                                    const int32_t se = order[staged];
                                    const bool sr = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + se] >= 0;
                                    bool held = false;
                                    if (!sr) for (int i = 0; i < nd_n; ++i) held = held || nd_slot[i] == stage_next;
                                    if (held) { if (transfer_counts) PfRouteCensus::add(nd_reuse); native_flush(); }
                                }
                                if (!stage_one(staged)) return false;
                                ++staged;
                            }
                            const int32_t e = order[j];
                            if (stage_of[j] < 0) {
                                ++stats_.experts_resident;
                                if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                            } else {
                                pt.mark(kPfWaitCopy, cs);
                                publication::require(strata::gpu::stream_wait_event(m.cs, m.copied[stage_of[j]]), "strata::gpu::stream_wait_event(m.cs, m.copied[stage_of[j]])", strata::gpu::last_error);
                                if (!compute(j, slot_ptr(stage_of[j]), stage_of[j])) return false;
                            }
                        }
                    } else {
                        // the streamed walk: this layer's entries [k, kend) in id order; an entry the routing did not
                        // pick only gives its slot back
                        size_t k = seq_start[(size_t) l];
                        const size_t kend = seq_start[(size_t) l + 1];
                        auto release_to = [&](int32_t e_stop) {
                            while (k < kend && seq[k].e < e_stop) {
                                publication::skipped_entry(
                                    [&] { if (native_batch) native_flush(); if (group_gather) flush(); },
                                    [&] { wait_issued(k); },
                                    [&] {
                                        publication::require(strata::gpu::event_record(m.used[k % (size_t) m.ring], m.cs), "skipped entry event_record", strata::gpu::last_error);
                                        m.used_of[k % (size_t) m.ring] = (int) (k % (size_t) m.ring);
                                        consumed = ++k;
                                        give_back(consumed);
                                    });
                            }
                        };
                        for (size_t j = nf; j < order.size(); ++j) {   // from nf on: id order again
                            const int32_t e = order[j];
                            release_to(e);
                            if (k < kend && seq[k].e == e) {
                                const int sl = (int) (k % (size_t) m.ring);
                                pt.mark(kPfWaitCopy, cs);
                                if (native_batch && k - consumed >= (size_t) m.ring) native_flush();
                                wait_issued(k);
                                if (native_batch) nd_consumed = k + 1;
                                if (group_gather) {
                                    gg_consumed = k + 1;   // given back by the flush that gathers it
                                } else {
                                    publication::require(strata::gpu::stream_wait_event(m.cs, m.copied[sl]), "strata::gpu::stream_wait_event(m.cs, m.copied[sl])", strata::gpu::last_error);
                                }
                                if (!compute(j, m.stage_dev[sl], sl)) return false;
                                ++k;
                                if (!group_gather && !native_batch) {
                                    consumed = k;
                                    give_back(consumed);
                                }
                            } else {
                                ++stats_.experts_resident;
                                if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                            }
                        }
                        release_to(nd_expert_end);
                    }
                    if (native_batch) native_flush();
                    if (native_batch && transfer_counts)
                        std::fprintf(stderr, "[prefill-native-dequant-group-v2] device=%d lb=%lld le=%lld chunk_offset=%lld pos=%lld tokens=%lld layer=%lld fused_prefix=%llu cpu_experts=%lld gpu_order=%llu gu_admitted=%llu gu_declined=%llu down_admitted=%llu down_declined=%llu gu_covered=%llu down_covered=%llu gu_fallback_experts=%llu down_fallback_experts=%llu flushes=%llu held_reuse_flushes=%llu released_slots=%llu\n",
                                     m.device, (long long) LB, (long long) LE, (long long) c0, (long long) p0, (long long) T,
                                     (long long) l, (unsigned long long) nf, (long long) n_cpu, (unsigned long long) order.size(),
                                     (unsigned long long) nd_ga, (unsigned long long) nd_gd,
                                     (unsigned long long) nd_da, (unsigned long long) nd_dd, (unsigned long long) nd_ge,
                                     (unsigned long long) nd_de, (unsigned long long) nd_gfe, (unsigned long long) nd_dfe, (unsigned long long) nd_flushes, (unsigned long long) nd_reuse,
                                     (unsigned long long) nd_released);
                    stager_done.complete();
                    } else {
                    // Stage ahead: the copy stream moves blobs host -> device while the compute stream works.
                    int stage_next = 0;
                    std::vector<int> stage_of(order.size(), -1);
                    // the unpinned ones are copied to pinned buffers by the stager's threads, in this order
                    std::vector<int> job_of(order.size(), -1);
                    if (!stream_all) {
                        std::vector<Stager::Job> js;
                        for (size_t j = 0; j < order.size(); ++j) {
                            const int32_t e = order[j];
                            if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                            const bool tr = m.src->transient(l, e);   // copied by the source (never pinned)
                            if (!tr && m.src->pinned(l, e)) continue;
                            const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
                            if (!tr && !b) { err = "prefill: expert source has no blob"; return false; }
                            job_of[j] = (int) js.size();
                            js.push_back({b, (size_t) lay.blob_bytes(l), tr ? m.src : nullptr, (int32_t) l, e});
                        }
                        m.stager->start(std::move(js));
                    }
                    StagerDone stager_done{stream_all ? nullptr : m.stager.get(), m.cs, m.copy};
                    // MMQ in place (MMQ_RING): expert j's blob in ring slot j % MMQ_RING, its group's slots released
                    // together; else the STAGE ring, a slot at a time
                    const bool mmq_direct = use_mmq && m.mring != nullptr;
                    auto slot_ptr = [&](int sl) { return mmq_direct ? m.mring + (size_t) sl * MMQ_SLOT() : m.stage_dev[sl]; };
                    auto stage_one = [&](size_t j) -> bool {
                        const int32_t e = order[j];
                        const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                        if (resident) return true;
                        int sl = (int) (j % MMQ_RING);
                        if (!mmq_direct) {
                            sl = stage_next;
                            stage_next = (stage_next + 1) % STAGE;
                        }
                        const int par = (int) (j / MMQ_GROUP % 2);
                        auto wait_slot = [&] {
                            if (mmq_direct) {
                                if (m.mgrp_live[par]) publication::require(strata::gpu::stream_wait_event(m.copy, m.mgrp_used[par]), "strata::gpu::stream_wait_event", strata::gpu::last_error);
                            } else if (m.stage_live[sl]) {
                                publication::require(strata::gpu::stream_wait_event(m.copy, m.used[m.used_of[sl]]), "strata::gpu::stream_wait_event(m.copy, m.used[m.used_of[sl]])", strata::gpu::last_error);
                            }
                        };
                        const auto th = Clock::now();
                        const bool tr = m.src->transient(l, e);
                        const uint8_t* b = tr ? nullptr : m.src->blob(l, e);
                        if (!tr && !b) { err = "prefill: expert source has no blob"; return false; }
                        const size_t bytes = (size_t) lay.blob_bytes(l);
                        const bool direct = !tr && m.src->pinned(l, e);
                        const auto timing_kind = direct ? IssuerTiming::PinnedDirect :
                            (tr ? IssuerTiming::TransientStaged : IssuerTiming::UnpinnedStaged);
                        if (direct) {
                            // DMA straight from the page-locked arena: the copy stream only waits for the slot
                            wait_slot();
                            publication::require(issuer_timing.measure((size_t) (l - LB), timing_kind, false, [&] { return strata::gpu::copy_async(slot_ptr(sl), b, bytes, m.copy); }), "copy_async", strata::gpu::last_error);
                            if (transfer_counts && bytes != 0) chunk_transfers[(size_t) (l - LB)].add(1, bytes);
                            ++stats_.experts_dma;
                        } else {
                            // copied to a pinned buffer by the stager (waits only if it is behind), then DMA
                            const uint8_t* hb = issuer_timing.measure((size_t) (l - LB), timing_kind, true, [&] { return m.stager->wait(job_of[j]); });
                            if (m.stager->failed.exchange(false)) { err = "prefill: the expert source could not copy a blob"; return false; }
                            wait_slot();
                            publication::require(issuer_timing.measure((size_t) (l - LB), timing_kind, false, [&] { return strata::gpu::copy_async(slot_ptr(sl), hb, bytes, m.copy); }), "copy_async", strata::gpu::last_error);
                            if (transfer_counts && bytes != 0) chunk_transfers[(size_t) (l - LB)].add(1, bytes);
                            m.stager->issued_one(job_of[j], m.copy);
                        }
                        publication::require(strata::gpu::event_record(m.copied[sl], m.copy), "strata::gpu::event_record(m.copied[sl], m.copy)", strata::gpu::last_error);
                        if (!mmq_direct) m.stage_live[sl] = true;
                        stage_of[j] = sl;
                        issuer_timing.entry((size_t) (l - LB), timing_kind, bytes);
                        stats_.ms_experts_host += ms_since(th);
                        ++stats_.experts_streamed;
                        return true;
                    };
                    // The streamed walk gathers an MMQ group in one launch, after one wait on its last streamed copy
                    // (the copy queue is in order), and releases its ring slots with one event (upstream b3096357,
                    // e8de49eb): a wait, a gather and a mark an expert were three submissions each, and the RTX 4070's
                    // copy engine stood idle behind them.  Its slots go back to the issuer only then; a skipped ring
                    // entry first flushes the open group (`flush`), so at most a group's entries are held back.
                    // STRATA_PREFILL_GROUP_GATHER=0: one wait, gather and mark an expert.
                    static const bool group_env = [] {
                        const char* v = std::getenv("STRATA_PREFILL_GROUP_GATHER");
                        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                    }();
                    const bool group_gather = group_env && stream_all && use_mmq && !mmq_direct;
                    if (census) { census->mmq_direct = mmq_direct; census->group_gather = group_gather; }
                    mmq::GatherGroup gg;
                    int gg_slots[MMQ_GROUP];
                    int gg_nslots = 0;            // ring slots gathered by the next flush
                    size_t gg_consumed = consumed;   // what the issuer may have back after the next flush
                    auto flush = [&]() {
                        if (gg.n > gg.first) {
                            const auto& f = lay.fmt[(size_t) l];
                            if (gg_nslots > 0) {
                                pt.mark(kPfWaitCopy, cs);
                                publication::require(strata::gpu::stream_wait_event(m.cs, m.copied[gg_slots[gg_nslots - 1]]), "strata::gpu::stream_wait_event(m.cs, m.copied[gg_slots[gg_nslots - 1]])", strata::gpu::last_error);
                                pt.mark(kPfDequant, cs);
                            }
                            const bool accepted = mmq::gather_native_group(gg, f.up_off, mmq_gub / 2, f.down_off, mmq_db,
                                                                           m.grp_gu, mmq_gub, m.grp_d, mmq_db, m.cs);
                            if (census) PfRouteCensus::add(accepted ? census->group_accepted : census->group_declined);
                            if (!accepted) {
                                for (int i = gg.first; i < gg.n; ++i) {   // not 16-byte aligned: one at a time
                                    const uint8_t* b = gg.blob[i];
                                    mmq::gather_native(b, b + f.up_off, mmq_gub / 2, b + f.down_off, mmq_db,
                                                       m.grp_gu + i * mmq_gub, m.grp_d + i * mmq_db, m.cs);
                                    if (census) PfRouteCensus::add(census->individual_gather);
                                }
                            }
                            if (gg_nslots > 0) {
                                const int rel = gg_slots[gg_nslots - 1];
                                publication::require(strata::gpu::event_record(m.used[rel], m.cs), "strata::gpu::event_record(m.used[rel], m.cs)", strata::gpu::last_error);
                                for (int i = 0; i < gg_nslots; ++i) m.used_of[gg_slots[i]] = rel;
                            }
                            gg_nslots = 0;
                            gg.first = gg.n;
                        }
                        if (gg_consumed > consumed) {
                            consumed = gg_consumed;
                            give_back(consumed);
                        }
                    };
                    // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
                    // expert) is released once the blob is read
                    auto compute = [&](size_t j, const uint8_t* blob_dev, int slot) -> bool {
                        pt.mark(kPfDequant, cs);
                        if (use_mmq) {
                            const size_t q = j % MMQ_GROUP;
                            const auto& f = lay.fmt[(size_t) l];
                            if (mmq_direct) {
                                // a resident expert into its ring slot (the slot's last group is done: same stream)
                                if (slot < 0)
                                    publication::require(strata::gpu::copy_async(slot_ptr((int) (j % MMQ_RING)), blob_dev,
                                                            (size_t) lay.blob_bytes(l), m.cs), "copy_async", strata::gpu::last_error);
                            } else if (group_gather) {
                                gg.blob[q] = blob_dev;
                                gg.n = (int) q + 1;
                                if (slot >= 0) gg_slots[gg_nslots++] = slot;
                                if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
                                flush();
                                gg = mmq::GatherGroup{};
                            } else {
                                // gather the expert into its group slot (its GGUF blocks; mmq_plan takes native packs)
                                mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off,
                                                   mmq_db, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                                if (census) PfRouteCensus::add(census->individual_gather);
                                if (slot >= 0) { publication::require(strata::gpu::event_record(m.used[slot], m.cs), "strata::gpu::event_record", strata::gpu::last_error); m.used_of[slot] = slot; }
                            }
                            if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
                            // the group's products: gate/up, swiglu, the group's H to int8, down
                            const size_t j0 = j - q, g = j0 / MMQ_GROUP, n = order.size();
                            const int ngx = (int) (q + 1);
                            const int64_t r0 = m.bounds_host[j0], nr = m.bounds_host[j + 1] - r0;
                            int64_t maxr = 0;
                            for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                            pt.mark(kPfGemmGU, cs);
                            const uint8_t* ring0 = mmq_direct ? slot_ptr((int) (j0 % MMQ_RING)) : nullptr;
                            mmq::Product gu;
                            gu.w = mmq_direct ? ring0 : m.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N;
                            gu.expert_bytes = mmq_direct ? MMQ_SLOT() : mmq_gub;
                            gu.n = ngx; gu.xq = m.Xq; gu.bounds = m.bounds_dev + j0; gu.ids = m.ids_identity;
                            gu.total_rows = T * K; gu.max_rows = maxr; gu.dst = m.GU; gu.ld_dst = 1280;
                            m.mmq_ctx->run(gu, m.cs);
                            mmq::swiglu(m.GU + r0 * 1280, m.H + r0 * 640, nr, 640, !lay.native, m.cs);
                            pt.mark(kPfGemmD, cs);
                            mmq::quantize(m.H + r0 * 640, nullptr, m.Hq, mmq_dt, 640, 640, nr, m.cs);
                            mmq::Product dn;
                            dn.w = mmq_direct ? ring0 + f.down_off : m.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640;
                            dn.expert_bytes = mmq_direct ? MMQ_SLOT() : mmq_db;
                            dn.n = ngx; dn.xq = m.Hq; dn.bounds = m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1);
                            dn.ids = m.ids_identity; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = m.Dm + r0 * N;
                            dn.ld_dst = N;
                            m.mmq_ctx->run(dn, m.cs);
                            if (mmq_direct) {
                                const int par = (int) (g % 2);
                                publication::require(strata::gpu::event_record(m.mgrp_used[par], m.cs), "strata::gpu::event_record(m.mgrp_used[par], m.cs)", strata::gpu::last_error);
                                m.mgrp_live[par] = true;
                            }
                            return true;
                        }
                        // dequantize into the group's slot q; the group's products once it is full (or the last)
                        const int G = g_expert_group, q = (int) ((j - nf) % (size_t) G);   // groups from nf on
                        uint16_t *gu16 = m.dq_gu + q * GU_ELEMS, *d16 = m.dq_d + q * D_ELEMS;
                        if (lay.native) {
                            // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas
                            const auto& f = lay.fmt[(size_t) l];
                            strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                                               gu16, m.cs);
                            if (census) PfRouteCensus::add(census->native_dequant_gu);
                            strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, d16, m.cs);
                            if (census) PfRouteCensus::add(census->native_dequant_down);
                        } else {
                            blob_dequant_f16(blob_dev, gu16, d16, m.cs);
                        }
                        if (slot >= 0) { publication::require(strata::gpu::event_record(m.used[slot], m.cs), "strata::gpu::event_record", strata::gpu::last_error); m.used_of[slot] = slot; }
                        if (q + 1 < G && j + 1 < order.size()) return true;
                        const size_t j0 = j - (size_t) q;
                        fallback_products(j0, j + 1);
                        return true;
                    };
                    if (!stream_all) {
                        size_t staged = nf;
                        const size_t lookahead = mmq_direct ? (size_t) MMQ_GROUP : (size_t) STAGE - 1;
                        for (size_t j = nf; j < order.size(); ++j) {
                            while (staged < order.size() && staged <= j + lookahead) {
                                if (!stage_one(staged)) return false;
                                ++staged;
                            }
                            const int32_t e = order[j];
                            if (stage_of[j] < 0) {
                                ++stats_.experts_resident;
                                if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                            } else {
                                pt.mark(kPfWaitCopy, cs);
                                publication::require(strata::gpu::stream_wait_event(m.cs, m.copied[stage_of[j]]), "strata::gpu::stream_wait_event(m.cs, m.copied[stage_of[j]])", strata::gpu::last_error);
                                if (!compute(j, slot_ptr(stage_of[j]), stage_of[j])) return false;
                            }
                        }
                    } else {
                        // the streamed walk: this layer's entries [k, kend) in id order; an entry the routing did not
                        // pick only gives its slot back
                        size_t k = seq_start[(size_t) l];
                        const size_t kend = seq_start[(size_t) l + 1];
                        auto release_to = [&](int32_t e_stop) {
                            while (k < kend && seq[k].e < e_stop) {
                                publication::skipped_entry(
                                    [&] { if (group_gather) flush(); },
                                    [&] { wait_issued(k); },
                                    [&] {
                                        publication::require(strata::gpu::event_record(m.used[k % (size_t) m.ring], m.cs), "skipped entry event_record", strata::gpu::last_error);
                                        m.used_of[k % (size_t) m.ring] = (int) (k % (size_t) m.ring);
                                        consumed = ++k;
                                        give_back(consumed);
                                    });
                            }
                        };
                        for (size_t j = nf; j < order.size(); ++j) {   // from nf on: id order again
                            const int32_t e = order[j];
                            release_to(e);
                            if (k < kend && seq[k].e == e) {
                                const int sl = (int) (k % (size_t) m.ring);
                                pt.mark(kPfWaitCopy, cs);
                                wait_issued(k);
                                if (group_gather) {
                                    gg_consumed = k + 1;   // given back by the flush that gathers it
                                } else {
                                    publication::require(strata::gpu::stream_wait_event(m.cs, m.copied[sl]), "strata::gpu::stream_wait_event(m.cs, m.copied[sl])", strata::gpu::last_error);
                                }
                                if (!compute(j, m.stage_dev[sl], sl)) return false;
                                ++k;
                                if (!group_gather) {
                                    consumed = k;
                                    give_back(consumed);
                                }
                            } else {
                                ++stats_.experts_resident;
                                if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                            }
                        }
                        release_to(m.g->n_expert);
                    }
                    stager_done.complete();
                    }
                    pt.mark(kPfCombine, cs);
                    if (cpu_fut.valid()) {   // set_cpu_pool: the CPU's rows into Dm's tail
                        if (cpu_share_env() < 0.0) publication::require(strata::gpu::event_record(m.cpu_ev[1], m.cs), "strata::gpu::event_record", strata::gpu::last_error);   // after the GPU's experts
                        if (!cpu_fut.get()) { err = "prefill: the CPU experts failed"; return false; }
                        if (cpu_share_env() < 0.0 && n_stream > n_cpu) {
                            m.cpu_pend = true;
                            m.pend_cpu_ms = cpu_ms;
                            m.pend_n_cpu = n_cpu;
                            m.pend_n_gpu = n_stream - n_cpu;
                        }
                        stats_.cpu_share = cpu_share_env() >= 0.0 ? cpu_share_env() : m.cpu_share_now;
                        publication::require(strata::gpu::copy_async(m.Dm + (size_t) rows_gpu * N, m.cpu_rows, (size_t) rows_cpu * N * sizeof(float),
                                                m.cs), "copy_async", strata::gpu::last_error);
                        stats_.experts_cpu += n_cpu;
                        if (static bool said = false; !said) {
                            said = true;
                            std::fprintf(stderr, "strata: STRATA_PREFILL_CPU_SHARE: the CPU pool computes %lld of a layer's "
                                         "%lld streamed experts (share %.2f)\n", (long long) n_cpu, (long long) n_stream,
                                         stats_.cpu_share);
                        }
                    }
                    moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);
                    // debug: STRATA_DBG_NAN=1 reports the first layer of a chunk whose MoE produced non-finite values
                    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {
                        publication::cleanup_require(strata::gpu::stream_sync(m.cs), "strata::gpu::stream_sync(m.cs)", strata::gpu::last_error);
                        auto bad = [&](const float* d, int64_t n) {
                            std::vector<float> h((size_t) n);
                            publication::cleanup_require(strata::gpu::copy(h.data(), d, (size_t) n * 4), "diagnostic copy", strata::gpu::last_error);
                            int64_t c = 0;
                            for (float v : h) c += !std::isfinite(v);
                            return c;
                        };
                        // (the CPU's rows of GU and H are never written by the GPU: not counted, upstream 4322e241)
                        const int64_t bgu = bad(m.GU, rows_gpu * 1280), bdm = bad(m.Dm, T * K * N), bbo = bad(m.bo, T * N);
                        const int64_t bh = m.H ? bad(m.H, rows_gpu * 640) : -1;
                        static int64_t reported = -1;
                        if ((bgu || bdm || bbo || bh > 0) && reported != stats_.chunks) {
                            reported = stats_.chunks;
                            std::fprintf(stderr, "strata dbg: layer %lld (mmq %d, types %d/%d, %zu experts): non-finite GU %lld "
                                         "H %lld Dm %lld bo %lld of T %lld\n", (long long) l, (int) use_mmq, mmq_gt, mmq_dt,
                                         order.size(), (long long) bgu, (long long) bh, (long long) bdm, (long long) bbo,
                                         (long long) T);
                        }
                    }
                }
                // ---- the hyper-connection write of this half; F-2: fused with the next half's norm when nothing else
                // touches R in between (not the stage's last half, not before the PLE block of layer 1, not under a
                // control vector)
                const int64_t nl = half == 0 ? l : l + 1;
                // a steered layer's write, vector and next norm as one pass too (upstream 6448f0f7; the same bits);
                // STRATA_CVEC_FUSE=0: gr_write, cvec_apply, then the next half's own norm
                static const bool cvec_fuse = [] {
                    const char* e = std::getenv("STRATA_CVEC_FUSE");
                    return e == nullptr || std::strtol(e, nullptr, 10) != 0;
                }();
                const bool steered = half == 1 && strata::kernels::cvec().covers(l);
                const bool fuse = nl < LE && !(half == 1 && nl == 1 && ple_on) && (!steered || cvec_fuse);
                const core::WeightRef* wnn = nullptr;
                if (fuse) {
                    const core::LayerView vn(*m.wt, nl);
                    wnn = need(vn, half == 0 ? "hc_ffn_norm.weight" : "hc_attn_norm.weight", err);
                    if (!wnn) return false;
                }
                if (wnn && steered) {
                    gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, l, (const float*) wnn->data, EPS, m.xn, m.xn16, T, m.cs,
                                          m.xn16_lo);
                    normed = true;
                } else if (wnn) {
                    gr_write_norm_rs(m.R, m.bo, m.inj, HC, (const float*) wnn->data, EPS, m.xn, m.xn16, T, m.cs,
                                     m.xn16_lo);
                    normed = true;
                } else {
                    gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
                    if (steered)   // --control-vector-scaled
                        strata::kernels::cvec_apply(m.R, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
                }
            }
        }
        if (!ple_land()) return false;   // a stage that ends before layer 1: the rows land anyway, the next gather starts
        if (issuer.joinable()) {
            issuer.join();
            issuer_failure.rethrow();
            stats_.ms_experts_host += iss_ms;
            stats_.experts_streamed += iss_streamed;
            stats_.experts_dma += iss_dma;
        }
        if (m.stager && m.stager->failed.exchange(false)) {   // a transient source's copy_blob failed
            err = "prefill: the expert source could not copy a blob";
            return false;
        }
        chunk_stager_done.complete();
        issuer_timing.emit(m.device, LB, LE, c0, p0, T, stream_all, threaded_issue, m.ring,
                           (uint64_t) (stats_.experts_streamed - issuer_streamed_begin));
        if (transfer_counts)
            transfer_chunks.push_back({c0, p0, T, stream_all, threaded_issue, std::move(chunk_transfers),
                                       std::move(chunk_census)});
        if (!m.layer_major_active() || LE == g.n_layers) stats_.tokens += T;
        core::progress_at("reading the prompt (batched): finishing the chunk from token", p0);
        pt.mark(kPfStart, cs);
        if (next_ != nullptr) {
            // the rows to the host buffer the next stage read two chunks ago (it has finished: waited below)
            float* h = m.hand[hand_buf];
            if (!strata::gpu::copy_async(h, m.R, (size_t) T * D * 4, m.cs) ||
                !strata::gpu::stream_sync(m.cs)) {
                err = std::string("prefill: the layer split's hand-off: ") + strata::gpu::last_error();
                return false;
            }
            // this stage's state is at the chunk's end now (synced) and moves on with the next chunk below
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (next_run.valid() && !next_run.get()) { err = next_err; return false; }
            next_->hand_in_ = h;
            next_run = std::async(std::launch::async, [this, tokens, c0, T, p0, &next_err] {
                return next_->run(tokens + c0, T, p0, next_err);
            });
            hand_buf ^= 1;
            continue;   // the last stage reports the chunk (on_chunk)
        }
        if (const char* dump = std::getenv("STRATA_PREFILL_DUMP_R")) {   // debug: the final residuals, every 64th
            publication::cleanup_require(strata::gpu::stream_sync(m.cs), "strata::gpu::stream_sync(m.cs)", strata::gpu::last_error);                                  // position (A/B quality of this path)
            if (std::FILE* f = std::fopen(dump, c0 == 0 ? "wb" : "ab")) {
                std::vector<float> row((size_t) D);
                for (int64_t t = (64 - p0 % 64) % 64; t < T; t += 64) {
                    publication::cleanup_require(strata::gpu::copy(row.data(), m.R + t * D, (size_t) D * 4), "diagnostic copy", strata::gpu::last_error);
                    const int64_t pos = p0 + t;
                    std::fwrite(&pos, sizeof pos, 1, f);
                    std::fwrite(row.data(), 4, row.size(), f);
                }
                std::fclose(f);
            }
        }
        if (const char* dump = std::getenv("STRATA_PREFILL_DUMP_R_ALL")) {
            // draft-layer distillation data (opt-in, upstream 48d10a49): every position's final multi-stream residual
            // as BF16 (round-to-nearest-even), rows in position order, appended across chunks and requests:
            // [n][hc*n_embd]
            publication::cleanup_require(strata::gpu::stream_sync(m.cs), "strata::gpu::stream_sync(m.cs)", strata::gpu::last_error);
            if (std::FILE* f = std::fopen(dump, "ab")) {
                constexpr int64_t kRows = 512;
                std::vector<float> rows((size_t) (kRows * D));
                std::vector<uint16_t> out((size_t) (kRows * D));
                for (int64_t t0 = 0; t0 < T; t0 += kRows) {
                    const int64_t nr = std::min<int64_t>(kRows, T - t0);
                    publication::cleanup_require(strata::gpu::copy(rows.data(), m.R + t0 * D, (size_t) (nr * D) * 4), "diagnostic copy", strata::gpu::last_error);
                    for (int64_t i = 0; i < nr * D; ++i) {
                        uint32_t u;
                        std::memcpy(&u, &rows[(size_t) i], 4);
                        u += 0x7fffu + ((u >> 16) & 1u);
                        out[(size_t) i] = (uint16_t) (u >> 16);
                    }
                    std::fwrite(out.data(), 2, (size_t) (nr * D), f);
                }
                std::fclose(f);
            }
        }
        if (on_chunk || on_stage_chunk) {
            const auto toc = Clock::now();
            if (!strata::gpu::stream_sync(m.cs)) {
                err = std::string("prefill: ") + strata::gpu::last_error();
                return false;
            }
            const auto toc2 = Clock::now();
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (on_chunk && !on_chunk(m.R, T, p0, err)) return false;
            host_sync_ms += std::chrono::duration<double, std::milli>(toc2 - toc).count();
            host_chunk_ms += ms_since(toc2);
        }
    }
    if (next_run.valid() && !next_run.get()) { err = next_err; return false; }
    ss.ple_prev[0] = prev[0];
    ss.ple_prev[1] = prev[1];
    if (std::getenv("STRATA_DBG_NAN") != nullptr) {   // debug: the state the prompt leaves for the token path
        publication::cleanup_require(strata::gpu::stream_sync(m.cs), "strata::gpu::stream_sync(m.cs)", strata::gpu::last_error);
        auto bad = [&](const float* d, int64_t n) {
            std::vector<float> h((size_t) n);
            publication::cleanup_require(strata::gpu::copy(h.data(), d, (size_t) n * 4), "diagnostic copy", strata::gpu::last_error);
            int64_t c = 0;
            double mx = 0;
            for (float v : h) { c += !std::isfinite(v); if (std::isfinite(v)) mx = std::max(mx, (double) std::fabs(v)); }
            std::fprintf(stderr, " %lld non-finite (max |x| %.3g)", (long long) c, mx);
        };
        const int64_t last = (n - 1) % m.T;
        std::fprintf(stderr, "strata dbg: prompt end: last residual row");
        bad(m.R + last * D, D);
        if (ss.ple.ready()) { std::fprintf(stderr, "; PLE history"); bad(ss.ple.hist, (int64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM); }
        std::fprintf(stderr, "; GDN state 0");
        bad(ss.gdn_state, 64 * 1024);
        std::fprintf(stderr, "\n");
    }
    if (!strata::gpu::stream_sync(m.cs)) {
        err = std::string("prefill: ") + strata::gpu::last_error();
        return false;
    }
    if (!m.layer_major_active()) stats_.ms_total += ms_since(t_start);
    service_scope.complete();  // Existing checked terminal compute sync succeeded; no diagnostic wait.
    if (pt.on) {
        pt.fold();
        double total = 0.0;
        for (double v : pt.ms) total += v;
        std::string line;
        char b[96];
        for (int i = 0; i < kPfCount; ++i) {
            if (pt.ms[i] <= 0.0) continue;
            std::snprintf(b, sizeof b, " %s %.0f (%.1f%%)", kPfNames[i], pt.ms[i], total > 0 ? 100.0 * pt.ms[i] / total : 0.0);
            line += b;
        }
        std::fprintf(stderr, "strata prefill timing: %lld tokens, GPU timeline %.0f ms, wall %.0f ms, host staging %.0f ms:%s\n",
                     (long long) n, total, ms_since(t_start), stats_.ms_experts_host, line.c_str());
        std::fprintf(stderr, "strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) %.0f ms, "
                             "waiting for each chunk %.0f ms, after each chunk (the draft layer, progress) %.0f ms, "
                             "PLE %.0f ms\n", host_setup_ms, host_sync_ms, host_chunk_ms, stats_.ms_ple);
        std::fprintf(stderr, "strata prefill timer overhead: host wall CPU API overhead: mark %.3f ms "
                             "(marker calls %zu, new event creations %zu), fold %.3f ms "
                             "(fold calls %zu, completed intervals %zu); not GPU service or timer calibration\n",
                     pt.mark_host_ms, pt.marker_calls, pt.new_events, pt.fold_host_ms, pt.fold_calls, pt.completed_intervals);
    }
    if (std::getenv("STRATA_STATE_HASH_GDN") != nullptr) {   // debug: the GDN states as the prompt path leaves them
        publication::cleanup_require(strata::gpu::stream_sync(m.cs), "strata::gpu::stream_sync(m.cs)", strata::gpu::last_error);
        std::vector<uint8_t> b((size_t) gdn_floats * 4);
        std::string line;
        char h[8];
        for (int64_t i = 0; i < ss.gdn_alloc; ++i) {   // the GDN layers this session owns
            publication::cleanup_require(strata::gpu::copy(b.data(), ss.gdn_state + (size_t) i * gdn_floats, b.size()), "diagnostic copy", strata::gpu::last_error);
            uint64_t x = 1469598103934665603ull;
            for (uint8_t c : b) x = (x ^ c) * 1099511628211ull;
            std::snprintf(h, sizeof(h), "%04llx ", (unsigned long long) (x & 0xffff));
            line += h;
        }
        std::fprintf(stderr, "strata prefill: GDN_HASH %s\n", line.c_str());
    }
    if (transfer_counts) {
        // All issuer threads have joined, expert copy/stager drains succeeded and
        // the terminal compute drain/optional diagnostics have succeeded. Sum
        // before emitting any ledger so overflow cannot publish a partial summary.
        PfExpertTransfers mode_total[2], total;
        for (const auto& chunk : transfer_chunks)
            for (const auto& layer : chunk.layers)
                mode_total[chunk.stream_all ? 1 : 0].add(layer.calls, layer.bytes);
        for (const auto& mode : mode_total) total.add(mode.calls, mode.bytes);
        const auto& layout = strata::kernels::cpu::expert_layout();
        for (const auto& chunk : transfer_chunks) {
            for (size_t i = 0; i < chunk.layers.size(); ++i) {
                const int64_t l = LB + (int64_t) i;
                const auto& count = chunk.layers[i];
                const int gu_type = layout.native ? layout.fmt[(size_t) l].gu_type : -1;
                const int down_type = layout.native ? layout.fmt[(size_t) l].d_type : -1;
                std::fprintf(stderr, "strata prefill expert H2D: device=%d stage=[%lld,%lld) chunk_offset=%lld pos=%lld "
                                     "tokens=%lld layer=%lld mode=%s issuer=%s format=%s gu_type=%d down_type=%d "
                                     "blob_bytes=%llu submitted_calls=%llu submitted_bytes=%llu\n",
                             m.device, (long long) LB, (long long) LE, (long long) chunk.offset, (long long) chunk.pos,
                             (long long) chunk.tokens, (long long) l, chunk.stream_all ? "full_nonresident_stream" : "routed",
                             chunk.threaded ? "thread" : "inline", layout.native ? "native_gguf" : "canonical_q2_0",
                             gu_type, down_type, (unsigned long long) layout.blob_bytes(l),
                             (unsigned long long) count.calls, (unsigned long long) count.bytes);
            }
        }
        // Separate grammar: successful wrapper returns and host routing facts.
        for (const auto& chunk : transfer_chunks) {
            for (size_t i = 0; i < chunk.census.size(); ++i) {
                const auto& c = chunk.census[i];
                std::fprintf(stderr, "strata prefill route census v1: device=%d stage=[%lld,%lld) chunk_offset=%lld pos=%lld "
                                     "tokens=%lld layer=%lld mode=%s issuer=%s routing_observed=%d mmq_enabled=%d "
                                     "plan_fallback=%d mmq_direct=%d group_gather=%d routed_rows=%llu expected_rows=%llu "
                                     "active_experts=%llu active_nonresident=%llu zero_nonresident=%llu "
                                     "gather_group_accepted_calls=%llu gather_group_declined_calls=%llu gather_individual_calls=%llu "
                                     "native_dequant_gu_calls=%llu native_dequant_down_calls=%llu completion=checked\n",
                             m.device, (long long) LB, (long long) LE, (long long) chunk.offset, (long long) chunk.pos,
                             (long long) chunk.tokens, (long long) LB + (long long) i,
                             chunk.stream_all ? "full_nonresident_stream" : "routed", chunk.threaded ? "thread" : "inline",
                             (int) c.routing_observed, (int) c.mmq_enabled, (int) c.plan_fallback,
                             (int) c.mmq_direct, (int) c.group_gather, (unsigned long long) c.routed_rows,
                             (unsigned long long) c.expected_rows, (unsigned long long) c.active_experts,
                             (unsigned long long) c.active_nonresident, (unsigned long long) c.zero_nonresident,
                             (unsigned long long) c.group_accepted, (unsigned long long) c.group_declined,
                             (unsigned long long) c.individual_gather, (unsigned long long) c.native_dequant_gu,
                             (unsigned long long) c.native_dequant_down);
            }
        }
        std::fprintf(stderr, "strata prefill expert H2D total: scope=this_prefill_stage_run device=%d stage=[%lld,%lld) "
                             "pos=%lld tokens=%lld chunks=%zu routed_calls=%llu routed_bytes=%llu "
                             "full_stream_calls=%llu full_stream_bytes=%llu submitted_calls=%llu submitted_bytes=%llu "
                             "completion=checked; request extents, not PCIe-exclusive bytes or DMA service\n",
                     m.device, (long long) LB, (long long) LE, (long long) pos0, (long long) n, transfer_chunks.size(),
                     (unsigned long long) mode_total[0].calls, (unsigned long long) mode_total[0].bytes,
                     (unsigned long long) mode_total[1].calls, (unsigned long long) mode_total[1].bytes,
                     (unsigned long long) total.calls, (unsigned long long) total.bytes);
    }
    return true;
} catch (const std::exception& e) {
    err = std::string("prefill: ") + e.what();
    if (impl_->cs) publication::cleanup_require(strata::gpu::stream_sync(impl_->cs), "release/catch cs drain", strata::gpu::last_error);
    if (impl_->copy) publication::cleanup_require(strata::gpu::stream_sync(impl_->copy), "release/catch copy drain", strata::gpu::last_error);
    return false;
} catch (...) {
    err = "prefill: unknown asynchronous submission failure";
    if (impl_->cs) publication::cleanup_require(strata::gpu::stream_sync(impl_->cs), "release/catch cs drain", strata::gpu::last_error);
    if (impl_->copy) publication::cleanup_require(strata::gpu::stream_sync(impl_->copy), "release/catch copy drain", strata::gpu::last_error);
    return false;
}

}  // namespace strata::prefill
