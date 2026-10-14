// Proposed source SHA256: 95dca6962025079f38f13714da47bc25e0b44c95b305be674633c7fae32a87e4
// UNRUN CPU fixture. Source placeholders are replaced by prepare_cpu_fixture.py.
// No CUDA/SYCL headers, libraries, GPU, intentional driver error or product import.
// Main compiles after patch application, with -std=c++17 -pthread -I<root>/include.
#define STRATA_ENABLE_SYCL 1
#include "strata/prefill/stage_ownership.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <vector>
#include <cerrno>
#include <csignal>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

enum cudaError_t { cudaSuccess = 0, cudaErrorUnknown = 999 };
using cudaStream_t = void*;
struct FakeEvent { int id; };
using cudaEvent_t = FakeEvent*;
constexpr unsigned cudaHostAllocDefault = 0, cudaEventDisableTiming = 0;
constexpr int cudaMemcpyHostToDevice = 0, RING_MAX = 1024;
enum class Case { WorkSuccess, WaitFailure, DeviceFailure, RecordFailure,
                  IssuerSuccess, CopyFailure, WaitEventFailure, CopiedRecordFailure,
                  ReleaseSuccess, DrainFailure };
struct Stats { int64_t experts_dma = 0, experts_streamed = 0; double ms_experts_host = 0; };
struct Probe {
    Case scenario;
    std::atomic<int> allocations{0}, events{0}, waits{0}, copies{0}, records{0}, wait_events{0}, drains{0}, frees{0};
    std::atomic<int> ready[2]{};
    size_t issuer_issued = 0;
    Stats stats;
    uint8_t host[16][8], device[16][8];
};
static Probe* probe;
static const uint8_t pattern[8] = {1, 2, 3, 4, 5, 6, 7, 8};
const char* cudaGetErrorString(cudaError_t code) { return code == cudaSuccess ? "success" : "fake sticky adapter failure"; }
cudaError_t cudaGetDevice(int* out) { *out = 0; return cudaSuccess; }
cudaError_t cudaSetDevice(int) { return probe->scenario == Case::DeviceFailure ? cudaErrorUnknown : cudaSuccess; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
cudaError_t cudaHostAlloc(void** out, size_t n, unsigned) {
    const int slot = probe->allocations.fetch_add(1);
    if (slot >= 16 || n > 8) return cudaErrorUnknown;
    *out = probe->host[slot]; return cudaSuccess;
}
cudaError_t cudaFreeHost(void*) { probe->frees.fetch_add(1); return cudaSuccess; }
cudaError_t cudaFree(void*) { probe->frees.fetch_add(1); return cudaSuccess; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* out, unsigned) {
    *out = new FakeEvent{probe->events.fetch_add(1)}; return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t event) { delete event; return cudaSuccess; }
cudaError_t cudaEventSynchronize(cudaEvent_t) {
    probe->waits.fetch_add(1);
    // Deliberately no completion action. The result must stop the real worker.
    return probe->scenario == Case::WaitFailure ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t) {
    probe->records.fetch_add(1);
    if (probe->scenario == Case::RecordFailure) return cudaErrorUnknown;
    if (probe->scenario == Case::CopiedRecordFailure && event->id == 100) return cudaErrorUnknown;
    return cudaSuccess;
}
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) {
    probe->wait_events.fetch_add(1);
    return probe->scenario == Case::WaitEventFailure ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, int, cudaStream_t) {
    probe->copies.fetch_add(1);
    if (probe->scenario == Case::CopyFailure) return cudaErrorUnknown;
    std::memcpy(dst, src, bytes); return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
    probe->drains.fetch_add(1);
    return probe->scenario == Case::DrainFailure ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
namespace core {
struct ExpertSource { bool copy_blob(int32_t, int32_t, uint8_t*) { return false; } };
}

struct Stager {
    // D-5: the pinned ring's depth (STRATA_STAGER_RING, default 16) - how far the host copies can run ahead of the
    // DMAs of the unpinned experts' blobs
    int kRing = 16;
    // `from` set: the blob is copied by the source itself (CS-T: a GGUF read in place assembles it from its three
    // role slices; a pointer to it would not live as long as the queue)
    struct Job { const uint8_t* src; size_t bytes; core::ExpertSource* from = nullptr; int32_t l = 0, e = 0; };
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<cudaEvent_t> dma_done;
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
        if (const char* v = std::getenv("STRATA_STAGER_RING")) kRing = std::clamp(std::atoi(v), 2, 256);
        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = cudaHostAlloc((void**) &buf[i], blob_bytes, cudaHostAllocDefault) == cudaSuccess;
            if (!pinned[i]) {
                cudaGetLastError();
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            if (cudaEventCreateWithFlags(&dma_done[i], cudaEventDisableTiming) != cudaSuccess) return false;
        }
        STRATA_PREFILL_STAGE_CHECK(cudaGetDevice(&device), "Stager::init device");
        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] { work(); });
        return true;
    }
    ~Stager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) cudaEventDestroy(dma_done[i]);
            if (buf[i] && pinned[i]) cudaFreeHost(buf[i]);
        }
    }
    void work() {
        STRATA_PREFILL_STAGE_CHECK(cudaSetDevice(device), "Stager::work device");
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
                if (j >= kRing)   // job j - kRing's DMA from this buffer is queued
                    while (issued.load(std::memory_order_acquire) <= j - kRing) std::this_thread::yield();
                // and done - for a generation's first kRing jobs that is the previous generation's last DMA from
                // the buffer, which nothing else waits for when a chunk ends without a sync (no MTP) or the DMA
                // was a ring entry the routing skipped (an event never recorded returns at once)
                STRATA_PREFILL_STAGE_CHECK(cudaEventSynchronize(dma_done[b]), "Stager::work host-slot reuse");
                const Job& jb = jobs[(size_t) j];
                if (jb.from == nullptr) std::memcpy(buf[b], jb.src, jb.bytes);
                else if (!jb.from->copy_blob(jb.l, jb.e, buf[b])) {
                    std::fprintf(stderr, "prefill: the expert source could not copy expert %d of layer %d\n", jb.e, jb.l);
                    std::abort();
                }
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
        if (js.empty()) return;
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
        while (!ready[(size_t) j].load(std::memory_order_acquire)) std::this_thread::yield();
        return buf[j % kRing];
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    void issued_one(int j, cudaStream_t copy) {
        STRATA_PREFILL_STAGE_CHECK(cudaEventRecord(dma_done[j % kRing], copy), "Stager::issued_one host-slot DMA");
        issued.store(j + 1, std::memory_order_release);
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};


using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point) { return 0.0; }
struct FakePayload { void add(size_t, bool, bool) {} };
struct FakeObservation { bool on = false; FakePayload payload; };
struct FakeLayout { size_t blob_bytes(int32_t) const { return 8; } };
void extracted_issuer(Stager* stager) {
    struct StreamEntry { int32_t l, e; const uint8_t* blob; int job; };
    std::vector<StreamEntry> seq{{0, 0, nullptr, 0}};
    size_t& issued = probe->issuer_issued;
    FakeEvent copied{100}, used{200};
    struct M {
        int ring = 1;
        bool stage_live[16]{};
        int used_of[16]{};
        uint8_t* stage_dev[16]{};
        cudaEvent_t copied[16]{}, used[16]{};
        cudaStream_t copy = reinterpret_cast<void*>(1);
        Stager* stager;
    } m;
    m.stager = stager;
    m.stage_dev[0] = probe->device[0];
    m.copied[0] = &copied;
    m.used[0] = &used;
    m.stage_live[0] = true; // reuse an already live device slot: exercise the actual wait
    FakeLayout lay0;
    FakeObservation ro;
    Stats& stats_ = probe->stats;
    const bool copy_events = true;
        auto issue_until = [&](size_t limit) {
            limit = std::min(limit, seq.size());
            while (issued < limit) {
                const StreamEntry& en = seq[issued];
                const int sl = (int) (issued % (size_t) m.ring);
                const auto th = Clock::now();
                const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                if (copy_events && m.stage_live[sl]) STRATA_PREFILL_STAGE_CHECK(cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0), "inline issuer stage copy");
                if (ro.on) ro.payload.add(bytes, en.job < 0, m.stage_live[sl]);
                if (en.job < 0) {
                    STRATA_PREFILL_STAGE_CHECK(cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes, cudaMemcpyHostToDevice, m.copy), "inline issuer stage copy");
                    ++stats_.experts_dma;
                } else {
                    const uint8_t* hb = m.stager->wait(en.job);
                    STRATA_PREFILL_STAGE_CHECK(cudaMemcpyAsync(m.stage_dev[sl], hb, bytes, cudaMemcpyHostToDevice, m.copy), "inline issuer stage copy");
                    m.stager->issued_one(en.job, m.copy);
                }
                if (copy_events) STRATA_PREFILL_STAGE_CHECK(cudaEventRecord(m.copied[sl], m.copy), "inline issuer stage copy");
                m.stage_live[sl] = true;
                stats_.ms_experts_host += ms_since(th);
                ++stats_.experts_streamed;
                ++issued;
            }
        };

    issue_until(1);
}

struct FakeImpl {
    cudaStream_t cs = reinterpret_cast<void*>(1), copy = reinterpret_cast<void*>(1);
    cudaEvent_t copied[RING_MAX]{}, used[RING_MAX]{}, ple_copied[2]{};
    void* hand[2]{}, *ple_emb_host[2]{}, *grp_host = nullptr;
    std::vector<uint8_t> ple_pageable[2];
    std::vector<void*> owned;
};
void extracted_release(FakeImpl* impl_) {
    if (!impl_) return;
    if (impl_->cs) STRATA_PREFILL_STAGE_CHECK(cudaStreamSynchronize(impl_->cs), "Prefill::release compute drain");
    if (impl_->copy) STRATA_PREFILL_STAGE_CHECK(cudaStreamSynchronize(impl_->copy), "Prefill::release copy drain");
    for (int i = 0; i < RING_MAX; ++i) {
        if (impl_->copied[i]) cudaEventDestroy(impl_->copied[i]);
        if (impl_->used[i]) cudaEventDestroy(impl_->used[i]);
    }
    for (int b = 0; b < 2; ++b) {
        if (impl_->hand[b]) cudaFreeHost(impl_->hand[b]);
        if (impl_->ple_copied[b]) cudaEventDestroy(impl_->ple_copied[b]);
        if (impl_->ple_emb_host[b] && impl_->ple_pageable[b].empty()) cudaFreeHost(impl_->ple_emb_host[b]);
    }
    if (impl_->copy && impl_->copy != impl_->cs) cudaStreamDestroy(impl_->copy);
    if (impl_->grp_host) cudaFreeHost(impl_->grp_host);
    for (void* p : impl_->owned) cudaFree(p);
}


struct Shared {
    Probe probe{};
    alignas(Stager) unsigned char storage[sizeof(Stager)];
};

int child_case(Stager* stager) {
    // Child-only env and core-file setting; no effect on the live engine or parent.
    setenv("STRATA_STAGER_RING", "16", 1);
    const rlimit no_core{0, 0};
    if (setrlimit(RLIMIT_CORE, &no_core)) return 20;
    if (!stager->init(8, 1)) return 21;
    // Use mapped ready flags so the parent can inspect them after child termination.
    // start() sees enough capacity and therefore runs its actual code without allocation.
    stager->ready.reset(probe->ready);
    stager->ready_cap = 2;
    switch (probe->scenario) {
    case Case::WorkSuccess:
    case Case::WaitFailure:
    case Case::DeviceFailure:
        stager->start(std::vector<Stager::Job>{{pattern, sizeof pattern}});
        stager->wait(0);
        stager->issued_one(0, reinterpret_cast<void*>(1));
        stager->finish();
        break;
    case Case::RecordFailure:
        stager->issued_one(0, reinterpret_cast<void*>(1));
        break;
    case Case::IssuerSuccess:
    case Case::CopyFailure:
    case Case::WaitEventFailure:
    case Case::CopiedRecordFailure:
        std::memcpy(probe->host[0], pattern, sizeof pattern); // a completed predecessor job
        probe->ready[0].store(1, std::memory_order_release);
        extracted_issuer(stager);
        break;
    case Case::ReleaseSuccess:
    case Case::DrainFailure: {
        FakeImpl impl;
        impl.owned.push_back(probe->device[0]);
        extracted_release(&impl);
        break;
    }
    }
    // The actual destructor must not delete[] the fixture's mmap-owned ready array.
    stager->ready.release();
    stager->~Stager();
    return 0;
}

struct Spec { Case scenario; const char* name; bool aborts; const char* site; };
bool run_case(const Spec& spec) {
    void* mapping = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) return false;
    Shared* shared = new(mapping) Shared;
    probe = &shared->probe;
    probe->scenario = spec.scenario;
    std::memset(probe->host, 0x7a, sizeof probe->host);
    std::memset(probe->device, 0x7a, sizeof probe->device);
    Stager* stager = new(shared->storage) Stager;
    int diagnostic_pipe[2];
    if (pipe(diagnostic_pipe)) { munmap(mapping, sizeof(Shared)); return false; }
    const pid_t pid = fork();
    if (pid == 0) {
        close(diagnostic_pipe[0]);
        if (dup2(diagnostic_pipe[1], STDERR_FILENO) < 0) _exit(22);
        close(diagnostic_pipe[1]);
        _exit(child_case(stager));
    }
    close(diagnostic_pipe[1]);
    if (pid < 0) { close(diagnostic_pipe[0]); munmap(mapping, sizeof(Shared)); return false; }
    int status = 0;
    bool terminated = false;
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (Clock::now() < deadline) {
        const pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid) { terminated = true; break; }
        if (result < 0 && errno != EINTR) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!terminated) { kill(pid, SIGKILL); while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {} }
    char diagnostic[4096]{};
    const ssize_t got = read(diagnostic_pipe[0], diagnostic, sizeof diagnostic - 1);
    close(diagnostic_pipe[0]);
    bool good = terminated && (spec.aborts ? WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT
                                          : WIFEXITED(status) && WEXITSTATUS(status) == 0);
    if (spec.aborts) {
        good = good && got > 0 && std::strstr(diagnostic, spec.site) &&
               std::strstr(diagnostic, "operation=") && std::strstr(diagnostic, "error=999") &&
               std::strstr(diagnostic, "aborting native") && probe->frees.load() == 0;
    } else good = good && got == 0;
    switch (spec.scenario) {
    case Case::WaitFailure:
        good = good && probe->ready[0].load() == 0 && probe->host[0][0] == 0x7a &&
               stager->issued.load() == 0 && probe->copies.load() == 0;
        break;
    case Case::DeviceFailure:
        good = good && probe->ready[0].load() == 0 && probe->host[0][0] == 0x7a &&
               probe->waits.load() == 0 && stager->issued.load() == 0;
        break;
    case Case::RecordFailure:
        good = good && stager->issued.load() == 0 && probe->records.load() == 1;
        break;
    case Case::CopyFailure:
        good = good && stager->issued.load() == 0 && probe->issuer_issued == 0 &&
               probe->device[0][0] == 0x7a && probe->records.load() == 0;
        break;
    case Case::WaitEventFailure:
        good = good && stager->issued.load() == 0 && probe->issuer_issued == 0 &&
               probe->copies.load() == 0 && probe->records.load() == 0;
        break;
    case Case::CopiedRecordFailure:
        // DMA-event record succeeded, so host watermark 1 is valid. The separate
        // copied-event failure must still block device copy/issuer acknowledgment.
        good = good && stager->issued.load() == 1 && probe->issuer_issued == 0 &&
               probe->stats.experts_streamed == 0 && probe->records.load() == 2;
        break;
    case Case::DrainFailure:
        good = good && probe->drains.load() == 1 && probe->frees.load() == 0;
        break;
    case Case::WorkSuccess:
        good = good && probe->ready[0].load() == 1 && stager->issued.load() == (1 << 30) &&
               std::memcmp(probe->host[0], pattern, sizeof pattern) == 0 &&
               probe->records.load() == 1 && probe->frees.load() == 16;
        break;
    case Case::IssuerSuccess:
        good = good && probe->issuer_issued == 1 && probe->stats.experts_streamed == 1 &&
               std::memcmp(probe->device[0], pattern, sizeof pattern) == 0 &&
               probe->copies.load() == 1 && probe->records.load() == 2 && probe->frees.load() == 16;
        break;
    case Case::ReleaseSuccess:
        good = good && probe->drains.load() == 2 && probe->frees.load() == 17;
        break;
    }
    std::printf("%s %s exit_status=%d bounded=%d\n", good ? "PASS" : "FAIL", spec.name, status, terminated);
    if (!good && got > 0) std::fputs(diagnostic, stderr);
    // Do not invoke a parent destructor on child-allocated private heap pointers.
    // The child's healthy destructor is tested; each shared probe is retired here.
    munmap(mapping, sizeof(Shared));
    return good;
}

int main() {
    const Spec cases[] = {
        {Case::WaitFailure, "failed wait blocks overwrite/ready", true, "Stager::work host-slot reuse"},
        {Case::DeviceFailure, "failed worker device blocks job", true, "Stager::work device"},
        {Case::RecordFailure, "failed record blocks issued", true, "Stager::issued_one host-slot DMA"},
        {Case::CopyFailure, "failed H2D blocks issuer acknowledgment", true, "inline issuer stage copy"},
        {Case::WaitEventFailure, "failed dependency blocks H2D", true, "inline issuer stage copy"},
        {Case::CopiedRecordFailure, "failed copied event blocks device acknowledgment", true, "inline issuer stage copy"},
        {Case::DrainFailure, "failed teardown drain blocks frees", true, "Prefill::release compute drain"},
        {Case::WorkSuccess, "worker success retains bytes/ready/events", false, ""},
        {Case::IssuerSuccess, "issuer success retains bytes/events/ack", false, ""},
        {Case::ReleaseSuccess, "successful existing drains precede frees", false, ""},
    };
    bool good = true;
    for (const auto& spec : cases) good = run_case(spec) && good;
    return good ? 0 : 1;
}
