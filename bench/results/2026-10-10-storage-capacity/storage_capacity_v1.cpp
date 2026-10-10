// Linux/POSIX, C++20 -pthread. Read-only hardware/file-path control, no GPU.
// O_DIRECT acceptance does not prove bypass of every filesystem cache or SSD traffic.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <pthread.h>
#include <unistd.h>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;
static_assert(Clock::is_steady, "steady host clock required");
constexpr uint64_t kSize = 28800138432ULL, kTableOffset = 192, kTableBytes = 28800138240ULL;
constexpr uint64_t kAlignment = 4096, kSequentialBytes = 1ULL << 20, kRandomCount = 65536;
constexpr uint64_t kBegin = (kTableOffset + kAlignment - 1) / kAlignment * kAlignment;
constexpr uint64_t kEnd = (kTableOffset + kTableBytes) / kAlignment * kAlignment;
constexpr size_t kGuardBytes = 64, kMaxWorkers = 64, kWorkerStackBytes = 256ULL << 10;
constexpr const char* kShard =
    "/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf";
static_assert(kTableOffset + kTableBytes == kSize && kEnd > kBegin);
static_assert((kEnd - kBegin) / kSequentialBytes <= kRandomCount);
static_assert(kMaxWorkers * (kSequentialBytes + kGuardBytes + kWorkerStackBytes + kAlignment) +
              kRandomCount * sizeof(uint64_t) + kSequentialBytes + kGuardBytes + (4ULL << 20) < (256ULL << 20));
static_assert(sizeof(off_t) >= 8, "64-bit pread offsets required");

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static uint64_t decimal(const char* text) {
    const size_t length = std::strlen(text);
    require(length > 0 && length <= 20, "decimal width");
    uint64_t value = 0;
    for (size_t i = 0; i < length; ++i) {
        require(text[i] >= '0' && text[i] <= '9', "nondecimal argument");
        const unsigned digit = unsigned(text[i] - '0');
        require(value <= (UINT64_MAX - digit) / 10, "decimal overflow");
        value = value * 10 + digit;
    }
    return value;
}
static uint64_t splitmix(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}
static void fnv_word(uint64_t& hash, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 255;
        hash *= 1099511628211ULL;
    }
}
static uint64_t fnv_bytes(const unsigned char* bytes, size_t size) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i]; hash *= 1099511628211ULL;
    }
    return hash;
}
static std::string json_string(const char* text) {
    std::string out = "\"";
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(text); *p; ++p) {
        if (*p == '"' || *p == '\\') { out += '\\'; out += char(*p); }
        else if (*p < 32) {
            char escaped[7];
            std::snprintf(escaped, sizeof escaped, "\\u%04x", unsigned(*p));
            out += escaped;
        } else out += char(*p);
    }
    return out + '"';
}
static void append_u64(std::string& output, const char* key, uint64_t value, bool comma = true) {
    if (comma) output += ',';
    output += json_string(key); output += ':'; output += std::to_string(value);
}
static void append_double(std::string& output, const char* key, double value) {
    require(value >= 0 && value <= std::numeric_limits<double>::max(), "nonfinite result");
    char number[64];
    const int n = std::snprintf(number, sizeof number, "%.17g", value);
    require(n > 0 && size_t(n) < sizeof number, "result number width");
    output += ','; output += json_string(key); output += ':'; output += number;
}
struct Config {
    std::string path, mode;
    size_t workers = 0;
    uint64_t seed = 0, block_bytes = 0, requests = 0, available_requests = 0;
    bool sequential = false, qualify = false;
    uint64_t offset(uint64_t job) const noexcept {
        return kBegin + (sequential ? job * kSequentialBytes :
            splitmix(seed ^ job) % ((kEnd - kBegin) / kAlignment) * kAlignment);
    }
};
static Config parse(int argc, char** argv) {
    require(argc == 5 || argc == 6,
        "usage: storage_capacity_v1 SHARD sequential|random WORKERS SEED [--qualify]");
    require(std::strlen(argv[1]) > 0 && std::strlen(argv[1]) <= 4096, "path width");
    require(std::strcmp(argv[1], kShard) == 0, "specialized fixture requires exact original shard path");
    require(std::strlen(argv[2]) > 0 && std::strlen(argv[2]) <= 16, "mode width");
    Config c; c.path = argv[1]; c.mode = argv[2];
    require(c.mode == "sequential" || c.mode == "random", "mode");
    const uint64_t workers = decimal(argv[3]);
    require(workers == 1 || workers == 16 || workers == 64, "workers must be1,16,64");
    c.workers = size_t(workers); c.seed = decimal(argv[4]);
    c.sequential = c.mode == "sequential";
    require(!c.sequential || workers != 64, "sequential workers must be1 or16");
    if (argc == 6) {
        require(std::strcmp(argv[5], "--qualify") == 0, "unknown option");
        c.qualify = true;
    }
    c.block_bytes = c.sequential ? kSequentialBytes : kAlignment;
    c.available_requests = c.sequential ? (kEnd - kBegin) / kSequentialBytes : kRandomCount;
    c.requests = c.qualify ? std::max<uint64_t>(16, workers) : c.available_requests;
    require(c.requests >= workers && c.requests <= kRandomCount &&
            (!c.sequential || c.requests <= c.available_requests), "bounded request count");
    require(c.block_bytes <= UINT64_MAX / c.requests, "logical byte count overflow");
    return c;
}
struct PosixError : std::runtime_error {
    int code;
    PosixError(const char* message, int value) : std::runtime_error(message), code(value) {}
};
struct ThreadAttributes {
    pthread_attr_t value{};
    ThreadAttributes() {
        int code = ::pthread_attr_init(&value);
        if (code) throw PosixError("pthread_attr_init failed", code);
        try {
            code = ::pthread_attr_setstacksize(&value, kWorkerStackBytes);
            if (code) throw PosixError("bounded worker stack admission failed", code);
            code = ::pthread_attr_setguardsize(&value, kAlignment);
            if (code) throw PosixError("worker stack guard admission failed", code);
        } catch (...) { ::pthread_attr_destroy(&value); throw; }
    }
    ~ThreadAttributes() { ::pthread_attr_destroy(&value); }
};
struct Thread {
    pthread_t value{};
    bool active = false;
    Thread(const pthread_attr_t& attributes, void* (*entry)(void*), void* context) {
        const int code = ::pthread_create(&value, &attributes, entry, context);
        if (code) throw PosixError("pthread_create failed", code);
        active = true;
    }
    Thread(const Thread&) = delete; Thread& operator=(const Thread&) = delete;
    Thread(Thread&& other) noexcept : value(other.value), active(std::exchange(other.active,false)) {}
    void join() {
        if (!active) return;
        const int code = ::pthread_join(value,nullptr);
        if (code) throw PosixError("pthread_join failed", code);
        active = false;
    }
    ~Thread() {
        if (active) {
            static constexpr char failure[] = "{\"success\":false,\"reason\":\"unjoined worker; no cleanup\"}\n";
            (void)::write(STDERR_FILENO,failure,sizeof failure-1); std::_Exit(1);
        }
    }
};
struct Fd {
    int value = -1;
    explicit Fd(int descriptor) : value(descriptor) { if (value < 0) throw PosixError("read-only open failed",errno); }
    Fd(const Fd&) = delete; Fd& operator=(const Fd&) = delete;
    ~Fd() { if (value >= 0) ::close(value); }
};
struct Buffer {
    unsigned char* data = nullptr;
    size_t bytes = 0;
    explicit Buffer(size_t length) : bytes(length) {
        require(bytes > 0 && bytes <= kSequentialBytes && bytes % kAlignment == 0, "buffer bound");
        void* allocation = nullptr;
        const int error = ::posix_memalign(&allocation, kAlignment, bytes + kGuardBytes);
        if (error) throw std::runtime_error("aligned buffer allocation failed");
        data = static_cast<unsigned char*>(allocation);
        std::memset(data + bytes, 0xa5, kGuardBytes);
    }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : data(std::exchange(other.data, nullptr)), bytes(other.bytes) {}
    Buffer& operator=(Buffer&&) = delete;
    ~Buffer() { std::free(data); }
    bool guard_ok() const noexcept {
        for (size_t i = 0; i < kGuardBytes; ++i) if (data[bytes + i] != 0xa5) return false;
        return true;
    }
};
static struct stat file_stat(int fd) {
    struct stat value{};
    if (::fstat(fd, &value) != 0) throw PosixError("fstat failed",errno);
    require(S_ISREG(value.st_mode) && value.st_size >= 0 && uint64_t(value.st_size) == kSize,
            "fixture must be regular and exact expected size");
    return value;
}
static bool same_stat(const struct stat& a, const struct stat& b) noexcept {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
        a.st_mode == b.st_mode && a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
        a.st_mtim.tv_nsec == b.st_mtim.tv_nsec && a.st_ctim.tv_sec == b.st_ctim.tv_sec &&
        a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
static bool read_bound(const Config& c, uint64_t job, uint64_t offset) noexcept {
    return job < c.requests && offset >= kBegin && offset % kAlignment == 0 &&
        offset >= kTableOffset && offset <= kEnd &&
        c.block_bytes <= kEnd - offset && c.block_bytes <= kSize - offset &&
        offset <= uint64_t(std::numeric_limits<off_t>::max());
}
struct State {
    std::atomic<bool> stop{false}, released{false};
    std::atomic<size_t> ready{0};
    std::atomic<uint64_t> next{0}, completed{0}, bytes{0}, syscalls{0}, eintr{0};
    std::atomic<uint64_t> inflight{0}, peak{0};
    std::atomic_flag error_lock = ATOMIC_FLAG_INIT;
    bool have_error = false;
    uint64_t error_job = UINT64_MAX, error_offset = 0;
    int error_errno = 0;
    std::array<char,512> error_text{};
    void release() noexcept {
        released.store(true, std::memory_order_release); released.notify_all();
    }
    void fail(const char* message, int code, uint64_t job, uint64_t offset) noexcept {
        while (error_lock.test_and_set(std::memory_order_acquire)) {}
        if (!have_error) {
            have_error = true; error_errno = code; error_job = job; error_offset = offset;
            std::snprintf(error_text.data(), error_text.size(), "%s", message);
        }
        error_lock.clear(std::memory_order_release);
        stop.store(true, std::memory_order_release); release(); ready.notify_all();
    }
};
struct Worker {
    uint64_t assignments = 0, completed = 0, final_offset = 0, final_fnv = 0;
    bool have_final = false;
};
struct Inflight {
    State& state;
    explicit Inflight(State& s) noexcept : state(s) {
        const uint64_t active = state.inflight.fetch_add(1, std::memory_order_relaxed) + 1;
        uint64_t peak = state.peak.load(std::memory_order_relaxed);
        while (active > peak && !state.peak.compare_exchange_weak(peak, active, std::memory_order_relaxed)) {}
    }
    ~Inflight() { state.inflight.fetch_sub(1, std::memory_order_relaxed); }
};
struct JoinGuard {
    State& state;
    std::vector<Thread>& threads;
    void join() { for (auto& thread : threads) thread.join(); }
    ~JoinGuard() {
        state.stop.store(true, std::memory_order_release); state.release();
        try { join(); } catch (...) {
            // Do not close descriptors/free buffers while a worker might still use them.
            static constexpr char failure[] = "{\"success\":false,\"reason\":\"thread join failed; no cleanup\"}\n";
            (void)::write(STDERR_FILENO, failure, sizeof failure - 1); std::_Exit(1);
        }
    }
};
static void worker(const Config& c, State& state, size_t index, int fd, Buffer& buffer,
                   Worker& receipt, std::vector<uint64_t>& latency) noexcept {
    uint64_t job = UINT64_MAX, offset = 0;
    try {
        state.ready.fetch_add(1, std::memory_order_release); state.ready.notify_all();
        state.released.wait(false, std::memory_order_acquire);
        job = index; // Reserve one job per worker, including the short qualification.
        while (!state.stop.load(std::memory_order_acquire) && job < c.requests) {
            offset = c.offset(job);
            ++receipt.assignments;
            if (!read_bound(c, job, offset) || reinterpret_cast<uintptr_t>(buffer.data) % kAlignment) {
                state.fail("request/buffer alignment or table bound", 0, job, offset); return;
            }
            const auto start = Clock::now();
            ssize_t got = -1;
            int code = 0;
            {
                Inflight active(state);
                for (;;) {
                    state.syscalls.fetch_add(1, std::memory_order_relaxed);
                    got = ::pread(fd, buffer.data, size_t(c.block_bytes), off_t(offset));
                    if (got >= 0) break;
                    code = errno;
                    if (code != EINTR) break;
                    state.eintr.fetch_add(1, std::memory_order_relaxed);
                    if (state.stop.load(std::memory_order_acquire)) break;
                }
            }
            const auto end = Clock::now();
            if (got != ssize_t(c.block_bytes)) {
                state.fail(got < 0 ? "pread error (including stopped EINTR retry)" : "short pread rejected",
                           got < 0 ? code : 0, job, offset);
                return;
            }
            if (end < start) { state.fail("backward host latency clock", 0, job, offset); return; }
            if (!buffer.guard_ok()) { state.fail("buffer canary changed", 0, job, offset); return; }
            latency[size_t(job)] = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
            ++receipt.completed; receipt.have_final = true; receipt.final_offset = offset;
            state.completed.fetch_add(1, std::memory_order_relaxed);
            state.bytes.fetch_add(c.block_bytes, std::memory_order_relaxed);
            job = state.next.fetch_add(1, std::memory_order_relaxed);
        }
    } catch (const std::exception& error) { state.fail(error.what(), 0, job, offset); }
      catch (...) { state.fail("unknown worker exception", 0, job, offset); }
}
struct WorkerContext {
    const Config* config = nullptr;
    State* state = nullptr;
    size_t index = 0;
    int fd = -1;
    Buffer* buffer = nullptr;
    Worker* receipt = nullptr;
    std::vector<uint64_t>* latency = nullptr;
};
static void* worker_entry(void* opaque) noexcept {
    auto& c = *static_cast<WorkerContext*>(opaque);
    worker(*c.config,*c.state,c.index,c.fd,*c.buffer,*c.receipt,*c.latency);
    return nullptr;
}
static uint64_t quantile(const std::vector<uint64_t>& sorted, unsigned percentile) {
    require(!sorted.empty() && percentile >= 1 && percentile <= 100, "quantile bound");
    const size_t rank = (sorted.size() * percentile + 99) / 100; // nearest rank
    return sorted[rank - 1];
}
static void reference_read(int fd, Buffer& buffer, uint64_t offset) {
    ssize_t got;
    do { got = ::pread(fd, buffer.data, buffer.bytes, off_t(offset)); } while (got < 0 && errno == EINTR);
    if (got < 0) throw PosixError("untimed buffered reference read failed",errno);
    require(got == ssize_t(buffer.bytes), "untimed buffered reference read short");
    require(buffer.guard_ok(), "reference canary changed");
}
static std::string run(const Config& c, State& state) {
    constexpr int flags = O_RDONLY | O_DIRECT | O_CLOEXEC | O_NOFOLLOW;
    const Fd direct(::open(c.path.c_str(), flags));
    const struct stat original = file_stat(direct.value);
    const int actual_flags = ::fcntl(direct.value, F_GETFL);
    if (actual_flags < 0) throw PosixError("fcntl F_GETFL failed",errno);
    require((actual_flags & O_ACCMODE) == O_RDONLY &&
            (actual_flags & O_DIRECT), "actual direct read-only flags missing");
    std::vector<Buffer> buffers; buffers.reserve(c.workers);
    for (size_t i = 0; i < c.workers; ++i) buffers.emplace_back(size_t(c.block_bytes));
    std::vector<Worker> workers(c.workers);
    std::vector<uint64_t> latency(size_t(c.requests), UINT64_MAX);
    ThreadAttributes attributes;
    std::vector<WorkerContext> contexts(c.workers);
    std::vector<Thread> threads; threads.reserve(c.workers);
    state.next.store(c.workers, std::memory_order_relaxed);
    JoinGuard guard{state, threads}; // Destroyed before buffers and descriptor on every path.
    for (size_t i = 0; i < c.workers; ++i) {
        contexts[i] = WorkerContext{&c,&state,i,direct.value,&buffers[i],&workers[i],&latency};
        threads.emplace_back(attributes.value,worker_entry,&contexts[i]);
    }
    while (state.ready.load(std::memory_order_acquire) < c.workers) {
        const size_t observed = state.ready.load(std::memory_order_acquire);
        if (observed == c.workers) break;
        state.ready.wait(observed, std::memory_order_acquire);
    }
    const auto start = Clock::now();
    state.release();
    guard.join();
    const auto end = Clock::now(); // Includes gate handoff, completed calls and joins.
    if (state.have_error) throw std::runtime_error(state.error_text.data());
    require(!state.stop.load() && state.inflight.load() == 0, "worker stop/inflight reconciliation");
    const uint64_t expected_bytes = c.requests * c.block_bytes;
    require(state.syscalls.load() == c.requests + state.eintr.load(), "pread/EINTR syscall reconciliation");
    require(state.completed.load() == c.requests && state.bytes.load() == expected_bytes,
            "complete request/byte reconciliation");
    require(end > start, "nonpositive whole-operation clock");
    const double wall_seconds = std::chrono::duration<double>(end-start).count();
    uint64_t worker_completed = 0, assignment_sum = 0, assignment_max = 0, assignment_min = UINT64_MAX;
    const Fd buffered(::open(c.path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    require(same_stat(original, file_stat(buffered.value)), "buffered descriptor/original stat differs");
    Buffer reference(size_t(c.block_bytes));
    uint64_t final_checks = 0, final_checked_bytes = 0, final_fnv = 14695981039346656037ULL;
    for (size_t i = 0; i < c.workers; ++i) {
        auto& w = workers[i];
        require(w.have_final && w.completed > 0 && w.assignments == w.completed, "worker final/count reconciliation");
        require(buffers[i].guard_ok(), "final worker canary changed");
        require(w.final_offset >= kBegin && w.final_offset <= kEnd && c.block_bytes <= kEnd - w.final_offset,
                "final reference offset bound");
        reference_read(buffered.value, reference, w.final_offset);
        require(std::memcmp(buffers[i].data, reference.data, size_t(c.block_bytes)) == 0,
                "final surviving worker buffer differs from buffered reference");
        w.final_fnv = fnv_bytes(buffers[i].data, size_t(c.block_bytes)); // Full hash, untimed.
        fnv_word(final_fnv, i); fnv_word(final_fnv, w.final_offset); fnv_word(final_fnv, w.final_fnv);
        ++final_checks; final_checked_bytes += c.block_bytes;
        worker_completed += w.completed; assignment_sum += w.assignments;
        assignment_max = std::max(assignment_max, w.assignments);
        assignment_min = std::min(assignment_min, w.assignments);
    }
    require(worker_completed == c.requests && assignment_sum == c.requests &&
            state.peak.load() >= 1 && state.peak.load() <= c.workers, "assignment/peak reconciliation");
    uint64_t latency_sum = 0, minimum_offset = UINT64_MAX, maximum_end = 0;
    uint64_t order_fnv = 14695981039346656037ULL;
    for (uint64_t job = 0; job < c.requests; ++job) {
        const uint64_t ns = latency[size_t(job)];
        require(ns != UINT64_MAX && ns <= UINT64_MAX - latency_sum, "missing/overflow latency");
        latency_sum += ns;
        const uint64_t offset = c.offset(job);
        require(read_bound(c, job, offset), "post-read planned offset bound");
        minimum_offset = std::min(minimum_offset, offset);
        maximum_end = std::max(maximum_end, offset + c.block_bytes);
        fnv_word(order_fnv, job); fnv_word(order_fnv, offset); fnv_word(order_fnv, c.block_bytes);
    }
    std::sort(latency.begin(), latency.end());
    require(same_stat(original, file_stat(direct.value)) &&
            same_stat(original, file_stat(buffered.value)), "original file stat changed");
    std::string output = "{\"success\":true,\"scope\":\"read-only hardware/file-path control; not PLE, model or proven SSD peak\",";
    output += "\"mode\":" + json_string(c.mode.c_str()) + ",\"shard\":" + json_string(c.path.c_str());
    output += ",\"qualification_only\":" + std::string(c.qualify ? "true" : "false");
    output += ",\"timing_claim\":" + std::string(c.qualify ? "false" : "true");
    output += ",\"warmup_requests\":0,\"payload_validation\":\"full byte equality of final surviving buffer per worker only; all calls/counts/bytes checked\"";
    output += ",\"cache_note\":\"O_DIRECT accepted; filesystem fallback/cache behavior and physical SSD bytes are not established\"";
    append_u64(output,"workers",c.workers); append_u64(output,"worker_stack_bytes",kWorkerStackBytes);
    append_u64(output,"worker_stack_guard_bytes",kAlignment); append_u64(output,"seed",c.seed);
    append_u64(output,"block_bytes",c.block_bytes); append_u64(output,"requests",c.requests);
    append_u64(output,"available_sequential_requests",c.sequential ? c.available_requests : 0);
    append_u64(output,"completed_requests",state.completed.load()); append_u64(output,"logical_bytes",expected_bytes);
    append_u64(output,"pread_syscalls",state.syscalls.load()); append_u64(output,"eintr_retries",state.eintr.load());
    append_double(output,"wall_seconds_observed",wall_seconds);
    // Qualification emits observations without throughput/IOPS or a service timing claim.
    if (!c.qualify) {
        append_double(output,"logical_GBps",double(expected_bytes)/wall_seconds/1e9);
        append_double(output,"IOPS",double(c.requests)/wall_seconds);
    }
    append_u64(output,"latency_clock_calls",2*c.requests);
    append_u64(output,"latency_median_ns",quantile(latency,50));
    append_u64(output,"latency_p95_ns",quantile(latency,95));
    append_u64(output,"latency_p99_ns",quantile(latency,99));
    append_u64(output,"latency_sum_ns",latency_sum);
    output += ",\"latency_definition\":\"two steady-clock calls per request; pread including EINTR retries plus inflight atomics; nearest-rank quantiles; overlapping latencies are not additive wall\"";
    append_u64(output,"actual_peak_inflight_requests",state.peak.load());
    output += ",\"inflight_definition\":\"worker pread/retry region, not physical device queue depth\"";
    append_u64(output,"worker_assignments_min",assignment_min); append_u64(output,"worker_assignments_max",assignment_max);
    append_u64(output,"table_offset",kTableOffset); append_u64(output,"table_bytes",kTableBytes);
    append_u64(output,"aligned_begin",kBegin); append_u64(output,"aligned_end",kEnd);
    append_u64(output,"min_requested_offset",minimum_offset); append_u64(output,"max_requested_end",maximum_end);
    append_u64(output,"requested_bounding_span_bytes",maximum_end-minimum_offset);
    append_u64(output,"planned_job_order_fnv",order_fnv);
    output += ",\"sequential_definition\":\"ascending planned job-index offsets; concurrent workers may reorder issue/completion\"";
    append_u64(output,"sequential_tail_omitted_bytes",c.sequential ? (kEnd-kBegin)-c.available_requests*c.block_bytes : 0);
    append_u64(output,"table_alignment_prefix_omitted_bytes",kBegin-kTableOffset);
    append_u64(output,"table_alignment_suffix_omitted_bytes",kSize-kEnd);
    output += ",\"random_definition\":\"SplitMix64(seed XOR job_index) modulo aligned block count; replacement sampling, independent of worker/order\"";
    append_u64(output,"final_buffer_checks",final_checks); append_u64(output,"final_buffer_checked_bytes",final_checked_bytes);
    append_u64(output,"final_buffers_fnv",final_fnv);
    append_u64(output,"requested_open_flags",uint64_t(flags)); append_u64(output,"actual_f_getfl",uint64_t(actual_flags));
    output += ",\"original_stat\":{\"dev\":" + std::to_string(uint64_t(original.st_dev));
    append_u64(output,"ino",uint64_t(original.st_ino)); append_u64(output,"size",uint64_t(original.st_size));
    output += ",\"mtime_sec\":" + std::to_string(int64_t(original.st_mtim.tv_sec));
    append_u64(output,"mtime_nsec",uint64_t(original.st_mtim.tv_nsec));
    output += ",\"ctime_sec\":" + std::to_string(int64_t(original.st_ctim.tv_sec));
    append_u64(output,"ctime_nsec",uint64_t(original.st_ctim.tv_nsec)); output += '}';
    output += ",\"stat_unchanged\":true,\"workers_detail\":[";
    for (size_t i = 0; i < c.workers; ++i) {
        if (i) output += ',';
        output += "{\"worker\":" + std::to_string(i);
        append_u64(output,"assignments",workers[i].assignments); append_u64(output,"completed",workers[i].completed);
        append_u64(output,"final_offset",workers[i].final_offset); append_u64(output,"final_buffer_fnv",workers[i].final_fnv);
        output += '}';
    }
    output += "]}\n";
    return output;
}
int main(int argc, char** argv) {
    State state;
    try {
        const Config config = parse(argc, argv); // All CLI admission precedes opening any file.
        const std::string output = run(config, state); // Every thread joined/FD closed before PASS emission.
        require(std::fwrite(output.data(),1,output.size(),stdout) == output.size() && std::fflush(stdout) == 0,
                "success receipt output failed");
        return 0;
    } catch (const std::exception& error) {
        try {
            std::string output = "{\"success\":false,\"reason\":" + json_string(error.what());
            append_u64(output,"completed_requests",state.completed.load());
            append_u64(output,"completed_logical_bytes",state.bytes.load());
            if (state.have_error) {
                output += ",\"worker_error\":" + json_string(state.error_text.data());
                append_u64(output,"job",state.error_job); append_u64(output,"offset",state.error_offset);
                output += ",\"errno\":" + std::to_string(state.error_errno);
            } else {
                const auto* posix = dynamic_cast<const PosixError*>(&error);
                output += ",\"errno\":" + std::to_string(posix ? posix->code : 0);
            }
            output += "}\n";
            (void)std::fwrite(output.data(),1,output.size(),stderr); (void)std::fflush(stderr);
        } catch (...) {
            static constexpr char fallback[] = "{\"success\":false,\"reason\":\"failure receipt allocation failed\"}\n";
            (void)::write(STDERR_FILENO,fallback,sizeof fallback-1);
        }
        return 1;
    } catch (...) {
        static constexpr char failure[] = "{\"success\":false,\"reason\":\"unknown main exception\"}\n";
        (void)::write(STDERR_FILENO,failure,sizeof failure-1); return 1;
    }
}
