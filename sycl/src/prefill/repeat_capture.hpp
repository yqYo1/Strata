// Opt-in, bounded physical-row diagnostic. No device work while disabled.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <memory>
#include <vector>
#include <cerrno>
#include <cstdlib>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <stdexcept>

namespace strata::prefill::detail {
#if defined(_WIN32)
// This temporary diagnostic uses POSIX private-file checks. Disabled builds
// remain portable; an explicit request on Windows fails instead of weakening
// the file ownership or byte-budget contract.
class RepeatCapture {
public:
    static bool enabled() {
        static const bool on = [] { const char* p = std::getenv("STRATA_PREFILL_REPEAT_CAPTURE"); return p && *p; }();
        return on;
    }
    static RepeatCapture& instance() { static RepeatCapture c; return c; }
    void record(sycl::queue&, int64_t, const char*, int64_t, int64_t, int64_t, int64_t, int64_t, uint64_t, const void*) {
        throw std::runtime_error("repeat capture: this diagnostic requires POSIX private-file support");
    }
};
#else
class RepeatCapture {
    int fd_ = -1;
    uint64_t bytes_ = 0, records_ = 0;
    bool retirement_unknown_ = false;
    std::mutex mutex_;
    static constexpr uint64_t limit = 128ULL * 1024 * 1024;
    void write_all(const void* p, size_t n) {
        const char* b = static_cast<const char*>(p);
        while (n) {
            const ssize_t k = ::write(fd_, b, n);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) throw std::runtime_error("repeat capture: file write failed (diagnostic incomplete)");
            b += k; n -= (size_t) k;
        }
    }
public:
    ~RepeatCapture() {
        if (fd_ >= 0 && ::close(fd_) != 0)
            std::fprintf(stderr, "strata: repeat capture close failed; diagnostic incomplete\n");
    }
    static bool enabled() {
        static const bool on = [] { const char* p = std::getenv("STRATA_PREFILL_REPEAT_CAPTURE"); return p && *p; }();
        return on;
    }
    static RepeatCapture& instance() { static RepeatCapture c; return c; }
    // Header: 12 native-endian uint64 words, then a zero-padded 32-byte phase,
    // then elements * 4 payload bytes. All payload types are float32 (1) or int32 (2).
    // magic,version,layer,p0,T,rowfirst,rowcount,elements,type,rowwidth,records,headerbytes.
    void record(sycl::queue& queue, int64_t layer, const char* phase, int64_t p0, int64_t T,
                int64_t rowfirst, int64_t rows, int64_t width, uint64_t type, const void* device) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (retirement_unknown_)
            throw std::runtime_error("repeat capture: prior readback completion unknown (diagnostic incomplete)");
        if (rows <= 0 || width <= 0 || (uint64_t) width > limit / 4 / (uint64_t) rows)
            throw std::runtime_error("repeat capture: invalid shape (diagnostic incomplete)");
        const uint64_t elements = (uint64_t) rows * width, size = elements * 4;
        if (fd_ < 0) {
            const char* path = std::getenv("STRATA_PREFILL_REPEAT_CAPTURE");
            fd_ = ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK, 0600);
            struct stat st{};
            if (fd_ < 0 || ::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != ::geteuid() ||
                (st.st_mode & 077) || st.st_nlink != 1 || st.st_size < 0) {
                if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
                throw std::runtime_error("repeat capture: requires a fresh owned private regular file (diagnostic incomplete)");
            }
            bytes_ = (uint64_t) st.st_size;
            std::fprintf(stderr, "strata: repeat capture enabled; synchronous CPU readbacks can mask timing-dependent failures\n");
        }
        constexpr uint64_t headerbytes = 12 * 8 + 32;
        if (bytes_ > limit || size + headerbytes > limit - bytes_)
            throw std::runtime_error("repeat capture: 128 MiB budget exceeded (diagnostic incomplete)");
        auto host = std::make_unique<std::vector<uint32_t>>((size_t) elements);
        // Drain prior errors before issuing a copy, and keep its destination alive
        // through completion even if the asynchronous error handler throws.
        queue.wait_and_throw();
        try { queue.memcpy(host->data(), device, (size_t) size).wait_and_throw(); }
        catch (...) {
            try { queue.wait(); }
            catch (...) {
                // A failed drain does not prove that the destination retired.
                // Keep this one bounded allocation until process exit; later
                // captures cannot issue another readback or reuse its storage.
                retirement_unknown_ = true;
                (void) host.release();
                std::fprintf(stderr, "strata: repeat capture readback completion unknown after failed drain; "
                                     "destination quarantined until process exit; diagnostic incomplete\n");
            }
            throw; // Preserve the original submit/event error after either drain outcome.
        }
        const uint64_t header[12] = {0x5354524152505431ULL, 1, (uint64_t) layer, (uint64_t) p0,
            (uint64_t) T, (uint64_t) rowfirst, (uint64_t) rows, elements, type, (uint64_t) width,
            records_, headerbytes};
        char name[32]{}; std::strncpy(name, phase, sizeof(name) - 1);
        write_all(header, sizeof(header)); write_all(name, sizeof(name)); write_all(host->data(), (size_t) size);
        bytes_ += headerbytes + size; ++records_;
        std::fprintf(stderr, "strata: repeat capture record=%llu layer=%lld phase=%s row=%lld rows=%lld bytes=%llu\n",
            (unsigned long long) records_, (long long) layer, phase, (long long) rowfirst, (long long) rows,
            (unsigned long long) bytes_);
    }
};
#endif
} // namespace strata::prefill::detail
