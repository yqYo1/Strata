#pragma once
// Main-authored host-only diagnostic control; repository LICENSE.
// Caller drains its GPU work before a gate is published. No runtime/env changes.
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <utility>

namespace strata::sycl_backend::decode_profile_gate {
struct Config {
    std::string directory;
    uint64_t request = 0;
    unsigned timeout_ms = 10000;
    bool valid = true;
};

inline Config configuration(const char* directory, const char* request) {
    if (!directory && !request) return {};
    Config c;
    c.valid = directory && *directory == '/' && std::strlen(directory) <= 4096 && request && *request;
    if (c.valid) {
        const auto end = request + std::strlen(request);
        const auto parsed = std::from_chars(request, end, c.request);
        c.valid = parsed.ec == std::errc{} && parsed.ptr == end && c.request > 0;
        c.directory = directory;
    }
    return c;
}

inline Config environment() {
    return configuration(std::getenv("STRATA_SYCL_DECODE_PROFILE_DIR"),
                         std::getenv("STRATA_SYCL_DECODE_PROFILE_REQUEST"));
}

class Gate {
    Config config_;
    uint64_t request_ = 0, last_round_ = 0;
    size_t next_ = 0;
    int directory_ = -1;
    bool begun_ = false, failed_ = false;
    static constexpr std::array<uint64_t, 4> boundaries{0, 8, 16, 24};

    bool valid_config() const noexcept {
        return config_.valid && config_.timeout_ms > 0 && config_.timeout_ms <= 30000 &&
            ((config_.directory.empty() && config_.request == 0) ||
             (!config_.directory.empty() && config_.directory.front() == '/' &&
              config_.directory.size() <= 4096 && config_.request > 0));
    }

    bool fail(std::string& error, const char* text) {
        failed_ = true;
        error = std::string("decode profile gate: ") + text;
        return false;
    }

    bool exchange(uint64_t completed, std::string& error) {
        if (directory_ < 0) {
            directory_ = ::open(config_.directory.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
            struct stat s{};
            if (directory_ < 0 || ::fstat(directory_, &s) || !S_ISDIR(s.st_mode) ||
                s.st_uid != ::geteuid() || (s.st_mode & 07777) != 0700)
                return fail(error, "private0700 directory required");
        }
        struct stat old{};
        errno = 0;
        if (::fstatat(directory_, "ack", &old, AT_SYMLINK_NOFOLLOW) == 0 || errno != ENOENT)
            return fail(error, "acknowledgement exists before gate publication");
        timespec raw{};
        if (::clock_gettime(CLOCK_MONOTONIC_RAW, &raw) || raw.tv_sec < 0)
            return fail(error, "raw clock unavailable");
        const auto now = uint64_t(raw.tv_sec) * 1000000000ULL + uint64_t(raw.tv_nsec);
        char event[192]{}, expected[128]{};
        const auto sequence = static_cast<unsigned long long>(next_ + 1);
        const int n = std::snprintf(event, sizeof(event), "v1 %llu %llu %llu %llu\n",
            static_cast<unsigned long long>(request_), sequence,
            static_cast<unsigned long long>(completed), static_cast<unsigned long long>(now));
        const int a = std::snprintf(expected, sizeof(expected), "v1 %llu %llu %llu\n",
            static_cast<unsigned long long>(request_), sequence, static_cast<unsigned long long>(completed));
        if (n <= 0 || size_t(n) >= sizeof(event) || a <= 0 || size_t(a) >= sizeof(expected))
            return fail(error, "event extent");
        const int output = ::openat(directory_, "event.tmp", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (output < 0) return fail(error, "event create");
        const auto written = ::write(output, event, size_t(n));
        const int closed = ::close(output);
        if (written != n || closed || ::renameat(directory_, "event.tmp", directory_, "event"))
            return fail(error, "event publication");
        std::fprintf(stderr, "strata decode gate: request=%llu sequence=%llu rounds=%llu raw_ns=%llu\n",
            static_cast<unsigned long long>(request_), sequence,
            static_cast<unsigned long long>(completed), static_cast<unsigned long long>(now));
        std::fflush(stderr);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(config_.timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            const int input = ::openat(directory_, "ack", O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
            if (input >= 0) {
                struct stat s{};
                char received[128]{};
                const bool regular = ::fstat(input, &s) == 0 && S_ISREG(s.st_mode) &&
                    s.st_uid == ::geteuid() && s.st_nlink == 1 && s.st_size == a &&
                    (s.st_mode & 07777) == 0600;
                const auto bytes = regular ? ::read(input, received, sizeof(received)) : -1;
                const int close_status = ::close(input);
                if (!regular || bytes != a || close_status || std::memcmp(received, expected, size_t(a)) ||
                    ::unlinkat(directory_, "ack", 0))
                    return fail(error, "acknowledgement identity/type/extent differs");
                return true;
            }
            if (errno != ENOENT) return fail(error, "acknowledgement open");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return fail(error, "acknowledgement deadline");
    }

public:
    Gate(Config config, uint64_t request) : config_(std::move(config)), request_(request) {}
    ~Gate() { if (directory_ >= 0) ::close(directory_); }
    Gate(const Gate&) = delete;
    Gate& operator=(const Gate&) = delete;
    bool enabled() const noexcept { return config_.request != 0 && config_.request == request_; }
    bool complete() const noexcept {
        return valid_config() && !failed_ && (!enabled() || next_ == boundaries.size());
    }

    template<class Drain> bool checkpoint(uint64_t completed_rounds, Drain&& drain, std::string& error) {
        if (failed_) return fail(error, "earlier checkpoint failed");
        if (!valid_config())
            return fail(error, "invalid immutable startup configuration");
        if (!enabled()) return true; // No filesystem, clocks, GPU call or allocation when unset.
        if ((!begun_ && completed_rounds != 0) ||
            (begun_ && (last_round_ == UINT64_MAX || completed_rounds != last_round_ + 1)))
            return fail(error, "round sequence differs");
        begun_ = true;
        last_round_ = completed_rounds;
        if (next_ == boundaries.size() || completed_rounds < boundaries[next_]) return true;
        if (completed_rounds != boundaries[next_]) return fail(error, "checkpoint boundary skipped");
        try {
            if (!drain()) return fail(error, "GPU completion not established");
        } catch (...) { return fail(error, "GPU completion threw"); }
        if (!exchange(completed_rounds, error)) return false;
        ++next_;
        return true;
    }
};
} // namespace strata::sycl_backend::decode_profile_gate
