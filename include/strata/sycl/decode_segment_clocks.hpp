#pragma once

#include <cuda_runtime.h>
#include <array>
#include <charconv>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace strata::sycl_backend::decode_segment_clocks {

struct Config { std::uint64_t request; bool device; };
inline Config configuration() {
    static const auto config = []() -> Config {
        const char* text = std::getenv("STRATA_SYCL_DECODE_SEGMENT_CLOCK_REQUEST");
        if (!text) return {};
        const bool wall_only = std::strncmp(text, "wall:", 5) == 0;
        if (wall_only) text += 5;
        std::uint64_t value = 0;
        const char* end = text + std::strlen(text);
        auto parsed = std::from_chars(text, end, value);
        if (parsed.ec == std::errc{} && parsed.ptr == end) return {value, value != 0 && !wall_only};
        std::fprintf(stderr, "strata decode segment clocks: invalid request id\n");
        return {};
    }();
    return config;
}

// The 48-layer, unsplit T4 verifier has 49 sequential layer segments.
// Segment i contains the previous layer's post and the incoming layer's pre.
constexpr unsigned layers = 48;
struct Sample {
    compat::GraphEnvelopeTimestamps device;
    std::int64_t host_wait_ns;
    std::int64_t pool_callback_ns;
    int status;
};

class Request;
inline thread_local Request* active = nullptr;

class Request {
    Request* previous_;
    std::uint64_t id_;
    unsigned windows_ = 0;
    std::unique_ptr<std::array<Sample, layers + 1>> samples_;
public:
    explicit Request(std::uint64_t id) : previous_(active), id_(id) {
        active = id != 0 && configuration().request == id ? this : nullptr;
    }
    ~Request() { active = previous_; }
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;

    bool sample_next_window() {
        if (windows_ >= 12 || ++windows_ <= 8) return false;
        if (configuration().device && !samples_) samples_ = std::make_unique<std::array<Sample, layers + 1>>();
        return true;
    }

    void record(unsigned segment, cudaEvent_t completion, std::int64_t host_wait_ns) {
        auto& sample = samples_->at(segment);
        sample.host_wait_ns = host_wait_ns;
        sample.pool_callback_ns = 0;
        sample.status = int(compat::graph_completion_timestamps(completion, &sample.device));
    }

    void record_pool(unsigned segment, std::int64_t elapsed_ns) {
        samples_->at(segment).pool_callback_ns = elapsed_ns;
    }

    void emit_window(std::int64_t position, std::int64_t elapsed_ns) const {
        std::fprintf(stderr, "strata decode sample: request=%" PRIu64 " window=%u pos=%" PRId64
            " device=%d wall_ns=%" PRId64 "\n", id_, windows_, position, int(configuration().device), elapsed_ns);
    }

    // Called after the sampled window's existing compute and copy drains.
    void emit(std::int64_t position, std::int64_t qsa_interval) const {
        for (unsigned i = 0; i < samples_->size(); ++i) {
            const auto& s = (*samples_)[i];
            const auto& d = s.device;
            std::fprintf(stderr, "strata decode segment: request=%" PRIu64 " window=%u pos=%" PRId64
                " qsa_interval=%" PRId64 " segment=%u status=%d fields=%u submit=%" PRIu64
                " start=%" PRIu64 " end=%" PRIu64 " resolution_ns=%" PRIu64
                " host_wait_ns=%" PRId64 " pool_callback_ns=%" PRId64 "\n",
                id_, windows_, position, qsa_interval, i, s.status, d.available_fields,
                d.submit, d.start, d.end, d.timer_resolution, s.host_wait_ns, s.pool_callback_ns);
        }
    }
};

} // namespace strata::sycl_backend::decode_segment_clocks
