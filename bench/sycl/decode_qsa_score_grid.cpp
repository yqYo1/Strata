// Main-authored for Strata. This fixture uses the production SYCL graph shim.
#include "strata/kernels/qsa_select.hpp"
#include "strata/sycl/runtime.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strata::kernels::decode_score_grid {
void qsa_block_scores(const float*, const float*, const float*, const int32_t*, int64_t,
                      int64_t, const QsaShapes&, float*, void*, int64_t);
}

namespace {
using namespace strata::kernels;
using strata::sycl_backend::Allocation;
using strata::sycl_backend::MemoryKind;
using strata::sycl_backend::Runtime;
namespace compat = strata::sycl_backend::compat;
using Clock = std::chrono::steady_clock;
constexpr int capacity = 262144, blocks = capacity / 4 + 2, guard = 16;
constexpr float score_guard = -12345.5f;
constexpr int32_t id_guard = -123456789;
constexpr int replays = 512, batches = 3;
constexpr int profile_replays = 5;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void check(cudaError_t result, const char* message) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(message) + " " + cudaGetErrorString(result));
}
void retire(cudaError_t code) noexcept { if (code != cudaSuccess) std::abort(); }
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

class ReplayGraphs {
    cudaGraph_t recording_ = nullptr;
    cudaEvent_t complete_ = nullptr;
    std::array<cudaGraphExec_t, 4> executable_{};
    bool capturing_ = false;
    cudaStream_t stream_ = nullptr;
public:
    void open() {
        check(cudaStreamCreate(&stream_), "score graph stream");
        check(cudaEventCreateWithFlags(&complete_, cudaEventDisableTiming), "score completion event");
    }
    ~ReplayGraphs() {
        if (capturing_) retire(cudaStreamEndCapture(stream_, &recording_));
        if (stream_) retire(cudaStreamSynchronize(stream_));
        for (auto graph : executable_) if (graph) retire(cudaGraphExecDestroy(graph));
        if (recording_) retire(cudaGraphDestroy(recording_));
        if (complete_) retire(cudaEventDestroy(complete_));
        if (stream_) retire(cudaStreamDestroy(stream_));
    }
    template<class Record> double capture(int slot, size_t expected_nodes, bool profile, Record record) {
        const auto start = Clock::now();
        check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "score begin capture");
        capturing_ = true;
        record(stream_);
        check(cudaStreamEndCapture(stream_, &recording_), "score end capture");
        capturing_ = false;
        size_t nodes = 0;
        check(cudaGraphGetNodes(recording_, nullptr, &nodes), "score graph nodes");
        require(nodes == expected_nodes, "unexpected score graph topology");
        const auto status = profile
            ? compat::graph_instantiate_profiled(&executable_[slot], recording_)
            : cudaGraphInstantiate(&executable_[slot], recording_, 0ull);
        check(status, "score instantiate graph");
        check(cudaGraphDestroy(recording_), "score destroy source graph");
        recording_ = nullptr;
        return elapsed(start);
    }
    void run(int slot) {
        check(compat::graph_launch_with_completion(executable_[slot], stream_, complete_), "score launch");
        check(cudaEventSynchronize(complete_), "score replay completion");
    }
    compat::GraphEnvelopeTimestamps envelope() const {
        compat::GraphEnvelopeTimestamps sample{};
        check(compat::graph_completion_timestamps(complete_, &sample), "score graph timestamps");
        require((sample.available_fields & 6) == 6 && sample.timer_resolution > 0 &&
                sample.end > sample.start && sample.end - sample.start >= sample.timer_resolution,
                "invalid score graph device interval");
        return sample;
    }
};

template<class T> class Buffer {
    std::shared_ptr<Runtime> rt_;
    Allocation buffer_;
public:
    Buffer(const std::shared_ptr<Runtime>& rt, size_t n)
        : rt_(rt), buffer_(rt, n * sizeof(T), MemoryKind::Device) {}
    T* data() { return buffer_.as<T>(); }
    void put(const std::vector<T>& values) {
        require(values.size() * sizeof(T) == buffer_.size(), "score fixture upload extent");
        rt_->wait(rt_->compute().memcpy(data(), values.data(), buffer_.size()));
    }
    std::vector<T> get() {
        std::vector<T> values(buffer_.size() / sizeof(T));
        rt_->wait(rt_->compute().memcpy(values.data(), data(), buffer_.size()));
        return values;
    }
};

float seed_value(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return float(int(state & 0xffffu) - 32768) / 32768.f;
}

void check_outputs(int width, int cap, const std::vector<int32_t>& steps,
                   const std::vector<float>& scores, const std::vector<int32_t>& ids) {
    for (int q = 0; q < width; ++q) {
        const auto* step = steps.data() + q * kStepCount;
        const auto* score = scores.data() + guard + q * blocks;
        const auto* id = ids.data() + guard + q * cap;
        const int cells = step[kStepNKv], live = cells < 0 ? 0 : step[kStepNBid] + 1;
        for (int b = 0; b < live; ++b) require(std::isfinite(score[b]) && score[b] >= 0, "missing valid score");
        for (int b = live; b < blocks; ++b) require(score[b] == score_guard, "changed inactive score");
        const int used = cells < 0 ? 0 : step[kStepWidth];
        require(std::is_sorted(id, id + used) && std::adjacent_find(id, id + used) == id + used,
                "unordered or duplicated IDs");
        for (int i = 0; i < used; ++i) require(id[i] >= 0 && id[i] < cells, "selected ID outside the physical step");
        for (int i = used; i < cap; ++i) require(id[i] == id_guard, "changed inactive ID");
    }
    for (int i = 0; i < guard; ++i) {
        require(scores[size_t(i)] == score_guard && scores[scores.size() - 1 - size_t(i)] == score_guard,
                "changed score guard");
        require(ids[size_t(i)] == id_guard && ids[ids.size() - 1 - size_t(i)] == id_guard, "changed ID guard");
    }
}

void run(const std::shared_ptr<Runtime>& rt, int width, std::string_view mode) {
    const auto shape = qsa_real_shapes();
    const int cap = int(qsa_selection_width(capacity, shape));
    std::vector<float> pooled(size_t(blocks) * 128), dead(128), query(size_t(width) * 512);
    uint32_t seed = 0x8341c5d7u;
    for (auto* values : {&pooled, &dead, &query}) for (float& x : *values) x = seed_value(seed);
    std::vector<int32_t> steps(size_t(width) * kStepCount, 0);
    std::vector<float> unset_scores(size_t(width) * blocks + 2 * guard, score_guard);
    std::vector<int32_t> unset_ids(size_t(width) * cap + 2 * guard, id_guard);
    Buffer<float> kp(rt, pooled.size()), kd(rt, dead.size()), qq(rt, query.size()), scores(rt, unset_scores.size());
    Buffer<int32_t> ss(rt, steps.size()), ids(rt, unset_ids.size());
    ReplayGraphs graphs; // Retire the graphs and stream before releasing their USM owners.
    graphs.open();
    kp.put(pooled); kd.put(dead); qq.put(query);
    auto set_steps = [&](int context) {
        for (int q = 0; q < width; ++q) {
            const int n = context == -1 ? -1 : context - width + q + 1;
            auto* step = steps.data() + q * kStepCount;
            step[kStepPos] = n - 1;
            step[kStepNKv] = n;
            step[kStepNBid] = n / 4;
            step[kStepWidth] = n < 0 ? 0 : int32_t(qsa_selection_width(n, shape));
        }
        ss.put(steps);
    };
    set_steps(96'204);
    const bool profile = mode == "profile";
    const int order_offset = mode == "reverse" ? 1 : 0;
    std::array<double, 4> setups{};
    for (int i = 0; i < 4; ++i) {
        const int slot = i ^ order_offset;
        const bool grid = slot % 2, with_selector = slot >= 2;
        setups[slot] = graphs.capture(slot, with_selector ? 2 : 1, profile, [&](void* stream) {
            auto score = grid ? decode_score_grid::qsa_block_scores : qsa_block_scores;
            score(kp.data(), kd.data(), qq.data(), ss.data(), width, blocks, shape, scores.data() + guard, stream, -1);
            if (with_selector)
                qsa_block_topk(scores.data() + guard, ss.data(), width, blocks, cap, shape, ids.data() + guard, stream);
        });
    }
    std::cout << "setup width=" << width;
    for (int slot = 0; slot < 4; ++slot) std::cout << " ms" << slot << '=' << setups[slot];
    std::cout << std::endl;

    auto compare = [&](int context) {
        set_steps(context);
        scores.put(unset_scores); ids.put(unset_ids);
        graphs.run(2);
        const auto original = scores.get(); const auto expected_ids = ids.get();
        scores.put(unset_scores); ids.put(unset_ids);
        graphs.run(3);
        const auto candidate = scores.get(); const auto actual_ids = ids.get();
        require(std::memcmp(candidate.data(), original.data(), original.size() * sizeof(float)) == 0 &&
                actual_ids == expected_ids, "score bits or selected IDs changed across grid replay");
        check_outputs(width, cap, steps, candidate, actual_ids);
    };

    if (mode == "smoke") {
        compare(97); compare(98); compare(97); compare(-1);
    } else if (mode == "validate") {
        for (int end : {96'204, capacity}) for (int offset = 0; offset < 4; ++offset) compare(end - offset);
        std::fill(pooled.begin(), pooled.end(), 0.f);
        std::fill(dead.begin(), dead.end(), 0.f);
        kp.put(pooled); kd.put(dead);
        compare(96'203); compare(capacity); compare(96'203); compare(-1);
    } else if (profile) {
        uint64_t previous_end = 0;
        for (int context : {96'204, capacity}) {
            compare(context);
            for (int stage = 0; stage < 2; ++stage) {
                graphs.run(stage * 2);
                graphs.run(stage * 2 + 1);
                for (int i = 0; i < profile_replays; ++i) {
                    for (int order = 0; order < 2; ++order) {
                        const int arm = (order + i) % 2;
                        const auto start = Clock::now();
                        graphs.run(stage * 2 + arm);
                        const double wall_ms = elapsed(start);
                        const auto sample = graphs.envelope();
                        require(sample.start >= previous_end, "overlapping sequential score graph samples");
                        previous_end = sample.end;
                        std::cout << "device width=" << width << " context=" << context
                                  << " stage=" << stage << " replay=" << i << " arm=" << arm
                                  << " ns=" << sample.end - sample.start << " tick_ns=" << sample.timer_resolution
                                  << " wall_ms=" << wall_ms << std::endl;
                    }
                }
            }
            compare(context);
        }
    } else {
        for (int context : {96'204, capacity}) {
            compare(context);
            for (int stage = 0; stage < 2; ++stage) {
                for (int rep = 0; rep < batches; ++rep) {
                    const auto timer = Clock::now();
                    for (int i = 0; i < replays; ++i) (void) Clock::now();
                    const double clock_ms = elapsed(timer);
                    for (int order = 0; order < 2; ++order) {
                        const int arm = (order + rep + order_offset) % 2;
                        const int slot = stage * 2 + arm;
                        graphs.run(slot);
                        const auto start = Clock::now();
                        for (int i = 0; i < replays; ++i) graphs.run(slot);
                        const double ms = elapsed(start);
                        require(ms >= 10.0 && clock_ms <= ms * 0.01, "invalid scorer timing batch");
                        std::cout << "sample width=" << width << " context=" << context << " stage=" << stage
                                  << " batch=" << rep << " arm=" << arm << " replays=" << replays
                                  << " ms=" << ms << " clock_ms=" << clock_ms << std::endl;
                    }
                }
            }
            compare(context);
        }
    }
    std::cout << "checked width=" << width << " scores=PASS ids=PASS guards=PASS replay=PASS" << std::endl;
}
} // namespace

int main(int argc, char** argv) try {
    require(argc == 2, "use smoke|validate|forward|reverse|profile");
    const std::string_view mode(argv[1]);
    require(mode == "smoke" || mode == "validate" || mode == "forward" || mode == "reverse" || mode == "profile",
            "invalid score mode");
    auto runtime = strata::sycl_backend::runtime_for();
    const auto name = runtime->device().get_info<sycl::info::device::name>();
    require(name.find("B570") != std::string::npos, "expected Arc B570");
    std::cout << std::fixed << std::setprecision(9);
    const int widths = mode == "smoke" ? 1 : 4;
    for (int width = 1; width <= widths; ++width) run(runtime, width, mode);
    runtime->wait();
    std::cout << "decode QSA scorer completed" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "decode QSA scorer: " << e.what() << '\n';
    return 1;
}
