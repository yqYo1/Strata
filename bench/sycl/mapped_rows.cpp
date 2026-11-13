// Main-authored bounded copy mechanism fixture; repository LICENSE.
// Explicit test parameters select separate functions; no environment mutation.
#include "strata/kernels/elementwise.hpp"
#include "strata/sycl/compat/cuda_runtime.h"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
namespace kernels = strata::kernels;
namespace compat = strata::sycl_backend::compat;
constexpr size_t guard = 64, max_rows = 40, max_width = 2560;
constexpr size_t capacity = max_rows * max_width + 2 * guard + 4;
constexpr std::array<uint32_t, 10> patterns{
    0, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000,
    0xff800000, 0x7fc00001, 0x7fc12345, 1, 0x80000001};

void need(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}
void check(cudaError_t error, const char* why) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(why) + ": " + cudaGetErrorString(error));
}
uint32_t bits(float value) { return std::bit_cast<uint32_t>(value); }

struct Fixture {
    cudaStream_t stream = nullptr;
    cudaEvent_t completion = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    float *host = nullptr, *mapped = nullptr, *device = nullptr;
    int32_t *device_hits = nullptr, *device_count = nullptr;
    std::vector<float> source = std::vector<float>(capacity);
    std::vector<float> initial = std::vector<float>(capacity);
    std::vector<float> readback = std::vector<float>(capacity);
    std::array<int32_t, max_rows> hits{};
    int32_t count = 0;
    int64_t rows = 0, width = 0;
    size_t offset = guard;
    bool selected = false;

    void open() {
        check(cudaSetDevice(0), "device select");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "device properties");
        need(std::string(prop.name).find("B570") != std::string::npos, "B570 required");
        check(cudaStreamCreate(&stream), "stream create");
        need(stream->has_property<sycl::property::queue::in_order>(), "in-order stream required");
        const auto sizes = stream->get_device().get_info<sycl::info::device::sub_group_sizes>();
        need(std::find(sizes.begin(), sizes.end(), size_t(32)) != sizes.end() &&
             stream->get_device().get_info<sycl::info::device::max_work_group_size>() >= 256,
             "subgroup32/workgroup256 required");
        check(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming), "completion create");
        check(cudaHostAlloc(&host, capacity * sizeof(float), cudaHostAllocMapped), "host source allocation");
        check(cudaHostGetDevicePointer(&mapped, host, 0), "mapped source view");
        check(cudaMalloc(&device, capacity * sizeof(float)), "device destination");
        check(cudaMalloc(&device_hits, max_rows * sizeof(int32_t)), "device hit rows");
        check(cudaMalloc(&device_count, sizeof(int32_t)), "device hit count");
    }
    void drain() { check(cudaStreamSynchronize(stream), "stream completion"); }
    void retire_graph() {
        drain();
        if (executable) { check(cudaGraphExecDestroy(executable), "graph executable retire"); executable = nullptr; }
        if (graph) { check(cudaGraphDestroy(graph), "source graph retire"); graph = nullptr; }
    }
    void reset(int64_t nr, int64_t nw, size_t shift, int nhit, unsigned seed) {
        drain();
        need(nr >= 0 && nr <= int64_t(max_rows) && nw >= 0 && nw <= int64_t(max_width) &&
             shift <= 1 && nhit >= 0 && nhit <= nr, "bounded fixture geometry");
        rows = nr; width = nw; count = nhit; offset = guard + shift;
        const float sentinel = std::bit_cast<float>(uint32_t(0x4b123456));
        std::fill(source.begin(), source.end(), sentinel);
        std::fill(initial.begin(), initial.end(), sentinel);
        for (size_t i = 0; i < size_t(rows * width); ++i)
            source[offset + i] = std::bit_cast<float>(patterns[(i + seed) % patterns.size()]);
        hits.fill(-77);
        for (int i = 0; i < count; ++i) hits[size_t(i)] = (i * 17) % int(rows);
        std::memcpy(host, source.data(), capacity * sizeof(float));
        check(cudaMemcpyAsync(device, initial.data(), capacity * sizeof(float), cudaMemcpyHostToDevice, stream), "destination guards");
        check(cudaMemcpyAsync(device_hits, hits.data(), hits.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "hit rows upload");
        check(cudaMemcpyAsync(device_count, &count, sizeof(count), cudaMemcpyHostToDevice, stream), "hit count upload");
        drain();
    }
    bool dispatch(bool candidate) {
        if (candidate)
            return kernels::copy_rows_from_mapped_subgroup(device + offset, mapped + offset,
                rows, width, device_hits, device_count, stream);
        kernels::copy_rows_from_mapped(device + offset, mapped + offset,
            rows, width, device_hits, device_count, stream);
        return false;
    }
    void capture(bool candidate) {
        need(!graph && !executable && rows > 0 && width > 0, "fresh nonempty capture");
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture begin");
        selected = dispatch(candidate);
        check(cudaStreamEndCapture(stream, &graph), "capture end");
        size_t nodes = 0;
        check(cudaGraphGetNodes(graph, nullptr, &nodes), "graph node count");
        need(nodes == 1, "one production copy kernel required");
        const bool wanted = candidate && width % 128 == 0 &&
            ((uintptr_t(device + offset) | uintptr_t(mapped + offset)) & 15) == 0;
        need(selected == wanted, "actual capture dispatch differs");
        check(cudaGraphInstantiate(&executable, graph, 0ull), "graph instantiate");
    }
    void replay() {
        check(compat::graph_launch_with_completion(executable, stream, completion), "graph replay");
        check(cudaEventSynchronize(completion), "actual copy completion");
    }
    void verify() {
        check(cudaMemcpyAsync(readback.data(), device, capacity * sizeof(float), cudaMemcpyDeviceToHost, stream), "guarded destination readback");
        std::array<int32_t, max_rows> seen_hits{};
        int32_t seen_count = -1;
        check(cudaMemcpyAsync(seen_hits.data(), device_hits, hits.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream), "hit rows readback");
        check(cudaMemcpyAsync(&seen_count, device_count, sizeof(count), cudaMemcpyDeviceToHost, stream), "hit count readback");
        drain();
        need(seen_hits == hits && seen_count == count, "readonly hit metadata changed");
        for (size_t i = 0; i < capacity; ++i) {
            uint32_t expected = bits(initial[i]);
            if (i >= offset && i < offset + size_t(rows * width)) {
                const int64_t row = int64_t((i - offset) / size_t(width));
                bool hit = false;
                for (int j = 0; j < count; ++j) hit |= hits[size_t(j)] == row;
                expected = hit ? 0u : bits(source[i]);
            }
            need(bits(readback[i]) == expected, "copy result or destination guard differs");
            need(bits(host[i]) == bits(source[i]), "host source or guard modified");
        }
    }
    void refresh(unsigned seed) {
        // Previous replay completed: the host owns this source until the next
        // enqueue. No host write overlaps a device read. Preparation is untimed.
        if (rows && width)
            source[offset + seed % size_t(rows * width)] = std::bit_cast<float>(patterns[seed % patterns.size()]);
        std::memcpy(host, source.data(), capacity * sizeof(float));
    }
    void close() {
        retire_graph();
        check(cudaEventDestroy(completion), "completion retire"); completion = nullptr;
        check(cudaFree(device_count), "count retire"); device_count = nullptr;
        check(cudaFree(device_hits), "hit rows retire"); device_hits = nullptr;
        check(cudaFree(device), "destination retire"); device = nullptr;
        check(cudaFreeHost(host), "host source retire"); host = mapped = nullptr;
        check(cudaStreamDestroy(stream), "stream retire"); stream = nullptr;
    }
};

void correctness(Fixture& f) {
    int cases = 0, replays = 0;
    for (bool candidate : {false, true}) {
        for (const auto shape : std::array<std::array<int, 3>, 9>{{
                {1, 128, 0}, {3, 128, 0}, {1, 2560, 0}, {10, 2560, 0},
                {20, 2560, 0}, {30, 2560, 0}, {40, 2560, 0}, {10, 19, 0}, {10, 2560, 1}}}) {
            f.reset(shape[0], shape[1], size_t(shape[2]), 0, 1);
            f.capture(candidate);
            for (int count : {0, shape[0] / 4, shape[0]}) {
                // Same graph, fresh source and hit metadata: those values must
                // be read at replay rather than frozen during construction.
                f.reset(shape[0], shape[1], size_t(shape[2]), count, unsigned(2 + count));
                f.replay(); f.verify(); ++replays;
            }
            f.retire_graph(); ++cases;
        }
    }
    f.reset(0, 2560, 0, 0, 1);
    need(!f.dispatch(true), "empty operation must not select a kernel"); f.drain(); f.verify();
    int refused = 0;
    auto rejects = [&](auto function) {
        try { function(); }
        catch (const std::invalid_argument&) { ++refused; return; }
        throw std::runtime_error("invalid span/dimension was admitted");
    };
    rejects([&] { kernels::copy_rows_from_mapped_subgroup(f.device, f.mapped, -1, 128, f.device_hits, f.device_count, f.stream); });
    rejects([&] { kernels::copy_rows_from_mapped_subgroup(f.device, f.mapped, std::numeric_limits<int64_t>::max(), 128, f.device_hits, f.device_count, f.stream); });
    rejects([&] { kernels::copy_rows_from_mapped_subgroup(f.device, nullptr, 1, 128, f.device_hits, f.device_count, f.stream); });
    rejects([&] { kernels::copy_rows_from_mapped_subgroup(f.mapped, f.mapped, 1, 128, f.device_hits, f.device_count, f.stream); });
    rejects([&] { kernels::copy_rows_from_mapped_subgroup(f.device, f.mapped + capacity - 1, 1, 128, f.device_hits, f.device_count, f.stream); });
    need(refused == 5 && cases == 18 && replays == 54, "complete correctness coverage");
    std::cout << "correctness=PASS captures=" << cases << " replays=" << replays
              << " refused=" << refused << " bitwise=PASS guards=PASS readonly=PASS dynamic_metadata=PASS\n";
}

double clock_pair_cost() {
    int64_t total = 0;
    for (int i = 0; i < 10000; ++i) {
        const auto a = Clock::now(), b = Clock::now();
        total += std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    }
    return double(total) / 10000;
}
double batch(Fixture& f, unsigned calls, bool update, unsigned& seed) {
    int64_t total = 0;
    for (unsigned i = 0; i < calls; ++i) {
        if (update) f.refresh(++seed);
        const auto a = Clock::now();
        f.replay();
        const auto b = Clock::now();
        total += std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    }
    return double(total);
}
void benchmark(Fixture& f, bool candidate) {
    const double pair_ns = clock_pair_cost();
    int samples = 0;
    for (int rows : {10, 20, 30, 40}) for (int pattern = 0; pattern < 3; ++pattern)
        for (bool update : {false, true}) {
            const int count = pattern == 0 ? 0 : pattern == 1 ? rows / 4 : rows;
            f.reset(rows, int(max_width), 0, count, 7);
            f.capture(candidate);
            unsigned calls = 64, seed = 1;
            double warmup = 0;
            do { warmup += batch(f, 512, update, seed); }
            while (warmup < 500000000.);
            f.verify();
            double calibration = 0;
            for (;;) {
                calibration = batch(f, calls, update, seed);
                if (calibration >= 750000000.) break;
                need(calls < 262144, "calibration call cap"); calls *= 2;
            }
            for (int rep = 0; rep < 3; ++rep) {
                const double elapsed = batch(f, calls, update, seed);
                f.verify();
                const double observer_fraction = pair_ns * calls / elapsed;
                if (elapsed < 250000000. || observer_fraction > .01)
                    throw std::runtime_error("ineligible duration/timer cost rows=" + std::to_string(rows) +
                        " hits=" + std::to_string(count) + " update=" + std::to_string(update) +
                        " calls=" + std::to_string(calls) + " elapsed_ns=" + std::to_string(elapsed) +
                        " calibration_ns=" + std::to_string(calibration) +
                        " timer_fraction=" + std::to_string(observer_fraction));
                std::cout << "{\"type\":\"sample\",\"rows\":" << rows << ",\"width\":2560,\"hits\":" << count
                    << ",\"producer_update\":" << (update ? "true" : "false") << ",\"subgroup_dispatch\":" << (f.selected ? "true" : "false")
                    << ",\"rep\":" << rep << ",\"calls\":" << calls << ",\"elapsed_ns\":" << elapsed
                    << ",\"ns_per_call\":" << elapsed / calls << ",\"clock_pair_ns\":" << pair_ns
                    << ",\"timer_fraction\":" << observer_fraction << ",\"calibration_ns\":" << calibration
                    << ",\"warmup_ns\":" << warmup << "}\n";
                ++samples;
            }
            f.retire_graph();
        }
    need(samples == 72, "complete fixed component matrix");
    std::cout << "benchmark=PASS samples=72 actual_graph_nodes=1 timing=launch_and_wait preparation=excluded physical_bus_bytes=unknown\n";
}
} // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf << std::setprecision(17);
    try {
        need((argc == 2 && std::string(argv[1]) == "correctness") ||
             (argc == 3 && std::string(argv[1]) == "bench" &&
              (std::string(argv[2]) == "scalar" || std::string(argv[2]) == "subgroup")), "use correctness|bench scalar|subgroup");
        const char* capture = std::getenv("STRATA_SYCL_VERIFY_NATIVE_CAPTURE");
        need(capture && std::string(capture) == "0", "qualified ordinary capture required");
        auto* fixture = new Fixture;
        fixture->open();
        if (argc == 2) correctness(*fixture);
        else benchmark(*fixture, std::string(argv[2]) == "subgroup");
        // The owner checks actual loaded providers while allocations and the
        // context are still alive; this is outside every timed interval.
        std::cout << "MAPPED_ROWS_WAIT\n";
        std::string reply;
        need(bool(std::getline(std::cin, reply)) && reply == "QUIT", "supervised retirement required");
        fixture->close(); delete fixture;
        std::cout << "fixture=PASS checked_teardown=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fixture=FAIL error=" << error.what() << '\n' << std::flush;
        std::_Exit(1); // Never free source/graph storage after uncertain work.
    }
}
