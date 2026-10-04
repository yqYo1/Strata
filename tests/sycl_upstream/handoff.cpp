#include "strata/sycl_upstream/handoff.hpp"
#include "strata/sycl_upstream/runtime.hpp"
#include <atomic>
#include <chrono>
#include <future>
#include <immintrin.h>
#include <iostream>
#include <vector>
using namespace strata::sycl_upstream;
using Clock = std::chrono::steady_clock;
struct alignas(64) Flag { uint32_t value; };
void check(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
int main() try {
    Runtime rt{sycl::device{sycl::gpu_selector_v}};
    sycl::queue allocation(rt.context(), rt.device(), sycl::property::queue::in_order{});
    const auto stream = rt.create_stream(true);
    std::cout << "device=" << rt.device().get_info<sycl::info::device::name>()
              << " usm_atomic_host=" << rt.device().has(sycl::aspect::usm_atomic_host_allocations)
              << " usm_atomic_shared=" << rt.device().has(sycl::aspect::usm_atomic_shared_allocations) << std::endl;
    constexpr int n = 4096, k = 10, layers = 48;
    auto* flags = sycl::aligned_alloc_host<Flag>(64, 3, allocation);
    auto* published = sycl::aligned_alloc_host<float>(64, n, allocation);
    auto* reply = sycl::aligned_alloc_host<float>(64, n, allocation);
    auto* ids_out = sycl::aligned_alloc_host<int32_t>(64, k, allocation);
    auto* ids_reply = sycl::aligned_alloc_host<int32_t>(64, k, allocation);
    auto* weights_out = sycl::aligned_alloc_host<float>(64, k, allocation);
    auto* x = sycl::malloc_device<float>(n, allocation);
    auto* ids = sycl::malloc_device<int32_t>(k, allocation);
    auto* weights = sycl::malloc_device<float>(k, allocation);
    auto* status = sycl::malloc_shared<uint32_t>(layers, allocation);
    auto* polls = sycl::malloc_shared<uint32_t>(layers, allocation);
    check(flags && published && reply && ids_out && ids_reply && weights_out && x && ids && weights && status && polls,
          "handoff allocation");
    auto* seq = &flags[0].value; auto* ack = &flags[1].value; auto* skip = &flags[2].value;
    *seq = *ack = *skip = 0;
    std::vector<float> input(n), output(n), w(k);
    std::vector<int32_t> initial_ids(k), final_ids(k);
    bool bounded = true;
    auto pipeline = [&] {
        for (int layer = 0; layer < layers; ++layer) rt.enqueue(stream, [=](sycl::queue& q) {
            doorbell_publish(q, x, ids, weights, n, k, published, ids_out, weights_out, seq);
            doorbell_wait(q, ack, seq, bounded ? HandoffWaitLimit{1000000, status + layer, polls + layer} : HandoffWaitLimit{});
            copy_from_mapped(q, x, reply, n);
            return copy_i32_from_mapped(q, ids, ids_reply, k);
        });
    };
    size_t handshakes = 0, values = 0, active_waits = 0;
    for (bool diagnostic : {true, false}) for (bool captured : {false, true}) {
        bounded = diagnostic;
        Runtime::Graph graph;
        if (captured) {
            const auto before = *seq;
            rt.begin_capture(stream); pipeline(); graph = rt.end_capture(stream);
            check(*seq == before && graph.node_count() == size_t(layers * 4), "handoff capture ran or lost commands");
        }
        const int repeats = captured ? 8 : 4;
        for (int run = 0; run < repeats; ++run) {
            const int seed = (captured ? 8192 : 0) + run * 512;
            for (int i = 0; i < n; ++i) input[i] = float(seed + i % 127);
            for (int j = 0; j < k; ++j) { initial_ids[j] = seed + j; w[j] = float(run + j) / 8; }
            rt.copy(stream, x, input.data(), n * sizeof(float));
            rt.copy(stream, ids, initial_ids.data(), k * sizeof(int32_t));
            rt.copy(stream, weights, w.data(), k * sizeof(float)); rt.synchronize(stream);
            std::fill(status, status + layers, 0);
            std::fill(polls, polls + layers, 0);
            const uint32_t base = *seq;
            std::promise<void> ready; auto waiting = ready.get_future();
            auto service = std::async(std::launch::async, [&] {
                bool ok = true; size_t checked = 0, rings = 0;
                ready.set_value();
                for (int layer = 0; layer < layers; ++layer) {
                    const uint32_t want = base + layer + 1;
                    const auto deadline = Clock::now() + std::chrono::seconds(5);
                    while (*static_cast<volatile uint32_t*>(seq) < want && Clock::now() < deadline) _mm_pause();
                    if (*static_cast<volatile uint32_t*>(seq) != want) { ok = false; break; }
                    std::atomic_thread_fence(std::memory_order_acquire);
                    for (int i = 0; i < n; ++i) {
                        ok &= published[i] == float(seed + i % 127 + layer * 3);
                        reply[i] = published[i] + 3; ++checked;
                    }
                    for (int j = 0; j < k; ++j) {
                        ok &= ids_out[j] == seed + j + layer * 2 && weights_out[j] == w[j];
                        ids_reply[j] = ids_out[j] + 2; checked += 2;
                    }
                    if (layer == 0) {
                        // Hold the first reply so the bounded variant records
                        // actual polling, not just an already-ready flag.
                        const auto release = Clock::now() + std::chrono::milliseconds(1);
                        while (Clock::now() < release) _mm_pause();
                    }
                    // Match the original x86 host publication sequence. The
                    // GPU keeps running; no runtime query or sync occurs here.
                    std::atomic_thread_fence(std::memory_order_seq_cst); _mm_sfence();
                    *static_cast<volatile uint32_t*>(ack) = want; ++rings;
                }
                return std::tuple{ok, checked, rings};
            });
            waiting.wait();
            if (captured) rt.launch(graph, stream); else pipeline();
            const auto [ok, checked, rings] = service.get(); // No driver calls during host service.
            rt.synchronize(stream);
            bool no_timeout = true;
            for (int layer = 0; layer < layers; ++layer) no_timeout &= status[layer] == 0;
            rt.copy(stream, output.data(), x, n * sizeof(float));
            rt.copy(stream, final_ids.data(), ids, k * sizeof(int32_t)); rt.synchronize(stream);
            bool final_ok = true;
            for (int i = 0; i < n; ++i) final_ok &= output[i] == input[i] + layers * 3;
            for (int j = 0; j < k; ++j) final_ok &= final_ids[j] == initial_ids[j] + layers * 2;
            if (!(ok && no_timeout && final_ok))
                std::cerr << "captured=" << captured << " run=" << run << " rings=" << rings
                          << " payload=" << ok << " no_timeout=" << no_timeout << " final=" << final_ok << '\n';
            check(ok && no_timeout && final_ok, "duplex mapped handoff failed");
            if (diagnostic) {
                check(polls[0] > 0, "delayed first reply did not exercise an active GPU wait");
                ++active_waits;
            }
            handshakes += rings; values += checked + n + k;
        }
    }
    // Exercise GE, cancellation's UINT32_MAX and the resident-plan skip branch,
    // including a changed skip value on replay. A bounded failed wait must be
    // reported explicitly rather than accepted because later copies ran.
    *ack = 3; *skip = 0; status[0] = 0;
    rt.enqueue(stream, [&](sycl::queue& q) { return wait_flag_ge(q, ack, 2, {8, status}); });
    rt.synchronize(stream); check(status[0] == 0, "GE comparison");
    *ack = UINT32_MAX;
    rt.enqueue(stream, [&](sycl::queue& q) { return wait_flag_ge(q, ack, 1234, {8, status}); });
    rt.synchronize(stream); check(status[0] == 0, "cancellation high-water flag");
    *ack = 0; *skip = 7;
    rt.begin_capture(stream);
    rt.enqueue(stream, [&](sycl::queue& q) { return wait_flag_ge_or(q, ack, 7, skip, {8, status}); });
    auto conditional = rt.end_capture(stream);
    rt.launch(conditional, stream); rt.synchronize(stream); check(status[0] == 0, "resident skip");
    *skip = 6; rt.launch(conditional, stream); rt.synchronize(stream);
    check(status[0] == 1, "changed skip must enter wait and report exhaustion");
    status[0] = 0; *ack = 7; rt.launch(conditional, stream); rt.synchronize(stream);
    check(status[0] == 0, "GE graph recovers with new host input");
    const auto before = *seq;
    rt.enqueue(stream, [&](sycl::queue& q) { return doorbell_ring(q, seq); }); rt.synchronize(stream);
    check(*seq == before + 1, "standalone ring");
    bool rejected = false;
    try { rt.enqueue(stream, [&](sycl::queue& q) { return doorbell_wait(q, ack, seq, {1, nullptr}); }); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "timeout without reporting storage accepted");
    // Upstream's null/empty guards must not insert native graph nodes or
    // discard the preceding command's ordering event.
    rt.begin_capture(stream);
    rt.enqueue(stream, [&](sycl::queue& q) { return doorbell_ring(q, nullptr); });
    rt.enqueue(stream, [&](sycl::queue& q) { return copy_from_mapped(q, nullptr, nullptr, 0); });
    bool empty = false;
    try { rt.end_capture(stream); } catch (const std::invalid_argument&) { empty = true; }
    check(empty, "no-op handoff made an executable graph");
    rt.begin_capture(stream);
    rt.enqueue(stream, [&](sycl::queue& q) {
        doorbell_ring(q, seq); doorbell_wait(q, nullptr, seq);
        return copy_i32_from_mapped(q, nullptr, nullptr, -1);
    });
    auto noops = rt.end_capture(stream);
    check(noops.node_count() == 1, "no-op changed graph dependencies or count");
    const auto old_seq = *seq;
    rt.launch(noops, stream); rt.synchronize(stream); check(*seq == old_seq + 1, "no-op tail lost preceding ring");
    constexpr size_t large_n = 65540; // Requires a second iteration after the 64-block cap.
    auto* large_host = sycl::aligned_alloc_host<float>(64, large_n + 4, allocation);
    auto* large_device = sycl::malloc_device<float>(large_n + 4, allocation);
    check(large_host && large_device, "large mapped-copy allocation");
    for (size_t i = 0; i < large_n + 4; ++i) large_host[i] = i < large_n ? float(i % 811 - 400.0) : -17.f;
    rt.copy(stream, large_device, large_host, (large_n + 4) * sizeof(float)); rt.synchronize(stream);
    for (size_t i = 0; i < large_n; ++i) large_host[i] += 3.f;
    rt.enqueue(stream, [&](sycl::queue& q) { return copy_from_mapped(q, large_device, large_host, large_n); });
    std::vector<float> large_out(large_n + 4);
    rt.copy(stream, large_out.data(), large_device, (large_n + 4) * sizeof(float)); rt.synchronize(stream);
    for (size_t i = 0; i < large_n + 4; ++i) check(large_out[i] == large_host[i], "mapped-copy grid stride or tail overwrite");
    sycl::free(large_host, allocation); sycl::free(large_device, allocation);
    rt.synchronize_device();
    for (void* p : std::initializer_list<void*>{flags, published, reply, ids_out, ids_reply, weights_out, x, ids, weights, status, polls})
        sycl::free(p, allocation);
    std::cout << "PASS handshakes=" << handshakes << " exact_payload_values=" << values
              << " direct_pipelines=8 graph_replays=16 graph_layers=48 capture_nodes=192"
              << " active_waits=" << active_waits
              << " ge_skip_replays=3 bounded_exhaustion=1 default_unbounded_pipelines=12 noop_graph_nodes=1 large_copy_values=65544 no_driver_calls_during_service=1\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
