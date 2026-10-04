#include "strata/sycl_upstream/runtime.hpp"
#include "strata/sycl_upstream/mmq_product.hpp"
#include "strata/sycl_upstream/mmq_stages.hpp"
#include "upstream_cpu_dequant.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <vector>
using namespace strata::sycl_upstream;
using namespace std::chrono_literals;
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void rejects(F f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid capture operation was accepted");
}
void write(Runtime& rt, Runtime::Stream stream, int* dst, int value) {
    rt.enqueue(stream, [=](sycl::queue& q){ return q.single_task([=]{ *dst = value; }); });
}
void add(Runtime& rt, Runtime::Stream stream, const int* src, int* dst, int value) {
    rt.enqueue(stream, [=](sycl::queue& q){ return q.single_task([=]{ *dst = *src + value; }); });
}
struct Gate {
    std::promise<void> release;
    std::shared_future<void> future = release.get_future().share();
    bool opened = false;
    void open() { if (!opened) { release.set_value(); opened = true; } }
    ~Gate() { open(); }
};
int main() try {
    Runtime rt{sycl::device{sycl::gpu_selector_v}};
    std::cout << "device=" << rt.device().get_info<sycl::info::device::name>() << '\n';
    const auto a = rt.create_stream(true), b = rt.create_stream(true), ordinary = rt.create_stream();
    auto* data = sycl::malloc_shared<int>(16, rt.device(), rt.context());
    check(data, "allocation"); std::fill(data, data + 16, 0);
    int scenarios = 0, replays = 0;
    {
        auto temporary = rt.create_stream();
        rt.begin_capture(temporary);
        add(rt, temporary, data, data + 1, 7);
        add(rt, temporary, data + 1, data + 2, 11);
        check(rt.capturing(temporary), "capture state");
        auto graph = rt.end_capture(temporary);
        check(graph.node_count() == 2 && data[1] == 0 && data[2] == 0, "recording executed work or lost nodes");
        rt.destroy_stream(temporary);
        Runtime::Event done;
        for (int i = 0; i < 12; ++i) {
            data[0] = i * 13;
            rt.launch(graph, i % 2 ? a : b); rt.record(done, i % 2 ? a : b); rt.synchronize(done);
            check(data[1] == i * 13 + 7 && data[2] == i * 13 + 18, "replay used stale input"); ++replays;
        }
        ++scenarios;
    }
    {
        Runtime::Event fork, joined;
        data[3] = 0; data[4] = 0;
        rt.begin_capture(a);
        add(rt, a, data, data + 1, 1); rt.record(fork, a);
        rt.wait_event(b, fork); add(rt, b, data + 1, data + 2, 2); rt.record(joined, b);
        rt.wait_event(a, joined); add(rt, a, data + 2, data + 3, 3);
        auto graph = rt.end_capture(a);
        check(!rt.capturing(a) && !rt.capturing(b) && data[3] == 0, "fork capture executed work");
        for (int i = 0; i < 8; ++i) {
            data[0] = 100 + i;
            rt.launch(graph, ordinary); rt.synchronize(ordinary);
            check(data[3] == 106 + i, "capture event fork/join"); ++replays;
        }
        rejects([&]{ rt.query(fork); }); // Internal graph event is not exported as a completion event.
        rt.record(fork, a); rt.synchronize(fork); check(rt.query(fork), "event rerecord after capture");
        ++scenarios;
    }
    {
        std::atomic<int> calls{0};
        auto* host = sycl::malloc_host<int>(2, rt.context());
        auto* dev = sycl::malloc_device<int>(2, rt.device(), rt.context());
        check(host && dev, "copy graph allocation");
        host[0] = 0; host[1] = -1;
        rt.begin_capture(a);
        rt.copy(a, dev, host, sizeof(int)); add(rt, a, dev, dev + 1, 5);
        rt.copy(a, host + 1, dev + 1, sizeof(int));
        rt.host_function(a, [&]{ calls.fetch_add(1); });
        auto graph = rt.end_capture(a);
        check(calls == 0 && host[1] == -1, "capture ran host task or copy");
        for (int i = 0; i < 5; ++i) {
            host[0] = i * 17; rt.launch(graph, a); rt.synchronize(a);
            check(host[1] == i * 17 + 5 && calls == i + 1, "graph copy/host function replay"); ++replays;
        }
        graph = Runtime::Graph{};
        sycl::free(host, rt.context()); sycl::free(dev, rt.context()); ++scenarios;
    }
    {
        rejects([&]{ rt.begin_capture(0); });
        rt.begin_capture(a); rejects([&]{ rt.end_capture(a); });
        check(!rt.capturing(a), "empty capture not cleared");
        rt.begin_capture(a); write(rt, a, data + 4, 999);
        rejects([&]{ rt.begin_capture(a); });
        auto wrong_thread = std::async(std::launch::async, [&]{ rejects([&]{ rt.end_capture(a); }); });
        wrong_thread.get();
        rejects([&]{ rt.end_capture(b); });
        rt.abort_capture(a); check(data[4] == 0, "abort executed capture");
        rt.begin_capture(a); write(rt, a, data + 4, 999);
        rejects([&]{ rt.synchronize(a); }); rejects([&]{ rt.end_capture(a); });
        rt.begin_capture(ordinary); write(rt, ordinary, data + 4, 999);
        rejects([&]{ write(rt, 0, data + 4, 888); }); rejects([&]{ rt.end_capture(ordinary); });
        Runtime::Event fork;
        rt.begin_capture(a); rt.record(fork, a); rt.wait_event(b, fork); write(rt, b, data + 4, 999);
        rejects([&]{ rt.end_capture(a); });
        check(!rt.capturing(a) && !rt.capturing(b) && data[4] == 0, "unjoined capture not discarded");
        rt.begin_capture(a);
        bool failed = false;
        try { rt.enqueue(a, [&](sycl::queue& q)->sycl::event {
            q.single_task([=]{ data[4] = 999; }); throw std::runtime_error("capture body failed");
        }); } catch (const std::runtime_error&) { failed = true; }
        rejects([&]{ rt.end_capture(a); }); check(failed && data[4] == 0, "failed capture was executable");
        rt.begin_capture(a); rt.begin_capture(b);
        write(rt, a, data + 4, 999); write(rt, b, data + 4, 888);
        rejects([&]{ rt.synchronize_device(); });
        rejects([&]{ rt.end_capture(a); }); rejects([&]{ rt.end_capture(b); });
        rt.begin_capture(a); rt.begin_capture(b);
        rt.record(fork, a); rejects([&]{ rt.wait_event(b, fork); });
        rejects([&]{ rt.end_capture(a); }); rejects([&]{ rt.end_capture(b); });
        check(data[4] == 0, "invalidating multiple captures executed work");
        write(rt, a, data + 4, 42); rt.synchronize(a); check(data[4] == 42, "queue not restored after rejected capture");
        ++scenarios;
    }
    {
        data[5] = 0;
        rt.begin_capture(a);
        rt.enqueue(a, [=](sycl::queue& q){ return q.single_task([=]{ ++data[5]; }); });
        auto graph = rt.end_capture(a);
        rt.launch(graph, a); rt.synchronize(a); ++replays; // Warm before the bounded host observation.
        Gate gate; const auto held = gate.future;
        rt.host_function(a, [held]{ held.wait(); });
        auto submitted = std::async(std::launch::async, [&]{ rt.launch(graph, a); rt.launch(graph, b); });
        const bool asynchronous = submitted.wait_for(2s) == std::future_status::ready;
        if (!asynchronous) { gate.open(); submitted.get(); throw std::runtime_error("graph launch waited for prior host task"); }
        submitted.get();
        const bool second_pending = !rt.query(b);
        std::promise<void> started; auto entering = started.get_future();
        auto destroyed = std::async(std::launch::async, [&]{ started.set_value(); graph = Runtime::Graph{}; });
        entering.wait(); const bool released = destroyed.wait_for(2s) == std::future_status::ready;
        // Management must also preserve a released executable still in flight.
        auto temporary = rt.create_stream(true); rt.destroy_stream(temporary);
        const bool still_pending = !rt.query(b);
        gate.open(); destroyed.get(); rt.synchronize_device(); replays += 2;
        check(second_pending && released && still_pending && data[5] == 3, "graph replay serialization or destruction lifetime");
        ++scenarios;
    }
    size_t mmq_comparisons = 0;
    {
        constexpr int cols = 512, ff = 512, total = 3;
        sycl::queue allocation_queue(rt.context(), rt.device(), sycl::property::queue::in_order{});
        const size_t gu_bytes = mmq_matrix_bytes(GGML_TYPE_Q8_0, ff * 2, cols);
        const size_t down_bytes = mmq_matrix_bytes(GGML_TYPE_Q8_0, cols, ff);
        auto* guw = sycl::malloc_shared<block_q8_0>(gu_bytes / sizeof(block_q8_0) + 128, allocation_queue);
        auto* dw = sycl::malloc_shared<block_q8_0>(down_bytes / sizeof(block_q8_0) + 128, allocation_queue);
        auto* x = sycl::malloc_shared<float>(total * cols, allocation_queue);
        auto* gu = sycl::malloc_shared<float>(total * 2 * ff, allocation_queue);
        auto* h = sycl::malloc_shared<float>(total * ff, allocation_queue);
        auto* dst = sycl::malloc_shared<float>(total * cols, allocation_queue);
        auto* ids = sycl::malloc_shared<int32_t>(total, allocation_queue);
        auto* bounds = sycl::malloc_shared<int32_t>(2, allocation_queue);
        auto* xq = sycl::malloc_shared<MmqBlock>(mmq_q8_bytes(total, cols) / sizeof(MmqBlock), allocation_queue);
        auto* hq = sycl::malloc_shared<MmqBlock>(mmq_q8_bytes(total, ff) / sizeof(MmqBlock), allocation_queue);
        check(guw && dw && x && gu && h && dst && ids && bounds && xq && hq, "MMQ graph allocation");
        for (int which = 0; which < 2; ++which) {
            auto* w = which ? dw : guw;
            const size_t count = (which ? down_bytes : gu_bytes) / sizeof(block_q8_0) + 128;
            for (size_t b0 = 0; b0 < count; ++b0) {
                w[b0].d = sycl::half(1.f / 256);
                for (int k = 0; k < 32; ++k) w[b0].qs[k] = int((b0 + k * 13) % 7) - 3;
            }
        }
        bounds[0] = 0; bounds[1] = total;
        MmqProduct pg{guw,GGML_TYPE_Q8_0,2*ff,cols,gu_bytes,1,xq,bounds,ids,total,total,gu,2*ff};
        MmqProduct pd{dw,GGML_TYPE_Q8_0,cols,ff,down_bytes,1,hq,bounds,ids,total,total,dst,cols};
        const int units = rt.device().get_info<sycl::info::device::max_compute_units>();
        const auto gp = mmq_plan(pg, units), dp = mmq_plan(pd, units);
        auto* scratch = sycl::malloc_device<float>(std::max(gp.scratch_bytes, dp.scratch_bytes) / 4 + 1, allocation_queue);
        check(scratch, "MMQ graph scratch");
        auto body = [&](sycl::queue& q) {
            mmq_iota(q, ids, total); mmq_quantize(q, x, nullptr, xq, GGML_TYPE_Q8_0, cols, cols, total);
            mmq_product(q, pg, gp, scratch); mmq_swiglu(q, gu, h, total, ff, false);
            mmq_quantize(q, h, nullptr, hq, GGML_TYPE_Q8_0, ff, ff, total);
            return mmq_product(q, pd, dp, scratch);
        };
        std::fill(dst, dst + total * cols, -1234567.f);
        rt.begin_capture(a); rt.enqueue(a, body); auto graph = rt.end_capture(a);
        check(graph.node_count() == 8 && dst[0] == -1234567.f, "MMQ capture node count or execution");
        for (int run = 0; run < 4; ++run) {
            for (int i = 0; i < total * cols; ++i) x[i] = float((i * 17 + run * 23) % 701 - 350) / 701.f;
            rt.enqueue(a, body); rt.synchronize(a);
            std::vector<float> reference(dst, dst + total * cols);
            std::fill(dst, dst + total * cols, -1234567.f);
            rt.launch(graph, b); rt.synchronize(b); ++replays;
            for (int i = 0; i < total * cols; ++i) {
                check(std::isfinite(dst[i]) && dst[i] == reference[i], "captured MMQ chain disagrees with direct chain");
                ++mmq_comparisons;
            }
        }
        graph = Runtime::Graph{};
        sycl::free(scratch, allocation_queue); sycl::free(guw, allocation_queue); sycl::free(dw, allocation_queue);
        sycl::free(x, allocation_queue); sycl::free(gu, allocation_queue); sycl::free(h, allocation_queue); sycl::free(dst, allocation_queue);
        sycl::free(ids, allocation_queue); sycl::free(bounds, allocation_queue); sycl::free(xq, allocation_queue); sycl::free(hq, allocation_queue);
        ++scenarios;
    }
    sycl::free(data, rt.context());
    std::cout << "PASS graph_scenarios=" << scenarios << " replays=" << replays
              << " exact_mmq_graph_outputs=" << mmq_comparisons << " fork_join=1 invalidation=1 async_launch_and_teardown=1\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
