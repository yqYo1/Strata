// Authored for Strata; repository LICENSE applies. Mechanism/ownership probe,
// NOT an LLM throughput benchmark. Compare serial copy/read with a two-queue
// pipeline under the unchanged qualified Runtime and startup safety settings.
#include "strata/prefill/expert_copy_pipeline.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

namespace sb = strata::sycl_backend;
namespace pd = strata::prefill::detail;
using Clock = std::chrono::steady_clock;
constexpr size_t Guard = 64;
constexpr uint32_t Sentinel = 0xa5a5a5a5;
void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
uint32_t word(size_t index, int expert, unsigned seed) {
    return (uint32_t(index) * 747796405u ^ uint32_t(expert + 1) * 2891336453u ^ seed * 277803737u) | 1u;
}
void gate(const char* stage) {
    if (!std::getenv("STRATA_TRANSFER_PROBE_GATES")) return;
    std::cout << "GATE stage=" << stage << std::endl;
    std::string input;
    need(bool(std::getline(std::cin, input)) && input == "continue", "controller acknowledgement");
}
struct Drain {
    sycl::queue queue;
    std::shared_ptr<sb::Runtime> runtime;
    ~Drain() {
        try { queue.wait_and_throw(); runtime->check(); } catch (...) { std::_Exit(86); }
    }
};
struct Interval { uint64_t begin, end; };
Interval interval(const sycl::event& e) {
    return {e.get_profiling_info<sycl::info::event_profiling::command_start>(),
            e.get_profiling_info<sycl::info::event_profiling::command_end>()};
}
uint64_t overlap(std::vector<Interval> a, std::vector<Interval> b) {
    auto merge = [](std::vector<Interval> rows) {
        std::sort(rows.begin(), rows.end(), [](auto x, auto y) { return x.begin < y.begin; });
        std::vector<Interval> out;
        for (auto r : rows) {
            need(r.end >= r.begin, "device event interval reversed");
            if (out.empty() || out.back().end < r.begin) out.push_back(r);
            else out.back().end = std::max(out.back().end, r.end);
        }
        return out;
    };
    a = merge(std::move(a)); b = merge(std::move(b));
    uint64_t total = 0;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        const auto first = std::max(a[i].begin, b[j].begin), last = std::min(a[i].end, b[j].end);
        if (last > first) total += last - first;
        if (a[i].end < b[j].end) ++i; else ++j;
    }
    return total;
}
void timing_controls() {
    std::vector<uint64_t> times;
    auto prior = Clock::now();
    for (int i = 0; i < 1000; ++i) {
        const auto now = Clock::now();
        times.push_back(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now - prior).count()));
        prior = now;
    }
    std::sort(times.begin(), times.end());
    std::cout << "TIMER samples=1000 p99_ns=" << times[990] << " max_ns=" << times.back() << '\n';
    auto runtime = sb::runtime_for();
    auto q = runtime->make_queue(true);
    for (int i = -1; i < 5; ++i) {
        const auto begin = Clock::now();
        q.ext_oneapi_submit_barrier().wait_and_throw(); runtime->check();
        std::cout << "CONTROL repetition=" << i
            << " wall_s=" << std::chrono::duration<double>(Clock::now() - begin).count() << '\n';
    }
}

void batch(bool pipeline, size_t bytes, int loops, int count, int repetition,
           bool cancel = false, bool reject_controls = false) {
    need(bytes && bytes % 256 == 0 && loops >= 1 && count >= 1 && count <= 64, "bounded fixture geometry");
    auto runtime = sb::runtime_for();
    auto queue = runtime->make_queue(true);
    const size_t words = bytes / 4, stride = bytes + 256, source_words = 512 * stride / 4;
    const unsigned seed = unsigned(repetition + 7);
    std::vector<uint32_t> host(source_words + 2 * Guard, Sentinel);
    auto* source = reinterpret_cast<const uint8_t*>(host.data() + Guard);
    std::vector<int32_t> order(size_t(count), 0);
    // Skips/gaps make overfetch and assuming expert==ordinal observable.
    for (int e = 0; e < count; ++e) order[size_t(e)] = e * 3 + 1;
    for (auto e : order) for (size_t i = 0; i < words; ++i)
        host[Guard + size_t(e) * stride / 4 + i] = word(i, e, seed);
    sb::Allocation bank(runtime, 2 * stride + 2 * Guard * 4, sb::MemoryKind::Device);
    sb::Allocation output(runtime, size_t(count) * bytes + 2 * Guard * 4, sb::MemoryKind::Device);
    sb::Allocation status(runtime, 4, sb::MemoryKind::Device);
    auto* device = bank.as<uint8_t>() + Guard * 4;
    auto* results = output.as<uint32_t>() + Guard;
    auto* bad = status.as<uint32_t>();
    std::vector<uint32_t> observed(output.size() / 4), bank_observed(bank.size() / 4);
    Drain cleanup{queue, runtime}; // Before every borrowed host/device owner can unwind.
    queue.memset(bank.data(), 0xa5, bank.size());
    queue.memset(output.data(), 0xa5, output.size());
    queue.memset(status.data(), 0, 4);
    queue.wait_and_throw(); runtime->check();
    std::vector<sycl::event> copies, readers;
    copies.reserve(size_t(count)); readers.reserve(size_t(count) * 2);
    uint32_t flags = ~uint32_t(0);
    size_t completed = 0, uploads = 0;
    unsigned rejected = 0;
    bool cancelled = false;
    const auto begin = Clock::now();
    try {
        std::unique_ptr<pd::ExpertCopyPipeline> pipe;
        if (pipeline) {
            if (reject_controls) {
                auto reject = [&](auto action) {
                    bool caught = false;
                    try { action(); } catch (const std::invalid_argument&) { caught = true; }
                    need(caught, "invalid admission accepted"); ++rejected;
                };
                reject([&] { pd::ExpertCopyPipeline p(device, 2 * stride, stride, source, 512 * stride - 1, stride, bytes, order, &queue); });
                reject([&] { pd::ExpertCopyPipeline p(device, stride, stride, source, 512 * stride, stride, bytes, order, &queue); });
                reject([&] { pd::ExpertCopyPipeline p(device, 2 * stride, stride, source, 512 * stride, stride, stride + 1, order, &queue); });
                reject([&] { pd::ExpertCopyPipeline p(device, 2 * stride, stride, source, 512 * stride, stride, bytes, {1, 1}, &queue); });
                reject([&] { pd::ExpertCopyPipeline p(device, 2 * stride, stride, source, 512 * stride, stride, bytes, {2, 1}, &queue); });
                reject([&] { pd::ExpertCopyPipeline p(device, 2 * stride, stride, source, 512 * stride, stride, bytes, {512}, &queue); });
                reject([&] { pd::ExpertCopyPipeline p(device, 2 * stride, stride, nullptr, 512 * stride, stride, bytes, order, &queue); });
            }
            pipe = std::make_unique<pd::ExpertCopyPipeline>(device, 2 * stride, stride, source,
                512 * stride, stride, bytes, order, &queue);
            pipe->start();
            if (reject_controls) {
                auto reject = [&](auto action) {
                    bool caught = false;
                    try { action(); } catch (const std::invalid_argument&) { caught = true; }
                    need(caught, "invalid active order accepted"); ++rejected;
                };
                reject([&] { pipe->start(); });
                reject([&] { pipe->consumed(0); });
                reject([&] { pipe->acquire(1); });
                reject([&] { pipe->retire(); });
            }
        }
        for (int e = 0; e < count; ++e) {
            const int expert = order[size_t(e)];
            const uint32_t* blob = nullptr;
            if (pipe) {
                const auto lease = pipe->acquire(size_t(e));
                need(lease.expert == expert && lease.ordinal == size_t(e), "pipeline ticket mismatch");
                blob = reinterpret_cast<const uint32_t*>(lease.data);
                copies.push_back(lease.uploaded);
            } else {
                auto* destination = device + size_t(e % 2) * stride;
                copies.push_back(queue.memcpy(destination, source + size_t(expert) * stride, bytes));
                blob = reinterpret_cast<const uint32_t*>(destination);
            }
            readers.push_back(queue.parallel_for(sycl::range<1>(words), [=](sycl::id<1> id) {
                const size_t i = id[0];
                uint32_t x = blob[i];
                if (x != word(i, expert, seed))
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device, sycl::access::address_space::global_space>(*bad).fetch_or(1);
                // Synthetic dependent integer work; its service is reported.
                // It neither models XMX GEMM nor predicts LLM throughput.
                for (int step = 0; step < loops; ++step) x = x * 1664525u + 1013904223u;
                results[size_t(e) * words + i] = x;
            }));
            // A separate final reader detects release before all consumers.
            readers.push_back(queue.parallel_for(sycl::range<1>(words), [=](sycl::id<1> id) {
                const size_t i = id[0];
                if (blob[i] != word(i, expert, seed))
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device, sycl::access::address_space::global_space>(*bad).fetch_or(2);
            }));
            ++completed;
            if (pipe) pipe->consumed(size_t(e));
            if (cancel && e == std::min(count - 1, 4)) {
                if (pipe) uploads = pipe->copies();
                throw std::runtime_error("owned cancellation");
            }
        }
        if (pipe) { pipe->retire(); uploads = pipe->copies(); }
        queue.wait_and_throw(); runtime->check();
    } catch (const std::runtime_error& error) {
        need(cancel && std::string(error.what()) == "owned cancellation", error.what());
        cancelled = true;
        queue.wait_and_throw(); runtime->check();
    }
    const auto end = Clock::now();
    need(cancelled == cancel, "cancellation route mismatch");
    queue.memcpy(&flags, status.data(), 4);
    queue.memcpy(observed.data(), output.data(), output.size());
    queue.memcpy(bank_observed.data(), bank.data(), bank.size());
    queue.wait_and_throw(); runtime->check();
    need(flags == 0, "GPU reader saw overwritten/wrong expert bytes");
    uint32_t a = 1, b = 0;
    for (int i = 0; i < loops; ++i) { a *= 1664525u; b = b * 1664525u + 1013904223u; }
    for (size_t e = 0; e < size_t(count); ++e) for (size_t i = 0; i < words; ++i)
        need(observed[Guard + e * words + i] == (e < completed ? a * word(i, order[e], seed) + b : Sentinel),
            "complete output mismatch");
    for (size_t i = 0; i < Guard; ++i)
        need(observed[i] == Sentinel && observed[observed.size() - 1 - i] == Sentinel &&
             bank_observed[i] == Sentinel && bank_observed[bank_observed.size() - 1 - i] == Sentinel &&
             host[i] == Sentinel && host[host.size() - 1 - i] == Sentinel, "outer guard changed");
    for (size_t slot = 0; slot < 2; ++slot) for (size_t i = words; i < stride / 4; ++i)
        need(bank_observed[Guard + slot * stride / 4 + i] == Sentinel, "device slot tail guard changed");
    for (int e = 0; e < 512; ++e) {
        const bool used = e > 0 && (e - 1) % 3 == 0 && (e - 1) / 3 < count;
        for (size_t i = 0; i < stride / 4; ++i)
            need(host[Guard + size_t(e) * stride / 4 + i] == (used && i < words ? word(i, e, seed) : Sentinel),
                "immutable source or source padding changed");
    }
    need(!pipeline || uploads == std::min(size_t(count), completed + (cancel ? pd::ExpertCopyPipeline::depth : 0)),
         "pipeline copy count mismatch");
    std::vector<Interval> copy_times, reader_times;
    uint64_t copy_ns = 0, reader_ns = 0;
    for (const auto& e : copies) { auto x = interval(e); copy_ns += x.end - x.begin; copy_times.push_back(x); }
    for (const auto& e : readers) { auto x = interval(e); reader_ns += x.end - x.begin; reader_times.push_back(x); }
    const auto overlap_ns = overlap(copy_times, reader_times);
    uint64_t first = UINT64_MAX, last = 0;
    for (const auto& x : copy_times) { first = std::min(first, x.begin); last = std::max(last, x.end); }
    for (const auto& x : reader_times) { first = std::min(first, x.begin); last = std::max(last, x.end); }
    std::cout << "PIPELINE arm=" << (pipeline ? "pipeline" : "serial") << " bytes=" << bytes
        << " loops=" << loops << " repetition=" << repetition << " experts=" << count
        << " completed=" << completed << " requested_uploads=" << (pipeline ? uploads : completed)
        << " observed_uploads=" << copies.size() << " readers=" << readers.size()
        << " wall_s=" << std::chrono::duration<double>(end - begin).count()
        << " copy_s=" << double(copy_ns) / 1e9 << " reader_s=" << double(reader_ns) / 1e9
        << " overlap_s=" << double(overlap_ns) / 1e9 << " device_span_s=" << double(last - first) / 1e9
        << " cancelled=" << int(cancelled)
        << " admission_rejections=" << rejected << " source=PASS guards=PASS output=PASS drain=PASS\n";
}

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(12) << std::unitbuf;
        const bool smoke = argc == 2 && std::string(argv[1]) == "--smoke";
        const bool compare = argc == 3 && std::string(argv[1]) == "--compare" &&
            (std::string(argv[2]) == "0" || std::string(argv[2]) == "1" || std::string(argv[2]) == "2");
        need(smoke || compare, "choose --smoke or --compare 0|1|2");
        gate("before-runtime");
        auto rt = sb::runtime_for();
        need(rt->device().is_gpu() && rt->device().get_info<sycl::info::device::vendor_id>() == 0x8086,
             "Intel GPU required");
        std::cout << "DEVICE name=" << rt->device().get_info<sycl::info::device::name>() << '\n';
        gate("runtime-ready");
        timing_controls();
        if (smoke) {
            batch(false, 4096, 16, 8, 0);
            batch(true, 4096, 16, 8, 0, false, true);
            batch(true, 4096, 16, 8, 1, true);
            batch(true, 4096, 16, 1, 2); // fewer experts than pipeline depth
        } else {
            const int process = argv[2][0] - '0';
            for (const size_t bytes : {size_t(1971200), size_t(2662400)})
                for (const int loops : {16, 256}) {
                    // First use stays separate. Five paired subsequent batches
                    // alternate order, with the starting arm crossed by process.
                    for (int repetition = -1; repetition < 5; ++repetition)
                        for (int arm = 0; arm < 2; ++arm)
                            batch(bool((arm + repetition + 1 + process) % 2), bytes, loops, 32,
                                  repetition);
                }
        }
        rt->wait(); rt->check();
        std::cout << "DONE mechanism_only=1 decode_changed=0\n";
        gate("before-exit");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL expert_pipeline=" << error.what() << '\n';
        std::cerr.flush(); std::_Exit(1);
    }
}
