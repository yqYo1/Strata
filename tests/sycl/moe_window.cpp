// Main-authored MIT Strata fixture. Exercise production route/gather/combine;
// the synthetic expert witness deliberately does not stand in for GEMM math.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/draft_uploads.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_window_routes.hpp"
#include "strata/prefill/native16.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
namespace pd = strata::prefill::detail;
namespace pn = strata::prefill::native16;
namespace pk = strata::prefill;
namespace kb = strata::kernels;
constexpr size_t N = 2560, K = 10, J = 8192, guard_bytes = 256;
void need(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct ExpectedStop {};
enum class Fault { none, source, slot, expert, down };

// Every owner is constructed before the armed lease below. No constructor
// enqueues work, so an allocation failure also leaves no borrowed host source.
template<class T> struct Buffer {
    sb::Allocation owner;
    size_t size;
    Buffer(const std::shared_ptr<sb::Runtime>& rt, size_t n)
        : owner(rt, n * sizeof(T) + 2 * guard_bytes, sb::MemoryKind::Device), size(n) {}
    T* data() const {
        return reinterpret_cast<T*>(owner.as<unsigned char>() + guard_bytes);
    }
    void poison(sycl::queue& q) const { q.memset(owner.data(), 0xa5, owner.size()); }
    void guards(const std::shared_ptr<sb::Runtime>& rt) const {
        std::array<unsigned char, 2 * guard_bytes> observed{};
        // A read failure must not unwind the temporary host DMA destination.
        try {
            auto& q = rt->compute();
            q.memcpy(observed.data(), owner.data(), guard_bytes);
            q.memcpy(observed.data() + guard_bytes, data() + size, guard_bytes);
            rt->wait();
        } catch (...) { sb::cleanup_fail_stop("MoE fixture guard read did not retire"); }
        for (auto byte : observed) need(byte == 0xa5, "MoE fixture guard overwritten");
    }
};

// Distinguish each source row at a fixed column with finite FP16 bit patterns.
// The oracle uses original token IDs, never the route map or expert job list.
uint16_t input_bits(size_t row, size_t col, unsigned salt) {
    return uint16_t(0x0800u + (row * 97 + col * 13 + salt * 131) % 0x5000u);
}
uint16_t witness(uint16_t x, int expert) {
    return kb::bf16_from_f32(kb::f32_from_f16(x) + float(expert - 256) / 16.f);
}
float weight(size_t row, size_t slot, unsigned salt) {
    return float(1 + (row + slot * 3 + salt) % 7) / 128.f;
}
uint16_t shared_bits(size_t row, size_t col, unsigned salt) {
    return kb::bf16_from_f32(float(int((row + col * 3 + salt) % 17) - 8) / 32.f);
}
void record_mismatch(uint32_t* errors) {
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
        sycl::memory_scope::device, sycl::access::address_space::global_space>(*errors).fetch_add(1);
}

void run(const std::shared_ptr<sb::Runtime>& rt, int rows, bool concentrated,
         bool stop_after_gather, Fault fault, unsigned salt) {
    const size_t routes = size_t(rows) * K;
    std::vector<int32_t> ids(routes);
    for (int t = 0; t < rows; ++t) for (size_t k = 0; k < K; ++k)
        ids[size_t(t) * K + k] = int32_t(((concentrated ? salt : size_t(t) * 17 + salt) + k * 51) % 512);
    // Independent reference: stable-sort original token-slot indices by expert.
    // Production instead uses counts, prefix offsets, and scatter cursors.
    std::vector<int32_t> order(routes), expected_source(routes), expected_expert(routes), expected_slot(routes);
    std::iota(order.begin(), order.end(), int32_t(0));
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return ids[size_t(a)] < ids[size_t(b)]; });
    for (size_t at = 0; at < routes; ++at) {
        const size_t original_slot = size_t(order[at]);
        expected_source[at] = int32_t(original_slot / K);
        expected_expert[at] = ids[original_slot];
        expected_slot[original_slot] = int32_t(at);
    }
    const auto plan = pd::MoeWindowRoutes::build(rows, ids.data(), ids.size());
    need(plan->source_rows() == expected_source && plan->slot_map() == expected_slot,
         "route builder differs from independent stable-sort reference");
    size_t covered = 0;
    for (const auto& job : plan->jobs()) {
        need(job.rows > 0 && job.rows <= int(J) && job.global_row >= 0 &&
             size_t(job.global_row) == covered && size_t(job.rows) <= routes - covered,
             "route jobs have a gap, overlap, or invalid extent");
        need(expected_expert[covered] == job.expert &&
             expected_expert[covered + size_t(job.rows) - 1] == job.expert,
             "route job has the wrong expert");
        covered += size_t(job.rows);
    }
    need(covered == routes, "route jobs do not cover the entire Down plane");
    if (concentrated && rows > int(J)) {
        size_t split_experts = 0;
        for (auto n : plan->counts()) if (n == rows) ++split_experts;
        need(split_experts == K && plan->jobs().size() == 2 * K,
             "fixture did not exercise ne == window and split expert jobs");
    }
    Buffer<uint16_t> mixed(rt, size_t(rows) * N), shared(rt, size_t(rows) * N),
        down(rt, routes * N), gathered(rt, std::min<size_t>(rows, J) * N);
    Buffer<int32_t> source(rt, routes), slots(rt, routes), oracle_ids(rt, routes),
        gold_source(rt, routes), gold_expert(rt, routes), gold_slot(rt, routes);
    Buffer<float> weights(rt, routes), gates(rt, size_t(rows));
    Buffer<uint32_t> errors(rt, 4); // maps, exact gather/expert, Down, final combine
    std::array<uint32_t, 4> observed_errors{};
    bool drained_with_owners = false, stopped = false;
    auto& q = rt->compute();
    auto* m = mixed.data(); auto* sh = shared.data(); auto* d = down.data();
    auto* g = gathered.data(); auto* src = source.data(); auto* sl = slots.data();
    auto* original = oracle_ids.data(); auto* w = weights.data();
    auto* sg = gates.data(); auto* bad = errors.data();
    auto* gs = gold_source.data(); auto* ge = gold_expert.data(); auto* gl = gold_slot.data();
    // `mixed` becomes the output only after the final gather, as in the engine.
    try {
        pd::DraftUploadGuard lease([&]() noexcept {
            try { rt->wait(); drained_with_owners = true; return true; }
            catch (...) { return false; }
        }, []() noexcept { sb::cleanup_fail_stop("MoE fixture could not drain with route owners live"); });
        lease.arm();
        mixed.poison(q); shared.poison(q); down.poison(q); gathered.poison(q);
        source.poison(q); slots.poison(q); oracle_ids.poison(q);
        gold_source.poison(q); gold_expert.poison(q); gold_slot.poison(q);
        weights.poison(q); gates.poison(q); errors.poison(q);
        q.memcpy(src, plan->source_rows().data(), routes * sizeof(int32_t));
        q.memcpy(sl, plan->slot_map().data(), routes * sizeof(int32_t));
        q.memcpy(original, ids.data(), routes * sizeof(int32_t));
        q.memcpy(gs, expected_source.data(), routes * sizeof(int32_t));
        q.memcpy(ge, expected_expert.data(), routes * sizeof(int32_t));
        q.memcpy(gl, expected_slot.data(), routes * sizeof(int32_t));
        q.memset(bad, 0, 4 * sizeof(uint32_t));
        if (fault == Fault::source) q.single_task([=]() { src[0] = (src[0] + 1) % rows; });
        if (fault == Fault::slot) q.single_task([=]() { sl[0] = (sl[0] + 1) % int32_t(routes); });
        q.parallel_for(sycl::range<1>(routes), [=](sycl::id<1> index) {
            const size_t i = index[0];
            if (src[i] != gs[i] || sl[i] < 0 || sl[i] >= int32_t(routes) || sl[i] != gl[i])
                record_mismatch(bad);
        });
        q.parallel_for(sycl::range<1>(size_t(rows) * N), [=](sycl::id<1> index) {
            const size_t i = index[0], t = i / N, col = i % N;
            m[i] = input_bits(t, col, salt); sh[i] = shared_bits(t, col, salt);
        });
        q.parallel_for(sycl::range<1>(routes), [=](sycl::id<1> index) {
            const size_t i = index[0]; w[i] = weight(i / K, i % K, salt);
        });
        q.memset(sg, 0, size_t(rows) * sizeof(float));
        for (const auto& job : plan->jobs()) {
            const size_t offset = size_t(job.global_row), count = size_t(job.rows);
            const int expert = fault == Fault::expert && offset == 0 ? (job.expert + 1) % 512 : job.expert;
            sb::validate_device_span(m, size_t(rows) * N * sizeof(uint16_t), q);
            sb::validate_device_span(src + offset, count * sizeof(int32_t), q);
            sb::validate_device_span(d + offset * N, count * N * sizeof(uint16_t), q);
            pk::gather_rows16(m, src + offset, g, job.rows, N, &q);
            if (stop_after_gather) throw ExpectedStop{};
            q.parallel_for(sycl::range<1>(count * N), [=](sycl::id<1> index) {
                const size_t i = index[0], routed_row = offset + i / N;
                const size_t row = size_t(gs[routed_row]);
                if (g[i] != input_bits(row, i % N, salt) || expert != ge[routed_row]) record_mismatch(bad + 1);
                d[offset * N + i] = witness(g[i], expert);
            });
        }
        if (fault == Fault::down) q.single_task([=]() { d[size_t(gl[0]) * N] ^= uint16_t(1); });
        // Read the trusted expected slot, never a possibly faulty actual slot.
        // This detects a wrong Down witness even if final BF16 rounding masks it.
        q.parallel_for(sycl::range<1>(routes * N), [=](sycl::id<1> index) {
            const size_t i = index[0], original_slot = i / N, col = i % N;
            const auto want = witness(input_bits(original_slot / K, col, salt), original[original_slot]);
            if (d[size_t(gl[original_slot]) * N + col] != want) record_mismatch(bad + 2);
        });
        pn::combine(d, sl, w, sh, sg, m, rows, int64_t(routes), &q);
        q.parallel_for(sycl::range<1>(size_t(rows) * N), [=](sycl::id<1> index) {
            const size_t i = index[0], t = i / N, col = i % N;
            float expected = 0;
            for (size_t k = 0; k < K; ++k)
                expected = sycl::fma(weight(t, k, salt),
                    kb::f32_from_bf16(witness(input_bits(t, col, salt), original[t * K + k])), expected);
            const auto want = kb::bf16_from_f32(expected + kb::f32_from_bf16(shared_bits(t, col, salt)) * .5f);
            if (m[i] != want) record_mismatch(bad + 3);
        });
        q.memcpy(observed_errors.data(), bad, sizeof(observed_errors));
        rt->wait(); drained_with_owners = true; lease.retire();
    } catch (const ExpectedStop&) { stopped = true; }
    need(drained_with_owners, "fixture did not retire before releasing upload owners");
    need(stopped == stop_after_gather, "fixture stop injection was not reached");
    if (!stopped) {
        if (fault == Fault::none)
            need(std::all_of(observed_errors.begin(), observed_errors.end(), [](auto count) { return count == 0; }),
                 "MoE route/gather/Down/combine witness mismatch");
        else if (fault == Fault::source)
            need(observed_errors[0] > 0 && observed_errors[1] > 0,
                 "independent map/gather checks missed corrupt source");
        else if (fault == Fault::slot)
            need(observed_errors[0] > 0 && observed_errors[1] == 0 && observed_errors[2] == 0,
                 "independent map check missed corrupt slot or unrelated check failed");
        else if (fault == Fault::expert)
            need(observed_errors[0] == 0 && observed_errors[1] > 0,
                 "exact expert identity check missed wrong expert");
        else if (fault == Fault::down)
            need(observed_errors[0] == 0 && observed_errors[1] == 0 && observed_errors[2] == 1,
                 "pre-combine check missed single corrupt Down element");
    }
    mixed.guards(rt); shared.guards(rt); down.guards(rt); gathered.guards(rt);
    source.guards(rt); slots.guards(rt); oracle_ids.guards(rt);
    gold_source.guards(rt); gold_expert.guards(rt); gold_slot.guards(rt);
    weights.guards(rt); gates.guards(rt); errors.guards(rt);
    std::printf("moe_window case rows=%d concentrated=%d salt=%u stopped=%d fault=%d jobs=%zu checks_completed=%d PASS\n",
                rows, int(concentrated), salt, int(stopped), int(fault), plan->jobs().size(), int(!stopped));
    if (!stopped) std::printf("moe_window counters maps=%u gather_expert=%u down=%u combine=%u\n",
                              observed_errors[0], observed_errors[1], observed_errors[2], observed_errors[3]);
    std::fflush(stdout);
}
} // namespace

int main(int argc, char** argv) {
    try {
        need(argc == 1 || (argc == 3 && std::string(argv[1]) == "--case"),
             "usage: sycl_moe_window_test [--case smoke|single|hot8193|hot16384|balanced16384|tail16383|unwind|negative-source|negative-slot|negative-expert|negative-down]");
        const std::string name = argc == 1 ? "smoke" : argv[2];
        int rows = 3; bool hot = false, stop = false;
        Fault fault = Fault::none;
        if (name == "single") rows = 1;
        else if (name == "hot8193") { rows = 8193; hot = true; }
        else if (name == "hot16384") { rows = 16384; hot = true; }
        else if (name == "balanced16384") rows = 16384;
        else if (name == "tail16383") rows = 16383;
        else if (name == "unwind") stop = true;
        else if (name == "negative-source") fault = Fault::source;
        else if (name == "negative-slot") fault = Fault::slot;
        else if (name == "negative-expert") fault = Fault::expert;
        else if (name == "negative-down") fault = Fault::down;
        else need(name == "smoke", "unknown MoE fixture case");
        const auto rt = sb::runtime_for();
        // Separate owners and different routes/operands; no process env mutation.
        run(rt, rows, hot, stop, fault, 1); run(rt, rows, hot, stop, fault, 37);
        std::puts("moe_window PASS (synthetic route/gather/combine only; no model/GEMM/state/speed claim)");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "moe_window FAIL: %s\n", error.what());
        return 1;
    }
}
