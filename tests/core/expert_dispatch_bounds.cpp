// Main-authored production-dispatch bounds controls; repository LICENSE.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace core = strata::core;
namespace {
void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct Source final : core::ExpertSource {
    std::vector<uint8_t> raw = std::vector<uint8_t>(cpu::BLOB, 0);
    int calls = 0;
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        need(layer == 0 && expert == 0, "source coordinates");
        ++calls;
        return raw.data();
    }
};
void check(cpu::NativeGuPhase phase, int k, bool overflow) {
    Source source;
    cpu::ExpertPool pool(1, false, true);
    core::ExpertDispatch d;
    d.pool = &pool;
    d.src = &source;
    d.n_expert = 1;
    d.native_gu_phase = phase;
    // No cache was admitted: even an explicit experimental policy must retain
    // the existing canonical path. This test never initializes a GPU owner.
    d.compact_gu_policy = cpu::compact_gu::BatchPolicy::MixedNt;
    std::array<float, cpu::H> x{};
    std::vector<int32_t> ids(size_t(k), 0);
    constexpr float guard = 12345.25f;
    std::vector<float> output(size_t(k) * cpu::H + 2, guard);
    core::expert_pool_dispatch_multi(d, x.data(), ids.data(), 1, k, output.data() + 1);
    need(output.front() == guard && output.back() == guard, "output guard");
    need(source.calls == 1, "one distinct expert source read");
    if (overflow) {
        need(d.failed && d.fail && std::strstr(d.fail, "native job token capacity") &&
            d.fail_expert == 0 && d.layers == 0, "duplicate expert overflow refusal");
        need(d.jobs_multi[0].nt == cpu::MAXT && d.multi_entries == cpu::MAXT,
            "refuse before writing the next fixed-array entry");
        need(std::all_of(output.begin(), output.end(), [](float v) { return v == guard; }),
            "overflow must not publish CPU output");
    } else {
        need(!d.failed && d.layers == 1 && d.multi_entries == k && d.multi_misses == 1,
            "legal final native job slot");
        for (size_t i = 1; i + 1 < output.size(); ++i) {
            uint32_t bits;
            std::memcpy(&bits, &output[i], sizeof bits);
            need((bits & 0x7fffffffu) == 0, "actual canonical zero-weight result");
        }
    }
}
}
int main() try {
    need(!cpu::expert_layout().native, "fresh process canonical layout");
    static_assert(cpu::MAXT + 2 <= 128);
    for (auto phase : {cpu::NativeGuPhase::Prefill, cpu::NativeGuPhase::Decode}) {
        check(phase, cpu::MAXT, false);
        check(phase, cpu::MAXT + 1, true);
        check(phase, cpu::MAXT + 2, true);
    }
    std::puts("expert-dispatch-bounds: passed cases=6 phase=2 GPU_executed=0");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "expert-dispatch-bounds: %s\n", e.what());
    return 1;
}
