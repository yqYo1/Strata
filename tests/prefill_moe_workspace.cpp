// Authored for Strata; repository LICENSE applies. Pure CPU capacity checks.
// No SYCL/runtime/model dependency and no inference-performance claim.
#include "strata/prefill/moe_accum_plan.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using Plan = strata::prefill::detail::MoeAccumPlan;
constexpr uint64_t legacy_r = 8192ull * 10240 * 4;
// Eight guarded takes in native14508's gdn_set_bytes(8192). The actual
// shared region is max(GDN,QSA,MoE), hence cannot be smaller than this.
constexpr uint64_t gdn_region_lower_bound = 1180698624;

void require(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}
template<class F> void refuses(F function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid capacity was accepted");
}

void inspect(const Plan& plan) {
    require(plan.rows > Plan::maximum_rows && plan.rows <= Plan::long_prefill_maximum_rows,
            "long-prefill range");
    require(plan.accumulator_primary_rows == 65536 && plan.primary_rows == 65536,
            "unchanged chunk-aligned bank boundaries");
    uint64_t region = 0, extra = plan.accumulator_tail_allocation_bytes();
    unsigned relocated = 0;
    for (size_t i = 0; i < size_t(Plan::Plane::count); ++i) {
        const auto& span = plan.span(Plan::Plane(i));
        const bool moved = i == size_t(Plan::Plane::gathered) || i == size_t(Plan::Plane::logits);
        require((span.storage == Plan::Storage::extra) == moved, "unexpected relocated plane");
        auto& used = moved ? extra : region;
        require(span.offset == used && span.offset % 256 == 0 &&
                span.reserved >= span.bytes + 256 && span.reserved % 256 == 0,
                "plane overlap, alignment or missing guard");
        used += span.reserved;
        relocated += moved;
    }
    require(relocated == 2 && region == plan.region_bytes && extra == plan.extra_bytes,
            "owner extent conservation");
    require(plan.primary_bytes <= legacy_r && region + plan.tail_bytes <= gdn_region_lower_bound,
            "mixed bank exceeds its existing owner");
    for (int64_t at = 0; at < plan.rows; at += Plan::chunk) {
        const auto count = std::min(Plan::chunk, plan.rows - at);
        const auto mixed = plan.mixed_piece(at,count), accum = plan.accumulator_piece(at,count);
        require(mixed.tail == (at >= 65536) && accum.tail == mixed.tail &&
                mixed.bytes == uint64_t(count)*2560*2 &&
                accum.bytes == uint64_t(count)*2560*4, "logical chunk mapping");
    }
}

int main() try {
    const auto ordinary = Plan::make_scratch_borrowed(81920,legacy_r,gdn_region_lower_bound);
    require(ordinary.region_bytes == 1059131904 && ordinary.tail_bytes == 83886080 &&
            ordinary.extra_bytes == 0, "ordinary layout changed");
    for (const auto& span : ordinary.planes)
        require(span.storage == Plan::Storage::region, "ordinary path acquired extra storage");
    refuses([]{Plan::make_split_scratch(131071,legacy_r,gdn_region_lower_bound);});
    uint64_t checked = 0;
    for (int64_t rows = 81921; rows <= 131072; ++rows) {
        const auto next = Plan::make_long_prefill_scratch(rows,legacy_r,gdn_region_lower_bound);
        inspect(next); ++checked;
        if (rows <= Plan::split_maximum_rows) {
            const auto prior = Plan::make_split_scratch(rows,legacy_r,gdn_region_lower_bound);
            require(prior.primary_bytes == next.primary_bytes && prior.tail_bytes == next.tail_bytes &&
                    prior.accumulator_tail_bytes() == next.accumulator_tail_bytes(),
                    "tensor payload or rounding/storage contract changed");
            for (size_t i = 0; i < prior.planes.size(); ++i)
                require(prior.planes[i].bytes == next.planes[i].bytes &&
                        prior.planes[i].reserved == next.planes[i].reserved,
                        "a plane's payload or guard changed");
        }
    }
    const auto last = Plan::make_long_prefill_scratch(131071,legacy_r,gdn_region_lower_bound);
    require(checked == 49152 && last.extra_bytes == 729799424 &&
            last.region_bytes + last.tail_bytes == 1177025536,
            "131K byte census differs from the source arithmetic");
    constexpr uint64_t free_observed = 4245643264, stride = 2662400;
    constexpr uint64_t prefix_and_status = 131072ull*2056+256;
    const uint64_t candidate = 131071ull*20480 + stride + prefix_and_status +
                               (512ull<<20) + last.extra_bytes;
    require(candidate == 4223151104 && free_observed-candidate == 22492160,
            "historical free-memory scenario differs");
    for (int64_t rows : std::array<int64_t,8>{-1,0,8192,81920,131073,262143,262144,INT64_MAX})
        refuses([&]{Plan::make_long_prefill_scratch(rows,legacy_r,gdn_region_lower_bound);});
    refuses([]{Plan::make_long_prefill_scratch(131071,0,gdn_region_lower_bound);});
    refuses([]{Plan::make_long_prefill_scratch(131071,legacy_r,1);});
    refuses([&]{last.mixed_piece(65535,8192);});
    refuses([&]{last.accumulator_piece(131072,1);});
    std::cout << "prefill_moe_workspace CPU_PASS rows_checked=" << checked
              << " moved_planes=2 extra_131071=" << last.extra_bytes
              << " historical_margin=22492160 GPU=UNRUN runtime_admission=UNRUN"
                 " model_math=UNRUN model_performance=UNRUN physical262144_model=UNRUN\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
}
