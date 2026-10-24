// Authored for Strata; repository LICENSE applies. No GPU or runtime access.
#include "strata/prefill/expert_bank_plan.hpp"
#include "strata/core/expert_layer_view.hpp"
#include <cstdio>

using namespace strata::prefill::detail;
using Selection = ExpertBankPlan::Selection;
constexpr uint64_t Stride = 2662400, Free = 4257304576;
constexpr uint64_t Other = 241555584, Headroom = 512ull << 20;
size_t negative = 0;
void need(bool ok) { if (!ok) throw std::runtime_error("expert bank invariant failed"); }
template<class Error = std::invalid_argument, class F> void rejects(F f) {
    bool caught = false;
    try { f(); } catch (const Error&) { caught = true; }
    need(caught); ++negative;
}
ExpertBankPlan plan(int slots, uint64_t free = Free, int64_t rows = 117488,
                    int64_t requested = -1, bool supported = true, uint64_t max_alloc = 4ull << 30) {
    return ExpertBankPlan::make(rows, 8192, 10240, Stride, free, Other, Headroom,
        max_alloc, requested, LayerMajorResidualDtype::bf16, slots, supported);
}

int main() try {
    std::array<uint8_t, 128> source{};
    using View = strata::core::ResidentExpertLayer;
    const auto view = View::make(source.data(), 64, 16, 3, 16);
    need(view.data == source.data() + 16 && view.bytes == 48 && view.stride == 16 && view.experts == 3);
    need(View::make(source.data(), 64, 0, 4, 16).bytes == 64);
    need(View::make(source.data(), 64, 63, 1, 1).data == source.data() + 63);
    need(View::make(source.data(), 64, 1, 3, 7).bytes == 21); // host memcpy has no256B stride requirement
    need(!View::make(source.data(), 64, 17, 3, 16).data);
    need(!View::make(source.data(), 64, 65, 1, 1).data);
    need(!View::make(source.data(), 64, 64, 1, 1).data);
    need(!View::make(source.data(), 0, 0, 1, 1).data);
    need(!View::make(nullptr, 64, 0, 1, 1).data);
    need(!View::make(source.data(), 64, 0, -1, 1).data);
    need(!View::make(source.data(), 64, 0, 0, 1).data);
    need(!View::make(source.data(), 64, 0, 1, 0).data);
    need(!View::make(source.data(), 64, 0, std::numeric_limits<int64_t>::max(), 3).data);
    need(!View::make(source.data(), std::numeric_limits<uint64_t>::max(), 0, 1, 1).data);
    std::puts("resident_layer_view valid_cases=4 rejected_extents=10 PASS");
    const auto original = plan(0);
    need(!original.staged() && original.slots == 512 && original.device_bytes == 1363148800);
    need(original.residual.gpu_rows == 98304 && original.residual.host_bytes == 392888320);
    const auto smaller = plan(384);
    need(smaller.staged() && smaller.slots == 384 && smaller.device_bytes == 1022361600);
    need(smaller.host_upload_bytes == 1363148800 && smaller.residual.gpu_rows == 117488);
    need(smaller.residual.host_bytes == 0 && smaller.residual.total_bytes == 2406154240);
    need(original.device_bytes - smaller.device_bytes == 340787200);
    const uint64_t exact = Other + Headroom + 1022361600 + 2406154240;
    need(plan(384, exact).staged());
    const auto below = plan(384, exact - 1);
    need(!below.staged() && below.selection == Selection::insufficient && below.slots == 512);
    need(below.residual.gpu_rows == 98304);
    const auto explicit_prefix = plan(384, Free, 117488, 81920);
    need(!explicit_prefix.staged() && explicit_prefix.selection == Selection::explicit_prefix);
    need(explicit_prefix.residual.gpu_rows == 81920);
    need(plan(384, Free, 117488, 117488).staged());
    need(plan(1).staged() && plan(1).device_bytes == Stride);
    need(plan(512).selection == Selection::insufficient);
    need(plan(384, Free, 117488, -1, false).selection == Selection::unsupported);
    for (int64_t rows : {81921, 81937, 90112}) {
        const auto short_tail = plan(384, Free, rows);
        need(short_tail.selection == Selection::unsupported && short_tail.slots == 512);
        need(short_tail.residual.gpu_rows == rows); // all GPU, but normal FFN tail
    }
    need(plan(384, Free, 90113).staged()); //81920+8193: both accumulator windows
    // Hypothetical stricter devices: total free bytes must not bypass the
    // single-allocation limit. This fixture performs no device allocation.
    const auto two_gib = plan(384, Free, 117488, -1, true, 2ull << 30);
    need(!two_gib.staged() && two_gib.slots == 512 && two_gib.residual.gpu_rows == 98304);
    const auto smaller_limit = plan(384, Free, 117488, -1, true, 1536ull << 20);
    need(!smaller_limit.staged() && smaller_limit.residual.gpu_rows == 73728);
    need(plan(384, Free, 117488, -1, true, 2406154240).staged());
    need(!plan(384, Free, 117488, -1, true, 2406154239).staged());
    const auto full_context = plan(384, Free, 262144);
    need(!full_context.staged() && full_context.slots == 512);
    need(full_context.residual.rows == 262144 && full_context.residual.gpu_rows == 98304);
    need(full_context.residual.gpu_bytes + full_context.residual.host_bytes == 5368709120ull);
    rejects([] { (void)plan(-1); });
    rejects([] { (void)plan(513); });
    rejects([] { (void)plan(384, 0); });
    rejects([] { (void)plan(0, Free, 117488, 117489); });
    rejects([] { (void)plan(384, Free, 117488, -1, true, 0); });
    rejects([] { (void)plan(384, Free, 117488, -1, true, 1ull << 30); });
    rejects([] { (void)plan(384, Free, 117488, 117488, true, 2ull << 30); });
    rejects([] { (void)ExpertBankPlan::make(117488,8192,10240,Stride-1,Free,Other,Headroom,
        4ull<<30,-1,LayerMajorResidualDtype::bf16,384,true); });
    rejects<std::overflow_error>([] { (void)ExpertBankPlan::make(117488,8192,10240,
        uint64_t(1)<<63,Free,Other,Headroom,4ull<<30,-1,LayerMajorResidualDtype::bf16,384,true); });
    std::puts("expert_bank_capacity full117488=GPU partial_request=preserved fallback262144=original exact_boundary=PASS short_tail_fallback_cases=3 single_alloc_limit_cases=4");

    // Observed standalone262144-context budget. Materializing both BF16 K/V
    // tensors must not prevent the existing full-bank host-tail path. Nothing
    // here permits shrinking decode placement or the512MiB safety headroom.
    constexpr uint64_t ActualFree = 2039808000, FullPrefix = 538968064;
    auto prefix_plan = [&](int64_t rows, uint64_t prefix, uint64_t free, int64_t request=-1,
                           uint64_t prefix_limit=10132807680ull, int64_t bank_slots=384) {
        return ExpertBankPrefixPlan::make(prefix, prefix_limit, [&](uint64_t admitted) {
            return ExpertBankPlan::make(rows,8192,10240,Stride,free,admitted,Headroom,
                prefix_limit,request,LayerMajorResidualDtype::bf16,bank_slots,true);
        });
    };
    const auto short_prefix = prefix_plan(16384,16384*2056ull,ActualFree);
    need(short_prefix.prefix_bytes==16384*2056ull && short_prefix.bank.staged());
    need(short_prefix.bank.residual.gpu_rows==16384);
    for (int64_t rows : {262143,262144}) {
        const auto full_prefix = prefix_plan(rows,FullPrefix,ActualFree);
        need(full_prefix.prefix_bytes==0 && !full_prefix.bank.staged() && full_prefix.bank.slots==512);
        need(full_prefix.bank.residual.gpu_rows==0 && full_prefix.bank.residual.host_bytes==uint64_t(rows)*20480);
    }
    constexpr uint64_t Minimum = 512*Stride+Headroom;
    need(prefix_plan(262143,FullPrefix,Minimum+FullPrefix).prefix_bytes==FullPrefix);
    need(prefix_plan(262143,FullPrefix,Minimum+FullPrefix-1).prefix_bytes==0);
    need(prefix_plan(262143,FullPrefix,Minimum).prefix_bytes==0);
    // A long nonzero-prefix request may have only8193 new rows and a legal
    // one-slot bank. Mandatory owners fit a stricter hypothetical device limit
    // while the full262144-cell prefix needs its own one-shot check.
    need(prefix_plan(8193,FullPrefix,ActualFree,-1,FullPrefix,1).prefix_bytes==FullPrefix);
    need(prefix_plan(8193,FullPrefix,ActualFree,-1,FullPrefix-1,1).prefix_bytes==0);
    unsigned budget_rejected=0;
    try { (void)prefix_plan(262143,FullPrefix,Minimum-1); }
    catch (const ExpertBankBudgetError&) { ++budget_rejected; }
    try { (void)prefix_plan(16384,16384*2056ull,Free,16385); }
    catch (const std::invalid_argument&) { ++budget_rejected; }
    try { (void)prefix_plan(262143,std::numeric_limits<uint64_t>::max(),ActualFree); }
    catch (const std::overflow_error&) { ++budget_rejected; }
    try { (void)prefix_plan(262143,FullPrefix,Minimum+FullPrefix,-1,0); }
    catch (const std::invalid_argument&) { ++budget_rejected; }
    try { (void)prefix_plan(8193,FullPrefix,ActualFree,-1,128ull<<20,1); }
    catch (const ExpertBankBudgetError&) { ++budget_rejected; }
    need(budget_rejected==5);
    std::puts("expert_bank_prefix_budget short_kept=PASS full262143_host_tail=PASS full262144_host_tail=PASS exact=PASS single_allocation=PASS invalid=PASS");

    // A host oracle models complete slot contents; accepted enqueue updates it.
    // Two ascending scans must retain three permanent hits without scan eviction.
    ExpertBankSlots slots(4);
    std::array<int,4> contents{-1,-1,-1,-1};
    int uploads = 0, hits = 0;
    for (int round = 0; round < 2; ++round) {
        for (int expert : {0,1,2,3,4,5}) {
            auto t = slots.begin(expert);
            if (t.upload) { contents[size_t(t.slot)] = expert; slots.queued(t); ++uploads; }
            else ++hits;
            need(contents[size_t(t.slot)] == expert);
        }
    }
    need(uploads == 9 && hits == 3 && slots.permanent_used() == 3);
    rejects([&] { (void)slots.begin(-1); });
    rejects([&] { (void)slots.begin(512); });
    auto pending = slots.begin(6);
    rejects([&] { (void)slots.begin(7); });
    rejects([&] { slots.reset(); });
    auto wrong = pending; ++wrong.expert;
    rejects([&] { slots.queued(wrong); });
    slots.queued(pending);
    rejects([&] { slots.queued(pending); });
    auto next = slots.begin(7);
    rejects([&] { slots.queued(pending); });
    slots.queued(next);
    slots.reset();
    need(slots.permanent_used() == 0 && slots.begin(0).upload);
    rejects([] { ExpertBankSlots bad(0); });
    rejects([] { ExpertBankSlots bad(513); });
    ExpertBankSlots one(1);
    for (int expert : {0,511,0}) {
        auto t = one.begin(expert);
        need(t.upload && !t.permanent && t.slot == 0);
        one.queued(t);
    }
    need(one.permanent_used() == 0);
    need(negative == 18);
    std::printf("expert_bank_slots uploads=%d hits=%d negative_checks=%zu PASS\n",uploads,hits,negative);
    const size_t packed_negative_begin = negative;
    const uint64_t bank_bytes = 384 * Stride;
    const std::array<uint64_t,7> layer_strides{2176000,1510400,1971200,1715200,2329600,1868800,2662400};
    const std::array<int,7> capacities{469,512,512,512,438,512,384};
    for (size_t i = 0; i < layer_strides.size(); ++i) {
        const auto geometry = ExpertBankLayout::packed(bank_bytes, layer_strides[i]);
        need(geometry.slots == capacities[i] && geometry.stride == layer_strides[i]);
        need(geometry.used_bytes <= bank_bytes && geometry.backing_bytes == bank_bytes);
        need(geometry.offset(geometry.slots - 1) + geometry.stride == geometry.used_bytes);
        rejects([&] { (void)geometry.offset(-1); });
        rejects([&] { (void)geometry.offset(geometry.slots); });
        ExpertBankSlots mapped(geometry.slots, geometry.full_fit());
        std::array<int,512> oracle{}; oracle.fill(-1);
        unsigned copied = 0, reused = 0;
        for (int layer = 0; layer < 2; ++layer) {
            mapped.reset();
            for (int round = 0; round < 2; ++round) for (int expert = 0; expert < 512; ++expert) {
                const auto t = mapped.begin(expert);
                if (t.upload) { oracle[size_t(t.slot)] = expert; mapped.queued(t); ++copied; }
                else ++reused;
                need(oracle[size_t(t.slot)] == expert);
                need(geometry.offset(t.slot) + geometry.stride <= bank_bytes);
            }
        }
        const auto permanent = unsigned(geometry.slots - (geometry.full_fit() ? 0 : 1));
        need(copied == 2 * (1024 - permanent) && reused == 2 * permanent);
    }
    const auto one_exact = ExpertBankLayout::packed(256,256);
    need(one_exact.slots == 1 && one_exact.used_bytes == 256 && !one_exact.full_fit());
    need(ExpertBankLayout::packed(512*256-1,256).slots == 511);
    need(ExpertBankLayout::packed(512*256,256).full_fit());
    rejects([] { (void)ExpertBankLayout::packed(0,256); });
    rejects([] { (void)ExpertBankLayout::packed(256,0); });
    rejects([] { (void)ExpertBankLayout::packed(255,256); });
    rejects([] { (void)ExpertBankLayout::packed(512,255); });
    rejects([] { ExpertBankSlots invalid(511,true); });
    std::printf("expert_bank_packed formats=7 layers_each=2 full_fit512=PASS boundary=PASS negative_checks=%zu PASS\n",
        negative - packed_negative_begin);
    std::puts("expert_bank_plan PASS GPU=UNRUN model=UNRUN physical262144_RESTORE=UNQUALIFIED");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr,"expert_bank_plan failed: %s\n",e.what());
    return 1;
}
