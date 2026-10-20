#include "strata/core/expert_slot_sizes.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
void require(bool ok) { if (!ok) std::exit(1); }
template <class F> void rejects(F&& f) {
    try { f(); } catch (const std::exception&) { return; }
    std::exit(1);
}
template <class F> void test(const char* name, F&& f) {
    f(); std::printf("cache_tail case=%s status=PASS\n", name);
}
}

int main() {
    using strata::core::ExpertCacheTail;
    const uint64_t offsets[] = {0, 256, 768, 1536};
    const ExpertCacheTail sized(3, 1536, offsets, 768), uniform(4, 4096, nullptr, 1024);
    test("sized_boundary", [&] { require(sized.slots_for(768) == 1 && sized.bytes_from(2) == 768); });
    test("sized_cross_boundary", [&] { require(sized.slots_for(769) == 2 && sized.bytes_from(1) == 1280); });
    test("sized_all", [&] { require(sized.slots_for(1281) == 3 && sized.bytes_from(0) == 1536); });
    test("zero_need", [&] { require(sized.slots_for(0) == 0 && sized.bytes_from(3) == 0 && uniform.slots_for(0) == 0); });
    test("no_fit", [&] { require(sized.slots_for(1537) == 4 && sized.slots_for(UINT64_MAX) == 4 &&
                                 uniform.slots_for(UINT64_MAX) == 5); });
    test("uniform_boundary", [&] { require(uniform.slots_for(1024) == 1 && uniform.slots_for(1025) == 2 &&
                                           uniform.bytes_from(3) == 1024); });
    test("layer_tail_floor", [&] {
        const auto slots = strata::core::per_layer_slot_sizes({257, 511, 769}, 192);
        std::vector<uint64_t> off{0};
        for (const auto bytes : slots) off.push_back(off.back() + strata::core::expert_slot_aligned_bytes(bytes));
        const ExpertCacheTail tail(192, off.back(), off.data(), 769);
        require(tail.slots_for(65536) == 64 && 192 - tail.slots_for(65536) == 128 &&
                tail.bytes_from(128) == 65536 && 192 - tail.slots_for(65537) == 127);
    });
    test("bounds", [&] { rejects([&] { (void)sized.bytes_from(-1); }); rejects([&] { (void)sized.bytes_from(4); }); });
    test("invalid_layout", [&] {
        const uint64_t bad[] = {0, 512, 256, 1536};
        rejects([&] { ExpertCacheTail x(3, 1536, bad, 768); });
        rejects([&] { ExpertCacheTail x(3, 1535, offsets, 768); });
        rejects([&] { ExpertCacheTail x(4, 4095, nullptr, 1024); });
        rejects([&] { ExpertCacheTail x(4, 0, nullptr, UINT64_MAX); });
    });
    test("request_rounding", [] {
        using strata::core::prefill_request_chunk;
        require(prefill_request_chunk(1,8192)==256 && prefill_request_chunk(256,8192)==256 &&
                prefill_request_chunk(257,8192)==512 && prefill_request_chunk(8193,8192)==8192 &&
                prefill_request_chunk(INT64_MAX,8192)==8192 && prefill_request_chunk(0,8192)==0);
    });
    using strata::core::occupied_expert_tail_first;
    test("occupied_exact_tail", [] {
        const int32_t rows[]={0,127,132,-1,240};
        require(occupied_expert_tail_first(128,220,512,rows,5)==220);
    });
    test("occupied_widened_tail", [] {
        const int32_t rows[]={0,127,132,-1,131};
        const auto first=occupied_expert_tail_first(128,500,512,rows,5);
        const ExpertCacheTail tail(512,512*1024,nullptr,1024);
        require(first==132 && tail.bytes_from(first)>=tail.bytes_from(500));
    });
    test("occupied_prefix_only", [] {
        const int32_t rows[]={0,127,-1,-1};
        require(occupied_expert_tail_first(128,500,512,rows,4)==-1 &&
                occupied_expert_tail_first(128,500,512,nullptr,0)==-1);
    });
    test("occupied_invalid", [] {
        for (int32_t slot:{-2,512}) rejects([&]{(void)occupied_expert_tail_first(128,500,512,&slot,1);});
        rejects([]{(void)occupied_expert_tail_first(-1,500,512,nullptr,0);});
        rejects([]{(void)occupied_expert_tail_first(128,127,512,nullptr,0);});
        rejects([]{(void)occupied_expert_tail_first(128,512,512,nullptr,0);});
        rejects([]{(void)occupied_expert_tail_first(128,500,512,nullptr,1);});
    });
    std::puts("cache_tail_fixture cases=14 status=PASS");
}
