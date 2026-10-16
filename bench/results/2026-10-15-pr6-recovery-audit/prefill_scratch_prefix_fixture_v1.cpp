// Compile only after applying the reviewed alias proposal. Include the actual
// production translation unit to exercise its internal Alloc and prefix planner;
// do not duplicate the planner or link the old prefill.cpp archive member.
#include "prefill.cpp"
#include <iostream>

namespace {
void require_fixture(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
struct Span {
    const void* pointer;
    size_t bytes;
    const char* name;
};
void check_prefix(size_t rows, bool alias) {
    using namespace strata::prefill;
    Alloc count;
    count.count_only = true;
    bool ok = true;
    (void)take_scratch_prefix(count, rows, alias, ok);
    require_fixture(ok && count.used > 0, "count-only production planner");
    std::unique_ptr<uint8_t[]> memory(new uint8_t[size_t(count.used)+256]);
    std::memset(memory.get()+count.used, 0xa7, 256);
    Alloc actual;
    actual.base = memory.get();
    actual.cap = count.used;
    const auto p = take_scratch_prefix(actual, rows, alias, ok);
    require_fixture(ok && actual.used == count.used, "count/borrowed prefix differ");
    require_fixture((p.xn != nullptr) == gr_unfused(), "unfused normalized image");
    require_fixture((p.xn16_lo != nullptr) == bf16x2_hc() &&
                    (p.lo16_lo != nullptr) == bf16x2_hc() &&
                    (p.mixed_bf_lo != nullptr) == bf16x2(), "low image branches");
    require_fixture(alias ? p.emb == p.mixed && p.emb == p.bo :
                    p.emb != p.mixed && p.emb != p.bo && p.mixed != p.bo,
                    "shared N-wide identities");
    const std::vector<Span> spans = {
        {p.emb, rows*N*4, "emb"}, {p.R, rows*D*4, "R"},
        {p.xn, rows*D*4, "xn"}, {p.grs, rows*HC*4, "grs"},
        {p.xn16, rows*D*2, "xn16"}, {p.lo, rows*LR*4, "lo"},
        {p.lo16, rows*LR*2, "lo16"}, {p.gated, rows*D*4, "gated"},
        {p.inj, rows*HC*4, "inj"}, {p.mixed, rows*N*4, "mixed"},
        {p.mixed_bf, rows*N*2, "mixed_bf"}, {p.mixed_h, rows*N*2, "mixed_h"},
        {p.bo, rows*N*4, "bo"}, {p.xn16_lo, rows*D*2, "xn16_lo"},
        {p.lo16_lo, rows*LR*2, "lo16_lo"}, {p.mixed_bf_lo, rows*N*2, "mixed_bf_lo"}};
    const uintptr_t begin = reinterpret_cast<uintptr_t>(memory.get());
    for (size_t i = 0; i < spans.size(); ++i) {
        const auto& a = spans[i];
        if (!a.pointer) continue;
        const uintptr_t ab = reinterpret_cast<uintptr_t>(a.pointer);
        require_fixture(ab >= begin && ab-begin <= actual.used &&
                        a.bytes <= actual.used-(ab-begin), "payload outside counted prefix");
        for (size_t j = 0; j < i; ++j) {
            const auto& b = spans[j];
            if (!b.pointer) continue;
            const uintptr_t bb = reinterpret_cast<uintptr_t>(b.pointer);
            const bool shared = alias && a.pointer == p.emb && b.pointer == p.emb;
            require_fixture(shared || ab+a.bytes <= bb || bb+b.bytes <= ab,
                            "unexpected live prefix overlap");
        }
    }
    for (size_t i = 0; i < 256; ++i)
        require_fixture(memory[count.used+i] == 0xa7, "borrowed end canary");
    Alloc too_small;
    too_small.base = memory.get();
    too_small.cap = count.used-1;
    bool rejected = true;
    (void)take_scratch_prefix(too_small, rows, alias, rejected);
    require_fixture(!rejected && too_small.used <= too_small.cap,
                    "one-byte-short prefix admitted");
    for (size_t i = 0; i < 256; ++i)
        require_fixture(memory[count.used+i] == 0xa7, "failed take changed canary");
    std::cout << "prefix rows=" << rows << " alias=" << alias
              << " bytes=" << count.used << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        require_fixture(argc == 2, "expected cached flag mode argument");
        const int expected = std::stoi(argv[1]);
        require_fixture(strata::prefill::scratch_alias_mode() == expected, "immutable strict flag");
        if (expected < 0) {
            strata::core::ModelGeometry geometry;
            strata::core::SessionState session;
            bool rejected = false;
            try { (void)strata::prefill::Prefill::bytes_needed(geometry, session, 8192); }
            catch (const std::invalid_argument&) { rejected = true; }
            require_fixture(rejected, "invalid flag reached geometry/GPU planner");
            std::cout << "prefix_flag PASS rejected_before_geometry no_Prefill_instance no_GPU_API\n";
            return 0;
        }
        for (size_t rows : {size_t(1), size_t(17), size_t(256), size_t(4096), size_t(8192)}) {
            check_prefix(rows, false);
            check_prefix(rows, true);
        }
        std::cout << "prefix_layout PASS cases=10 production_Alloc_and_prefix no_GPU_API\n";
    } catch (const std::exception& e) {
        std::cerr << "prefix fixture failed: " << e.what() << '\n';
        return 1;
    }
}
