
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <algorithm>
#define _MM_HINT_T0 3
static uintptr_t base_address, end_address;
static std::vector<uintptr_t> captured;
static uint64_t reads = 0;
static void capture_prefetch(const char* p, int hint) {
    const auto a = reinterpret_cast<uintptr_t>(p);
    if (hint != _MM_HINT_T0 || a < base_address || a >= end_address) {
        std::fprintf(stderr, "invalid hint address %llu outside [%llu,%llu)\n",
                     (unsigned long long)a, (unsigned long long)base_address,
                     (unsigned long long)end_address);
        std::exit(3);
    }
    captured.push_back(a);
    volatile uint8_t observed = *reinterpret_cast<const uint8_t*>(p);
    (void)observed;
    ++reads;
}
int prefetch_distance() {
    static const int d = [] {
        const char* v = std::getenv("STRATA_IQ_PREFETCH");
        if (!v) return 2048;
        int distance = 0;
        const auto parsed = std::from_chars(v, v + std::strlen(v), distance);
        // Invalid, overflowing and nonpositive values disable the hint. No atoi overflow.
        return parsed.ec == std::errc{} && *parsed.ptr == '\0' && distance > 0 ? distance : 0;
    }();
    return d;
}
inline void rows_ahead(const uint8_t* blk, const uint8_t* end, int pf) {
    if (pf <= 0) return;
    const size_t remaining = static_cast<size_t>(end - blk);
    const size_t distance = static_cast<size_t>(pf);
    if (distance >= remaining) return;
    capture_prefetch(reinterpret_cast<const char*>(blk + distance), _MM_HINT_T0);
    if (remaining - distance > 64)
        capture_prefetch(reinterpret_cast<const char*>(blk + distance + 64), _MM_HINT_T0);
}

int main(int argc, char** argv) {
    if (argc != 2) return 9;
    const int expected = std::atoi(argv[1]);
    if (prefetch_distance() != expected || prefetch_distance() != expected) return 10;
    const size_t lengths[] = {0,1,2,63,64,65,66,127,128,129,360,720,2048,2112,4096,65536};
    const int distances[] = {INT_MIN,-2048,-1,0,1,2,63,64,65,512,1024,2048,INT_MAX,expected};
    uint64_t cases = 0;
    for (size_t length : lengths) {
        // Prefix/suffix sentinels ensure the oracle checks the caller's logical
        // matrix/assigned range, not just the larger allocation's address range.
        std::vector<uint8_t> storage(length + 96, 0xa5);
        const uint8_t* begin = storage.data() + 32;
        const uint8_t* end = begin + length;
        base_address = reinterpret_cast<uintptr_t>(begin);
        end_address = reinterpret_cast<uintptr_t>(end);
        std::vector<size_t> offsets;
        for (size_t i=0; i<=std::min<size_t>(length,256); ++i) offsets.push_back(i);
        offsets.push_back(length / 2); offsets.push_back(length);
        if (length) offsets.push_back(length-1);
        if (length > 1) offsets.push_back(length-2);
        std::sort(offsets.begin(), offsets.end());
        offsets.erase(std::unique(offsets.begin(),offsets.end()),offsets.end());
        for (size_t offset : offsets) for (int pf : distances) {
            captured.clear();
            rows_ahead(begin+offset,end,pf);
            std::vector<uintptr_t> wanted;
            // Independent integer-address oracle: enumerate the original two
            // hint targets and retain exactly the targets in the logical range.
            if (pf > 0) for (uint64_t extra : {uint64_t(0), uint64_t(64)}) {
                const uint64_t target = uint64_t(offset) + uint64_t(pf) + extra;
                if (target < length) wanted.push_back(base_address + target);
            }
            if (captured != wanted) return 11;
            ++cases;
        }
        for (uint8_t v : storage) if (v != 0xa5) return 12;
    }
    std::printf("{\"cases\":%llu,\"bounded_reads\":%llu,\"parsed_distance\":%d}\n",
                (unsigned long long)cases,(unsigned long long)reads,expected);
}
