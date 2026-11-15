#include "strata/program/full_context_diagnostic.hpp"
#include "strata/program/diagnostic_kv_ranges.hpp"

#include <cstdio>
#include <initializer_list>
#include <limits>
#include <vector>

int main() {
    namespace d = strata::program::full_context_diagnostic;
    int cases = 0;
    auto check = [&](bool ok) { ++cases; return ok; };
    for (const auto context : {int64_t(-1), int64_t(0), int64_t(262143), int64_t(262144),
                               int64_t(262145), std::numeric_limits<int64_t>::max()}) {
        for (const size_t prompt : {size_t(0), size_t(262143), size_t(262144), size_t(262145),
                                    std::numeric_limits<size_t>::max()}) {
            for (const auto output : {int64_t(-1), int64_t(0), int64_t(1),
                                      std::numeric_limits<int64_t>::max()}) {
                const bool expected = context == 262144 && prompt == 262144 && output == 0;
                if (!check(d::request(context, prompt, output) == expected)) return 1;
            }
        }
    }
    if (!check(d::last_window(true, 262144, 262143, 1, 42, 42, 0))) return 1;
    if (!check(!d::last_window(false, 262144, 262143, 1, 42, 42, 0))) return 1;
    if (!check(!d::last_window(true, 262144, 262143, 1, 41, 42, 0))) return 1;
    if (!check(!d::last_window(true, 262144, 262143, 1, 42, 42, 1))) return 1;
    for (int64_t position : {int64_t(-1), int64_t(262142), int64_t(262144),
                             std::numeric_limits<int64_t>::max()})
        if (!check(!d::last_window(true, 262144, position, 1, 42, 42, 0))) return 1;
    for (int width : {-1, 0, 2, 4})
        if (!check(!d::last_window(true, 262144, 262143, width, 42, 42, 0))) return 1;
    if (!check(!d::last_window(true, 262143, 262143, 1, 42, 42, 0))) return 1;
    // First-window eligibility is captured before the loop clears its flag.
    bool first = true;
    const bool terminal = d::last_window(first, 262144, 262143, 1, 42, 42, 0);
    first = false;
    if (!check(terminal && !d::last_window(first, 262144, 262143, 1, 42, 42, 0))) return 1;
    const int bounds_cases = cases;
    namespace diag = strata::program::diagnostic;
    auto hash = [](const uint8_t* p, size_t n, uint64_t h) {
        for (size_t i=0;i<n;++i) { h ^= p[i]; h *= UINT64_C(1099511628211); }
        return h;
    };
    // Independent oracle: the original nested page/head/cell walk. Exercise
    // every interval, including two partial heads at either end and empty spans.
    for (int64_t ps : {int64_t(1),int64_t(4),int64_t(16)})
        for (int64_t heads : {int64_t(1),int64_t(2),int64_t(4)})
            for (int64_t width : {int64_t(1),int64_t(8),int64_t(128),int64_t(256),int64_t(512)}) {
                const int64_t cells=19;
                std::vector<uint8_t> bytes(size_t((cells+ps-1)/ps*heads*ps*width));
                for (size_t i=0;i<bytes.size();++i) bytes[i]=uint8_t(i*131+(i>>5)+17);
                for (int64_t begin=0;begin<=cells;++begin) for (int64_t end=begin;end<=cells;++end) {
                    uint64_t expected=UINT64_C(1469598103934665603),actual=expected;
                    size_t seen=0;
                    for (int64_t pg=begin/ps;pg*ps<end;++pg)
                        for (int64_t hd=0;hd<heads;++hd)
                            for (int64_t cell=0;cell<ps;++cell)
                                if (begin<=pg*ps+cell && pg*ps+cell<end) {
                                    const size_t offset=size_t(((pg*heads+hd)*ps+cell)*width);
                                    expected=hash(bytes.data()+offset,size_t(width),expected);
                                }
                    const bool ok=diag::kv_hash_spans(begin,end,ps,heads,width,[&](size_t offset,size_t n) {
                        if (offset>bytes.size() || n>bytes.size()-offset || !n) return false;
                        actual=hash(bytes.data()+offset,n,actual);seen+=n;return true;
                    });
                    if (!check(ok && actual==expected && seen==size_t((end-begin)*heads*width))) return 1;
                }
            }
    size_t calls=0;
    if (!check(diag::kv_hash_spans(0,262144,4,2,256,[&](size_t offset,size_t n) {
        ++calls;return offset==0 && n==size_t(262144)*2*256;
    }) && calls==1)) return 1;
    calls=0;
    if (!check(!diag::kv_hash_spans(1,19,4,2,256,[&](size_t,size_t) { return ++calls<2; }) && calls==2)) return 1;
    for (auto shape : {std::vector<int64_t>{-1,1,4,2,256},{2,1,4,2,256},{0,1,0,2,256},
                       {0,1,4,0,256},{0,1,4,2,0},
                       {0,std::numeric_limits<int64_t>::max(),4,2,256},
                       {0,1,4,std::numeric_limits<int64_t>::max(),256}}) {
        calls=0;
        if (!check(!diag::kv_hash_spans(shape[0],shape[1],shape[2],shape[3],shape[4],
            [&](size_t,size_t) { ++calls;return true; }) && calls==0)) return 1;
    }
    std::printf("cases=%d full_context_request_and_terminal_bounds=PASS\n", bounds_cases);
    std::printf("bounds_cases=%d kv_hash_cases=%d byte_order_and_bounds=PASS full_extent_spans=1 GPU=UNRUN\n",
                bounds_cases,cases-bounds_cases);
}
