// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "strata/kernels/cpu/pool.hpp"
#include <cstdint>
#include <mutex>

namespace strata::core {
// Raw verifier capacity is not the actual number of routed token rows.
inline bool decode_route_ledger_eligible(bool ordinary, int batch, int stages,
                                        bool mtp, int suffix, int lookup, int capacity) {
    return ordinary && batch == 0 && stages == 1 && !mtp && suffix == 0 &&
           lookup == 0 && capacity >= 2;
}
inline bool decode_route_ledger_dispatch_eligible(bool native, int64_t actual_nt,
                                                 int remote_count) {
    return native && actual_nt == 1 && remote_count == 0;
}

// Expected counts are a coverage check, never a replacement for observed counts.
struct DecodeRouteCoverage {
    uint64_t expected_calls = 0;
    bool expected_known = false, complete = false;
};
inline DecodeRouteCoverage decode_route_ledger_coverage(int64_t layers, int64_t emitted,
                                                        uint64_t observed, bool normal_close,
                                                        bool invalid, bool overflow) {
    DecodeRouteCoverage result;
    if (layers < 1 || layers > 64 || emitted < 0 ||
        (uint64_t) emitted > UINT64_MAX / (uint64_t) layers) return result;
    result.expected_calls = (uint64_t) emitted * (uint64_t) layers;
    result.expected_known = true;
    result.complete = normal_close && !invalid && !overflow && observed == result.expected_calls;
    return result;
}

// Fixed 64 x 4 x 8 buckets. Host-only writes after existing pool completion.
class DecodeRouteLedger {
public:
    struct Bucket { uint64_t groups=0, entries=0, gu=0, down=0, blob=0; };
    Bucket rows[64][4][8]{};
    strata::kernels::cpu::NativeFmt fmt[64]{};
    std::mutex mutex;
    int64_t layers=0;
    uint64_t layer_calls=0, calls_by_layer[64]{};
    bool active=true, invalid=false, overflow=false;
    void add(uint64_t& v, uint64_t n) {
        if (n > UINT64_MAX-v) { v=UINT64_MAX; overflow=true; } else v+=n;
    }
    void record(int64_t l, const int32_t* ids, const int32_t* kind, int64_t n,
                const strata::kernels::cpu::ExpertJobMulti* jobs, int count,
                const strata::kernels::cpu::NativeFmt& f, bool mapped = false) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!active) return;
        // Validate the fixed envelope and pointers before any array access.
        if (layers < 1 || layers > 64 || l < 0 || l >= 64 || l >= layers ||
            n < 1 || n > 32 || count < 0 || count > n ||
            !ids || !kind || (count > 0 && !jobs)) {
            invalid = true; return;
        }
        const auto& expected = fmt[l];
        if (f.gu_type < 0 || f.d_type < 0 || f.gu_type != expected.gu_type ||
            f.d_type != expected.d_type || f.bytes == 0 || f.up_off == 0 ||
            f.up_off >= f.down_off || f.down_off >= f.bytes ||
            f.up_off != expected.up_off || f.down_off != expected.down_off ||
            f.bytes != expected.bytes) {
            invalid = true; return;
        }
        add(layer_calls,1); add(calls_by_layer[l],1);
        auto group = [&](int owner,int nt) {
            if (owner<0 || owner>3 || nt<1 || nt>8) { invalid=true; return; }
            auto& b=rows[l][owner][nt-1];
            add(b.groups,1); add(b.entries,nt); add(b.gu,f.down_off);
            add(b.down,f.bytes-f.down_off); add(b.blob,f.bytes);
        };
        for (int j=0;j<count;++j) group(0,jobs[j].nt);
        for (int64_t i=0;i<n;++i) {
            if (kind[i]<0) continue;
            if (kind[i]>1) { invalid=true; continue; }
            bool first=true;
            for (int64_t j=0;j<i;++j) if (ids[j]==ids[i]) first=false;
            if (!first) continue;
            int nt=0;
            for (int64_t j=i;j<n;++j) if (ids[j]==ids[i]) {
                if (kind[j]!=kind[i]) invalid=true;
                ++nt;
            }
            group(kind[i]==0 ? 1 : mapped ? 3 : 2,nt);
        }
    }
};
static_assert(sizeof(DecodeRouteLedger)<=128*1024,"ledger storage budget");
}
