// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host-only fixture. Main owns compilation/execution.
#include "strata/core/decode_route_ledger.hpp"
#include <cassert>
#include <cstdint>
#include <initializer_list>
int main() {
    using strata::core::decode_route_ledger_eligible;
    using strata::core::decode_route_ledger_dispatch_eligible;
    for (int capacity : {2, 4}) {
        assert(decode_route_ledger_eligible(true,0,1,false,0,0,capacity));
        assert(decode_route_ledger_dispatch_eligible(true,1,0));
        assert(!decode_route_ledger_dispatch_eligible(true,2,0));
    }
    assert(!decode_route_ledger_eligible(true,0,1,false,0,0,0));
    assert(!decode_route_ledger_eligible(true,0,1,true,0,0,4));
    assert(!decode_route_ledger_eligible(true,0,1,false,3,0,4));
    assert(!decode_route_ledger_eligible(true,0,2,false,0,0,4));
    using strata::core::decode_route_ledger_coverage;
    const auto normal=decode_route_ledger_coverage(48,1,48,true,false,false);
    assert(normal.expected_known && normal.expected_calls==48 && normal.complete);
    assert(decode_route_ledger_coverage(48,128,6144,true,false,false).complete);
    assert(!decode_route_ledger_coverage(48,1,47,true,false,false).complete);
    assert(!decode_route_ledger_coverage(48,1,49,true,false,false).complete);
    assert(!decode_route_ledger_coverage(48,1,48,false,false,false).complete);
    assert(!decode_route_ledger_coverage(48,1,48,true,true,false).complete);
    assert(!decode_route_ledger_coverage(48,1,48,true,false,true).complete);
    assert(!decode_route_ledger_coverage(48,INT64_MAX,0,true,false,false).expected_known);
    assert(!decode_route_ledger_coverage(65,1,65,true,false,false).expected_known);
    assert(!decode_route_ledger_coverage(48,-1,0,true,false,false).expected_known);
    strata::core::DecodeRouteLedger ledger;
    ledger.layers=1;
    auto& f=ledger.fmt[0]; f.gu_type=21; f.d_type=20; f.up_off=300; f.down_off=600; f.bytes=1000;
    strata::kernels::cpu::ExpertJobMulti jobs[2]{};
    jobs[0].nt=1; jobs[1].nt=2;
    int32_t ids[]={1,2,2,3,4,4}, kind[]={-1,-1,-1,0,1,1};
    ledger.record(0,ids,kind,6,jobs,2,f);
    assert(ledger.layer_calls==1 && ledger.calls_by_layer[0]==1 && !ledger.invalid);
    assert(ledger.rows[0][0][0].groups==1 && ledger.rows[0][0][1].entries==2);
    assert(ledger.rows[0][1][0].groups==1 && ledger.rows[0][2][1].groups==1);
    assert(ledger.rows[0][2][1].blob==1000 && ledger.rows[0][2][1].gu==600);
    assert(ledger.rows[0][2][1].down==400); // extent once/group, not once/entry
    ledger.record(0,ids,kind,6,jobs,2,f,true);
    assert(ledger.rows[0][3][1].groups==1 && ledger.rows[0][3][1].entries==2);
    uint64_t counter=UINT64_MAX-1; ledger.add(counter,2);
    assert(counter==UINT64_MAX && ledger.overflow);
    ledger.record(64,ids,kind,6,jobs,2,f); assert(ledger.invalid);
    ledger.invalid=false; ledger.layers=65;
    ledger.record(64,ids,kind,6,jobs,2,f); assert(ledger.invalid);
    ledger.layers=1; ledger.invalid=false;
    auto bad=f; bad.down_off=bad.bytes+1;
    ledger.record(0,ids,kind,6,jobs,2,bad); assert(ledger.invalid);
    ledger.invalid=false; bad=f; bad.up_off=bad.down_off;
    ledger.record(0,ids,kind,6,jobs,2,bad); assert(ledger.invalid);
    ledger.invalid=false;
    ledger.record(0,nullptr,kind,6,jobs,2,f); assert(ledger.invalid);
    ledger.invalid=false;
    ledger.record(0,ids,kind,6,nullptr,2,f); assert(ledger.invalid);
    ledger.invalid=false; jobs[0].nt=9;
    ledger.record(0,ids,kind,6,jobs,2,f); assert(ledger.invalid);
    ledger.invalid=false; kind[3]=2;
    ledger.record(0,ids,kind,6,jobs,0,f); assert(ledger.invalid);
    const auto calls=ledger.layer_calls; ledger.active=false;
    ledger.record(0,ids,kind,6,jobs,0,f); assert(ledger.layer_calls==calls);
}
