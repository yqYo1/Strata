// Proposal-only CPU fixture: compile against the proposed header after Main applies it.
#include "strata/core/decode_route_ledger.hpp"
#include <cassert>
int main() {
    strata::core::DecodeRouteLedger x;
    x.layers=2; x.workers=5; x.service.worker_limit=5;
    // Tests aggregation, not NativeFmt kernel geometry or pool publication.
    x.fmt[0].gu_type=21; x.fmt[0].d_type=21; x.fmt[1]=x.fmt[0];
    x.dispatch_attempts=1;
    x.format_service_begin(0,x.fmt[0]);
    assert(x.service_pending && !x.format_service_complete());
    x.service.slots[0].call[1].record(0,10,0,20,true);
    x.service.slots[1].call[0].record(0,30,0,40,true);
    x.service.slots[5].call[2].record(0,50,0,60,true);
    x.format_service_end();
    assert(x.format_service_complete() && x.format_count==1);
    assert(x.format_service[0].group[1][0].cpu_ns==30);
    ++x.dispatch_attempts;
    x.format_service_begin(1,x.fmt[1]); x.format_service_end();
    assert(x.format_service_complete() && x.format_count==1); // zero-work dispatch
    ++x.service.slots[1].call[0].cpu_ns;
    assert(!x.format_service_complete()); // unassigned service rejected
    --x.service.slots[1].call[0].cpu_ns;
    ++x.dispatch_attempts; x.fmt[1].d_type=22;
    x.format_service_begin(1,x.fmt[1]);
    x.service.slots[1].call[2].record(0,7,0,9,true);
    x.format_service_end();
    assert(x.format_service_complete() && x.format_count==2);
    x.format_service[1].group[1][2].cpu_ns=UINT64_MAX;
    assert(!x.format_service_complete());
    x.format_overflow=true;
    assert(!x.format_service_complete());
}
