// Proposal only. Main compiles/runs after applying source.diff; no device/kernel work.
#include "strata/core/decode_route_ledger.hpp"
#include <cassert>
#include <memory>
#include <string>
using strata::core::DecodeRouteLedger;
using strata::core::DecodeLedgerOutput;
using strata::kernels::cpu::NativeFmt;
NativeFmt format(int type=21) {
    NativeFmt f{}; f.gu_type=f.d_type=type; f.gu_act=f.d_act=8;
    f.n_embd=2560; f.n_ff=640; f.gu_row=f.d_row=8;
    f.up_off=5120; f.down_off=10240; f.bytes=30720;
    f.act_bytes=f.h_bytes=8; return f;
}
std::unique_ptr<DecodeRouteLedger> fresh() {
    auto x=std::make_unique<DecodeRouteLedger>();
    x->layers=2; x->k=2; x->experts=16; x->workers=5; x->service.worker_limit=5;
    x->fmt[0]=x->fmt[1]=format(); return x;
}
void dispatch(DecodeRouteLedger& x,int l,bool work) {
    ++x.dispatch_attempts;
    x.format_service_begin(l,x.fmt[l]);
    if (work) {
        x.service.slots[0].call[1].record(0,10,0,20,true);
        x.service.slots[1].call[0].record(0,30,0,40,true);
        x.service.slots[5].call[2].record(0,50,0,60,true);
    }
    x.format_service_end();
}
void negative_receipt(DecodeRouteLedger& x,bool expect_unbounded=false) {
    assert(!x.format_service_complete());
    auto bytes=std::make_unique<char[]>(DecodeLedgerOutput::kCap);
    DecodeLedgerOutput output{bytes.get(),DecodeLedgerOutput::kCap};
    x.receipt(output,"error",0,0,0,0);
    assert(output.used<=DecodeLedgerOutput::kCap);
    const std::string text(bytes.get(),output.used);
    assert(text.find("service_format=1")!=std::string::npos);
    assert(text.find("status=error complete=0 coverage_complete=0")!=std::string::npos);
    assert(text.find("service_format_close complete=0")!=std::string::npos);
    if (expect_unbounded) assert(!output.bounded);
}
int main() {
    auto x=fresh(); dispatch(*x,0,true); assert(x->format_service_complete());
    dispatch(*x,1,false); assert(x->format_service_complete() && x->format_count==1);
    x->fmt[1]=format(20); dispatch(*x,1,true);
    assert(x->format_service_complete() && x->format_count==2);
    assert(x->format_service[0].group[1][0].cpu_ns==30);
    ++x->service.slots[1].call[0].cpu_ns; negative_receipt(*x);
    --x->service.slots[1].call[0].cpu_ns;
    assert(x->format_service_complete());
    ++x->dispatch_attempts; negative_receipt(*x); --x->dispatch_attempts;
    x->format_service_begin(0,x->fmt[0]); negative_receipt(*x);
    x->format_service_end(); // now extra delta dispatch must also be rejected
    negative_receipt(*x);
    x=fresh(); dispatch(*x,0,true); x->format_overflow=true; negative_receipt(*x);
    x=fresh(); dispatch(*x,0,true);
    x->format_service[0].group[1][0].cpu_ns=UINT64_MAX;
    x->fmt[1]=format(20); dispatch(*x,1,true); negative_receipt(*x); // summed overflow
    x=fresh(); dispatch(*x,0,true);
    x->format_service[0].group[1][0].cpu_ns=UINT64_MAX;
    dispatch(*x,0,true); assert(x->format_overflow); negative_receipt(*x);
    for (int workers:{0,129}) {
        x=fresh(); dispatch(*x,0,true); x->workers=workers;
        negative_receipt(*x,true);
        x->format_service_begin(0,x->fmt[0]); assert(x->format_invalid);
    }
    x=fresh(); x->layers=65;
    x->format_service_begin(64,format()); assert(x->format_invalid);
    negative_receipt(*x,true);
    for (int representative:{-1,64}) {
        x=fresh(); dispatch(*x,0,true); x->format_service[0].layer=representative;
        negative_receipt(*x,true);
        x->format_service_begin(0,x->fmt[0]); assert(x->format_invalid);
    }
    for (int count:{-1,17}) {
        x=fresh(); dispatch(*x,0,true); x->format_count=count;
        negative_receipt(*x,true);
        x->format_service_begin(0,x->fmt[0]); assert(x->format_invalid);
    }
    x=fresh(); dispatch(*x,0,true); x->fmt[0].gu_row=0;
    negative_receipt(*x,true);
    x=fresh(); x->layers=0; x->format_service_begin(0,format());
    assert(x->format_invalid); negative_receipt(*x,true);
    // 16 groups fit; a seventeenth distinct actual metadata group is rejected.
    x=fresh(); x->layers=17;
    for (int l=0;l<17;++l) x->fmt[l]=format(l);
    for (int l=0;l<16;++l) dispatch(*x,l,false);
    assert(x->format_count==16 && x->format_service_complete());
    dispatch(*x,16,false); assert(x->format_invalid); negative_receipt(*x);
}
