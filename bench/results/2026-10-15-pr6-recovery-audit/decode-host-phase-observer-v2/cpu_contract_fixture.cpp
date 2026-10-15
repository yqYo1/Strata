// SPDX-FileCopyrightText: 2026 yayoi
// SPDX-License-Identifier: LGPL-3.0-or-later
// PROPOSAL ONLY. Not compiled or executed by the implementation agent.
// Main after both Direct arms close, serially:
// c++ -std=c++17 -Wall -Wextra -Werror -Iinclude PRIVATE/cpu_contract_fixture.cpp -o PRIVATE/cpu_contract
#include "strata/core/decode_host_phases.hpp"
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <stdexcept>
using O=strata::core::DecodeHostPhaseObserver;
using W=strata::core::DecodeHostWindow;
using B=strata::core::DecodeHostPhaseBorrow;
namespace {
uint64_t time_ns=0,reads=0;
bool clock_ok=true;
bool fake_clock(uint64_t& n) noexcept { ++reads; n=time_ns; return clock_ok; }
void reset() { time_ns=reads=0; clock_ok=true; }
std::string receipt(const O& o,bool joined=true,uint64_t n=1) {
    auto bytes=std::make_unique<char[]>(O::OutputCap+16);
    std::memset(bytes.get()+O::OutputCap,0x5a,16);
    const size_t used=o.receipt(bytes.get(),7,"length",joined,n,n);
    assert(used<=O::OutputCap);
    for (unsigned i=0;i<16;++i) assert(bytes[O::OutputCap+i]==0x5a);
    return std::string(bytes.get(),used);
}
void phase(W& w,O::Phase p,uint64_t n,int layer=-1) { time_ns=n; w.phase(p,layer); }
// Independent explicit timeline: parent 200 ns, exclusive other 80 ns,
// finalreadout 20 ns, ten named leaf intervals of 10 ns apiece.
void one_window(O* slot,int T=1) {
    const uint64_t start=time_ns;
    W w(slot,T);
    phase(w,O::Capture,start+10);
    phase(w,O::CaptureCommit,start+20);
    phase(w,O::HostStaging,start+30);
    phase(w,O::InitialGraphSubmit,start+40);
    phase(w,O::Other,start+50);
    phase(w,O::LayerEventWait,start+60,0);
    phase(w,O::Other,start+70);
    phase(w,O::PoolCallback,start+80,0);
    phase(w,O::Other,start+90);
    phase(w,O::CopyStreamSync,start+100,0);
    phase(w,O::Other,start+110);
    phase(w,O::NextGraphSubmit,start+120,0);
    phase(w,O::Other,start+130);
    phase(w,O::TailComputeSync,start+140);
    phase(w,O::Other,start+150);
    phase(w,O::TailCopySync,start+160);
    phase(w,O::Other,start+170);
    phase(w,O::FinalReadout,start+180);
    time_ns=start+200; w.finish(true);
}
void off_and_success() {
    reset(); one_window(nullptr); assert(reads==0);
    reset(); O o(1,fake_clock); O* slot=nullptr;
    {
        B borrow(slot,&o); assert(slot==&o && o.attached);
        one_window(slot);
        assert(o.global[O::Other].value.sum==80);
        assert(o.global[O::FinalReadout].value.sum==20);
        assert(o.global[O::RunTotal].value.sum==200);
        assert(!o.structurally_complete(1,1)); // attached close must refuse
    }
    assert(slot==nullptr && !o.attached && o.structurally_complete(1,1));
    const auto text=receipt(o);
    assert(text.find("close request=7 complete=1 ")!=std::string::npos);
    assert(text.find("p50_lo_ns=4 p50_hi_ns=15")!=std::string::npos);
    assert(receipt(o,false).find("close request=7 complete=0 ")!=std::string::npos);
    assert(receipt(o,true,2).find("close request=7 complete=0 ")!=std::string::npos);
}
void request_borrow_and_unwind() {
    reset(); O* slot=nullptr;
    {
        O o(1,fake_clock);
        bool caught=false;
        try {
            B borrow(slot,&o);
            W w(slot,1); time_ns=10; w.phase(O::Capture);
            throw std::runtime_error("original request failure");
        } catch (const std::runtime_error& e) {
            caught=true; assert(std::strcmp(e.what(),"original request failure")==0);
        }
        assert(caught);
        assert(slot==nullptr && !o.attached && !o.in_window && o.failed==1);
        assert(receipt(o).find("close request=7 complete=0 ")!=std::string::npos);
    }
    // A later request has fresh counters and cannot observe the preceding owner.
    O next(1,fake_clock);
    { B borrow(slot,&next); one_window(slot); borrow.detach(); borrow.detach(); }
    assert(slot==nullptr && next.structurally_complete(1,1));
    // RESTORE/prompt/capture outside the borrow has no observer pointer.
    const uint64_t before=reads; one_window(slot); assert(reads==before);
    O old(1,fake_clock),fresh(1,fake_clock);
    {
        B first(slot,&old);
        { B collision(slot,&fresh); assert(slot==nullptr && !old.invalid && fresh.invalid); }
        // Only first's known owner is diagnosed when first detaches.
        assert(old.attached && !old.invalid); first.detach(); assert(old.invalid);
    }
    assert(slot==nullptr && !old.attached && !fresh.attached);
}
void clock_and_overflow_failures() {
    reset(); O o(1,fake_clock); O* slot=nullptr;
    { B borrow(slot,&o); time_ns=100; W w(slot,1); time_ns=99; w.phase(O::Capture); }
    assert(o.invalid && !o.structurally_complete(1,1));
    reset(); O f(1,fake_clock);
    { B borrow(slot,&f); clock_ok=false; W w(slot,1); }
    assert(f.invalid && f.failed==1);
    reset(); O over(1,fake_clock);
    over.global[O::Other].value.sum=UINT64_MAX;
    over.record(over.global[O::Other],1); assert(over.overflow);
    assert(receipt(over).find("complete=0 ")!=std::string::npos);
    O invalid_t(1,fake_clock);
    { B borrow(slot,&invalid_t); const auto n=reads; W w(slot,5); assert(reads==n); }
    assert(invalid_t.invalid);
    O invalid_g(1,fake_clock);
    { B borrow(slot,&invalid_g); W w(slot,1); w.envelope(false); }
    assert(invalid_g.invalid);
}
void fixed_marginals_and_bins() {
    reset(); O o(1,fake_clock); O* slot=nullptr;
    { B borrow(slot,&o); for (int n=0;n<65;++n) one_window(slot,1+n%4); }
    assert(o.structurally_complete(65,65));
    assert(o.window_band[0].value.count==1 && o.window_band[1].value.count==7);
    assert(o.window_band[2].value.count==56 && o.window_band[3].value.count==1);
    assert(o.by_t[0][O::RunTotal].value.count==17);
    assert(o.by_t[1][O::RunTotal].value.count==16);
    assert(o.layer[0][0].count==65);
    for (unsigned b=0;b<O::Bins;++b) {
        assert(O::bin(O::low(b))==b && O::bin(O::high(b))==b);
    }
    O::Distribution d;
    // Distinct raw samples: no exact median is claimed; values 4 and 15 share a bin.
    o.record(d,4); o.record(d,15); o.record(d,16); o.record(d,63);
    assert(O::quantile_bin(d,50)==2 && O::quantile_bin(d,95)==3);
    ++o.by_t[0][O::RunTotal].hist[0]; // corrupt one marginal must reject completeness
    assert(!o.structurally_complete(65,65));
}
std::unique_ptr<O> valid_one_layer_65() {
    reset(); auto o=std::make_unique<O>(1,fake_clock); O* slot=nullptr;
    { B borrow(slot,o.get()); for (int n=0;n<65;++n) one_window(slot,1+n%4); }
    assert(o->structurally_complete(65,65)); return o;
}
std::unique_ptr<O> valid_two_layers() {
    reset(); auto o=std::make_unique<O>(2,fake_clock); O* slot=nullptr;
    {
        B borrow(slot,o.get()); W w(slot,1);
        phase(w,O::Capture,10); phase(w,O::CaptureCommit,20); phase(w,O::HostStaging,30);
        phase(w,O::InitialGraphSubmit,40); phase(w,O::Other,50);
        for (int l=0;l<2;++l) {
            const uint64_t offset=uint64_t(l)*80;
            phase(w,O::LayerEventWait,60+offset,l); phase(w,O::Other,70+offset);
            phase(w,O::PoolCallback,80+offset,l); phase(w,O::Other,90+offset);
            phase(w,O::CopyStreamSync,100+offset,l); phase(w,O::Other,110+offset);
            phase(w,O::NextGraphSubmit,120+offset,l); phase(w,O::Other,130+offset);
        }
        phase(w,O::TailComputeSync,220); phase(w,O::Other,230);
        phase(w,O::TailCopySync,240); phase(w,O::Other,250);
        phase(w,O::FinalReadout,260); time_ns=280; w.finish(true);
    }
    assert(o->structurally_complete(1,1)); return o;
}
void reject_corruption(const O& o,uint64_t n) {
    assert(!o.structurally_complete(n,n));
    assert(receipt(o,true,n).find("close request=7 complete=0 ")!=std::string::npos);
}
void band_count_corruption() {
    const auto base=valid_one_layer_65(); auto o=std::make_unique<O>(*base);
    ++o->window_band[0].value.count; reject_corruption(*o,65);
}
void band_sum_corruption() {
    const auto base=valid_one_layer_65(); auto o=std::make_unique<O>(*base);
    ++o->window_band[0].value.sum; reject_corruption(*o,65);
}
void band_hist_corruption() {
    const auto base=valid_one_layer_65(); auto o=std::make_unique<O>(*base);
    // Keep local histogram count constant, move one sample to the wrong bin.
    const unsigned actual=O::bin(200);
    --o->window_band[0].hist[actual]; ++o->window_band[0].hist[0];
    reject_corruption(*o,65);
}
void band_ordinal_corruption() {
    const auto base=valid_one_layer_65(); auto o=std::make_unique<O>(*base);
    // Global count/sum/hist are unchanged by this swap; ordinal counts must reject it.
    const auto saved=o->window_band[0];
    o->window_band[0]=o->window_band[1]; o->window_band[1]=saved;
    reject_corruption(*o,65);
}
void band_checked_arithmetic() {
    const auto base=valid_one_layer_65(); auto o=std::make_unique<O>(*base);
    o->window_band[0].value.sum=UINT64_MAX; o->window_band[1].value.sum=1;
    reject_corruption(*o,65);
    o=std::make_unique<O>(*base);
    o->window_band[0].hist[0]=UINT64_MAX;
    o->window_band[0].hist[1]=1; // individual histogram count overflow
    reject_corruption(*o,65);
}
void layer_sum_corruption() {
    const auto base=valid_two_layers();
    for (unsigned p=0;p<O::LayerPhases;++p) {
        auto o=std::make_unique<O>(*base);
        ++o->layer[1][p].sum; reject_corruption(*o,1);
    }
}
void layer_count_corruption() {
    const auto base=valid_two_layers();
    for (unsigned p=0;p<O::LayerPhases;++p) {
        auto o=std::make_unique<O>(*base);
        ++o->layer[1][p].count; reject_corruption(*o,1);
    }
}
void layer_counts_vs_global_corruption() {
    const auto base=valid_two_layers();
    for (int p=O::LayerEventWait;p<=O::NextGraphSubmit;++p) {
        auto o=std::make_unique<O>(*base);
        // Keep global/T count and histogram conservation valid. Layer total
        // stays 2, while the corresponding global phase claims 3 samples.
        ++o->global[p].value.count; ++o->by_t[0][p].value.count;
        ++o->global[p].hist[0]; ++o->by_t[0][p].hist[0];
        reject_corruption(*o,1);
    }
}
void layer_checked_arithmetic() {
    const auto base=valid_two_layers();
    for (unsigned p=0;p<O::LayerPhases;++p) {
        auto o=std::make_unique<O>(*base);
        o->layer[0][p].sum=UINT64_MAX; o->layer[1][p].sum=1;
        reject_corruption(*o,1);
    }
}

void output_budget() {
    O o(64,fake_clock);
    auto fill=[](O::Distribution& d) {
        d.value={UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX};
        for (auto& n:d.hist) n=UINT64_MAX; // worst decimal width, intentionally invalid totals
    };
    for (auto& d:o.global) fill(d);
    for (auto& row:o.by_t) for (auto& d:row) fill(d);
    for (auto& d:o.window_band) fill(d);
    for (auto& row:o.layer) for (auto& m:row) m={UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX};
    const auto text=receipt(o);
    assert(text.size()<O::OutputCap-O::FooterReserve);
    assert(text.find("close request=7 complete=0 ")!=std::string::npos);
    assert(text.find("output_bounded=1 ")!=std::string::npos);
    auto buf=std::make_unique<char[]>(O::OutputCap);
    O::Output out{buf.get()};
    out.used=O::OutputCap-O::FooterReserve-1;
    assert(!out.append(false,"longer than remaining room"));
    assert(!out.bounded && out.append(true,"complete=0 reason=output_budget\n"));
}
}
int main() {
    off_and_success(); request_borrow_and_unwind(); clock_and_overflow_failures();
    fixed_marginals_and_bins();
    band_count_corruption(); band_sum_corruption(); band_hist_corruption();
    band_ordinal_corruption(); band_checked_arithmetic();
    layer_sum_corruption(); layer_count_corruption(); layer_counts_vs_global_corruption();
    layer_checked_arithmetic(); output_budget();
    std::puts("decode_host_phases CPU contract PASS");
}
