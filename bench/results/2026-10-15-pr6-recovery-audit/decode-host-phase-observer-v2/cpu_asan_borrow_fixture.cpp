// SPDX-FileCopyrightText: 2026 yayoi
// SPDX-License-Identifier: LGPL-3.0-or-later
// PROPOSAL ONLY: independently compile/run with ASan+UBSan after both Direct suites close.
// No environment mutation, SYCL runtime, GPU, server or pool dependency.
#include "strata/core/decode_host_phases.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
using O=strata::core::DecodeHostPhaseObserver;
using W=strata::core::DecodeHostWindow;
using B=strata::core::DecodeHostPhaseBorrow;
namespace {
uint64_t clock_ns=0;
bool fake_clock(uint64_t& ns) noexcept { ns=++clock_ns; return true; }
void normal_window(O* slot) {
    assert(slot!=nullptr);
    W window(slot,1);
    window.phase(O::Capture); window.phase(O::CaptureCommit);
    window.phase(O::HostStaging); window.phase(O::InitialGraphSubmit);
    window.phase(O::Other);
    window.phase(O::LayerEventWait,0); window.phase(O::Other);
    window.phase(O::PoolCallback,0); window.phase(O::Other);
    window.phase(O::CopyStreamSync,0); window.phase(O::Other);
    window.phase(O::NextGraphSubmit,0); window.phase(O::Other);
    window.phase(O::TailComputeSync); window.phase(O::Other);
    window.phase(O::TailCopySync); window.phase(O::Other);
    window.phase(O::FinalReadout); window.finish(true);
}
void later_fresh_request(O*& slot) {
    assert(slot==nullptr);
    O fresh(1,fake_clock);
    {
        B borrow(slot,&fresh);
        assert(slot==&fresh && fresh.attached && !fresh.invalid);
        normal_window(slot);
    }
    assert(slot==nullptr && !fresh.attached && fresh.structurally_complete(1,1));
}
void stale_pointer_is_not_dereferenced() {
    O* slot=nullptr;
    {
        auto dead=std::make_unique<O>(1,fake_clock);
        slot=dead.get();
    } // slot now holds a freed heap address. V1 slot->invalid is an ASan UAF here.
    O refused(1,fake_clock);
    {
        B borrow(slot,&refused);
        assert(slot==nullptr && refused.invalid && !refused.attached && !refused.bound_once);
    }
    later_fresh_request(slot);
}
void live_collision_diagnoses_only_owned_objects() {
    O prior(1,fake_clock),refused(1,fake_clock);
    O* slot=nullptr;
    {
        B first(slot,&prior);
        {
            B collision(slot,&refused);
            assert(slot==nullptr && refused.invalid && !refused.attached);
            assert(prior.attached && !prior.invalid); // fresh borrow must not touch slot pointee
        }
        assert(prior.attached && !prior.invalid);
        first.detach(); // first knows prior's lifetime; now it can diagnose its own lost slot
        assert(prior.invalid && !prior.attached);
    }
    assert(slot==nullptr);
    later_fresh_request(slot);
}
struct OriginalError : std::runtime_error {
    const int marker;
    OriginalError() : std::runtime_error("original inference error"),marker(731) {}
};
void throwing_request(O*& slot,O& owner) {
    B borrow(slot,&owner); W window(slot,1); window.phase(O::Capture);
    throw OriginalError();
}
int returned_error_request(O*& slot,O& owner) {
    B borrow(slot,&owner); W window(slot,1); window.phase(O::Capture);
    return 73; // diagnostic destructors must preserve the original return value
}
void original_errors_and_later_request() {
    O* slot=nullptr;
    O failed(1,fake_clock);
    bool caught=false;
    try { throwing_request(slot,failed); }
    catch (const OriginalError& e) {
        caught=true;
        assert(e.marker==731 && std::strcmp(e.what(),"original inference error")==0);
    }
    assert(caught && slot==nullptr && failed.failed==1 && !failed.in_window && !failed.attached);
    assert(!failed.structurally_complete(1,0));
    later_fresh_request(slot);
    O returned(1,fake_clock);
    assert(returned_error_request(slot,returned)==73);
    assert(slot==nullptr && returned.failed==1 && !returned.in_window && !returned.attached);
    later_fresh_request(slot);
}
}
int main() {
    stale_pointer_is_not_dereferenced();
    live_collision_diagnoses_only_owned_objects();
    original_errors_and_later_request();
    std::puts("decode_host_phases independent ASan borrow contract PASS");
}
