#include "layer_reuse_plan.hpp"
#include <cstdlib>
#include <iostream>
using namespace strata::prefill::reuse;
static void require(bool value, const char* what) {
    if (!value) { std::cerr << what << '\n'; std::exit(1); }
}
int main() {
    constexpr uint64_t MiB = 1024 * 1024, GiB = 1024 * MiB;
    Request r;
    r.tokens=65536; r.chunk=4096; r.residual_width=10240;
    r.experts=512; r.max_blob=2662400;
    r.free_device=1299*MiB; r.reserve_device=700*MiB;
    r.available_host=16*GiB; r.reserve_host=2*GiB;
    r.fresh=r.single_gpu=r.checkpoints_disabled=r.callbacks_final_rows_only=true;
    r.state_invalidation_contract=r.layer_runner_ready=true;
    Plan p; std::string reason;
    require(plan(r,p,reason), "bounded resident subset must fit projected budget");
    require(p.residual_bytes==2684354560ull && p.chunks==16, "64K FP32 geometry");
    require(p.resident_experts==235 && !p.full_layer && p.slot_bytes==625664000ull,
            "do not pretend 512 expert slots fit current free bytes");
    Slice s;
    require(residual_slice(p,r.tokens,r.residual_width,65530,6,s), "last suffix");
    require(s.offset+s.bytes==p.residual_bytes, "suffix exact boundary");
    require(!residual_slice(p,r.tokens,r.residual_width,65530,7,s), "reject overrun");
    r.tokens=262144;
    require(plan(r,p,reason) && p.residual_bytes==10737418240ull, "full physical context byte arithmetic");
    r.available_host=8*GiB;
    require(!plan(r,p,reason) && p.slot_bytes==0, "full context unchanged fallback on host capacity");
    r.available_host=16*GiB; r.tokens=UINT64_MAX;
    require(!plan(r,p,reason), "overflow before allocation");
    r.tokens=65536; r.checkpoints_disabled=false;
    require(!plan(r,p,reason), "no mixed-layer chunk checkpoints");
    r.checkpoints_disabled=true; r.layer_runner_ready=false;
    require(!plan(r,p,reason), "v1 scaffold must not advertise an implemented scheduler");
    Lease lease; uint64_t g=0, next=0;
    require(lease.begin(g), "first lease");
    require(!lease.close(g) && !lease.record_drain(g,true), "unrecorded / unjoined is unsafe");
    require(lease.producers_joined(g), "producer join");
    require(!lease.record_drain(g,false) && !lease.close(g) && !lease.may_release(),
            "drain failure quarantine: no free or fallback after mutation");
    require(lease.record_drain(g,true) && lease.close(g), "successful full drain permits release");
    require(lease.begin(next) && next!=g && !lease.record_drain(g,true), "stale proof rejected");
    require(lease.producers_joined(next) && lease.record_drain(next,true) && lease.close(next), "second lease");
    auto ids=select_experts({true,false,true,false,false},2);
    require(ids==std::vector<uint64_t>({1,3}), "bounded stable selection excludes ordinary cache");
    std::cout << "PASS host planner/lease protocol only; no model or GPU execution\n";
}
