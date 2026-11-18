// Main-authored; repository LICENSE. Independent byte-copy and failure oracle.
#include "strata/prefill/fullfit_upload_plan.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

using namespace strata::prefill::detail;
static void check(bool value) { if (!value) throw std::runtime_error("full-fit upload oracle failed"); }
template<class F> static void refuses(F&& operation) {
    try { operation(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid full-fit upload accepted");
}
static void bytes(const std::vector<int32_t>& ids,uint64_t source_stride,size_t calls) {
    const auto bank=ExpertBankLayout::packed(512*256,256);
    const auto plan=PrefillFullfitUploadPlan::make(ids,bank,source_stride);
    std::vector<uint8_t> source(512*source_stride),destination(bank.backing_bytes,0xa5);
    for (int e=0;e<512;++e)
        for (size_t i=0;i<source_stride;++i) source[size_t(e)*source_stride+i]=uint8_t((e+i*5)%251);
    ExpertBankSlots slots(512,true);size_t observed=0;uint64_t copied=0;
    enqueue_prefill_fullfit_upload(plan,slots,[&](const PrefillFullfitCopy& run) {
        check(slots.permanent_used()==int32_t(run.first)); // not published before acceptance
        std::memcpy(destination.data()+run.destination_offset,source.data()+run.source_offset,run.bytes);
        ++observed;copied+=run.bytes;return true;
    });
    check(observed==calls && copied==ids.size()*256 && copied==plan.payload_bytes());
    for (size_t i=0;i<ids.size();++i) {
        const auto hit=slots.begin(ids[i]);check(!hit.upload && hit.permanent && hit.slot==int32_t(i));
        check(std::memcmp(destination.data()+i*256,source.data()+size_t(ids[i])*source_stride,256)==0);
    }
    check(std::all_of(destination.begin()+ids.size()*256,destination.end(),[](uint8_t v){return v==0xa5;}));
}
int main() try {
    const auto bank=ExpertBankLayout::packed(512*256,256);
    const std::vector<int32_t> ids{2,3,4,8,9,21};
    bytes(ids,256,6);bytes(ids,512,6);bytes({0,1,510,511},256,4);
    bytes({},256,0);bytes({511},256,1);
    std::vector<int32_t> all;for (int e=0;e<512;++e) all.push_back(e);
    bytes(all,256,512);
    for (const auto& invalid:std::vector<std::vector<int32_t>>{{-1},{512},{3,3},{3,2}})
        refuses([&]{PrefillFullfitUploadPlan::make(invalid,bank,256);});
    refuses([&]{PrefillFullfitUploadPlan::make(ids,bank,128);});
    refuses([&]{PrefillFullfitUploadPlan::make(ids,bank,std::numeric_limits<uint64_t>::max()-255);});
    refuses([&]{PrefillFullfitUploadPlan::make(ids,ExpertBankLayout::packed(511*256,256),256);});
    ExpertBankSlots partial(511);const auto plan=PrefillFullfitUploadPlan::make(ids,bank,256);
    refuses([&]{enqueue_prefill_fullfit_upload(plan,partial,[](auto){return true;});});
    ExpertBankSlots failed(512,true);unsigned attempts=0;
    refuses([&]{enqueue_prefill_fullfit_upload(plan,failed,[&](auto){return ++attempts!=3;});});
    check(attempts==3 && failed.permanent_used()==2);
    const auto unpublished=failed.begin(4);check(unpublished.upload && unpublished.slot==2);
    ExpertBankSlots nonempty(512,true);const auto first=nonempty.begin(1);nonempty.queued(first);
    refuses([&]{enqueue_prefill_fullfit_upload(plan,nonempty,[](auto){return true;});});
    const std::vector<int32_t> routed{511,2,42, 42,0,511, 1,512,2};
    const auto prefix = prefill_prefix_expert_ids(routed, 2, 3);
    check(prefix == std::vector<int32_t>({0,2,42,511}));
    bytes(prefix,256,4);
    refuses([&]{prefill_prefix_expert_ids(routed,3,3);});
    refuses([&]{prefill_prefix_expert_ids({0,-1},1,2);});
    refuses([&]{prefill_prefix_expert_ids({2,2},1,2);});
    refuses([&]{prefill_prefix_expert_ids(routed,4,3);});
    refuses([&]{prefill_prefix_expert_ids(routed,0,3);});
    refuses([&]{prefill_prefix_expert_ids(routed,1,0);});
    refuses([&]{prefill_prefix_expert_ids(routed,1,513);});
    std::puts("prefill full-fit upload oracle: byte_cases=6 refusals=10 failure_publication=checked");
    std::puts("prefill prefix expert IDs: sparse_and_repeated=checked refusals=7 checked_copy=pass");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr,"%s\n",error.what());return 1; }
