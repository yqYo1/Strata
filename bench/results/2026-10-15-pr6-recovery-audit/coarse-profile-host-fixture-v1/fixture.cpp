// Main's host-only fault/lifetime/count/output fixture; no runtime or GPU.
#include "cuda_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>
struct MockEvent { bool pending=false,recorded=false; unsigned stamp=0; };
static std::vector<MockEvent*> live;
static std::string mode;
static unsigned created=0,destroyed=0,records=0,elapsed_calls=0,queries=0,max_live=0,tick=0;
static unsigned create_attempts=0,create_fault=0,record_fault=0;
static bool injected=false;
static void need(bool condition,const char* why) {
    if(!condition){std::fprintf(stderr,"HOST_FIXTURE_FAIL %s\n",why);std::abort();}
}
cudaError_t cudaEventCreate(cudaEvent_t* out) {
    ++create_attempts;*out=nullptr;
    if(create_attempts==create_fault){injected=true;return cudaErrorUnknown;}
    *out=new MockEvent;live.push_back(*out);++created;
    max_live=std::max(max_live,unsigned(live.size()));return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t event) {
    need(event && !event->pending,"event destroyed before simulated queue retirement");
    auto it=std::find(live.begin(),live.end(),event);need(it!=live.end(),"foreign or duplicate event destroy");
    live.erase(it);delete event;++destroyed;
    if(mode=="destroy-error" && !injected){injected=true;return cudaErrorUnknown;}
    return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t event,cudaStream_t) {
    need(event && !event->pending,"event reused before simulated queue retirement");
    event->pending=event->recorded=true;event->stamp=++tick;++records;
    // An uncertain submit can leave work pending: caller must still drain.
    if(records==record_fault){injected=true;return cudaErrorUnknown;}
    return cudaSuccess;
}
cudaError_t cudaEventQuery(cudaEvent_t event) {
    need(event && event->recorded,"query of unrecorded event");
    need(!event->pending,"query before existing simulated drain");++queries;
    if(mode=="query-error" && !injected){injected=true;return cudaErrorUnknown;}
    return cudaSuccess;
}
cudaError_t cudaEventElapsedTime(float* ms,cudaEvent_t a,cudaEvent_t b) {
    need(a && b && a->recorded && b->recorded && !a->pending && !b->pending,
         "elapsed-time call introduces pending-event wait");
    need(b->stamp>=a->stamp,"reversed mock event order");++elapsed_calls;
    if(mode=="elapsed-error" && !injected){injected=true;return cudaErrorUnknown;}
    *ms=float(b->stamp-a->stamp);
    if(mode=="nan" && !injected){injected=true;*ms=std::numeric_limits<float>::quiet_NaN();}
    if(mode=="negative" && !injected){injected=true;*ms=-1;}
    return cudaSuccess;
}
static void existing_drain() { for(auto* e:live)e->pending=false; }
#include "strata/prefill/layermajor_coarse_profile.hpp"
using Profile=strata::prefill::detail::LayerMajorCoarseProfile;
int main(int argc,char** argv) {
    need(argc==2,"one declared host arm");mode=argv[1];
    const bool normal=mode=="normal32";
    if(mode=="create-1")create_fault=1;
    else if(mode=="create-7")create_fault=7;
    else if(mode=="create-12")create_fault=12;
    else if(mode=="record-1")record_fault=1;
    else if(mode=="record-2")record_fault=2;
    else need(normal || mode=="query-error" || mode=="elapsed-error" || mode=="nan" ||
              mode=="negative" || mode=="double-begin" || mode=="early-return" ||
              mode=="destroy-error","known host arm");
    {
        Profile profile;profile.open();
        for(int layer=0;layer<48 && profile.usable;++layer) {
            profile.layer=layer;profile.begin(Profile::upload,nullptr);
            if(mode=="double-begin"){
                profile.begin(Profile::upload,nullptr);injected=true;
            }
            profile.end(Profile::upload,nullptr);existing_drain();profile.retired(Profile::upload);
            for(int chunk=0;chunk<32 && profile.usable;++chunk) {
                const bool device=chunk<4;
                if(layer) {
                    profile.rows[layer].input_bytes[int(device)]+=8192ULL*10240*4;
                    profile.begin(Profile::input,nullptr,device);profile.end(Profile::input,nullptr);
                }
                const auto attention=layer%4==3?Profile::qsa:Profile::gdn;
                profile.begin(attention,nullptr);profile.end(attention,nullptr);
                profile.begin(Profile::moe,nullptr);profile.end(Profile::moe,nullptr);
                existing_drain();profile.chunk_retired();
                if(layer!=47) {
                    profile.rows[layer].store_bytes[int(device)]+=8192ULL*10240*4;
                    profile.begin(Profile::store,nullptr,device);profile.end(Profile::store,nullptr);
                    existing_drain();profile.retired(Profile::store);
                }
            }
            if(mode=="early-return")break;
        }
        // Mirrors only the caller contract; actual wrapper lease ordering is
        // separately reviewed and must still be qualified on the real backend.
        existing_drain();
        if(normal) {
            need(profile.usable && !profile.errors,"normal observer status");
            for(int layer=0;layer<48;++layer) {
                const auto& row=profile.rows[layer];
                need(row.samples[Profile::upload]==1 && row.samples[Profile::moe]==32,"bounded upload/MoE counts");
                need(row.samples[Profile::input]==unsigned(layer?32:0) &&
                     row.samples[Profile::store]==unsigned(layer==47?0:32),"residual layer boundary counts");
                need(row.samples[Profile::qsa]==unsigned(layer%4==3?32:0) &&
                     row.samples[Profile::gdn]==unsigned(layer%4==3?0:32),"all QSA/GDN layer counts");
            }
            need(records==12256 && elapsed_calls==6128 && queries==12256,"physical-length marker accounting");
        } else if(mode!="early-return" && mode!="destroy-error") {
            need(injected && !profile.usable && profile.errors,"injected observer error is incomplete");
        }
        profile.complete=normal || mode=="destroy-error";profile.wrapper_ms=profile.complete?1000:-1;
        profile.close();
        if(mode=="destroy-error")need(injected && !profile.usable && profile.errors,"destroy error is incomplete");
        profile.emit();
    }
    need(live.empty() && created==destroyed && max_live<=12,"all bounded event owners retired exactly once");
    std::printf("HOST_ONLY mode=%s created=%u destroyed=%u max_live=%u records=%u; no GPU/model qualification\n",
                mode.c_str(),created,destroyed,max_live,records);
}
