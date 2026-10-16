#include "strata/prefill/layer_major_kv_prefix_reuse.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace strata::prefill::detail;
static void check(bool ok,const char* message) {
    if(!ok) { std::cerr<<"FAIL: "<<message<<'\n'; std::exit(1); }
}
template<class F> static void rejects(F action,const char* message) {
    bool caught=false;
    try { action(); } catch(const std::exception&) { caught=true; }
    check(caught,message);
}
int main() {
    check(!layer_major_kv_reuse_requested(nullptr) && !layer_major_kv_reuse_requested("0") &&
          layer_major_kv_reuse_requested("1"),"strict flag defaults off");
    for(const char* bad:{"","01"," 1","1 ","true","-1","2"})
        rejects([&]{layer_major_kv_reuse_requested(bad);},"invalid flag rejected");
    for(const int64_t pos0:{int64_t(0),int64_t(1),int64_t(3),int64_t(4),int64_t(5)}) {
        LayerMajorKvPrefixReuse state(19,pos0,4,262144);
        uint64_t avoided=0;
        for(int64_t qsa=0;qsa<12;++qsa) {
            const int64_t layer=4*qsa+3;
            check(state.pass_matches(layer-1,layer),"GDN pass has no active QSA stage generation");
            state.begin_layer(layer,qsa);
            check(state.pass_matches(layer,layer+1) && !state.pass_matches(layer-1,layer),"active layer scope exact");
            rejects([&]{state.begin_layer(layer,qsa);},"active QSA layer cannot reenter");
            rejects([&]{state.finish_layer(layer);},"incomplete layer cannot publish completion");
            for(int64_t row=0;row<19;row+=4) {
                const int64_t count=std::min<int64_t>(4,19-row),p=pos0+row;
                rejects([&]{state.chunk(layer,qsa,p+1,count);},"nonmonotonic position rejected without advancing");
                rejects([&]{state.chunk(layer,qsa+1,p,count);},"different QSA generation rejected");
                rejects([&]{state.chunk(layer,qsa,p,count+1);},"wrong chunk/tail extent rejected");
                const auto decision=state.chunk(layer,qsa,p,count);
                check(decision.seed==(row==0),"seed exactly first chunk per QSA layer");
                check(decision.seed_blocks==(row==0?pos0/4+(pos0%4!=0):0),"first partial page restored once");
                if(row) avoided+=uint64_t(p/4+(p%4!=0));
                rejects([&]{state.chunk(layer,qsa,p,count);},"duplicate chunk cannot reuse stale generation");
            }
            state.finish_layer(layer);
            rejects([&]{state.chunk(layer,qsa,pos0,4);},"finished layer cannot reuse its stale stage");
        }
        check(state.complete() && state.seeds()==12 && state.reused_chunks()==48 &&
              state.seed_blocks()==12*uint64_t(pos0/4+(pos0%4!=0)) &&
              state.avoided_prefix_blocks()==avoided,"complete layer and logical traffic counts");
        rejects([&]{state.begin_layer(47,11);},"completed request cannot reenter");
        layer_major_kv_require_idle(nullptr);
        rejects([&]{layer_major_kv_require_idle(&state);},"public mutation/reentry requires idle lease");
    }
    LayerMajorKvPrefixReuse ordered(19,3,4,262144);
    rejects([&]{ordered.begin_layer(7,1);},"cannot skip first QSA layer");
    rejects([&]{ordered.begin_layer(2,0);},"GDN cannot become QSA stage owner");
    // A cancelled/failed call has only request-local metadata. A fresh call at
    // the SAME session position must seed, even if the previous call advanced.
    ordered.begin_layer(3,0); check(ordered.chunk(3,0,3,4).seed,"initial call seeds");
    LayerMajorKvPrefixReuse resumed(19,3,4,262144); resumed.begin_layer(3,0);
    check(resumed.chunk(3,0,3,4).seed,"new call never trusts previous stage validity");
    // Opaque encoded-cell simulation driven by the actual production helper:
    // compare retained stage against the old copy-every-chunk control. Both
    // host mirrors receive identical appends. Padding is not valid KV until
    // appended; compare every valid prior/current cell, including partial pages.
    LayerMajorKvPrefixReuse simulated(19,3,4,32);
    std::vector<uint32_t> stage(32,0xdeadbeefu),control_stage(32,0xbad0bad0u);
    for(int64_t qsa=0;qsa<12;++qsa) {
        const int64_t layer=4*qsa+3;
        std::vector<uint32_t> host(32),control_host(32);
        for(size_t i=0;i<host.size();++i) host[i]=uint32_t(qsa+1)*1000000u+uint32_t(i);
        control_host=host;
        simulated.begin_layer(layer,qsa);
        for(int64_t p=3;p<22;p+=4) {
            const int64_t count=std::min<int64_t>(4,22-p);
            const auto decision=simulated.chunk(layer,qsa,p,count);
            std::copy_n(control_host.begin(),size_t((p/4+(p%4!=0))*4),control_stage.begin());
            if(decision.seed) std::copy_n(host.begin(),size_t(decision.seed_blocks*4),stage.begin());
            for(int64_t i=p;i<p+count;++i) {
                const uint32_t encoded=0x80000000u|(uint32_t(qsa)<<20)|uint32_t(i);
                stage[size_t(i)]=host[size_t(i)]=encoded;
                control_stage[size_t(i)]=control_host[size_t(i)]=encoded;
            }
            check(std::equal(stage.begin(),stage.begin()+p+count,control_stage.begin()),
                  "all valid encoded cells match copy-every-chunk through partial pages/layer changes");
            check(host==control_host,"authoritative host mirror remains unchanged by reuse policy");
        }
        simulated.finish_layer(layer);
    }
    check(simulated.complete(),"all simulated QSA layers completed");
    for(const int64_t chunk:{int64_t(1),int64_t(4096),int64_t(8192)}) {
        LayerMajorKvPrefixReuse full(262144,0,chunk,262144);
        full.begin_layer(3,0);
        for(int64_t p=0;p<262144;p+=chunk) {
            const auto d=full.chunk(3,0,p,std::min<int64_t>(chunk,262144-p));
            check(d.seed==(p==0),"planned final context position has monotonic exact spans");
        }
        full.finish_layer(3);
    }
    LayerMajorKvPrefixReuse final_partial(3,262141,8192,262144);
    final_partial.begin_layer(3,0);
    const auto tail=final_partial.chunk(3,0,262141,3);
    check(tail.seed && tail.seed_blocks==65536,"nonzero final partial page seed extent");
    final_partial.finish_layer(3);
    rejects([]{LayerMajorKvPrefixReuse x(4,262141,4,262144);},"context overrun rejected");
    rejects([]{LayerMajorKvPrefixReuse x(1,-1,4,262144);},"negative position rejected");
    rejects([]{LayerMajorKvPrefixReuse x(0,0,4,262144);},"empty request rejected");
    rejects([]{LayerMajorKvPrefixReuse x(1,0,8193,262144);},"unsupported chunk rejected");
    rejects([]{LayerMajorKvPrefixReuse x(INT64_MAX,0,4,262144);},"huge row extent rejected");
    rejects([]{LayerMajorKvPrefixReuse x(1,0,4,262145);},"context maximum remains bounded");
    unsigned char memory[256]{};
    check(layer_major_kv_disjoint(memory,64,memory+64,128),"adjacent allocations disjoint");
    check(!layer_major_kv_disjoint(memory,65,memory+64,128),"stage/draft scratch overlap rejected");
    rejects([&]{layer_major_kv_disjoint(nullptr,1,memory,1);},"null scratch/store rejected");
    rejects([&]{layer_major_kv_disjoint(memory,UINT64_MAX,memory+64,1);},"address-span overflow rejected");
    std::cout<<"PASS layer-major KV prefix reuse host scheduling/address contracts; no GPU or physical-context qualification\n";
}
