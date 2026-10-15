// Main-only CPU test proposal. No product/runtime/model dependencies.
#include "cpu_gu_reuse_observer.hpp"
#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <vector>
using O=strata::core::CpuGuReuseObserver;
struct Ref {
    std::vector<unsigned> order;
    uint64_t bytes=0,hits=0,misses=0,evictions=0,hit_tokens=0;
    void touch(unsigned id,unsigned nt,uint64_t budget,const int* fmt) {
        auto it=std::find(order.begin(),order.end(),id);
        if(it!=order.end()){++hits;hit_tokens+=nt;order.erase(it);}
        else {++misses;uint64_t sz=O::payload(fmt[id/512]);while(bytes+sz>budget){unsigned v=order.back();order.pop_back();bytes-=O::payload(fmt[v/512]);++evictions;}bytes+=sz;}
        order.insert(order.begin(),id);
    }
};
std::vector<uint64_t> decode(const std::string& output) {
    const std::string prefix="expert_jobs_uleb128_base64 ";auto p=output.find(prefix);assert(p!=std::string::npos);p+=prefix.size();
    std::string alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";unsigned acc=0,bits=0;std::vector<unsigned> raw;
    for(;output[p]!='\n'&&output[p]!='=';++p){auto v=alphabet.find(output[p]);assert(v!=std::string::npos);acc=(acc<<6)|(unsigned)v;bits+=6;if(bits>=8){bits-=8;raw.push_back((acc>>bits)&255);}}
    std::vector<uint64_t> values;uint64_t v=0;unsigned shift=0;
    for(unsigned b:raw){assert(shift<64);v|=uint64_t(b&127)<<shift;if(b&128)shift+=7;else{values.push_back(v);v=0;shift=0;}}
    assert(shift==0);return values;
}
std::string receipt(const O& o,uint64_t jobs,uint64_t tokens,bool ledger=true) {
    std::vector<char> guarded(O::OutputCap+32,'Q');size_t n=o.receipt(guarded.data(),17,jobs,tokens,ledger);
    assert(n<O::OutputCap);for(size_t i=O::OutputCap;i<guarded.size();++i)assert(guarded[i]=='Q');return {guarded.data(),n};
}
int main() {
    int fmt[48];for(int l=0;l<48;++l)fmt[l]=l%3==0?18:l%3==1?22:21;
    auto o=std::make_unique<O>(fmt);Ref ref[4];uint64_t counts[O::N][4]{};uint64_t tokens=0;
    // Mixed formats, multiple job nt, deterministic short/long recurrence and evictions.
    uint32_t rng=314159;
    for(unsigned i=0;i<16000;++i){rng=rng*1664525u+1013904223u;unsigned id=i%5?((rng>>8)%O::N):17;unsigned nt=1+(rng%4);
        o->touch(id/512,id%512,nt);o->completed(1,nt);++counts[id][nt-1];tokens+=nt;
        for(unsigned b=0;b<4;++b)ref[b].touch(id,nt,O::Budgets[b],fmt);
    }
    assert(o->conserved(16000,tokens));assert(o->unique+o->retouches==16000);
    uint64_t reference_unique=0;for(auto& row:counts){bool seen=false;for(auto c:row)seen|=c!=0;reference_unique+=seen;}
    assert(o->unique==reference_unique&&o->retouches==16000-reference_unique);
    for(unsigned b=0;b<4;++b){auto& c=o->cache[b];auto& r=ref[b];assert(c.hits==r.hits&&c.misses==r.misses&&c.evictions==r.evictions&&c.bytes==r.bytes&&c.hit_tokens==r.hit_tokens);
        unsigned node=c.head;for(unsigned id:r.order){assert(node==id);node=c.next[node];}assert(node==O::Nil);
    }
    auto text=receipt(*o,16000,tokens);assert(text.find("complete=1")!=std::string::npos);auto decoded=decode(text);assert(decoded.size()==O::N*4);
    for(unsigned id=0;id<O::N;++id)for(unsigned n=0;n<4;++n)assert(decoded[id*4+n]==counts[id][n]);
    assert(receipt(*o,16001,tokens).find("complete=0")!=std::string::npos);
    assert(receipt(*o,16000,tokens,false).find("complete=0")!=std::string::npos);
    // One grouped CPU job per expert, with true IDs distinct from job-array slots;
    // GPU-owned expert 9 must not appear. Repeated route 7 shares its nt=2 job.
    auto grouped=std::make_unique<O>(fmt);int ids[]={7,9,3,7};bool cpu[]={true,false,true,true};int map[512];std::fill(map,map+512,-1);int origin[4],nt[4]{},nj=0;
    for(unsigned i=0;i<4;++i)if(cpu[i]){int e=ids[i];if(map[e]<0){map[e]=nj;origin[nj++]=e;}++nt[map[e]];}
    for(int j=0;j<nj;++j)grouped->touch(0,origin[j],nt[j]);grouped->completed(nj,3);
    assert(grouped->conserved(2,3));assert(grouped->entries[7].jobs[1]==1&&grouped->entries[3].jobs[0]==1&&grouped->entries[9].jobs[0]==0);
    // OFF wrapper leaves both observer and arithmetic unchanged, with no observer call.
    uint64_t before=grouped->jobs;O* off=nullptr;float a=3.5f;auto call=[&](O* p){if(p)p->touch(0,7,1);return a*a+2;};assert(call(off)==14.25f&&grouped->jobs==before);
    // Fresh requests are independently cold even for the same expert/model.
    auto fresh=std::make_unique<O>(fmt);fresh->touch(0,7,1);fresh->completed(1,1);
    assert(fresh->cache[0].hits==0&&fresh->cache[0].misses==1&&fresh->conserved(1,1));
    // Exercise ULEB128 width changes and both base64 padding cases independently.
    for(unsigned extra=1;extra<=2;++extra){auto widths=std::make_unique<O>(fmt);widths->entries[0].jobs[0]=extra==1?128:16384;auto decoded_width=decode(receipt(*widths,0,0));assert(decoded_width.size()==O::N*4&&decoded_width[0]==(extra==1?128:16384));}
    fmt[0]=99;auto unknown=std::make_unique<O>(fmt);unknown->touch(0,4,4);unknown->completed(1,4);assert(unknown->conserved(1,4)&&unknown->unsupported==1&&unknown->cache[0].hits==0&&unknown->cache[0].misses==0);
    for(auto bad:std::array<std::array<int,3>,5>{{{{-1,0,1}},{{48,0,1}},{{0,512,1}},{{0,0,0}},{{0,0,5}}}}){auto x=std::make_unique<O>(fmt);x->touch(bad[0],bad[1],bad[2]);assert(x->invalid&&x->jobs==0);}
    auto over=std::make_unique<O>(fmt);over->jobs=UINT64_MAX;over->touch(1,1,1);assert(over->overflow&&over->invalid);uint64_t j=over->jobs;over->touch(1,2,1);assert(over->jobs==j);
    // Worst-case legal-width counters exceed output capacity: no partial detail survives.
    auto huge=std::make_unique<O>(fmt);for(auto& e:huge->entries)for(auto& c:e.jobs)c=UINT64_MAX;
    auto full=receipt(*huge,0,0);assert(full.find("reason=output_cap")!=std::string::npos&&full.find("expert_jobs")==std::string::npos);
    std::puts("PASS exact byte-budget LRU/reference; nt/expert conservation; origin grouping; unknown-format bypass; off wrapper; overflow latch; base64 roundtrip; bounded complete/incomplete output");
}
