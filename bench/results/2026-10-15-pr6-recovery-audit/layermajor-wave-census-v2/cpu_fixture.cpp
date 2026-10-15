// SOURCE ONLY: Main owns CPU compilation/ASan/UBSan and execution.
#include "proposed/include/strata/prefill/layermajor_shape_census.hpp"
#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
using C=strata::prefill::detail::LayerMajorShapeCensus;
static C::Format format(){return {23,42,1360,180,640*1360,1280*1360,1280*1360+2560*180};}
static std::array<int32_t,512> residency(){std::array<int32_t,512> r{};for(int e=0;e<512;++e)r[e]=e;return r;}
static std::string finish(C& c){FILE* f=std::tmpfile();assert(f);c.complete(f);std::rewind(f);std::string s;char b[4096];size_t n;while((n=std::fread(b,1,sizeof(b),f)))s.append(b,n);std::fclose(f);return s;}
// Independent token->expert sets/maps, not the observer masks or wave bins.
struct Ref{std::set<int> ids;std::map<int,uint64_t> rows;uint64_t incidences=0,tokens=0,chunks=0;};
struct Expected{uint64_t waves=0,partial=0,unions=0,low=512,high=0,duplicates=0,rows=0,max_ne=0,partial_chunks=0,partial_tokens=0;std::array<uint64_t,16> hist{};};
static void flush(Ref& x,Expected& r,int q){if(!x.chunks)return;++r.waves;const auto u=x.ids.size();r.unions+=u;r.low=std::min<uint64_t>(r.low,u);r.high=std::max<uint64_t>(r.high,u);r.duplicates+=x.incidences-u;if(x.chunks<uint64_t(q)){++r.partial;r.partial_chunks=x.chunks;r.partial_tokens=x.tokens;}for(auto [id,n]:x.rows){(void)id;r.rows+=n;r.max_ne=std::max(r.max_ne,n);unsigned bin=0;while(bin<15 && n>=(uint64_t(1)<<(bin+1)))++bin;++r.hist[bin];}x=Ref{};}
static void check(const C::WaveSummary& x,const Expected& y){assert(x.waves==y.waves && x.partial_waves==y.partial && x.union_sum==y.unions && x.union_min==y.low && x.union_max==y.high && x.duplicate_incidences==y.duplicates && x.expert_rows==y.rows && x.max_ne==y.max_ne && x.hist==y.hist && x.partial_chunks==y.partial_chunks && x.partial_tokens==y.partial_tokens);}
static void feed(C& c,int layer,int64_t n,int64_t cap,bool concentrated){auto res=residency();c.begin(layer,format(),res.data(),true);std::array<Ref,2> windows;std::array<Expected,2> expected;int ordinal=0;for(int64_t off=0;off<n;off+=cap,++ordinal){const auto t=std::min(cap,n-off);std::array<int32_t,512> counts{};std::set<int> active;std::map<int,uint64_t> routed;for(int64_t token=0;token<t;++token)for(int k=0;k<10;++k){const int e=concentrated?k:int((token*17+k*31+ordinal*13+layer*7)%512);++counts[e];active.insert(e);++routed[e];}for(int i=0;i<2;++i){auto& w=windows[i];w.ids.insert(active.begin(),active.end());w.incidences+=active.size();w.tokens+=t;++w.chunks;for(auto [id,ne]:routed)w.rows[id]+=ne;if(w.chunks==uint64_t(i?4:2))flush(w,expected[i],i?4:2);}c.counts(layer,t,counts.data(),counts.size());for(int e=511;e>=0;--e)if(counts[e])c.issue(true,layer,counts[e],e,e,-1);for(int e=0;e<512;++e)if(counts[e])c.issue(false,layer,counts[e],e,e,-1);}for(int i=0;i<2;++i)flush(windows[i],expected[i],i?4:2);c.end(layer);assert(!c.failure);for(int i=0;i<2;++i)check(c.record[layer].wave[i],expected[i]);}
static std::unique_ptr<C> started(int64_t n=1){auto c=std::make_unique<C>(n,0,1);auto r=residency();c->begin(0,format(),r.data(),true);std::array<int32_t,512> count{};for(int i=0;i<10;++i)count[i]=1;c->counts(0,1,count.data(),count.size());return c;}
int main(){
    static_assert(sizeof(C)<=4u<<20);
    for(auto shape:std::vector<std::pair<int64_t,int64_t>>{{1,1},{13,3},{10,3},{32769,8192}}){auto c=std::make_unique<C>(shape.first,17,shape.second);for(int l=0;l<48;++l)feed(*c,l,shape.first,shape.second,false);auto text=finish(*c);assert(!c->failure && text.size()<=C::output_cap && text.find("\"wave_schema_version\":1")!=std::string::npos && text.find("\"complete\":true")!=std::string::npos);}
    // Exact q*8192 boundary: bin14 at16384, bin15 at32768.
    {auto c=std::make_unique<C>(32768,0,8192);feed(*c,0,32768,8192,true);assert(c->record[0].wave[0].hist[14]==20 && c->record[0].wave[1].hist[15]==10);c->finalized=true;}
    {auto c=started();c->issue(true,0,1,99,99,-1);assert(c->failure);c->finalized=true;} // same ne, wrong inactive expert
    {auto c=started();c->issue(true,0,1,0,0,-1);c->issue(true,0,1,0,0,-1);assert(c->failure);c->finalized=true;}
    {auto c=started();for(int e=0;e<10;++e)c->issue(true,0,1,e,e,-1);c->end(0);assert(c->failure);c->finalized=true;}
    {auto c=started(2);std::array<int32_t,512> count{};for(int i=0;i<10;++i)count[i]=1;c->counts(0,1,count.data(),count.size());assert(c->failure);c->finalized=true;} // missing issuer before next counts
    // New checked counters and per-expert addition fault injection.
    {auto c=std::make_unique<C>(1,0,1);auto r=residency();c->begin(0,format(),r.data(),true);c->windows[0].ne[0]=UINT32_MAX;std::array<int32_t,512> count{};for(int i=0;i<10;++i)count[i]=1;c->counts(0,1,count.data(),count.size());assert(c->failure);c->finalized=true;}
    {auto c=started();c->record[0].wave[0].union_sum=UINT64_MAX;for(int e=0;e<10;++e){c->issue(true,0,1,e,e,-1);c->issue(false,0,1,e,e,-1);}c->end(0);assert(c->failure);c->finalized=true;}
    {auto c=std::make_unique<C>(1,0,1);for(auto& r:c->record)r.hist.fill(UINT64_MAX);c->retired=48;auto text=finish(*c);assert(c->failure && text.find("output_overflow")!=std::string::npos && text.find("\"layers\"")==std::string::npos);}
    // Unbuffered output error exercises the original completion/I/O contract.
    {auto c=std::make_unique<C>(1,0,1);for(int l=0;l<48;++l)feed(*c,l,1,1,false);FILE* bad=std::fopen("/dev/full","w");assert(bad);std::setvbuf(bad,nullptr,_IONBF,0);c->complete(bad);assert(c->failure && std::string(c->failure)=="output_io_incomplete");std::fclose(bad);}
    {auto c=std::make_unique<C>(262144,1,8192);assert(c->failure);c->finalized=true;}
}
