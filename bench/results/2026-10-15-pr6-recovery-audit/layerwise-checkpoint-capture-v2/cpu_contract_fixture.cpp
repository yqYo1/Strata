// PROPOSAL ONLY: author has not compiled or run this fixture.
// Compile with fixture_stubs FIRST, proposed/include SECOND, product include THIRD.
// This includes the EXACT candidate LayerwiseCheckpoints implementation. Only
// GPU-dependent core types/sizing/validation are replaced with explicit host stubs.
#include "strata/program/layerwise_checkpoint.hpp"
#include <cassert>
#ifdef NDEBUG
#error "CPU contract fixture requires assertions; compile with -UNDEBUG"
#endif
#include <cstring>
#include <iostream>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

namespace cpu_fixture { inline std::optional<uint64_t> available = 64ull << 30; }
namespace strata::core {
std::optional<uint64_t> conversation_available_memory() { return cpu_fixture::available; }
}
using strata::program::LayerwiseCheckpoints;
using strata::program::layerwise_capture_enabled;
using Check = strata::core::ConversationCheckpoint;
using Images = std::vector<strata::core::ConversationImageKey>;
using Chain = std::vector<Check>;
using Ids = std::vector<int64_t>;
struct Tag {
    uint64_t kind, layer, prefix, value;
    bool operator==(const Tag&) const = default;
};
static_assert(sizeof(Tag)==32);

static int64_t token(int64_t p) { return (17*p+13)%100003; }
static Tag oracle_tag(uint64_t kind, int64_t layer, int64_t prefix) {
    // Mathematical prefix function independent of captured state/masks/offsets.
    // Every actual fake-copy byte is checked against its expected owning prefix.
    return {kind,uint64_t(layer),uint64_t(prefix),
            uint64_t(prefix)*uint64_t(prefix+1)/2*uint64_t(layer+1)+kind};
}
static Ids ids(int64_t n) {
    Ids out(size_t(n),0);
    for (int64_t p=0;p<n;++p) out[size_t(p)] = token(p);
    return out;
}
static void put_tag(std::vector<uint8_t>& v, size_t off, Tag value) {
    assert(off <= v.size() && sizeof(Tag) <= v.size()-off);
    std::memcpy(v.data()+off,&value,sizeof value);
}
static bool same(const Check& a,const Check& b) {
    return a.ids==b.ids && a.imgs==b.imgs && a.gdn==b.gdn && a.ple==b.ple &&
           a.tails==b.tails && a.dead==b.dead && a.block_pos==b.block_pos && a.used==b.used &&
           a.stage_parts.size()==b.stage_parts.size();
}
static bool same(const Chain& a,const Chain& b) {
    if (a.size()!=b.size()) return false;
    for(size_t i=0;i<a.size();++i) if(!same(a[i],b[i])) return false;
    return true;
}
static Check old_state(int64_t prefix,uint64_t use,const Ids& input,const Images& images={}) {
    Check c;
    c.ids.assign(input.begin(),input.begin()+prefix); c.used=use;
    for(const auto& im:images) if(im.start<prefix) c.imgs.push_back(im);
    c.gdn.resize(36*sizeof(Tag)); c.ple.resize(sizeof(Tag));
    c.tails.resize(12*sizeof(Tag)); c.dead.resize(12*sizeof(Tag)); c.block_pos.resize(12*sizeof(int32_t));
    int q=0,g=0;
    for(int layer=0;layer<48;++layer) {
        if(layer%4==3) {
            put_tag(c.tails,size_t(q)*sizeof(Tag),oracle_tag(2,layer,prefix));
            put_tag(c.dead,size_t(q)*sizeof(Tag),oracle_tag(3,layer,prefix));
            const int32_t pos=int32_t(prefix);
            std::memcpy(c.block_pos.data()+size_t(q)*sizeof(pos),&pos,sizeof pos); ++q;
        } else put_tag(c.gdn,size_t(g++)*sizeof(Tag),oracle_tag(1,layer,prefix));
    }
    put_tag(c.ple,0,oracle_tag(4,1,prefix)); return c;
}
struct FakeSession {
    strata::core::ModelGeometry geometry;
    strata::core::SessionState ss;
    std::array<Tag,36> recurrent{};
    std::array<Tag,12> tails{},dead{};
    std::array<int32_t,12> block{};
    std::array<strata::core::QsaState,12> qsa{};
    Tag ple{};
    FakeSession() {
        ss.gdn_state=reinterpret_cast<float*>(recurrent.data());
        ss.ple_hist=reinterpret_cast<float*>(&ple); ss.qsa_states=qsa.data();
        for(int q=0;q<12;++q) qsa[q]={&tails[q],&dead[q],&block[q]};
    }
    void advance(int layer,int64_t prefix) {
        // Independent storage layout derived from explicit groups of 4 layers.
        if(layer%4==3) {
            const int q=layer/4;
            tails[q]=oracle_tag(2,layer,prefix); dead[q]=oracle_tag(3,layer,prefix); block[q]=int32_t(prefix);
        } else recurrent[size_t((layer/4)*3+layer%4)]=oracle_tag(1,layer,prefix);
        if(layer==1) ple=oracle_tag(4,1,prefix);
    }
};
struct Node { int64_t pos; uint64_t stamp; };
struct Oracle { std::vector<Node> chain; uint64_t clock=0; std::vector<int64_t> periodic; };
static Oracle retention_oracle(const Chain& old,uint64_t clock,int cap,int64_t begin,int64_t end,
                               int64_t chunk,int64_t threshold,int64_t interval,bool boundary) {
    Oracle o; o.clock=clock;
    for(const auto& c:old) o.chain.push_back({int64_t(c.ids.size()),c.used});
    auto use=[&](int64_t p) {
        assert(o.clock<UINT64_MAX); ++o.clock;
        auto found=std::find_if(o.chain.begin(),o.chain.end(),[&](const Node& n){return n.pos==p;});
        if(found!=o.chain.end()) found->stamp=o.clock;
        else o.chain.push_back({p,o.clock});
        if(o.chain.size()>size_t(cap)) {
            auto first=o.chain.begin()+(cap>=2?1:0);
            auto victim=cap==1?first:std::min_element(first,o.chain.end(),
                [](const Node& a,const Node& b){return a.stamp<b.stamp;});
            o.chain.erase(victim);
        }
    };
    // Token-time independent oracle. No call to proposal arithmetic/planner or
    // production eviction helper. inspect actual ends while elapsed clock runs.
    int64_t elapsed=threshold>begin ? -(threshold-begin) : begin-threshold;
    int64_t count=0;
    for(int64_t p=begin;p<end;) {
        ++p; ++elapsed; ++count;
        if(count==chunk || p==end) {
            if(elapsed>=0) { o.periodic.push_back(p); use(p); elapsed=-interval; }
            count=0;
        }
    }
    if(boundary) use(end);
    return o;
}
struct CopyRecord { int layer; int64_t prefix; uint64_t kind; };
struct CopyRecorder {
    FakeSession& host;
    int layer=0; int64_t prefix=0;
    int calls=0,fail_at=-1;
    std::vector<CopyRecord> records;
    explicit CopyRecorder(FakeSession& h):host(h) {}
    bool copy(void* dst,const void* src,size_t bytes,std::string& error) {
        const int call=calls++;
        if(call==fail_at) { error="injected checked host-copy failure"; return false; }
        assert(dst && src);
        if(bytes==sizeof(Tag)) {
            Tag tag; std::memcpy(&tag,src,sizeof tag);
            if(src==&host.ple) {
                assert(layer==1 && tag==oracle_tag(4,1,prefix));
            } else {
                const uint64_t expected=layer%4==3 ?
                    (src==&host.tails[size_t(layer/4)]?2:3) : 1;
                assert(tag==oracle_tag(expected,layer,prefix));
            }
            records.push_back({layer,prefix,tag.kind});
        } else {
            assert(bytes==sizeof(int32_t) && layer%4==3 && src==&host.block[size_t(layer/4)]);
            int32_t p;std::memcpy(&p,src,sizeof p); assert(p==prefix);
            records.push_back({layer,prefix,5});
        }
        std::memcpy(dst,src,bytes); return true;
    }
};
static bool feed(LayerwiseCheckpoints& tx,FakeSession& host,CopyRecorder& copies,
                 int64_t begin,int64_t end,int64_t chunk,std::string& error,
                 bool omit_prompt=false,int stop_layer=-1,bool duplicate_prompt=false) {
    const uint64_t run=tx.identity();
    LayerwiseCheckpoints::Copy copy=[&](void* d,const void* s,size_t n,std::string& e) {
        return copies.copy(d,s,n,e);
    };
    for(int l=0;l<48;++l) {
        if(l==stop_layer) return false;
        for(int64_t at=begin;at<end;) {
            const int64_t done=at+std::min(chunk,end-at);
            host.advance(l,done); copies.layer=l;copies.prefix=done;
            if(!tx.layer(run,l,done,host.ss,host.geometry,copy,error)) return false;
            if(l==47 && !omit_prompt) {
                if(!tx.prompt(done,error)) return false;
                if(duplicate_prompt && !tx.prompt(done,error)) return false;
            }
            at=done;
        }
    }
    return true;
}
static void success_case(int64_t begin,int64_t end,int64_t chunk,int64_t threshold,int64_t interval,
                         int cap,Chain chain,uint64_t clock,bool boundary,const Images& images={}) {
    FakeSession host;auto input=ids(end);std::string error;LayerwiseCheckpoints tx;
    const Chain before=chain;const uint64_t old_clock=clock;
    const Oracle expected=retention_oracle(chain,clock,cap,begin,end,chunk,threshold,interval,boundary);
    assert(tx.begin(chain,clock,cap,input,images,host.ss,host.geometry,begin,end,chunk,threshold,interval,boundary,error));
    assert(same(chain,before) && clock==old_clock);
    CopyRecorder copies{host}; assert(feed(tx,host,copies,begin,end,chunk,error));
    assert(same(chain,before) && clock==old_clock); // still private, before refill
    assert(tx.commit(chain,clock,error)); // driver has proved run+expert restore+refill here
    assert(!tx.active() && clock==expected.clock && chain.size()==expected.chain.size());
    size_t fresh=0;
    for(size_t i=0;i<chain.size();++i) {
        const auto p=expected.chain[i].pos;
        const auto wanted=old_state(p,expected.chain[i].stamp,input,images);
        assert(same(chain[i],wanted));
        if(std::none_of(before.begin(),before.end(),[&](const Check& c){return c.ids.size()==size_t(p);})) ++fresh;
    }
    // 36 GDN + 3*12 QSA fields + 1 owning-layer PLE per NEW survivor.
    assert(copies.records.size()==fresh*(36+3*12+1));
    for(const auto& r:copies.records)
        assert(std::any_of(expected.chain.begin(),expected.chain.end(),[&](const Node& n){return n.pos==r.prefix;}));
    const auto ple_count=std::count_if(copies.records.begin(),copies.records.end(),[](const CopyRecord& r){return r.kind==4;});
    assert(size_t(ple_count)==fresh);
}

static void scheduling_and_retention() {
    for(const int64_t length:{32768,65536,262144})
        for(const int64_t chunk:{4096,8192})
            for(const int64_t interval:{1,16384,16385})
                for(int cap:{1,6})
                    success_case(0,length,chunk,interval,interval,cap,{},0,false);
    {
        const auto o=retention_oracle({},0,6,0,262144,8192,16384,16384,false);
        std::vector<int64_t> positions;for(const auto& n:o.chain) positions.push_back(n.pos);
        assert((positions==std::vector<int64_t>{16384,196608,212992,229376,245760,262144}));
    }
    success_case(0,34602,8192,16384,16384,6,{},0,false,{{0,123},{16384,456},{32768,789}});
    success_case(0,32768,8192,16384,16384,6,{},0,true); // duplicate periodic+root/turn stamp
    const auto input=ids(65536);
    Chain old{old_state(2048,9,input),old_state(8192,2,input),old_state(16384,8,input)};
    success_case(16384,65536,8192,32768,16384,6,old,10,true);
    success_case(16384,65536,8192,32768,16384,1,{old.back()},10,true);
    success_case(16384,32768,4096,32768,16384,6,old,10,true);
    success_case(4000,26000,8192,20384,16384,6,{old_state(2048,1,input)},2,true);
    // Overdue threshold after verifier windows; first actual batch chunk saves.
    success_case(17000,40000,8192,16384,16384,6,{old_state(2048,1,input)},2,false);
    // INT64_MAX sentinel is inactive at every supported physical prefix.
    success_case(0,65536,8192,INT64_MAX,INT64_MAX,6,{},0,false);
}
static void bad_order_and_failures() {
    const auto input=ids(32768);FakeSession host;std::string error;
    auto start=[&](LayerwiseCheckpoints& tx,const Chain& old,uint64_t clock) {
        assert(tx.begin(old,clock,6,input,{},host.ss,host.geometry,0,32768,8192,16384,16384,false,error));
    };
    LayerwiseCheckpoints::Copy inert=[](void* d,const void* s,size_t n,std::string&){std::memcpy(d,s,n);return true;};
    for(int fault=0;fault<5;++fault) {
        LayerwiseCheckpoints tx;Chain out;uint64_t clock=0;start(tx,out,clock);const uint64_t run=tx.identity();
        host.advance(0,8192);
        if(fault==0) assert(!tx.layer(run+1,0,8192,host.ss,host.geometry,inert,error));
        if(fault==1) assert(!tx.layer(run,1,8192,host.ss,host.geometry,inert,error));
        if(fault==2) assert(!tx.layer(run,0,16384,host.ss,host.geometry,inert,error));
        if(fault==3) {
            assert(tx.layer(run,0,8192,host.ss,host.geometry,inert,error));
            assert(!tx.layer(run,0,8192,host.ss,host.geometry,inert,error));
        }
        if(fault==4) assert(!tx.prompt(16384,error));
        assert(!tx.commit(out,clock,error)); assert(out.empty()&&clock==0);tx.discard();
    }
    for(int stopped:{0,1,23,47}) {
        LayerwiseCheckpoints tx;Chain out;uint64_t clock=0;start(tx,out,clock);CopyRecorder copies{host};
        assert(!feed(tx,host,copies,0,32768,8192,error,false,stopped));
        assert(!tx.commit(out,clock,error));assert(out.empty()&&clock==0);tx.discard();
    }
    for(bool duplicate:{false,true}) {
        LayerwiseCheckpoints tx;Chain out;uint64_t clock=0;start(tx,out,clock);CopyRecorder copies{host};
        const bool completed=feed(tx,host,copies,0,32768,8192,error,!duplicate,-1,duplicate);
        assert(completed==!duplicate);
        assert(!tx.commit(out,clock,error));assert(out.empty()&&clock==0);tx.discard();
    }
    // Every actual fake state-copy failure point: masks never publish an incomplete state.
    for(int fail=0;fail<2*(36+3*12+1);++fail) {
        LayerwiseCheckpoints tx;Chain out;uint64_t clock=0;start(tx,out,clock);CopyRecorder copies{host};copies.fail_at=fail;
        assert(!feed(tx,host,copies,0,32768,8192,error));
        assert(!tx.commit(out,clock,error));assert(out.empty()&&clock==0);tx.discard();
    }
    // Caller lifecycle failure/cancel after a successful full feed: actual tx is
    // discarded instead of committed, preserving REAL nonempty old state bytes.
    for(const char* failure:{"cancel","run","expert_restore","refill"}) {
        (void) failure;LayerwiseCheckpoints tx;Chain out{old_state(2048,1,input)};const Chain before=out;
        uint64_t clock=2;
        assert(tx.begin(out,clock,6,input,{},host.ss,host.geometry,8192,32768,8192,16384,16384,false,error));
        CopyRecorder copies{host};assert(feed(tx,host,copies,8192,32768,8192,error));
        tx.discard();assert(!tx.commit(out,clock,error));assert(same(out,before)&&clock==2);
    }
    // A previous run identity stays invalid after discard+begin on the same object.
    LayerwiseCheckpoints tx;Chain out;uint64_t clock=0;start(tx,out,clock);const auto stale=tx.identity();
    tx.discard();start(tx,out,clock);
    assert(tx.identity()!=stale && !tx.layer(stale,0,8192,host.ss,host.geometry,inert,error));tx.discard();
}
static void reread_regression() {
    FakeSession host;const auto input=ids(65536);std::string error;
    const int64_t resume=32768,reread_to=resume,read_from=0,chunk=8192;
    Chain old{old_state(2048,10,input),old_state(resume,12,input)};const Chain before=old;
    uint64_t clock=12;LayerwiseCheckpoints tx;
    // Negative control is the REAL v1 failure, not an empty-old/sentinel case.
    const bool v1_gate=true && read_from>=0 && reread_to-read_from>chunk;
    assert(v1_gate && !old.empty() && reread_to>chunk);
    assert(!tx.begin(old,clock,6,input,{},host.ss,host.geometry,
                     read_from,reread_to,chunk,INT64_MAX,16384,false,error));
    assert(error.find("old chain")!=std::string::npos);tx.discard();
    // The real v2 gate bypasses new capture throughout this request, including
    // segments after the initial reread, without changing its existing segment list.
    int64_t at=read_from;int whole_runs=0;
    for(const int64_t to:{reread_to,int64_t(2048),int64_t(49152),int64_t(65535)}) {
        if(to<=at) continue;
        assert(!layerwise_capture_enabled(true,false,reread_to,at,to,chunk));
        ++whole_runs;at=to; // stands for original whole Prefill::run, no new begin
    }
    assert(whole_runs==3 && same(old,before) && clock==12 && !tx.active());
    // Existing no-periodic sentinel schedule remains empty on the whole reread.
    const auto schedule=retention_oracle(old,clock,6,0,resume,chunk,INT64_MAX,16384,false);
    assert(schedule.periodic.empty() && schedule.clock==clock);
    // Existing root/turn calls still run outside capture. Their duplicate stamp
    // and new-boundary event semantics are independent of the bypass.
    const auto boundary=retention_oracle(old,clock,6,resume,49152,chunk,INT64_MAX,16384,true);
    assert(boundary.periodic.empty() && boundary.clock==13 && boundary.chain.back().pos==49152);
    assert(layerwise_capture_enabled(true,false,-1,resume,65536,chunk));
    assert(!layerwise_capture_enabled(false,false,-1,0,32768,chunk));
    assert(!layerwise_capture_enabled(true,true,-1,0,32768,chunk));
    assert(!layerwise_capture_enabled(true,false,-1,0,chunk,chunk));
}
static void callback_raii() {
    using Callback=std::function<int()>;
    LayerwiseCheckpoints owner; // outer lifetime, same declaration order as serve
    Callback callback=[] {return 19;};
    bool restored_before_owner_end=false;
    {
        strata::program::ScopedLayerwiseCallback guard(callback,true);
        callback=[&owner] {return owner.active()?1:2;};
        assert(callback()==2);
    }
    assert(callback()==19);restored_before_owner_end=true;
    try {
        strata::program::ScopedLayerwiseCallback guard(callback,true);
        callback=[&owner] {return owner.active()?1:3;};
        assert(callback()==3);throw std::runtime_error("injected run exception");
    } catch(const std::runtime_error&) {}
    assert(callback()==19 && restored_before_owner_end);
    // Exception before installation also restores a slot already borrowed.
    try {
        strata::program::ScopedLayerwiseCallback guard(callback,true);
        throw std::runtime_error("injected callback construction failure");
    } catch(const std::runtime_error&) {}
    assert(callback()==19);
    auto early_return=[&]() {
        strata::program::ScopedLayerwiseCallback guard(callback,true);
        callback=[&owner] {return owner.active()?1:7;};return;
    };
    early_return();assert(callback()==19);
    for(int i=0;i<1;++i) {
        strata::program::ScopedLayerwiseCallback guard(callback,true);
        callback=[] {return 8;};break;
    }
    assert(callback()==19);
    {
        strata::program::ScopedLayerwiseCallback guard(callback,false);
        assert(callback()==19); // no move, clear or assignment in disabled path
    }
    assert(callback()==19);
}
static void limits_and_admission() {
    using strata::program::layerwise_next;
    assert(layerwise_next(INT64_MAX-3,10)==INT64_MAX);
    assert(layerwise_next(INT64_MAX,INT64_MAX)==INT64_MAX);
    assert(layerwise_next(0,INT64_MAX)==INT64_MAX && layerwise_next(8,0)==8);
    bool thrown=false;try{(void)layerwise_next(-1,1);}catch(const std::invalid_argument&){thrown=true;}assert(thrown);
    thrown=false;try{(void)layerwise_next(0,-1);}catch(const std::invalid_argument&){thrown=true;}assert(thrown);
    assert(layerwise_capture_enabled(true,false,0,INT64_MAX-10,INT64_MAX,3));
    assert(!layerwise_capture_enabled(true,false,0,-1,INT64_MAX,3));
    FakeSession host;const auto input=ids(32768);std::string error;
    auto begin=[&](LayerwiseCheckpoints& tx,const Chain& old,uint64_t clock,int64_t cap,
                   int64_t at,int64_t end,int64_t chunk,int64_t threshold,int64_t interval) {
        return tx.begin(old,clock,cap,input,{},host.ss,host.geometry,at,end,chunk,threshold,interval,false,error);
    };
    for(int fault=0;fault<8;++fault) {
        LayerwiseCheckpoints tx;Chain old;
        int64_t cap=6,at=0,end=32768,chunk=8192,threshold=16384,interval=16384;
        if(fault==0)cap=0;
        if(fault==1)interval=0;
        if(fault==2)chunk=0;
        if(fault==3)end=INT64_MAX;
        if(fault==4)at=-1;
        if(fault==5)threshold=-1;
        if(fault==6)cap=INT64_MAX;
        if(fault==7){at=INT64_MAX-10;end=INT64_MAX;}
        assert(!begin(tx,old,0,cap,at,end,chunk,threshold,interval));assert(old.empty());tx.discard();
    }
    {
        LayerwiseCheckpoints tx;Chain old;
        assert(!begin(tx,old,UINT64_MAX,6,0,32768,8192,16384,16384));assert(old.empty());tx.discard();
    }
    for(auto memory:{std::optional<uint64_t>{},std::optional<uint64_t>{(8ull<<30)-1},
                     std::optional<uint64_t>{8ull<<30}}) {
        cpu_fixture::available=memory;LayerwiseCheckpoints tx;Chain old;
        assert(!begin(tx,old,0,6,0,32768,8192,16384,16384));assert(!tx.active());tx.discard();
    }
    cpu_fixture::available=64ull<<30;
    {
        LayerwiseCheckpoints tx;Chain old;auto bad=input;bad[4]=-1;
        assert(!tx.begin(old,0,6,bad,{},host.ss,host.geometry,0,32768,8192,16384,16384,false,error));
        bad[4]=int64_t(INT32_MAX)+1;
        assert(!tx.begin(old,0,6,bad,{},host.ss,host.geometry,0,32768,8192,16384,16384,false,error));
    }
    {
        LayerwiseCheckpoints tx;Chain old;host.geometry.qsa_interval=0;
        assert(!begin(tx,old,0,6,0,32768,8192,16384,16384));host.geometry.qsa_interval=4;
    }
    // Exercise ACTUAL checked byte arithmetic before any gigantic allocation.
    {
        const auto normal=strata::core::fixture_sizes;
        for(int fault=0;fault<2;++fault) {
            LayerwiseCheckpoints tx;Chain old;
            if(fault==0) strata::core::fixture_sizes.gdn=size_t(INT64_MAX)-3;
            else strata::core::fixture_sizes.tail=size_t(INT64_MAX)/4;
            assert(!begin(tx,old,0,6,0,32768,8192,16384,16384));
            assert(error.find("overflow")!=std::string::npos && old.empty());
            tx.discard();strata::core::fixture_sizes=normal;
        }
    }
    // A mutated published chain or clock is detected before any move occurs.
    {
        LayerwiseCheckpoints tx;Chain old{old_state(2048,1,input)};uint64_t clock=2;
        assert(begin(tx,old,clock,6,8192,32768,8192,16384,16384));CopyRecorder copies{host};
        assert(feed(tx,host,copies,8192,32768,8192,error));++clock;const auto before=old;
        assert(!tx.commit(old,clock,error));assert(same(old,before));tx.discard();
    }
}
int main() {
    scheduling_and_retention();bad_order_and_failures();reread_regression();callback_raii();limits_and_admission();
    std::cout<<"PASS: ACTUAL candidate LayerwiseCheckpoints and gate/RAII with host stubs; "
                "not core-validator/queue/RESTORE/numerical/GPU/performance qualification\n";
}
