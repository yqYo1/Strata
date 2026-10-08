
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
}
bool idle=false, retired=false, fail_second_download=false;
int downloads=0;
namespace dpct {
struct Queue {
    void wait_and_throw() { idle=true; }
    Queue& memcpy(void* d, const void* s, size_t n) {
        std::memcpy(d,s,n); return *this;
    }
    Queue& memset(void* d, int v, size_t n) { std::memset(d,v,n); return *this; }
};
using queue_ptr=Queue*;
struct Device { void queues_wait_and_throw() { idle=true; } };
Device& get_current_device() { static Device d; return d; }
}
namespace strata::kernels::cpu {
struct Layout {
    int64_t n_layers=2, n_expert=2;
    size_t blob_bytes(int64_t l) const { return l ? 32 : 16; }
};
const Layout& expert_layout() { static Layout l; return l; }
}
namespace core {
struct ExpertSource { virtual ~ExpertSource()=default; };
struct ArenaExpertSource : ExpertSource {
    bool available=true, missing=false;
    std::array<std::vector<uint8_t>,4> data;
    ArenaExpertSource() {
        for(int i=0;i<4;++i) data[i].assign(i<2?16:32,static_cast<uint8_t>(i+10));
    }
    bool mapped() const { return available; }
    const uint8_t* blob(int64_t l,int64_t e) { return missing ? nullptr : data[l*2+e].data(); }
};
struct ExpertCache {
    std::array<uint8_t,128> data{};
    // Profile order deliberately mixes layers. The64-byte segment boundary
    // cuts the third32-byte expert at its midpoint, rather than a slot edge.
    std::array<int64_t,5> offsets{0,32,48,80,96};
    std::array<int32_t,4> admission{1,3,0,2};
    int64_t mapped=128;
    bool bad_span=false, fail_partial_shrink=false, fail_grow_once=false;
    int shrinks=0, grows=0;
    int64_t bytes() const { return 96; }
    int64_t mapped_bytes() const { return mapped; }
    int64_t slots() const { return 4; }
    int64_t segment_bytes() const { return 64; }
    bool segmented() const { return true; }
    uint8_t* device_slot(int64_t s) { return data.data()+offsets[s]; }
    int64_t bytes_of(int64_t s) const { return offsets[s]-(bad_span&&s==2?1:0); }
    int32_t slot_of(int64_t l,int64_t e) const { return admission[l*2+e]; }
    bool shrink(int64_t keep,std::string& error) {
        ++shrinks; assert(idle);
        if(keep==mapped) return true;
        assert(retired);
        mapped=fail_partial_shrink ? 64 : keep;
        std::fill(data.begin()+mapped,data.end(),static_cast<uint8_t>(0xcc));
        if(fail_partial_shrink) { fail_partial_shrink=false; error="injected partial shrink"; return false; }
        return true;
    }
    bool grow(int64_t want,std::string& error) {
        ++grows; assert(idle);
        if(fail_grow_once) { fail_grow_once=false; error="injected grow"; return false; }
        mapped=want; return true;
    }
};
}
#include "lease.inc"
int main(int argc,char** argv) {
    assert(argc==2); const std::string name=argv[1];
    core::ExpertCache cache; core::ArenaExpertSource arena; dpct::Queue q;
    auto residency=cache.admission;
    if(name.find("adaptive")!=std::string::npos) std::swap(residency[2],residency[3]);
    for(int64_t l=0;l<2;++l) for(int64_t e=0;e<2;++e)
        std::memcpy(cache.device_slot(residency[l*2+e]),arena.blob(l,e),l?32:16);
    const auto before=cache.data;
    const bool snapshot=name.find("snapshot")!=std::string::npos;
    const char* fraction=name.find("control")!=std::string::npos?"0":name.find("half")!=std::string::npos?"0.5":"1";
    if(name=="invalid-nan") fraction="nan";
    if(name=="invalid-trailing") fraction="0.5x";
    if(name=="invalid-low") fraction="-0.1";
    if(name=="invalid-high") fraction="1.1";
    setenv("STRATA_PREFILL_CACHE_RELEASE_FRAC",fraction,1);
    setenv("STRATA_PREFILL_CACHE_RESTORE",snapshot?"snapshot":"ram",1);
    setenv("STRATA_PREFILL_CACHE_VERIFY","1",1);
    arena.available=name!="unmapped-source";
    arena.missing=name=="missing-blob";
    cache.bad_span=name=="wrong-span";
    if(name=="pre-mismatch") cache.data[0]^=1;
    cache.fail_partial_shrink=name=="partial-shrink-rollback";
    cache.fail_grow_once=name=="grow-retry";
    bool retire_fail=name=="retire-failure", refresh_fail=name=="callback-retry";
    int retires=0, refreshes=0;
    auto retire=[&](std::string& error) {
        ++retires; assert(idle);
        if(retire_fail) { error="injected retire"; return false; }
        retired=true; return true;
    };
    auto refresh=[&](const uint8_t* p,std::string& error) {
        ++refreshes; assert(idle&&retired&&cache.mapped==128&&p==cache.data.data());
        assert(std::equal(before.begin(),before.begin()+96,cache.data.begin()));
        if(refresh_fail) { refresh_fail=false; error="injected refresh"; return false; }
        return true;
    };
    std::string error;
    {
        CacheLease lease;
        bool ok=lease.suspend(&cache,&arena,residency.data(),&q,retire,refresh,error);
        const bool early_failure=name.rfind("invalid-",0)==0||name=="unmapped-source"||name=="missing-blob"||name=="wrong-span"||name=="pre-mismatch"||name=="retire-failure";
        if(early_failure) {
            assert(!ok&&!lease.active&&cache.mapped==128&&cache.shrinks==0&&!error.empty());
            assert(cache.data==before||name=="pre-mismatch");
        } else if(name=="partial-shrink-rollback") {
            assert(!ok&&lease.active&&cache.mapped==64&&retires==1);
            // Actual destructor must remap, restore the entire RAM payload and
            // rebuild the retired graph after a partially successful shrink.
        } else {
            assert(ok&&lease.active);
            const int64_t expected=name.find("control")!=std::string::npos?128:name.find("half")!=std::string::npos?64:0;
            assert(cache.mapped==expected&&lease.kept==expected);
            if(expected==64) assert(lease.restore_bytes==32&&(!snapshot?lease.span_bytes==32:true));
            if(name.find("adaptive")!=std::string::npos) assert(lease.remapped_experts==2);
            if(name=="grow-retry"||name=="callback-retry") {
                assert(!lease.restore(error)&&lease.active);
                assert(lease.restore(error));
            } else assert(lease.restore(error));
            assert(!lease.active&&cache.mapped==128);
            assert(std::equal(before.begin(),before.begin()+96,cache.data.begin()));
            assert(retires==(expected<128?1:0));
            assert(refreshes==(expected<128?(name=="callback-retry"?2:1):0));
        }
    }
    if(name=="partial-shrink-rollback") {
        assert(cache.mapped==128&&refreshes==1);
        assert(std::equal(before.begin(),before.begin()+96,cache.data.begin()));
    }
    std::printf("PASS %s\n",name.c_str());
}
