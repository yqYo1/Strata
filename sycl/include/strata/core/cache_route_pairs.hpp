// Host-only diagnostic. No routing, cache admission or kernel arithmetic decisions.
#pragma once
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <thread>
namespace strata::core {
class CacheRoutePairs {
public:
    static constexpr size_t cells = 48 * 512;
    struct Cell { uint64_t entries=0, hits=0, refused=0, offloaded=0, callbacks=0; };
    static bool enabled() {
        const char* s=std::getenv("STRATA_CACHE_ROUTE_PAIRS");
        if (!s || std::strcmp(s,"0")==0) return false;
        if (std::strcmp(s,"1")==0) return true;
        throw std::runtime_error("STRATA_CACHE_ROUTE_PAIRS must be exactly 0 or 1");
    }
    void begin(int64_t layers, int64_t experts, const int32_t* resident, size_t count, bool supported) {
        owner_=std::this_thread::get_id(); error_.store(false); ++ordinal_;
        table_={}; seen_={}; callbacks_=0; layers_=layers; experts_=experts;
        supported_=supported && layers==48 && experts==512 && resident && count==cells;
        if (supported_) std::memcpy(start_.data(),resident,sizeof(start_));
    }
    void callback() {
        if (!owned()) return;
        seen_={}; increment(callbacks_);
    }
    void unsupported_callback() { error_.store(true); }
    // Called immediately beside the existing validated classification scalar increments.
    void entry(int64_t layer, int64_t expert, int kind) {
        if (!owned()) return;
        if (layer<0 || layer>=48 || expert<0 || expert>=512) { error_.store(true); return; }
        auto& c=table_[size_t(layer)*512+size_t(expert)];
        increment(c.entries);
        increment(kind==0?c.hits:kind<0?c.refused:c.offloaded);
        if (!seen_[size_t(expert)]) { seen_[size_t(expert)]=true; increment(c.callbacks); }
    }
    bool report(FILE* f, const int32_t* resident, size_t count, int64_t hits, int64_t look,
                int64_t offloaded, bool dispatch_failed, bool request_complete = true) {
        const bool owner=std::this_thread::get_id()==owner_;
        bool unchanged=supported_ && resident && count==cells;
        if (unchanged) unchanged=std::memcmp(start_.data(),resident,sizeof(start_))==0;
        uint64_t entries=0,h=0,r=0,o=0,unique=0,nonzero=0;
        auto add = [&](uint64_t& total, uint64_t value) {
            if (value > std::numeric_limits<uint64_t>::max()-total) error_.store(true);
            else total += value;
        };
        if (owner && supported_) for(const auto& c:table_) {
            add(entries,c.entries);add(h,c.hits);add(r,c.refused);add(o,c.offloaded);add(unique,c.callbacks);
            nonzero+=c.entries!=0;
        }
        // DONE look excludes offloads; admissions/scalar callbacks therefore fail reconciliation.
        const bool reconciled=hits>=0 && look>=0 && offloaded>=0 && h==uint64_t(hits) &&
            r<=std::numeric_limits<uint64_t>::max()-h && h+r==uint64_t(look) && o==uint64_t(offloaded);
        const bool complete=supported_ && owner && !error_.load() && !dispatch_failed && request_complete && unchanged && reconciled && entries>0;
        std::fprintf(f,"CACHE_ROUTE_PAIRS_V1 BEGIN request=%llu layers=%lld experts=%lld enabled=1 supported=%d truncated=0 error=%d complete=%d\n",
            (unsigned long long)ordinal_,(long long)layers_,(long long)experts_,supported_,!owner||error_.load()||dispatch_failed,complete);
        std::fprintf(f,"CACHE_ROUTE_PAIRS_V1 RESIDENCY request=%llu cells=%zu unchanged=%d start_fnv1a64=%016llx end_fnv1a64=%016llx\n",
            (unsigned long long)ordinal_,supported_?cells:0,unchanged,
            (unsigned long long)(supported_?fingerprint(start_.data()):0),
            (unsigned long long)(supported_&&resident&&count==cells?fingerprint(resident):0));
        if(owner && supported_) for(size_t i=0;i<cells;++i) { const auto& c=table_[i];if(!c.entries)continue;
            std::fprintf(f,"CACHE_ROUTE_PAIRS_V1 PAIR %zu %zu %llu %llu %llu %llu %llu\n",
                i/512,i%512,(unsigned long long)c.entries,(unsigned long long)c.hits,(unsigned long long)c.refused,
                (unsigned long long)c.offloaded,(unsigned long long)c.callbacks);
        }
        std::fprintf(f,"CACHE_ROUTE_PAIRS_V1 END request=%llu cells=%llu callbacks=%llu entries=%llu hits=%llu refused=%llu offloaded=%llu callback_pairs=%llu done_hits=%lld done_look=%lld done_offload=%lld reconciled=%d complete=%d\n",
            (unsigned long long)ordinal_,(unsigned long long)nonzero,(unsigned long long)callbacks_,(unsigned long long)entries,
            (unsigned long long)h,(unsigned long long)r,(unsigned long long)o,(unsigned long long)unique,
            (long long)hits,(long long)look,(long long)offloaded,reconciled,complete);
        std::fflush(f); return complete;
    }
private:
    bool owned() { if(std::this_thread::get_id()!=owner_) { error_.store(true);return false; } return supported_; }
    void increment(uint64_t& v) { if(v==std::numeric_limits<uint64_t>::max())error_.store(true);else ++v; }
    static uint64_t fingerprint(const int32_t* p) {
        uint64_t h=14695981039346656037ULL;
        for(size_t i=0;i<cells;++i) { uint32_t v=uint32_t(p[i]);for(unsigned b=0;b<4;++b){h^=(v>>(8*b))&255;h*=1099511628211ULL;} }
        return h;
    }
    std::array<Cell,cells> table_{};
    std::array<int32_t,cells> start_{};
    std::array<bool,512> seen_{};
    std::atomic<bool> error_{false};
    std::thread::id owner_{};
    uint64_t ordinal_=0,callbacks_=0;
    int64_t layers_=0,experts_=0;
    bool supported_=false;
};
static_assert(sizeof(CacheRoutePairs)<2*1024*1024,"fixed diagnostic memory budget");
}
