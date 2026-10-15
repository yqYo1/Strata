// Proposal only. Issuer-owned request-cold CPU GU recurrence; no model data.
#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace strata::core {
struct CpuGuReuseObserver {
    static constexpr unsigned N=48*512, Nil=65535, OutputCap=256*1024;
    static_assert(N<Nil,"fixed expert IDs must fit the list link type");
    static constexpr uint64_t GiB=uint64_t(1)<<30;
    static constexpr std::array<uint64_t,4> Budgets={GiB/4,GiB,4*GiB,8*GiB};
    struct Entry { uint64_t jobs[4]{}; };
    struct Lru {
        std::array<uint16_t,N> prev{},next{};
        std::array<uint8_t,N> in{};
        unsigned head=Nil,tail=Nil;
        uint64_t bytes=0,hits=0,misses=0,evictions=0,hit_tokens=0;
    };
    std::array<Entry,N> entries{};
    std::array<Lru,4> cache{};
    std::array<int,48> formats{};
    uint64_t jobs=0,tokens=0,unique=0,retouches=0,unsupported=0;
    uint64_t format_nt[4][4]{}; // 18,22,21,other; job nt1..4
    bool invalid=false,overflow=false;
    uint64_t completed_jobs=0,completed_tokens=0;
    explicit CpuGuReuseObserver(const int* types) { for(unsigned i=0;i<48;++i) formats[i]=types[i]; }
    static uint64_t payload(int type) { return type==18||type==22 ? 2*640*10*266ULL : type==21 ? 2*640*10*262ULL : 0; }
    static unsigned group(int type) { return type==18?0:type==22?1:type==21?2:3; }
    bool add(uint64_t& dst,uint64_t value) {
        if (value>UINT64_MAX-dst) { invalid=overflow=true; return false; }
        dst+=value; return true;
    }
    void unlink(Lru& c,unsigned id) {
        unsigned p=c.prev[id],n=c.next[id];
        if(p==Nil)c.head=n;else c.next[p]=(uint16_t)n;
        if(n==Nil)c.tail=p;else c.prev[n]=(uint16_t)p;
    }
    void front(Lru& c,unsigned id) {
        c.prev[id]=Nil;c.next[id]=(uint16_t)c.head;
        if(c.head!=Nil)c.prev[c.head]=(uint16_t)id;else c.tail=id;
        c.head=id;
    }
    // Called once per completed grouping, ordered by actual job-array index.
    // No blob address enters this object or output. Unknown formats bypass LRU.
    void touch(int layer,int expert,int nt) {
        if(invalid)return;
        if(layer<0||layer>=48||expert<0||expert>=512||nt<1||nt>4){invalid=true;return;}
        const unsigned id=(unsigned)layer*512+(unsigned)expert;
        auto& e=entries[id]; bool seen=false;for(auto v:e.jobs)seen|=v!=0;
        if(!add(e.jobs[nt-1],1)||!add(jobs,1)||!add(tokens,(uint64_t)nt)||
           !add(seen?retouches:unique,1)||!add(format_nt[group(formats[layer])][nt-1],1))return;
        uint64_t size=payload(formats[layer]);if(!size){add(unsupported,1);return;}
        for(unsigned b=0;b<4;++b){auto& c=cache[b];
            if(c.in[id]){if(!add(c.hits,1)||!add(c.hit_tokens,(uint64_t)nt))return;unlink(c,id);}
            else {
                if(!add(c.misses,1))return;
                while(c.bytes>Budgets[b]-size){unsigned victim=c.tail;
                    if(victim==Nil){invalid=true;return;}
                    unlink(c,victim);c.in[victim]=0;
                    c.bytes-=payload(formats[victim/512]);if(!add(c.evictions,1))return;
                }
                c.in[id]=1;c.bytes+=size;
            }
            front(c,id);
        }
    }
    void completed(uint64_t count,uint64_t rows) { if(!invalid){add(completed_jobs,count);add(completed_tokens,rows);} }
    // Caller passes ledger counts only after its existing pool-parking boundary.
    bool conserved(uint64_t ledger_jobs,uint64_t ledger_rows) const {
        if(invalid||jobs!=ledger_jobs||tokens!=ledger_rows||completed_jobs!=jobs||completed_tokens!=tokens||unique+retouches!=jobs)return false;
        uint64_t ej=0,et=0,fg=0;
        for(const auto& e:entries)for(unsigned n=0;n<4;++n){if(e.jobs[n]>UINT64_MAX/(n+1)||e.jobs[n]>UINT64_MAX-ej||e.jobs[n]*(n+1)>UINT64_MAX-et)return false;ej+=e.jobs[n];et+=e.jobs[n]*(n+1);}
        for(auto& f:format_nt)for(auto v:f){if(v>UINT64_MAX-fg)return false;fg+=v;}
        if(ej!=jobs||et!=tokens||fg!=jobs)return false;
        for(unsigned b=0;b<4;++b)if(cache[b].hits+cache[b].misses!=jobs-unsupported||cache[b].bytes>Budgets[b])return false;
        return true;
    }
    struct Out {
        char* p;size_t used=0;bool ok=true;
        template<class... A> void line(const char* f,A... a){if(!ok)return;int n=std::snprintf(p+used,OutputCap-used,f,a...);if(n<0||(size_t)n>=OutputCap-used){ok=false;return;}used+=(size_t)n;}
        void byte(char c){if(!ok)return;if(used+1>=OutputCap){ok=false;return;}p[used++]=c;}
    };
    // Dense base64(ULEB128(job counts)) keeps the zero-heavy complete domain compact.
    // On any capacity failure discard the entire detailed output; never a partial hit claim.
    size_t receipt(char* dst,uint64_t request,uint64_t ledger_jobs,uint64_t ledger_rows,bool ledger_valid) const {
        Out o{dst};bool complete=ledger_valid&&conserved(ledger_jobs,ledger_rows);
        o.line("strata cpu_gu_reuse_v1 request=%llu complete=%d overflow=%d lifetime=request_cold budgets_ignore_live_ram=1 ram_floor_bytes=8589934592 admission_cap_bytes=8589934592 jobs=%llu tokens=%llu first_touches_unique=%llu retouches=%llu unsupported=%llu completed_jobs=%llu completed_tokens=%llu\n",(unsigned long long)request,complete,overflow,(unsigned long long)jobs,(unsigned long long)tokens,(unsigned long long)unique,(unsigned long long)retouches,(unsigned long long)unsupported,(unsigned long long)completed_jobs,(unsigned long long)completed_tokens);
        for(unsigned b=0;b<4;++b){auto& c=cache[b];o.line("lru budget=%llu hits=%llu misses=%llu hit_tokens=%llu evictions=%llu resident_payload_bytes=%llu\n",(unsigned long long)Budgets[b],(unsigned long long)c.hits,(unsigned long long)c.misses,(unsigned long long)c.hit_tokens,(unsigned long long)c.evictions,(unsigned long long)c.bytes);}
        for(unsigned f=0;f<4;++f)for(unsigned n=0;n<4;++n)o.line("format_nt group=%u nt=%u jobs=%llu tokens=%llu\n",f,n+1,(unsigned long long)format_nt[f][n],(unsigned long long)(format_nt[f][n]*(n+1)));
        o.line("formats");for(int f:formats)o.line(" %d",f);o.line("\nexpert_jobs_uleb128_base64 ");
        static constexpr char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const size_t start=o.used;
        unsigned acc=0,bits=0;
        auto raw=[&](unsigned x){acc=(acc<<8)|x;bits+=8;while(bits>=6){bits-=6;o.byte(alphabet[(acc>>bits)&63]);}};
        for(const auto& e:entries)for(uint64_t v:e.jobs){do{unsigned x=(unsigned)(v&127);v>>=7;raw(x|(v?128:0));}while(v);}
        if(bits)o.byte(alphabet[(acc<<(6-bits))&63]);
        // Standard base64 padding (encoded bytes modulo 3 inferred from output chars).
        while(o.ok&&(o.used-start)%4) o.byte('=');
        o.byte('\n');
        if(!o.ok){o={dst};o.line("strata cpu_gu_reuse_v1 request=%llu complete=0 reason=output_cap cap=262144\n",(unsigned long long)request);}
        return o.used;
    }
};
}
