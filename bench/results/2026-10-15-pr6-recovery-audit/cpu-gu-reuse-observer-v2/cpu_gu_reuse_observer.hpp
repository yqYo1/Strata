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
        uint64_t fill_payload_bytes=0,evicted_payload_bytes=0;
        uint64_t hits_format_nt[3][4]{},misses_format_nt[3][4]{};
        uint64_t resident_format[3]{};
    };
    std::array<Entry,N> entries{};
    // policy 0: every supported job; policy 1: supported nt1 jobs only.
    std::array<std::array<Lru,4>,2> cache{};
    uint64_t unsupported_nt[4]{};
    uint64_t bypassed_nt[2][4]{}; // supported format, excluded by policy; disjoint from unsupported
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
        uint64_t size=payload(formats[layer]);
        if(!size){add(unsupported,1);add(unsupported_nt[nt-1],1);return;}
        const unsigned f=group(formats[layer]);
        for(unsigned policy=0;policy<2;++policy) {
          if(policy==1 && nt!=1){if(!add(bypassed_nt[policy][nt-1],1))return;continue;}
          for(unsigned b=0;b<4;++b){auto& c=cache[policy][b];
            if(c.in[id]){if(!add(c.hits,1)||!add(c.hit_tokens,(uint64_t)nt)||!add(c.hits_format_nt[f][nt-1],1))return;unlink(c,id);}
            else {
                if(!add(c.misses,1)||!add(c.misses_format_nt[f][nt-1],1)||!add(c.fill_payload_bytes,size))return;
                while(c.bytes>Budgets[b]-size){unsigned victim=c.tail;
                    if(victim==Nil){invalid=true;return;}
                    unlink(c,victim);c.in[victim]=0;
                    const uint64_t victim_size=payload(formats[victim/512]);
                    const unsigned vf=group(formats[victim/512]);
                    if(!c.resident_format[vf]){invalid=true;return;}
                    --c.resident_format[vf];c.bytes-=victim_size;
                    if(!add(c.evictions,1)||!add(c.evicted_payload_bytes,victim_size))return;
                }
                c.in[id]=1;c.bytes+=size;if(!add(c.resident_format[f],1))return;
            }
            front(c,id);
          }
        }
    }
    void completed(uint64_t count,uint64_t rows) { if(!invalid){add(completed_jobs,count);add(completed_tokens,rows);} }
    // Caller passes ledger counts only after its existing pool-parking boundary.
    bool conserved(uint64_t ledger_jobs,uint64_t ledger_rows) const {
        if(invalid||overflow||jobs!=ledger_jobs||tokens!=ledger_rows||completed_jobs!=jobs||completed_tokens!=tokens||unique+retouches!=jobs)return false;
        uint64_t ej=0,et=0,fg=0;
        for(const auto& e:entries)for(unsigned n=0;n<4;++n){if(e.jobs[n]>UINT64_MAX/(n+1)||e.jobs[n]>UINT64_MAX-ej||e.jobs[n]*(n+1)>UINT64_MAX-et)return false;ej+=e.jobs[n];et+=e.jobs[n]*(n+1);}
        for(auto& f:format_nt)for(auto v:f){if(v>UINT64_MAX-fg)return false;fg+=v;}
        if(ej!=jobs||et!=tokens||fg!=jobs)return false;
        uint64_t un=0;
        for(unsigned n=0;n<4;++n) {
            if(unsupported_nt[n]!=format_nt[3][n] || unsupported_nt[n]>UINT64_MAX-un)return false;
            un+=unsupported_nt[n];
        }
        if(un!=unsupported)return false;
        for(unsigned policy=0;policy<2;++policy) {
            uint64_t bypass=0,eligible=0;
            for(unsigned n=0;n<4;++n) {
                uint64_t supported=0;for(unsigned f=0;f<3;++f){if(format_nt[f][n]>UINT64_MAX-supported)return false;supported+=format_nt[f][n];}
                const uint64_t excluded=policy==1&&n!=0?supported:0;
                if(bypassed_nt[policy][n]!=excluded || excluded>UINT64_MAX-bypass)return false;
                bypass+=excluded;
                if(supported-excluded>UINT64_MAX-eligible)return false;
                eligible+=supported-excluded;
            }
            if(eligible>jobs || unsupported>jobs-eligible || bypass!=jobs-eligible-unsupported)return false;
            for(unsigned b=0;b<4;++b){const auto& c=cache[policy][b];
                if(c.hits>eligible || c.misses!=eligible-c.hits || c.bytes>Budgets[b] ||
                   c.fill_payload_bytes<c.evicted_payload_bytes || c.fill_payload_bytes-c.evicted_payload_bytes!=c.bytes)return false;
                uint64_t h=0,m=0,ht=0,fill=0,resident=0,resident_count=0;
                for(unsigned f=0;f<3;++f){const int type=f==0?18:f==1?22:21;const uint64_t size=payload(type);
                    if(c.resident_format[f]>N || c.resident_format[f]>UINT64_MAX-resident_count || c.resident_format[f]>UINT64_MAX/size || c.resident_format[f]*size>UINT64_MAX-resident)return false;
                    resident+=c.resident_format[f]*size;resident_count+=c.resident_format[f];
                    for(unsigned n=0;n<4;++n){const uint64_t fh=c.hits_format_nt[f][n],fm=c.misses_format_nt[f][n];
                        const uint64_t expected=policy==1&&n!=0?0:format_nt[f][n];
                        if(fh>expected || fm!=expected-fh || fh>UINT64_MAX-h || fm>UINT64_MAX-m ||
                           fh>UINT64_MAX/(n+1) || fh*(n+1)>UINT64_MAX-ht || fm>UINT64_MAX/size || fm*size>UINT64_MAX-fill)return false;
                        h+=fh;m+=fm;ht+=fh*(n+1);fill+=fm*size;
                    }
                }
                if(c.evictions>c.misses || resident_count!=c.misses-c.evictions || h!=c.hits || m!=c.misses || ht!=c.hit_tokens || fill!=c.fill_payload_bytes || resident!=c.bytes)return false;
            }
        }
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
        o.line("strata cpu_gu_reuse_v2 request=%llu complete=%d overflow=%d lifetime=request_cold budgets_ignore_live_ram=1 ram_floor_bytes=8589934592 admission_cap_bytes=8589934592 state_bytes=%zu jobs=%llu tokens=%llu first_touches_unique=%llu retouches=%llu unsupported=%llu completed_jobs=%llu completed_tokens=%llu\n",(unsigned long long)request,complete,overflow,sizeof(*this),(unsigned long long)jobs,(unsigned long long)tokens,(unsigned long long)unique,(unsigned long long)retouches,(unsigned long long)unsupported,(unsigned long long)completed_jobs,(unsigned long long)completed_tokens);
        for(unsigned policy=0;policy<2;++policy) {
            for(unsigned n=0;n<4;++n)o.line("policy_skipped policy=%u nt=%u unsupported=%llu bypassed_supported_nt=%llu\n",policy,n+1,(unsigned long long)unsupported_nt[n],(unsigned long long)bypassed_nt[policy][n]);
            for(unsigned b=0;b<4;++b){const auto& c=cache[policy][b];
                o.line("lru policy=%u policy_name=%s budget=%llu expanded18_bytes=3404800 expanded22_bytes=3404800 expanded21_bytes=3353600 hits=%llu misses=%llu fill_payload_bytes=%llu hit_tokens=%llu evictions=%llu evicted_payload_bytes=%llu resident_payload_bytes=%llu\n",policy,policy?"nt1_only":"all_supported",(unsigned long long)Budgets[b],(unsigned long long)c.hits,(unsigned long long)c.misses,(unsigned long long)c.fill_payload_bytes,(unsigned long long)c.hit_tokens,(unsigned long long)c.evictions,(unsigned long long)c.evicted_payload_bytes,(unsigned long long)c.bytes);
                for(unsigned f=0;f<3;++f)for(unsigned n=0;n<4;++n)
                    o.line("lru_format_nt policy=%u budget=%llu group=%u nt=%u hits=%llu misses=%llu\n",policy,(unsigned long long)Budgets[b],f,n+1,(unsigned long long)c.hits_format_nt[f][n],(unsigned long long)c.misses_format_nt[f][n]);
            }
        }
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
        if(!o.ok){o={dst};o.line("strata cpu_gu_reuse_v2 request=%llu complete=0 reason=output_cap cap=262144\n",(unsigned long long)request);}
        return o.used;
    }
};
static_assert(sizeof(CpuGuReuseObserver)<=2621440,"observer state must remain <=2.5 MiB");
}
