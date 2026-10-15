# Historical one-shot authoring transform retained for provenance, not a test.
# The final reviewed header is authoritative. This transform is not idempotent.
raise SystemExit("Do not rerun: use write_proposal.py to regenerate diff/pins from the final header")
from pathlib import Path
p=Path('/home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/cpu-gu-reuse-observer-v2/cpu_gu_reuse_observer.hpp')
s=p.read_text()
s=s.replace('        uint64_t bytes=0,hits=0,misses=0,evictions=0,hit_tokens=0;', '''        uint64_t bytes=0,hits=0,misses=0,evictions=0,hit_tokens=0;
        uint64_t fill_payload_bytes=0,evicted_payload_bytes=0;
        uint64_t hits_format_nt[3][4]{},misses_format_nt[3][4]{};
        uint64_t resident_format[3]{};''')
s=s.replace('    std::array<Lru,4> cache{};', '''    // policy 0: every supported job; policy 1: supported nt1 jobs only.
    std::array<std::array<Lru,4>,2> cache{};
    uint64_t unsupported_nt[4]{};
    uint64_t bypassed_nt[2][4]{}; // supported format, excluded by policy; disjoint from unsupported''')
s=s.replace('uint64_t size=payload(formats[layer]);if(!size){add(unsupported,1);return;}\n        for(unsigned b=0;b<4;++b){auto& c=cache[b];', '''uint64_t size=payload(formats[layer]);
        if(!size){add(unsupported,1);add(unsupported_nt[nt-1],1);return;}
        const unsigned f=group(formats[layer]);
        for(unsigned policy=0;policy<2;++policy) {
          if(policy==1 && nt!=1){if(!add(bypassed_nt[policy][nt-1],1))return;continue;}
          for(unsigned b=0;b<4;++b){auto& c=cache[policy][b];''')
s=s.replace('if(c.in[id]){if(!add(c.hits,1)||!add(c.hit_tokens,(uint64_t)nt))return;unlink(c,id);}', 'if(c.in[id]){if(!add(c.hits,1)||!add(c.hit_tokens,(uint64_t)nt)||!add(c.hits_format_nt[f][nt-1],1))return;unlink(c,id);}')
s=s.replace('                if(!add(c.misses,1))return;', '                if(!add(c.misses,1)||!add(c.misses_format_nt[f][nt-1],1)||!add(c.fill_payload_bytes,size))return;')
s=s.replace('                    c.bytes-=payload(formats[victim/512]);if(!add(c.evictions,1))return;', '''                    const uint64_t victim_size=payload(formats[victim/512]);
                    const unsigned vf=group(formats[victim/512]);
                    if(!c.resident_format[vf]){invalid=true;return;}
                    --c.resident_format[vf];c.bytes-=victim_size;
                    if(!add(c.evictions,1)||!add(c.evicted_payload_bytes,victim_size))return;''')
s=s.replace('                c.in[id]=1;c.bytes+=size;', '                c.in[id]=1;c.bytes+=size;if(!add(c.resident_format[f],1))return;')
s=s.replace('            front(c,id);\n        }\n    }', '            front(c,id);\n          }\n        }\n    }')
a=s.index('        for(unsigned b=0;b<4;++b)if(cache[b]')
b=s.index('        return true;',a)
s=s[:a]+'''        uint64_t un=0;
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
                if(supported-excluded>UINT64_MAX-eligible)return false;eligible+=supported-excluded;
            }
            if(eligible>jobs || unsupported>jobs-eligible || bypass!=jobs-eligible-unsupported)return false;
            for(unsigned b=0;b<4;++b){const auto& c=cache[policy][b];
                if(c.hits>eligible || c.misses!=eligible-c.hits || c.bytes>Budgets[b] ||
                   c.fill_payload_bytes<c.evicted_payload_bytes || c.fill_payload_bytes-c.evicted_payload_bytes!=c.bytes)return false;
                uint64_t h=0,m=0,ht=0,fill=0,resident=0;
                for(unsigned f=0;f<3;++f){const int type=f==0?18:f==1?22:21;const uint64_t size=payload(type);
                    if(c.resident_format[f]>UINT64_MAX/size || c.resident_format[f]*size>UINT64_MAX-resident)return false;
                    resident+=c.resident_format[f]*size;
                    for(unsigned n=0;n<4;++n){const uint64_t fh=c.hits_format_nt[f][n],fm=c.misses_format_nt[f][n];
                        const uint64_t expected=policy==1&&n!=0?0:format_nt[f][n];
                        if(fh>expected || fm!=expected-fh || fh>UINT64_MAX-h || fm>UINT64_MAX-m ||
                           fh>UINT64_MAX/(n+1) || fh*(n+1)>UINT64_MAX-ht || fm>UINT64_MAX/size || fm*size>UINT64_MAX-fill)return false;
                        h+=fh;m+=fm;ht+=fh*(n+1);fill+=fm*size;
                    }
                }
                if(h!=c.hits || m!=c.misses || ht!=c.hit_tokens || fill!=c.fill_payload_bytes || resident!=c.bytes)return false;
            }
        }
'''+s[b:]
s=s.replace('cpu_gu_reuse_v1','cpu_gu_reuse_v2')
a=s.index('        for(unsigned b=0;b<4;++b){auto& c=cache[b];o.line')
b=s.index('        for(unsigned f=0;f<4;++f)for(unsigned n=0;n<4;++n)o.line',a)
s=s[:a]+'''        for(unsigned policy=0;policy<2;++policy) {
            for(unsigned n=0;n<4;++n)o.line("policy_skipped policy=%u nt=%u unsupported=%llu bypassed_supported_nt=%llu\\n",policy,n+1,(unsigned long long)unsupported_nt[n],(unsigned long long)bypassed_nt[policy][n]);
            for(unsigned b=0;b<4;++b){const auto& c=cache[policy][b];
                o.line("lru policy=%u budget=%llu expanded18_bytes=3404800 expanded22_bytes=3404800 expanded21_bytes=3353600 hits=%llu misses=%llu fill_payload_bytes=%llu hit_tokens=%llu evictions=%llu evicted_payload_bytes=%llu resident_payload_bytes=%llu\\n",policy,(unsigned long long)Budgets[b],(unsigned long long)c.hits,(unsigned long long)c.misses,(unsigned long long)c.fill_payload_bytes,(unsigned long long)c.hit_tokens,(unsigned long long)c.evictions,(unsigned long long)c.evicted_payload_bytes,(unsigned long long)c.bytes);
                for(unsigned f=0;f<3;++f)for(unsigned n=0;n<4;++n)
                    o.line("lru_format_nt policy=%u budget=%llu group=%u nt=%u hits=%llu misses=%llu\\n",policy,(unsigned long long)Budgets[b],f,n+1,(unsigned long long)c.hits_format_nt[f][n],(unsigned long long)c.misses_format_nt[f][n]);
            }
        }
'''+s[b:]
s=s.replace('admission_cap_bytes=8589934592 jobs=', 'admission_cap_bytes=8589934592 state_bytes=%zu jobs=')
s=s.replace('(unsigned long long)request,complete,overflow,(unsigned long long)jobs', '(unsigned long long)request,complete,overflow,sizeof(*this),(unsigned long long)jobs')
s=s.replace('};\n}', '};\nstatic_assert(sizeof(CpuGuReuseObserver)<=2621440,"observer state must remain <=2.5 MiB");\n}')
p.write_text(s)
