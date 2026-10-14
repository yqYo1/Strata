from pathlib import Path
import hashlib,json,difflib
root=Path.cwd(); out=root/'bench/results/2026-10-15-pr6-recovery-audit/decode-format-service-v1'
paths=['include/strata/core/decode_route_ledger.hpp','src/core/expert_source.cpp','include/strata/kernels/cpu/service_budget.hpp','src/kernels/cpu/pool.cpp']
pins={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
h=(root/paths[0]).read_text(); s=(root/paths[1]).read_text()
insert='''    // Issuer-only aggregation. No worker writer, clock or atomic is added.
    struct CompactService { uint64_t calls=0,cpu_ns=0,wall_ns=0; };
    struct FormatService { CompactService group[2][6]{}; int layer=-1; } format_service[16];
    CompactService service_before[2][6]{};
    int format_count=0,service_pair=-1;
    uint64_t service_dispatches=0;
    bool service_pending=false,format_invalid=false,format_overflow=false;
    void format_add(uint64_t& dst,uint64_t n) {
        if (n>UINT64_MAX-dst) { dst=UINT64_MAX; format_overflow=true; } else dst+=n;
    }
    bool service_snapshot(CompactService (&dst)[2][6]) {
        if (workers<1 || workers>128 || service.invalid.load(std::memory_order_relaxed) || service.overflow) return false;
        for (int g=0;g<2;++g) for (int r=0;r<6;++r) {
            strata::kernels::cpu::ServiceCounter c;
            for (int i=g?1:0;i<=(g?workers:0);++i) {
                if (service.slots[i].overflow) return false;
                c.merge(service.slots[i].call[r]);
            }
            if (!c.complete()) return false;
            dst[g][r]={c.calls,c.cpu_ns,c.wall_ns};
        }
        return true;
    }
    // First binding owns fresh zero counters; later calls follow the previous
    // native run's done + acquire wait_parked return and detached publication.
    void format_service_begin(int64_t l,const strata::kernels::cpu::NativeFmt& actual) {
        if (service_pending || l<0 || l>=layers || !same_format(fmt[l],actual)) { format_invalid=true; return; }
        int p=0;
        for (;p<format_count;++p) if (same_format(fmt[format_service[p].layer],actual)) break;
        if (p==format_count) {
            if (format_count==16) { format_invalid=true; return; }
            format_service[p].layer=(int)l; ++format_count;
        }
        if (!service_snapshot(service_before)) { format_invalid=true; return; }
        service_pair=p; service_pending=true;
    }
    // Caller invokes only after native return through existing final parked wait.
    void format_service_end() {
        if (!service_pending || service_pair<0 || service_pair>=format_count) { format_invalid=true; return; }
        CompactService now[2][6]{};
        if (!service_snapshot(now)) { format_invalid=true; return; }
        for (int g=0;g<2;++g) for (int r=0;r<6;++r) {
            const auto& a=service_before[g][r]; const auto& b=now[g][r];
            if (b.calls<a.calls || b.cpu_ns<a.cpu_ns || b.wall_ns<a.wall_ns) { format_invalid=true; continue; }
            auto& c=format_service[service_pair].group[g][r];
            format_add(c.calls,b.calls-a.calls); format_add(c.cpu_ns,b.cpu_ns-a.cpu_ns); format_add(c.wall_ns,b.wall_ns-a.wall_ns);
        }
        format_add(service_dispatches,1); service_pending=false; service_pair=-1;
    }
    bool format_service_complete() const {
        if (format_invalid || format_overflow || service_pending || service_dispatches!=dispatch_attempts || format_count<1 || format_count>16) return false;
        for (int g=0;g<2;++g) for (int r=0;r<6;++r) {
            strata::kernels::cpu::ServiceCounter c;
            for (int i=g?1:0;i<=(g?workers:0);++i) c.merge(service.slots[i].call[r]);
            if (!c.complete()) return false;
            uint64_t calls=0,cpu=0,wall=0;
            for (int p=0;p<format_count;++p) {
                const auto& b=format_service[p].group[g][r];
                if (b.calls>UINT64_MAX-calls || b.cpu_ns>UINT64_MAX-cpu || b.wall_ns>UINT64_MAX-wall) return false;
                calls+=b.calls; cpu+=b.cpu_ns; wall+=b.wall_ns;
            }
            if (calls!=c.calls || cpu!=c.cpu_ns || wall!=c.wall_ns) return false;
        }
        return true;
    }
'''
h=h.replace('    struct Layer {',insert+'    struct Layer {',1)
h=h.replace('!service.complete(workers)) return false;','!service.complete(workers) || !format_service_complete()) return false;',1)
marker='        for (int ph=0;ph<2;++ph) {\n            uint64_t claimed'
receipt='''        for (int p=0;p<format_count && p<16;++p) {
            const auto& f=fmt[format_service[p].layer];
            out.append(false,"service_format pair=%d representative_layer=%d gu=%d down=%d gu_act=%d down_act=%d gu_row=%zu down_row=%zu act_bytes=%zu h_bytes=%zu\\n",p,format_service[p].layer,f.gu_type,f.d_type,f.gu_act,f.d_act,f.gu_row,f.d_row,f.act_bytes,f.h_bytes);
            for (int g=0;g<2;++g) for (int r=0;r<6;++r) {
                const auto& c=format_service[p].group[g][r];
                if (c.calls) out.append(false,"service_format_counter pair=%d group=%s role=%d calls=%llu cpu_ns=%llu wall_ns=%llu\\n",p,g?"workers":"issuer",r,(unsigned long long)c.calls,(unsigned long long)c.cpu_ns,(unsigned long long)c.wall_ns);
            }
        }
        out.append(false,"service_format_close complete=%d invalid=%d overflow=%d pending=%d dispatches=%llu pairs=%d\\n",format_service_complete(),format_invalid,format_overflow,service_pending,(unsigned long long)service_dispatches,format_count);
'''
h=h.replace(marker,receipt+marker,1)
s=s.replace('            decode_observer->service.actual_t=(int)n_tok;','            decode_observer->format_service_begin(d.layers,lay.fmt[(size_t)d.layers]);\n            decode_observer->service.actual_t=(int)n_tok;',1)
s=s.replace('            decode_observer->pool_bound=false;','            decode_observer->pool_bound=false;\n            decode_observer->format_service_end();',1)
patch=''
for p,new in [(paths[0],h),(paths[1],s)]:
 old=(root/p).read_text(); patch+=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+p,tofile='b/'+p))
(out/'source.diff').write_text(patch)
(out/'pins.json').write_text(json.dumps({'status':'proposal_only_unapplied_unbuilt_untested','source_base_label_from_main':'current 5fbc/5183 tree; same source expected; no Git identity queried','source_sha256':pins,'source_diff_sha256':hashlib.sha256(patch.encode()).hexdigest()},indent=2)+'\n')
