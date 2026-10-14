from pathlib import Path
import difflib,json,hashlib
root=Path.cwd(); out=root/'bench/results/2026-10-15-pr6-recovery-audit/decode-format-service-v2'
# Reuse the archived generator in memory; never execute its archival writes.
script=(out.parent/'decode-format-service-v1/make_proposal.py').read_text()
script=script[:script.index("patch=''" )]
script=script.replace("out=root/'bench/results/2026-10-15-pr6-recovery-audit/decode-format-service-v1'", "out=root/'bench/results/2026-10-15-pr6-recovery-audit/decode-format-service-v2'")
ns={}; exec(compile(script,'v1-proposal-generator','exec'),ns)
h,s=ns['h'],ns['s']
h=h.replace('    void format_service_begin(int64_t l,const strata::kernels::cpu::NativeFmt& actual) {\n        if (service_pending || l<0 || l>=layers || !same_format(fmt[l],actual))', '''    bool format_representatives_valid() const {
        if (!envelope() || format_count<0 || format_count>16) return false;
        for (int p=0;p<format_count;++p) {
            const int l=format_service[p].layer;
            if (l<0 || l>=layers || l>=kLayers || !valid_format(fmt[l])) return false;
        }
        return true;
    }
    void format_service_begin(int64_t l,const strata::kernels::cpu::NativeFmt& actual) {
        if (!envelope() || !format_representatives_valid() || service_pending || l<0 || l>=layers || l>=kLayers || !valid_format(actual) || !same_format(fmt[l],actual))''')
h=h.replace('if (!service_pending || service_pair<0 || service_pair>=format_count)', 'if (!format_representatives_valid() || !service_pending || service_pair<0 || service_pair>=format_count)')
h=h.replace('if (format_invalid || format_overflow || service_pending || service_dispatches!=dispatch_attempts || format_count<1 || format_count>16)', 'if (!format_representatives_valid() || format_invalid || format_overflow || service_pending || service_dispatches!=dispatch_attempts || format_count<1)')
h=h.replace('schema=mtp4-route-service-v2 request=%llu coupled=%d input_sha256=', 'schema=mtp4-route-service-v2 service_format=1 request=%llu coupled=%d input_sha256=')
h=h.replace('        for (int p=0;p<format_count && p<16;++p) {\n            const auto& f=fmt[format_service[p].layer];', '''        if (!format_representatives_valid()) out.bounded=false;
        for (int p=0;p<format_count && p<16;++p) {
            const int representative=format_service[p].layer;
            if (layers<1 || layers>kLayers || representative<0 || representative>=layers || representative>=kLayers) {
                out.bounded=false; continue;
            }
            const auto& f=fmt[representative];
            if (!valid_format(f)) { out.bounded=false; continue; }''')
p='tests/cpu/decode_joined_mtp4.cpp'; t=(root/p).read_text()
t=t.replace('        if (!all_gpu) pool_dispatch(value,value.fmt[l],T);', '        value.format_service_begin(l,value.fmt[l]);\n        if (!all_gpu) pool_dispatch(value,value.fmt[l],T);\n        value.format_service_end();',1)
t=t.replace('    assert(complete(*value));\n    auto variable=', '    assert(complete(*value));\n    assert(value->format_service_complete() && value->service_dispatches==8 && value->format_count==2);\n    auto variable=',1)
t=t.replace('    assert(zero->complete(true,1,0,0));', '    assert(zero->complete(true,1,0,0));\n    assert(zero->format_service_complete() && zero->service_dispatches==2);',1)
patch=''
for p,new in [('include/strata/core/decode_route_ledger.hpp',h),('src/core/expert_source.cpp',s),('tests/cpu/decode_joined_mtp4.cpp',t)]:
 old=(root/p).read_text(); patch+=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+p,tofile='b/'+p))
(out/'source.diff').write_text(patch)
paths=ns['paths']+['tests/cpu/decode_joined_mtp4.cpp']
(out/'pins.json').write_text(json.dumps({'status':'proposal_only_unapplied_unbuilt_untested','head_label_from_main':'e0eb1622; product source bytes same as 5fbc; no Git queried','source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},'source_diff_sha256':hashlib.sha256(patch.encode()).hexdigest()},indent=2)+'\n')
