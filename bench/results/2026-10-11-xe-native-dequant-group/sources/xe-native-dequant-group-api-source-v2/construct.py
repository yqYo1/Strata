from pathlib import Path
import hashlib, json, difflib
b=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
r=b/'xe-native-dequant-group-api-source-v2'
w=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
f='src/kernels/xe/iq_kernels.cpp'
h='include/strata/kernels/iq_kernels.hpp'
base=(b/'xe-native-dequant-group-api-source-v1/replacement'/f).read_bytes()
header=(b/'xe-native-dequant-group-api-source-v1/replacement'/h).read_bytes()
sha=lambda x:hashlib.sha256(x).hexdigest()
if sha(base)!='7a109e4123bcd6c6cdf8b11ff411657eaf9cd8ed81b0d7f4fdab59285461cdf7':raise RuntimeError('immutable base pin mismatch')
if sha(header)!='61513ccb71831d1065bd25829541c508adf3ae58e1e3df6d3bc0c9681cb54f6d':raise RuntimeError('immutable header pin mismatch')
if (w/f).read_bytes()!=base or (w/h).read_bytes()!=header:raise RuntimeError('live IQ bytes differ from delegated v1')
helper='''// Constant subscripts keep the captured pointer table available for scalar replacement. Whether the compiler
// actually removes private-array traffic is an IR / component qualification question, not a source guarantee.
// Pass the table by reference: this helper does not make another by-value array copy.
#if defined(__clang__) || defined(__GNUC__)
__attribute__((always_inline))
#endif
inline const void* dequant_group_pointer(const void* const (&p)[kIqDequantGroup], int expert) {
    switch (expert) {
        case 0: return p[0];
        case 1: return p[1];
        case 2: return p[2];
        case 3: return p[3];
        case 4: return p[4];
        case 5: return p[5];
        case 6: return p[6];
        case 7: return p[7];
        case 8: return p[8];
        case 9: return p[9];
        case 10: return p[10];
        case 11: return p[11];
        case 12: return p[12];
        case 13: return p[13];
        case 14: return p[14];
        case 15: return p[15];
        default: return nullptr;   // unreachable after the host slot-bounds check and the expert ND-range
    }
}

'''
s=base.decode()
anchor='// All group checks precede queue submission; no allocation, queue or synchronization is introduced.\n'
oldgu='dq_dispatch(t, parity ? g.up[expert] : g.gate[expert], i,'
newgu='dq_dispatch(t, parity ? dequant_group_pointer(g.up, expert) : dequant_group_pointer(g.gate, expert), i,'
olddown='dq_dispatch(t, g.down[expert], i,'
newdown='dq_dispatch(t, dequant_group_pointer(g.down, expert), i,'
for old in [anchor,oldgu,olddown]:
    if s.count(old)!=1:raise RuntimeError('anchor not unique')
new=s.replace(anchor,helper+anchor,1).replace(oldgu,newgu,1).replace(olddown,newdown,1)
recovered=new.replace(helper,'',1).replace(newgu,oldgu,1).replace(newdown,olddown,1)
if recovered.encode()!=base:raise RuntimeError('reverse substitution mismatch')
for sub,data in [('base/'+f,base),('base/'+h,header),('replacement/'+f,new.encode())]:
    p=r/sub;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
patch=''.join(difflib.unified_diff(s.splitlines(True),new.splitlines(True),fromfile='a/'+f,tofile='b/'+f))
(r/'candidate.patch').write_text(patch)
deps=[w/'AGENTS.md',Path('/home/yayoi/orca/workspaces/Strata/sycl/docs/ARTIFACT_RETENTION.md'),b/'xe-native-dequant-group-component-gpu-root-v1-free-full-host/record.json',b/'research-20261009/implementation-xe-native-dequant-group-api-v1.txt',b/'research-20261009/implementation-xe-native-dequant-group-byte-probe-v1.txt',b/'research-20261009/implementation-xe-native-dequant-prefill-batch-v1.txt',b/'research-20261009/implementation-xe-native-dequant-group-component-v1.txt',Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009/bench/results/2026-10-11-xe-prefill-route-census32k/report-registry-v127.json')]
record=json.loads(deps[2].read_bytes())
medians=[]
for gu,down in [(18,20),(18,42),(21,20),(21,42),(22,20),(22,42),(23,20)]:
    arms={}
    for arm in ['individual','group']:
        rows=[x for x in record['samples'] if x['gu_type']==gu and x['down_type']==down and x['n']==16 and x['arm']==arm]
        med=lambda key:(sorted(x[key] for x in rows)[2]+sorted(x[key] for x in rows)[3])/2
        arms[arm]={'host_submit_ms':med('host_submit_ms'),'host_submit_checked_drain_ms':med('host_submit_checked_drain_ms'),'samples':len(rows),'repetitions':4}
    medians.append({'gu_type':gu,'down_type':down,'n':16,'arms':arms,'individual_over_group_checked_drain':arms['individual']['host_submit_checked_drain_ms']/arms['group']['host_submit_checked_drain_ms']})
(r/'v1-context-medians.json').write_text(json.dumps({'source_record':str(deps[2]),'source_sha256':sha(deps[2].read_bytes()),'scope':'read/paraphrase of Main-owned previous v1 results; no new execution','medians':medians},indent=2)+'\n')
m={'version':2,'qualification':'SOURCE ONLY; compiler/IR/parity/component unexecuted','implementation_model':'gpt-6.1-sol','context_HEAD_supplied':'b3e1de06076b692614ef51512f5501838d4b6f2b','base_identity':'v1 full replacement; applied live IQ bytes match at text-construction boundary, not clean-tree claim','files':[{'path':f,'base_sha256':sha(base),'base_bytes':len(base),'replacement_sha256':sha(new.encode()),'replacement_bytes':len(new.encode()),'reverse_exact_substitutions_recover_v1':True},{'path':h,'sha256':sha(header),'bytes':len(header),'unchanged':True,'replacement_needed':False}],'patch_sha256':sha(patch.encode()),'preserved':'all admission checks, API declarations, geometry, decoder/math, old individual functions, stream contracts','local_md_present':(w/'LOCAL.md').exists(),'dependencies':[{'path':str(p),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}for p in deps]}
(r/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(m['files'],indent=2)); print('patch_sha256',m['patch_sha256'])
