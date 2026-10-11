from pathlib import Path
import hashlib,json,difflib
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
A=B/'xe-native-dequant-group-v2-component-controller-source-v1'
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
WS=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
pins={'src/kernels/xe/iq_kernels.cpp':'75f2ac1dfdb337c424bba19e62c34159a1bde547a776b9b5d6cf7975dfc29d50','include/strata/kernels/iq_kernels.hpp':'61513ccb71831d1065bd25829541c508adf3ae58e1e3df6d3bc0c9681cb54f6d','src/prefill/prefill.cpp':'1b30e3de22b30f4eb2160756c9fdde623abe87a2360fe029f0767be33aa9d664'}
def once(t,x,y):
 assert t.count(x)==1,(x,t.count(x))
 return t.replace(x,y,1)
outputs={}
for kind in ['build','gpu']:
 old=B/f'xe-native-dequant-group-component-{kind}-v1.py';t=old.read_text();original=t
 for x,y in [('component-build-root-v1','component-build-root-v2'),('component-gpu-root-v1','component-gpu-root-v2'),('full-build-root-v17','full-build-root-v18'),('asan-full-build-root-v14','asan-full-build-root-v15'),('lints-baseline-compare-root-v12','lints-baseline-compare-root-v13'),('byte-gpu-root-v3','byte-gpu-root-v4')]:t=t.replace(x,y)
 t=once(t,"from pathlib import Path\n","from pathlib import Path\nEXPECTED_API_V2_PINS="+repr(pins)+"\n")
 if kind=='build':
  t=once(t,"source_base='b3e1de06076b692614ef51512f5501838d4b6f2b'","source_base='d73288cba04f728dadfd69e9797729c1a10af3ab',api_version=2")
  t=once(t,"  m.require(all(m.sha(W/f)==h for f,h in v['source_pins'].items())", "  m.require(all(v['source_pins'].get(f)==h for f,h in EXPECTED_API_V2_PINS.items()),'fresh exact constant-selector APIv2 release pins')\n  m.require(all(m.sha(W/f)==h for f,h in v['source_pins'].items())")
 else:
  t=once(t,"mode=a.mode,matrix=a.matrix,model_executed=False", "mode=a.mode,matrix=a.matrix,api_version=2,context_head='d73288cba04f728dadfd69e9797729c1a10af3ab',model_executed=False")
  t=once(t,"  r['source_pins']=proofs['release']['source_pins'];", "  m.require(all(proofs['release']['source_pins'].get(f)==h for f,h in EXPECTED_API_V2_PINS.items()),'fresh exact constant-selector APIv2 release pins')\n  m.require(proofs['probe_build'].get('api_version')==2 and proofs['probe_build']['mode']==a.mode,'fresh mode-specific APIv2 component build')\n  m.require(proofs['probe_build']['probe_source']['sha256']=='f9168793027bfd3aa62fd8c8ff607b98e9b25d9ece54fe31d54d1945439b89a7','unchanged Main corrected component source')\n  r['timing_interpretation']='Completed queue tag endpoint includes issuance gaps and end-tag work; not exclusive kernel time. Component only, no model throughput claim.'\n  r['probe_source']=proofs['probe_build']['probe_source']\n  r['source_pins']=proofs['release']['source_pins'];")
  t=once(t,"  r['source_stable']=all(m.sha(W/f)==h for f,h in r.get('source_pins',{}).items());", "  r['source_stable']=all(m.sha(W/f)==h for f,h in r.get('source_pins',{}).items()) and bool(r.get('probe_source') and m.sha(Path(r['probe_source']['path']))==r['probe_source']['sha256']);")
 target=A/f'{kind}_v2.py';assert not target.exists();target.write_text(t)
 patch=''.join(difflib.unified_diff(original.splitlines(True),t.splitlines(True),fromfile=str(old),tofile=str(target)))
 (A/f'{kind}_v2.patch').write_text(patch);outputs[str(target)]={'sha256':hashlib.sha256(t.encode()).hexdigest(),'bytes':len(t.encode())}
deps=[W/'AGENTS.md',WS/'docs/ARTIFACT_RETENTION.md',WS/'bench/results/2026-10-11-xe-native-dequant-group/report-registry-v128.json',B/'xe-native-dequant-group-component-build-v1.py',B/'xe-native-dequant-group-component-gpu-v1.py',B/'xe-native-dequant-group-component-source-v2/native_dequant_group_component.cpp',B/'xe-native-dequant-group-component-analysis-v1.json',B/'xe-native-dequant-group-optimized-ir-analysis-v1/analysis.json',B/'xe-native-dequant-group-component-gpu-root-v1-free-full-host/record.json',B/'xe-native-dequant-group-component-gpu-root-v1-free-full-tags/record.json',B/'xe-llvm-v711-source-v5/direct_owner.py',B/'xe-source-llvm-census32k-v3/runtime.py',B/'xe-source-llvm-census32k-root-v3/counts/record.json',B/'xe-source-llvm-small-health-root-v3/record.json',B/'xe-source-llvm-profiling-tags-root-v1/record.json',Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011/sycl/tools/recover-xe.sh')]
deps += [B/'research-20261009'/f'implementation-xe-native-dequant-group-{name}.txt' for name in ['api-v1','api-v2','byte-probe-v1','component-v1']]
deps += [B/'xe-native-dequant-group-api-source-v2/replacement/src/kernels/xe/iq_kernels.cpp',W/'include/strata/kernels/iq_kernels.hpp',W/'src/prefill/prefill.cpp',A/'construct.py',A/'build_v2.patch',A/'gpu_v2.patch']
identities=[]
for p in deps:
 data=p.read_bytes();identities.append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
manifest={'qualification':'source-only construction; no AST/import/build/test/lint/GPU execution','context_head':'d73288cba04f728dadfd69e9797729c1a10af3ab','expected_api_v2_pins':pins,'outputs':outputs,'dependencies':identities,'local_md_present':(W/'LOCAL.md').exists(),'fresh_future_prerequisites':['xe-e32-{mode}-full-build-root-v18','xe-e32-{mode}-asan-full-build-root-v15','xe-e32-lints-baseline-compare-root-v13','xe-native-dequant-group-byte-gpu-root-v4-free-full'],'sharing':'No mode sharing is authorized or implemented. Each mode requires its own fresh component build and smoke. Main can review a future freshly generated samebinarybothmodes proof separately.'}
(A/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(outputs,indent=2))
