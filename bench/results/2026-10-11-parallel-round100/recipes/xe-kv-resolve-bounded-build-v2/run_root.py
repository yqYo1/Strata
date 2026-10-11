"""Main-owned actual-production bounded KV probe compilation/lint; no GPU."""
import fcntl, importlib.util, json, time
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
ORIGINAL=B/'xe-kv-resolve-bounded-source-v2';S=B/'xe-kv-resolve-bounded-source-v3';LLVM=W/'.tools/intel-llvm-v711/install';O=B/'xe-llvm-v711-source-v5/direct_owner.py'
s=importlib.util.spec_from_file_location('owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','owner pin')
ENV=dict(PATH=str(LLVM/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LLVM/'lib'))
L=W/'.lint';LE=dict(ENV,PATH=str(L/'venv/bin')+':'+str(L/'bin')+':'+str(L/'cppcheck/usr/bin')+':'+ENV['PATH'],LD_LIBRARY_PATH=str(L/'cppcheck/usr/lib/x86_64-linux-gnu')+':'+str(LLVM/'lib')+':/opt/intel/oneapi/compiler/2026.1/lib')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=B/'xe-kv-resolve-bounded-build-root-v2';m.require(not out.exists(),'immutable output');out.mkdir()
 r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),owner_sha256=m.sha(O),gpu_executed=False,model_executed=False,binaries={},scope='Actual production device.cpp/gpu.cpp/atomic-patched kv_stream.cpp linked directly. Standalone FP16+INT8 format probe; not BLAS/oneMath coverage.',total_wall_budget_seconds=900,raw_log_budget_bytes=64<<20)
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;begin=time.monotonic();save()
 def run(label,args,env=ENV,wall=120):
  left=900-(time.monotonic()-begin);m.require(left>0,'whole build deadline')
  e,so,se=owner.run(label,args,env,W,wall=min(wall,left),cpu=int(wall)*4,text_cap=64<<20,total_cap=64<<20,rss_cap=16<<30)
  m.require(m.closed(e),'closed '+label);return e,so.read_text(),se.read_text()
 try:
  p=B/'xe-kv-resolve-atomic-source-commit-v1.json';v=json.loads(p.read_text());m.require(v['passed'] and v['complete'] and not v['active'],'published qualified KV source');r['source_revision']=v['committed_revision'];r['source_publication_sha256']=m.sha(p)
  e,so,se=run('source-head',['/usr/bin/git','rev-parse','HEAD']);m.require(e['exit_code']==0 and so.strip()==r['source_revision'],'actual current revision')
  e,so,se=run('source-status',['/usr/bin/git','status','--porcelain']);m.require(e['exit_code']==0 and not so.strip(),'clean current source')
  q=B/'xe-kv-resolve-bounded-main-review-v3.json';review=json.loads(q.read_text());m.require(review['main_reviewed'] and m.sha(S/'kv_resolve_bounded.cpp')==review['source_sha256'] and m.sha(ORIGINAL/'manifest.json')==review['manifest_sha256'],'Main reviewed actual probe');r['probe_sha256']=review['source_sha256'];r['review_sha256']=m.sha(q);r['manifest_sha256']=review['manifest_sha256']
  manifest=json.loads((ORIGINAL/'manifest.json').read_text());r['production_dependencies_pinned']={f:x['sha256'] for f,x in manifest['production_dependencies_pinned'].items()};r['production_dependencies_pinned']['src/kernels/xe/kv_stream.cpp']='cd3bb7877dd848497a1d7d50cbcf75ae28cffdd187b5c18b6234b917520f7fbf'
  m.require(all(m.sha(W/f)==h for f,h in r['production_dependencies_pinned'].items()),'reviewed production source/headers')
  r['additional_header_pins']={f:m.sha(W/f) for f in ['src/kernels/xe/device_caps.hpp','include/strata/kernels/kv_q4.hpp','include/strata/kernels/kv_q8.hpp','include/strata/core/device.hpp']}
  t=json.loads((B/'xe-llvm-v711-source-root-v5/record.json').read_text());m.require(t['passed'] and t['complete'] and not t['active'],'qualified compiler');m.require(all(m.sha(Path(x['path']))==x['sha256'] for x in t['installed_files'].values()),'actual compiler installation')
  r['source_pins']=v['source_pins'];m.require(all(m.sha(W/f)==h for f,h in r['source_pins'].items()),'published qualified source identity')
  r['proof_sha256']={}
  for mode in ['free','contrib-llvm']:
   for kind,version in [('full-build','v13'),('asan-full-build','v10')]:
    q=B/('xe-e32-'+mode+'-'+kind+'-root-'+version+'/record.json');p=json.loads(q.read_text());m.require(p['passed'] and p['complete'] and not p['active'] and all(m.closed(c) for c in p['commands']) and p['source_pins']==r['source_pins'],'closed matching full qualification');r['proof_sha256'][mode+'-'+kind]=m.sha(q)
   defs=['-DSTRATA_NO_BLAS=1'] if mode=='free' else ['-USTRATA_NO_BLAS']
   binary=out/('kv-resolve-bounded-'+mode)
   args=[str(LLVM/'bin/clang++'),'-std=c++20','-O2','-Wall','-Wextra','-Werror','-fsycl','-fsycl-targets=spir64','-pthread','-I'+str(W/'include'),*defs,str(S/'kv_resolve_bounded.cpp'),str(W/'src/core/device.cpp'),str(W/'src/core/gpu.cpp'),str(W/'src/kernels/xe/kv_stream.cpp'),'-o',str(binary)]
   e,so,se=run('compile-'+mode,args,wall=180);m.require(e['exit_code']==0,'actual probe compile');r['binaries'][mode]=dict(path=str(binary),sha256=m.sha(binary),bytes=binary.stat().st_size)
   db=out/('lint-db-'+mode);db.mkdir();row=dict(directory=str(W),file=str(S/'kv_resolve_bounded.cpp'),arguments=[*args[:args.index(str(S/'kv_resolve_bounded.cpp'))],'-c',str(S/'kv_resolve_bounded.cpp'),'-o',str(out/('unused-'+mode+'.o'))]);(db/'compile_commands.json').write_text(json.dumps([row],indent=2)+'\n')
   e,so,se=run('lint-'+mode,['/bin/bash','tools/lint/run.sh',str(S/'kv_resolve_bounded.cpp')],env=dict(LE,STRATA_LINT_BUILD=str(db)))
   r.setdefault('lint',{})[mode]=dict(exit_code=e['exit_code'],stdout=so,stderr=se,compile_db_sha256=m.sha(db/'compile_commands.json'));m.require(e['exit_code']==0 and 'skipped (' not in so and 'possible secret' not in so and 'error:' not in se,'all probe lints execute successfully')
   baseline=json.loads((B/'xe-e32-lints-baseline-compare-root-v8/record.json').read_text());m.require(baseline['passed'] and baseline['complete'] and not baseline['active'],'actual existing-header lint comparison')
   messages=[x[3] for x in baseline['comparisons'][mode]['baseline_diagnostics'] if x[:3]==['include/strata/kernels/kv_stream.hpp',83,'warning']];m.require(len(messages)==1,'exact existing header warning')
   expected=str(W/'include/strata/kernels/kv_stream.hpp')+':83:92: warning: '+messages[0]
   warnings=[line for line in so.splitlines() if 'warning:' in line];m.require(warnings==[expected],'zero new standalone probe warnings; one actual proven unchanged header finding')
   r['lint'][mode]['baseline_header_proof_sha256']=m.sha(B/'xe-e32-lints-baseline-compare-root-v8/record.json');r['lint'][mode]['new_probe_findings']=[]
   for name in ['clang-tidy','cppcheck','codespell','reuse']:m.require('== '+name in so,'required probe tool '+name)
   e,so,se=run('invalid-cli-'+mode,[str(binary),'unexpected'],wall=10);m.require(e['exit_code']==2 and not so and se=='FAIL: no CLI arguments accepted; Runtime not initialized\n','bad CLI before GPU init');save()
  m.require(m.sha(S/'kv_resolve_bounded.cpp')==r['probe_sha256'] and all(m.sha(W/f)==h for f,h in {**r['source_pins'],**r['production_dependencies_pinned'],**r['additional_header_pins']}.items()),'all source bytes stable');r['complete']=True
 except BaseException as e:r['error']=repr(e)
 finally:
  r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','binaries']}));raise SystemExit(0 if r['passed'] else 1)
