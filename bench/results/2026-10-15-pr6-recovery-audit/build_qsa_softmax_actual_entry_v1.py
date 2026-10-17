#!/usr/bin/env python3
"""Compile dual actual row-major softmax API fixture, only after model closure."""
import datetime,fcntl,hashlib,importlib.util,json,re,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,sha256=h)
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 root=B/'main-qsa-softmax-actual-entry-build-v1';dst=A/'own-qsa-softmax-actual-entry-build-v1.json';assert not root.exists() and not dst.exists()
 op=A/'direct_owner_transient_artifacts_v1.py';assert pin(op)['sha256']=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a';m=module('softmax_actual_owner',op)
 package=A/'own-vllm-xe2-qualification-v1';mp=package/'manifest.json';assert pin(mp)['sha256']=='7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58';meta=json.loads(mp.read_text());hp=package/'main_controller.py';assert pin(hp)['sha256']==meta['package_files']['main_controller.py']['sha256'];guard=module('softmax_actual_guard',hp)
 bp=A/'qsa-score-gemm-entry-engine-build-v2.json';build=json.loads(bp.read_text());gates=[bp]
 for arm in [0,1]:gates.append(A/f'qsa-score-gemm-entry-model-gemm{arm}-r0-tokens65536-warm4-v1.json')
 for g in gates:
  d=json.loads(g.read_text());assert d['passed'] and d['complete'] and d['ownership_closed'] and not d['active'] and d['fresh_faults']==[] and d['boot_unchanged'] and not d.get('driver_dump_exists') and not d.get('driver_dump_entries')
  for c in d['commands']:
   assert m.closed(c) and c['exit_code']==0
   for old in c['owners']+[c['direct_identity']]:
    now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
 pkg=A/'own-qsa-softmax-actual-entry-v1';proposal=pkg/'manifest.json';mdata=json.loads(proposal.read_text());dispatcher=pkg/'dispatcher.cpp';assert pin(dispatcher)['sha256']==mdata['dispatcher_sha256']
 for n,h in mdata['source_pins'].items():assert pin(P/n)['sha256']==h
 fixture=pkg/'fixture.cpp';assert pin(fixture)['sha256']==mdata['fixture']['sha256'];libroot=Path(build['binary']['path']).parent
 libs=[libroot/n for n in ['libstrata_kernels.a','libstrata_core.a','libstrata_sycl_runtime.a']]
 frozen=dict(build['frozen_files'])
 for p in [*gates,proposal,dispatcher,fixture,op,mp,hp,Path(__file__),L/'bin/clang++',*libs]:frozen[str(p)]=pin(p)
 assert pin(L/'bin/clang++')['sha256']=='1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard.no_model_owners(meta['known_model_files']);guard.driver_guard()
  root.mkdir(mode=0o700);(root/'tmp').mkdir()
  env={**build['environment'],'TMPDIR':str(root/'tmp'),'XDG_CACHE_HOME':str(root/'cache')}
  rec=dict(schema='strata-qsa-softmax-actual-entry-build-v1',passed=False,complete=False,active=False,GPU_executed=False,commands=[],boot_id=build['boot_id'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),base_build=pin(bp),frozen_files=frozen,library_inputs={str(p):pin(p) for p in libs},environment=env,provenance=mdata['provenance'],scope='Same test binary dispatches actual current optimized serial versus subgroup row-major APIs; unchanged independent oracle/cases, owner-retained host DMA fixture. No old private packed kernel is compiled or selected. Archive output hashes first observed here as build inputs, not retroactive receipt claims.',production_selected=False,model_qualified=False)
  owner=None
  def save():
   rec['active']=bool(owner and owner.active is not None);rec['ownership_closed']=bool(owner and owner.active is None and all(m.closed(c) for c in owner.commands));(root/'record.json').write_text(json.dumps(rec,indent=2)+'\n')
  owner=m.Owner(root/'owned',save);rec['commands']=owner.commands
  def verify():
   assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==build['boot_id']
   for p,v in frozen.items():assert pin(p)==v,p
  try:
   verify();exe=root/'sycl_qsa_softmax_actual_entry'
   argv=[str(L/'bin/clang++'),'-std=c++20','-O3','-DNDEBUG','-DSTRATA_ENABLE_SYCL=1','-Dqsa_bf16_prefix_batch=qsa_prefix_actual_entry','-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-fno-fast-math','-ffp-contract=off','-fsycl','-fsycl-id-queries-range=size_t','-Wall','-Wextra','-Werror',str(dispatcher),str(fixture),*map(str,libs),'-o',str(exe)]
   c,so,se=owner.run('compile-link',argv,env,P,wall=300,cpu=900,rss_cap=8<<30,text_cap=64<<20,file_cap=128<<20,total_cap=1<<30);rec.update(stdout=so.read_text(),stderr=se.read_text());assert m.closed(c) and c['exit_code']==0
   c,so,se=owner.run('readelf',['/usr/bin/readelf','-d',str(exe)],env,P,wall=30,cpu=60,rss_cap=1<<30,text_cap=1<<20,file_cap=1<<20,total_cap=2<<20);assert m.closed(c) and c['exit_code']==0;rec['dynamic']=so.read_text()
   needed=re.findall(r'Shared library: \[([^\]]+)\]',rec['dynamic']);assert set(needed)=={'libsycl.so.9','libstdc++.so.6','libm.so.6','libgcc_s.so.1','libc.so.6'},needed
   verify();guard.driver_guard();rec.update(binary=pin(exe),passed=True,complete=True)
  except BaseException as e:
   import traceback
   rec.update(error=repr(e),traceback=traceback.format_exc())
  finally:
   rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();dst.write_bytes((root/'record.json').read_bytes())
  print(json.dumps(dict(receipt=str(dst),passed=rec['passed'],complete=rec['complete'],error=rec.get('error'))),flush=True)
  return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
