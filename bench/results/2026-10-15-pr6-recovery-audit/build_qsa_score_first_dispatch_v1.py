#!/usr/bin/env python3
"""Build private exact-root-tree head mapping only after both model arms close."""
import datetime,fcntl,hashlib,importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,sha256=h)
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 root=B/'main-qsa-score-first-dispatch-build-v1';out=A/'qsa-score-first-dispatch-build-v1.json';assert not root.exists() and not out.exists()
 gate_paths=[A/f'qsa-score-prefill-model-querytile{t}-tokens65536-warm4-v1.json' for t in [0,8]]
 cohort_path=A/'qsa-score-headparallel-cohort-v1.json';cohort=json.loads(cohort_path.read_text());assert cohort['passed'] and cohort['complete']
 for p in gate_paths:
  v=json.loads(p.read_text());assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['active'] and v['fresh_faults']==[] and not v['driver_dump_exists']
 engine_path=A/'qsa-score-prefill-engine-build-v1.json';engine=json.loads(engine_path.read_text());assert engine['passed'] and engine['complete'] and engine['ownership_closed']
 compiler=pin(L/'bin/clang++');assert compiler['sha256']=='1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
 owner_path=A/'direct_owner_transient_artifacts_v1.py';assert pin(owner_path)['sha256']=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a';owned=module('headparallel_build_owner',owner_path)
 package=A/'own-vllm-xe2-qualification-v1';meta=json.loads((package/'manifest.json').read_text());helper=package/'main_controller.py';assert pin(helper)['sha256']==meta['package_files']['main_controller.py']['sha256'];guard=module('headparallel_build_guard',helper)
 candidate=A/'qsa-score-first-dispatch-v1';proposal=candidate/'proposal.json';v=json.loads(proposal.read_text())
 for name,h in v['files'].items():assert pin(candidate/name)['sha256']==h
 libs=[Path(engine['binary']['path']).parent/n for n in ['libstrata_kernels.a','libstrata_core.a','libstrata_sycl_runtime.a']]
 # All archive outputs predate the qualified final executable; their exact
 # current hashes are frozen here for this new private build, not retroactively
 # claimed as old build-time pins.
 assert all(p.stat().st_mtime_ns<=Path(engine['binary']['path']).stat().st_mtime_ns for p in libs)
 inputs=[Path(__file__),proposal,cohort_path,owner_path,helper,engine_path,L/'bin/clang++',*libs,*gate_paths,*[candidate/n for n in v['files']],*list((P/'include').rglob('*.hpp')),P/'src/kernels/sycl/qsa_select.cpp']
 frozen={str(p):pin(p) for p in inputs}
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard.no_model_owners(meta['known_model_files']);guard.driver_guard();root.mkdir(mode=0o700);(root/'tmp').mkdir()
  env=dict(PATH=str(L/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(L/'lib'),TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
  rec=dict(schema='strata-qsa-score-first-dispatch-build-v1',passed=False,complete=False,active=False,GPU_executed=False,frozen_files=frozen,compiler=compiler,environment=env,commands=[],production_selected=False,model_PP_measured=False,
   provenance=v['provenance'],scope='Private head-parallel exact-root FP32 mapping with production query8 control; whole-score and ID original-kernel parity outside timing. Source reuse and fewer shuffle rounds are hypotheses, not hardware resource or speed proof.')
  owner=None
  def save():
   rec['active']=bool(owner and owner.active is not None);out.write_text(json.dumps(rec,indent=2)+'\n')
  owner=owned.Owner(root/'owned',save);rec['commands']=owner.commands
  def run(n,argv):
   c,so,se=owner.run(n,argv,env,P,wall=300,cpu=900,rss_cap=12<<30,text_cap=16<<20,file_cap=256<<20,total_cap=2<<30)
   assert owned.closed(c) and c['exit_code']==0,(n,se.read_text()[-6000:]);return so.read_text()
  try:
   assert all(pin(p)==v for p,v in frozen.items())
   flags=[compiler['path'],'-std=c++20','-O3','-DNDEBUG','-DSTRATA_ENABLE_SYCL=1','-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-fno-fast-math','-ffp-contract=off','-fsycl','-fsycl-id-queries-range=size_t','-Wall','-Wextra','-Werror']
   exe=root/'sycl_qsa_score_first_dispatch_bench'
   run('compile-link',flags+[str(candidate/'fixture.cpp'),str(candidate/'qsa_score_headparallel.cpp'),*map(str,libs),'-o',str(exe)])
   rec['binary']=pin(exe);rec['dynamic']=run('readelf',['/usr/bin/readelf','-d',str(exe)])
   assert all(pin(p)==v for p,v in frozen.items());rec.update(passed=True,complete=True)
  except BaseException as e:
   import traceback
   rec.update(error=repr(e),traceback=traceback.format_exc())
  finally:
   rec['ownership_closed']=owner.active is None and all(owned.closed(c) for c in owner.commands);rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
  print(json.dumps(dict(receipt=str(out),passed=rec['passed'],error=rec.get('error'))),flush=True);return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
