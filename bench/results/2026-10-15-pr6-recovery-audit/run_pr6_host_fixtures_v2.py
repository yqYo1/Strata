#!/usr/bin/env python3
"""Main-only ordinary C++ helper fixtures; no model/SYCL/Level Zero execution.
Repository LICENSE. Tests six reconstructed helper contracts, not GPU safety.
"""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,os,sys,time
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
C=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-pr6-prefill-restore')
D=C/'validation/pr6-restore'
OWNER=A.parents[2]/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
LLVM=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):
 p=Path(p);return dict(path=str(p),realpath=str(p.resolve(strict=True)),bytes=p.stat().st_size,sha256=sha(p))
def main():
 root=B/'main-pr6-host-fixtures-v2';target=A/'own-pr6-host-fixtures-v2.json'
 assert not root.exists() and not target.exists() and sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
 spec=importlib.util.spec_from_file_location('pr6_fixture_owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 names=['draft_uploads','range_guard','token_admission','allocated_fields','validation_control','bounded_join'];inputs=[D/'fixtures/CMakeLists.txt']+[D/'fixtures'/(n+'.cpp') for n in names]+[D/'include/strata/prefill/draft_uploads.hpp',D/'include/strata/prefill/range_guard.hpp',D/'include/strata/core/token_admission.hpp',D/'include/strata/core/allocated_fields.hpp',D/'include/strata/core/validation_control.hpp',D/'include/strata/prefill/retirement.hpp']
 compiler=LLVM/'bin/clang++';base=json.loads((A/'own-vllm-resource-timeline-v3/manifest.json').read_text());cp=next(v for v in base['file_pins'].values() if v['realpath']==str(compiler.resolve()))
 assert sha(compiler)==cp['sha256'];pins=[pin(p) for p in inputs]
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
  start=time.monotonic();record=dict(schema='strata-pr6-host-fixtures-v1',status='RUNNING',passed=False,complete=False,active=False,GPU_executed=False,source_checkpoint='3da01f7148f5536f444c8f471c65e2d47c58bc41',source_pins=pins,controller=pin(__file__),compiler=cp,commands=[],samples=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),scope='Standalone ordinary C++20 fixtures only; no engine/CMake/native GPU/runtime initialization. 256K array is HOST boundary/lease fixture, not physical GPU context qualification.',expected_child_outcomes='draft_uploads fixture asserts its own failed-drain children exit73 or SIGABRT; parent must exit0. Core limit zero only in owned fixture descendants.',retention=dict(owner='Main',consumer='Host-helper gate for reconstructed PR6; source-only runtime still GPU-held',raw_budget_bytes=64<<20,review_deadline_utc='2026-10-17T03:00:00Z'))
  owner=None
  def persist():
   record['active']=bool(owner and owner.active is not None);(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
  owner=m.Owner(root/'owned',persist);record['commands']=owner.commands
  env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(root),LD_LIBRARY_PATH=str(LLVM/'lib'))
  record['environment']=env
  try:
   for name in names:
    assert time.monotonic()-start<240
    binary=root/name
    argv=[str(compiler),'-std=c++20','-O2','-Wall','-Wextra','-Werror','-fno-sycl','-pthread','-I'+str(D/'include'),str(D/'fixtures'/(name+'.cpp')),'-o',str(binary)]
    for label,args in [('build-'+name,argv),('test-'+name,[str(binary)])]:
     c,out,err=owner.run(label,args,env,root,wall=45 if label.startswith('build') else 20,cpu=40 if label.startswith('build') else 15,rss_cap=2<<30,text_cap=8<<20,total_cap=128<<20,file_cap=8<<20)
     record['samples'].append(dict(label=label,exit_code=c.get('exit_code'),stdout=out.read_text(),stderr=err.read_text(),elapsed_seconds=c['elapsed_seconds']))
     assert m.closed(c) and c['exit_code']==0
    record.setdefault('binaries',{})[name]=pin(binary)
   for p in pins:assert pin(p['path'])==p
   assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
   record.update(passed=True,complete=True,status='PASS_HOST_HELPERS_ONLY')
  except BaseException as e:record.update(error=repr(e),status='FAIL',passed=False,complete=False)
  finally:
   record['ownership_closed']=owner.active is None and all(m.closed(c) and c['exit_code']==0 for c in owner.commands)
   for c in owner.commands:
    for old in c['owners']+[c['direct_identity']]:
     now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
   if not record['ownership_closed']:record.update(status='FAIL',passed=False,complete=False)
   record.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
   persist()
   with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
  print(json.dumps(dict(status=record['status'],receipt=str(target),sha256=sha(target),GPU_executed=False)))
  return 0 if record['passed'] else 1
if __name__=='__main__':sys.exit(main())
