import fcntl,importlib.util,json,time
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE=Path(__file__).resolve().parent
WX=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
LLVM=WX/'.tools/intel-llvm-v711/install'
OWNER=B/'xe-llvm-v711-source-v5/direct_owner.py'
spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.require(m.sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
BASE=dict(PATH=str(LLVM/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LLVM/'lib'))
HELPERS=['prefill_publication','commit_transaction','full_context_probe','mtp_completion','context_bounds']
HEADERS=['include/strata/prefill/publication.hpp','include/strata/core/commit_transaction.hpp','include/strata/program/full_context_probe.hpp','include/strata/core/mtp_completion.hpp','include/strata/core/context_bounds.hpp']
def main():
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  proof=B/'xe-llvm-v711-source-root-v5/record.json';toolchain=json.loads(proof.read_text())
  m.require(toolchain['passed'] and not toolchain['active'] and toolchain['complete'],'closed compiler')
  runtime_proof=B/'xe-llvm-v711-host-sanitizer-runtime-root-v2/record.json';runtime=json.loads(runtime_proof.read_text())
  m.require(runtime['passed'] and not runtime['active'] and runtime['complete'],'closed matching host sanitizer runtimes')
  out=B/'xe-e32-source-llvm-host-helpers-root-v1';m.require(not out.exists(),'immutable output');out.mkdir()
  r=dict(active=True,complete=False,passed=False,commands=[],tests=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),owner_sha256=m.sha(OWNER),compiler_receipt_sha256=m.sha(proof),host_runtime_receipt_sha256=m.sha(runtime_proof),model_executed=False,gpu_executed=False,scope='Five actual production helpers, new source LLVM host Release and host ASan/UBSan; not a full engine/GPU/physical262144 qualification',total_wall_budget_seconds=600,raw_log_budget_bytes=32<<20)
  def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
  owner=m.Owner(out/'commands',save);r['commands']=owner.commands;start=time.monotonic();save()
  def cmd(label,args,env=BASE,wall=40):
   m.require(time.monotonic()-start<600,'whole helper deadline')
   e,so,se=owner.run(label,args,env,HERE,wall=min(wall,600-(time.monotonic()-start)),cpu=int(wall),text_cap=8<<20,total_cap=32<<20,rss_cap=8<<30,file_cap=64<<20)
   m.require(m.closed(e) and e['exit_code']==0,'normal closed '+label)
   return so.read_text()
  try:
   for item in toolchain['installed_files'].values():m.require(m.sha(Path(item['path']))==item['sha256'],'installed compiler identity')
   for item in runtime['installed_runtime_files']:m.require(m.sha(Path(item['path']))==item['sha256'],'host sanitizer runtime identity')
   paths=HEADERS+['tools/test_'+h+'.cpp' for h in HELPERS]
   r['source_pins']={f:m.sha(WX/f) for f in paths};save()
   r['compiler_version']=cmd('compiler-version',[str(LLVM/'bin/clang++'),'--version'])
   for mode in ['release','asan-ubsan']:
    for h in HELPERS:
     binary=out/(h+'-'+mode)
     flags=['-std=c++20','-O2','-DNDEBUG','-UNDEBUG','-Wall','-Wextra','-Werror','-I'+str(WX/'include')]
     if mode=='asan-ubsan':flags+=['-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
     cmd('compile-'+h+'-'+mode,[str(LLVM/'bin/clang++'),*flags,str(WX/('tools/test_'+h+'.cpp')),'-o',str(binary)],wall=60)
     env=dict(BASE)
     if mode=='asan-ubsan':env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
     output=cmd('test-'+h+'-'+mode,[str(binary)],env=env,wall=30)
     r['tests'].append(dict(helper=h,mode=mode,passed=True,binary_sha256=m.sha(binary),output=output));save()
   r['source_stable']=all(m.sha(WX/f)==v for f,v in r['source_pins'].items());m.require(r['source_stable'],'five completed helpers stable')
   r['complete']=True
  except BaseException as exc:r['error']=repr(exc)
  finally:
   r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
  print(json.dumps(dict(passed=r['passed'],error=r.get('error'),tests=r['tests'],record=str(out/'record.json'))))
  return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
