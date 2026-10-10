import fcntl,hashlib,importlib.util,json,os,shutil,tarfile,urllib.request
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-b570-prefill-publication-20261011');OLD=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/eval-b570-20261010');HERE=Path(__file__).resolve().parent
p=B/'xestrata-clean-64k-comparison-v4/direct_owner.py';s=importlib.util.spec_from_file_location('owner',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(p)=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686','owner pin')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=B/'xe-prefill-lint-root-v3';m.require(not out.exists(),'new output');out.mkdir(mode=0o700)
 r=dict(active=True,complete=False,passed=False,gpu_executed=False,model_executed=False,started_utc=m.utc(),controller_sha256=m.sha(__file__),source_sha256=json.loads((HERE/'source-pins.json').read_text()))
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 own=m.Owner(out/'commands',save);r['commands']=own.commands
 env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
 def run(label,args,cwd=W,wall=60,require=True,environment=None):
  e,so,se=own.run(label,args,environment or env,cwd,wall=wall,text_cap=8<<20,total_cap=64<<20,rss_cap=6<<30,cpu=wall)
  if require:m.require(m.closed(e) and e['exit_code']==0,'normal '+label)
  return e,so,se
 save()
 try:
  lint=W/'.lint';downloads=lint/'downloads'
  run('download-cppcheck-dependency',['/usr/bin/apt-get','download','libtinyxml2-10'],cwd=downloads,wall=60)
  ds=list(downloads.glob('libtinyxml2-10*.deb'));m.require(len(ds)==1,'one dependency');run('unpack-cppcheck-dependency',['/usr/bin/dpkg-deb','-x',str(ds[0]),str(lint/'cppcheck')])
  r['tinyxml_package_sha256']=m.sha(ds[0])
  cfg=lint/'cppcheck/usr/bin/cfg'
  if not cfg.exists():cfg.symlink_to('../lib/x86_64-linux-gnu/cppcheck/cfg',target_is_directory=True)
  db=json.loads((OLD/'build/eval-b570-icpx-v1/compile_commands.json').read_text());newdb=[]
  for v in db:
   n=dict(v);n['file']=n['file'].replace(str(OLD),str(W));n['command']=n['command'].replace(str(OLD/'include'),str(W/'include')).replace(' -c '+str(OLD),' -c '+str(W));newdb.append(n)
  build=lint/'build';build.mkdir(exist_ok=True);(build/'compile_commands.json').write_text(json.dumps(newdb,indent=2)+'\n');r['lint_compile_commands_sha256']=m.sha(build/'compile_commands.json')
  le=dict(env,PATH=str(lint/'venv/bin')+':'+str(lint/'bin')+':'+str(lint/'cppcheck/usr/bin')+':/usr/bin:/bin',STRATA_LINT_BUILD=str(build),LD_LIBRARY_PATH=str(lint/'cppcheck/usr/lib/x86_64-linux-gnu')+':/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/mkl/2026.1/lib')
  baseline_env=dict(le,STRATA_LINT_BUILD=str(OLD/'build/eval-b570-icpx-v1'))
  run('baseline-tidy',['/opt/intel/oneapi/compiler/2026.1/bin/compiler/clang-tidy','--quiet','-p',baseline_env['STRATA_LINT_BUILD'],'--config-file='+str(OLD/'tools/lint/clang-tidy.yaml'),str(OLD/'src/prefill/prefill.cpp')],cwd=OLD,wall=180,environment=baseline_env,require=False)
  e,so,se=run('lint',['/bin/bash','tools/lint/run.sh','src/prefill/prefill.cpp','include/strata/prefill/publication.hpp','tools/test_prefill_publication.cpp'],wall=240,environment=le,require=False)
  r['lint_exit_code']=e['exit_code'];r['lint_closed']=m.closed(e);r['stdout_sha256']=m.sha(so);r['stderr_sha256']=m.sha(se);r['passed']=bool(m.closed(e) and e['exit_code']==0);r['complete']=True
 except BaseException as exc:r['error']=repr(exc)
 finally:r['active']=own.active is not None;r['finished_utc']=m.utc();save()
 print(json.dumps(dict(record=str(out/'record.json'),passed=r['passed'],error=r.get('error'))))
 raise SystemExit(0 if r['passed'] else 1)
