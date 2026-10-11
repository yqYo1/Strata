import fcntl,importlib.util,json,time
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011');HERE=Path(__file__).resolve().parent
O=B/'xe-llvm-v711-source-v5/direct_owner.py';s=importlib.util.spec_from_file_location('o',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
LLVM=W/'.tools/intel-llvm-v711/install';LINT=W/'.lint'
BASE=dict(PATH=str(LINT/'venv/bin')+':'+str(LINT/'bin')+':'+str(LINT/'cppcheck/usr/bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LINT/'cppcheck/usr/lib/x86_64-linux-gnu')+':'+str(LLVM/'lib')+':/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/mkl/2026.1/lib',STRATA_LINT_BUILD=str(W/'build/free'))
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=B/'xe-e32-lints-root-v1';m.require(not out.exists(),'new output');out.mkdir()
 r=dict(active=True,complete=False,passed=False,commands=[],controller_sha256=m.sha(__file__),owner_sha256=m.sha(O),gpu_executed=False,model_executed=False,mode='free source LLVM build / oneAPI clang-tidy per repository lint instruction',total_wall_budget_seconds=1500,raw_log_budget_bytes=64<<20,tool_store=str(LINT.resolve()),scope='Mandatory changed/new-file lints; no source changes during run')
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save();begin=time.monotonic()
 def run(label,args,wall=60,require=True):
  m.require(time.monotonic()-begin<1500,'whole lint deadline');e,so,se=owner.run(label,args,BASE,W,wall=min(wall,1500-(time.monotonic()-begin)),cpu=int(wall)*4,file_cap=64<<20,text_cap=64<<20,total_cap=64<<20,rss_cap=16<<30)
  m.require(m.closed(e),'normal closed '+label)
  if require:m.require(e['exit_code']==0,'zero exit '+label)
  return e,so.read_text(),se.read_text()
 try:
  for name in ['xe-e32-free-full-build-root-v5','xe-e32-contrib-llvm-full-build-root-v5','xe-e32-free-asan-full-build-root-v4']:
   p=B/name/'record.json';x=json.loads(p.read_text());m.require(x['passed'] and not x['active'],'closed release build');r.setdefault('release_proofs',{})[name]=m.sha(p)
  r['paths']=sorted(set(run('modified-paths',['/usr/bin/git','diff','--name-only','HEAD'])[1].splitlines()+run('new-paths',['/usr/bin/git','ls-files','--others','--exclude-standard'])[1].splitlines()));r['source_pins']={p:m.sha(W/p) for p in r['paths']};m.require(len(r['paths'])==25,'expected candidate file set')
  r['compile_commands_sha256']=m.sha(W/'build/free/compile_commands.json');r['lint_script_sha256']=m.sha(W/'tools/lint/run.sh')
  for p in ['bin/gitleaks','venv/bin/codespell','venv/bin/reuse','cppcheck/usr/bin/cppcheck']:m.require((LINT/p).is_file(),'required tool '+p)
  run('install-cmake-lint',[str(LINT/'venv/bin/python'),'-m','pip','install','--disable-pip-version-check','cmakelang==0.6.13','PyYAML==6.0.2'],wall=180)
  r['python_lint_distributions']=run('python-lint-versions',[str(LINT/'venv/bin/python'),'-m','pip','show','codespell','reuse','cmakelang','PyYAML'])[1]
  for tool in ['cppcheck','cmake-lint']:r.setdefault('tool_versions',{})[tool]=run(tool+'-version',[tool,'--version'])[1]
  e,so,se=run('required-lint-script',['/bin/bash','tools/lint/run.sh',*r['paths']],wall=1200,require=False);r['lint_exit_code']=e['exit_code'];r['lint_stdout_sha256']=__import__('hashlib').sha256(so.encode()).hexdigest();r['lint_stderr_sha256']=__import__('hashlib').sha256(se.encode()).hexdigest();r['skipped_lints']=[v for v in so.splitlines() if 'skipped (' in v];r['source_stable']=all(m.sha(W/p)==h for p,h in r['source_pins'].items());m.require(r['source_stable'],'source stable during lints');r['complete']=True
 except BaseException as e:r['error']=repr(e)
 finally:r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error') and r.get('lint_exit_code')==0 and not r.get('skipped_lints'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','lint_exit_code','skipped_lints']}));raise SystemExit(0 if r['passed'] else 1)
