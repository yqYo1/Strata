import fcntl,importlib.util,json,re,time
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
LLVM=W/'.tools/intel-llvm-v711/install'
SNAP=B/'xe-parity-failure-host-source-snapshot-v1';STAGE=B/'xe-parity-failure-host-source-v1';SRC=STAGE/'tools/test_parity_failure_host.cpp'
O=B/'xe-llvm-v711-source-v5/direct_owner.py';s=importlib.util.spec_from_file_location('o',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
ENV=dict(PATH=str(LLVM/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LLVM/'lib'))
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/'xe-parity-failure-host-root-v1';m.require(not out.exists(),'immutable new output');out.mkdir()
 r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),owner_sha256=m.sha(O),source_sha256=m.sha(SRC),gpu_executed=False,model_executed=False,scope='Actual production checked_transfer host template callback return/standard/unknown exception boundary and buffered stderr flush; no Runtime, actual GPU submission/wait/async, private oneMath or leak qualification of _Exit branches',raw_log_budget_bytes=1<<20,total_wall_budget_seconds=240,cases=[])
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save();begin=time.monotonic()
 def run(label,args,expected=0,env=ENV,wall=20):
  m.require(time.monotonic()-begin<240,'whole host check deadline');e,so,se=owner.run(label,args,env,STAGE,wall=min(wall,240-(time.monotonic()-begin)),cpu=int(wall)*2,text_cap=1<<20,total_cap=128<<20,rss_cap=4<<30,file_cap=64<<20)
  m.require(m.closed(e) and e['exit_code']==expected,'normal expected closed '+label);return so.read_text(),se.read_text()
 try:
  m.require(r['source_sha256']=='1c856cea9a385703753901730483d6053e9b4b43cc94a253aa33904a4e2e62d5','reviewed host harness')
  m.require(m.sha(STAGE/'manifest.json')=='532c76d3fcd60ab98e48ef63a1273773de5249a38670e5f9e61e8005238bc9ee','harness source manifest')
  snap=json.loads((SNAP/'manifest.json').read_text());m.require(all(m.sha(SNAP/f)==h and m.sha(W/f)==h for f,h in snap['source_pins'].items()),'immutable exact production headers')
  r['header_pins']=snap['source_pins'];r['source_manifest_sha256']=m.sha(STAGE/'manifest.json')
  proof=B/'xe-llvm-v711-source-root-v5/record.json';compiler=json.loads(proof.read_text());m.require(compiler['passed'] and not compiler['active'] and compiler['complete'],'closed source compiler')
  m.require(all(m.sha(Path(v['path']))==v['sha256'] for v in compiler['installed_files'].values()),'installed compiler identity');r['compiler_receipt_sha256']=m.sha(proof)
  proof=B/'xe-llvm-v711-host-sanitizer-runtime-root-v2/record.json';rt=json.loads(proof.read_text());m.require(rt['passed'] and not rt['active'] and rt['complete'],'closed source sanitizer runtimes')
  m.require(all(m.sha(Path(v['path']))==v['sha256'] for v in rt['installed_runtime_files']),'installed host sanitizer runtime identity');r['host_sanitizer_receipt_sha256']=m.sha(proof)
  for arm in ['release','asan-ubsan']:
   binary=out/('parity_failure_host_'+arm);flags=['-fsanitize=address,undefined','-fno-omit-frame-pointer'] if arm=='asan-ubsan' else []
   run('compile-'+arm,[str(LLVM/'bin/clang++'),'-std=c++20','-O1','-g','-Wall','-Wextra','-Werror',*flags,'-I',str(SNAP/'src/kernels'),'-I',str(SNAP/'include'),'-isystem',str(LLVM/'include'),'-isystem',str(LLVM/'include/sycl'),str(SRC),'-o',str(binary)],wall=60)
   so,se=run('dependencies-'+arm,['/usr/bin/readelf','-d',str(binary)])
   needed=re.findall(r'\(NEEDED\).+?\[(.+?)\]',so);m.require(needed and all(not any(v in k for v in ['libsycl','libur_','libze_','libumf','libmkl']) for k in needed),'host-only binary dependencies')
   r.setdefault('binaries',{})[arm]=dict(path=str(binary),sha256=m.sha(binary),bytes=binary.stat().st_size,needed=needed)
   env=dict(ENV,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1') if arm=='asan-ubsan' else ENV
   for case in ['happy','standard','unknown','invalid']:
    status=0 if case=='happy' else 2 if case=='invalid' else 1
    so,se=run(arm+'-'+case,[str(binary),case],expected=status,env=env,wall=10)
    if case=='happy':expected=['START case=happy','CALLBACK case=happy count=1','CALLBACK_RETURN case=happy','AFTER_CALL case=happy','HAPPY_OK callback_count=1 checksum=130560','STAGING_DTOR case=happy bytes=1024','MAIN_RETURN case=happy status=0','ATEXIT']
    elif case=='invalid':expected=['usage: parity_failure_host happy|standard|unknown']
    else:
     msg='injected standard failure' if case=='standard' else 'unknown exception'
     expected=['START case='+case,'CALLBACK case='+case+' count=1','PRE_FAILURE case='+case,'parity: host '+case+': '+msg+'; terminating before caller staging can unwind']
    m.require(not so and se.splitlines()==expected,'exact complete host failure-scope protocol '+arm+'/'+case)
    r['cases'].append(dict(arm=arm,case=case,exit_code=status,stderr=se.splitlines(),stdout_empty=True,passed=True));save()
   m.require(m.sha(binary)==r['binaries'][arm]['sha256'],'binary stable')
  r['source_stable']=m.sha(SRC)==r['source_sha256'] and all(m.sha(SNAP/f)==h and m.sha(W/f)==h for f,h in r['header_pins'].items());m.require(r['source_stable'],'actual production/source identity stable');r['complete']=True
 except BaseException as e:r['error']=repr(e)
 finally:r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','cases']}));raise SystemExit(0 if r['passed'] else 1)
