import fcntl,importlib.util,json,shlex
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');HERE=Path(__file__).resolve().parent
X=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011');LLVM=X/'.tools/intel-llvm-v711/install'
O=B/'xe-llvm-v711-source-v5/direct_owner.py';s=importlib.util.spec_from_file_location('o',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
env=dict(PATH=str(LLVM/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LLVM/'lib'))
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=B/'xe-host-sanitizer-flags-root-v1';m.require(not out.exists(),'new output');out.mkdir()
 r=dict(active=True,complete=False,passed=False,commands=[],gpu_executed=False,model_executed=False,controller_sha256=m.sha(__file__),source_sha256=m.sha(HERE/'probe.cpp'),references=['https://clang.llvm.org/docs/ClangCommandLineReference.html'],scope='Compiler driver actual host-only sanitizers, no execution of any compiled code')
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
 def run(label,args):
  e,so,se=owner.run(label,args,env,HERE,wall=90,cpu=360,file_cap=64<<20,text_cap=8<<20,total_cap=16<<20,rss_cap=4<<30)
  m.require(m.closed(e) and e['exit_code']==0,'normal '+label);return so.read_text(),se.read_text()
 try:
  proof=B/'xe-llvm-v711-source-root-v5/record.json';v=json.loads(proof.read_text());m.require(v['passed'] and not v['active'],'closed compiler');r['compiler_receipt_sha256']=m.sha(proof)
  flags=['-std=c++20','-O2','-Wall','-Wextra','-Werror','-Xarch_host','-fsanitize=address,undefined','-Xarch_host','-fno-omit-frame-pointer']
  for kind,args in [('sycl',['-fsycl']),('native',[])]:
   _,dry=run(kind+'-driver',[str(LLVM/'bin/clang++'),*args,*flags,'-###','-c',str(HERE/'probe.cpp'),'-o',str(out/(kind+'.o'))])
   cc=[shlex.split(l) for l in dry.splitlines() if '"-cc1"' in l];h=[c for c in cc if any('x86_64-' in x for x in c)];d=[c for c in cc if '-fsycl-is-device' in c];m.require(len(h)==1 and any(x.startswith('-fsanitize=') and 'address' in x and 'signed-integer-overflow' in x for x in h[0]),'actual host sanitizer args');m.require(all(not any(x.startswith('-fsanitize=') for x in c) for c in d),'device not sanitized');m.require(bool(d)==(kind=='sycl'),'expected device compilation job')
   r.setdefault('driver_validation',{})[kind]=dict(host_jobs=len(h),device_jobs=len(d),host_sanitizers=True,device_sanitizers=False,dry_sha256=__import__('hashlib').sha256(dry.encode()).hexdigest())
   run(kind+'-compile',[str(LLVM/'bin/clang++'),*args,*flags,'-c',str(HERE/'probe.cpp'),'-o',str(out/(kind+'.o'))])
   nm,_=run(kind+'-symbols',['/usr/bin/nm','-u',str(out/(kind+'.o'))]);m.require('__asan' in nm and '__ubsan' in nm,'host object actually instrumented');r['driver_validation'][kind]['host_object_instrumented']=True
  r['complete']=True
 except BaseException as e:r['error']=repr(e)
 finally:r['active']=owner.active is not None;r['passed']=r['complete'] and not r.get('error') and not r['active'];save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','driver_validation']}));raise SystemExit(0 if r['passed'] else 1)
