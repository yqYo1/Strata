"""Compile changed prefill TU, replace one cached archive member, relink.
No GPU work; the frozen original libraries and numerical math remain pinned.
"""
from pathlib import Path
import fcntl,hashlib,json,os,shlex,subprocess,sys,types
B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-phase-validity-20261010')
T=W.parent/'perf-sycl-decode-pool-phase-timing-v0141-20261009'
D=T/'build-sycl-decode-pool-phase-timing-v0141-v1'
HEAD='185fa78098bee4be01ac81ad4f791673d24246f3'
OUT=B/'prefill-phase-validity-cpu-build-v1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ident(p):return dict(bytes=Path(p).stat().st_size,sha256=sha(p))
def git(root,*a):return subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert git(W,'rev-parse','HEAD')==HEAD and not git(W,'status','--porcelain')
 assert git(T,'rev-parse','HEAD')=='7d0105f2a942ac72ca20fef69d4c2e7960653de3' and not git(T,'status','--porcelain')
 prior=B/'decode-pool-phase-timing-v0141-private-build-v2/record.json'
 assert sha(prior)=='3632b1514544ef2dd89046c112b823c75a27d0215aca2fce58c4b8df9e43d7f8'
 cpu=json.loads((B/'prefill-phase-validity-cpu-contract-v1/record.json').read_text());assert cpu['passed'] and not cpu['active']
 source=W/'sycl/src/prefill/prefill.cpp';assert sha(source)==cpu['source_sha256']
 old=(T/'sycl/src/prefill/prefill.cpp').read_text();new=source.read_text()
 assert old.split('struct PfTimer {',1)[0]==new.split('struct PfTimer {',1)[0]
 assert old.split('\n// Profile the memcpy commands themselves',1)[1]==new.split('\n// Profile the memcpy commands themselves',1)[1]
 parent=B/'run_gdn_gate_factor_probe_v2.py';assert sha(parent)=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 owner_source=parent.read_text().split('\ndef parse_probe(',1)[0]
 for a,b in [('(16 << 30, 16 << 30)','(32 << 30, 32 << 30)'),('(120, 121)','(600, 601)'),('rss <= 2 << 30','rss <= 12 << 30')]:
  assert owner_source.count(a)==1;owner_source=owner_source.replace(a,b)
 m=types.ModuleType('prefill_validity_build_owner');exec(compile(owner_source,str(parent),'exec'),m.__dict__);m.W=OUT
 assert not OUT.exists();OUT.mkdir(mode=0o700);o=m.Owner(OUT)
 original_commands=json.loads((D/'compile_commands.json').read_text())
 entries=[e for e in original_commands if e['file']==str(T/'sycl/src/prefill/prefill.cpp')];assert len(entries)==1
 orig_compile=shlex.split(entries[0]['command']);compile_args=[s.replace(str(T),str(W)) for s in orig_compile]
 compile_args[compile_args.index('-o')+1]=str(OUT/'prefill.cpp.o')
 link_raw=subprocess.check_output(['ninja','-C',str(D),'-t','commands','strata'],text=True).splitlines()[-1]
 assert link_raw.startswith(': && ') and link_raw.endswith(' && :')
 orig_link=shlex.split(link_raw[5:-5]);link=list(orig_link)
 link[link.index('-o')+1]=str(OUT/'strata')
 for i,s in enumerate(link):
  if s=='libstrata_prefill.a':link[i]=str(OUT/'libstrata_prefill.a')
  elif not s.startswith(('-','/')) and s.endswith(('.a','.o')):link[i]=str(D/s)
 dependencies={str(D/s):ident(D/s) for s in orig_link if not s.startswith(('-','/')) and s.endswith(('.a','.o'))}
 r=dict(active=True,complete=False,passed=False,gpu_tested=False,adopted=False,root=str(W),commit=HEAD,compiled_engine_base_commit='7d0105f2a942ac72ca20fef69d4c2e7960653de3',controller=ident(__file__),source_sha256=sha(source),original_compile=orig_compile,changed_compile=compile_args,original_link=orig_link,changed_link=link,cached_dependencies_before=dependencies,prior_build_receipt=ident(prior),CPU_contract_receipt=ident(B/'prefill-phase-validity-cpu-contract-v1/record.json'),commands=o.commands,scope='Diagnostic-only changed PfTimer translation unit. Unchanged closed original7d0105 engine/CPU/kernel libraries, only prefill.cpp.o archive member replaced, production flags preserved.',started_utc=m.utc())
 def save():
  p=OUT/'record.json.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(OUT/'record.json')
 o.persist=save
 def run(label,args,env,wall=120):
  e,so,se=o.run(label,args,env,wall=wall,text_cap=32<<20,file_cap=128<<20);save();assert m.completed(e) and e['exit_code']==0,(label,e);return so
 try:
  clean=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
  e=run('toolchain-environment',['/bin/bash','-c','source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && env -0'],clean,wall=30)
  discovered=dict(x.decode().split('=',1) for x in e.read_bytes().split(b'\0') if b'=' in x)
  env=dict(clean)
  for k in ['PATH','LD_LIBRARY_PATH','LIBRARY_PATH','CPATH','ONEAPI_ROOT','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','MKLROOT']:
   if k in discovered:env[k]=discovered[k]
  r['environment']=env;save()
  run('compile-prefill',compile_args,env,wall=1200)
  run('copy-original-archive',['/usr/bin/cp',str(D/'libstrata_prefill.a'),str(OUT/'libstrata_prefill.a')],clean)
  run('replace-prefill-member',['/usr/bin/ar','r',str(OUT/'libstrata_prefill.a'),str(OUT/'prefill.cpp.o')],clean)
  members=subprocess.check_output(['ar','t',str(D/'libstrata_prefill.a')],text=True).splitlines()
  assert members==subprocess.check_output(['ar','t',str(OUT/'libstrata_prefill.a')],text=True).splitlines()
  compared=[]
  for name in members:
   before=subprocess.check_output(['ar','p',str(D/'libstrata_prefill.a'),name]);after=subprocess.check_output(['ar','p',str(OUT/'libstrata_prefill.a'),name])
   equal=before==after;assert equal==(name!='prefill.cpp.o')
   compared.append(dict(member=name,byte_identical=equal,before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest()))
  r['archive_member_comparison']=compared;save()
  run('link-strata',link,env,wall=240)
  r.update(binary=str(OUT/'strata'),binary_sha256=sha(OUT/'strata'),passed=True,complete=True)
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:
  try:
   assert {p:ident(p) for p in dependencies}==dependencies
   assert sha(source)==r['source_sha256']
   assert git(W,'rev-parse','HEAD')==HEAD and not git(W,'status','--porcelain')
   assert git(T,'rev-parse','HEAD')=='7d0105f2a942ac72ca20fef69d4c2e7960653de3' and not git(T,'status','--porcelain')
  except BaseException as e:r['final_pin_error']=repr(e);r['passed']=False
  r.update(active=o.active is not None,finished_utc=m.utc());r['passed']=r['passed'] and not r['active'] and all(m.completed(e) for e in o.commands);save()
  print(json.dumps({k:r.get(k) for k in ['passed','complete','active','error','binary_sha256']}),flush=True)
 if not r['passed']:sys.exit(1)
