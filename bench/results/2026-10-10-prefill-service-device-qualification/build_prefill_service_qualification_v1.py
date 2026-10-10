"""Root builds two changed SYCL TUs plus a small qualification; no GPU work."""
from pathlib import Path
import fcntl,hashlib,json,shlex,subprocess,sys,types
B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-qualification-20261010')
T=W.parent/'perf-sycl-decode-pool-phase-timing-v0141-20261009'
D=T/'build-sycl-decode-pool-phase-timing-v0141-v1'
HEAD='7c5ad75a45c7e188ecef7a947b49b829dc5c5710'
OUT=B/'prefill-service-qualification-cpu-build-v1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ident(p):return dict(bytes=Path(p).stat().st_size,sha256=sha(p))
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert git(W,'rev-parse','HEAD')==HEAD and not git(W,'status','--porcelain')
 assert git(T,'rev-parse','HEAD')=='7d0105f2a942ac72ca20fef69d4c2e7960653de3' and not git(T,'status','--porcelain')
 contract=B/'prefill-expert-service-ledger-cpu-contract-v2/record.json'
 cr=json.loads(contract.read_text());assert cr['passed'] and not cr['active']
 prior=B/'decode-pool-phase-timing-v0141-private-build-v2/record.json'
 assert sha(prior)=='3632b1514544ef2dd89046c112b823c75a27d0215aca2fce58c4b8df9e43d7f8'
 parent=B/'run_gdn_gate_factor_probe_v2.py';assert sha(parent)=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 owner_source=parent.read_text().split('\ndef parse_probe(',1)[0]
 for a,b in [('(16 << 30, 16 << 30)','(32 << 30, 32 << 30)'),('(120, 121)','(600, 601)'),('rss <= 2 << 30','rss <= 12 << 30')]:
  assert owner_source.count(a)==1;owner_source=owner_source.replace(a,b)
 module=types.ModuleType('prefill_service_build_owner');exec(compile(owner_source,str(parent),'exec'),module.__dict__);module.W=OUT
 assert not OUT.exists();OUT.mkdir(mode=0o700);owner=module.Owner(OUT)
 commands=json.loads((D/'compile_commands.json').read_text())
 changed={}
 for rel,name in [('sycl/src/prefill/prefill.cpp','prefill.cpp.o'),('sycl/src/prefill/gemm.dp.cpp','gemm.dp.cpp.o')]:
  entry=[e for e in commands if e['file']==str(T/rel)];assert len(entry)==1
  args=[s.replace(str(T),str(W)) for s in shlex.split(entry[0]['command'])]
  args[args.index('-o')+1]=str(OUT/name);changed[name]=args
 qual=list(changed['gemm.dp.cpp.o']);qual[qual.index('-o')+1]=str(OUT/'qualify_prefill_f16_event.cpp.o')
 qual[qual.index('-c')+1]=str(W/'sycl/tools/qualify_prefill_f16_event.cpp')
 raw=subprocess.check_output(['ninja','-C',str(D),'-t','commands','strata'],text=True).splitlines()[-1]
 assert raw.startswith(': && ') and raw.endswith(' && :')
 original_link=shlex.split(raw[5:-5]);link=list(original_link);link[link.index('-o')+1]=str(OUT/'strata')
 for i,arg in enumerate(link):
  if arg=='libstrata_prefill.a':link[i]=str(OUT/arg)
  elif not arg.startswith(('-','/')) and arg.endswith(('.a','.o')):link[i]=str(D/arg)
 qual_link=list(link);qual_link[qual_link.index('-o')+1]=str(OUT/'qualify_prefill_f16_event')
 main=str(D/'CMakeFiles/strata.dir/src/program/generate.cpp.o');assert qual_link.count(main)==1
 qual_link[qual_link.index(main)]=str(OUT/'qualify_prefill_f16_event.cpp.o')
 dependencies={str(D/arg):ident(D/arg) for arg in original_link if not arg.startswith(('-','/')) and arg.endswith(('.a','.o'))}
 source_files=['include/strata/prefill/gemm.hpp','sycl/include/dpct/blas_utils.hpp','sycl/include/dpct/detail/blas_utils_detail.hpp','sycl/include/strata/prefill_service_ledger.hpp','sycl/src/prefill/gemm.dp.cpp','sycl/src/prefill/prefill.cpp','sycl/tools/test_prefill_service_ledger.py','sycl/tools/qualify_prefill_f16_event.cpp']
 sources={rel:ident(W/rel) for rel in source_files}
 r=dict(active=True,complete=False,passed=False,gpu_tested=False,adopted=False,root=str(W),commit=HEAD,
  compiled_engine_base_commit='7d0105f2a942ac72ca20fef69d4c2e7960653de3',controller=ident(__file__),
  sources=sources,original_link=original_link,changed_compile=changed,qualification_compile=qual,
  changed_link=link,qualification_link=qual_link,cached_dependencies_before=dependencies,
  prior_build_receipt=ident(prior),CPU_contract_receipt=ident(contract),commands=owner.commands,
  scope='Default-off diagnostic service path; prefill and Gemm TUs replaced in closed cached baseline archive. Other numerical libraries byte-pinned to base. Small qualification compiled with identical SYCL flags. No full fresh dependency rebuild or model/device/full-context result.',started_utc=module.utc())
 def save():
  p=OUT/'record.json.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(OUT/'record.json')
 owner.persist=save
 def run(label,args,env,wall=120):
  e,so,se=owner.run(label,args,env,wall=wall,text_cap=32<<20,file_cap=128<<20);save()
  assert module.completed(e) and e['exit_code']==0,(label,e);return so
 try:
  clean=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
  out=run('toolchain-environment',['/bin/bash','-c','source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && env -0'],clean,30)
  discovered=dict(x.decode().split('=',1) for x in out.read_bytes().split(b'\0') if b'=' in x)
  env=dict(clean)
  for key in ['PATH','LD_LIBRARY_PATH','LIBRARY_PATH','CPATH','ONEAPI_ROOT','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','MKLROOT']:
   if key in discovered:env[key]=discovered[key]
  r['environment']=env;save()
  for name,args in changed.items():run('compile-'+name,args,env,1200)
  run('compile-qualification',qual,env,300)
  run('copy-original-archive',['/usr/bin/cp',str(D/'libstrata_prefill.a'),str(OUT/'libstrata_prefill.a')],clean)
  for name in changed:run('replace-'+name,['/usr/bin/ar','r',str(OUT/'libstrata_prefill.a'),str(OUT/name)],clean)
  members=subprocess.check_output(['ar','t',str(D/'libstrata_prefill.a')],text=True).splitlines()
  assert members==subprocess.check_output(['ar','t',str(OUT/'libstrata_prefill.a')],text=True).splitlines()
  comparisons=[]
  for name in members:
   before=subprocess.check_output(['ar','p',str(D/'libstrata_prefill.a'),name]);after=subprocess.check_output(['ar','p',str(OUT/'libstrata_prefill.a'),name])
   equal=before==after;assert equal==(name not in changed)
   comparisons.append(dict(member=name,byte_identical=equal,before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest()))
  r['archive_member_comparison']=comparisons;save()
  run('link-strata',link,env,300)
  run('link-qualification',qual_link,env,300)
  r.update(binaries={name:ident(OUT/name) for name in ('strata','qualify_prefill_f16_event')},passed=True,complete=True)
 except BaseException as error:r['error']=type(error).__name__+': '+str(error)
 finally:
  try:
   assert {p:ident(p) for p in dependencies}==dependencies
   assert {rel:ident(W/rel) for rel in sources}==sources
   assert git(W,'rev-parse','HEAD')==HEAD and not git(W,'status','--porcelain')
   assert git(T,'rev-parse','HEAD')=='7d0105f2a942ac72ca20fef69d4c2e7960653de3' and not git(T,'status','--porcelain')
  except BaseException as error:r['final_pin_error']=repr(error);r['passed']=False
  r.update(active=owner.active is not None,finished_utc=module.utc())
  r['passed']=r['passed'] and not r['active'] and all(module.completed(e) for e in owner.commands);save()
  print(json.dumps({k:r.get(k) for k in ['passed','complete','active','error','binaries']}),flush=True)
 if not r['passed']:sys.exit(1)
