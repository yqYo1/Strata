"""Main-owned actual modified SYCL translation-unit compile; no link/GPU run."""
from pathlib import Path
import fcntl,hashlib,json,shutil,types
B=Path(__file__).parent
PARENT=B/'run_gdn_gate_factor_probe_v2.py'
P=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-events-20261010')
OLD=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-qualification-20261010')
SRC=B/'prefill-gemm-service-only-v1/prefill.cpp'
HDR=B/'prefill-gemm-service-only-v1/include/strata/prefill_route_census.hpp'
OUT=B/'prefill-gemm-only-tu-root-v1'
def ident(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(bytes=p.stat().st_size,sha256=h)
def save_json(p,j):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert ident(PARENT)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 assert ident(SRC)['sha256']=='1cf7e911a555e8611c1368a4d18d4ba81c95027bb8de77f89ad65ac64b05fc3c'
 assert ident(HDR)['sha256']=='c673382e9f4bf63919489cf00daa71bec793ff7b4584c203291380b827dc110b'
 assert ident(P/'sycl/src/prefill/prefill.cpp')['sha256']=='f6cbc4acb9f84b19480a25892a2e6864896a3e3ab0f68ace3c3f51a788591f69'
 q=json.loads((B/'prefill-gemm-only-contract-root-v1/record.json').read_text());assert q['passed'] and q['complete'] and not q['active']
 assert not OUT.exists();OUT.mkdir(mode=0o700);overlay=OUT/'include/strata';overlay.mkdir(parents=True);shutil.copyfile(HDR,overlay/HDR.name)
 SERVICE=HDR.parent/'prefill_service_ledger.hpp';assert ident(SERVICE)['sha256']=='7d473f68dcfdb48660c50cff0e75c6191e5b3339ed95901a1aa4286f38bbff06';shutil.copyfile(SERVICE,overlay/SERVICE.name)
 m=types.ModuleType('root_prefill_census_compile_owner');exec(compile(PARENT.read_text().split('\ndef parse_probe(',1)[0],str(PARENT),'exec'),m.__dict__);m.W=OUT
 owner=m.Owner(OUT);base=B/'prefill-service-qualification-cpu-build-v1/record.json';br=json.loads(base.read_text());assert br['passed'] and not br['active'];env=br['environment']
 cmd=[s.replace(str(OLD),str(P)) for s in br['changed_compile']['prefill.cpp.o']]
 cmd[cmd.index('-o')+1]=str(OUT/'prefill.cpp.o');cmd[cmd.index('-c')+1]=str(SRC);cmd[1:1]=['-I'+str(OUT/'include'),'-iquote',str(P/'sycl/src/prefill')]
 r=dict(active=True,complete=False,passed=False,linked=False,gpu_tested=False,model_executed=False,adopted=False,started_utc=m.utc(),scope='GEMM-only source and both headers SYCL TU compile; copy profiling properties unchanged with unchanged prior production policy and pinned includes; not link/GPU/model qualification',controller=ident(__file__),parent_owner=ident(PARENT),prior_recipe_receipt=ident(base),CPU_contract_receipt=ident(B/'prefill-gemm-only-contract-root-v1/record.json'),source=ident(SRC),header=ident(HDR),service_header=ident(SERVICE),baseline_caller=ident(P/'sycl/src/prefill/prefill.cpp'),quoted_relative_include_files={str(p):ident(p) for p in [P/'src/prefill/mmq_resident_sort.hpp',P/'src/prefill/wmma_gemm.h']},effective_command=cmd,environment=env,commands=owner.commands)
 def save():save_json(OUT/'record.json',r)
 owner.persist=save
 try:
  e,so,se=owner.run('compile-SYCL-prefill',cmd,env,wall=120,text_cap=8<<20,file_cap=8<<20)
  assert m.completed(e) and e['exit_code']==0,('compile',e)
  r['object']=ident(OUT/'prefill.cpp.o');r.update(complete=True,passed=True)
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:
  r.update(active=owner.active is not None,finished_utc=m.utc());r['passed']=r['passed'] and not r['active'] and all(m.completed(e) for e in owner.commands);save();print(json.dumps({k:r.get(k) for k in ['active','complete','passed','error']},ensure_ascii=False),flush=True)
 if not r['passed']:raise SystemExit(1)
