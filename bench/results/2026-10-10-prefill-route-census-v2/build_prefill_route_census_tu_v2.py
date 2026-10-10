"""Main-owned actual modified SYCL translation-unit compile; no link/GPU run."""
from pathlib import Path
import fcntl,hashlib,json,shutil,types
B=Path(__file__).parent
PARENT=B/'run_gdn_gate_factor_probe_v2.py'
P=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-events-20261010')
OLD=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-qualification-20261010')
SRC=B/'prefill-route-census-v2/prefill.cpp'
HDR=B/'prefill-route-census-v2/prefill_route_census.hpp'
OUT=B/'prefill-route-census-tu-root-v2'
def ident(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(bytes=p.stat().st_size,sha256=h)
def save_json(p,j):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert ident(PARENT)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 assert ident(SRC)['sha256']=='905f43162ae079b6785584da226d950cec14b13bd413ca51ab1a48f49de6ebaf'
 assert ident(HDR)['sha256']=='3545e3d4ffdb8f0ff0e43f36854b56bd28fe5a4fd02b2e42afaca6cd50f4160a'
 assert ident(P/'sycl/src/prefill/prefill.cpp')['sha256']=='f6cbc4acb9f84b19480a25892a2e6864896a3e3ab0f68ace3c3f51a788591f69'
 q=json.loads((B/'prefill-route-census-contract-root-v2/record.json').read_text());assert q['passed'] and q['complete'] and not q['active']
 assert not OUT.exists();OUT.mkdir(mode=0o700);overlay=OUT/'include/strata';overlay.mkdir(parents=True);shutil.copyfile(HDR,overlay/HDR.name)
 m=types.ModuleType('root_prefill_census_compile_owner');exec(compile(PARENT.read_text().split('\ndef parse_probe(',1)[0],str(PARENT),'exec'),m.__dict__);m.W=OUT
 owner=m.Owner(OUT);base=B/'prefill-service-qualification-cpu-build-v1/record.json';br=json.loads(base.read_text());assert br['passed'] and not br['active'];env=br['environment']
 cmd=[s.replace(str(OLD),str(P)) for s in br['changed_compile']['prefill.cpp.o']]
 cmd[cmd.index('-o')+1]=str(OUT/'prefill.cpp.o');cmd[cmd.index('-c')+1]=str(SRC);cmd[1:1]=['-I'+str(OUT/'include'),'-iquote',str(P/'sycl/src/prefill')]
 r=dict(active=True,complete=False,passed=False,linked=False,gpu_tested=False,model_executed=False,adopted=False,started_utc=m.utc(),scope='Actual modified caller/header SYCL TU compile with unchanged prior production policy and pinned includes; not link/GPU/model qualification',controller=ident(__file__),parent_owner=ident(PARENT),prior_recipe_receipt=ident(base),CPU_contract_receipt=ident(B/'prefill-route-census-contract-root-v2/record.json'),source=ident(SRC),header=ident(HDR),baseline_caller=ident(P/'sycl/src/prefill/prefill.cpp'),quoted_relative_include_files={str(p):ident(p) for p in [P/'src/prefill/mmq_resident_sort.hpp',P/'src/prefill/wmma_gemm.h']},effective_command=cmd,environment=env,commands=owner.commands)
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
