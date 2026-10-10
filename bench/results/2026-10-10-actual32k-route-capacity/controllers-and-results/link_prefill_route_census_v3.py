"""Root-only finite link, one private archive replacement. No GPU execution."""
from pathlib import Path
import fcntl,hashlib,json,shutil,types
B=Path(__file__).parent;OUT=B/'prefill-route-census-linked-root-v3';PARENT=B/'run_gdn_gate_factor_probe_v2.py'
PRIOR=B/'prefill-service-qualification-cpu-build-v1/record.json';TU=B/'prefill-route-census-tu-root-v2/record.json';CPU=B/'prefill-census-admission-root-v4/record.json'
def ident(p):
 with Path(p).open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(bytes=Path(p).stat().st_size,sha256=h)
def save_json(p,j):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert ident(PARENT)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 prior=json.loads(PRIOR.read_text());tu=json.loads(TU.read_text());cpu=json.loads(CPU.read_text())
 assert all(x['passed'] and not x['active'] for x in [prior,tu,cpu]);assert len(cpu['results'])==64
 assert ident(PRIOR)['sha256']==tu['prior_recipe_receipt']['sha256']
 for p,h in prior['cached_dependencies_before'].items():assert ident(p)==h
 obj=B/'prefill-route-census-tu-root-v2/prefill.cpp.o';assert ident(obj)==tu['object']
 caller=B/'prefill-route-census-v2/prefill.cpp';hdr=caller.parent/'prefill_route_census.hpp'
 assert ident(caller)==tu['source'] and ident(hdr)==tu['header']
 assert not OUT.exists();OUT.mkdir(mode=0o700)
 m=types.ModuleType('root_census_link');exec(compile(PARENT.read_text().split('\ndef parse_probe(',1)[0],str(PARENT),'exec'),m.__dict__);m.W=OUT
 owner=m.Owner(OUT);env=prior['environment'];old=B/'prefill-service-qualification-cpu-build-v1/libstrata_prefill.a';arc=OUT/'libstrata_prefill.a'
 r=dict(active=True,complete=False,passed=False,gpu_tested=False,model_executed=False,adopted=False,started_utc=m.utc(),controller=ident(__file__),parent_owner=ident(PARENT),prior_receipt=ident(PRIOR),CPU_admission_receipt=ident(CPU),TU_receipt=ident(TU),caller=ident(caller),header=ident(hdr),object=ident(obj),prior_archive=ident(old),environment=env,cached_dependencies_before=prior['cached_dependencies_before'],commands=owner.commands,member_comparison=[],scope='Exact qualified v2 object replaces only prefill.cpp.o in prior service-ledger archive, retaining gemm/service and all cached inputs; link FSIZE64MiB admits38MiB expected binary. v2 failed normally at overly small8MiB root FSIZE; no GPU/model qualification.')
 def save():save_json(OUT/'record.json',r)
 owner.persist=save
 try:
  e,so,se=owner.run('archive-list-before',['/usr/bin/ar','t',str(old)],env,wall=15,file_cap=8<<20);assert m.completed(e) and e['exit_code']==0
  names=so.read_text().splitlines();assert names.count('prefill.cpp.o')==1 and len(names)==len(set(names))
  before={}
  for i,n in enumerate(names):
   e,so,se=owner.run(f'member-before-{i}',['/usr/bin/ar','p',str(old),n],env,wall=15,file_cap=8<<20);assert m.completed(e) and e['exit_code']==0;before[n]=ident(so)
  shutil.copyfile(old,arc)
  e,so,se=owner.run('archive-replace',['/usr/bin/ar','r',str(arc),str(obj)],env,wall=30,file_cap=8<<20);assert m.completed(e) and e['exit_code']==0
  e,so,se=owner.run('archive-index',['/usr/bin/ar','s',str(arc)],env,wall=30,file_cap=8<<20);assert m.completed(e) and e['exit_code']==0
  for i,n in enumerate(names):
   e,so,se=owner.run(f'member-after-{i}',['/usr/bin/ar','p',str(arc),n],env,wall=15,file_cap=8<<20);assert m.completed(e) and e['exit_code']==0;h=ident(so)
   assert h==(ident(obj) if n=='prefill.cpp.o' else before[n]);r['member_comparison'].append(dict(member=n,before=before[n],after=h,replaced=n=='prefill.cpp.o'))
  cmd=[str(arc) if p==str(old) else str(OUT/'strata') if p==str(old.parent/'strata') else p for p in prior['changed_link']];r['effective_link']=cmd;save()
  e,so,se=owner.run('link',cmd,env,wall=120,file_cap=64<<20);assert m.completed(e) and e['exit_code']==0,('link',e)
  for p,h in prior['cached_dependencies_before'].items():assert ident(p)==h
  assert ident(old)==r['prior_archive'] and ident(obj)==tu['object']
  r.update(complete=True,passed=True,linked=True,binary=str(OUT/'strata'),binary_identity=ident(OUT/'strata'),archive_identity=ident(arc),all_other_members_unchanged=True,cached_inputs_unchanged=True)
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:
  r.update(active=owner.active is not None,finished_utc=m.utc());r['passed']=r['passed'] and not r['active'] and all(m.completed(e) for e in owner.commands);save();print(json.dumps({k:r.get(k) for k in ['active','complete','passed','binary_identity','error']},ensure_ascii=False),flush=True)
 if not r['passed']:raise SystemExit(1)
