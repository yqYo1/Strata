"""Main-only bounded CPU gates against exact v3; no GPU/model execution."""
from pathlib import Path
import fcntl,hashlib,json,types
B=Path(__file__).parent;PARENT=B/'run_gdn_gate_factor_probe_v2.py'
SRC=B/'prefill_census_admission_v5.cpp';HDR=B/'prefill-route-census-v3/prefill_route_census.hpp';OUT=B/'prefill-census-admission-root-v5'
def ident(p):
 with Path(p).open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(bytes=Path(p).stat().st_size,sha256=h)
def save_json(p,j):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert ident(PARENT)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 assert ident(HDR)['sha256']=='c673382e9f4bf63919489cf00daa71bec793ff7b4584c203291380b827dc110b'
 assert not OUT.exists();OUT.mkdir(mode=0o700)
 m=types.ModuleType('root_census_admission');exec(compile(PARENT.read_text().split('\ndef parse_probe(',1)[0],str(PARENT),'exec'),m.__dict__);m.W=OUT
 owner=m.Owner(OUT);env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
 positives=['resident','ne10','threaded','mmq','fused-fallback','full-context-host','request32k']
 negatives=['n-low','n-high','bad-env','alloc1','alloc2','copy-before','copy-between','copy-after','meta-small','meta-large','unknown-branch','native-branch','null-cnt','duplicate-layer','product-before','missing-call','missing-down','duplicate-product','unselected-product','resident-copy','output-cap','early-return','fatal']
 ids=[f'id-{actor}-{axis}{value}' for actor in ['plan','copy','call','product','lookup'] for axis,values in [('l',[-1,1,3,64]),('e',[-1,16,512])] for value in values]
 r=dict(active=True,complete=False,passed=False,adopted=False,gpu_tested=False,model_executed=False,started_utc=m.utc(),controller=ident(__file__),parent_owner=ident(PARENT),source=ident(SRC),header=ident(HDR),commands=owner.commands,results=[],scope='Exact real header, test-only global nothrow allocation and clock_gettime interceptors. Two heap failures, admitted IDs/population, sticky copy fault, cap/no partial output, fatal and destructor controls. Fullcontext is host accounting only.')
 def save():save_json(OUT/'record.json',r)
 owner.persist=save
 try:
  exe=OUT/'contract'
  e,so,se=owner.run('compile',['/usr/bin/g++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-pthread','-I'+str(HDR.parent),str(SRC),'-o',str(exe)],env,wall=120,file_cap=8<<20);assert m.completed(e) and e['exit_code']==0,('compile',e)
  r['binary']=ident(exe)
  for mode in ['off','zero']+positives+negatives+ids:
   e,so,se=owner.run(mode,[str(exe),mode],env,wall=120 if mode=='output-cap' else 15,file_cap=8<<20);assert m.completed(e) and e['exit_code']==0,(mode,e)
   data=se.read_bytes();rows=[json.loads(x) for x in data.splitlines()];layers=[x for x in rows if x['kind']=='prefill_route_layer'];rr=[x for x in rows if x['kind']=='prefill_route_census_receipt'];obs=json.loads(so.read_text()) if so.stat().st_size else None
   if mode in ['off','zero']:ok=not rows and obs['allocations']==0 and obs['clock_calls']==0
   elif mode in positives:
    z=rr[0] if len(rr)==1 else {};n=262144 if mode=='full-context-host' else 32767 if mode=='request32k' else 32768;c=(n+8191)//8192
    ok=len(rr)==1 and z.get('valid') and len(layers)==c and z['requested_tokens']==n and z['chunks']==c and z['layers']==c and z['calls']==10*c and z['routed_rows']==10*n and obs['allocations']==2 and obs['clock_calls']>0 and sum(obs['allocation_sizes'])<=34<<20
    ok=ok and z['transfer_calls']==(9*c if mode=='threaded' else 0) and z['transfer_bytes']==(9*c*1971200 if mode=='threaded' else 0) and z['routed_transfer_calls']==(8*c if mode=='threaded' else 0)
    text=b''.join(x+b'\n' for x in data.splitlines() if json.loads(x)['kind']=='prefill_route_layer');h=14695981039346656037
    for byte in text:h=((h^byte)*1099511628211)&((1<<64)-1)
    ok=ok and z['output_bytes']==len(text) and z['histogram_fnv1a64']==f'{h:016x}'
   else:
    ok=len(rr)==1 and not rr[0]['valid'] and not layers
    if mode not in ['fatal']:
     ok=ok and rr[0]['calls']==rr[0]['routed_rows']==rr[0]['transfer_calls']==rr[0]['transfer_bytes']==rr[0]['output_bytes']==0 and rr[0]['histogram_fnv1a64']=='0000000000000000'
    expected={'output-cap':'output_cap_or_format_failure','alloc1':'allocation_failure','alloc2':'allocation_failure','early-return':'request_early_return_or_exception','fatal':'owned_contract_fatal'}.get(mode)
    if expected:ok=ok and rr[0]['reason']==expected
   r['results'].append(dict(mode=mode,contract_passed=bool(ok),stdout=obs,observed_rows=rows));save()
   assert ok,('contract mismatch',mode,obs,rr)
  r.update(complete=True,passed=True)
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:
  r.update(active=owner.active is not None,finished_utc=m.utc());r['passed']=r['passed'] and not r['active'] and all(m.completed(e) for e in owner.commands);save();print(json.dumps({k:r.get(k) for k in ['active','complete','passed','error']},ensure_ascii=False),flush=True)
 if not r['passed']:raise SystemExit(1)
