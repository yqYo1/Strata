"""Root-only CPU admission; no model or GPU run. Original handoff stays immutable."""
from pathlib import Path
import fcntl,hashlib,json,types
B=Path(__file__).parent
PARENT=B/'run_gdn_gate_factor_probe_v2.py'
SOURCE=B/'prefill_route_census_contract_v1.cpp'
HEADER=B/'prefill-route-census-v2/prefill_route_census.hpp'
OUT=B/'prefill-route-census-contract-root-v2'
def ident(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'bytes':p.stat().st_size,'sha256':h}
def save_json(p,j):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert ident(PARENT)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 assert ident(HEADER)['sha256']=='3545e3d4ffdb8f0ff0e43f36854b56bd28fe5a4fd02b2e42afaca6cd50f4160a'
 assert not OUT.exists();OUT.mkdir(mode=0o700)
 m=types.ModuleType('prefill_census_cpu_owner');exec(compile(PARENT.read_text().split('\ndef parse_probe(',1)[0],str(PARENT),'exec'),m.__dict__);m.W=OUT
 owner=m.Owner(OUT);env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
 r=dict(active=True,complete=False,passed=False,adopted=False,gpu_tested=False,started_utc=m.utc(),controller=ident(__file__),parent_owner=ident(PARENT),source=ident(SOURCE),header=ident(HEADER),commands=owner.commands,results=[],scope='CPU contract against exact unmodified Sol header; admission requires every negative to suppress layer rows and emit one invalid receipt')
 def save():save_json(OUT/'record.json',r)
 owner.persist=save
 try:
  binary=OUT/'contract'
  e,so,se=owner.run('compile',['/usr/bin/g++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-pthread','-I'+str(HEADER.parent),str(SOURCE),'-o',str(binary)],env,wall=120,file_cap=8<<20)
  assert m.completed(e) and e['exit_code']==0,('compile',e)
  r['binary']=ident(binary)
  for mode,valid in [('off',None),('zero',None),('bad-env',False),('resident',True),('mixed-stream-all',True),('mmq',True),('duplicate-call',False),('resident-copy',False),('outside-layer-copy',False)]:
   e,so,se=owner.run(mode,[str(binary),mode],env,wall=15,file_cap=8<<20)
   assert m.completed(e) and e['exit_code']==0,(mode,e)
   data=se.read_bytes();rows=[json.loads(x) for x in data.splitlines()];layers=[x for x in rows if x['kind']=='prefill_route_layer'];receipts=[x for x in rows if x['kind']=='prefill_route_census_receipt']
   ok=not rows if valid is None else len(receipts)==1 and receipts[0]['valid'] is valid and (len(layers)==4 if valid else not layers)
   if valid is True:
    z=receipts[0];ok=ok and z['calls']==40 and z['routed_rows']==327680 and z['layers']==4 and z['chunks']==4
    ok=ok and z['transfer_calls']==(36 if mode=='mixed-stream-all' else 0) and z['routed_transfer_calls']==(32 if mode=='mixed-stream-all' else 0) and z['transfer_bytes']==(36*1971200 if mode=='mixed-stream-all' else 0)
    text=b''.join(x+b'\n' for x in data.splitlines() if json.loads(x)['kind']=='prefill_route_layer');h=14695981039346656037
    for byte in text:h=((h^byte)*1099511628211)&((1<<64)-1)
    ok=ok and z['output_bytes']==len(text) and z['histogram_fnv1a64']==f'{h:016x}'
   r['results'].append(dict(mode=mode,expected_valid=valid,contract_passed=bool(ok),observed_rows=rows));save()
  r['complete']=True;r['passed']=all(x['contract_passed'] for x in r['results'])
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:
  r.update(active=owner.active is not None,finished_utc=m.utc());r['passed']=r['passed'] and not r['active'] and all(m.completed(e) for e in owner.commands);save();print(json.dumps({k:r.get(k) for k in ['active','complete','passed','error']},ensure_ascii=False),flush=True)
 if not r['passed']:raise SystemExit(1)
