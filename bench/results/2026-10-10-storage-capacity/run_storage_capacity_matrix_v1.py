from pathlib import Path
import subprocess,json,hashlib,datetime
B=Path(__file__).parent
controller=B/'run_storage_capacity_v2.py'
base=[('random',1),('sequential',1),('random',16),('sequential',16),('random',64)]
plan=[dict(mode=mode,workers=workers,repeat=repeat) for repeat in (1,2,3) for mode,workers in (base if repeat%2 else list(reversed(base)))]
out=B/'storage-capacity-v2-matrix-plan-v1.json';assert not out.exists()
d=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),plan=plan,closed=[],active=False,complete=False,passed=False,scope='Serial sameoriginalPLEfile;3freshprocesses each5shapes. Firstfullrandomw1r1alreadyclosed. No GPU/model, tuning/property/cachechange, no purephysicalSSDpeakclaim.')
def save():
 p=out.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2)+'\n');p.replace(out)
save()
for cell in plan:
 receipt=B/f"storage-capacity-v2-measure-{cell['mode']}-w{cell['workers']}-r{cell['repeat']}/record.json"
 if receipt.exists():
  assert cell==dict(mode='random',workers=1,repeat=1)
  r=json.loads(receipt.read_text());assert r['passed'] and r['complete'] and not r['active'];status=0
 else:
  d['active']=True;d['current']=cell;save();status=subprocess.run(['/usr/bin/python3',str(controller),'measure',cell['mode'],str(cell['workers']),str(cell['repeat'])]).returncode;r=json.loads(receipt.read_text());d['active']=False
 d['closed'].append(dict(**cell,returncode=status,path=str(receipt),sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),passed=r['passed'],complete=r['complete'],active=r['active']));save()
 if status or not r['passed'] or not r['complete'] or r['active']:raise SystemExit('Rejected matrix cell; no retry/gate changes')
assert hashlib.sha256(controller.read_bytes()).hexdigest()==d['controller_sha256']
properties=subprocess.check_output(['zfs','get','-Hp','-o','property,value','recordsize,compression,compressratio,primarycache,secondarycache,checksum,copies','rpool/USERDATA/yayoi'],text=True)
assert properties==json.loads((B/'storage-capacity-metadata-v1.json').read_text())['commands']['dataset']['stdout']
d['complete']=True;d['passed']=True;d['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print('STORAGE_MATRIX_CLOSED_PASS',flush=True)
