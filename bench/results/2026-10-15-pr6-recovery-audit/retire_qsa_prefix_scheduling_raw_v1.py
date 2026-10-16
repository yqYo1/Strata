#!/usr/bin/env python3
"""Main exclusive lock: retire committed, closed scheduling experiment logs only."""
import fcntl,hashlib,json,subprocess,importlib.util
from pathlib import Path
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
out=A/'retention-qsa-prefix-scheduling-raw-v1.json';assert not out.exists()
helper=P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec=importlib.util.spec_from_file_location('retention_owner',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Production engine artifacts and every tensor/session/profile remain out of scope.
prefixes=['main-qsa-bf16-prefix-parallel-probability-','main-qsa-bf16-prefix-probability-bench-','main-qsa-bf16-prefix-score-pad-','main-qsa-bf16-prefix-parallel-pad-']
r=dict(schema='strata-qsa-scheduling-raw-retention-v1',scope='Closed successful API/build logs and duplicate record copies only; keep private sources/ELFs/objects, production builds, all tensors/sessions and required profiles',deleted=[],canonical_receipts=[])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 roots=sorted(p for p in B.iterdir() if p.is_dir() and any(p.name.startswith(x) for x in prefixes))
 planned=[]
 for root in roots:
  canonical=A/('own-'+root.name.removeprefix('main-')+'.json')
  if not canonical.exists():raise RuntimeError('missing canonical '+str(canonical))
  v=json.loads(canonical.read_text());assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['active']
  committed=subprocess.check_output(['git','show','HEAD:'+str(canonical.relative_to(P))],cwd=P)
  assert committed==canonical.read_bytes()
  for c in v['commands']:
   assert m.closed(c) and c['exit_code']==0
   for old in c['owners']+[c['direct_identity']]:
    now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
  r['canonical_receipts'].append(dict(path=str(canonical),sha256=sha(canonical)))
  for p in (root/'owned').rglob('*'):
   if p.is_file() and p.suffix in ['.stdout','.stderr']:planned.append((p,'Closed successful raw trace/build/journal; useful result/commands/ELF/identity/fault evidence committed'))
  q=root/'record.json'
  if q.exists():assert q.read_bytes()==canonical.read_bytes();planned.append((q,'Byte-identical private duplicate of committed canonical receipt'))
 for p,why in planned:
  st=p.stat();r['deleted'].append(dict(path=str(p),sha256=sha(p),bytes=st.st_size,allocated_bytes=st.st_blocks*512,reason=why))
 r['logical_bytes']=sum(x['bytes'] for x in r['deleted']);r['allocated_bytes']=sum(x['allocated_bytes'] for x in r['deleted'])
 out.write_text(json.dumps(r,indent=2)+'\n')
 for p,_ in planned:p.unlink()
 r.update(complete=True,files_deleted=len(planned));out.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(files_deleted=len(planned),logical_bytes=r['logical_bytes'],allocated_bytes=r['allocated_bytes'])))
