#!/usr/bin/env python3
"""Retire only committed, closed BF16 v1 success logs and duplicate records."""
import fcntl,hashlib,json,subprocess
from pathlib import Path
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 dst=A/'qsa-bf16-late-flags-raw-retirement-v1.json';assert not dst.exists()
 receipts=sorted([*A.glob('own-qsa-bf16-late-flags-*-v1.json'),*A.glob('own-qsa-bf16-flags-bench-*-v1.json')]);assert len(receipts)==13,len(receipts)
 result=dict(schema='strata-qsa-bf16-late-flags-retention-v1',source_sha256=digest(Path(__file__)),deleted=[],receipts=[],reason='All oracle/finite/guard/input/repeat, samples, runtime maps/environment, UR counts, owner/fault outcomes committed. No remaining fullAPI consumer; do not alter original receipts. Keep binaries/objects/source/archives.')
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  planned=[]
  for p in receipts:
   r=json.loads(p.read_text());assert r['passed'] and r['complete'] and not r['active'] and r['ownership_closed']
   if r['GPU_executed']:assert r['native_exit_code']==0 and not r['fresh_faults'] and not r['driver_dump_entries']
   rel=p.relative_to(P);committed=subprocess.check_output(['git','show','HEAD:'+str(rel)],cwd=P)
   assert committed==p.read_bytes()
   root=B/p.stem.replace('own-','main-',1);record=root/'record.json';assert record.read_bytes()==p.read_bytes()
   result['receipts'].append(dict(path=str(p),sha256=digest(p)))
   planned.append(record)
   for c in r['commands']:
    for name,v in c['logs'].items():
     log=root/'owned'/name;assert log.is_file() and log.stat().st_size==v['bytes'] and digest(log)==v['sha256']
     planned.append(log)
  assert len(planned)==len(set(planned))
  for p in planned:result['deleted'].append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
  result['logical_bytes']=sum(x['bytes'] for x in result['deleted']);result['physical_freed_bytes']='not measured'
  for p in planned:p.unlink()
  result['complete']=True;dst.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(files=len(planned),logical_bytes=result['logical_bytes'],manifest=str(dst))))
if __name__=='__main__':main()
