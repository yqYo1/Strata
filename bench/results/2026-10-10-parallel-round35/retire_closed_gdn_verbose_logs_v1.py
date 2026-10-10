from pathlib import Path
import datetime,fcntl,hashlib,json,subprocess,shutil
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');Q=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-quad-pipeline-20261010');W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009');A=W/'bench/results/2026-10-10-parallel-round35';OUT=B/'closed-gdn-verbose-retirement-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not OUT.exists()
 candidates=[]
 for archive,manifest,allowed in [('2026-10-10-gdn-quad-qualification','raw-file-identities.json',{'gdn-quad-cpu-build-v2','gdn-quad-initial-gates-v1','gdn-quad-prefix-carry-gates-v1'}),('2026-10-10-gdn-quad-quiet-32k','raw-file-identities.json',{'gdn-quad-quiet-cpu-build-v1','gdn-quad-quiet-gates-v1'}),('2026-10-10-gdn-host-event-diagnostic','compact-log-identities.json',{'gdn-host-event-cpu-build-v1','gdn-host-event-runtime-v1'})]:
  mp=Q/'bench/results'/archive/manifest;rows=json.loads(mp.read_text());assert subprocess.run(['git','-C',str(Q),'ls-files','--error-unmatch',str(mp.relative_to(Q))],stdout=subprocess.DEVNULL).returncode==0
  for row in rows:
   p=Path(row['path'])
   if p.parent.name not in allowed or not p.exists() or row['bytes']<50000:continue
   if not (p.suffix=='.trace' or (p.suffix=='.stderr' and p.name.startswith(('gpu-short','prefix-'))) or p.name=='build.stdout'):continue
   assert p.is_file() and not p.is_symlink() and sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
   receipt=p.parent/'record.json';r=json.loads(receipt.read_text());assert r['passed'] and r['complete'] and not r['active'] and not r['cleanup'] and not r['survivors']
   candidates.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'closed_original_receipt':str(receipt),'receipt_sha256':sha(receipt),'committed_replacement_manifest':str(mp),'replacement_manifest_sha256':sha(mp),'replacement_archive':archive,'reason':'Closed successful API/strace/compiler repetition. Useful correctness/commands/env/pins/individual samples retained in committed compact archive. No active raw-log consumer; researchers read committed evidence only.','status':'prepared'})
 assert candidates and len({r['path'] for r in candidates})==len(candidates)
 record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_owns_serial_lock':True,'preserve_failed_receipts_and_raw_failure_evidence':True,'keep_individual_measurement_and_correctness_rows':True,'keep_required_binaries_fixtures_and_research_snapshots':True,'entries':candidates}
 OUT.write_text(json.dumps(record,indent=2)+'\n')
 for row in candidates:
  p=Path(row['path']);assert sha(p)==row['sha256'];p.unlink();row['status']='deleted';OUT.write_text(json.dumps(record,indent=2)+'\n')
 record['deleted_files']=len(candidates);record['deleted_bytes']=sum(x['bytes'] for x in candidates);record['complete']=True;OUT.write_text(json.dumps(record,indent=2)+'\n')
 shutil.copyfile(OUT,A/OUT.name);shutil.copyfile(__file__,A/Path(__file__).name)
 p=A/'REPORT.md';p.write_text(p.read_text()+f"\n\nClosed successful GDN verbose logs retired under the root-owned serial lock: {len(candidates)} files/{record['deleted_bytes']} bytes. Exact path/hash/closed receipt/committed replacement/reason are in closed-gdn-verbose-retirement-v1.json. Full individual measurement/correctness rows, failures, required binaries/data and active research snapshots remain. This is scoped retirement of superseded trace/compiler repetitions, not deletion of needed65GiB experiment data. Original statuses were not rewritten.\n")
 identity=A/'archive-file-identities.json';old=json.loads(identity.read_text());new={str(p.relative_to(A)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(A.rglob('*')) if p.is_file() and p!=identity};assert set(old)<=set(new)
 for n,v in old.items():
  if n!='REPORT.md':assert v==new[n],n
 identity.write_text(json.dumps(new,indent=2)+'\n');print(json.dumps({k:record[k] for k in ['deleted_files','deleted_bytes','complete']}))
