#!/usr/bin/env python3
"""Retire closed successful scorer logs, preserving canonical immutable receipts."""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,re,subprocess
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
owner=module('head_retire_owner',A/'direct_owner_transient_artifacts_v1.py')
package=A/'own-vllm-xe2-qualification-v1';meta=json.loads((package/'manifest.json').read_text());guard=module('head_retire_guard',package/'main_controller.py')
paths=[A/'qsa-score-headparallel-build-v1.json',A/'qsa-score-first-dispatch-build-v1.json',A/'qsa-score-headparallel-engine-build-v1.json']
paths+=list(A.glob('qsa-score-headparallel-suite-?-v1.json'))
paths+=list(A.glob('qsa-score-headparallel-?-*-?-v1.json'))
paths+=list(A.glob('qsa-score-first-dispatch-*-0-v1.json'))
qualification=json.loads((A/'qsa-score-headparallel-entry-qualification-v1.json').read_text());assert qualification['passed'] and qualification['complete']
paths += [Path(x['path']) for x in qualification['samples']]
assert len(paths)==37,len(paths)
out=A/'qsa-score-headparallel-retention-v1.json';assert not out.exists()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard.no_model_owners(meta['known_model_files']);guard.driver_guard()
 records=[];deletions=[];warnings=[]
 for p in paths:
  r=json.loads(p.read_text());assert r['passed'] and r['complete'] and r['ownership_closed'] and not r.get('active')
  if r.get('GPU_executed'):assert r['fresh_faults']==[] and not r['driver_dump_entries'] and r['boot_unchanged']
  root=B/('main-'+p.stem);assert root.is_dir()
  for c in r['commands']:
   assert owner.closed(c) and c['exit_code']==0
   for old in c['owners']+[c['direct_identity']]:
    now=owner.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
   for ext in ['stdout','stderr']:
    f=root/'owned'/(c['label']+'.'+ext)
    if not f.exists():continue
    assert f.is_file() and not f.is_symlink()
    text=f.read_text(errors='replace');lines=[x for x in text.splitlines() if re.search(r'(?i)warning:|\b(?:UR|ZE)_RESULT_ERROR_|cleanup failed|unsupported',x)]
    assert sum(len(x.encode()) for x in lines)<65536
    if lines:warnings.append({'path':str(f),'lines':lines})
    deletions.append({**pin(f),'reason':'Closed normal-exit log; canonical receipt retains command/environment/source or binary identity, individual timings/oracles/guard/ownership/fault results; cache/phase lines preserved for first-dispatch diagnostics. Warning/error findings are extracted in this manifest.'})
  duplicate=root/'record.json'
  if duplicate.exists():
   assert duplicate.read_bytes()==p.read_bytes();deletions.append({**pin(duplicate),'reason':'Byte-identical duplicate of the canonical Git measurement receipt.'})
  records.append(pin(p))
 # Generated preparation/patch copies have no independent consumer after source commit.
 for f in [Path('/tmp/strata-prepare-qsa-first-dispatch.py'),Path('/tmp/strata-prepare-qsa-head-entry.py'),A/'qsa-score-headparallel-entry-v1.patch']:
  if f.exists():deletions.append({**pin(f),'reason':'Preparation/patch duplicate; actual source and guarded build/run recipes committed in e3323ed0, and Main source review/actual-entry qualification complete.'})
 rec=dict(schema=1,recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,source=pin(__file__),owner='Main / qualified QSA score headparallel and first-dispatch diagnostics',serialized_lock=str(B/'owned-v0141-measurement.lock'),closed_receipts=records,all_owned_processes_absent=True,warning_error_findings=warnings,deleted=deletions,deleted_logical_bytes=sum(x['bytes'] for x in deletions),retained_consumers=['Canonical immutable receipts and all individual samples in Git.','New engine/binaries/archives/source/build tree retained for actual model comparison and resource analysis.','R1030 and active R1033 source snapshots left intact; no research fixtures or model/RESTORE captures retired.'],free_space_change_claimed=False)
 out.write_text(json.dumps(rec,indent=2)+'\n');assert json.loads(out.read_text())['closed_receipts']==records
 for x in deletions:
  f=Path(x['path']);assert pin(f)=={k:x[k] for k in ['path','bytes','sha256']};f.unlink()
 print(json.dumps(dict(manifest=str(out),files=len(deletions),logical_bytes=rec['deleted_logical_bytes'])))
