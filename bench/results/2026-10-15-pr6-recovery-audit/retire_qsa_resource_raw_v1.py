#!/usr/bin/env python3
"""Retire closed, committed diagnostic duplicates; no source/fixture status changes."""
import fcntl,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2];B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 target=A/'qsa-resource-raw-retirement-v1.json';assert not target.exists()
 ownerpath=P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py';assert sha(ownerpath)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
 spec=importlib.util.spec_from_file_location('qsa_raw_retirement_owner',ownerpath);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 names=['own-pr6-provider-contract-v1.json','own-qsa-resource-smoke-v1.json','own-qsa-resource-first-v1.json','own-qsa-resource-clean-v1.json','own-qsa-resource-comparison-v1.json','own-qsa-resource-build-v1.json','own-qsa-resource-build-v2.json','own-qsa-resource-build-v3.json','own-pr6-harness-build-v2.json','own-pr6-harness-build-v3.json','own-pr6-provider-build-v1.json']
 roots=['main-pr6-provider-contract-v1','main-qsa-resource-smoke-v1','main-qsa-resource-first-v1','main-qsa-resource-clean-v1','main-qsa-resource-analysis-v1','main-qsa-resource-build-v1','main-qsa-resource-build-v2','main-qsa-resource-build-v3','main-pr6-harness-build-v2','main-pr6-harness-build-v3','main-pr6-provider-build-v1']
 receipts={name:json.loads((A/name).read_text()) for name in names}
 def committed(path):assert subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(P))],cwd=P)==path.read_bytes(),str(path)
 planned={}
 def add(path,reason):
  path=Path(path);assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(B.resolve()),str(path)
  planned[str(path)]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),reason=reason,removed=False)
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name,rootname in zip(names,roots):
   p=A/name;j=receipts[name];committed(p);assert not j['active'] and j['ownership_closed']
   for c in j['commands']:
    assert m.closed(c)
    for old in c['owners']+[c['direct_identity']]:
     now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
   root=B/rootname
   record=root/'record.json';assert record.read_bytes()==p.read_bytes();add(record,'Byte-identical duplicate of committed structured receipt; not a resume checkpoint.')
   for q in (root/'owned').glob('*'):
    if q.is_file() and q.suffix in ('.stdout','.stderr'):add(q,'Closed full API/build/comparator output: exactcase stdout/timings/checks/exit/maps retained in committed receipt; distinct compile errors preserved in committed representative windows.')
   cache=root/'cache'
   if cache.exists():
    for q in cache.rglob('*'):
     if q.is_file() and not q.is_symlink():add(q,'Closed diagnostic private shader cache; warmups excluded and no current fixture consumer.')
  comp=receipts['own-qsa-resource-comparison-v1.json'];assert comp['passed'] and comp['complete'] and comp['all_full_frames_equal']
  for row in comp['frames']:
   assert sha(row['path'])==row['sha256'];add(row['path'],'All10 initial frames fully finite/byte-compared and fingerprints committed; no unresolved numerical mismatch or future RESTORE consumer.')
  old=receipts['own-qsa-resource-build-v2.json']['binary'];assert old['sha256']!='6be5e0d378233e831cdb839e52b398f4da5a4a1909cef9780f77f6ff452b017c';assert sha(old['path'])==old['sha256'];add(old['path'],'Superseded ungated fixture binary; actual11nativeprocess cohort used retained v3 binary.')
  profilepath=A/'hardware-resource-profile-v1.json';modelpath=A/'own-vllm-resource-timeline-model-v4.json'
  committed(profilepath);committed(modelpath);profile=json.loads(profilepath.read_text());model=json.loads(modelpath.read_text())
  assert model['passed'] and model['complete'] and not model['active']
  assert len(profile['per_name_count_sum_min_max_reconciliation']['rows'])==154 and len(profile['long_QSA_samples'])==60
  for c in model['commands']:
   assert m.closed(c)
   for old in c['owners']+[c['direct_identity']]:
    now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
  sample=profile['sample_pin'];assert sha(sample['path'])==sample['sha256']=='f0593196a2167882ad15c7fe86a5672d3a808f235a7d5181eddad5aa442dcd23'
  add(sample['path'],'Canonical raw profiler timeline currentconsumer closed: all154 name summaries exactly reconciled,60QSA samples and unions/copybytes/gaps/CPU facts committed; OA collection branch stopped and matched component comparison complete. Remaining causalhost gaps/internalunit scarcity require different evidence, not this unpaired raw timeline.')
  for name in ['pr6-build-v2-compile-error-window-v1.json','pr6-build-v2-compile-error-window-v1.stdout','pr6-build-v3-compile-error-window-v1.json','pr6-build-v3-compile-error-window-v1.stdout','hardware-resource-followup-v7.json']:committed(A/name)
  record=dict(schema='strata-qsa-resource-raw-retirement-v1',passed=False,complete=False,GPU_executed=False,source_sha256=sha(__file__),original_receipt_statuses_unchanged=True,individual_measurement_samples_kept=True,
   all_original_receipts_verified_in_Git=True,frames_bytecomparison_consumed=True,
   canonical_timeline_consumer_closed=True,prior_canonical64MiB_exception_superseded=True,
   canonical_retention_reason='Per-name154 exact summaries,60individualQSA samples and all scoped busy/copy/CPU findings in Git; no named current consumer requires all534498 timestamp rows.',
   retained=['Current resource fixture v3 binary+source and frozen current archives for next controlled candidate; review at next component cohort/2026-10-17T03:00Z.',
    'Original all individual timings/failed statuses/ownership/environment/runtime identities and compact compile windows in Git.',
    'PR6dependency snapshots/build inputs/providerbinary retained for next fullbuild and native64bit oracle.',
    'No model/tensor/session/physical262144/RESTORE fixtures or unresolved xe driver dumps selected.'],
   files=list(planned.values()),total_logical_bytes=sum(v['bytes'] for v in planned.values()))
  target.write_text(json.dumps(record,indent=2)+'\n')
  for row in record['files']:
   path=Path(row['path']);assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'];path.unlink();row['removed']=True;target.write_text(json.dumps(record,indent=2)+'\n')
  record.update(passed=True,complete=True);target.write_text(json.dumps(record,indent=2)+'\n')
  print(json.dumps(dict(receipt=str(target),files=len(record['files']),bytes=record['total_logical_bytes'],MiB=record['total_logical_bytes']/2**20,all_removed=all(v['removed'] for v in record['files']))))
if __name__=='__main__':main()
