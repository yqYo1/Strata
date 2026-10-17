#!/usr/bin/env python3
"""Closed model raw-log/duplicate and unconsumed first-build retirement."""
import datetime,fcntl,hashlib,importlib.util,json,os,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def pin(p):
 p=Path(p)
 if p.is_symlink():return dict(path=str(p),kind='symlink',link=os.readlink(p),bytes=len(os.readlink(p).encode()))
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),kind='file',bytes=p.stat().st_size,sha256=h)
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 out=A/'qsa-score-gemm-entry-model-retention-v1.json';assert not out.exists()
 comparison=A/'qsa-score-gemm-entry-model-comparison-v1.json';cmp=json.loads(comparison.read_text());assert cmp['passed'] and cmp['complete']
 op=A/'direct_owner_transient_artifacts_v1.py';assert pin(op)['sha256']=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a';owner=module('gemm_model_retire_owner',op)
 pkg=A/'own-vllm-xe2-qualification-v1';meta=json.loads((pkg/'manifest.json').read_text());assert pin(pkg/'manifest.json')['sha256']=='7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58';assert pin(pkg/'main_controller.py')['sha256']==meta['package_files']['main_controller.py']['sha256'];guard=module('gemm_model_retire_guard',pkg/'main_controller.py')
 receipts=[A/x['receipt'] for x in cmp['arms']];receipts.append(A/'qsa-score-gemm-entry-engine-build-v1.json')
 rec=dict(schema=1,passed=False,complete=False,source=pin(__file__),comparison=pin(comparison),owner='Main / closed integrated score-GEMM model comparison',canonical=[],retained_diagnostics=[],deleted=[],retained_consumers=['Current full v2 engine/archives remain for current-path profiling, actual serial/subgroup fixture and quality work.','All canonical result bytes and individual samples remain in Git. Model entry/exit identity records and generated reproducible wrapper/config sources retained until quality/adoption review.'],free_space_change_claimed=False)
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard.no_model_owners(meta['known_model_files']);guard.driver_guard()
  deletions=[];dirs=[]
  for receipt in receipts:
   d=json.loads(receipt.read_text());assert d['passed'] and d['complete'] and d['ownership_closed'] and not d['active'] and d['fresh_faults']==[] and d['boot_unchanged'];assert not d.get('driver_dump_exists') and not d.get('driver_dump_entries');rec['canonical'].append(pin(receipt))
   for c in d['commands']:
    assert owner.closed(c) and c['exit_code']==0
    for old in c['owners']+[c['direct_identity']]:
     now=owner.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
   if 'config' not in d:continue
   root=Path(d['config']['log']).parents[1];assert root.parent==B and not root.is_symlink()
   for f in (root/'owned').iterdir():
    assert f.is_file() and not f.is_symlink();text=f.read_text(errors='replace')
    keep=[x for x in text.splitlines() if re.search(r'(?i)warning:|error|failed|cleanup|retire|^strata mtp: prompt expert phase=|^strata generate: loaded |^strata serve: prompt |^strata MoE |^strata QSA score prefill:',x)]
    assert sum(len(x.encode())+1 for x in keep)<65536
    if keep:rec['retained_diagnostics'].append(dict(path=str(f),lines=keep))
    deletions.append({**pin(f),'retained_in':str(receipt),'reason':'Closed successful raw protocol/progress/journal log; all request samples, identity, env, status, source and no-fault facts in canonical receipt; nonduplicate allocator/transfer/cache/admission diagnostic lines retained here.'})
   duplicates=[(root/'result.json',d['result'])]
   duplicates += [(f,d['benchmark_result']) for f in (root/'stock-results').glob('*/results.json')]
   for f,expected in duplicates:
    if f.exists():
     assert json.loads(f.read_text())==expected;deletions.append({**pin(f),'retained_in':str(receipt),'reason':'Semantic JSON duplicate already embedded in canonical model receipt; exact original file hash recorded.'})
  # First full build was never GPU-executed and has no remaining consumer;
  # the admission-fixed v2 is independently built, qualified and still retained.
  tree=B/'main-qsa-score-gemm-entry-engine-build-v1/build';assert tree.is_dir() and not tree.is_symlink()
  for current,subdirs,files in os.walk(tree,followlinks=False):
   here=Path(current);dirs.append(here)
   for name in list(subdirs):
    q=here/name
    if q.is_symlink():deletions.append({**pin(q),'reason':'Superseded unexecuted v1 build output, no current consumer; Git source64ec70ec, compile/provenance/binary hashes in canonical build receipt and independent qualified v2 retained.'});subdirs.remove(name)
   for name in files:deletions.append({**pin(here/name),'reason':'Superseded unexecuted v1 build output, no current consumer; source64ec70ec/commands/compiler/source/output hashes remain in canonical receipt; independent qualified v2 engine and archives retained.'})
  assert len({x['path'] for x in deletions})==len(deletions)
  rec.update(deletion_manifest=deletions,planned_file_count=len(deletions),planned_logical_bytes=sum(x['bytes'] for x in deletions),planned_empty_directories=[str(x) for x in sorted(dirs,key=lambda p:len(p.parts),reverse=True)],deleted_file_count=0,deleted_logical_bytes=0,removed_empty_directories=[],recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  out.write_text(json.dumps(rec,indent=2)+'\n')
  try:
   for x in deletions:
    f=Path(x['path']);observed=pin(f);assert observed=={k:x[k] for k in observed};f.unlink();rec['deleted_file_count']+=1;rec['deleted_logical_bytes']+=x['bytes']
   for f in sorted(dirs,key=lambda p:len(p.parts),reverse=True):f.rmdir();rec['removed_empty_directories'].append(str(f))
   for x in rec['canonical']:assert pin(x['path'])==x
   rec.update(passed=True,complete=True,canonical_bytes_unchanged=True)
  except BaseException as e:
   rec['error']=repr(e);raise
  finally:out.write_text(json.dumps(rec,indent=2)+'\n')
 print(json.dumps(dict(manifest=str(out),files=rec['deleted_file_count'],logical_bytes=rec['deleted_logical_bytes'])))
if __name__=='__main__':main()
