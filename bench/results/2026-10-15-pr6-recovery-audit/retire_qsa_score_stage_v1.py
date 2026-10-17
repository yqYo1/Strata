#!/usr/bin/env python3
"""Retire closed successful QSA score traces and Git-restorable research mirrors."""
import datetime,fcntl,hashlib,importlib.util,json,re,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=A/'qsa-score-stage-retention-v1.json';assert not out.exists()
 assert json.loads((A/'qsa-score-querytile-decision-v1.json').read_text())['complete']
 helper=A/'direct_owner_transient_artifacts_v1.py'
 assert sha(helper)=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
 s=importlib.util.spec_from_file_location('qsa_score_retention_owner',helper);owned=importlib.util.module_from_spec(s);s.loader.exec_module(owned)
 names=['qsa-select-stage-build-v1','qsa-select-stage-smoke-v1','qsa-score-querytile-build-v1']
 names += [f'qsa-select-stage-{ctx}-{rep}-v1' for ctx in [32768,78453,131072,262144] for rep in range(3)]
 names += [f'qsa-score-querytile-smoke-{tile}-v1' for tile in [0,2,4,8]]
 names += [f'qsa-score-querytile-{tile}-78453-0-v1' for tile in [2,4]]
 names += [f'qsa-score-querytile-{tile}-{ctx}-{rep}-v1' for ctx in [32768,78453,262144] for tile in [0,8] for rep in range(3)]
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  deletions=[];originals={};warnings=[]
  def select(path,reason,replacement,restore=None):
   assert path.is_file() and not path.is_symlink()
   row=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),reason=reason,replacement=replacement)
   if restore:row['restore_from_git']=restore
   deletions.append(row)
  for name in names:
   receipt=A/(name+'.json');v=json.loads(receipt.read_text())
   assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['active']
   assert not v.get('fresh_faults') and not v.get('driver_dump_entries')
   originals[str(receipt)]=sha(receipt)
   for c in v['commands']:
    assert owned.closed(c) and c['exit_code']==0
    for old in c['owners']+[c['direct_identity']]:
     now=owned.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
   root=B/('main-'+name)
   for raw in (root/'owned').rglob('*'):
    if raw.is_file() and raw.suffix in ['.stdout','.stderr','.log','.jsonl']:
     if raw.suffix=='.stderr':
      lines=raw.read_text().splitlines()
      selected=sorted(set(x for x in lines if re.search(r'(?i)\b(warning|error|failed|failure|unsupported|not supported)\b',x)))
      # Keep unusual diagnostics for an explicit review instead of discarding them.
      if selected:
       warnings.append(dict(path=str(raw),sha256=sha(raw),lines=selected[:50],count=len(selected),consumer='Main review any unexpected warning before retiring this raw stderr',review_point='Next QSA scorer entry integration'))
       continue
     select(raw,'Closed successful API/build/journal text; cases, timing samples, runtime mappings, kernel counts, ownership and fault boundaries preserved.',str(receipt))
   duplicate=root/'record.json'
   if duplicate.exists():
    assert json.loads(duplicate.read_text())==v
    select(duplicate,'Exact structured duplicate of retained original receipt.',str(receipt))
  sourcepin=A/'research-qsa-score-layout-round1024-source-pin-v1.json';pin=json.loads(sourcepin.read_text());originals[str(sourcepin)]=sha(sourcepin)
  for name,row in pin['files'].items():
   path=Path(row['path']);raw=subprocess.check_output(['git','show',pin['commit']+':'+name],cwd=A.parents[2]);assert hashlib.sha256(raw).hexdigest()==row['sha256']==sha(path)
   select(path,'Research and native review settled; exact source is reachable in Git and no live worker consumes the mirror.',str(sourcepin),dict(commit=pin['commit'],path=name))
  mirror=Path(next(iter(pin['files'].values()))['path']).parents[3]/'manifest.json'
  # The source root is explicit; never infer and traverse outside this named mirror.
  mirror=B/'research-qsa-score-layout-round1024-v1/manifest.json'
  assert json.loads(mirror.read_text())==pin
  select(mirror,'Duplicate source-pin manifest retained in repository.',str(sourcepin))
  rec=dict(schema='strata-qsa-score-stage-retention-v1',complete=False,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=deletions,original_receipts=originals,warnings_kept=warnings,deleted=0,logical_bytes_removed=0,
   scope='Only closed successful raw text, exact receipt duplicates and Git-restorable research mirrors. Keep candidate source/binary/object/static archives for named next engine-entry integration and full-score baseline comparison; preserve all original outcomes and all timing samples.')
  def save():out.write_text(json.dumps(rec,indent=2)+'\n')
  save()
  try:
   for row in deletions:
    path=Path(row['path']);assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes'];path.unlink()
    rec['deleted']+=1;rec['logical_bytes_removed']+=row['bytes'];save()
   for p,h in originals.items():assert sha(p)==h
   rec['complete']=True
  finally:save()
 print(json.dumps({k:rec[k] for k in ['complete','deleted','logical_bytes_removed','warnings_kept']}),flush=True)
if __name__=='__main__':main()
