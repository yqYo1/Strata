#!/usr/bin/env python3
"""Retire reviewed successful component traces/dumps; preserve original FAIL/PASS."""
import fcntl,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2];B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha


def main():
 module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  names=[f'own-prompt-k-cellmajor-full-d{d}-a{a}-r0-v2.json' for d in (0,1) for a in (0,1)]
  names += [f'own-prompt-k-cellmajor-bench-d1-a{a}-r{r}-v2.json' for a in (0,1) for r in range(3)]
  names += ['own-prompt-k-cellmajor-build-v2.json','own-prompt-k-cellmajor-build-v3.json','own-prompt-k-cellmajor-comparison-v2.json','own-prompt-k-cellmajor-full-comparison-v2.json']
  refs={n:json.loads((A/n).read_text()) for n in names};pins={n:sha(A/n) for n in names}
  for n,r in refs.items():
   require(not r['active'],'closed receipt')
   if n=='own-prompt-k-cellmajor-build-v2.json':require(not r['passed'] and not r['complete'] and not r['GPU_executed'] and 'unused parameter' in r['error'],'original build FAIL unchanged')
   else:require(r['passed'] and r['complete'],'closed successful test/build/analysis')
   for c in r.get('commands',[]):
    require(module.closed(c),'closed ownership')
    for old in c.get('owners',[]):
     now=module.identity(old['pid']);require(now is None or now['start_ticks']!=old['start_ticks'],'owned identity absent')
  require(not refs['own-prompt-k-cellmajor-comparison-v2.json']['candidate_adopted'],'negative component decision preserved')
  roots=[Path(refs[n]['dump']['path']).parent for n in names[:10]]+[B/'main-own-prompt-k-cellmajor-build-root-v2',B/'main-own-prompt-k-cellmajor-rebuild-root-v3']
  keep=Path(refs['own-prompt-k-cellmajor-build-v3.json']['retained_reference_binary']['path'])
  require(sha(keep)==refs['own-prompt-k-cellmajor-build-v3.json']['retained_reference_binary']['sha256'],'one current old-native reference')
  compact={};files=[]
  for root in roots:
   trace=root/'owned/component.stderr'
   if trace.exists():
    text=trace.read_text();compact[root.name]=dict(stderr_sha256=sha(trace),stderr_bytes=trace.stat().st_size,first=text.splitlines()[:8],last=text.splitlines()[-8:])
   for f in sorted(root.rglob('*')):
    if f.is_file() and f!=keep:
     require(not f.is_symlink(),'no symlink retirement');files.append((f,'Closed component results/dump comparison and individual samples retained; duplicate success traces/requestless metadata/raw outputs or resolved strict-build diagnostics no current consumer.'))
  compact['resolved_build_error']=refs['own-prompt-k-cellmajor-build-v2.json']['error']
  compact['fixture_fix_sha256']=sha(A/'own-prompt-k-cellmajor-unused-fixture-fix-v1.json')
  for version in ('own-prompt-k-cellmajor-v1','own-prompt-k-cellmajor-v2'):
   folder=A/version;manifest=json.loads((folder/'manifest.json').read_text())
   for f in sorted((folder/'files').rglob('*')):
    if f.is_file():
     relative=str(f.relative_to(folder/'files'));require(sha(f)==manifest['files'][relative]['proposal_sha256'],'exact reviewed full mirror')
     files.append((f,'Superseded completed proposal full mirror; compact diff/manifest/reconstruction and actual source inGit. Padded proposal has its own current snapshot.'))
  output=A/'closed-prompt-k-component-compact-v2.json';target=A/'closed-prompt-k-component-retention-v2.json';require(not output.exists() and not target.exists(),'fresh retention')
  output.write_text(json.dumps(dict(passed=True,complete=True,active=False,original_receipts=pins,evidence=compact,recipe_sha256=sha(Path(__file__))),indent=2)+'\n')
  manifest=dict(passed=True,complete=False,active=False,original_receipts=pins,compact=dict(path=output.name,sha256=sha(output)),
   files=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f),reason=reason) for f,reason in files],
   kept_reference=dict(path=str(keep),bytes=keep.stat().st_size,sha256=sha(keep),consumer='Main own-prompt-k-padded-v1 old/defaultOFF whole-prefill regression reference',budget_bytes=32<<20,next_review='after padded candidate rejection/adoption or reference replacement'),original_FAIL_PASS_samples_unchanged=True,finished_utc=module.utc())
  target.write_text(json.dumps(manifest,indent=2)+'\n')
  require(all(sha(A/n)==h for n,h in pins.items()),'original receipts unchanged before retirement')
  for entry in manifest['files']:
   f=Path(entry['path']);require(sha(f)==entry['sha256'] and f.stat().st_size==entry['bytes'],'exact file identity');f.unlink()
  require(all(sha(A/n)==h for n,h in pins.items()),'original receipt statuses/bytes unchanged')
  manifest.update(complete=True,files_retired=len(files),logical_bytes_retired=sum(f['bytes'] for f in manifest['files']));target.write_text(json.dumps(manifest,indent=2)+'\n')
  print(json.dumps({k:manifest[k] for k in ('files_retired','logical_bytes_retired')}),flush=True)


if __name__=='__main__':main()
