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
  names=[f'own-qsa-select-full-varied-a{a}-r0-v1.json' for a in range(3)]
  names += [f'own-qsa-select-bench-{p}-a{a}-r{r}-v1.json' for p in ('varied','equal') for a in range(3) for r in range(3)]
  names += ['own-qsa-select-build-v1.json','own-qsa-select-full-comparison-v1.json','own-qsa-select-bench-comparison-v1.json']
  refs={n:json.loads((A/n).read_text()) for n in names};pins={n:sha(A/n) for n in names}
  for n,r in refs.items():
   require(not r['active'] and r['passed'] and r['complete'],'closed successful test/build/analysis')
   for c in r.get('commands',[]):
    require(module.closed(c),'closed ownership')
    for old in c.get('owners',[]):
     now=module.identity(old['pid']);require(now is None or now['start_ticks']!=old['start_ticks'],'owned identity absent')
  decision=refs['own-qsa-select-bench-comparison-v1.json']
  require(not decision['candidate_adopted'] and all(decision['component_positive'].values()),'positive component outcome and model nonadoption preserved')
  roots=[Path(refs[n]['dump']['path']).parent for n in names[:21]]+[B/'main-own-qsa-select-build-root-v1']
  build=refs['own-qsa-select-build-v1.json']
  keep=Path(build['retained_reference_binary']['path'])
  require(sha(keep)==build['retained_reference_binary']['sha256'],'current old-native reference')
  require(sha(build['retained_prompt_component']['path'])==build['retained_prompt_component']['sha256'],'current actualJIT component carrier')
  compact={};files=[]
  for root in roots:
   trace=root/'owned/component.stderr'
   if trace.exists():
    text=trace.read_text();compact[root.name]=dict(stderr_sha256=sha(trace),stderr_bytes=trace.stat().st_size,first=text.splitlines()[:8],last=text.splitlines()[-8:])
   for f in sorted(root.rglob('*')):
    if f.is_file() and f!=keep:
     require(not f.is_symlink(),'no symlink retirement');files.append((f,'Closed component results/dump comparison and individual samples retained; duplicate success traces/requestless metadata/raw outputs or resolved strict-build diagnostics no current consumer.'))
  compile_commands=Path(build['build_path'])/'compile_commands.json'
  if compile_commands.exists():
   entries=json.loads(compile_commands.read_text())
   compact['compiled_changed_files']=[v for v in entries if v['file'].endswith(('/src/kernels/sycl/qsa_select.cpp','/tests/sycl/qsa_select_radix.cpp'))]
   require(len(compact['compiled_changed_files'])==2,'both actual compiler command entries')
   compact['compile_commands_sha256']=sha(compile_commands)
  compact['build_stdout']=(roots[-1]/'owned/incremental-sycl-build.stdout').read_text()
  require(len(compact['build_stdout'])<32768,'compact successful build output')
  for name in ('own-qsa-select-radix4-v1','own-qsa-select-count-mapping-v1'):
   folder=A/name;proposal=json.loads((folder/'manifest.json').read_text())
   for subtree in ('base','files'):
    for f in sorted((folder/subtree).rglob('*')):
     if f.is_file():
      relative=str(f.relative_to(folder/subtree))
      expected=proposal['files'][subtree+'/'+relative]['sha256'] if name.endswith('radix4-v1') else proposal['files'][relative]['proposal_sha256']
      require(sha(f)==expected,'exact reviewed full mirror')
      files.append((f,'Applied own selector source mirror; compact patch/manifest/prepare and current source inGit. Comparison consumers use qualified actual source, not duplicate mirrors.'))
  output=A/'closed-own-qsa-select-compact-v1.json';target=A/'closed-own-qsa-select-retention-v1.json';require(not output.exists() and not target.exists(),'fresh retention')
  output.write_text(json.dumps(dict(passed=True,complete=True,active=False,original_receipts=pins,evidence=compact,recipe_sha256=sha(Path(__file__))),indent=2)+'\n')
  manifest=dict(passed=True,complete=False,active=False,original_receipts=pins,compact=dict(path=output.name,sha256=sha(output)),
   files=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f),reason=reason) for f,reason in files],
   kept_reference=dict(path=str(keep),bytes=keep.stat().st_size,sha256=sha(keep),consumer='Main own-qsa-select-radix4-v1 old/defaultOFF whole-prefill regression reference',budget_bytes=32<<20,next_review='after selector candidate rejection/adoption or reference replacement'),original_FAIL_PASS_samples_unchanged=True,finished_utc=module.utc())
  target.write_text(json.dumps(manifest,indent=2)+'\n')
  require(all(sha(A/n)==h for n,h in pins.items()),'original receipts unchanged before retirement')
  for entry in manifest['files']:
   f=Path(entry['path']);require(sha(f)==entry['sha256'] and f.stat().st_size==entry['bytes'],'exact file identity');f.unlink()
  require(all(sha(A/n)==h for n,h in pins.items()),'original receipt statuses/bytes unchanged')
  manifest.update(complete=True,files_retired=len(files),logical_bytes_retired=sum(f['bytes'] for f in manifest['files']));target.write_text(json.dumps(manifest,indent=2)+'\n')
  print(json.dumps({k:manifest[k] for k in ('files_retired','logical_bytes_retired')}),flush=True)


if __name__=='__main__':main()
