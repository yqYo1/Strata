#!/usr/bin/env python3
"""Preserve original strict-build FAIL and rebuild after fixture-only warning fix."""
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
  old_path=A/'own-prompt-k-cellmajor-build-v2.json';fix_path=A/'own-prompt-k-cellmajor-unused-fixture-fix-v1.json'
  old=json.loads(old_path.read_text());fix=json.loads(fix_path.read_text())
  require(not old['passed'] and not old['active'] and not old['GPU_executed'] and 'unused parameter' in old['error'],'exact closed original strict-build failure')
  require(all(module.closed(c) for c in old['commands']),'old ownership closed')
  sources=dict(old['source_pins']);require(sources[fix['path']]==fix['before_sha256'],'same fixture base');sources[fix['path']]=fix['proposed_sha256']
  require(all(sha(P/n)==h for n,h in sources.items()),'only reviewed fixture fix')
  root=B/'main-own-prompt-k-cellmajor-rebuild-root-v3';target=A/'own-prompt-k-cellmajor-build-v3.json';require(not root.exists() and not target.exists(),'fresh rebuild')
  root.mkdir(mode=0o700);build=Path(old['build_path'])
  record=dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],references={p.name:sha(p) for p in (old_path,fix_path)},source_pins=sources,environment=old['environment'],build_path=str(build),compiler=old['compiler'],compiler_sha256=old['compiler_sha256'],canonical_policy_passed=old['canonical_policy_passed'],retained_reference_binary=old['retained_reference_binary'],transformations=old['transformations'],fixture_correction=fix,recipe_sha256=sha(Path(__file__)),component_qualified=False,performance_qualified=False,quality_qualified=False,restore_qualified=False,physical_262144_qualified=False,scope='Fixture-only unused scratch warning correction under strict flags; no kernel/expectation change. Build only; original FAIL unchanged.')
  def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
  owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
  try:
   c,out,err=owner.run('incremental-sycl-rebuild',['cmake','--build',str(build),'--target','strata','sycl_prompt_k_cellmajor_test','--parallel','2'],record['environment'],P,wall=1200,cpu=14400,rss_cap=32<<30,text_cap=16<<20,total_cap=1<<30,file_cap=512<<20)
   require(module.closed(c) and c['exit_code']==0,'rebuild: '+err.read_text()[-3000:]+'\n'+out.read_text()[-3000:])
   require(all(sha(P/n)==h for n,h in sources.items()),'compiled source stable')
   record.update(passed=True,complete=True,binary=str(build/'strata'),binary_sha256=sha(build/'strata'),component_binary=str(build/'sycl_prompt_k_cellmajor_test'),component_binary_sha256=sha(build/'sycl_prompt_k_cellmajor_test'))
  except BaseException as e:record['error']=repr(e);raise
  finally:
   record.update(active=owner.active is not None,finished_utc=module.utc())
   if record['active'] or not all(module.closed(c) for c in owner.commands):record.update(passed=False,complete=False)
   persist()
   with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
  print(json.dumps(dict(passed=record['passed'],binary_sha256=record['binary_sha256'],component_binary_sha256=record['component_binary_sha256'])),flush=True)


if __name__=='__main__':main()
