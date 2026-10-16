#!/usr/bin/env python3
"""Main serial GU-diagnostic retirement and incremental actual SYCL build."""
import fcntl,json,os,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,P,load_owner,require,sha

def main():
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  previous=A/'license-clean-prefill-build-v1.json';old=json.loads(previous.read_text())
  require(old['passed'] and old['complete'] and not old['active'],'prior build closed')
  require(all(sha(P/n)==h for n,h in old['source_pins'].items()),'unchanged post-v1 source')
  proposal=A/'remove-xestrata-diagnostics-v2';pins=json.loads((proposal/'pins.json').read_text());patch=proposal/'after-v1.diff'
  require(sha(patch)==pins['diff_sha256']=='035ebf5710d58ea4c74604827ea81f63489d12c480559e606d5bdc84b95fbf43','reviewed full GU retirement')
  for n,v in pins['files'].items():require(sha(P/n)==v['base_sha256'],'exact GU patch base '+n)
  root=B/'main-license-clean-prefill-build-root-v2';target=A/'license-clean-prefill-build-v2.json';require(not root.exists() and not target.exists(),'fresh build record')
  build=Path(old['build_path']);env=old['environment'];module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
  for c in old['commands']:
   require(module.closed(c),'prior build owners closed')
   for ident in c['owners']:
    now=module.identity(ident['pid']);require(now is None or now['start_ticks']!=ident['start_ticks'],'prior build owner retired')
  products={str(p):sha(p) for p in build.rglob('*') if p.is_file() and (p.suffix in ('.o','.a') or p.name=='strata')}
  root.mkdir(mode=0o700)
  record=dict(passed=False,complete=False,active=True,applied=False,GPU_executed=False,commands=[],environment=env,
   recipe_sha256=sha(Path(__file__)),references={p.name:sha(p) for p in (previous,proposal/'pins.json',A/'research-cpu-gu-origin-round717-v1.txt')},
   source_revision=old['source_revision'],patch_sha256=sha(patch),removal_files=pins['files'],build_path=str(build),
   unchanged_safety_and_math=old['unchanged_safety_and_math'],compiler=old['compiler'],compiler_sha256=old['compiler_sha256'],
   configure_argv=old['configure_argv'],compile_flag_proof=old['compile_flag_proof'],
   toolchain_location_note=old['toolchain_location_note'],quality_qualified=False,performance_qualified=False,restore_qualified=False,physical_model_262144_qualified=False,
   scope='Known derived route/service and provenance-unknown GU diagnostics retired; actual SYCL recompile only. No blanket license, model runtime/quality, RESTORE or performance certification.')
  def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
  owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
  def run(label,argv,wall=60):
   c,out,err=owner.run(label,argv,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),rss_cap=32<<30,text_cap=8<<20,total_cap=3<<30,file_cap=256<<20)
   require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-4000:]+'\n'+out.read_text()[-2500:]);print('PASS '+label,flush=True)
  try:
   run('check-GU-retirement',['git','apply','--check',str(patch)])
   run('apply-GU-retirement',['git','apply',str(patch)]);record['applied']=True
   for n,v in pins['files'].items():require(not(P/n).exists() if v['deleted'] else sha(P/n)==v['proposed_sha256'],'exact GU retirement '+n)
   require(all(sha(P/n)==h for n,h in record['unchanged_safety_and_math'].items()),'independent safety and math unchanged')
   names=subprocess.check_output(['git','ls-files','--','src','include','cmake','CMakeLists.txt','third_party','tests/sycl'],cwd=P,text=True).splitlines()
   record['source_pins']={n:sha(P/n) for n in sorted(names) if (P/n).is_file()};persist()
   run('build-license-clean-v2',['cmake','--build',str(build),'--target','strata','--parallel','2'],1800)
   require(all(sha(P/n)==h for n,h in record['source_pins'].items()),'source stable during build')
   record['superseded_prior_build_products']=[dict(path=n,old_sha256=h,new_sha256=sha(Path(n)),reason='Unexecuted intermediate GU-unbound build superseded by full diagnostic retirement; no current v1 binary consumer.') for n,h in products.items() if sha(Path(n))!=h]
   record.update(binary=str(build/'strata'),binary_sha256=sha(build/'strata'),compile_commands_sha256=sha(build/'compile_commands.json'),passed=True,complete=True)
  except BaseException as error:record['error']=repr(error);raise
  finally:
   record.update(active=owner.active is not None,finished_utc=module.utc())
   if record['active'] or not all(module.closed(c) for c in owner.commands):record.update(passed=False,complete=False)
   persist()
   with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
  print(json.dumps(dict(passed=record['passed'],binary_sha256=record.get('binary_sha256'))),flush=True)
if __name__=='__main__':main()
