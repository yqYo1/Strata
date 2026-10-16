#!/usr/bin/env python3
"""Main-owned removal and fresh actual SYCL build. No GPU execution."""
import fcntl,json,os,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,P,load_owner,require,sha

def main():
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  proposal=A/'remove-xestrata-diagnostics-v1'
  pins=json.loads((proposal/'pins.json').read_text()); patch=proposal/'compactsource.diff'
  require(sha(patch)==pins['diff_sha256']=='a15e84f29ddcb1009a0db268311809371ac3cc8ec58b1c1c50b1e6a5a4348fa3','complete reviewed removal')
  require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip()==pins['immutable_base'],'immutable source base')
  for n,v in pins['files'].items():require(sha(P/n)==v['base_sha256'],'exact source base '+n)
  module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
  diagnostic=A/'native-dense-bank-direct1-v1.json'; comparison=A/'native-dense-bank-direct-comparison-v1.json'
  for path in (diagnostic,comparison):
   r=json.loads(path.read_text());require(r['passed'] and r['complete'] and not r['active'],'preceding actual comparison closed')
   for c in r.get('commands',[]):
    require(module.closed(c),'prior owned command closed')
    for old in c['owners']:
     now=module.identity(old['pid']);require(now is None or now['start_ticks']!=old['start_ticks'],'prior owner retired')
  require(not r.get('first_difference'),'same-native Direct comparison exact')
  seed=json.loads((B/'main-primary-native-build-root-v1/record.json').read_text())
  require(seed['passed'] and seed['complete'] and not seed['active'],'qualified configure source')
  env=dict(seed['environment']);compiler=Path(seed['compiler']['path'])
  require(sha(compiler)==seed['compiler']['sha256'],'unchanged compiler')
  require(not any('TOKEN' in k or 'API_KEY' in k or k=='LD_PRELOAD' or k.startswith(('UNITRACE_','PTI_')) for k in env),'bounded build environment')
  root=B/'main-license-clean-prefill-build-root-v1';target=A/'license-clean-prefill-build-v1.json'
  require(not root.exists() and not target.exists(),'fresh build')
  build=root/'build';configure=list(next(c['argv'] for c in seed['commands'] if c['label']=='configure-primary-native'))
  configure[configure.index('-B')+1]=str(build)
  configure+=['-DSTRATA_PREFILL_MMQ=OFF','-DSTRATA_BUILD_PREFILL_HOST_TESTS=ON']
  invariant_names=['src/core/mtp.cpp','src/core/verify.cpp','src/core/session.cpp','src/prefill/prefill.cpp','src/prefill/layer_major.inc','src/kernels/sycl/prompt_attention.cpp','src/kernels/sycl/decode_attention.cpp']
  invariant_pins={n:sha(P/n) for n in invariant_names}
  root.mkdir(mode=0o700)
  record=dict(passed=False,complete=False,active=True,GPU_executed=False,applied=False,commands=[],environment=env,
   recipe_sha256=sha(Path(__file__)),source_revision=pins['immutable_base'],patch_sha256=sha(patch),removal_files=pins['files'],
   references={p.name:sha(p) for p in (diagnostic,comparison,proposal/'pins.json',A/'research-xestrata-removal-hooks-round715-v1.txt',A/'research-xestrata-provenance-round716-v1.txt')},
   compiler=str(compiler),compiler_sha256=sha(compiler),configure_argv=configure,build_path=str(build),unchanged_safety_and_math=invariant_pins,
   scope='Removal of XeStrata-derived decode route/service diagnostics and actual fresh production SYCL compile. No numerical, runtime, physical context, RESTORE, task-quality or performance qualification.',
   performance_qualified=False,quality_qualified=False,restore_qualified=False,physical_model_262144_qualified=False,
   GU_observer_request_binding_available=False,
   toolchain_location_note='Unchanged official IntelLLVM/runtime installation lives under a reference worktree tools directory. This build imports no XeStrata source. Installation path is not source provenance.')
  def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
  owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
  def run(label,argv,wall=90):
   c,out,err=owner.run(label,argv,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),rss_cap=32<<30,text_cap=8<<20,total_cap=3<<30,file_cap=256<<20)
   require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-4000:]+'\n'+out.read_text()[-2500:])
   print('PASS '+label,flush=True)
  try:
   run('check-removal-patch',['git','apply','--check',str(patch)])
   run('apply-removal-patch',['git','apply',str(patch)]);record['applied']=True
   for n,v in pins['files'].items():
    require(not (P/n).exists() if v['deleted'] else sha(P/n)==v['proposed_sha256'],'exact removal result '+n)
   require(all(sha(P/n)==h for n,h in invariant_pins.items()),'independent safety and math retained')
   for n in ('src/core/expert_source.cpp','src/program/generate.cpp','include/strata/core/expert_source.hpp','include/strata/kernels/cpu/pool.hpp','src/kernels/cpu/pool.cpp'):
    text=(P/n).read_text()
    require(not any(x in text for x in ('DecodeRouteLedger','DecodeServiceBudgetScope','NativeServiceBudget','service_budget_','decode_service.','decode_observer','decode_service_budget.hpp','decode_route_ledger.hpp')),'no derived route/service implementation '+n)
   names=subprocess.check_output(['git','ls-files','--','src','include','cmake','CMakeLists.txt','third_party','tests/sycl'],cwd=P,text=True).splitlines()
   record['source_pins']={n:sha(P/n) for n in sorted(names) if (P/n).is_file()};persist()
   run('configure-license-clean',configure,300)
   run('build-license-clean',['cmake','--build',str(build),'--target','strata','--parallel','2'],1800)
   cc=json.loads((build/'compile_commands.json').read_text());proof={}
   for n in ('src/program/generate.cpp','src/core/expert_source.cpp','src/kernels/cpu/pool.cpp','src/core/mtp.cpp','src/prefill/prefill.cpp'):
    rows=[x for x in cc if x['file']==str(P/n)]
    require(rows,'actual production source resolver '+n)
    if '/cpu/' not in n:require(all('-fsycl-id-queries-range=size_t' in x['command'] for x in rows),'large context size_t launch '+n)
    proof[n]=rows
   require(all(sha(P/n)==h for n,h in record['source_pins'].items()),'source stable during build')
   record.update(compile_flag_proof=proof,compile_commands_sha256=sha(build/'compile_commands.json'),binary=str(build/'strata'),binary_sha256=sha(build/'strata'),passed=True,complete=True)
  except BaseException as e:record['error']=repr(e);raise
  finally:
   record.update(active=owner.active is not None,finished_utc=module.utc())
   if record['active'] or not all(module.closed(c) for c in owner.commands):record.update(passed=False,complete=False)
   persist()
   with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
  print(json.dumps(dict(passed=record['passed'],binary_sha256=record.get('binary_sha256'))),flush=True)
if __name__=='__main__':main()
