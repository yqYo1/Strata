#!/usr/bin/env python3
"""Analyze closed component arms; preserve every sample and reject the slower candidate."""
import fcntl,json,statistics,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2];B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha


def main():
 module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  target=A/'own-prompt-k-cellmajor-comparison-v2.json';require(not target.exists(),'fresh component comparison')
  names=[f'own-prompt-k-cellmajor-full-d{d}-a{a}-r0-v2.json' for d in (0,1) for a in (0,1)]
  order=[(0,0),(1,0),(1,1),(0,1),(0,2),(1,2)]
  names += [f'own-prompt-k-cellmajor-bench-d1-a{a}-r{r}-v2.json' for a,r in order]
  records={n:json.loads((A/n).read_text()) for n in names}
  for r in records.values():
   require(r['passed'] and r['complete'] and not r['active'] and not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists'],'closed healthy arm')
   for c in r['commands']:
    require(module.closed(c),'command closed')
    for old in c.get('owners',[]):
     now=module.identity(old['pid']);require(now is None or now['start_ticks']!=old['start_ticks'],'identity retired')
   require(sha(Path(r['dump']['path']))==r['dump']['sha256'],'raw identity')
  reference=records[names[0]]
  require(all(r['source_pins']==reference['source_pins'] and r['binary_sha256']==reference['binary_sha256'] and r['boot_id']==reference['boot_id'] for r in records.values()),'same compiled source/binary/boot')
  for d in (0,1):require(records[f'own-prompt-k-cellmajor-full-d{d}-a0-r0-v2.json']['dump']['sha256']==records[f'own-prompt-k-cellmajor-full-d{d}-a1-r0-v2.json']['dump']['sha256'],'full exact OFF/ON')
  bench=[records[n] for n in names[4:]]
  ignored={'STRATA_SYCL_PROMPT_K_CELL_MAJOR','XDG_CACHE_HOME'}
  require(all({k:v for k,v in r['environment'].items() if k not in ignored}=={k:v for k,v in bench[0]['environment'].items() if k not in ignored} for r in bench),'matched clean environment')
  require(all('UR_ENABLE_LAYERS' not in r['environment'] and 'UR_LOG_TRACING' not in r['environment'] for r in bench),'no collector/validation in timed arms')
  require(len({r['dump']['sha256'] for r in bench})==1,'all256query dumps byteequal')
  actual=sorted(bench,key=lambda r:next(c['started_utc'] for c in r['commands'] if c['label']=='component'))
  require([(r['arm'],r['index']) for r in actual]==order,'actual balanced independent order')
  samples={str(a):[r['component_total_ms'] for r in bench if r['arm']==a] for a in (0,1)}
  medians={a:statistics.median(v) for a,v in samples.items()};change=100*(medians['1']/medians['0']-1)
  record=dict(passed=True,complete=True,active=False,GPU_executed=False,references={n:sha(A/n) for n in names},recipe_sha256=sha(Path(__file__)),
   binary_sha256=reference['binary_sha256'],source_pins=reference['source_pins'],full_exact_pairs=2,full_cases_per_process=16,full_dump_sha256=reference['dump']['sha256'],
   benchmark_geometry=dict(nq=256,cap=2051,repeats=20,counts='13,67,then2051',scalar_oracle_queries=[0,1,2,255],all_output_finite=True,tail_guards=True),
   actual_order=order,samples_ms=samples,medians_ms=medians,median_time_change_percent=change,all_benchmark_bytes_equal=True,benchmark_dump_sha256=bench[0]['dump']['sha256'],
   candidate_adopted=False,decision='Reject unpadded cell-major K mapping: every measured ON time exceeds every OFF time in this matched component screen.' if min(samples['1'])>max(samples['0']) else 'No adoption; component evidence insufficient.',
   performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
   scope='Actual component clean independent processes, no model PP or task-quality claim. Timing includes submission/queue/drain. Slower result does not prove SLM conflicts; candidate padding is separate hypothesis.',finished_utc=module.utc())
  with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
  print(json.dumps({k:record[k] for k in ('passed','samples_ms','medians_ms','median_time_change_percent','candidate_adopted','decision')}),flush=True)


if __name__=='__main__':main()
