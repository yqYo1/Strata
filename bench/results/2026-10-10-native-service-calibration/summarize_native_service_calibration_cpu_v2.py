"""Root offline round bootstrap; explicit microservice, never engine adoption."""
from pathlib import Path
import csv
import datetime
import fcntl
import hashlib
import json
import math
import random
import statistics

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
EXPECTED={
 (6,3):'dffb75710b481c0c21905612ca4b9028333e6593926b782e318099fe8bfa073e',
 (0,3):'8053a2aa4df45216fd168508e3540b2428f019a81725efd2cf17e66ed19eadcb',
 (0,4):'95d1419ef44a252ad49c409d5d06a4235de8ffc8b05fb8c2fb543bddb9bad43e',
 (6,4):'fb33b2294adf63156792df9cd7c6f0c548e3d2aef168136f56922f073f209c7e',
}


def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def ci(values,key):
 assert len(values)==25 and all(math.isfinite(v) and v>0 for v in values)
 rng=random.Random(int(hashlib.sha256(key.encode()).hexdigest(),16))
 reps=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(10000))
 lo,hi=reps[249],reps[9749]
 med=statistics.median(values)
 return dict(samples=len(values),median=med,bootstrap95=[lo,hi],relative_interval_width=(hi-lo)/med,
             coefficient_of_variation=statistics.stdev(values)/statistics.mean(values),
             minimum=min(values),maximum=max(values))


def main():
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  out=B/'native-service-calibration-22-20-nt1-summary-v2';out.mkdir(mode=0o700)
  result=dict(scope='Exact22/20NT1, layer-balanced real routed weights, synthetic inputs, homogeneous six-job CPU pool; not inference latency/adoption',
              controller_sha256=sha(__file__),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              affinity_invalidation_sha256=sha(B/'native-service-calibration-affinity-invalidation-v1.json'),
              old_r1_r2_timing_interpretation_valid=False,
              bootstrap='10000 resampled 25-round medians per fresh process/cohort/arm; deterministic SHA256 seed; central 95 percentiles indices249/9749; no per-ID pseudoreplication',
              profile_order=[{'tasks':6,'repeat':3},{'tasks':0,'repeat':3},{'tasks':0,'repeat':4},{'tasks':6,'repeat':4}],
              process_rounds_not_pooled_as_independent=True,inference_run=False,performance_eligible=False,adopted=False,
              full_lifecycle_passed=False,profiles=[],comparisons=[])
  for task,rep in ((6,3),(0,3),(0,4),(6,4)):
   root=B/f'native-service-calibration-22-20-nt1-timing-tasks{task}-batch6-r{rep}'
   rp=root/'record.json';d=json.loads(rp.read_text())
   assert d['passed'] and d['complete'] and not d['active'] and not d['cleanup'] and not d['survivors']
   if (task,rep) in EXPECTED:assert sha(rp)==EXPECTED[(task,rep)]
   assert d['configuration']==dict(tasks=task,batch=6,nt=1,gu=22,down=20,round_seed=2026101001)
   command=d['commands'][-1];assert command['exit_code']==0 and 'LD_DEBUG' not in command['environment']
   assert command['argv'][0]==d['binary']['path'] and command['label']=='uninstrumented-rounds'
   assert not any(x in command['argv'] for x in ('strace','perf','--correctness-only'))
   samples=root/'uninstrumented-rounds/stdout.csv'
   assert command['files'][str(samples)]['sha256']==sha(samples)
   all_rows=list(csv.reader(samples.read_text().splitlines()))
   placement=[r for r in all_rows if r[:2]==['PLACEMENT','observed_allowed_list']]
   planned=[int(r[3]) for r in all_rows if r[:2] in (['PLACEMENT','planned_host'],['PLACEMENT','planned_worker'])]
   assert len(placement)==6 and len({r[3] for r in placement})==6
   observed=[int(r[4].strip()) for r in placement]
   assert len(set(observed))==6 and sorted(observed)==sorted(planned)==list(range(6))
   assert next(r for r in all_rows if r[:1]==['TASK_PLAN'])==['TASK_PLAN',str(task),str(task or 18)]
   rows=[r for r in all_rows if r[:1]==['ROUND'] and r[1]!='cohort']
   assert len(rows)==250
   stats={}
   for co in (0,1):
    for arm in ('GU','FFquant','Down','direct_complete','pool'):
     rr=sorted([r for r in rows if int(r[1])==co and r[2]==arm],key=lambda r:int(r[3]))
     assert [int(r[3]) for r in rr]==list(range(25))
     # Direct wrappers have one call/expert; pool has32 calls of6 jobs.
     # Pool units remain whole192-job cohort wall, never per-job C_direct.
     divisor=1 if arm=='pool' else 192
     stats[f'{co}/{arm}']={**ci([float(r[12])/divisor for r in rr],f'{task}/{rep}/{co}/{arm}'),
                          'units':'ms_per_192_job_cohort' if arm=='pool' else 'ms_per_expert_call'}
     if arm=='pool':
      for phase,col in (('pool_GU',13),('pool_Q',14),('pool_Down',15)):
       stats[f'{co}/{phase}']={**ci([float(r[col]) for r in rr],f'{task}/{rep}/{co}/{phase}'),'units':'ms_per_192_job_cohort'}
   gates={}
   for arm in ('GU','FFquant','Down','direct_complete','pool','pool_GU','pool_Q','pool_Down'):
    train,hold=stats[f'0/{arm}'],stats[f'1/{arm}']
    overlap=max(train['bootstrap95'][0],hold['bootstrap95'][0])<=min(train['bootstrap95'][1],hold['bootstrap95'][1])
    narrow=train['relative_interval_width']<=.05 and hold['relative_interval_width']<=.05
    gates[arm]=dict(train_holdout_intervals_overlap=overlap,both_relative_width_at_most_5_percent=narrow,
                    train_holdout_median_relative_gap=hold['median']/train['median']-1,qualified=overlap and narrow)
   result['profiles'].append(dict(tasks=task,effective_tasks=task or 18,repeat=rep,receipt_path=str(rp),receipt_sha256=sha(rp),
                                  samples_sha256=sha(samples),actual_singleton_cpus=observed,planned_cpus=planned,placement_gate_passed=True,stats=stats,R46_gates=gates,
                                  observation_before=command['start_observation'],observation_after=command['end_observation']))
  for rep in (3,4):
   six=next(p for p in result['profiles'] if p['tasks']==6 and p['repeat']==rep)
   eighteen=next(p for p in result['profiles'] if p['tasks']==0 and p['repeat']==rep)
   comparisons={key:six['stats'][key]['median']/eighteen['stats'][key]['median'] for key in six['stats']}
   result['comparisons'].append(dict(repeat=rep,six_over_eighteen_median_ratio=comparisons,
                                    scope='Controlled homogeneous 6-job pool; cross-layer shuffled batches differ from production per-layer variable/mixed-NT callbacks. Direct components are controls, not task-factor treatment.'))
  result['all_direct_component_fit_gates_passed']=all(p['R46_gates'][a]['qualified'] for p in result['profiles'] for a in ('GU','FFquant','Down','direct_complete'))
  result['interpretation']='If any fit/holdout gates fail, those cells are unresolved and cannot weight route-demand or establish a task candidate. Whole-engine >=32K and full262144 gates remain mandatory. No pool-ms sum or cohort-ms division is an inference/per-job latency estimate.'
  rp=out/'summary.json';rp.write_text(json.dumps(result,indent=2)+'\n')
  lines=['# Actual native CPU service calibration,22/20/NT1','',
         'Four new fresh processes r3/r4, counterbalanced6/18/18/6 task profiles with six verified distinct singleton CPUs0..5;25 measured rounds per arm/cohort in each process. All recorded per-round samples remain. This is a layer-balanced host-resident weight characterization with synthetic activations, not model throughput.',
         '', '| profile | cohort | GU ms/call | quant ms/call | Down ms/call | complete ms/call | pool cohort ms | pool fit gate |',
         '|---|---|---:|---:|---:|---:|---:|---|']
  for p in result['profiles']:
   for co,name in ((0,'train'),(1,'holdout')):
    v=[p['stats'][f'{co}/{a}']['median'] for a in ('GU','FFquant','Down','direct_complete','pool')]
    lines.append(f"| tasks{p['effective_tasks']} r{p['repeat']} | {name} | "+' | '.join(f'{x:.6f}' for x in v)+f" | {p['R46_gates']['pool']['qualified']} |")
  lines+=['','95% bootstrap intervals, relative widths, CV, every fit/holdout gate and per-process ratios are in summary.json. Failing gates remain failed; additional repeats do not turn distinct prompts or experts into independent workloads. Exact native/independent NT1 GU+Down20, quantizer, repeat and partition checks passed in each process.','',result['interpretation'],'']
  (out/'REPORT.md').write_text('\n'.join(lines))
  print(json.dumps(dict(summary=str(rp),sha256=sha(rp),all_direct_fit_gates=result['all_direct_component_fit_gates_passed'],comparisons=result['comparisons'])))


if __name__=='__main__':main()
