#!/usr/bin/env python3
"""Summarize every closed K schedule sample; no model-speed extrapolation."""
import collections,hashlib,json,math,statistics
from pathlib import Path
A=Path(__file__).resolve().parent
def pin(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
 out=A/'qsa-kstaging-component-comparison-v1.json';assert not out.exists()
 names=[f'own-qsa-kstaging-{arm}-{mode}-v1.json' for arm in ['baseline','span4','row'] for mode in ['smoke','component']]
 names += ['own-qsa-kstaging-all-tiny-v1.json','own-qsa-kstaging-all-perf-v1.json']
 receipts={};records={}
 for name in names:
  p=A/name;d=json.loads(p.read_text());assert d['passed'] and d['complete'] and d['ownership_closed'] and not d['active'] and not d['fresh_faults'] and not d['driver_dump_entries'] and d['boot_unchanged']
  assert all(c['exit_code']==0 and c['direct_child_reaped'] and c['session_empty'] and c['observation_complete'] for c in d['commands'])
  receipts[name]=pin(p);records[name]=d
 d=records['own-qsa-kstaging-all-perf-v1.json'];groups=collections.defaultdict(list)
 for line in d['individual_samples']:
  row=dict(x.split('=',1) for x in line.split()[1:])
  for k in ['cells','queries','cap','round','order','span','compared']:row[k]=int(row[k])
  row['milliseconds']=float(row['milliseconds']);assert math.isfinite(row['milliseconds']) and row['milliseconds']>0
  assert row['compared']==row['queries']*24*256
  assert row['exact_bits']==row['guards']==row['status']=='PASS'
  groups[(row['cells'],row['queries'],row['cap'],row['pattern'])].append(row)
 assert len(groups)==3;comparisons=[]
 for key,samples in groups.items():
  assert len(samples)==18 and len({(s['round'],s['order']) for s in samples})==18
  arms={}
  for span in [1,4,256]:
   selected=[s for s in samples if s['span']==span];assert len(selected)==6 and {s['round'] for s in selected}==set(range(6))
   ms=[s['milliseconds'] for s in selected];arms[str(span)]={'individual_samples':selected,'median_ms':statistics.median(ms),'mean_ms':statistics.mean(ms),'stdev_ms':statistics.stdev(ms)}
  baseline=arms['1']['median_ms'];relative={str(span):{'median_time_reduction_percent':100*(1-arms[str(span)]['median_ms']/baseline),'median_speed_ratio':baseline/arms[str(span)]['median_ms']} for span in [4,256]}
  comparison={'shape':dict(zip(['cells','queries','cap','pattern'],key)),'arms':arms,'candidate_relative_to_baseline':relative};comparisons.append(comparison)
  print(json.dumps({'shape':comparison['shape'],'median_ms':{k:v['median_ms'] for k,v in arms.items()},'relative':relative}))
 build=A/'own-qsa-kstaging-build-v3.json';b=json.loads(build.read_text());assert b['passed'] and b['complete'] and b['ownership_closed']
 result=dict(schema=1,passed=True,source=pin(Path(__file__)),receipts=receipts,build=pin(build),binary=b['binary'],perf_binary=b['perf_binary'],proposal=pin(A/'own-qsa-kstaging-v1/manifest.json'),comparisons=comparisons,numerical_contract='Only K producer ownership changes. Matrix layout, operand bits and arithmetic/reduction/barrier order remain fixed. Component oracle is unchanged. Full output bit equality is a discriminating layout-change check, not a general cross-LLM precision/token equality adoption gate.',measurement='Same process per-shape baseline/span4/span256 with rotating order;6timed samples perarm after explicit warmups. Wall time includes host admission, logical prefix refresh, attention, checked drain; excludes uploads/readbacks/checks. Validation layer remains on, full API trace off in timed perf process. This is not isolated attention device time, physical DRAM/cache/SLM traffic, occupancy or saturation.',source_integration=False,adoption=False,whole_model_speed_qualified=False,physical262144_model_RESTORE_qualified=False,original_failures_preserved=['own-qsa-kstaging-build-v1.json:post-build utility artifact monitor cap; no GPU executed','own-qsa-kstaging-build-v2.json:SYCL unbundled temporary host-object path differs from fat object in map; no GPU executed'],next='Use the closed component evidence to decide a real long-prefill engine experiment. Keep old packed candidate and C16/C64 closed. Preserve existing full-context numerical failure and observed short-decode cost; no default promotion.',ownership_closed=True,fresh_faults=[])
 out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
