#!/usr/bin/env python3
"""Record matched per-process medians and unchanged full-output fingerprints."""
import datetime,hashlib,json,statistics
from pathlib import Path
A=Path(__file__).resolve().parent
out=A/'qsa-score-querytile-decision-v1.json';assert not out.exists()
cohort=json.loads((A/'qsa-score-querytile-cohort-v1.json').read_text());assert cohort['passed'] and cohort['complete'] and cohort['active_sample'] is None
rows=[]
for ctx in [32768,78453,262144]:
 row=dict(context=ctx,arms={},exact_same_scores_and_ids=True)
 reference=None
 for tile in [0,8]:
  processes=[]
  for rep in range(3):
   p=A/f'qsa-score-querytile-{tile}-{ctx}-{rep}-v1.json';v=json.loads(p.read_text())
   assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['active'] and v['native_exit_code']==0
   assert not v['fresh_faults'] and not v['driver_dump_entries'] and v['boot_unchanged']
   assert all(x in v['cases'][0] for x in ['baseline_bits=PASS','guards=PASS','inputs=PASS','repeat_bits=PASS'])
   identity=[s for s in v['progress'] if s.startswith('inputs ') or s.startswith('case ')]
   if reference is None:reference=identity
   assert identity==reference,(ctx,tile,rep,'input/output fingerprints')
   med={stage:{key:statistics.median(float(s[key]) for s in v['samples'] if s['stage']==stage) for key in
       ['total_wall_ms','submit_wall_ms','submit_thread_cpu_ms','wait_wall_ms','wait_thread_cpu_ms','total_thread_cpu_ms']}
       for stage in ['score','topk','pair']}
   processes.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),medians=med))
  row['arms'][str(tile)]=dict(processes=processes,stages={stage:dict(median_ms=statistics.median(p['medians'][stage]['total_wall_ms'] for p in processes),SD_of_process_medians_ms=statistics.stdev(p['medians'][stage]['total_wall_ms'] for p in processes)) for stage in ['score','topk','pair']})
 row['timing_reduction_percent']={stage:100*(1-row['arms']['8']['stages'][stage]['median_ms']/row['arms']['0']['stages'][stage]['median_ms']) for stage in ['score','topk','pair']}
 rows.append(row)
rec=dict(schema='strata-qsa-score-querytile-decision-v1',passed=True,complete=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,
 selection='Query tile8 for next engine integration/evaluation. Private component candidate only; no product math/default/environment selector has been changed.',
 production_selected=False,model_PP_measured=False,quality_qualified=False,physical262144_model_qualified=False,RESTORE_qualified=False,
 math='Unchanged FP32 input/storage, four-term per-lane products/sums, five XOR subgroup reductions and head summation; per-query dead key/tail bonus. Whole score and ID bit parity checked versus original GPU kernel in each process.',
 resources='WG32/SG32, eight sequential queries per subgroup, four common key FP32 elements per lane, no SLM or XMX, about8x fewer workgroups at a multiple-of8 query count. Output bytes and useful dot products unchanged. Logical pooled loads can be amortized across8queries; these are source-level counts, not measured physical DRAM traffic, cache hit rates, GRF allocation or occupancy.',
 limit='Finite seeded synthetic keys/queries; all256rows are checked against baseline, sampled335scores against a fixed FP32 roundoff bound and allselected IDs against independent cell ranking. Full engine/real activations, fullmodel262144/RESTORE, quality and sustaineddecode are pending.',
 next='Integrate a dedicated-prefill-only opt-in scorer entry without changing decode/reference/short-prefill, rebuild and qualify the actual entry, then compare official bench.sh long and required agent/long-thinking generation paths. Keep the1000PP/70TG goal active.')
out.write_text(json.dumps(rec,indent=2)+'\n')
for row in rows:print(row['context'],row['timing_reduction_percent'])
