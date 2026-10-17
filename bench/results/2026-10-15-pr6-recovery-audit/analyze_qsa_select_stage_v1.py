#!/usr/bin/env python3
"""Reduce all qualified per-process QSA stage samples, preserving originals."""
from pathlib import Path
import json, statistics, hashlib, datetime
A = Path(__file__).resolve().parent
rows = []
for ctx in [32768, 78453, 131072, 262144]:
    row = dict(context=ctx, capacity=131072 if ctx <= 131072 else 262144, queries=256, stages={})
    for stage in ['score', 'topk', 'pair']:
        procs = []
        for i in range(3):
            p = A / f'qsa-select-stage-{ctx}-{i}-v1.json'
            v = json.loads(p.read_text())
            assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults']
            vals = [s for s in v['samples'] if s['stage'] == stage]
            med = {k: statistics.median(float(t[k]) for t in vals) for k in [
                'submit_wall_ms', 'submit_thread_cpu_ms', 'wait_wall_ms', 'wait_thread_cpu_ms',
                'total_wall_ms', 'total_thread_cpu_ms']}
            procs.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), medians=med))
        totals = [p['medians']['total_wall_ms'] for p in procs]
        row['stages'][stage] = dict(processes=procs, median_of_process_medians_ms=statistics.median(totals),
                                    process_median_SD_ms=statistics.stdev(totals))
    s = row['stages']
    score, top = [s[k]['median_of_process_medians_ms'] for k in ['score', 'topk']]
    row['score_fraction_of_isolated_stage_sum'] = score / (score + top)
    rows.append(row)
record = dict(schema='strata-qsa-select-stage-analysis-v1',
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), passed=True, rows=rows,
    decision='Prioritize score mapping/matrix work over top-k; scalar FP32 score owns84-87% of synthetic selection stage sum. CPU submit is small, calling-thread CPU during waits mostly mirrors wall. This is evidence of polling, not evidence of CPU math bottleneck.',
    limits=['Synthetic finite seeded queries/keys, not a real activation trace; cannot directly replace the historical15.8s combined model interval.',
            'Separate drained-stage timings and pair timings include submission/waits; not physical GPU busy, instruction issue, bandwidth counters or DRAM/PCIe bytes.',
            '262144 passes here only for this scorer/selector fixture and guards; full-model262144/RESTORE remain unqualified.',
            'Score sampled FP64 oracle uses fixed FP32 roundoff budget for unchanged FP32 arithmetic, not an adoption rule for future BF16 algorithms.',
            'No model throughput, quality or sustained decode measured in this fixture.'],
    next='Independently authored multi-query subgroup scorer, matched full score-bit/ID comparison, then repeated whole-path timing before adoption.')
out = A / 'qsa-select-stage-analysis-v1.json'
assert not out.exists()
out.write_text(json.dumps(record, indent=2) + '\n')
