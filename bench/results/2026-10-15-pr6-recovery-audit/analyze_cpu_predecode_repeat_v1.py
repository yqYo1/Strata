#!/usr/bin/env python3
import hashlib
import json
import statistics
from pathlib import Path

A = Path(__file__).resolve().parent


def main():
    basepath, repeatpath = A / 'cpu-predecode-probe-v1.json', A / 'cpu-predecode-repeat-v1.json'
    base, repeat = [json.loads(p.read_text()) for p in [basepath, repeatpath]]
    assert all(d['passed'] and d['complete'] and not d['active'] for d in [base, repeat])
    result = dict(passed=False, source_receipts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [basepath, repeatpath]},
        comparisons=[], independent_processes=4, distinct_experts=64, nt=1,
        GPU_executed=False, adopted=False, model_performance_qualified=False,
        scope='Initial64expert process plus3 independent repeats. No actual model hit-rate, cold expansion amortization, multithread bandwidth, or full-engine benefit claim. Powersave governor, endpoints pinned, TSC is not core cycles.')
    sets = [dict(repetition='initial', rows=base['timings']['e64-nt1']['rows'])] + repeat['repetitions']
    for run in sets:
        for ty in [18, 22, 21]:
            for pair, control in [(0, 'ggml'), (1, 'existing_exact')]:
                blocks = []
                for rd in range(12):
                    s = [s for s in run['rows'] if s['kind'] == 'timing_sample' and s['type'] == ty and s['pair'] == pair and s['round'] == rd]
                    assert len(s) == 4 and sorted(s['slot'] for s in s) == [0,1,2,3]
                    block = dict(round=rd)
                    for clock in ['thread_cpu_ns_per_output_row', 'wall_ns_per_output_row']:
                        a = statistics.mean(x[clock] for x in s if x['arm'] == 'expanded_exact')
                        b = statistics.mean(x[clock] for x in s if x['arm'] == control)
                        block[clock] = dict(candidate=a, control=b, ratio=a/b)
                    blocks.append(block)
                cpu = [b['thread_cpu_ns_per_output_row']['ratio'] for b in blocks]
                wall = [b['wall_ns_per_output_row']['ratio'] for b in blocks]
                result['comparisons'].append(dict(repetition=run['repetition'], type=ty, control=control,
                    paired_blocks=blocks, median_CPU_ratio=statistics.median(cpu), median_wall_ratio=statistics.median(wall),
                    minimum_CPU_ratio=min(cpu), maximum_CPU_ratio=max(cpu)))
    result['passed'] = True
    (A / 'cpu-predecode-repeat-summary-v1.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, comparisons=len(result['comparisons']), independent_processes=4)))


if __name__ == '__main__': main()
