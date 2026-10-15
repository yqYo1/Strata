#!/usr/bin/env python3
"""Cross-check interval arithmetic and the independent provider timing table."""
import importlib.util
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0,str(A))
import analyze_prefill_window_profile_v2 as analyzer
from analyze_device_profile_v1 import parse


def main():
    cases = 0
    for a,b in [([],[]), ([(1,2)],[(1,2)]),
                ([(3,4),(1,2),(2,3)],[(1,4)]),
                ([(0,4),(1,2),(3,5)],[(0,5)]),
                ([(1,1),(2,3)],[(1,1),(2,3)])]:
        assert analyzer.merged(a)==b; cases+=1
    for a,b,want in [([],[],0), ([(1,3)],[(2,4)],1),
                     ([(1,2),(3,5)],[(0,4)],2),
                     ([(1,4)],[(1,4)],3), ([(0,1)],[(1,2)],0)]:
        assert analyzer.overlap(a,b)==want; cases+=1
    for invalid in [[(2,1)],[(0,float('inf'))],[(float('nan'),2)]]:
        try: analyzer.merged(invalid)
        except RuntimeError: pass
        else: raise AssertionError('invalid interval accepted')
        cases+=1
    assert analyzer.role('zeCommandListAppendMemoryCopy(M2D)[42]')=='provider_memory_copy'
    cases+=1
    comparisons=[]
    for mode,receipt_name in [('component','prefill-window-profile-component-v3-compact.json'),
                              ('long','prefill-window-profile-long-v4.json')]:
        summary=A/f'prefill-window-{mode}-interval-summary-v2.json'
        d=json.loads(summary.read_text())
        assert d['passed'] and d['recipe_sha256']==analyzer.sha(Path(analyzer.__file__))
        receipt=A/receipt_name
        r=json.loads(receipt.read_text())
        assert analyzer.sha(receipt)==d['receipt_sha256']
        table=Path(r['native_trace']['path']).with_name('device_timing.txt')
        pinned=next(f for f in r['provider_files'] if f['path']==str(table))
        assert analyzer.sha(table)==pinned['sha256']
        parsed=parse(table.read_text())
        expected={x['kernel']:x for x in parsed['kernels']}
        actual={}
        for x in d['GPU_operations']:
            # ESIMD provider names use SIMD_ANY rather than SIMD<number>.
            # v2 keeps those few geometries separate; the timing table groups
            # them under the base kernel. Sum every original group for the join.
            name=re.sub(r'\[SIMD_ANY .*\]$','',x['name'])
            group=actual.setdefault(name,dict(calls=0,duration_us=0.0))
            group['calls']+=x['calls']; group['duration_us']+=x['duration_us']
        assert set(actual)==set(expected)
        deltas=[]
        for name,x in actual.items():
            y=expected[name]
            assert x['calls']==y['calls']
            delta=x['duration_us']*1000-y['total_ns']
            assert abs(delta)<1.0,(name,delta)
            deltas.append(abs(delta))
        assert abs(d['GPU_service_sum_us']*1000-parsed['total_device_service_ns'])<1.0
        old=A/f'prefill-window-{mode}-interval-summary-v1.json'
        previous=json.loads(old.read_text())
        for field in ['GPU_service_sum_us','GPU_busy_union_us','GPU_gap_within_span_us',
                      'CPU_wait_union_us','CPU_wait_overlap_GPU_busy_us',
                      'CPU_wait_without_visible_GPU_us']:
            assert abs(d[field]-previous[field])<0.001
        assert sum(x['calls'] for x in d['GPU_operations'])==sum(x['calls'] for x in previous['GPU_operations'])
        comparisons.append(dict(mode=mode,summary_sha256=analyzer.sha(summary),
            receipt_sha256=analyzer.sha(receipt), independent_table_sha256=analyzer.sha(table),
            GPU_operation_groups=len(actual), GPU_calls=sum(x['calls'] for x in actual.values()),
            device_table_total_ns=parsed['total_device_service_ns'],
            max_per_operation_total_difference_ns=max(deltas),
            original_v1_totals_conserved=True, all_events_and_raw_SHA_checked_by_analyzer=True))
    result=dict(passed=True,GPU_executed=False,host_arithmetic_cases=cases,
        recipe_sha256=analyzer.sha(Path(__file__)),analyzer_sha256=analyzer.sha(Path(analyzer.__file__)),
        comparisons=comparisons,
        earlier_attempt=dict(passed=False,error='Exact raw-name table join rejected SIMD_ANY geometry suffixes.',
            remedy='Normalize only known ESIMD geometry suffix and sum all original groups for the independent table join; no samples/timing/events dropped.'),
        scope='Host interval contracts, actual full raw event/SHA accounting, and independent unitrace device timing totals. No speed, quality, physical maximum-context or RESTORE qualification.')
    with (A/'prefill-window-interval-summary-host-v2.json').open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(dict(passed=True,host_arithmetic_cases=cases,modes=[x['mode'] for x in comparisons])))


if __name__=='__main__':
    main()
