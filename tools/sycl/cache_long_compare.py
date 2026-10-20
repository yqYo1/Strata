#!/usr/bin/env python3
"""Retained-engine117K comparison decisions; execution follows long admission.

The two arms differ in engine revision and scratch/cache placement. This is a
route comparison, not an isolated kernel experiment or sustained decode test.
"""
import math
import statistics

from measurement_guards import require

ORDER = (0, 1, 1, 0, 0, 1)
PAIRS = ((0, 1), (3, 2), (4, 5))


def comparison_plan():
    return dict(order=list(ORDER), pairs=[list(p) for p in PAIRS],
        arms=['retained396-owned128', 'current-borrowed1256'],
        requests_per_process=4, actual_fresh_input_tokens=117489, output_cap=24,
        thinking=True, workload_id='native_gdn_20261018',
        classification='Route-level prefill comparison on the current runtime; engine and placement differ.',
        prerequisites='Close1256-slot117K all-GPU/two-bank admission, then a separately checked old-owned117K request. Verify both retained source/recipe/binary/fixture sets before every process. Freeze full benchmark/config/runtime/model inputs before collection.',
        decision='Compare first-use separately. For each pair use the ratio of subsequent-request median throughput (three fresh requests/process). Faster screen requires all three ratios>1 and geomean>1.02; slower screen requires all<1 and geomean<1/1.02. Otherwise inconclusive. Nonregression requires geomean>=0.98 and every pair>=0.95. The1000 milestone requires each of three candidate process medians>=1000tok/s.',
        stopping='Stop on first process, input, route, fault, ownership, budget or sample-eligibility failure. Never retry a root or replace failed samples. No speed-based early stopping.',
        limits='Two old-first pairs and one new-first pair represent both chronology directions without exact balance. Four requests share a process; the three process pairs are the repetition units.24-output checks do not qualify decode, agent-session, quality or physical262144/session/RESTORE. Neither screen adopts a model route.')


def summarize(processes):
    require(len(processes) == 6 and [r['cohort'] for r in processes] == list(range(6)) and
        [r['arm'] for r in processes] == list(ORDER), 'complete frozen six-process order required')
    require(len({(r['native_identity']['pid'],r['native_identity']['start_ticks']) for r in processes}) == 6,
        'independent native processes required')
    for key in ('benchmark_fingerprint', 'machine_identity', 'model_assets_fingerprint'):
        require(all(r[key] == processes[0][key] for r in processes), 'cross-process input changed: '+key)
    for arm in (0,1):
        group=[r for r in processes if r['arm'] == arm]
        require(all(r['binary_sha256'] == group[0]['binary_sha256'] and
            r['build_source_commit'] == group[0]['build_source_commit'] for r in group), 'arm source/binary changed')
    medians=[]
    first=[]
    for process in processes:
        require(process['passed'] and process['ownership_closed'] and not process['fresh_faults'] and
            process['observer_records_absent'] and process['actual_route_checked'] and len(process['requests']) == 4,
            'failed, observed, unqualified or partial process')
        speeds=[]
        for run,sample in enumerate(process['requests']):
            require(sample['run'] == run and all(sample.get(k) is True for k in
                ('ok','eligible','stream_complete','thinking_requested','reasoning_observed')) and
                sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == 117489 and
                sample['actual_reused_tokens'] == 0 and 1 <= sample['generated_tokens'] <= 24 and
                sample['prompt_tps_source'] == 'engine', 'ineligible or changed request')
            require(type(sample['prompt_ms']) in (int,float) and math.isfinite(sample['prompt_ms']) and
                sample['prompt_ms'] > 0, 'invalid prompt clock')
            speeds.append(117489000/sample['prompt_ms'])
        first.append(speeds[0]); medians.append(statistics.median(speeds[1:]))
    for baseline,candidate in PAIRS:
        require(all(processes[baseline]['requests'][i]['request_sha256'] ==
            processes[candidate]['requests'][i]['request_sha256'] for i in range(4)), 'paired input body changed')
    ratios=[medians[b]/medians[a] for a,b in PAIRS]
    geo=math.exp(statistics.mean(math.log(r) for r in ratios))
    decision=('candidate_faster_screen' if all(r>1 for r in ratios) and geo>1.02 else
        'candidate_slower_screen' if all(r<1 for r in ratios) and geo<1/1.02 else 'inconclusive_variability_or_small_effect')
    return dict(prefill_decision=decision,subsequent_pair_ratios=ratios,subsequent_ratio_geomean=geo,
        first_use_tps=first,subsequent_median_tps=medians,
        prefill_nonregression=geo>=0.98 and all(r>=0.95 for r in ratios),
        candidate1000tok_s_all_process_medians=all(medians[i]>=1000 for i,r in enumerate(processes) if r['arm']==1),
        adoption_qualified=False,sustained_decode_qualified=False,limits=comparison_plan()['limits'])
