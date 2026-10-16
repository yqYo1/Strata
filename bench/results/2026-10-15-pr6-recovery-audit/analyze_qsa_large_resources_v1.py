#!/usr/bin/env python3
"""Reconcile nine closed8192-query processes and their full-frame comparisons."""
import hashlib
import json
from pathlib import Path
import statistics

A = Path(__file__).resolve().parent
ORDER = ['cleanA0', 'cleanB0', 'cleanC0', 'cleanC1', 'cleanB1',
         'cleanA1', 'cleanA2', 'cleanC2', 'cleanB2']


def pin(p):
    return dict(path=str(p), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    processes = []; binary = None; environment = None; last_start = None
    for name in ORDER:
        path = A / f'own-qsa-large-resource-{name}-v1.json'
        comparison = A / f'own-qsa-large-resource-compare-{name}-v1.json'
        j = json.loads(path.read_text()); c = json.loads(comparison.read_text())
        for r in [j, c]:
            assert r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
        assert j['boot_unchanged'] and not j['fresh_faults'] and not j['driver_dump_entries']
        assert j['mode'] == 'clean' and len(j['cases']) == 1
        case = j['cases'][0]
        assert not case['traced'] and case['exit_code'] == 0 and len(case['samples']) == 10
        assert c['result']['status'] == 'PASS' and c['result']['frames'][1]['exact_reference']
        assert all(v['finite'] and v['bytes'] == 201326616 for v in c['result']['frames'])
        if binary is None:
            binary = j['binary']; environment = j['native_environment']
        assert j['binary'] == binary and j['native_environment'] == environment | {
            'XDG_CACHE_HOME': j['native_environment']['XDG_CACHE_HOME']}
        native = [v for v in j['commands'] if v['label'] == 'qsa-0']
        assert len(native) == 1
        start = native[0]['direct_identity']['start_ticks']
        assert last_start is None or start > last_start
        last_start = start
        medians = {k: statistics.median(s[k] for s in case['samples'])
                   for k in case['samples'][0] if k != 'sample'}
        processes.append(dict(case=name, receipt=pin(path), comparison=pin(comparison),
            native_identity=native[0]['direct_identity'], samples=case['samples'], medians=medians,
            frame=case['frame'], frame_status='Full comparison completed; candidate retired separately.'))
    arms = {}
    for arm in 'ABC':
        rows = [p for p in processes if p['case'][5] == arm]
        assert len(rows) == 3
        arms[arm] = dict(process_medians_ms=[p['medians']['whole_wall_ms'] for p in rows],
            median_of_process_medians={k: statistics.median(p['medians'][k] for p in rows)
                                       for k in rows[0]['medians']},
            minimum_ms=min(s['whole_wall_ms'] for p in rows for s in p['samples']),
            maximum_ms=max(s['whole_wall_ms'] for p in rows for s in p['samples']))
    base = arms['A']['median_of_process_medians']['whole_wall_ms']
    record = dict(schema='strata-qsa-large-resource-comparison-v1', analysis_source=pin(Path(__file__)),
        GPU_executed_by_analysis=False, passed=True, processes=processes, order=ORDER,
        independent_processes_per_arm=3, timed_samples_per_process=10, warmups_per_process=2,
        binary=binary, native_environment=environment, arms=arms,
        controls=dict(queries=8192, selected_width=2051, query_heads=24, KV_heads=2, dim=256,
            total_workgroups=16384, allocated_device_bytes_all_arms=823485888,
            selected_unique_KV_scale_bytes_A_C=2165856, selected_unique_KV_scale_bytes_B=277229568,
            workgroups_per_command_A_B=16384, workgroups_per_command_C=1024,
            commands_per_operator_A_B=1, commands_per_operator_C=16),
        ratios=dict(B_over_A=arms['B']['median_of_process_medians']['whole_wall_ms']/base,
            C_over_A=arms['C']['median_of_process_medians']['whole_wall_ms']/base),
        interpretation=[
            'At8192queries,128x selected-address working set increases median wall time about20%; '
            'this is sensitivity to the changed address set, not measured physical DRAM traffic or cache counters.',
            'Splitting the same16384groups into16commands costs about5%, unlike the old128query '
            '2.63x case; do not attribute production QSA cost to the small-grid result.',
            'Neither controlled change identifies XMX/SLM/register/cache saturation, nor tests a faster model implementation.',
            'All90 timed samples and nine full initial-frame finite/byte comparisons pass; no new fault or GPU hang.',
            'Other read-only research workers began during the later processes. No CPU isolation or '
            'physical utilization measurement is claimed. Individual samples and order are retained.'],
        performance_adopted=False, model_PP_measured=False,
        scope='Unchanged production QSA kernel on synthetic uniform selected IDs, not complete model prefill.',
        retention='All comparisons closed. Canonical and consumed successful raw diagnostics can now retire '
                  'after this summary and its evidence are committed; keep source/binary for future current tests.')
    out = A / 'hardware-resource-followup-v8.json'
    assert not out.exists()
    out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(receipt=str(out), passed=True, arms=arms, ratios=record['ratios'])))


if __name__ == '__main__':
    main()
