#!/usr/bin/env python3
"""Summarize closed copy-geometry samples without treating host waits as GPU time."""
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

A = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    ordered = sorted(values)
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                p95_nearest_rank=ordered[math.ceil(.95 * len(values)) - 1], maximum=max(values))


def main():
    result = dict(passed=False, GPU_executed=False, adopted=False, performance_qualified=False,
                  model_performance_qualified=False, inputs={}, comparisons=[],
                  interpretation='Per-case host queue-completion spans. No exclusive GPU service, PCIe wire bandwidth, model throughput, or quality claim. ABBA blocks are descriptive; no statistical significance claim.')
    for cap in [128, 512]:
        path = A / f'kv-copy-groups-graph-benchmark-{cap}-v3.json'
        d = json.loads(path.read_text())
        assert d['passed'] and d['complete'] and not d['active']
        assert not d['fresh_faults'] and not d['driver_dump_exists'] and d['boot_unchanged']
        assert len(d['samples']) == 720
        result['inputs'][path.name] = sha(path)
        cases = defaultdict(list)
        for s in d['samples']:
            assert s['oracle_exact'] and s['timing_mode'] == 'host_clean'
            assert s['submission_mode'] == 'graph_replay' and s['bandwidth_GBps'] is None
            cases[s['fmt'], s['case']].append(s)
        for (fmt, case), rows in sorted(cases.items()):
            rows.sort(key=lambda s: s['sample'])
            assert [s['sample'] for s in rows] == list(range(40))
            blocks = []
            for b in range(10):
                block = rows[b * 4:b * 4 + 4]
                assert [s['configured_group_cap'] for s in block] == [0, cap, cap, 0]
                clocks = {}
                for clock in ['host_whole_queue_ns', 'host_operation_submit_ns']:
                    base = statistics.mean(s[clock] for s in [block[0], block[3]])
                    candidate = statistics.mean(s[clock] for s in [block[1], block[2]])
                    clocks[clock] = dict(base_mean=base, candidate_mean=candidate, ratio=candidate / base)
                blocks.append(dict(block=b, samples=[s['sample'] for s in block], clocks=clocks))
            arms = {}
            for c in [0, cap]:
                arm = [s for s in rows if s['configured_group_cap'] == c]
                arms[str(c)] = {clock: summary([s[clock] for s in arm]) for clock in
                               ['host_whole_queue_ns', 'host_operation_submit_ns']}
            result['comparisons'].append(dict(cap=cap, fmt=fmt, case=case, arms=arms,
                queue_median_ratio=arms[str(cap)]['host_whole_queue_ns']['median'] / arms['0']['host_whole_queue_ns']['median'],
                paired_blocks=blocks, paired_queue_ratio=summary([b['clocks']['host_whole_queue_ns']['ratio'] for b in blocks])))
    result['passed'] = True
    (A / 'kv-copy-groups-summary-v1.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, comparisons=len(result['comparisons']), samples=1440)))


if __name__ == '__main__':
    main()
