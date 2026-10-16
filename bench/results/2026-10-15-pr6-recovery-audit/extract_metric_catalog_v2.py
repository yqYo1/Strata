#!/usr/bin/env python3
"""Main-owned offline extraction of a closed full metric enumeration.
Repository LICENSE; no runtime, device or provider operations.
"""
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path

A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def pin(p):
    with p.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=digest)


def main():
    root = B / 'main-metric-enum-query-v2'
    source = root / 'record.json'
    target = A / 'own-metric-full-catalog-evidence-v2.json'
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not target.exists()
        assert pin(source)['sha256'] == 'bdcbb85dadd3f54240330ef8fdf7d741fcd9cd6fb37b464630d4388287952707'
        receipt = json.loads(source.read_text())
        assert receipt['passed'] and receipt['complete'] and receipt['ownership_closed'] and not receipt['active']
        assert not receipt['fault_window']['fresh_faults']
        for command in receipt['commands']:
            assert command['exit_code'] == 0 and not command['survivors']
            for old in command['owners'] + [command['direct_identity']]:
                path = Path('/proc', str(old['pid']), 'stat')
                if path.exists():
                    assert int(path.read_text().rsplit(')', 1)[1].split()[19]) != old['start_ticks']
        command = next(c for c in receipt['commands'] if c['label'] == 'native-enumeration')
        stdout = root / 'owned/native-enumeration.stdout'
        identity = pin(stdout)
        assert {k: identity[k] for k in ('bytes', 'sha256')} == command['logs'][stdout.name]
        raw = stdout.read_text()
        assert raw.endswith('\n')
        rows = [json.loads(line) for line in raw.splitlines()]
        pending = None
        counts = Counter()
        for row in rows:
            if row['kind'] == 'entry':
                assert pending is None
                pending = row['api']
            elif row['kind'] == 'result':
                assert pending == row['api'] and row['code'] == 0 and row['name'] == 'ZE_RESULT_SUCCESS'
                counts[pending] += 1
                pending = None
        assert pending is None
        groups = [row for row in rows if row['kind'] == 'group']
        metrics = [row for row in rows if row['kind'] == 'metric']
        summary, = [row for row in rows if row['kind'] == 'summary']
        assert summary == receipt['enumeration_summary']
        assert len(groups) == summary['groups'] == 24
        assert len(metrics) == sum(g['metric_count'] for g in groups) == summary['sampled_metrics'] == 1414
        for group in groups:
            values = [m for m in metrics if all(m[k] == group[k] for k in ('driver', 'root', 'group'))]
            assert len(values) == group['metric_count']
            assert [m['metric'] for m in values] == list(range(len(values)))
        chosen, = [g for g in groups if g['target'] and g['name'] == 'ComputeBasic' and g['sampling_flags'] == 1]
        catalog = [m for m in metrics if all(m[k] == chosen[k] for k in ('driver', 'root', 'group'))]
        assert len(catalog) == 83
        result = dict(schema='strata-metric-full-catalog-evidence-v2', status='ENUMERATION_PASS',
            receipt=pin(source), stdout=identity, build_receipt=pin(B / 'main-metric-enum-build-v2/record.json'),
            binary=json.loads((B / 'main-metric-enum-build-v2/record.json').read_text())['binary_candidate'],
            summary=summary, API_success_counts=dict(counts), target_compute_group=chosen,
            target_compute_metrics=catalog,
            identities=[r for r in rows if r['kind'] in ('driver', 'root', 'pci', 'pci_interface')],
            groups=groups, metrics=metrics,
            collection_qualified=False,
            limits='All metric properties were enumerated; no context, queue, command, activation, query or streamer created. Descriptions/units are driver metadata, not measured utilization or a bottleneck verdict.',
            reviewer='Main', retention=dict(owner='Main', consumer='Bounded tiny metric-query schema and resource-counter interpretation',
                next_review_utc='2026-10-17T03:00:00Z', raw_stdout_budget_bytes=1 << 20))
        target.write_text(json.dumps(result, indent=2) + '\n')
        copy = A / 'own-metric-enumeration-query-v2.json'
        assert not copy.exists()
        copy.write_bytes(source.read_bytes())
        print(json.dumps(dict(evidence=pin(target), groups=len(groups), metric_properties=len(metrics),
            target_group=chosen['name'], target_columns=len(catalog), collection_qualified=False)))


if __name__ == '__main__':
    main()
