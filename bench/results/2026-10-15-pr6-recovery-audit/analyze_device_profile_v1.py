#!/usr/bin/env python3
"""Parse pinned unitrace text tables; retain aggregate service, not busy-union claims."""
import csv
import hashlib
import json
import re
from pathlib import Path

A = Path(__file__).resolve().parent


def parse(text):
    rows, properties = [], []
    for fields in csv.reader(text.splitlines(), skipinitialspace=True):
        fields = [f.strip() for f in fields]
        if len(fields) == 7 and fields[1].isdigit() and fields[2].isdigit():
            rows.append(dict(kernel=fields[0], calls=int(fields[1]), total_ns=int(fields[2]),
                reported_percent=float(fields[3]), average_ns=int(fields[4]), minimum_ns=int(fields[5]), maximum_ns=int(fields[6])))
        elif len(fields) == 8 and fields[1] in ['JIT', 'AOT']:
            # Library kernels can report SIMD=ANY; retain that provider value.
            properties.append(dict(kernel=fields[0], compiled=fields[1], SIMD=int(fields[2]) if fields[2].isdigit() else fields[2],
                arguments=int(fields[3]), SLM_bytes_per_group=int(fields[4]), private_bytes_per_thread=int(fields[5]),
                spill_bytes_per_thread=int(fields[6]), GRF_bytes_per_thread=int(fields[7])))
    device = re.search(r'Total Device Time for L0 backend \(ns\):\s+(\d+)', text)
    assert rows and device
    assert sum(r['total_ns'] for r in rows) == int(device[1])
    return dict(total_device_service_ns=int(device[1]), kernels=rows, properties=properties,
        scope='Sum of recorded device operation durations, including copies. Concurrent durations may overlap; not device busy union or whole-process critical path.')


def main():
    p = A / 'kv-resolver-profile-v1.json'
    d = json.loads(p.read_text())
    assert d['passed'] and d['complete'] and not d['active'] and d['collected']
    assert not d['fresh_faults'] and d['boot_unchanged'] and not d['driver_dump_exists']
    tables = [f for f in d['provider_files'] if Path(f['path']).name == 'device_timing.txt']
    assert len(tables) == 1
    result = parse(tables[0]['text'])
    resolver = [k for k in result['kernels'] if 'kv_stream_resolve_with_copy_groups' in k['kernel']]
    assert len(resolver) == 3
    sequential = next(k for k in resolver if 'lambda()#1' in k['kernel'])
    legacy = next(k for k in resolver if 'nd_item<1>)#1' in k['kernel'])
    bounded = next(k for k in resolver if 'nd_item<1>)#2' in k['kernel'])
    assert sequential['calls'] == 3 * 6 * 42 == 756
    assert legacy['calls'] == bounded['calls'] == 3 * 6 * 21 == 378
    result.update(passed=True, source_receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        graph_replay_counts_qualified=True, model_profile_executed=False,
        roles=dict(ordered_resolver=sequential, legacy_copy=legacy, bounded_copy=bounded),
        comparison_limit='Mixed formats/cases, including 8192 misses, warmups, reset and oracle transfers. Not all-hit/int8-only durations. Do not compare copy variants as a clean performance trial.')
    (A / 'kv-resolver-profile-summary-v1.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, recorded_operations=sum(k['calls'] for k in result['kernels']),
        graph_resolver_calls=sequential['calls'], device_service_seconds=result['total_device_service_ns']/1e9)))


if __name__ == '__main__':
    main()
