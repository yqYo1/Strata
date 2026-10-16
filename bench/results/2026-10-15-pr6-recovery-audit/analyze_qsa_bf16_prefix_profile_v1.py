#!/usr/bin/env python3
"""Attribute the closed BF16-prefix diagnostic without inferring PMU utilization."""
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fields(line):
    return dict(re.findall(r'(\w+)=(\S+)', line))


def parse_table(lines, vector_keys, integer_vectors):
    assert len(lines) == 50
    header = fields(lines[0])
    assert header['complete'] == '1' and header['errors'] == header['first_status'] == '0'
    rows = []
    for line in lines[1:]:
        row = {}
        for k, v in fields(line).items():
            if k in vector_keys:
                row[k] = [(int(x) if k in integer_vectors else float(x)) for x in v.split(',')]
                assert all(math.isfinite(x) and x >= 0 for x in row[k])
            else:
                row[k] = int(v)
                assert k == 'layer' or row[k] >= 0
        rows.append(row)
    assert [x['layer'] for x in rows] == list(range(48)) + [-1]
    total = rows[-1]
    for k, v in total.items():
        if k == 'layer' or k == 'cap':
            continue
        if isinstance(v, list):
            assert all(len(row[k]) == len(v) for row in rows)
            for i, x in enumerate(v):
                summed = sum(row[k][i] for row in rows[:-1])
                assert summed == x if k in integer_vectors else math.isclose(summed, x, rel_tol=2e-5, abs_tol=.1)
        else:
            assert sum(row[k] for row in rows[:-1]) == v
    return dict(header=header, rows=rows)


def main():
    dst = A / 'qsa-bf16-prefix-profile-summary-v1.json'
    assert not dst.exists()
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        src = A / 'own-qsa-bf16-prefix-profile-bf16-prefix-r0-v1.json'
        r = json.loads(src.read_text())
        assert r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
        assert r['fresh_faults'] == [] and r['boot_unchanged'] and not r['driver_dump_exists']
        owner_path = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
        assert sha(owner_path) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
        spec = importlib.util.spec_from_file_location('profile_summary_owner', owner_path)
        owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner)
        for c in r['commands']:
            assert owner.closed(c) and c['exit_code'] == 0
            for old in c['owners'] + [c['direct_identity']]:
                now = owner.identity(old['pid'])
                assert now is None or now['start_ticks'] != old['start_ticks']
        vectors = {'wall_ms', 'queue_ms', 'samples', 'input_bytes_host_device',
                   'store_bytes_host_device', 'input_queue_host_device_ms', 'store_queue_host_device_ms'}
        ints = {'samples', 'input_bytes_host_device', 'store_bytes_host_device'}
        coarse = parse_table(r['coarse_profile_lines'], vectors, ints)
        fine = parse_table(r['qsa_profile_lines'], {'queue_ms', 'host_wall_ms', 'samples'}, {'samples'})
        assert coarse['header']['v'] == '2' and fine['header']['v'] == '1'
        assert fine['header']['wrapper_ms'] == coarse['header']['wrapper_ms']
        request, = r['benchmark_result']['results']['prefill_only']['requests']
        assert request['actual_prompt_tokens'] == 39262 and request['actual_read_tokens'] == 39262
        assert request['actual_reused_tokens'] == 0 and request['eligible'] and request['ok']
        n, chunks = 39261, 5
        for row in coarse['rows'][:-1]:
            l = row['layer']
            assert row['samples'] == [0 if l == 0 else chunks, chunks if l % 4 == 3 else 0,
                                      0 if l % 4 == 3 else chunks, chunks, 1, 0 if l == 47 else chunks]
        total = coarse['rows'][-1]
        assert total['samples'] == [235, 60, 180, 240, 48, 235]
        expected_residual = [47 * (n - 32768) * 10240 * 2, 47 * 32768 * 10240 * 2]
        assert total['input_bytes_host_device'] == total['store_bytes_host_device'] == expected_residual
        for row in fine['rows']:
            l = row['layer']
            count = chunks * 12 if l == -1 else chunks if l % 4 == 3 else 0
            queries = n * 12 if l == -1 else n if count else 0
            assert row['samples'] == [count] * 7
            assert row['expected_chunks'] == row['dispatch_chunks'] == count
            assert row['expected_queries'] == row['query_total'] == row['matrix_queries'] + row['fallback_queries'] == queries
            assert row['matrix_chunks'] + row['fallback_chunks'] == count
            assert row['staged_chunks'] == 0 and row['cap'] == (2051 if count else 0)
        cpu = {}
        for kind, key in [('wait', 'cpu_wait_lines'), ('host', 'cpu_host_lines')]:
            line, = r[key]
            cpu[kind] = {k: float(v) if '.' in v else int(v) for k, v in re.findall(r'([a-z_][a-z_0-9]*)=([0-9]+(?:\.[0-9]+)?)', line)}
        wait = cpu['wait']
        assert wait['available'] == wait['complete'] == 1 and wait['open_waits'] == wait['wait_throws'] == 0
        duration = float(coarse['header']['wrapper_ms'])
        assert math.isfinite(duration) and duration > 0
        queue = dict(zip(coarse['header']['queue_order'].split(','), total['queue_ms']))
        wall = dict(zip(coarse['header']['wall_order'].split(','), total['wall_ms']))
        qsa = dict(zip(fine['header']['sections'].split(','), fine['rows'][-1]['queue_ms']))
        host = dict(zip(fine['header']['sections'].split(','), fine['rows'][-1]['host_wall_ms']))
        raw = B / 'main-qsa-bf16-prefix-profile-bf16-prefix-r0-v1/owned/engine.stderr'
        assert sha(raw) == r['native_log_sha256']
        host_lines = [line for line in raw.read_text().splitlines() if line.startswith('strata native dense host profile:')]
        result = dict(schema='strata-bf16-prefix-profile-summary-v1', passed=True, complete=True,
                      GPU_executed=False, source_receipt=dict(path=str(src), sha256=sha(src)),
                      analysis_recipe_sha256=sha(__file__), binary_sha256=r['binary_sha256'],
                      source_commit=r['source_commit'], input_tokens=39262, batched_tokens=n,
                      request_sha256=request['request_sha256'], wrapper_ms=duration,
                      diagnostic_prompt_ms=request['prompt_ms'], diagnostic_prompt_tps=request['prompt_tps'],
                      coarse=coarse, fine=fine, queue_interval_ms=queue, host_wall_ms=wall,
                      qsa_interval_ms=qsa, qsa_host_wall_ms=host, cpu=cpu,
                      issuing_thread_cpu_fraction_in_explicit_wait=wait['wait_cpu_s']/wait['main_cpu_s'],
                      host_lifecycle_lines=host_lines, normal_ownership_closed=True, fresh_faults=[],
                      physical_bus_or_memory_counters_collected=False,
                      GPU_execution_unit_utilization_collected=False, performance_adoption=False,
                      limits='Instrumented n1; nonexclusive queue intervals include host gaps. CPU stamps apply to issuing thread only. Copy bytes are requested payload, not wire/DRAM traffic. No counter-based PCIe/VRAM/XMX saturation result, clean speed or full-context/RESTORE qualification.')
        dst.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k: result[k] for k in ['wrapper_ms', 'queue_interval_ms', 'host_wall_ms',
                         'qsa_interval_ms', 'cpu', 'issuing_thread_cpu_fraction_in_explicit_wait']}))


if __name__ == '__main__':
    main()
