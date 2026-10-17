#!/usr/bin/env python3
"""Summarize the closed, current-path diagnostic; never run a workload."""
import hashlib
import json
import math
import re
from pathlib import Path

A = Path(__file__).resolve().parent


def artifact(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def fields(line):
    return dict(re.findall(r'(\w+)=([^ ]+)', line))


def numbers(line, name, labels):
    values = list(map(float, fields(line)[name].split(',')))
    assert len(values) == len(labels) and all(math.isfinite(x) and x >= 0 for x in values)
    return dict(zip(labels, values))


def compact(row):
    return {key: value for key, value in row.items() if key != 'raw_row'}


def main():
    source = A / 'own-qsa-gemm-unitrace-direct-model-v1.json'
    target = A / 'qsa-gemm-current-resource-profile-v1.json'
    if target.exists():
        previous = json.loads(target.read_text())
        assert previous['passed'] and previous['source_receipt'] == artifact(source)
    d = json.loads(source.read_text())
    assert d['passed'] and d['complete'] and d['ownership_closed'] and not d['active']
    assert d['boot_unchanged'] and not d['fresh_faults'] and not d['driver_dump_exists']
    assert d['engine_sha256'] == '1e7bf1cb5a7a8ec9b91e46bedbc2d7b7de3f0f7cbddbeeddecee80461a688b26'
    assert d['native_environment']['STRATA_SYCL_PREFILL_SCORE_GEMM'] == '1'
    assert d['native_environment']['STRATA_SYCL_PREFILL_QSA_MATH'] == 'bf16-prefix-optimized'
    assert d['collection']['proved']
    evidence = d['model_evidence']
    coarse = evidence['coarse_profile_lines']
    qsa = evidence['qsa_profile_lines']
    assert fields(coarse[0])['complete'] == fields(qsa[0])['complete'] == '1'
    c = [line for line in coarse if line.startswith('strata lm profile: layer=-1 ')]
    q = [line for line in qsa if line.startswith('strata lm qsa profile: layer=-1 ')]
    assert len(c) == len(q) == 1
    c, q = c[0], q[0]
    coarse_wall = numbers(c, 'wall_ms', fields(coarse[0])['wall_order'].split(','))
    coarse_queue = numbers(c, 'queue_ms', fields(coarse[0])['queue_order'].split(','))
    qsa_sections = fields(qsa[0])['sections'].split(',')
    qsa_queue = numbers(q, 'queue_ms', qsa_sections)
    qsa_host = numbers(q, 'host_wall_ms', qsa_sections)
    cpu = {k: float(v) for k, v in fields(evidence['cpu_wait_lines'][0]).items()}
    host = {k: float(v) for k, v in fields(evidence['cpu_host_lines'][0]).items()}
    assert cpu['available'] == cpu['complete'] == host['available'] == 1
    assert host['open_regions'] == 0
    outside_host_cpu = sum(v for k, v in host.items() if k.endswith('_outside_wait_cpu_s'))
    outside_host_wall = sum(v for k, v in host.items() if k.endswith('_outside_wait_wall_s'))
    assert abs(cpu['main_cpu_s'] - cpu['wait_cpu_s'] - outside_host_cpu - host['unclassified_cpu_s']) < 1e-6
    assert abs(cpu['wall_s'] - cpu['wait_wall_s'] - outside_host_wall - host['unclassified_wall_s']) < 1e-6
    collection = d['collection']
    for kind, key in [('API', 'host_API_rows'), ('Device', 'device_rows')]:
        assert sum(v['time_ns'] for v in collection[key]) == collection['totals_ns'][kind]
    request = evidence['suite']['results']['prefill_only']['requests']
    assert len(request) == 1 and request[0]['actual_read_tokens'] == 78453 and request[0]['actual_reused_tokens'] == 0
    result = dict(
        schema='strata-current-score-gemm-resource-profile-v1', passed=True,
        source_receipt=artifact(source), analysis_source=artifact(Path(__file__)),
        source_commit=d['source_commit'], binary_sha256=d['engine_sha256'],
        scope='One official bench.sh thinkingON78K request plus initialization, warmup,24generated tokens and teardown; diagnostic process, not clean speed or model quality.',
        placement=evidence['placement'], request=request[0],
        wrapper_ms=float(fields(coarse[0])['wrapper_ms']),
        coarse_wall_ms=coarse_wall, coarse_queue_ms=coarse_queue,
        requested_copy_fields={k: fields(c)[k] for k in ['upload_bytes', 'input_bytes_host_device', 'store_bytes_host_device']},
        qsa_queue_ms=qsa_queue, qsa_host_wall_ms=qsa_host,
        qsa_dispatch={k: fields(q)[k] for k in ['expected_chunks', 'expected_queries', 'matrix_chunks', 'fallback_chunks', 'matrix_queries', 'fallback_queries', 'staged_chunks']},
        cpu=cpu, host=host,
        main_CPU_fraction_inside_explicit_wait=cpu['wait_cpu_s'] / cpu['main_cpu_s'],
        whole_process_device_seconds=collection['totals_ns']['Device'] / 1e9,
        whole_process_API_elapsed_seconds=collection['totals_ns']['API'] / 1e9,
        top_device_rows=[compact(row) for row in sorted(collection['device_rows'], key=lambda row: row['time_ns'], reverse=True)[:20]],
        top_API_rows=[compact(row) for row in sorted(collection['host_API_rows'], key=lambda row: row['time_ns'], reverse=True)[:12]],
        attention_kernel_properties=[compact(row) for row in collection['kernel_properties'] if 'prefix_batch_impl<33, false, false>' in row['name']],
        all_kernel_properties_consumer='Full aggregate tables and kernel properties remain in source_receipt; no duplicate full table here.',
        interpretation=[
            'Device/API totals are nonexclusive whole-process aggregates, not prefill-only time, utilization or a critical-path partition.',
            'Queue markers include host submission gaps; host-wall sections include helper waits. Phase categories may overlap. Existing drains are used; event markers and collector/validation still add overhead.',
            'CPU fields partition only the calling thread; explicit wait CPU can be completion polling. Other runtime threads and unclassified CPU remain distinct.',
            'Requested upload/residual bytes describe source operations, not measured PCIe wire traffic. DeviceTiming rows supply no physical traffic/occupancy proof.',
            'Kernel properties are compiled metadata; dynamic local-accessor SLM and actual occupancy are not established.',
            'Physical262144 model/RESTORE, task quality and long decode remain unqualified by this24token probe.'
        ],
        ownership_closed=True, fresh_faults=[], driver_dump_exists=False,
        adoption=False, clean_speed=False)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(summary=str(target), **artifact(target))))


if __name__ == '__main__':
    main()
