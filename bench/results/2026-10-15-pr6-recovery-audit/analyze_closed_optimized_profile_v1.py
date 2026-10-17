#!/usr/bin/env python3
"""Analyze the independently closed good arm; keep the paired failure intact."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    dst = A / 'optimized-profile-after-packed-stall-summary-v1.json'
    assert not dst.exists()
    pair_path = A / 'qsa-bf16-prefix-tilepack-profile-pair-v2.json'
    pair = json.loads(pair_path.read_text())
    assert not pair['passed'] and not pair['complete'] and pair['active_controller'] is None
    item = pair['completed'][0]
    source = Path(item['path'])
    assert item['arm'] == 'bf16-prefix-optimized' and sha(source) == item['sha256']
    r = json.loads(source.read_text())
    assert r['passed'] and r['complete'] and not r['active'] and r['ownership_closed']
    assert r['fresh_faults'] == [] and r['boot_unchanged'] and not r['driver_dump_exists']
    owner = module('closed_owner', A / 'direct_owner_transient_artifacts_v1.py')
    for c in r['commands']:
        assert owner.closed(c) and c['exit_code'] == 0
        for old in c['owners'] + [c['direct_identity']]:
            now = owner.identity(old['pid'])
            assert now is None or now['start_ticks'] != old['start_ticks']
    helpers = A / 'analyze_qsa_bf16_prefix_tilepack_profile_pair_v2.py'
    helper = module('profile_helpers', helpers)
    parser_path = A / 'analyze_qsa_bf16_prefix_profile_v1.py'
    assert sha(parser_path) == '53e078c8af454403b8e878991d5c5799f2a7d0a711a6c989f367e34020b4b0c7'
    parser = module('profile_parser', parser_path)
    vector = {'wall_ms', 'queue_ms', 'samples', 'input_bytes_host_device', 'store_bytes_host_device',
              'input_queue_host_device_ms', 'store_queue_host_device_ms'}
    coarse = parser.parse_table(r['coarse_profile_lines'], vector, {'samples', 'input_bytes_host_device', 'store_bytes_host_device'})
    fine = parser.parse_table(r['qsa_profile_lines'], {'queue_ms', 'host_wall_ms', 'samples'}, {'samples'})
    assert fine['header']['wrapper_ms'] == coarse['header']['wrapper_ms']
    request, = r['benchmark_result']['results']['prefill_only']['requests']
    assert request['ok'] and request['eligible'] and request['actual_prompt_tokens'] >= 32768 and request['actual_reused_tokens'] == 0
    census = r['shape_census']
    assert census['tokens'] == request['actual_read_tokens'] - 1
    shapes = helper.shape_summary(census)
    wait = helper.numeric_line(r['cpu_wait_lines'][0])
    host = helper.numeric_line(r['cpu_host_lines'][0])
    total, qsa = coarse['rows'][-1], fine['rows'][-1]
    small = shapes['buckets'][0]
    payload_rate = total['upload_bytes'] / total['queue_ms'][4] * 1000
    records = []
    endpoint = Path('/sys/bus/pci/devices/0000:05:00.0').resolve()
    for path in [endpoint, *endpoint.parents]:
        if not (path / 'vendor').is_file():
            continue
        row = dict(address=path.name, path=str(path))
        for key in ('vendor', 'device', 'class', 'current_link_speed', 'current_link_width', 'max_link_speed', 'max_link_width'):
            if (path / key).exists():
                row[key] = (path / key).read_text().strip()
        records.append(row)
    result = dict(schema='strata-single-closed-profile-after-paired-stall-v1', passed=True, complete=True,
                  generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), GPU_executed=False,
                  recipe_sha256=sha(__file__), helper_sha256=sha(helpers),
                  source_receipt=dict(path=str(source), sha256=sha(source)),
                  original_pair=dict(path=str(pair_path), sha256=sha(pair_path), passed=False, complete=False),
                  actual_prompt_tokens=request['actual_prompt_tokens'], batched_tokens=census['tokens'],
                  request_sha256=request['request_sha256'], binary_sha256=r['binary_sha256'],
                  diagnostic_prompt_ms=request['prompt_ms'], wrapper_ms=float(coarse['header']['wrapper_ms']),
                  coarse=coarse, QSA=fine, CPU_wait=wait, CPU_host=host, shapes=shapes,
                  sensors=helper.sensors_summary(r['sysfs_observation']),
                  PCI_topology_observed_after_stopped_diagnostic=records,
                  small_1_to_15_rows=dict(expert_calls=small['expert_chunk_calls'],
                                        expert_call_fraction=small['expert_chunk_calls'] / shapes['expert_chunk_calls'],
                                        routed_rows=small['routed_rows'],
                                        routed_row_fraction=small['routed_rows'] / shapes['routed_rows_per_GEMM_kind']),
                  requested_upload_bytes_per_inclusive_queue_second=payload_rate,
                  performance_adopted=False, model_quality_qualified=False,
                  physical262144_RESTORE_qualified=False,
                  limits=['Only the independently closed optimized arm is analyzed; the paired comparison remains failed.',
                          'One diagnostic request, not a repeated clean throughput result.',
                          'Queue marker intervals include host submission gaps and cannot establish GPU utilization or exclusive kernel time.',
                          'Logical dequant writes and requested copy bytes are not physical device/cache/PCIe counters.',
                          'PCI endpoint/ancestor sysfs values are link reports, not throughput; endpoint x1/2.5 must not be used as the whole external path.',
                          '8K per-layer histograms do not reveal aligned expert intersections across neighboring pieces.'])
    dst.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('actual_prompt_tokens', 'wrapper_ms', 'small_1_to_15_rows', 'requested_upload_bytes_per_inclusive_queue_second')}))


if __name__ == '__main__':
    main()
