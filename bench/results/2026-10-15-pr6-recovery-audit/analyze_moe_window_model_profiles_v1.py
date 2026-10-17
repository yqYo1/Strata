#!/usr/bin/env python3
"""Compare closed, matched MoE-window diagnostics without inferring utilization."""
import datetime
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
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def numbers(line):
    return {k: float(v) if '.' in v else int(v) for k, v in
            re.findall(r'([A-Za-z_][A-Za-z_0-9]*)=([0-9]+(?:\.[0-9]+)?)', line)}


def main():
    out = A / 'moe-window-model-profile-comparison-v1.json'
    assert not out.exists()
    parser_path = A / 'analyze_qsa_bf16_prefix_profile_v1.py'
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(parser_path) == '53e078c8af454403b8e878991d5c5799f2a7d0a711a6c989f367e34020b4b0c7'
    assert sha(owner_path) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    parser = module('window_profile_parser', parser_path)
    owner = module('window_profile_owner', owner_path)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cases, receipts = [], []
        for arm in (0, 2):
            path = A / f'moe-window-model-profile-window{arm}-v1.json'
            r = json.loads(path.read_text())
            assert r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
            assert r['fresh_faults'] == [] and r['boot_unchanged'] and not r['driver_dump_exists']
            for c in r['commands']:
                assert owner.closed(c) and c['exit_code'] == 0
                for old in c['owners'] + [c['direct_identity']]:
                    now = owner.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
            request, = r['benchmark_result']['results']['prefill_only']['requests']
            assert request['ok'] and request['eligible'] and request['actual_reused_tokens'] == 0
            assert request['actual_prompt_tokens'] == request['actual_read_tokens'] == 39262
            assert request['thinking_requested'] and request['reasoning_observed']
            vectors = {'wall_ms', 'queue_ms', 'samples', 'input_bytes_host_device', 'store_bytes_host_device',
                       'input_queue_host_device_ms', 'store_queue_host_device_ms'}
            coarse = parser.parse_table(r['coarse_profile_lines'], vectors,
                                        {'samples', 'input_bytes_host_device', 'store_bytes_host_device'})
            qsa = parser.parse_table(r['qsa_profile_lines'], {'queue_ms', 'host_wall_ms', 'samples'}, {'samples'})
            assert coarse['header']['wrapper_ms'] == qsa['header']['wrapper_ms']
            total = coarse['rows'][-1]
            wait, host = numbers(r['cpu_wait_lines'][0]), numbers(r['cpu_host_lines'][0])
            assert wait['available'] == wait['complete'] == 1
            assert wait['open_waits'] == wait['wait_throws'] == wait['clock_or_accounting_error'] == 0
            assert host['available'] == 1 and host['open_regions'] == 0
            assert math.isclose(wait['main_cpu_s'], wait['wait_cpu_s'] + wait['outside_wait_cpu_s'], abs_tol=1e-6)
            cases.append(dict(window=arm, receipt=dict(path=str(path), sha256=sha(path)),
                              request=request, wrapper_ms=float(coarse['header']['wrapper_ms']),
                              queue_interval_ms=dict(zip(coarse['header']['queue_order'].split(','), total['queue_ms'])),
                              wall_ms=dict(zip(coarse['header']['wall_order'].split(','), total['wall_ms'])),
                              qsa_interval_ms=dict(zip(qsa['header']['sections'].split(','), qsa['rows'][-1]['queue_ms'])),
                              requested_upload_bytes=total['upload_bytes'],
                              requested_input_bytes_host_device=total['input_bytes_host_device'],
                              requested_store_bytes_host_device=total['store_bytes_host_device'],
                              window_counters=numbers(r['window_lines'][0]) if arm else {},
                              CPU_wait=wait, CPU_host=host, coarse=coarse, QSA=qsa))
            receipts.append(r)
        baseline, candidate = cases
        for key in ('binary', 'boot_id', 'runtime_files', 'benchmark_options', 'model_identity',
                    'pack_metadata_sha256', 'MTP_metadata_sha256', 'external_profile_sha256', 'build_source_pins'):
            assert receipts[0][key] == receipts[1][key], key
        ignored_env = {'XDG_CACHE_HOME', 'STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT',
                       'STRATA_SYCL_PREFILL_MOE_WINDOW'}
        assert {k: v for k, v in receipts[0]['environment'].items() if k not in ignored_env} == {
            k: v for k, v in receipts[1]['environment'].items() if k not in ignored_env}
        assert baseline['request']['request_sha256'] == candidate['request']['request_sha256']
        counters = candidate['window_counters']
        assert counters['windows'] == 96 and counters['source_H2D_bytes'] == 62914560
        assert counters['dequants'] <= counters['jobs']
        differences = {}
        for section in ('queue_interval_ms', 'qsa_interval_ms', 'CPU_wait', 'CPU_host'):
            differences[section] = {k: candidate[section][k] - v for k, v in baseline[section].items()
                                    if k in candidate[section]}
        result = dict(schema='strata-moe-window-model-profile-comparison-v1', passed=True, complete=True,
                      generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      recipe_sha256=sha(__file__), parser_sha256=sha(parser_path), cases=cases,
                      same_binary_sha256=receipts[0]['binary']['sha256'],
                      wrapper_time_fraction=candidate['wrapper_ms'] / baseline['wrapper_ms'],
                      diagnostic_prompt_time_fraction=candidate['request']['prompt_ms'] / baseline['request']['prompt_ms'],
                      differences_candidate_minus_baseline=differences,
                      performance_adopted=False, general_quality_qualified=False,
                      physical262144_RESTORE_qualified=False,
                      limits=['One traced/validated diagnostic per arm; clean repeated speed is a separate measurement.',
                              'Nonexclusive queue intervals include host submission gaps; they are not exclusive GPU service or utilization.',
                              'Requested payload bytes are not measured PCIe/DRAM/cache traffic.',
                              '24 output tokens are a completion probe, not a decode benchmark or general quality test.',
                              'Output hashes describe observations; exact text equality is not a quality gate.'])
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(passed=True, comparison=str(out),
                             wrapper_time_fraction=result['wrapper_time_fraction'],
                             cases=[{k: c[k] for k in ('window', 'wrapper_ms', 'queue_interval_ms', 'window_counters')} for c in cases])))


if __name__ == '__main__':
    main()
