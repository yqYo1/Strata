#!/usr/bin/env python3
"""Compare two closed real-model cohorts; never start a workload."""
import hashlib
import json
import statistics
from pathlib import Path

A = Path(__file__).resolve().parent


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def summary(values):
    return dict(samples=values, n=len(values), median=statistics.median(values),
                mean=statistics.mean(values), stdev=statistics.stdev(values))


def normalized_environment(value):
    return {key: item for key, item in value.items() if key not in (
        'XDG_CACHE_HOME', 'STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT',
        'STRATA_SYCL_PREFILL_QSA_MATH')}


def normalized_args(args):
    args = list(args)
    args[args.index('--expert-profile') + 1] = '<same-original-profile-private-clone>'
    return args


def main():
    result_path = A / 'qsa-softmax-actual-model-comparison-v1.json'
    assert not result_path.exists()
    arms, references = {}, {}
    for arm in ('serial', 'subgroup'):
        path = A / f'qsa-softmax-actual-model-{arm}-r0-tokens65536-warm4-v1.json'
        value = json.loads(path.read_text())
        assert value['passed'] and value['complete'] and value['ownership_closed'] and not value['active']
        assert value['boot_unchanged'] and not value['fresh_faults'] and not value['driver_dump_exists']
        assert all(command['exit_code'] == 0 and command['direct_child_reaped'] and command['session_empty'] for command in value['commands'])
        assert value['score_gemm'] == 1 and value['window'] == 10
        assert value['environment']['STRATA_SYCL_PREFILL_QSA_MATH'] == {'serial': 'bf16-prefix-optimized', 'subgroup': 'bf16-prefix-subgroup'}[arm]
        assert value['environment']['STRATA_PREFILL_LAYER_MAJOR_PROFILE'] == value['environment']['STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE'] == value['environment']['STRATA_PREFILL_CPU_WAIT_PROFILE'] == '0'
        assert 'UR_ENABLE_LAYERS' not in value['environment'] and 'STRATA_TRACE' not in value['environment']
        assert value['single_engine_proof']['four_successful_fresh_requests']
        arms[arm], references[arm] = value, pin(path)
    serial, subgroup = arms['serial'], arms['subgroup']
    for key in ('binary', 'build_source_pins', 'runtime_files', 'model_identity', 'pack_metadata_sha256',
                'MTP_metadata_sha256', 'external_profile_sha256', 'benchmark_options', 'boot_id'):
        assert serial[key] == subgroup[key], key
    assert normalized_environment(serial['environment']) == normalized_environment(subgroup['environment'])
    assert normalized_args(serial['config']['args']) == normalized_args(subgroup['config']['args'])
    requests = {arm: value['benchmark_result']['results']['prefill_only']['requests'] for arm, value in arms.items()}
    for rows in requests.values():
        assert len(rows) == 4 and [row['run'] for row in rows] == list(range(4))
        assert all(row['ok'] and row['eligible'] and row['stream_complete'] and row['thinking_requested'] and
                   row['reasoning_observed'] and row['actual_prompt_tokens'] == row['actual_read_tokens'] == 78453 and
                   row['actual_reused_tokens'] == 0 and row['generated_tokens'] == 24 for row in rows)
    assert [(row['request_sha256'], row['actual_read_tokens']) for row in requests['serial']] == [(row['request_sha256'], row['actual_read_tokens']) for row in requests['subgroup']]
    groups = {}
    for arm, rows in requests.items():
        groups[arm] = dict(first=rows[0], subsequent=rows[1:],
            prefill=summary([row['prompt_tps'] for row in rows[1:]]),
            prompt_ms=summary([row['prompt_ms'] for row in rows[1:]]),
            short_decode_probe=summary([row['decode_tps'] for row in rows[1:]]))
    paired = [dict(run=a['run'], request_sha256=a['request_sha256'],
        prefill_tps_delta=b['prompt_tps'] - a['prompt_tps'],
        prompt_ms_saved=a['prompt_ms'] - b['prompt_ms'],
        short_decode_tps_delta=b['decode_tps'] - a['decode_tps'],
        output_hash_equal=a['output_sha256'] == b['output_sha256'],
        reasoning_hash_equal=a['reasoning_sha256'] == b['reasoning_sha256']) for a, b in zip(requests['serial'], requests['subgroup'])]
    ratio = groups['subgroup']['prefill']['median'] / groups['serial']['prefill']['median']
    result = dict(schema=1, passed=True, references=references, analysis=pin(Path(__file__)),
        binary=serial['binary'], configuration='Same engine/context131072/chunk8192/all-GPU-BF16-residual/scoreGEMM1/MoEwindow10; immutable attention selector is the only semantic environment difference.',
        groups=groups, paired_requests=paired, subsequent_median_prefill_improvement_percent=(ratio - 1) * 100,
        subsequent_median_prompt_ms_saved=groups['serial']['prompt_ms']['median'] - groups['subgroup']['prompt_ms']['median'],
        interpretation='Observed small long-prefill improvement in one fresh process per arm; three matched subsequent requests share process/host/cache state. First requests are kept separate. No cross-process variance elimination or causal mechanism claim.',
        decode_decision='The24token decode probes are retained individually and are slower in this subgroup cohort. Decode implementation was unchanged. This does not qualify sustained decode, explain the change or justify ignoring it as noise; long-generation evaluation is required before adoption.',
        adoption=False, default_changed=False, task_quality_qualified=False,
        physical262144_RESTORE_qualified=False, sustained_decode_qualified=False,
        next='Keep both experimental selectors; prioritize the larger remaining attention K-staging question. No default promotion from this1percent result; preserve the observed decode cost and physical-context/numerical failures.',
        remaining_prefill_goal=dict(target_tps=1000, measured_experimental_subsequent_median=groups['subgroup']['prefill']['median'],
            target_78453token_seconds=78.453, more_latency_seconds_to_remove=groups['subgroup']['prompt_ms']['median'] / 1000 - 78.453),
        ownership_closed=True, fresh_faults=[])
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(receipt=str(result_path), prefill_improvement_percent=result['subsequent_median_prefill_improvement_percent'],
        arms={arm: dict(PP=groups[arm]['prefill'], short_decode=groups[arm]['short_decode_probe']) for arm in groups}), ensure_ascii=False))


if __name__ == '__main__':
    main()
