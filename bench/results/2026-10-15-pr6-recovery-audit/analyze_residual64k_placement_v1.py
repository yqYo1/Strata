#!/usr/bin/env python3
"""Compare closed, interleaved 64K+ placement samples without profiler claims."""
from pathlib import Path
import hashlib
import json
import statistics

A = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError('Run without Python -O; comparison gates must remain enabled')
    cohort_path = A / 'residual64k-placement-clean-cohort-v1.json'
    cohort = json.loads(cohort_path.read_text())
    assert cohort['passed'] and cohort['complete'] and cohort['active_sample'] is None
    assert cohort['order'] == [['gpu32768', 0], ['gpuall', 0], ['gpuall', 1],
                               ['gpu32768', 1], ['gpu32768', 2], ['gpuall', 2]]
    rows = []
    common_environment = None
    common_input = None
    variable_environment = {'XDG_CACHE_HOME', 'STRATA_PREFILL_LAYER_MAJOR_R_GPU',
                            'STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'}
    for item in cohort['completed']:
        path = Path(item['path'])
        assert digest(path) == item['sha256']
        sample = json.loads(path.read_text())
        assert sample['passed'] and sample['complete'] and sample['ownership_closed']
        assert not sample['active'] and not sample['fresh_faults'] and not sample['driver_dump_exists']
        assert sample['boot_unchanged'] and sample['binary'] == cohort['binary']
        assert sample['result']['real_engine_exit']['exit_code'] == 0
        environment = sample['environment']
        assert 'STRATA_TRACE' not in environment and 'UR_ENABLE_LAYERS' not in environment
        assert environment['STRATA_SYCL_PREFILL_COMMAND_PROFILE'] == '0'
        comparable_environment = {k: v for k, v in environment.items() if k not in variable_environment}
        if common_environment is None:
            common_environment = comparable_environment
        assert common_environment == comparable_environment
        request, = sample['benchmark_result']['results']['prefill_only']['requests']
        assert request['ok'] and request['eligible'] and request['stream_complete']
        assert request['thinking_requested'] and request['reasoning_observed']
        assert request['actual_prompt_tokens'] == request['actual_read_tokens'] >= 65536
        assert request['actual_reused_tokens'] == 0
        # Prefill-only deliberately caps output at24; finish_reason=length is
        # expected here. A complete stream does not establish answer quality.
        assert request['generated_tokens'] == 24
        input_identity = (request['request_sha256'], request['actual_prompt_tokens'])
        if common_input is None:
            common_input = input_identity
        assert common_input == input_identity
        args = sample['config']['args']
        assert args[args.index('--max-context') + 1] == '131072'
        assert args[args.index('--kv-resident') + 1] == '131072'
        arm = item['window']
        placement, = sample['placement_lines']
        expected_rows = request['actual_prompt_tokens'] - 1 if arm == 'gpuall' else 32768
        assert f'residual_gpu_rows={expected_rows} ' in placement
        if arm == 'gpuall':
            assert 'residual_host=0 ' in placement
        rows.append(dict(arm=arm, repetition=item['repetition'], prompt_tps=request['prompt_tps'],
                         request=request, receipt=str(path), sha256=item['sha256'],
                         placement=placement, window_counters=sample['window_lines']))
    assert len(rows) == 6
    summaries = {}
    for arm in ('gpu32768', 'gpuall'):
        selected = [row for row in rows if row['arm'] == arm]
        assert sorted(row['repetition'] for row in selected) == [0, 1, 2]
        speeds = [row['prompt_tps'] for row in selected]
        summaries[arm] = dict(n=3, samples=speeds, mean=statistics.mean(speeds),
                              median=statistics.median(speeds), stdev=statistics.stdev(speeds),
                              min=min(speeds), max=max(speeds))
    baseline, candidate = summaries['gpu32768'], summaries['gpuall']
    separated = candidate['min'] > baseline['max']
    gain = 100 * (candidate['median'] / baseline['median'] - 1)
    removed_bytes = 2 * 47 * (common_input[1] - 1 - 32768) * 10240 * 2
    result = dict(schema='strata-residual64k-placement-clean-analysis-v1', complete=True,
                  controller_sha256=cohort['source_sha256'], analyzer_sha256=digest(__file__),
                  cohort=dict(path=str(cohort_path), sha256=digest(cohort_path)),
                  binary=cohort['binary'], samples=rows, summaries=summaries,
                  actual_prompt_tokens=common_input[1], request_sha256=common_input[0],
                  median_gain_percent=gain, candidate_min_exceeds_baseline_max=separated,
                  performance_win_in_this_cohort=separated,
                  repeated_prefill_above600=candidate['min'] > 600,
                  repeated_prefill_above1000=candidate['min'] > 1000,
                  production_adopted=False, general_quality_qualified=False,
                  physical262144_RESTORE_qualified=False, sustained_decode_qualified=False,
                  removed_requested_BF16_interlayer_host_copy_bytes=removed_bytes,
                  scope='Official benchmark prompt timing, six fresh processes, thinking ON with24-token '
                        'completion probes, same input and binary, context131072/chunk8192/window2. '
                        'This is a bundled placement/window-eligibility change. Copy bytes are source-derived '
                        'requested bytes, not measured PCIe wire traffic. Window counters omit baseline '
                        'single-chunk work and cannot compare total expert dequantization. '
                        'No GPU busy, isolated bottleneck, bandwidth, general quality or sustained decode claim.')
    output = A / 'residual64k-placement-clean-analysis-v1.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['summaries', 'median_gain_percent',
                    'performance_win_in_this_cohort', 'repeated_prefill_above600']}))


if __name__ == '__main__':
    main()
