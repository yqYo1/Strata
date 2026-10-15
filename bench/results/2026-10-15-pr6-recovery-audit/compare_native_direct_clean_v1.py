#!/usr/bin/env python3
"""Close the predeclared six-process Direct comparison; no engine execution."""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics

A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ORDER = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)]


def require(condition, why):
    if not condition:
        raise RuntimeError(why)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_env(record):
    env = dict(record['environment'])
    for key in ('STRATA_SYCL_PROMPT_DIRECT', 'XDG_CACHE_HOME',
                'STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'):
        require(key in env, 'missing predeclared variable ' + key)
        env.pop(key)
    return env


def stats(values):
    require(len(values) == 3 and all(math.isfinite(x) and x > 0 for x in values),
            'three finite positive independent samples')
    median = statistics.median(values)
    return dict(n=len(values), min=min(values), max=max(values), median=median,
                mean=statistics.mean(values), sample_stdev=statistics.stdev(values),
                median_absolute_deviation=statistics.median(abs(x-median) for x in values))


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = A/'native-direct-clean-comparison-v1.json'
        require(not output.exists(), 'immutable fresh comparison')
        base = None
        common_request = None
        samples = []
        references = {}
        for position, (arm, rep) in enumerate(ORDER):
            path = A/f'native-direct-clean-{arm}-r{rep}-v1.json'
            row = json.loads(path.read_text())
            require(row['passed'] and row['complete'] and not row['active'] and
                    row['performance_qualified'] and row['GPU_executed'], 'unqualified closed sample')
            require((row['direct_arm'], row['repetition'], row['measurement_order']) ==
                    (arm, rep, position), 'predeclared independent order')
            require(not row['fresh_faults'] and row['boot_unchanged'] and
                    not row['driver_dump_exists'], 'fault closure')
            require(row['result']['server_exit_code'] ==
                    row['result']['real_engine_exit']['exit_code'] == 0, 'normal native/server exit')
            for command in row['commands']:
                require(command['direct_child_reaped'] and command['session_empty'] and
                        command['observation_complete'] and not command['survivors'] and
                        not command['errors'] and not command['cleanup'], 'owned normal retirement')
            require(row['request_journal_verification']['valid'] and
                    row['request_journal_verification']['complete'] and
                    not row['request_journal_verification']['pending'], 'request journal closure')
            require(row['environment']['STRATA_SYCL_PROMPT_DIRECT'] == str(arm) and
                    row['config']['env']['STRATA_SYCL_PROMPT_DIRECT'] == str(arm), 'effective arm')
            require(row['environment'].get('STRATA_TRACE') is None and
                    all(row['environment'].get(k, '0') == '0' for k in
                        ('STRATA_PREFILL_PHASE_AUDIT', 'STRATA_PREFILL_BOUNDARY_AUDIT',
                         'STRATA_PREFILL_ROUTED_FP16_OBSERVE')), 'clean capture state')
            suite = row['benchmark_result']
            cohort = suite['results']['prefill_only']
            require(suite['complete'] and not suite['results']['failures'] and
                    suite['params']['think'] and suite['params']['mode'] == 'prefill-only' and
                    cohort['complete'] and cohort['eligible_runs'] == len(cohort['requests']) == 1,
                    'official prefill-only cohort')
            request = cohort['requests'][0]
            require(request['ok'] and request['eligible'] and not request['eligibility_errors'] and
                    request['stream_complete'] and request['thinking_requested'] and
                    request['reasoning_observed'] and request['generated_tokens'] == 24 and
                    request['actual_prompt_tokens'] == request['actual_read_tokens'] == 78453 and
                    request['actual_reused_tokens'] == 0 and request['prompt_ms'] > 0,
                    'same actual long input and completed end marker')
            if base is None:
                base = row
                common_request = request['request_sha256']
            else:
                for key in ('binary_sha256', 'source_pins', 'linked_inputs', 'benchmark_source',
                            'engine_args', 'model_identity', 'pack_metadata_sha256',
                            'MTP_metadata_sha256', 'benchmark_options', 'wrapper_template_sha256',
                            'boot_id', 'recipe_sha256', 'references', 'limits', 'placement_lines'):
                    require(row[key] == base[key], 'comparison identity changed: ' + key)
                require(normalized_env(row) == normalized_env(base), 'undeclared environment difference')
                left, right = dict(row['config']), dict(base['config'])
                for config in (left, right):
                    config.pop('exe'); config.pop('log')
                    config['env'] = dict(config['env'])
                    config['env'].pop('STRATA_SYCL_PROMPT_DIRECT')
                require(left == right, 'undeclared engine config difference')
                require(suite['comparison_identity'] == base['benchmark_result']['comparison_identity'],
                        'benchmark construction identity changed')
            require(request['request_sha256'] == common_request, 'actual request body changed')
            references[path.name] = sha(path)
            samples.append(dict(order=position, direct=arm, repetition=rep,
                                receipt=path.name, finished_utc=row['finished_utc'],
                                input_tokens=request['actual_read_tokens'],
                                prompt_ms=request['prompt_ms'], reported_prompt_tps=request['prompt_tps'],
                                derived_prompt_tps=78453*1000/request['prompt_ms'],
                                ttft_s=request['ttft_s'], request_sha256=common_request))
        arms = {}
        for arm in (0, 1):
            selected = [s for s in samples if s['direct'] == arm]
            arms[str(arm)] = {key:stats([s[key] for s in selected]) for key in
                             ('prompt_ms', 'reported_prompt_tps', 'derived_prompt_tps', 'ttft_s')}
        ratio = arms['0']['prompt_ms']['median']/arms['1']['prompt_ms']['median']
        result = dict(schema='owned-native-Direct-clean-comparison-v1', passed=True,
                      complete=True, active=False, GPU_executed=False,
                      recipe_sha256=sha(Path(__file__)), references=references,
                      binary_sha256=base['binary_sha256'], samples=samples, arms=arms,
                      direct1_median_prompt_speed_ratio=ratio,
                      direct1_median_prompt_speed_percent=(ratio-1)*100,
                      direct1_faster_than_all_direct0_samples=
                          arms['1']['prompt_ms']['max'] < arms['0']['prompt_ms']['min'],
                      statistical_significance_claimed=False, product_default_changed=False,
                      quality_qualified=False, decode_qualified=False,
                      physical_262144_qualified=False, restore_qualified=False,
                      scope='Three independent processes per arm, exact 78453-token workload, fixed '
                            'FP32/R32768/C8192/nativebank1/head0/QSA1/alias1; clean prompt-only '
                            'comparison. Twenty-four thinking tokens only confirm prompt completion. '
                            'Native lifecycle uses prior same-binary diagnostics, not clean TRACE observation. '
                            'Spread is reported, not removed by three repetitions.')
        with output.open('x') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
        print(json.dumps(dict(passed=True, arms=arms, direct1_speed_percent=(ratio-1)*100)))


if __name__ == '__main__':
    main()
