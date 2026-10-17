#!/usr/bin/env python3
"""Verify all six individual samples and describe this cohort's speed evidence."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = A / 'moe-window-clean-comparison-v1.json'
    assert not out.exists()
    source = A / 'moe-window-clean-cohort-v3.json'
    cohort = json.loads(source.read_text())
    assert cohort['passed'] and cohort['complete'] and cohort['active_sample'] is None
    assert cohort['order'] == [[0, 0], [2, 0], [2, 1], [0, 1], [0, 2], [2, 2]]
    assert len(cohort['completed']) == 6
    assert sha(A / 'run_moe_window_clean_cohort_v3.py') == cohort['source_sha256']
    helper = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(helper) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec = importlib.util.spec_from_file_location('clean_summary_owner', helper)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    samples, environments = [], []
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in cohort['completed']:
            path = Path(item['path'])
            assert sha(path) == item['sha256']
            r = json.loads(path.read_text())
            assert r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
            assert r['GPU_executed'] and r['fresh_faults'] == [] and r['boot_unchanged'] and not r['driver_dump_exists']
            assert r['binary'] == cohort['binary']
            for command in r['commands']:
                assert owner.closed(command) and command['exit_code'] == 0
                for old in command['owners'] + [command['direct_identity']]:
                    now = owner.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
            env = r['environment']
            assert 'STRATA_TRACE' not in env and 'UR_ENABLE_LAYERS' not in env
            assert not any(k.startswith(('UR_LOG_', 'UNITRACE_', 'PTI_')) for k in env)
            for key in ('STRATA_PREFILL_LAYER_MAJOR_PROFILE', 'STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE',
                        'STRATA_PREFILL_CPU_WAIT_PROFILE', 'STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS'):
                assert env[key] == '0'
            ignore = {'STRATA_SYCL_PREFILL_MOE_WINDOW', 'XDG_CACHE_HOME',
                      'STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'}
            environments.append({k: v for k, v in env.items() if k not in ignore})
            request, = r['benchmark_result']['results']['prefill_only']['requests']
            assert request == item['request'] and request['ok'] and request['eligible'] and request['stream_complete']
            assert request['actual_prompt_tokens'] == request['actual_read_tokens'] == 39262
            assert request['actual_reused_tokens'] == 0 and request['thinking_requested'] and request['reasoning_observed']
            assert request['generated_tokens'] == 24
            samples.append(dict(window=item['window'], repetition=item['repetition'],
                                receipt=dict(path=str(path), sha256=sha(path)), request=request))
        assert all(e == environments[0] for e in environments)
        assert len({s['request']['request_sha256'] for s in samples}) == 1
        summaries = {}
        for arm in (0, 2):
            subset = [s['request'] for s in samples if s['window'] == arm]
            rates = [s['prompt_tps'] for s in subset]
            summaries[str(arm)] = dict(n=len(subset), samples_tok_s=rates,
                                      median_tok_s=statistics.median(rates), min_tok_s=min(rates), max_tok_s=max(rates),
                                      median_prompt_ms=statistics.median(s['prompt_ms'] for s in subset))
            assert len(subset) == 3 and rates == cohort['summaries'][str(arm)]['samples']
        paired = []
        for repetition in range(3):
            r0, = [s['request'] for s in samples if s['window'] == 0 and s['repetition'] == repetition]
            r2, = [s['request'] for s in samples if s['window'] == 2 and s['repetition'] == repetition]
            paired.append(dict(repetition=repetition, baseline_tok_s=r0['prompt_tps'], candidate_tok_s=r2['prompt_tps'],
                               rate_gain_fraction=r2['prompt_tps'] / r0['prompt_tps'] - 1))
        result = dict(schema='strata-moe-window-clean-comparison-v1', passed=True, complete=True,
                      recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      source_cohort=dict(path=str(source), sha256=sha(source)), analysis_recipe_sha256=sha(__file__),
                      same_binary=cohort['binary'], samples=samples, summaries=summaries, paired=paired,
                      median_rate_gain_fraction=summaries['2']['median_tok_s'] / summaries['0']['median_tok_s'] - 1,
                      all_candidate_samples_above_all_baseline_samples=summaries['2']['min_tok_s'] > summaries['0']['max_tok_s'],
                      performance_evidence_qualified_for_this_cohort=True, performance_adopted=False,
                      general_quality_qualified=False, physical262144_RESTORE_qualified=False,
                      prefill600_milestone_achieved=False, prefill1000_goal_achieved=False, decode70_goal_achieved=False,
                      limits=['Six fresh native processes in counterbalanced order; three samples per arm estimate observed variation, not all future variation.',
                              '39K prompt/64K capacity, C8192/RGPU32768/BF16/optimized-QSA and no prefix reuse only.',
                              '24 thinking output tokens are a completion probe, not a decode-speed or general task-quality benchmark.',
                              'No exact output-text equality gate; output/reasoning hashes are retained as observations.',
                              'Failed v1 diagnostic-contaminated sample and v2 prelaunch port failure remain failed and are excluded, not replaced within this cohort.',
                              'No EU, DRAM, cache or physical PCIe saturation is established by these measurements.'])
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(comparison=str(out), summaries=summaries,
                             median_rate_gain_fraction=result['median_rate_gain_fraction'])))


if __name__ == '__main__':
    main()
