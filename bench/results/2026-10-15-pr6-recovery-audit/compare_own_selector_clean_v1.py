#!/usr/bin/env python3
"""Compare six closed independent long-prefill samples; preserve every sample."""
import datetime
import fcntl
import json
import math
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    order = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)]
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = A / 'own-selector-clean-comparison-v1.json'
        require(not target.exists(), 'fresh comparison')
        refs = {}; samples = []; previous_end = None; control = None
        ignored = {'STRATA_SYCL_SELECT_RADIX4', 'XDG_CACHE_HOME',
                   'STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'}
        for position, (arm, repetition) in enumerate(order):
            p = A / f'own-selector-clean-{arm}-r{repetition}-v1.json'
            r = json.loads(p.read_text()); refs[p.name] = sha(p)
            require(r['passed'] and r['complete'] and not r['active'] and r['performance_qualified']
                    and not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists'], 'healthy closed clean sample')
            require(r['selection_arm'] == arm and r['repetition'] == repetition
                    and r['measurement_order'] == position and r['dedicated_prefill_only'], 'actual declared arm/order')
            for c in r['commands']:
                require(module.closed(c), 'closed ownership')
                for old in c.get('owners', []):
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'owned identity absent')
            jobs = [c for c in r['commands'] if c['label'] == 'prefill-only']
            require(len(jobs) == 1, 'one independent sample process')
            job = jobs[0]
            start = datetime.datetime.fromisoformat(job['started_utc'])
            end = datetime.datetime.fromisoformat(job['finished_utc'])
            require(end > start and (previous_end is None or start >= previous_end), 'serial independent processes')
            previous_end = end
            effective = r['actual_experiment_environment']
            require(effective['STRATA_SYCL_SELECT_RADIX4'] == str(arm)
                    and effective['STRATA_SYCL_SELECT_CYCLIC'] == '0'
                    and effective['STRATA_SYCL_PROMPT_DIRECT'] == '1'
                    and effective['STRATA_PREFILL_CANONICAL_DENSE_BANK'] == '1'
                    and all(effective[k] == '0' for k in ('STRATA_PREFILL_LAYER_MAJOR_PROFILE',
                        'STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE', 'STRATA_PREFILL_BOUNDARY_AUDIT', 'STRATA_PREFILL_PHASE_AUDIT')), 'actual clean child controls')
            cohort = r['benchmark_result']['results']['prefill_only']
            require(cohort['complete'] and cohort['eligible_runs'] == len(cohort['requests']) == 1, 'complete individual workload')
            q = cohort['requests'][0]
            require(q['ok'] and q['eligible'] and q['actual_read_tokens'] == q['actual_prompt_tokens'] == 78453
                    and q['actual_reused_tokens'] == 0 and q['thinking_requested'] and q['generated_tokens'] == 24, 'same actual fresh input and scope')
            invariant = {k: r[k] for k in ('binary_sha256', 'source_pins', 'linked_inputs', 'benchmark_source',
                'engine_args', 'benchmark_options', 'model_identity', 'pack_metadata_sha256', 'MTP_metadata_sha256', 'boot_id', 'runtime_stack')}
            invariant['environment'] = {k: v for k, v in r['environment'].items() if k not in ignored}
            invariant['request_sha256'] = q['request_sha256']
            if control is None: control = invariant
            require(invariant == control, 'source/binary/work/environment fixed except explicit selector and private paths')
            require(all(math.isfinite(q[k]) and q[k] > 0 for k in ('prompt_tps', 'prompt_ms', 'ttft_s', 'total_s')), 'finite timing')
            samples.append(dict(arm=arm, repetition=repetition, order=position,
                started_utc=job['started_utc'], finished_utc=job['finished_utc'],
                owner=job['direct_identity'], prompt_tps=q['prompt_tps'], prompt_ms=q['prompt_ms'],
                ttft_s=q['ttft_s'], total_s=q['total_s'], prompt_tps_source=q['prompt_tps_source'],
                actual_input=78453, reuse=0, generated_tokens=24,
                output_sha256=q['output_sha256'], reasoning_sha256=q['reasoning_sha256']))
        groups = {}
        for arm in (0, 1):
            group = [v for v in samples if v['arm'] == arm]
            require(len(group) == 3, 'all3 independent samples per arm')
            groups[str(arm)] = {k: dict(values=[v[k] for v in group], median=statistics.median(v[k] for v in group),
                min=min(v[k] for v in group), max=max(v[k] for v in group))
                for k in ('prompt_tps', 'prompt_ms', 'ttft_s', 'total_s')}
        off, on = [groups[str(a)]['prompt_tps'] for a in (0, 1)]
        gain = (on['median'] / off['median'] - 1) * 100
        # Engineering decision preregistered before inspecting these samples;
        # this is deliberately not a statistical significance claim.
        separated = on['min'] > off['max']
        positive = bool(gain >= 1.0 and separated)
        result = dict(passed=True, complete=True, active=False, GPU_executed=False,
            recipe_sha256=sha(Path(__file__)), references=refs, actual_order=order,
            binary_sha256=control['binary_sha256'], request_sha256=control['request_sha256'], samples=samples,
            groups=groups, prompt_tps_median_gain_percent=gain, candidate_min_above_control_max=separated,
            preregistered_decision='Positive only if >=1% median PP gain and candidate min > control max over3 independent samples each; otherwise keep default0 and assess more samples/new settings. Not statistical significance.',
            positive_long_prefill_screen=positive, candidate_adopted=False,
            prefill_600_repeated=on['min'] > 600, prefill_1000_repeated=on['min'] >= 1000,
            decode_qualified=False, quality_qualified=False, physical_262144_qualified=False, restore_qualified=False,
            finished_utc=module.utc(), scope='Official bench.sh one actual78453 fresh input per owned native,6independent processes C8192FP32GPU32768; only dedicated-prefill selector flag differs. Output fingerprints descriptive, generated-token equality not required. No long decode/task-quality/context-boundary claim.')
        with target.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
        print(json.dumps(dict(passed=True, samples={a: g['prompt_tps']['values'] for a, g in groups.items()},
            median_gain_percent=gain, separated=separated, positive=positive)), flush=True)


if __name__ == '__main__': main()
