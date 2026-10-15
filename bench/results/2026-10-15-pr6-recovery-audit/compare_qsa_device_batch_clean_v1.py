#!/usr/bin/env python3
"""Compare all six predeclared independent-process clean QSA samples."""
import datetime
import fcntl
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from analyze_prefill_window_profile_v2 import B, P, load_owner, require, sha


def stats(values):
    return dict(n=len(values), values=values, median=statistics.median(values),
        min=min(values), max=max(values), mean=statistics.mean(values),
        sample_stdev=statistics.stdev(values))


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        order = [(0,0), (1,0), (1,1), (0,1), (0,2), (1,2)]
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        records, samples, references = [], [], {}
        previous_end = None
        ignored = {'XDG_CACHE_HOME', 'STRATA_MEASUREMENT_ENGINE_START',
            'STRATA_MEASUREMENT_ENGINE_EXIT', 'STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH'}
        for position,(arm,repetition) in enumerate(order):
            path = A/f'qsa-device-batch-clean-{arm}-r{repetition}-v1.json'
            r = json.loads(path.read_text())
            require(r['passed'] and r['complete'] and not r['active'] and r['performance_qualified'] and
                not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists'], 'closed healthy clean sample')
            require((r['qsa_batch_arm'], r['repetition'], r['measurement_order']) ==
                (arm, repetition, position), 'predeclared measurement order')
            require(r['environment']['STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH'] == str(arm), 'effective immutable arm')
            for c in r['commands']:
                require(module.closed(c), 'all owned commands closed')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'owned identity retired')
            runs = [c for c in r['commands'] if c['label'] == 'prefill-only']
            require(len(runs) == 1 and runs[0]['exit_code'] == 0, 'one normal native/server run')
            start = datetime.datetime.fromisoformat(runs[0]['started_utc'])
            end = datetime.datetime.fromisoformat(runs[0]['finished_utc'])
            require(start < end and (previous_end is None or previous_end <= start), 'nonoverlapping serial native runs')
            previous_end = end
            if records:
                first = records[0]
                require(all(r[k] == first[k] for k in ('binary_sha256', 'wrapper_sha256', 'source_pins',
                    'linked_inputs', 'engine_args', 'benchmark_options', 'benchmark_source', 'model_identity',
                    'pack_metadata_sha256', 'MTP_metadata_sha256', 'boot_id', 'recipe_sha256')), 'same implementation/model/work')
                require({k:v for k,v in r['environment'].items() if k not in ignored} ==
                    {k:v for k,v in first['environment'].items() if k not in ignored}, 'no uncontrolled environment difference')
            suite = r['benchmark_result']
            require(suite['complete'] and suite['params']['think'], 'complete thinking-enabled suite')
            requests = suite['results']['prefill_only']['requests']
            require(len(requests) == 1, 'one independent request per process')
            q = requests[0]
            require(q['ok'] and q['eligible'] and q['stream_complete'] and q['thinking_requested'] and
                q['actual_prompt_tokens'] == q['actual_read_tokens'] == 78453 and q['actual_reused_tokens'] == 0 and
                q['generated_tokens'] == 24 and q['finish_reason'] == 'length', 'same actual long request and output cap')
            require(r['request_journal_verification']['valid'] and r['request_journal_verification']['complete'], 'valid complete journal')
            require(all(math.isfinite(q[k]) and q[k] > 0 for k in ('prompt_ms', 'prompt_tps', 'ttft_s')), 'finite timing')
            if samples:
                require(q['request_sha256'] == samples[0]['request_sha256'], 'identical request body in all six samples')
            samples.append(dict(order=position, arm=arm, repetition=repetition, receipt=path.name,
                request_sha256=q['request_sha256'], actual_read_tokens=q['actual_read_tokens'],
                engine_prefill_tps=q['prompt_tps'], exact_prefill_tps=q['actual_read_tokens']/(q['prompt_ms']/1000),
                prompt_ms=q['prompt_ms'], ttft_s=q['ttft_s'], generated_tokens=q['generated_tokens'],
                decode_tps_unqualified=q['decode_tps'], output_sha256=q['output_sha256'],
                reasoning_sha256=q['reasoning_sha256'], started_utc=runs[0]['started_utc'],
                finished_utc=runs[0]['finished_utc']))
            references[path.name] = sha(path)
            records.append(r)
        cohorts = {}
        for arm in (0,1):
            cohort = [s for s in samples if s['arm'] == arm]
            require(len(cohort) == 3, 'three independent native processes per arm')
            cohorts[str(arm)] = {k:stats([s[k] for s in cohort]) for k in
                ('engine_prefill_tps', 'exact_prefill_tps', 'prompt_ms', 'ttft_s')}
        effect = 100*(cohorts['1']['exact_prefill_tps']['median']/cohorts['0']['exact_prefill_tps']['median']-1)
        result = dict(passed=True, complete=True, active=False, GPU_executed=False,
            recipe_sha256=sha(Path(__file__)), references=references, samples=samples, cohorts=cohorts,
            binary_sha256=records[0]['binary_sha256'], median_prefill_change_percent=effect,
            measured_clean_performance=True, performance_adopted=False, quality_qualified=False,
            physical_262144_qualified=False, restore_qualified=False, finished_utc=module.utc(),
            scope='Six serial independent processes in predeclared0,1,1,0,0,1 order; same78453-input request body, thinkingON,24-output cap. Clean long-prefill timing only. n3 does not establish statistical significance; short decode timing is not decode qualification. Retain individual samples.')
        target = A/'qsa-device-batch-clean-comparison-v1.json'
        with target.open('x') as out:
            json.dump(result, out, indent=2)
            out.write('\n')
        print(json.dumps(dict(passed=True, cohorts=cohorts, median_prefill_change_percent=effect)))


if __name__ == '__main__':
    main()
