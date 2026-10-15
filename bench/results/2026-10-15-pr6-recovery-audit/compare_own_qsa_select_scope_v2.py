#!/usr/bin/env python3
"""Main dedicated-prefill selector comparison; original/ref APIs also exact."""
import fcntl
import json
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id


def main(mode):
    require(mode in ('full', 'bench'), 'comparison mode')
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bp = A / 'own-prefill-select-scope-build-v1.json'
        build = json.loads(bp.read_text())
        require(build['passed'] and build['complete'] and not build['active'], 'build closed')
        require(all(sha(P / n) == h for n, h in build['source_pins'].items()), 'source stable')
        require(sha(build['component_binary']) == build['component_binary_sha256'], 'component stable')
        geometry = [('varied', a, 0) for a in range(3)] if mode == 'full' else [
            (p, a, i) for p in ('varied', 'equal') for a in range(3) for i in range(3)]
        records, refs, raw = [], {bp.name: sha(bp)}, {}
        for pattern, arm, index in geometry:
            path = A / f'own-qsa-select-scope-{mode}-{pattern}-a{arm}-r{index}-v2.json'
            x = json.loads(path.read_text())
            refs[path.name] = sha(path)
            require(x['passed'] and x['complete'] and not x['active'] and x['GPU_executed']
                and x['exit_code'] == 0 and not x['fresh_faults'] and x['boot_unchanged']
                and not x['driver_dump_exists'] and x['boot_id'] == boot_id(), 'healthy normal closure')
            require(x['mode'] == mode and x['pattern'] == pattern and x['arm'] == arm and x['index'] == index,
                    'arm labels')
            require(x['binary_sha256'] == build['component_binary_sha256'] and
                    x['compiled_source_receipt'] == dict(path=bp.name, sha256=sha(bp)), 'same compiled source')
            require(x['environment']['STRATA_SYCL_SELECT_RADIX4'] == str(int(arm == 1)) and
                    x['environment']['STRATA_SYCL_SELECT_CYCLIC'] == str(int(arm == 2)), 'actual immutable arm')
            for c in x['commands']:
                require(module.closed(c) and c['exit_code'] == 0, 'normal owned exit')
                for identity in c['owners']:
                    now = module.identity(identity['pid'])
                    require(now is None or now['start_ticks'] != identity['start_ticks'], 'owner absent')
            d = Path(x['dump']['path'])
            require(d.stat().st_size == x['dump']['bytes'] and sha(d) == x['dump']['sha256'], 'complete original dump')
            data = d.read_bytes()
            if pattern not in raw:
                raw[pattern] = data
            require(data == raw[pattern], 'all output/guard bytes exact within pattern')
            if mode == 'bench':
                require(not any(k in x['environment'] for k in ('UR_ENABLE_LAYERS', 'UR_LOG_TRACING',
                        'STRATA_TRACE', 'LD_PRELOAD')), 'clean timings')
            records.append((x, next(c['started_utc'] for c in x['commands'] if c['label'] == 'component')))
        records.sort(key=lambda pair: pair[1])
        result = dict(passed=True, complete=True, active=False, GPU_executed=False, references=refs,
            recipe_sha256=sha(Path(__file__)), binary_sha256=build['component_binary_sha256'],
            mode=mode, independent_processes=len(records), all_full_bytes_equal=True,
            actual_order=[[x['pattern'], x['arm'], x['index']] for x, _ in records],
            dump_sha256={p: next(x['dump']['sha256'] for x, _ in records if x['pattern'] == p) for p in raw},
            performance_qualified=False, quality_qualified=False, restore_qualified=False,
            physical_262144_qualified=False, candidate_adopted=False, finished_utc=module.utc())
        if mode == 'full':
            result.update(batches_per_process=17, queries_per_process=674, API_queries_per_process=2022, original_and_ref_oracle_outside_timing=True,
                physical_selector_rows_per_process=32,
                scope='Three immutable process arms, each674queries/17batches x3 APIs, original/ref alwaysMapping0, all output/guard bytes exact. '
                      'Expanded CPU oracle incl78452/78453 and262141..262144,8patterns,invalidsteps/hostvalidation. '
                      'Selector boundary coverage is not full engine physical-context/RESTORE/taskquality qualification.')
        else:
            full = A / 'own-qsa-select-scope-full-comparison-v2.json'
            g = json.loads(full.read_text())
            require(g['passed'] and g['complete'] and g['all_full_bytes_equal'], 'full exact gate')
            refs[full.name] = sha(full)
            samples = {p: {str(a): [x['component_total_ms'] for x, _ in records
                                   if x['pattern'] == p and x['arm'] == a] for a in range(3)}
                       for p in ('varied', 'equal')}
            medians = {p: {a: statistics.median(v) for a, v in arms.items()} for p, arms in samples.items()}
            result.update(samples_ms=samples, medians_ms=medians,
                time_change_percent={p: {a: (v / medians[p]['0'] - 1) * 100
                                         for a, v in arms.items() if a != '0'} for p, arms in medians.items()},
                component_positive={str(a): all(max(samples[p][str(a)]) < min(samples[p]['0'])
                                                for p in samples) for a in (1, 2)},
                geometry=dict(nq=256, n=78453, cap=2051, repeats=5,
                    all256queries_expanded_oracle=True, all_output_guards=True, score_and_step_inputs_unchanged=True),
                scope='18independent clean selector processes,9each inputpattern,3samples/arm/pattern. '
                      'Scores are synthetic; score producer excluded. Submission+drain timing only; '
                      'actual78453 whole-model correctness/performance and captured graph restore remain unqualified.')
        target = A / f'own-qsa-select-scope-{mode}-comparison-v2.json'
        with target.open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'references'}), flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2, 'full|bench')
    main(sys.argv[1])
