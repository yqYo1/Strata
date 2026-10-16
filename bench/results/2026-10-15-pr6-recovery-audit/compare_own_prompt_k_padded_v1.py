#!/usr/bin/env python3
"""Main serial exact component comparison; keep all independent samples."""
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
        build_path = A / 'own-prompt-k-padded-build-v1.json'
        build = json.loads(build_path.read_text())
        require(build['passed'] and build['complete'] and not build['active'], 'build closed')
        require(all(sha(P / n) == h for n, h in build['source_pins'].items()), 'source stable')
        require(sha(build['component_binary']) == build['component_binary_sha256'], 'fixture stable')
        geometry = [(d, a, 0) for d in (0, 1) for a in (0, 1, 2)] if mode == 'full' else [
            (1, a, i) for a in (0, 1, 2) for i in range(3)]
        records = []
        refs = {build_path.name: sha(build_path)}
        reference_bytes = None
        for direct, arm, index in geometry:
            path = A / f'own-prompt-k-padded-{mode}-d{direct}-a{arm}-r{index}-v1.json'
            x = json.loads(path.read_text())
            refs[path.name] = sha(path)
            require(x['passed'] and x['complete'] and not x['active'] and x['GPU_executed']
                and x['exit_code'] == 0 and not x['fresh_faults'] and x['boot_unchanged']
                and not x['driver_dump_exists'] and x['boot_id'] == boot_id(), 'actual healthy closure')
            require(x['mode'] == mode and x['direct'] == direct and x['arm'] == arm and x['index'] == index,
                    'exact arm labels')
            require(x['binary_sha256'] == build['component_binary_sha256'] and
                    x['source_pins'] == build['source_pins'], 'same compiled source')
            for c in x['commands']:
                require(module.closed(c) and c['exit_code'] == 0, 'owned normal exit')
                for identity in c['owners']:
                    now = module.identity(identity['pid'])
                    require(now is None or now['start_ticks'] != identity['start_ticks'], 'owned process absent')
            dump = Path(x['dump']['path'])
            require(dump.stat().st_size == x['dump']['bytes'] and sha(dump) == x['dump']['sha256'],
                    'full original dump identity')
            data = dump.read_bytes()
            if reference_bytes is None:
                reference_bytes = data
            require(data == reference_bytes, 'all full dumps byte-exact across arms')
            if mode == 'bench':
                require(not any(k in x['environment'] for k in ('UR_ENABLE_LAYERS', 'UR_LOG_TRACING',
                        'STRATA_TRACE', 'LD_PRELOAD')), 'clean timing environment')
            records.append((x, next(c['started_utc'] for c in x['commands'] if c['label'] == 'component')))
        records.sort(key=lambda pair: pair[1])
        result = dict(passed=True, complete=True, active=False, GPU_executed=False,
            references=refs, recipe_sha256=sha(Path(__file__)),
            binary_sha256=build['component_binary_sha256'], source_pins=build['source_pins'],
            mode=mode, independent_processes=len(records),
            actual_order=[[x['direct'], x['arm'], x['index']] for x, _ in records],
            dump_sha256=records[0][0]['dump']['sha256'], all_full_bytes_equal=True,
            performance_qualified=False, quality_qualified=False,
            restore_qualified=False, physical_262144_qualified=False, finished_utc=module.utc())
        if mode == 'full':
            old_path = A / 'own-prompt-k-cellmajor-comparison-v2.json'
            old = json.loads(old_path.read_text())
            require(result['dump_sha256'] == old['full_dump_sha256'], 'previous fixture full fingerprint')
            result.update(full_cases_per_process=16, full_cases=96,
                previous_full_dump_fingerprint_equal=True,
                scope='All six actual process dumps exact, Direct0/1 x mappings0/1/2, all16format/page/scaled cases. '
                      'This is component parity only, not model/task/context/RESTORE qualification.')
        else:
            full_path = A / 'own-prompt-k-padded-full-comparison-v1.json'
            full = json.loads(full_path.read_text())
            require(full['passed'] and full['complete'] and full['all_full_bytes_equal'], 'full parity first')
            result['references'][full_path.name] = sha(full_path)
            samples = {str(a): [x['component_total_ms'] for x, _ in records if x['arm'] == a]
                       for a in (0, 1, 2)}
            medians = {a: statistics.median(values) for a, values in samples.items()}
            positive = max(samples['2']) < min(samples['0'])
            result.update(samples_ms=samples, medians_ms=medians,
                pitch40_median_time_change_percent=(medians['2'] / medians['0'] - 1) * 100,
                pitch40_component_positive=positive, candidate_adopted=False,
                geometry=dict(nq=256, cap=2051, repeats=20, scalar_oracle_queries=[0,1,2,255],
                              counts='13,67,then2051', all_output_finite=True, tail_canaries=True),
                next_action='Matched actual long-model qualification, no performance adoption yet.' if positive else
                            'Reject pitch40 for adoption; do not run slower long-model candidate.',
                scope='Nine independent clean component processes, identical full output bytes. '
                      'Host submission/queue/drain interval; no bank/occupancy/kernel-service or whole-model speed claim.')
        target = A / f'own-prompt-k-padded-{mode}-comparison-v1.json'
        with target.open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('references', 'source_pins')}), flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2, 'full|bench')
    main(sys.argv[1])
