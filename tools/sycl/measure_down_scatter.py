#!/usr/bin/env python3
"""Main-owned selective Down/accumulation qualification; default-off engine hook."""
import fcntl
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'moe-down-scatter-component-v7'
BUILD = B / 'main-native-gdn-engine/record.json'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/moe-down-scatter-current.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'
TARGET = 'sycl_moe_down_scatter_bench'
PERF_ORDER = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P)


def controller_inputs(revision):
    """Pin the complete local Python/control inputs, separately from the build."""
    names = ('tools/sycl/measure_down_scatter.py', 'tools/sycl/measurement_owner.py',
             'tools/sycl/measurement_guards.py', 'tools/sycl/measurement_environment.json')
    pins = {}
    for name in names:
        original = git('show', revision + ':' + name)
        guards.require((P / name).read_bytes() == original,
                       'uncommitted or changed controller input: ' + name)
        pins[name] = hashlib.sha256(original).hexdigest()
    return pins


def compact_result(record):
    """Keep checked totals and all used timings; raw events stay in owned output."""
    result = dict(record)
    checks, samples = [], []
    columns = ('process', 'rows', 'irregular', 'round', 'order', 'arm', 'wall_ms', 'queue_interval_ms')
    for check in record['checks']:
        row = {k: v for k, v in check.items() if k != 'events'}
        events = check.get('events', [])
        if 'events' in check:
            counts = Counter((e.get('kind', 'missing'), e.get('status', 'missing')) for e in events)
            row['event_totals'] = [[kind, status, count] for (kind, status), count in sorted(counts.items())]
            row['event_count'] = len(events)
            row['immutable_input_failures'] = sum(e.get('inputs_immutable') is False for e in events)
            for event in events:
                if (event.get('kind') == 'sample' and isinstance(event.get('round'), int)
                        and event['round'] >= 0):
                    samples.append([event.get(name) for name in columns])
        checks.append(row)
    result['checks'] = checks
    if samples:
        result.update(timing_sample_columns=list(columns), timing_samples=samples,
            timing_sample_scope='Every measured sample, including unsuccessful gates; warmup durations and repeated shape/guard fields omitted only after validation. Full raw output is temporary.')
    return result


def public_timing_samples(checks):
    """Require six real immutable-arm processes and the full frozen shape set."""
    checks = [c for c in checks if c['mode'] == 'perf']
    guards.require(len(checks) == 6 and [c['command'][-3:] for c in checks] ==
        [['--bench-public', str(p), str(a)] for a, p in PERF_ORDER], 'public process order differs')
    guards.require(all(c['passed'] and c['ownership_closed'] and c['exit_code'] == 0 and
        all(isinstance(c['native_identity'][k], int) and c['native_identity'][k] > 0
            for k in ('pid','start_ticks')) for c in checks) and
        len({(c['native_identity']['pid'], c['native_identity']['start_ticks']) for c in checks}) == 6,
        'six independent closed public timing processes required')
    samples = []
    for check, (arm, process) in zip(checks, PERF_ORDER):
        guards.require(check['startup_down_selector'] == ('large' if arm else '0'),
                       'effective startup environment differs from public arm')
        routes = [e for e in check['events'] if e['kind'] == 'selected_route']
        guards.require(len(routes) == 14 and all(e['enabled'] == arm and
            e['selected'] == bool(arm and e['rows'] >= 2334) and e['false_route_no_write'] and
            e['expected_mask'] == 0 for e in routes), 'wrong public startup route')
        group = [e for e in check['events'] if e['kind'] == 'sample' and e['round'] >= 0]
        warm = [e for e in check['events'] if e['kind'] == 'sample' and e['round'] == -1]
        guards.require(len(check['events']) == 112 and len(warm) == 14 and
            {(e['rows'],e['irregular']) for e in warm} ==
            {(r,i) for r in (64,160,2333,2334,2335,5963,8192) for i in (0,1)} and
            len(group) == 84 and all(e['process'] == process and e['arm'] == arm
            for e in group), 'public process/arm labels differ')
        samples.extend(group)
    guards.require(len(samples) == 504 and
        {(e['rows'], e['irregular'], e['process'], e['round'], e['arm']) for e in samples} ==
        {(r, i, p, n, a) for r in (64, 160, 2333, 2334, 2335, 5963, 8192)
         for i in (0, 1) for p in range(3) for n in range(6) for a in (0, 1)} and
        all(e['public_dispatch'] == 1 and e['order'] == 0 and
            e['fused_route'] == bool(e['arm'] and e['rows'] >= 2334) and
            e['repeat'] == (256 if e['rows'] < 256 else 16) and
            e['window_rows'] == 78451 and e['accumulator_bytes'] == 78451*2560*4 and
            e['inputs_immutable'] and e['status'] == 'PASS' and
            all(math.isfinite(e[k]) and e[k] > 0 for k in ('wall_ms','queue_interval_ms'))
            for e in samples), 'public timing shape/coverage/granularity differs')
    return samples


def postchecks(record, boot, cursor):
    """A missing fault observation is failed qualification, never a clean run."""
    record.update(postchecks_complete=False, postcheck_errors=[], fresh_faults=None,
                  boot_unchanged=False, driver_after=None)
    try:
        after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
        record['fresh_faults'] = [line for line in after.splitlines() if re.search(
            r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', line)]
    except Exception as error:
        record['postcheck_errors'].append(dict(check='kernel_journal', error=str(error)))
    try:
        record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except Exception as error:
        record['postcheck_errors'].append(dict(check='boot_identity', error=str(error)))
    try:
        record['driver_after'] = guards.driver_guard()
    except Exception as error:
        record['postcheck_errors'].append(dict(check='driver_guard', error=str(error)))
    record['postchecks_complete'] = not record['postcheck_errors']
    record['passed'] = bool(record['passed'] and record['postchecks_complete'] and
        record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])


def main():
    guards.require(len(sys.argv) == 2 and sys.argv[1] in ('plan', 'ownership', 'smoke', 'validate', 'perf'),
                   'use measure_down_scatter.py plan|ownership|smoke|validate|perf')
    mode = sys.argv[1]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.driver_guard()
        build = json.loads(BUILD.read_text())
        guards.require(build['complete'] and build['passed'] and build['ownership_closed'] and
                       not build['active'], 'closed successful build required')
        guards.require(Path(build['project']['root']).resolve() == P,
                       'run from the measured engine worktree after integration')
        for source in [build['project'], *build['dependencies']]:
            for name, expected in source['source_sha256'].items():
                guards.require(digest(Path(source['root']) / name) == expected, 'build input changed: ' + name)
        fixture = build['fixtures'][TARGET]
        guards.require(digest(fixture['path']) == fixture['sha256'], 'fixture changed')
        if mode == 'plan':
            guards.require(not ROOT.exists(), 'inspect existing job; do not restart')
            prior_bytes = git('show', 'HEAD:' + str(RESULT.relative_to(P)))
            guards.require(prior_bytes == RESULT.read_bytes(), 'commit the preceding Down result before replacement')
            prior = json.loads(prior_bytes)
            guards.require(prior['ownership_closed'] and not prior['active'] and prior['mode'] != 'plan',
                           'closed original Down result required; preserve its status')
            revision = git('rev-parse', 'HEAD').decode().strip()
            pins = controller_inputs(revision)
            ROOT.mkdir(mode=0o700)
            record = dict(source_commit=build['source_commit'], fixture=fixture, checks=[],
                controller_source_commit=revision, controller_sha256=pins,
                build_record_sha256=digest(BUILD), run_root=str(ROOT), mode='plan',
                ownership_closed=True, GPU_executed=False,
                complete=False, passed=False, active=False, model_performance_adopted=False,
                model_quality_qualified=False, physical262144_model_RESTORE_qualified=False,
                consumer='One Down/scatter component comparison; retire raw logs/JIT after extracting samples.',
                prospective_gates='New production-matched comparison; v6 and uniform failures remain unchanged.54 ownership,45 traced small,110 full correctness events required on this exact fixture. Six immutable startup0/large processes in A0,B0,B1,A1,A2,B2 order,6 rounds over14 shapes:504 used samples. Both arms call the public try hook and fallback, including control flag0. Repeat256 for64/160 rows,16 otherwise; one warmup round per process/shape. Unchanged advance gate: candidate/base geomean<=0.97, no shape>1.05 and every>=2334-row shape improves in each process pair. Component advance only.',
                limits='FP16 Down2560x640, BF16-round then explicit FP32 FMA. Fixed78451-token FP32 accumulator; contiguous/distributed maps. Actual production startup-selected hook in both arms, with public dense+accumulation fallback. Compact below128,wide otherwise; cached-wide only when enabled at>=2334. Marker intervals include submission gaps; no physical traffic, occupancy, saturation or model speed inferred. This corrects comparison methodology and does not establish an engine speedup from the old low-level baseline.',
                previous_result=dict(commit=revision, path=str(RESULT.relative_to(P)),
                    sha256=hashlib.sha256(prior_bytes).hexdigest(), source_commit=prior['source_commit'],
                    mode=prior['mode'], passed=prior['passed'], complete=prior['complete'],
                    timing_status=prior.get('timing_status'),
                    advance_to_model=prior.get('resident_summary', {}).get('advance_to_model')),
                retention='Keep original individual samples in this result; retire successful API traces, routine logs and JIT caches after comparison closure and verified extraction. Preserve known-good fallback; do not retain another closed component binary.')
            (ROOT / 'plan.json').write_text(json.dumps(record, indent=2) + '\n')
            RESULT.write_text(json.dumps(record, indent=2) + '\n')
            print('Prepared fresh current-fixture qualification. Commit ' + str(RESULT.relative_to(P)) + ' before ownership.')
            return 0
        else:
            record = json.loads(RESULT.read_text())
            previous = {'ownership': 'plan', 'smoke': 'ownership', 'validate': 'smoke', 'perf': 'validate'}[mode]
            guards.require((mode == 'ownership' or (record['passed'] and record['complete'])) and
                record['ownership_closed'] and not record['active'] and record['mode'] == previous and
                record['source_commit'] == build['source_commit'] and record['fixture'] == fixture,
                'require matching successful preceding phase')
            committed = git('show', 'HEAD:' + str(RESULT.relative_to(P)))
            guards.require(committed == RESULT.read_bytes(), 'commit preceding result first')
            plan = json.loads((ROOT / 'plan.json').read_text())
            guards.require(record['run_root'] == str(ROOT) and
                all(record[key] == plan[key] for key in ('source_commit', 'fixture', 'controller_source_commit',
                    'controller_sha256', 'build_record_sha256', 'previous_result',
                    'prospective_gates', 'limits', 'retention', 'consumer')) and
                controller_inputs(record['controller_source_commit']) == record['controller_sha256'] and
                digest(BUILD) == record['build_record_sha256'], 'frozen component plan/input identity changed')
        guards.require(not (ROOT / ('owned-' + mode)).exists(), 'phase already started; inspect owner')
        settings = json.loads(SETTINGS.read_text())
        for path, info in settings['runtime_elfs'].items():
            guards.require(digest(path) == info['sha256'], 'runtime changed: ' + path)
        env = dict(settings['environment'])
        env.update(STRATA_SYCL_PREFILL_NATIVE16='bf16',
                   STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',
                   STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
                   STRATA_SYCL_PREFILL_DOWN_SCATTER='large')
        if mode != 'smoke':
            env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
            env.pop('UR_LOG_TRACING', None)
        if mode in ('ownership', 'perf'):
            for key in list(env):
                if key.startswith(('UR_LOG_', 'IGC_')) or key in ('UR_ENABLE_LAYERS', 'STRATA_TRACE'):
                    del env[key]
        for name in ('tmp', 'cache'):
            (ROOT / name).mkdir(exist_ok=True)
        env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = next(line.split(': ', 1)[1] for line in journal.splitlines() if line.startswith('-- cursor: '))
        jobs = [['--bench-public', str(p), str(a)] for a,p in PERF_ORDER] if mode == 'perf' else [['--' + mode]]
        if mode in ('smoke', 'validate'):
            jobs += [['--route-' + mode, str(flag)] for flag in (0, 1)]
        record.update(mode=mode, passed=False, complete=False, environment=env,
                      GPU_executed=mode != 'ownership')
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (ROOT / 'record.json').write_text(json.dumps(dict(record,
                owned_commands=owner.commands if owner else []), indent=2) + '\n')

        owner = supervision.Owner(ROOT / ('owned-' + mode), persist)
        try:
            for ordinal, args in enumerate(jobs):
                command = [fixture['path'], *args]
                child_env = dict(env)
                if args[0].startswith('--route-'):
                    child_env['STRATA_SYCL_PREFILL_DOWN_SCATTER'] = 'large' if args[1] == '1' else '0'
                if args[0] == '--bench-public':
                    child_env['STRATA_SYCL_PREFILL_DOWN_SCATTER'] = 'large' if args[2] == '1' else '0'
                entry, out, err = owner.run(str(ordinal), command, child_env, P, wall=1200 if mode == 'perf' else 600, cpu=2400 if mode == 'perf' else 1800,
                    rss_cap=6 << 30, file_cap=64 << 20, text_cap=64 << 20, total_cap=128 << 20)
                result = dict(mode=mode, command=command, exit_code=entry.get('exit_code'),
                    startup_down_selector=child_env['STRATA_SYCL_PREFILL_DOWN_SCATTER'],
                    native_identity={k: entry.get('direct_identity', {}).get(k) for k in ('pid','start_ticks')},
                    ownership_closed=supervision.closed(entry), events=[], passed=False,
                    closure=dict(observation_complete=entry.get('observation_complete'),
                        direct_child_reaped=entry.get('direct_child_reaped'), session_empty=entry.get('session_empty'),
                        owner_error=bool(entry.get('error') or entry.get('errors')),
                        cleanup_attempted=bool(entry.get('cleanup'))))
                record['checks'].append(result)
                persist()  # Retain closure even when stdout is truncated or parsing fails.
                stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
                events = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
                result['events'] = events
                guards.require(result['ownership_closed'] and result['exit_code'] == 0 and
                    'fixture_status=PASS' in stdout and not re.search(
                        r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|CHECKED_DRAIN_FAILED', stderr),
                    'fixture failed: ' + (stderr or stdout)[-4096:])
                routed = args[0].startswith('--route-')
                expected = (6 if mode == 'smoke' else 16) if routed else {'ownership': 54, 'smoke': 33, 'validate': 78, 'perf': 112}[mode]
                guards.require(len(events) == expected and all(e['status'] == 'PASS' for e in events),
                               'incomplete fixture results')
                if mode == 'ownership':
                    guards.require(sum(e['mapping'] == 2 for e in events) == 18 and all(
                        not e['metadata_missing'] and not e['metadata_multiple'] and not e['metadata_outside_tile'] and
                        (e['mapping'] != 2 or e['metadata_slots'] > 0) for e in events), 'row cache ownership incomplete')
                if routed:
                    flag = int(args[1])
                    guards.require(all(e['kind'] == 'selected_route' and e['enabled'] == flag and
                        e['selected'] == bool(flag and e['rows'] >= 2334) and e['false_route_no_write']
                        for e in events), 'selected dispatch/no-write evidence incomplete')
                    if mode == 'smoke':
                        guards.require([e['expected_mask'] for e in events] == [0, 0, 0, 1, 2, 3],
                                       'selected threshold or invalid-map masks incomplete')
                    else:
                        guards.require({(e['rows'], bool(e['source_first'])) for e in events} ==
                            {(r, irregular) for r in (1, 17, 160, 2333, 2334, 2335, 8191, 8192)
                             for irregular in (False, True)} and
                            all(e['window_rows'] == 2 * e['rows'] + 17 and e['expected_mask'] == 0
                                for e in events), 'selected tail/placement coverage incomplete')
                if mode == 'smoke' and not routed:
                    guards.require(sum(e['kind'] == 'exception_unwind' and e['queue_complete'] for e in events) == 3,
                                   'borrowed-queue exception retirement incomplete')
                    guards.require(sum(e['kind'] == 'admissions' and e['rejected'] == 16 for e in events) == 3 and
                        sum(e['kind'] == 'invalid_map' and e['case'] == 4 and e['expected_mask'] == 1 for e in events) == 3 and
                        sum(e['kind'] == 'invalid_map' and e['case'] == 6 and e['expected_mask'] == 3 for e in events) == 3,
                        'admission or mixed-invalid status checks incomplete')
                guards.require(all(e['inputs_immutable'] for e in events if e['kind'] in
                                   ('correctness','invalid_map','sample','selected_route')), 'input immutability not checked')
                result['passed'] = True; persist()
            if mode == 'perf':
                samples = public_timing_samples(record['checks'])
                summary = []
                for rows in (64, 160, 2333, 2334, 2335, 5963, 8192):
                    for irregular in (0, 1):
                        arms = []
                        for arm in (0, 1):
                            group = [e for e in samples if (e['rows'], e['irregular'], e['arm']) == (rows, irregular, arm)]
                            values = [e['queue_interval_ms'] for e in group]
                            guards.require(len(values) == 18 and all(v > 0 for v in values) and all(e['window_rows'] == 78451 and e['accumulator_bytes'] == 78451 * 2560 * 4 for e in group), 'missing samples or wrong working set')
                            arms.append(dict(arm=arm, sample_count=len(values), median_ms=statistics.median(values),
                                sample_sd_ms=statistics.stdev(values), process_medians_ms=[statistics.median(
                                    e['queue_interval_ms'] for e in group if e['process'] == process) for process in range(3)]))
                        summary.append(dict(rows=rows, irregular=irregular, arms=arms,
                            ratio=arms[1]['median_ms'] / arms[0]['median_ms']))
                ratio = statistics.geometric_mean(s['ratio'] for s in summary)
                record['resident_summary'] = dict(shapes=summary, geometric_mean_latency_ratio=ratio,
                    advance_to_model=ratio <= .97 and all(s['ratio'] <= 1.05 for s in summary) and all(
                        a < b for s in summary if s['rows'] >= 2334 for a, b in zip(
                            s['arms'][1]['process_medians_ms'], s['arms'][0]['process_medians_ms'])),
                    prospective_gate='New selective/separate geometric mean ratio<=0.97 across14 fixed78451-window shapes; no shape median>1.05; each of3 process medians improves at2334/2335/5963/8192 in both token placements. This only permits a later model experiment, not adoption.')
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            postchecks(record, boot, cursor)
            persist(); RESULT.write_text(json.dumps(compact_result(record), indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('mode', 'complete', 'passed', 'ownership_closed', 'error', 'resident_summary')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
