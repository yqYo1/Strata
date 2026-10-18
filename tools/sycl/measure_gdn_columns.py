#!/usr/bin/env python3
"""Main-owned exact-FP32 GDN scheduling qualification and resident comparison."""
import fcntl
import hashlib
import json
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
ROOT = B / 'gdn-columns-component-v1'
BUILD = B / 'main-native-gdn-engine/record.json'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/gdn-columns-current.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'
TARGET = 'sycl_gdn_columns_bench'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    guards.require(len(sys.argv) == 2 and sys.argv[1] in ('smoke', 'validate', 'perf', 'context'),
                   'use measure_gdn_columns.py smoke|validate|perf|context')
    mode = sys.argv[1]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.driver_guard()
        build = json.loads(BUILD.read_text())
        guards.require(build['complete'] and build['passed'] and build['ownership_closed'] and
                       not build['active'], 'closed successful build required')
        for source in [build['project'], *build['dependencies']]:
            for name, expected in source['source_sha256'].items():
                guards.require(digest(Path(source['root']) / name) == expected, 'build input changed: ' + name)
        fixture = build['fixtures'][TARGET]
        guards.require(digest(fixture['path']) == fixture['sha256'], 'fixture changed')
        if mode == 'smoke':
            guards.require(not ROOT.exists(), 'inspect the existing job; do not restart')
            ROOT.mkdir(mode=0o700)
            record = dict(source_commit=build['source_commit'], fixture=fixture, checks=[],
                complete=False, passed=False, active=False, model_performance_adopted=False,
                model_quality_qualified=False, physical262144_model_RESTORE_qualified=False,
                consumer='One exact-math GDN scheduling comparison; retire full logs/JIT after extracting needed samples.',
                prospective_gates='smoke with8 exact cases/6 invalid admissions,24 full-size/tail exact cases,3 rotated clean resident processes, then262144 recurrence/state-restore/decode continuity for the measured winner.',
                limits='Resident FP32 GDN128/16/48 only, no projection/conv/PCIe. Queue markers include host launch gaps and both candidate kernels, not isolated GPU service, occupancy or a hardware ceiling. Benchmark outputs are validated after the continuous train; correctness/context modes check each compared chunk. Full model quality/context/RESTORE remain separate gates.')
        else:
            record = json.loads(RESULT.read_text())
            previous = {'validate': 'smoke', 'perf': 'validate', 'context': 'perf'}[mode]
            guards.require(record['passed'] and record['ownership_closed'] and record['mode'] == previous and
                record['source_commit'] == build['source_commit'] and record['fixture'] == fixture,
                'require matching successful preceding phase')
            committed = subprocess.check_output(['git', 'show', 'HEAD:' + str(RESULT.relative_to(P))], cwd=P)
            guards.require(committed == RESULT.read_bytes(), 'commit preceding result first')
        guards.require(not (ROOT / ('owned-' + mode)).exists(), 'phase already started; inspect its owner')
        settings = json.loads(SETTINGS.read_text())
        for path, info in settings['runtime_elfs'].items():
            guards.require(digest(path) == info['sha256'], 'runtime changed: ' + path)
        env = dict(settings['environment'])
        if mode != 'smoke':
            env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
            env.pop('UR_LOG_TRACING', None)
        if mode == 'perf':
            for key in list(env):
                if key.startswith(('UR_LOG_', 'IGC_')) or key in ('UR_ENABLE_LAYERS', 'STRATA_TRACE'):
                    del env[key]
        for name in ('tmp', 'cache'):
            (ROOT / name).mkdir(exist_ok=True)
        env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = next(line.split(': ', 1)[1] for line in journal.splitlines() if line.startswith('-- cursor: '))
        jobs = ([['--bench', str(i)] for i in range(3)] if mode == 'perf' else
                [['--context', str(record['resident_summary']['winner'])]] if mode == 'context' else
                [['--' + mode]])
        record.update(mode=mode, passed=False, complete=False, environment=env)
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (ROOT / 'record.json').write_text(json.dumps(dict(record,
                owned_commands=owner.commands if owner else []), indent=2) + '\n')

        owner = supervision.Owner(ROOT / ('owned-' + mode), persist)
        try:
            for ordinal, args in enumerate(jobs):
                command = [fixture['path'], *args]
                entry, out, err = owner.run(str(ordinal), command, env, P, wall=360, cpu=720,
                    rss_cap=6 << 30, file_cap=64 << 20, text_cap=64 << 20, total_cap=64 << 20)
                stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
                events = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
                result = dict(mode=mode, command=command, exit_code=entry.get('exit_code'),
                    ownership_closed=supervision.closed(entry), events=events, passed=False)
                record['checks'].append(result)
                guards.require(result['ownership_closed'] and result['exit_code'] == 0 and
                    'fixture_status=PASS' in stdout and not re.search(
                        r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|CHECKED_DRAIN_FAILED', stderr),
                    'fixture failed: ' + (stderr or stdout)[-4096:])
                expected = {'smoke': 9, 'validate': 24, 'perf': 35, 'context': 1}[mode]
                guards.require(len(events) == expected and all(e['status'] == 'PASS' for e in events),
                               'incomplete fixture results')
                result['passed'] = True
                persist()
            if mode == 'perf':
                samples = [e for c in record['checks'] if c['mode'] == 'perf' for e in c['events']
                           if e['kind'] == 'sample' and e['round'] >= 0]
                summary = []
                for partitions in (0, 1, 2, 4, 8):
                    values = [e['queue_interval_ms'] for e in samples if e['partitions'] == partitions]
                    guards.require(len(values) == 18 and all(v > 0 for v in values), 'missing finite positive samples')
                    summary.append(dict(partitions=partitions, samples=len(values), median_ms=statistics.median(values),
                        min_ms=min(values), max_ms=max(values), sample_sd_ms=statistics.stdev(values),
                        process_medians_ms=[statistics.median(e['queue_interval_ms'] for e in samples
                            if e['partitions'] == partitions and e['process'] == process) for process in range(3)]))
                winner = min((s for s in summary if s['partitions'] in (2, 4, 8)), key=lambda s: s['median_ms'])
                record['resident_summary'] = dict(arms=summary, winner=winner['partitions'],
                    winner_latency_change_percent=(winner['median_ms'] / summary[0]['median_ms'] - 1) * 100,
                    scope='Three independent resident processes, six rotating measured rounds plus first-use round per arm; not model throughput or a saturation ceiling.')
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
            record['fresh_faults'] = [line for line in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', line)]
            record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            try: record['driver_after'] = guards.driver_guard()
            except Exception as error:
                record['driver_error'] = str(error); record['passed'] = False
            record['passed'] = bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])
            persist(); RESULT.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('mode', 'complete', 'passed', 'ownership_closed', 'error', 'resident_summary')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
