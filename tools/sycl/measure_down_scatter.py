#!/usr/bin/env python3
"""Main-owned Down/accumulation fusion gates; no engine dispatch adoption."""
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
ROOT = B / 'moe-down-scatter-component-v1'
BUILD = B / 'main-native-gdn-engine/record.json'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/moe-down-scatter-current.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'
TARGET = 'sycl_moe_down_scatter_bench'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    guards.require(len(sys.argv) == 2 and sys.argv[1] in ('ownership', 'smoke', 'validate', 'perf'),
                   'use measure_down_scatter.py ownership|smoke|validate|perf')
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
        if mode == 'ownership':
            guards.require(not ROOT.exists(), 'inspect existing job; do not restart')
            ROOT.mkdir(mode=0o700)
            record = dict(source_commit=build['source_commit'], fixture=fixture, checks=[],
                complete=False, passed=False, active=False, model_performance_adopted=False,
                model_quality_qualified=False, physical262144_model_RESTORE_qualified=False,
                consumer='One Down/scatter component comparison; retire raw logs/JIT after extracting samples.',
                prospective_gates='18 exhaustive host tile-ownership cases; traced small7-event correctness/admission/map gate;22 tail/split/irregular/rounding checks;3 independent resident processes with6 rotated rounds per shape/arm.',
                limits='FP16 Down2560x640, BF16-round then explicit FP32 FMA. Component only; existing engine route unchanged. Marker intervals include host submission gaps. No physical VRAM/PCIe traffic, occupancy, saturation or model speed is inferred.')
        else:
            record = json.loads(RESULT.read_text())
            previous = {'smoke': 'ownership', 'validate': 'smoke', 'perf': 'validate'}[mode]
            guards.require(record['passed'] and record['ownership_closed'] and record['mode'] == previous and
                record['source_commit'] == build['source_commit'] and record['fixture'] == fixture,
                'require matching successful preceding phase')
            committed = subprocess.check_output(['git', 'show', 'HEAD:' + str(RESULT.relative_to(P))], cwd=P)
            guards.require(committed == RESULT.read_bytes(), 'commit preceding result first')
        guards.require(not (ROOT / ('owned-' + mode)).exists(), 'phase already started; inspect owner')
        settings = json.loads(SETTINGS.read_text())
        for path, info in settings['runtime_elfs'].items():
            guards.require(digest(path) == info['sha256'], 'runtime changed: ' + path)
        env = dict(settings['environment'])
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
        jobs = [['--bench', str(i)] for i in range(3)] if mode == 'perf' else [['--' + mode]]
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
                entry, out, err = owner.run(str(ordinal), command, env, P, wall=360, cpu=1200,
                    rss_cap=6 << 30, file_cap=64 << 20, text_cap=64 << 20, total_cap=128 << 20)
                stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
                events = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
                result = dict(mode=mode, command=command, exit_code=entry.get('exit_code'),
                    ownership_closed=supervision.closed(entry), events=events, passed=False)
                record['checks'].append(result)
                guards.require(result['ownership_closed'] and result['exit_code'] == 0 and
                    'fixture_status=PASS' in stdout and not re.search(
                        r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|CHECKED_DRAIN_FAILED', stderr),
                    'fixture failed: ' + (stderr or stdout)[-4096:])
                expected = {'ownership': 18, 'smoke': 7, 'validate': 22, 'perf': 140}[mode]
                guards.require(len(events) == expected and all(e['status'] == 'PASS' for e in events),
                               'incomplete fixture results')
                result['passed'] = True; persist()
            if mode == 'perf':
                samples = [e for c in record['checks'] if c['mode'] == 'perf' for e in c['events'] if e['round'] >= 0]
                summary = []
                for rows in (64, 160, 2334, 5963, 8192):
                    for irregular in (0, 1):
                        arms = []
                        for arm in (0, 1):
                            group = [e for e in samples if (e['rows'], e['irregular'], e['arm']) == (rows, irregular, arm)]
                            values = [e['queue_interval_ms'] for e in group]
                            guards.require(len(values) == 18 and all(v > 0 for v in values), 'missing samples')
                            arms.append(dict(arm=arm, sample_count=len(values), median_ms=statistics.median(values),
                                sample_sd_ms=statistics.stdev(values), process_medians_ms=[statistics.median(
                                    e['queue_interval_ms'] for e in group if e['process'] == process) for process in range(3)]))
                        summary.append(dict(rows=rows, irregular=irregular, arms=arms,
                            ratio=arms[1]['median_ms'] / arms[0]['median_ms']))
                ratio = statistics.geometric_mean(s['ratio'] for s in summary)
                record['resident_summary'] = dict(shapes=summary, geometric_mean_latency_ratio=ratio,
                    advance_to_model=ratio <= .95 and all(s['ratio'] <= 1.05 for s in summary) and all(
                        a < b for s in summary if s['rows'] >= 2334 for a, b in zip(
                            s['arms'][1]['process_medians_ms'], s['arms'][0]['process_medians_ms'])),
                    prospective_gate='Geometric mean ratio<=0.95 across10 matched shapes; no shape median>1.05; each of3 process medians improves at2334/5963/8192 in both layouts. This only permits a later model experiment, not adoption.')
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
