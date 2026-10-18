#!/usr/bin/env python3
"""Main-owned direct/prefix/row-publication checks and a rotated same-binary row-publication timing."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'qsa-attention-current'
ENGINE = B / 'main-native-gdn-engine'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/qsa-attention-current.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'
TARGETS = ('sycl_qsa_bf16_late_flags_test', 'sycl_qsa_bf16_prefix_optimized_test',
           'sycl_qsa_bf16_prefix_rows_test')
BENCH = 'sycl_qsa_bf16_prefix_rows_bench'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    guards.require(len(sys.argv) == 2 and sys.argv[1] in ('smoke', 'component', 'perf'),
                   'use check_attention.py smoke|component|perf')
    mode = sys.argv[1]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.driver_guard()
        build = json.loads((ENGINE / 'record.json').read_text())
        guards.require(build['passed'] and build['complete'] and build['ownership_closed'] and
                       not build['active'], 'closed qualified build required')
        for group in [build['project'], *build['dependencies']]:
            for name, expected in group['source_sha256'].items():
                guards.require(digest(Path(group['root']) / name) == expected, 'build source changed: ' + name)
        fixtures = {name: build['fixtures'][name] for name in (*TARGETS, BENCH)}
        for item in fixtures.values():
            guards.require(digest(item['path']) == item['sha256'], 'fixture binary changed')
        if mode == 'smoke':
            guards.require(not ROOT.exists(), 'close/retire the previous check before replacing it')
            ROOT.mkdir(mode=0o700)
            record = dict(source_commit=build['source_commit'], fixtures=fixtures, checks=[],
                          passed=False, complete=False, active=False, model_performance_measured=False,
                          model_quality_qualified=False, physical262144_model_RESTORE_qualified=False,
                          consumer='Immediate prefix33 versus fused row publication same-binary component comparison; retire raw checks afterward.')
        else:
            record = json.loads(RESULT.read_text())
            guards.require(record['passed'] and record['ownership_closed'] and record['source_commit'] ==
                           build['source_commit'] and record['fixtures'] == fixtures, 'matching smoke required')
            guards.require(record['mode'] == ('component' if mode == 'perf' else 'smoke'),
                           'require smoke then component then perf on this exact build')
        guards.require(not (ROOT / ('owned-' + mode)).exists(), 'this check already started; inspect its receipt')
        settings = json.loads(SETTINGS.read_text())
        for path, item in settings['runtime_elfs'].items():
            guards.require(digest(path) == item['sha256'], 'runtime asset changed: ' + path)
        env = dict(settings['environment'])
        if mode == 'component':
            env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
            env.pop('UR_LOG_TRACING', None)
        if mode == 'perf':
            for key in list(env):
                if key.startswith('UR_LOG_') or key in ('UR_ENABLE_LAYERS', 'STRATA_TRACE'):
                    del env[key]
        for name in ('tmp', 'cache'):
            (ROOT / name).mkdir(exist_ok=True)
        env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = next(x.split(': ', 1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        record.update(passed=False, complete=False, mode=mode, environment=env)
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = supervision.Owner(ROOT / ('owned-' + mode), persist)
        try:
            targets = (BENCH,) if mode == 'perf' else (*TARGETS, BENCH) if mode == 'smoke' else TARGETS
            for name in targets:
                command = [fixtures[name]['path'], '--component' if mode == 'perf' else '--' + mode]
                entry, out, err = owner.run(name, command, env, P, wall=180, cpu=360,
                    rss_cap=4 << 30, file_cap=64 << 20, text_cap=64 << 20, total_cap=64 << 20)
                stdout = out.read_text(errors='replace')
                stderr = err.read_text(errors='replace')
                passed = (supervision.closed(entry) and entry.get('exit_code') == 0 and
                          'fixture_status=PASS' in stdout and not re.search(
                              r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed', stderr))
                record['checks'].append(dict(mode=mode, target=name, command=command,
                    binary_sha256=fixtures[name]['sha256'], passed=passed, exit_code=entry.get('exit_code'),
                    ownership_closed=supervision.closed(entry),
                    cases=[x for x in stdout.splitlines() if x.startswith(('case ', 'fixture_status='))],
                    samples=[dict(token.split('=', 1) for token in x.split()[1:])
                             for x in stdout.splitlines() if x.startswith('sample ')]))
                guards.require(passed, name + ': ' + (stderr or stdout)[-8192:])
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
            record['fresh_faults'] = [x for x in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]
            record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            try:
                record['driver_after'] = guards.driver_guard()
            except Exception as error:
                record['driver_guard_error'] = str(error)
                record['passed'] = False
            record['passed'] = bool(record['passed'] and record['ownership_closed'] and
                                    record['boot_unchanged'] and not record['fresh_faults'])
            persist()
            RESULT.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('mode', 'passed', 'complete', 'ownership_closed', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
