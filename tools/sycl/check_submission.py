#!/usr/bin/env python3
"""Main-owned immediate versus UR V2 batched submission acceptance; no speed claim."""
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
ROOT = B / 'submission-acceptance-v1'
ENGINE = B / 'main-native-gdn-engine'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/submission-acceptance-current.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    guards.require(sys.argv[1:] in (['control'], ['candidate']), 'use check_submission.py control|candidate')
    mode = sys.argv[1]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.driver_guard()
        build = json.loads((ENGINE / 'record.json').read_text())
        guards.require(build['passed'] and build['complete'] and build['ownership_closed'] and
                       not build['active'], 'closed build required')
        for group in [build['project'], *build['dependencies']]:
            for name, expected in group['source_sha256'].items():
                guards.require(digest(Path(group['root']) / name) == expected, 'build source changed: ' + name)
        fixture = build['fixtures']['sycl_submission_test']
        guards.require(digest(fixture['path']) == fixture['sha256'], 'fixture changed')
        if mode == 'control':
            guards.require(not ROOT.exists(), 'inspect/retire the prior owned comparison before reuse')
            ROOT.mkdir(mode=0o700)
            record = dict(source_commit=build['source_commit'], fixture=fixture, checks=[],
                          complete=False, passed=False, active=False, model_performance_measured=False,
                          model_quality_qualified=False, advance_to_model=False,
                          consumer='Bounded runtime compatibility gate before any forced-batched model run; retire normal full traces after recording outcomes.',
                          prospective_gate='Both separate processes pass10 numerical/guard/event/profiling/lifetime/teardown cases without faults. Any failed contract rejects advancing this runtime mode.',
                          limits='Startup flag plus pinned source/adapter identity; no hardware saturation or performance claim. Graph replay is excluded.')
        else:
            guards.require(RESULT.read_bytes() == subprocess.check_output(
                ['git', 'show', 'HEAD:' + str(RESULT.relative_to(P))], cwd=P), 'commit preceding control result')
            record = json.loads(RESULT.read_text())
            guards.require(record['passed'] and record['ownership_closed'] and record['mode'] == 'control' and
                           record['source_commit'] == build['source_commit'] and record['fixture'] == fixture,
                           'matching passed control required')
        settings = json.loads(SETTINGS.read_text())
        for path, item in settings['runtime_elfs'].items():
            guards.require(digest(path) == item['sha256'], 'runtime asset changed: ' + path)
        env = dict(settings['environment'])
        guards.require(env.get('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD') == '1', 'preserve copy-offload avoidance')
        if mode == 'candidate':
            env['UR_L0_V2_FORCE_BATCHED'] = '1'
        else:
            env.pop('UR_L0_V2_FORCE_BATCHED', None)
        for name in ('tmp', 'cache'):
            (ROOT / name).mkdir(exist_ok=True)
        env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = next(x.split(': ', 1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        record.update(mode=mode, complete=False, passed=False, ownership_closed=False, environment=env)
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = supervision.Owner(ROOT / ('owned-' + mode), persist)
        try:
            entry, out, err = owner.run('submission', [fixture['path'], '--submission'], env, P,
                wall=90, cpu=120, rss_cap=2 << 30, file_cap=64 << 20, text_cap=64 << 20, total_cap=64 << 20)
            stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
            cases = [x for x in stdout.splitlines() if x.startswith('case ')]
            passed = (supervision.closed(entry) and entry.get('exit_code') == 0 and len(cases) == 10 and
                      'fixture_status=PASS mode=submission' in stdout and not re.search(
                          r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed', stderr))
            record['checks'].append(dict(mode=mode, command=[fixture['path'], '--submission'],
                environment_selector=env.get('UR_L0_V2_FORCE_BATCHED', 'unset'), passed=passed,
                exit_code=entry.get('exit_code'), ownership_closed=supervision.closed(entry), cases=cases,
                stdout_sha256=digest(out), stderr_sha256=digest(err)))
            if not passed:
                selected = [x for x in stderr.splitlines() if re.search(
                    r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|profiling|cudaEventElapsedTime|terminate|exception|compat:', x)]
                record['error_excerpt'] = '\n'.join(selected[-40:])[-8192:]
            guards.require(passed, 'submission acceptance failed; inspect preserved compact error excerpt')
            record.update(complete=True, passed=True, advance_to_model=mode == 'candidate')
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
            record['advance_to_model'] = bool(record['advance_to_model'] and record['passed'])
            persist()
            RESULT.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('mode', 'passed', 'complete', 'ownership_closed', 'advance_to_model', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
