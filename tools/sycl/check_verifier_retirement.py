#!/usr/bin/env python3
"""Main-owned small production-verifier copy/retirement gate; no model or speed claim."""
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
ENGINE = B / 'main-native-gdn-engine'
ROOT = B / 'verifier-copy-safety-v2'
A = P / 'bench/results/2026-10-15-pr6-recovery-audit'
SETTINGS = P / 'tools/sycl/measurement_environment.json'
RESULT = A / 'verifier-copy-safety-current.json'
# Separately admitted diagnostic assets, unchanged from the closed CPU cohort.
# This does not refresh the qualified shared runtime or permit long model work.
DELTA = {
    '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395': (
        'dfa844e5afd8e8f74bf5ce9da3ea2837706d9116bc8efeaed90efba5acaf7b1c',
        '50adb6b727d027309039803033d41efeae4b248589ee363f321c0d538c7de3ff'),
    '/usr/lib/x86_64-linux-gnu/libudev.so.1.7.8': (
        '4298228175fa62a36af88b1afd4406cdd8b1bf166621f957d0dbbde45a698fe5',
        '0d6934adafb2ef3c640382440c14d6bd60f391e688c000796ffcea0c72849269'),
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    guards.require(len(sys.argv) == 1, 'no options; inspect a closed run before any retry')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.driver_guard()
        build_path = A / 'native-gdn-engine-build.json'
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + str(build_path.relative_to(P))], cwd=P)
        guards.require(committed == build_path.read_bytes(), 'commit the exact build result first')
        compact = json.loads(committed)
        build = json.loads((ENGINE / 'record.json').read_text())
        guards.require(build['passed'] and build['complete'] and build['ownership_closed'] and
            not build['active'] and all(build[k] == compact[k] for k in
            ('source_commit', 'binary', 'fixtures', 'targets', 'configure_argv', 'compiler')),
            'closed matching committed build required')
        sources = [build['project'], *build['dependencies']]

        def unchanged():
            for source in sources:
                for name, expected in source['source_sha256'].items():
                    guards.require(digest(Path(source['root']) / name) == expected, 'build input changed: ' + name)
            for item in [build['binary'], *build['fixtures'].values()]:
                guards.require(digest(item['path']) == item['sha256'], 'build output changed')

        unchanged()
        guards.require('tools/sycl/check_verifier_retirement.py' in build['project']['source_sha256'],
            'the build must pin this controller')
        settings = json.loads(SETTINGS.read_text())
        guards.require(set(DELTA) <= set(settings['runtime_elfs']), 'unknown diagnostic runtime delta')

        def runtime_unchanged():
            for name, item in settings['runtime_elfs'].items():
                expected = item['sha256']
                if name in DELTA:
                    old, expected = DELTA[name]
                    guards.require(old == item['sha256'], 'original runtime baseline differs')
                guards.require(digest(name) == expected, 'runtime asset changed: ' + name)

        runtime_unchanged()
        guards.require(not ROOT.exists(), 'inspect/close the existing owned check; no automatic retry')
        ROOT.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (ROOT / name).mkdir()
        env = dict(settings['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = next(x.split(': ', 1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        record = dict(source_commit=build['source_commit'], binary=build['binary'], fixtures=build['fixtures'],
            environment_reference=str(SETTINGS.relative_to(P)), environment_sha256=digest(SETTINGS),
            runtime_asset_delta={n: dict(qualified_sha256=v[0], candidate_sha256=v[1]) for n,v in DELTA.items()},
            checks=[], complete=False, passed=False, active=False, model_performance_measured=False,
            model_quality_qualified=False, nonresident_gpu_route_enabled=False,
            consumer='Prerequisite for an ordinary-Arena explicit-copy prototype. Retire normal traces after compact results are verified and committed.',
            gate='Same-build submission10 cases and verifier7 isolated cases (including empty-batch ordering and failed-window reuse/output refusal), checked owners and unchanged boot/runtime/source, no fresh GPU fault. Any failure blocks advancement.',
            limits='Synthetic bounded copies and intentional host-only async failures. No invalid GPU address or stalled kernel. No model, physical262144/session/RESTORE or throughput qualification.')
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (ROOT / 'record.json').write_text(json.dumps(dict(record, commands=owner.commands if owner else []), indent=2) + '\n')

        owner = supervision.Owner(ROOT / 'owned', persist)
        persist()
        try:
            for name, args in [('sycl_submission_test', ['--submission']), ('sycl_verifier_retirement_test', [])]:
                fixture = build['fixtures'][name]
                command = [fixture['path'], *args]
                entry, out, err = owner.run(name, command, env, P, wall=180, cpu=600,
                    rss_cap=4 << 30, file_cap=64 << 20, text_cap=64 << 20, total_cap=128 << 20)
                stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
                guards.require(supervision.closed(entry) and entry.get('exit_code') == 0,
                    'fixture failed: ' + (stdout + stderr)[-8192:])
                if name == 'sycl_submission_test':
                    cases = [x for x in stdout.splitlines() if x.startswith('case ')]
                    guards.require(len(cases) == 10 and 'fixture_status=PASS mode=submission' in stdout and
                        not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed', stderr),
                        'submission contracts failed: ' + (stdout + stderr)[-8192:])
                else:
                    cases = [x for x in stdout.splitlines() if x.startswith('case=')]
                    expected = [('success',0,'S'), ('empty',0,'Z'), ('partial',0,'P'), ('bounds',0,'B'),
                        ('query',0,'Q'), ('destructor',1,'D'), ('copy-error',1,'E')]
                    guards.require(cases == [f'case={n} expected_exit={e} markers={m} PASS' for n,e,m in expected]
                        and 'fixture_status=PASS cases=7' in stdout, 'verifier contracts failed: ' + (stdout + stderr)[-8192:])
                record['checks'].append(dict(command=command, cases=cases, passed=True,
                    stdout_sha256=digest(out), stderr_sha256=digest(err)))
            unchanged(); runtime_unchanged()
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
            record['fresh_faults'] = [x for x in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]
            record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            try: record['driver_after'] = guards.driver_guard()
            except Exception as error:
                record.update(driver_guard_error=str(error), passed=False)
            record['passed'] = bool(record['passed'] and record['ownership_closed'] and
                record['boot_unchanged'] and not record['fresh_faults'])
            persist(); RESULT.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('complete', 'passed', 'ownership_closed', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
