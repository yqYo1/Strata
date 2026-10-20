#!/usr/bin/env python3
"""Main-owned capacity and bounded two-bank gather qualification; no model claim."""
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
A = P / 'bench/results/2026-10-15-pr6-recovery-audit'
ROOT = B / 'moe-mixed-scratch-component-v1'
RESULT = A / 'moe-mixed-scratch-current.json'
TARGET = 'sycl_moe_accum_test'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def scratch_host():
    """Only host admission/geometry; not the actual cache, Prefill or GPU route."""
    root = B / 'prefill-cache-loan-host-v1'
    result = A / 'prefill-cache-loan-host-current.json'
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.require(not root.exists() and not result.exists(), 'closed host root is not retryable')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
        pins = {}
        for name in ('include/strata/prefill/scratch_allocation.hpp',
                     'include/strata/core/cache_scratch_lease.hpp', 'tests/core/prefill_scratch_test.cpp',
                     'tools/sycl/check_mixed_scratch.py', 'tools/sycl/measurement_guards.py',
                     'tools/sycl/measurement_owner.py'):
            raw = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=P)
            guards.require(raw == (P / name).read_bytes(), 'commit exact host inputs first: ' + name)
            pins[name] = hashlib.sha256(raw).hexdigest()
        compiler = '/usr/bin/g++'
        guards.require(digest(compiler) == '1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769',
                       'host compiler changed; commit a new plan')
        root.mkdir(mode=0o700)
        (root / 'tmp').mkdir()
        env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8', TMPDIR=str(root / 'tmp'))
        binary = root / 'prefill_scratch_host_test'
        steps = [('compile', [compiler, '-O2', '-std=c++20', '-Wall', '-Wextra', '-Werror',
                  '-I' + str(P / 'include'), str(P / 'tests/core/prefill_scratch_test.cpp'), '-o', str(binary)]),
                 ('host', [str(binary)])]
        record = dict(source_commit=revision, source_sha256=pins, environment=env,
            compiler=dict(path=compiler, sha256=digest(compiler)), commands=[c for _, c in steps],
            expected_cases=37, complete=False, passed=False, active=False, GPU_executed=False,
            budgets=dict(wall_per_command=30, cpu_seconds=60, rss_bytes=1 << 30,
                         raw_text_bytes=1 << 16, total_bytes=128 << 20),
            consumer='Current cache-loan protocol host gate; retire compiled fixture/raw output after compact verification.',
            limits='Host state, interval and layout arithmetic only. No actual ExpertCache ticket, GPU, generator/verifier, model speed, quality or physical262144/RESTORE qualification.')
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            record['owned_commands'] = owner.commands if owner else []
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = supervision.Owner(root / 'owned', persist)
        persist()
        try:
            for label, command in steps:
                entry, out, err = owner.run(label, command, env, P, wall=30, cpu=60,
                    rss_cap=1 << 30, file_cap=64 << 20, text_cap=1 << 16, total_cap=128 << 20)
                guards.require(supervision.closed(entry) and entry.get('exit_code') == 0,
                    label + ': ' + (err.read_text(errors='replace') + out.read_text(errors='replace'))[-4000:])
                if label == 'host':
                    lines = out.read_text().splitlines()
                    cases = [x for x in lines if x.startswith('prefill_scratch case=')]
                    guards.require(len(cases) == 37 and len(set(cases)) == 37 and
                        all(x.endswith(' status=PASS') for x in cases) and
                        lines[-1] == 'prefill_scratch_fixture cases=37 status=PASS', 'incomplete host checks')
                    record['checks'] = cases
            guards.require(all(digest(P / n) == h for n, h in pins.items()), 'host input changed')
            record.update(complete=True, passed=True,
                          fixture=dict(path=str(binary), sha256=digest(binary), bytes=binary.stat().st_size))
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            record['passed'] = bool(record['passed'] and record['ownership_closed'])
            persist()
            result.write_text(json.dumps({k: v for k, v in record.items() if k != 'owned_commands'}, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('complete', 'passed', 'ownership_closed', 'error')}))
        return 0 if record['passed'] else 1


def main():
    guards.require(sys.argv[1:] in (['plan'], ['smoke'], ['validate'], ['scratch-host']),
                   'use check_mixed_scratch.py plan|smoke|validate|scratch-host')
    mode = sys.argv[1]
    if mode == 'scratch-host':
        return scratch_host()
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
        pins = {}
        for name in ('include/strata/prefill/moe_accum_plan.hpp', 'tests/moe_accum_plan.cpp',
                     'tools/sycl/check_mixed_scratch.py'):
            raw = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=P)
            guards.require(raw == (P / name).read_bytes(), 'commit exact capacity test inputs first')
            pins[name] = hashlib.sha256(raw).hexdigest()
        gpu = mode != 'plan'
        if not gpu:
            guards.require(not ROOT.exists() and not RESULT.exists(), 'inspect/retire prior component first')
            ROOT.mkdir(mode=0o700)
            for name in ('tmp', 'cache'):
                (ROOT / name).mkdir()
            env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8')
            compiler = '/usr/bin/c++'
            version = subprocess.check_output([compiler, '--version'], env=env, text=True).splitlines()[0]
            binary = ROOT / 'plan-test'
            steps = [('compile-plan', [compiler, '-O2', '-std=c++20', '-Wall', '-Wextra', '-Werror',
                       '-I' + str(P / 'include'), str(P / 'tests/moe_accum_plan.cpp'), '-o', str(binary)]),
                     ('plan', [str(binary)])]
            record = dict(mode=mode, source_commit=revision, capacity_source_sha256=pins,
                compiler=dict(path=compiler, version=version, sha256=digest(compiler)), checks=[],
                limits='Capacity arithmetic and bit-copy gathers only. No helper lifetime, model speed, quality or physical262144/RESTORE qualification.')
        else:
            guards.require(RESULT.read_bytes() == subprocess.check_output(
                ['git', 'show', 'HEAD:' + str(RESULT.relative_to(P))], cwd=P), 'commit preceding gate first')
            record = json.loads(RESULT.read_text())
            guards.require(record['mode'] == ('plan' if mode == 'smoke' else 'smoke') and
                record['passed'] and record['ownership_closed'] and record['capacity_source_sha256'] == pins,
                'matching passed capacity/smoke gate required')
            build = json.loads((B / 'main-native-gdn-engine/record.json').read_text())
            guards.require(build['passed'] and build['complete'] and build['ownership_closed'] and not build['active'],
                           'closed current build required')
            for group in [build['project'], *build['dependencies']]:
                for name, expected in group['source_sha256'].items():
                    guards.require(digest(Path(group['root']) / name) == expected, 'engine input changed: ' + name)
            fixture = build['fixtures'][TARGET]
            guards.require(digest(fixture['path']) == fixture['sha256'], 'fixture changed')
            settings = json.loads((P / 'tools/sycl/measurement_environment.json').read_text())
            for name, identity in settings['runtime_elfs'].items():
                guards.require(digest(name) == identity['sha256'], 'runtime changed: ' + name)
            env = dict(settings['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
            env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                       UR_LOG_TRACING='level:info;flush:info;output:stderr')
            cases = ['smoke'] if mode == 'smoke' else ['contiguous', '65537', 'partial', 'full']
            steps = [(case, [fixture['path'], '--case', 'mixed-gather-' + case]) for case in cases]
            record.update(mode=mode, build_source_commit=build['source_commit'], fixture=fixture)
        record.update(complete=False, passed=False, active=False, GPU_executed=gpu, environment=env)
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        cursor = None
        if gpu:
            guards.driver_guard()
            journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
            cursor = next(x.split(': ', 1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            if owner:
                record['owned_commands'] = owner.commands
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = supervision.Owner(ROOT / ('owned-' + mode), persist)
        persist()
        try:
            for label, command in steps:
                entry, out, err = owner.run(label, command, env, P, wall=300, cpu=1200,
                    rss_cap=12 << 30, file_cap=128 << 20, text_cap=64 << 20, total_cap=1 << 30)
                guards.require(supervision.closed(entry) and entry.get('exit_code') == 0,
                    label + ': ' + (err.read_text(errors='replace') + out.read_text(errors='replace'))[-6000:])
                text = out.read_text()
                if gpu:
                    lines = [x for x in text.splitlines() if x.startswith('mixed_gather ')]
                    guards.require(len(lines) == 1 and 'admission_rejects=6 invalid_rows=4 status=4 inputs_immutable=1 guards=PASS PASS' in lines[0]
                        and 'fixture_status=PASS model_PP=UNRUN' in text, 'incomplete gather checks')
                    record['checks'].append(dict(case=label, result=lines[0], passed=True))
                elif label == 'plan':
                    guards.require('negative_checks=34 PASS' in text and
                        'borrowed_cases=6 borrowed_negative_checks=2 PASS' in text and 'moe_accum_plan PASS' in text,
                        'incomplete capacity/negative cases')
                    record['checks'].append(dict(case='plan', original_negative_cases=34,
                        new_borrowed_cases=6, new_borrowed_negative_cases=2, passed=True))
            if not gpu:
                record['plan_binary'] = dict(path=str(binary), bytes=binary.stat().st_size, sha256=digest(binary))
            guards.require(all(digest(P / n) == h for n, h in pins.items()), 'capacity/controller input changed')
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            record['fresh_faults'] = []
            if gpu:
                after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
                record['fresh_faults'] = [x for x in after.splitlines() if re.search(
                    r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]
                try:
                    record['driver_after'] = guards.driver_guard()
                except Exception as error:
                    record['driver_error'] = str(error)
                    record['passed'] = False
            record['passed'] = bool(record['passed'] and record['ownership_closed'] and
                                   record['boot_unchanged'] and not record['fresh_faults'])
            persist()
            RESULT.write_text(json.dumps({k: v for k, v in record.items() if k != 'owned_commands'}, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('mode', 'complete', 'passed', 'ownership_closed', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
