#!/usr/bin/env python3
"""Build and host-test a diagnostic exception observer; no GPU execution."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
O = B / 'main-primary-throw-observer-root-v1'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    build = json.loads((B / 'main-primary-size-t-range-build-root-v2/record.json').read_text())
    assert build['passed'] and build['complete'] and not build['active']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    assert not O.exists()
    O.mkdir()
    env = build['environment']
    compiler = env['PATH'].split(':')[0] + '/clang++'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = dict(active=True, passed=False, complete=False, GPU_executed=False, environment=env,
        commands=[], recipe_sha256=sha(Path(__file__)),
        source_pins={name: sha(D / name) for name in ['exception_backtrace.cpp', 'exception_backtrace_selftest.cpp']},
        consumer='one bounded corrected-range real-model diagnostic to identify any remaining SYCL throw',
        budget_bytes=16 << 20, review_point='after that diagnostic or replacement by a specific reproducer')

    def persist():
        (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

    owner = module.Owner(O / 'owned', persist)

    def run(label, argv, use_env=None):
        entry, out, err = owner.run(label, argv, use_env or env, P, wall=120, cpu=120,
            text_cap=64 << 20, rss_cap=2 << 30, total_cap=16 << 20, file_cap=8 << 20)
        record['commands'].append(entry)
        assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), err.read_text()[-1500:])
        return out.read_text(), err.read_text()

    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run('build-observer', [compiler, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-fPIC',
                '-shared', str(D / 'exception_backtrace.cpp'), '-ldl', '-o', str(O / 'observer.so')])
            run('build-observer-selftest', [compiler, '-std=c++20', '-O2', '-fsycl',
                '-fsycl-id-queries-range=size_t', str(D / 'exception_backtrace_selftest.cpp'),
                '-o', str(O / 'selftest')])
            use_env = dict(env, LD_PRELOAD=str(O / 'observer.so'))
            out, err = run('observer-selftest', [str(O / 'selftest')], use_env)
            assert 'original SYCL exception caught:' in out and 'strata diagnostic SYCL throw:' in err
            assert str(O / 'selftest') in err and str(O / 'observer.so') in err, err
            record['selftest'] = dict(stdout=out, stderr=err, original_exception_catch_preserved=True)
            record['binaries'] = {name: sha(O / name) for name in ['observer.so', 'selftest']}
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = False
            persist()
    print(json.dumps(dict(passed=True, GPU_executed=False)), flush=True)


if __name__ == '__main__':
    main()
