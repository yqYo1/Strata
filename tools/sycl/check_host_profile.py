#!/usr/bin/env python3
"""Main-owned CPU-only observer gates, before building a diagnostic engine."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import bench_engine as bench
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'host-profile-cpu-v2'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/prefill-host-cpu-current.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.require(not ROOT.exists(), 'inspect the existing CPU fixture receipt; do not restart it')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
        names = ['include/strata/sycl/cpu_wait_profile.hpp', 'tests/sycl_cpu_wait_profile.cpp',
                 'tools/sycl/check_host_profile.py', 'tools/sycl/bench_engine.py',
                 'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_guards.py']
        pins = {}
        for name in names:
            data = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=P)
            guards.require(data == (P / name).read_bytes(), 'commit CPU fixture input: ' + name)
            pins[name] = hashlib.sha256(data).hexdigest()
        compiler = Path('/usr/bin/g++').resolve()
        ROOT.mkdir(mode=0o700)
        (ROOT / 'tmp').mkdir()
        executable = ROOT / 'cpu_wait_profile'
        record = dict(source_commit=revision, source_sha256=pins,
            compiler=dict(path=str(compiler), sha256=digest(compiler)),
            complete=False, passed=False, active=False, GPU_executed=False,
            retention_consumer='Immediate host accounting qualification; retire CPU executable/raw output after compact evidence.')
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (ROOT / 'record.json').write_text(json.dumps(dict(record,
                owned_commands=owner.commands if owner else []), indent=2) + '\n')

        owner = supervision.Owner(ROOT / 'owned', persist)
        persist()
        try:
            env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', TMPDIR=str(ROOT / 'tmp'))
            record['environment'] = env
            command = [str(compiler), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-pthread',
                       '-I', str(P / 'include'), str(P / names[1]), '-o', str(executable)]
            record['compile_argv'] = command
            for label, argv in [('compile', command), ('accounting', [str(executable)])]:
                entry, out, err = owner.run(label, argv, env, P, wall=30, cpu=30,
                    rss_cap=1 << 30, text_cap=1 << 20, file_cap=8 << 20, total_cap=16 << 20)
                guards.require(supervision.closed(entry) and entry.get('exit_code') == 0,
                               label + ' failed: ' + (err.read_text() + out.read_text())[-4096:])
            record['binary'] = dict(path=str(executable), sha256=digest(executable))
            text = err.read_text()
            pieces = re.split(r'(?m)^cpu fixture case: (\w+)\n', text)
            guards.require(not pieces[0] and len(pieces) == 21, 'unexpected CPU fixture case framing')
            cases = dict(zip(pieces[1::2], pieces[2::2]))
            guards.require(list(cases) == ['disabled', 'valid', 'exception', 'nested_host',
                                          'host_inside_wait', 'nested_scope', 'status_nested_scope',
                                          'status_nested_part', 'status_part_inside_wait', 'status_exception'],
                           'missing/duplicate fixture cases')
            guards.require(not cases['disabled'], 'disabled scope emitted output')
            valid = bench.extract_cpu_observers(cases['valid'], 4)
            status_counts = dict.fromkeys(bench.CPU_STATUS_NAMES, 0)
            status_counts.update(precheck=1, drain=1)
            status = bench.extract_status_observer(cases['valid'], 1, status_counts)
            guards.require(all(valid['host'][name + '_count'] == '1' for name in
                               bench.CPU_HOST_BASE + bench.CPU_HOST_MOE), 'fixture region count mismatch')
            guards.require('fixture_status=PASS' in out.read_text(), 'missing CPU fixture success')
            rejected = []
            for name in ('exception', 'nested_host', 'host_inside_wait', 'nested_scope',
                         'status_nested_scope', 'status_nested_part', 'status_part_inside_wait', 'status_exception'):
                try:
                    bench.extract_cpu_observers(cases[name], 4)
                except (RuntimeError, KeyError, ValueError):
                    rejected.append(name)
                else:
                    raise RuntimeError('invalid fixture profile accepted: ' + name)
            mutations = {
                'duplicate_header': cases['valid'] + cases['valid'].splitlines()[0] + '\n',
                'wrong_version': cases['valid'].replace('version=4', 'version=3'),
                'throw_without_clock_error': cases['valid'].replace('moe_gather_throws=0', 'moe_gather_throws=1'),
                'bad_partition': re.sub(r'unclassified_cpu_s=[0-9.]+', 'unclassified_cpu_s=999', cases['valid']),
            }
            for name, value in mutations.items():
                try:
                    bench.extract_cpu_observers(value, 4)
                except (RuntimeError, KeyError, ValueError):
                    rejected.append(name)
                else:
                    raise RuntimeError('invalid profile mutation accepted: ' + name)
            for name, text in {
                'status_duplicate': cases['valid'] + next(x+'\n' for x in cases['valid'].splitlines()
                    if x.startswith('strata prefill status:')),
                'status_missing_call': cases['valid'].replace('precheck_count=1', 'precheck_count=0'),
                'status_bad_partition': cases['valid'].replace('drain_wait_cpu_s=', 'drain_wait_cpu_s=999'),
                'status_incomplete': cases['valid'].replace('scopes=1 completed=1', 'scopes=1 completed=0'),
            }.items():
                try:
                    bench.extract_status_observer(text, 1, status_counts)
                except (RuntimeError, KeyError, ValueError):
                    rejected.append(name)
                else:
                    raise RuntimeError('invalid status profile accepted: '+name)
            for name, expected in pins.items():
                guards.require(digest(P / name) == expected, 'CPU input changed during test: ' + name)
            record.update(complete=True, passed=True, valid_profile=valid, status_partition=status,
                          cases=list(cases), rejected_profiles=rejected,
                          limits='CPU observer lifetime/accounting only; no GPU arithmetic, model quality or performance qualification.')
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            record['passed'] = bool(record['passed'] and record['ownership_closed'])
            persist()
            RESULT.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('complete', 'passed', 'ownership_closed', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
