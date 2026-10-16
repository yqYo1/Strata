#!/usr/bin/env python3
"""Bounded, low-priority routing-metadata check; never loads a GPU runtime."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

D = Path(__file__).resolve().parent
A = D.parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = D / 'host-result-v2.json'
    root = B / 'main-moe-window-route-map-v2'
    if result.exists() or root.exists():
        raise RuntimeError('no receipt or artifact overwrite')
    helper = A / 'direct_owner_transient_artifacts_v1.py'
    if sha(helper) != 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a':
        raise RuntimeError('qualified owner hash mismatch')
    spec = importlib.util.spec_from_file_location('route_owner', helper)
    owner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner_module)
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir()
        (root / 'tmp').mkdir()
        sources = {str(p.relative_to(P)): sha(p) for p in (
            Path(__file__).resolve(), D / 'moe_window_routes.hpp', D / 'verify_routes.cpp')}
        compiler = Path(shutil.which('c++'))
        nice = Path(shutil.which('nice'))
        binary = root / 'verify_routes'
        env = {'PATH': '/home/yayoi/.nix-profile/bin:/usr/bin:/bin',
               'LANG': 'C', 'LC_ALL': 'C', 'TMPDIR': str(root / 'tmp'),
               'ASAN_OPTIONS': 'detect_leaks=1:abort_on_error=1',
               'UBSAN_OPTIONS': 'halt_on_error=1:print_stacktrace=1'}
        record = dict(schema='strata-moe-window-native-host-route-check-v1',
            passed=False, complete=False, GPU_executed=False, tensor_math_checked=False,
            source_commit_before_check='a9ba8a252d3021e2c69625acb81134f48876ebae',
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            sources=sources, owner_sha256=sha(helper),
            compiler=dict(path=str(compiler), resolved_path=str(compiler.resolve()), sha256=sha(compiler)),
            nice=dict(path=str(nice), resolved_path=str(nice.resolve()), sha256=sha(nice)), environment=env,
            supersedes_attempt=dict(path='host-result-v1.json', sha256=sha(D / 'host-result-v1.json'),
                reason='Preserve multicall executable symlink names; resolving nice to coreutils changed argv[0]. Original failure stays unchanged.'),
            limits=dict(priority=19, wall_seconds_per_command=45, CPU_seconds_per_process=15,
                        session_RSS_bytes=512 << 20, logs_bytes_per_command=64 << 10,
                        artifacts_bytes=16 << 20),
            scope='Actual C++ router ID -> stable expert source/slot/job metadata, '
                  'including ne==W. No SYCL, GEMM, queue/DMA, physical fit, state, RESTORE or speed proof.',
            foreign_measurement_policy='Do not modify the foreign benchmark. Only this bounded, '
                  'single-threaded nice19 host fixture may run; no inference or heavy SYCL builds.',
            retention=dict(root=str(root), budget_bytes=16 << 20,
                consumer='Main integration of the checked route map into the GPU window',
                review_point='First engine window build/check; retire standalone binary and duplicate logs '
                             'after compact evidence is verified and measurements have closed.'))
        def save():
            result.write_text(json.dumps(record, indent=2) + '\n')
        owner = owner_module.Owner(root / 'owned', save)
        record['commands'] = owner.commands
        save()
        try:
            flags = ['-std=c++17', '-O1', '-g1', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
                     '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-fno-omit-frame-pointer']
            commands = [
                ('version', [str(nice), '-n', '19', str(compiler), '--version']),
                ('compile', [str(nice), '-n', '19', str(compiler), *flags,
                             str(D / 'verify_routes.cpp'), '-o', str(binary)]),
                ('verify', [str(nice), '-n', '19', str(binary)])]
            for label, argv in commands:
                if not all(sha(P / p) == h for p, h in sources.items()):
                    raise RuntimeError('source changed during check')
                entry, stdout, stderr = owner.run(label, argv, env, P, wall=45, cpu=15,
                    rss_cap=512 << 20, text_cap=64 << 10, total_cap=16 << 20, file_cap=8 << 20)
                if not owner_module.closed(entry) or entry['exit_code'] != 0:
                    raise RuntimeError(label + ' did not complete cleanly')
                if label == 'version':
                    record['compiler']['version'] = stdout.read_text()
                elif label == 'compile':
                    record['binary'] = dict(path=str(binary), bytes=binary.stat().st_size, sha256=sha(binary))
                else:
                    record['checks'] = json.loads(stdout.read_text())
                    if not record['checks']['passed'] or stderr.stat().st_size:
                        raise RuntimeError('fixture failed or sanitizer output present')
            record.update(passed=True, complete=True, ownership_closed=all(
                owner_module.closed(e) for e in owner.commands))
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save()
        print(json.dumps(dict(passed=record['passed'], complete=record['complete'],
                              checks=record.get('checks'), receipt=str(result))), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
