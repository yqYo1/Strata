#!/usr/bin/env python3
"""Production metadata loader with host-only adapter stubs, normal and ASan/UBSan."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
O = B / 'main-primary-weight-index-host-root-v1'
T = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'

def main():
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = json.loads((B / 'main-primary-native-build-root-v1/record.json').read_text())
    assert not O.exists()
    O.mkdir()
    record = dict(active=True, passed=False, complete=False, GPU_executed=False,
                  environment=build['environment'], commands=[], source_pins={f: hashlib.sha256((P / f).read_bytes()).hexdigest()
                  for f in ['src/core/weights.cpp', 'tests/core/weights_index_host.cpp', 'include/strata/core/weights.hpp']})
    def persist():
        (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    owner = module.Owner(O / 'owned', persist)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for mode, flags in [('normal', ['-O2']), ('asan-ubsan', ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer'])]:
                exe = O / ('weights-index-' + mode)
                commands = [('compile-' + mode, [str(T / 'bin/clang++'), '-std=c++20', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(P / 'include'), '-I' + str(P / 'include/strata/sycl/compat'), *flags,
                    str(P / 'src/core/weights.cpp'), str(P / 'tests/core/weights_index_host.cpp'), '-o', str(exe)]),
                    ('run-' + mode, [str(exe), str(O / ('fixture-' + mode))])]
                for label, argv in commands:
                    entry, out, err = owner.run(label, argv, record['environment'], P, wall=120, total_cap=128 << 20)
                    record['commands'].append(entry)
                    assert module.closed(entry) and entry['exit_code'] == 0, (label, err.read_text()[-2000:])
            record.update(passed=True, complete=True)
        except Exception as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = False
            persist()
    print(json.dumps(dict(passed=record['passed'], GPU_executed=False)), flush=True)

if __name__ == '__main__':
    main()
