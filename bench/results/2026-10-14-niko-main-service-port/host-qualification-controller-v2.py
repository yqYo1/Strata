#!/usr/bin/env python3
"""Serial service/host/syntax checks; never executes a GPU or model."""
import fcntl
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
D = P / 'bench/results/2026-10-14-niko-main-service-port'
O = B / 'main-niko-primary-host-qualification-root-v2'
T = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
GG = Path('/home/yayoi/ghq/github.com/ggml-org/llama.cpp/.worktree/dep-ggml-3cf-xe-e32-20261011')
DEP = B / 'niko-service-test-deps-v1'
OWNER = B / 'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(OWNER.read_bytes()).hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('direct_owner', OWNER)
owner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner_module)
env = json.loads((B / 'main-layer-major-bulk-full-lints-root-v3/record.json').read_text())['environment']
env = {k: v for k, v in env.items() if k != 'STRATA_LINT_BUILD'}
env['PYTHONPATH'] = str(DEP) + ':' + str(P / 'tools') + ':' + str(P)
env['PIP_CONFIG_FILE'] = '/dev/null'

def main():
    assert not O.exists()
    O.mkdir()
    record = {'active': True, 'passed': False, 'complete': False, 'GPU_model_executed': False,
              'environment': env, 'commands': [], 'source_pins': {}, 'source_revision': subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip()}
    source_paths = [*json.loads((D / 'source-import.json').read_text())['files'],
        *[{'path': p} for p in ['src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'include/strata/sycl/checked_cleanup.hpp',
          'src/program/generate.cpp', 'include/strata/program/prompt_checkpoint.hpp', 'tests/sycl/checked_cleanup_host.cpp', 'tests/core/prompt_checkpoint_test.cpp', 'CMakeLists.txt']]]
    for row in source_paths:
        record['source_pins'][row['path']] = hashlib.sha256((P / row['path']).read_bytes()).hexdigest()
    def persist():
        (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    owner = owner_module.Owner(O / 'owned', persist)
    def run(label, argv, wall=300):
        entry, stdout, stderr = owner.run(label, [str(x) for x in argv], env, P,
            wall=wall, text_cap=16 << 20, total_cap=512 << 20, file_cap=32 << 20)
        record['commands'].append(entry)
        assert owner_module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), entry.get('exit_code'), stderr.read_text()[-2000:])
        print(json.dumps({'passed': label}), flush=True)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            python = '/home/yayoi/.local/share/strata-sycl/venv/bin/python'
            assert not DEP.exists(), 'Do not mutate another test dependency consumer'
            run('test-dependency', ['/home/yayoi/.nix-profile/bin/uv', '--no-config', '--no-progress', 'pip', 'install', '--python', python, '--index-url', 'https://pypi.org/simple', '--only-binary', ':all:', '--target', DEP, 'jsonschema==4.26.0'])
            versions = []
            for path in sorted(DEP.glob('*.dist-info/METADATA')):
                text = path.read_text()
                versions.append({'metadata': str(path), 'name': next(x[6:] for x in text.splitlines() if x.startswith('Name: ')),
                                 'version': next(x[9:] for x in text.splitlines() if x.startswith('Version: ')), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            record['test_dependencies'] = {'scope': 'project investigation test target only; serving venv unchanged', 'packages': versions,
                'jsonschema_source': 'https://pypi.org/project/jsonschema/4.26.0/', 'consumer': 'Niko service port and subsequent mandatory-feature tests',
                'byte_budget': 32 << 20, 'review_point': 'after main functional reconciliation or dependency supersession'}
            run('service-unit-schema', [python, D / 'run_service_tests.py', O / 'service-results.json'], wall=480)
            samples = json.loads((O / 'service-results.json').read_text())
            assert samples['passed'] and samples['tests_run'] >= 268
            record['service_result'] = {'tests_run': samples['tests_run'], 'skips': samples['skips'], 'path': str(O / 'service-results.json'), 'sha256': hashlib.sha256((O / 'service-results.json').read_bytes()).hexdigest()}
            clang = T / 'bin/clang++'
            base = [clang, '-std=c++20', '-Wall', '-Wextra', '-Werror', '-UNDEBUG', '-I' + str(P / 'include')]
            for case, path in [('cleanup', P / 'tests/sycl/checked_cleanup_host.cpp'), ('checkpoint', P / 'tests/core/prompt_checkpoint_test.cpp')]:
                for mode, flags in [('normal', ['-O2']), ('asan-ubsan', ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer'])]:
                    exe = O / (case + '-' + mode)
                    run(case + '-compile-' + mode, [*base, *flags, path, '-o', exe])
                    run(case + '-run-' + mode, [exe], wall=30)
            run('host-cmake-configure', ['cmake', '-S', P, '-B', O / 'cmake', '-G', 'Ninja', '-DCMAKE_CXX_COMPILER=' + str(clang), '-DSTRATA_ENABLE_CUDA=OFF', '-DSTRATA_ENABLE_HIP=OFF', '-DSTRATA_ENABLE_SYCL=OFF', '-DSTRATA_NATIVE_EXPERTS=OFF', '-DSTRATA_BUILD_TESTS=ON', '-DSTRATA_WERROR=ON', '-DSTRATA_PORTABLE=OFF'])
            run('host-cmake-build', ['cmake', '--build', O / 'cmake', '--target', 'sycl_checked_cleanup_host_test', 'prompt_checkpoint_test', '--parallel', '2'])
            run('host-cmake-ctest', ['ctest', '--test-dir', O / 'cmake', '-R', '^(sycl_checked_cleanup_host|prompt_checkpoint_test)$', '--output-on-failure'])
            syntax = [*base, '-fsycl', '-fsyntax-only', '-DSTRATA_ENABLE_SYCL=1', '-DSTRATA_NATIVE_EXPERTS=1',
                      '-I' + str(P / 'include/strata/sycl/compat'), '-I' + str(GG / 'ggml/include'), '-I' + str(GG / 'ggml/src')]
            run('SYCL-runtime-compat-syntax', [*syntax, P / 'src/sycl/runtime.cpp', P / 'src/sycl/compat.cpp'])
            run('SYCL-generator-syntax', [*syntax, P / 'src/program/generate.cpp'])
            for path, digest in record['source_pins'].items():
                assert hashlib.sha256((P / path).read_bytes()).hexdigest() == digest
            record.update({'passed': True, 'complete': True, 'host_cleanup_and_checkpoint_qualified': True,
                           'SYCL_link_device_execution_and_model_qualified': False, 'physical_262144_KV_qualified': False})
        except Exception as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = False
            persist()
            (D / 'host-qualification-controller-v2.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'complete': record['complete'], 'passed': record['passed']}), flush=True)

if __name__ == '__main__':
    main()
