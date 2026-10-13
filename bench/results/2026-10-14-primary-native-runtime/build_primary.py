#!/usr/bin/env python3
"""Build the actual primary SYCL runtime fixtures; no GPU execution."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = P / 'bench/results/2026-10-14-primary-native-runtime'
O = B / 'main-primary-native-build-root-v1'
T = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
GG = Path('/home/yayoi/ghq/github.com/ggml-org/llama.cpp/.worktree/dep-ggml-3cf-xe-e32-20261011')
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seed = P / 'bench/results/2026-10-14-niko-main-service-port/receipts/main-niko-primary-host-qualification-root-v2/record.json.gz'
    env = json.loads(gzip.decompress(seed.read_bytes()))['environment']
    env['PATH'] = str(T / 'bin') + ':/usr/bin:/bin'
    env['LD_LIBRARY_PATH'] = str(T / 'lib') + ':/opt/intel/oneapi/mkl/2026.1/lib'
    assert not any('TOKEN' in key or 'API_KEY' in key for key in env)
    assert not O.exists()
    O.mkdir()
    source = ['CMakeLists.txt', 'cmake/sycl_backend.cmake', 'src/sycl/runtime.cpp',
              'src/sycl/compat.cpp', 'include/strata/sycl/runtime.hpp',
              'include/strata/sycl/checked_cleanup.hpp', 'tests/sycl/runtime.cpp', 'tests/sycl/compat.cpp']
    record = dict(active=True, passed=False, complete=False, GPU_executed=False,
                  source_revision=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip(),
                  source_pins={f: sha(P / f) for f in source}, environment=env, commands=[],
                  compiler=dict(path=str(T / 'bin/clang++'), sha256=sha(T / 'bin/clang++')),
                  dependency_revision=subprocess.check_output(['git', '-C', str(GG), 'rev-parse', 'HEAD'], text=True).strip(),
                  build_consumer='primary small runtime and subsequent native kernel/model qualification',
                  build_byte_budget=4 << 30, review_point='after primary baseline qualification or recipe supersession')
    def persist():
        (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    owner = module.Owner(O / 'owned', persist)
    def run(label, argv, wall):
        entry, out, err = owner.run(label, [str(x) for x in argv], env, P, wall=wall,
                                    text_cap=64 << 20, total_cap=4 << 30, rss_cap=48 << 30, file_cap=256 << 20)
        record['commands'].append(entry)
        assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), err.read_text()[-3000:])
        print(json.dumps(dict(passed=label)), flush=True)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run('compiler-version', [T / 'bin/clang++', '--version'], 30)
            run('configure-primary-native', ['cmake', '-S', P, '-B', O / 'build', '-G', 'Ninja',
                '-DCMAKE_CXX_COMPILER=' + str(T / 'bin/clang++'), '-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DSTRATA_ENABLE_SYCL=ON',
                '-DSTRATA_ENABLE_CUDA=OFF', '-DSTRATA_ENABLE_HIP=OFF', '-DSTRATA_PORTABLE=OFF',
                '-DSTRATA_NATIVE_EXPERTS=ON', '-DSTRATA_CPU_ARCH=znver3', '-DGGML_NATIVE=ON',
                '-DSTRATA_GGML_DIR=' + str(GG), '-DSTRATA_SYCL_TESTS=ON', '-DSTRATA_BUILD_TESTS=OFF',
                '-DSTRATA_WERROR=ON', '-DMKL_DIR=/opt/intel/oneapi/mkl/2026.1/lib/cmake/mkl'], 180)
            run('build-primary-runtime-fixtures', ['cmake', '--build', O / 'build', '--target',
                'sycl_runtime_test', 'sycl_compat_test', '--parallel', '2'], 600)
            record['binaries'] = {f: sha(O / 'build' / f) for f in ['sycl_runtime_test', 'sycl_compat_test']}
            record['compile_commands_sha256'] = sha(O / 'build/compile_commands.json')
            for f, digest in record['source_pins'].items():
                assert sha(P / f) == digest
            record.update(passed=True, complete=True)
        except Exception as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = False
            persist()
    print(json.dumps(dict(passed=record['passed'], GPU_executed=False, build=str(O / 'build'))), flush=True)

if __name__ == '__main__':
    main()
