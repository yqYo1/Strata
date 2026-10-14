#!/usr/bin/env python3
"""Main-owned serialized production build and host concurrency qualification."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
O = B / 'main-primary-watchdog-lifecycle-build-root-v1'
BUILD = B / 'main-primary-native-build-root-v1/build'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
SEED = D / 'receipts/main-primary-size-t-range-build-root-v2/record.json.gz'
T = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seed = json.loads(gzip.decompress(SEED.read_bytes()))
    assert seed['passed'] and not seed['active']
    changed = {'CMakeLists.txt', 'include/strata/core/verify.hpp', 'src/core/verify.cpp', 'src/program/generate.cpp'}
    for name, digest in seed['source_pins'].items():
        if name not in changed:
            assert sha(P / name) == digest, name
    assert sha(BUILD / 'strata') == seed['binaries']['strata']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    assert not O.exists()
    O.mkdir()
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = set(seed['source_pins']) | changed | {'include/strata/core/watchdog_lifecycle.hpp', 'tests/cpu/watchdog_lifecycle.cpp'}
    record = dict(active=True, passed=False, complete=False, GPU_executed=False, commands=[],
        environment=seed['environment'], old_binary_sha256=sha(BUILD/'strata'),
        seed_receipt=str(SEED.relative_to(P)), seed_receipt_sha256=sha(SEED),
        source_revision=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip(),
        source_pins={name: sha(P/name) for name in sorted(paths)}, recipe_sha256=sha(Path(__file__)),
        proposal_sha256=sha(D/'proposed-watchdog-lifecycle.patch'), build_path=str(BUILD),
        scope='production watchdog/verifier lifecycle plus exact-helper CPU concurrency and ASan/UBSan; no GPU/model/performance claim',
        consumer='small GPU compat/runtime and real-model normal shutdown qualification',
        main_changes_to_proposal=['reject null registry entries', 'reject repeated watchdog start',
            'check actual mapped-flag atomic alignment', 'observe closing before lease-block assertions'],
        build_byte_budget=4 << 30)

    def persist():
        (O/'record.json').write_text(json.dumps(record, indent=2)+'\n')

    owner = module.Owner(O/'owned', persist)
    record['commands'] = owner.commands
    persist()

    def run(label, argv, wall=120, environment=None):
        entry, out, err = owner.run(label, argv, environment or record['environment'], P,
            wall=wall, cpu=wall, text_cap=64 << 20, rss_cap=48 << 30,
            total_cap=1 << 30, file_cap=256 << 20)
        assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), err.read_text()[-2500:], out.read_text()[-2500:])
        print(json.dumps(dict(passed=label)), flush=True)

    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run('configure-lifecycle', ['cmake', '-S', str(P), '-B', str(BUILD)], 180)
            run('build-native-and-host-lifecycle', ['cmake', '--build', str(BUILD), '--target',
                'strata', 'watchdog_lifecycle_host_test', '--parallel', '2'], 1500)
            run('cpu-lifecycle', [str(BUILD/'watchdog_lifecycle_host_test')], 30)
            sanitized = O/'watchdog-lifecycle-asan-ubsan'
            run('compile-cpu-asan-ubsan', [str(T/'bin/clang++'), '-std=c++20', '-O1', '-g', '-pthread',
                '-fno-omit-frame-pointer', '-fsanitize=address,undefined', '-I'+str(P/'include'),
                str(P/'tests/cpu/watchdog_lifecycle.cpp'), '-o', str(sanitized)], 120)
            env = dict(record['environment'], ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                       UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
            record['sanitizer_environment'] = env
            run('cpu-asan-ubsan', [str(sanitized)], 30, env)
            commands = json.loads((BUILD/'compile_commands.json').read_text())
            consumers = [x for x in commands if Path(x['file']).is_relative_to(P) and '-fsycl ' in x['command']]
            assert consumers and all('-fsycl-id-queries-range=size_t' in x['command'] for x in consumers)
            record['compile_flag_proof'] = dict(consumers=len(consumers), all_size_t=True,
                compile_commands_sha256=sha(BUILD/'compile_commands.json'))
            record['binaries'] = {name:sha(BUILD/name) for name in ['strata', 'watchdog_lifecycle_host_test',
                'sycl_range_contract_test', 'sycl_runtime_test', 'sycl_compat_test', 'sycl_prefill_mmq_test', 'sycl_gemm_test']}
            record['binaries']['watchdog-lifecycle-asan-ubsan'] = sha(sanitized)
            for name, digest in record['source_pins'].items():
                assert sha(P/name) == digest, name
            record['build_logical_bytes'] = sum(f.stat().st_size for f in BUILD.rglob('*') if f.is_file())
            assert record['build_logical_bytes'] <= record['build_byte_budget']
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = False
            persist()
    print(json.dumps(dict(passed=True, GPU_executed=False, native_binary=record['binaries']['strata'])), flush=True)


if __name__ == '__main__':
    main()
