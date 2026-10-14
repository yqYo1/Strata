#!/usr/bin/env python3
"""Serial current-build host-only routed FP16 observation; no kernel/math change."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE = Path(__file__).parent
BUILD = B / 'main-primary-native-build-root-v1/build'
O = B / 'main-primary-routed-fp16-census-build-root-v1'
BASELINE = B / 'main-primary-pre-census-native-root-v1'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
SEED = B / 'main-primary-compact-moe-build-root-v1/record.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    seed = json.loads(SEED.read_text())
    assert seed['passed'] and seed['complete'] and not seed['active']
    env = seed['environment']
    assert not any('TOKEN' in key or 'API_KEY' in key for key in env)
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists() and not BASELINE.exists()
        native = BUILD / 'strata'
        assert sha(native) == seed['binaries']['strata'] == '6361e706e2313049f8697e6cabe36a318737d541db323ba60fe13f6a5a223e09'
        changed = subprocess.check_output(['git', '-C', str(P), 'diff', '--name-only', '--', 'src', 'include', 'cmake'], text=True).splitlines()
        assert changed == ['src/prefill/prefill.cpp'], changed
        BASELINE.mkdir()
        shutil.copy2(native, BASELINE / 'strata')
        assert sha(BASELINE / 'strata') == sha(native)
        (BASELINE / 'consumer.json').write_text(json.dumps(dict(
            binary_sha256=sha(native), environment=env,
            consumer='Exact pre-census native for default-OFF numerical regression and observation overhead comparison',
            byte_budget=128 << 20, next_review='After route-census numerical/overhead comparison closes'), indent=2) + '\n')
        O.mkdir()
        paths = set(seed['source_pins']) | {'CMakeLists.txt', 'cmake/sycl_backend.cmake', 'src/prefill/prefill.cpp'}
        record = dict(active=True, passed=False, complete=False, GPU_executed=False, commands=[],
            environment=env, source_revision=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip(),
            source_pins={name: sha(P / name) for name in sorted(paths)},
            old_binary_sha256=sha(BASELINE / 'strata'), baseline_binary_path=str(BASELINE / 'strata'),
            recipe_sha256=sha(Path(__file__)), seed_receipt_sha256=sha(SEED), build_path=str(BUILD),
            scope='Default-OFF bounded exact route histogram and attempted blob-copy payload observation. No new device event/copy/wait, kernel math, CPU/decode, queue or compiler-flag change.',
            build_byte_budget=4 << 30, consumer='Main-owned model numerical and unchanged stock-client comparisons')
        patch = subprocess.check_output(['git', '-C', str(P), 'diff', '--', 'src/prefill/prefill.cpp'])
        (O / 'source.patch').write_bytes(patch)
        record['source_diff_sha256'] = hashlib.sha256(patch).hexdigest()
        def persist():
            (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(O / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        def run(label, argv, wall):
            entry, out, err = owner.run(label, argv, env, P, wall=wall, cpu=wall,
                text_cap=64 << 20, rss_cap=48 << 30, total_cap=1 << 30, file_cap=256 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), err.read_text()[-2500:], out.read_text()[-2500:])
            print(json.dumps(dict(passed=label)), flush=True)
        try:
            run('build-routed-census-and-existing-fixtures', ['cmake', '--build', str(BUILD), '--target',
                'strata', 'sycl_range_contract_test', 'sycl_gemm_test', 'sycl_prefill_mmq_test', '--parallel', '2'], 1500)
            run('host-range-contract', [str(BUILD / 'sycl_range_contract_test')], 120)
            commands = json.loads((BUILD / 'compile_commands.json').read_text())
            sycl = [x for x in commands if Path(x['file']).is_relative_to(P) and '-fsycl ' in x['command']]
            assert sycl and all('-fsycl-id-queries-range=size_t' in x['command'] for x in sycl)
            selected = [x for x in commands if x['file'].endswith('/src/prefill/prefill.cpp')]
            assert len(selected) == 1 and Path(selected[0]['file']) == P / 'src/prefill/prefill.cpp'
            record['compile_flag_proof'] = dict(consumers=len(sycl), all_size_t=True,
                compile_commands_sha256=sha(BUILD / 'compile_commands.json'), prefill=selected[0])
            record['binaries'] = {name: sha(BUILD / name) for name in
                ['strata', 'sycl_range_contract_test', 'sycl_gemm_test', 'sycl_prefill_mmq_test']}
            for name, pin in record['source_pins'].items():
                assert sha(P / name) == pin, name
            assert sha(BASELINE / 'strata') == record['old_binary_sha256']
            record['build_logical_bytes'] = sum(f.stat().st_size for f in BUILD.rglob('*') if f.is_file())
            assert record['build_logical_bytes'] <= record['build_byte_budget']
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = False
            persist()
        print(json.dumps(dict(passed=True, GPU_executed=False, binaries=record['binaries'])), flush=True)


if __name__ == '__main__':
    main()
