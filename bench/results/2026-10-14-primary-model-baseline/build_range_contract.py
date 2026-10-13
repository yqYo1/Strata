#!/usr/bin/env python3
"""Serial, bounded size_t-range build of the actual Niko-based primary engine."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = P / 'bench/results/2026-10-14-primary-model-baseline'
O = B / 'main-primary-size-t-range-build-root-v1'
BUILD = B / 'main-primary-native-build-root-v1/build'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
SEED = P / 'bench/results/2026-10-14-primary-native-runtime/receipts/main-primary-native-engine-build-root-v13/record.json.gz'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seed = json.loads(gzip.decompress(SEED.read_bytes()))
    assert seed['passed'] and not seed['active']
    for name, digest in seed['source_pins'].items():
        if name != 'cmake/sycl_backend.cmake':
            assert sha(P / name) == digest, name
    assert sha(BUILD / 'strata') == seed['binaries']['strata']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    assert not O.exists()
    O.mkdir()
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = set(seed['source_pins']) | {'CMakeLists.txt', 'cmake/sycl_backend.cmake',
                                      'tests/sycl/range_contract.cpp'}
    record = dict(active=True, passed=False, complete=False, GPU_executed=False,
        commands=[], environment=seed['environment'], old_binary_sha256=sha(BUILD / 'strata'),
        seed_receipt=str(SEED.relative_to(P)), seed_receipt_sha256=sha(SEED),
        source_revision=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip(),
        source_pins={name: sha(P / name) for name in sorted(paths)},
        recipe_sha256=sha(Path(__file__)), build_path=str(BUILD),
        scope='host global-range contract and complete native primary build; no model/performance/256K claim',
        consumer='correctness fixtures and primary real-model retry after range countermeasure',
        build_byte_budget=4 << 30)

    def persist():
        (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

    owner = module.Owner(O / 'owned', persist)

    def run(label, argv, wall=120):
        entry, out, err = owner.run(label, argv, record['environment'], P,
            wall=wall, cpu=wall, text_cap=64 << 20, rss_cap=48 << 30,
            total_cap=1 << 30, file_cap=256 << 20)
        record['commands'].append(entry)
        assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), err.read_text()[-2500:], out.read_text()[-2500:])
        print(json.dumps(dict(passed=label)), flush=True)

    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run('configure-size-t-range', ['cmake', '-S', str(P), '-B', str(BUILD)], 180)
            run('build-primary-and-range-fixtures', ['cmake', '--build', str(BUILD), '--target',
                'strata', 'sycl_range_contract_test', 'sycl_runtime_test', 'sycl_compat_test',
                'sycl_prefill_mmq_test', 'sycl_gemm_test', '--parallel', '2'], 1500)
            run('host-range-contract', [str(BUILD / 'sycl_range_contract_test')])
            commands = json.loads((BUILD / 'compile_commands.json').read_text())
            consumers = [x for x in commands if Path(x['file']).is_relative_to(P) and '-fsycl ' in x['command']]
            assert consumers and all('-fsycl-id-queries-range=size_t' in x['command'] for x in consumers)
            record['compile_flag_proof'] = dict(consumers=len(consumers), all_size_t=True,
                compile_commands_sha256=sha(BUILD / 'compile_commands.json'))
            record['binaries'] = {name: sha(BUILD / name) for name in ['strata', 'sycl_range_contract_test',
                'sycl_runtime_test', 'sycl_compat_test', 'sycl_prefill_mmq_test', 'sycl_gemm_test']}
            for name, digest in record['source_pins'].items():
                assert sha(P / name) == digest, name
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
