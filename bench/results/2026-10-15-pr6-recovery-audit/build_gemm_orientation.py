#!/usr/bin/env python3
"""Build only the standalone descriptor probe after the QSA jobs close."""
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
SEED = B / 'main-primary-gemm-orientation-build-root-v1/record.json'
O = B / 'main-primary-gemm-orientation-build-root-v2'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
TOOL = 'tools/sycl/gemm_orientation_bench.cpp'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seed = json.loads(SEED.read_text())
    assert seed['passed'] and seed['complete'] and not seed['active']
    build = Path(seed['build_path'])
    env = dict(seed['environment'])
    assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        closed = json.loads((D / 'qsa-layout-screen-summary.json').read_text())
        assert closed['passed'] and closed['all_fresh_fault_checks_zero']
        for name, pin in seed['source_pins'].items():
            if name != TOOL:
                assert sha(P / name) == pin, name
        assert sha(build / 'strata') == seed['production_binary_sha256']
        assert not O.exists()
        O.mkdir()
        record = dict(active=True, passed=False, complete=False, commands=[], GPU_executed=False,
            environment=env, source_revision=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip(),
            source_pins={n: sha(P / n) for n in sorted(set(seed['source_pins']) | {TOOL})},
            recipe_sha256=sha(Path(__file__)), seed_receipt_sha256=sha(SEED), build_path=str(build),
            production_binary_sha256=sha(build / 'strata'), control_binary=seed['control_binary'],
            build_byte_budget=4 << 30, performance_qualified=False, quality_qualified=False,
            physical_262144_qualified=False, adopted=False,
            scope='V2 standalone descriptor probe: validation actual/padded strides; timing actual model ld=N. Product GEMM/native unchanged. Same compiler/library. No GPU in build/host-only mode.')

        def persist():
            (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(O / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        try:
            for label, argv, wall in [('build-gemm-orientation', ['cmake', '--build', str(build), '--target',
                                       'sycl_gemm_orientation_bench', '--parallel', '2'], 1200),
                                      ('host-gemm-contracts', [str(build / 'sycl_gemm_orientation_bench'), '--host-only'], 30)]:
                entry, out, err = owner.run(label, argv, env, P, wall=wall,
                    cpu=wall * len(os.sched_getaffinity(0)), text_cap=64 << 20, rss_cap=48 << 30,
                    total_cap=1 << 30, file_cap=256 << 20)
                assert module.closed(entry) and entry['exit_code'] == 0, (label, err.read_text()[-3000:], out.read_text()[-3000:])
                if label == 'host-gemm-contracts':
                    assert 'PASS host-only' in out.read_text()
                    record['host_contracts_passed'] = True
            commands = json.loads((build / 'compile_commands.json').read_text())
            selected = [c for c in commands if c['file'] == str(P / TOOL)]
            assert len(selected) == 1 and '-fsycl-id-queries-range=size_t' in selected[0]['command']
            record['compile_flag_proof'] = selected[0]
            record['binary_sha256'] = sha(build / 'sycl_gemm_orientation_bench')
            assert sha(build / 'strata') == record['production_binary_sha256']
            assert all(sha(P / n) == pin for n, pin in record['source_pins'].items())
            record['build_logical_bytes'] = sum(p.stat().st_size for p in build.rglob('*') if p.is_file())
            assert record['build_logical_bytes'] <= record['build_byte_budget']
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = False
            persist()
            (D / 'gemm-orientation-build-v2.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=True, GPU_executed=False, binary_sha256=record['binary_sha256'])), flush=True)


if __name__ == '__main__':
    main()
