#!/usr/bin/env python3
"""Main-owned serial build; preserve the qualified native binary as a current control."""
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
SEED = B / 'main-primary-draft-upload-fix-build-root-v1/record.json'
O = B / 'main-primary-qsa-layout-build-root-v1'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
CHANGES = {'src/kernels/sycl/decode_attention.cpp', 'src/prefill/prefill.cpp',
           'include/strata/kernels/qsa_decode_attn.hpp', 'cmake/sycl_backend.cmake'}
FIXTURE = 'tests/sycl/qsa_fallback_layout.cpp'


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
        summary = json.loads((P / 'bench/results/2026-10-14-long-generation-benchmark/suite-v4-upload-fix-final-summary.json').read_text())
        assert summary['benchmark_protocol_qualified'] and all(summary['gates'].values())
        for name, pin in seed['source_pins'].items():
            if name not in CHANGES:
                assert sha(P / name) == pin, ('unreviewed source change', name)
        assert sha(build / 'strata') == seed['binaries']['strata']
        assert not O.exists()
        O.mkdir()
        control = O / 'control-strata'
        shutil.copy2(build / 'strata', control)
        assert sha(control) == seed['binaries']['strata']
        assert control.stat().st_size <= 128 << 20
        paths = set(seed['source_pins']) | {FIXTURE}
        record = dict(active=True, passed=False, complete=False, commands=[], GPU_executed=False,
                      source_revision=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip(),
                      source_pins={n: sha(P / n) for n in sorted(paths)}, environment=env,
                      seed_receipt_sha256=sha(SEED), recipe_sha256=sha(Path(__file__)), build_path=str(build),
                      old_build_path_superseded=True, control_binary=dict(path=str(control), sha256=sha(control),
                      bytes=control.stat().st_size, consumer='new natural-layout vs qualified old native numerical comparison',
                      byte_budget=128 << 20, next_review='after new OFF/layout0 model parity and kernel fixture qualification'),
                      changed_sources=sorted(CHANGES), new_fixture=FIXTURE, build_byte_budget=4 << 30,
                      performance_qualified=False, quality_qualified=False, physical_262144_qualified=False,
                      scope='Current root SYCL fallback Q-local layout only; natural decode API preserved. Reuse existing compiler/configure; no runtime/driver/queue policy change. No GPU executed by this build controller.')

        def persist():
            (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(O / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        try:
            entry, out, err = owner.run('build-qsa-layout',
                ['cmake', '--build', str(build), '--target', 'strata', 'sycl_qsa_fallback_layout_test', '--parallel', '2'],
                env, P, wall=1500, cpu=1500 * len(os.sched_getaffinity(0)), text_cap=64 << 20,
                rss_cap=48 << 30, total_cap=1 << 30, file_cap=256 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (entry.get('error'), err.read_text()[-3000:], out.read_text()[-3000:])
            entry, out, err = owner.run('host-layout-contracts',
                [str(build / 'sycl_qsa_fallback_layout_test'), '--host-only'],
                env, P, wall=30, cpu=30 * len(os.sched_getaffinity(0)), text_cap=64 << 20,
                rss_cap=8 << 30, total_cap=1 << 30, file_cap=64 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (entry.get('error'), err.read_text()[-3000:])
            assert 'PASS host geometry/layout/overflow contracts' in out.read_text()
            record['host_contracts_passed'] = True
            commands = json.loads((build / 'compile_commands.json').read_text())
            proof = {}
            for name in ['src/prefill/prefill.cpp', 'src/kernels/sycl/decode_attention.cpp', FIXTURE]:
                selected = [c for c in commands if c['file'] == str(P / name)]
                assert len(selected) == 1 and '-fsycl-id-queries-range=size_t' in selected[0]['command']
                proof[name] = selected[0]
            record['compile_flag_proof'] = proof
            record['compile_commands_sha256'] = sha(build / 'compile_commands.json')
            record['binaries'] = {n: sha(build / n) for n in ['strata', 'sycl_qsa_fallback_layout_test']}
            record['build_logical_bytes'] = sum(p.stat().st_size for p in build.rglob('*') if p.is_file())
            assert record['build_logical_bytes'] <= record['build_byte_budget']
            assert all(sha(P / n) == pin for n, pin in record['source_pins'].items())
            assert sha(control) == record['control_binary']['sha256']
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = False
            persist()
            (D / 'qsa-layout-build-v1.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=True, GPU_executed=False, binaries=record['binaries'])), flush=True)


if __name__ == '__main__':
    main()
