#!/usr/bin/env python3
"""Build primary engine and prompt parity fixtures after small runtime proof."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
BUILD = B / 'main-primary-native-build-root-v1'
O = B / 'main-primary-native-engine-build-root-v5'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prior = json.loads((BUILD / 'record.json').read_text())
    runtime = json.loads((B / 'main-primary-native-runtime-root-v1/record.json').read_text())
    assert prior['passed'] and runtime['passed'] and not runtime['active']
    assert not O.exists()
    O.mkdir()
    paths = sorted(set(Path(c['file']) for c in json.loads((BUILD / 'build/compile_commands.json').read_text())
                       if Path(c['file']).is_relative_to(P)))
    paths += sorted((P / 'include').rglob('*.hpp'))
    record = dict(active=True, passed=False, complete=False, GPU_executed=False, commands=[],
                  environment=prior['environment'], source_pins={str(f.relative_to(P)): sha(f) for f in paths},
                  compile_commands_sha256=sha(BUILD / 'build/compile_commands.json'),
                  reused_small_runtime_proof=str(B / 'main-primary-native-runtime-root-v1/record.json'),
                  build_path=str(BUILD / 'build'), budget_bytes=4 << 30,
                  consumer='primary parity tests and subsequent official-client model baseline')
    def persist():
        (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    owner = module.Owner(BUILD / 'engine-build-owned-v5', persist)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            configured, cfg_out, cfg_err = owner.run('configure-exp-api-check',
                ['cmake', '-S', str(P), '-B', str(BUILD / 'build')], record['environment'], P,
                wall=180, text_cap=64 << 20, total_cap=4 << 30, rss_cap=48 << 30, file_cap=256 << 20)
            record['commands'].append(configured)
            assert module.closed(configured) and configured['exit_code'] == 0, cfg_err.read_text()[-3000:]
            record['compile_commands_sha256'] = sha(BUILD / 'build/compile_commands.json')
            argv = ['cmake', '--build', str(BUILD / 'build'), '--target', 'strata',
                    'sycl_gemm_test', 'sycl_prefill_mmq_test', 'sycl_prefill_fused_test',
                    'sycl_native_single_dispatch_test', '--parallel', '2']
            entry, out, err = owner.run('primary-engine-and-parity-build', argv, record['environment'], P,
                                        wall=1500, cpu=1500, text_cap=64 << 20, total_cap=4 << 30,
                                        rss_cap=48 << 30, file_cap=256 << 20)
            record['commands'].append(entry)
            assert module.closed(entry) and entry['exit_code'] == 0, (entry.get('error'), err.read_text()[-4000:], out.read_text()[-4000:])
            names = ['strata', 'sycl_gemm_test', 'sycl_prefill_mmq_test',
                     'sycl_prefill_fused_test', 'sycl_native_single_dispatch_test']
            record['binaries'] = {name: sha(BUILD / 'build' / name) for name in names}
            for f, digest in record['source_pins'].items():
                assert sha(P / f) == digest
            record.update(passed=True, complete=True)
        except Exception as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = False
            persist()
    print(json.dumps(dict(passed=record['passed'], GPU_executed=False, binaries=record.get('binaries'))), flush=True)

if __name__ == '__main__':
    main()
