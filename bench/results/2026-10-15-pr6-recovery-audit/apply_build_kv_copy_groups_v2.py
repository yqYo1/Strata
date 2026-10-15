#!/usr/bin/env python3
"""Main serial integration/build of opt-in bounded KV copy launch geometry."""
import fcntl
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, sha


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seed_path = D / 'phase-audit-build-v1.json'
        seed = json.loads(seed_path.read_text())
        assert seed['passed'] and seed['complete'] and not seed['active']
        closed_paths = [D / n for n in ['phase-audit-long-fixed-v1.json', 'phase-audit-long-auto-v1.json',
                                         'boundary-audit-phase-off-long-auto-v2.json']]
        assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
        spec = importlib.util.spec_from_file_location('owner', OWNER)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for path in closed_paths:
            receipt = json.loads(path.read_text())
            assert receipt['complete'] and not receipt['active'] and not receipt['fresh_faults']
            assert receipt['boot_unchanged'] and not receipt['driver_dump_exists']
            assert receipt['source_pins'] == seed['source_pins']
            for command in receipt['commands']:
                assert module.closed(command)
                for identity in command['owners']:
                    live = module.identity(identity['pid'])
                    assert live is None or live['start_ticks'] != identity['start_ticks'] or live['state'] == 'Z'
        assert all(sha(P / n) == h for n, h in seed['source_pins'].items())
        build = Path(seed['build_path'])
        assert all(sha(build / n) == h for n, h in seed['binaries'].items())
        proposal = D / 'bounded-kv-copy-groups-v2'
        patch = proposal / 'source.diff'
        assert sha(patch) == '26e20f64b5780668512f0930992081c817990e6c55a2aba2106e7861734b3a2d'
        pins = json.loads((proposal / 'pins.json').read_text())
        generated = json.loads((proposal / 'generated.json').read_text())
        assert all(sha(P / n) == h for n, h in pins['product_inputs'].items())
        subprocess.run(['git','apply','--check',str(patch)],cwd=P,check=True)
        output = B / 'main-primary-kv-copy-groups-build-root-v2'
        assert not output.exists(); output.mkdir(mode=0o700)
        env = dict(seed['environment']); assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        result = dict(active=True, passed=False, complete=False, GPU_executed=False, commands=[],
                      environment=env, recipe_sha256=sha(Path(__file__)), seed_receipt_sha256=sha(seed_path),
                      closed_receipt_sha256={p.name:sha(p) for p in closed_paths}, applied_patch_sha256=sha(patch),
                      build_path=str(build), previous_binaries=seed['binaries'], adopted=False,
                      performance_qualified=False, quality_qualified=False, physical_262144_qualified=False,
                      restore_qualified=False, scope='Opt-in launch geometry; default unchanged. CPU partition/oracle checks and actual build only.')
        def persist(): (output / 'record.json').write_text(json.dumps(result, indent=2) + '\n')
        owner = module.Owner(output / 'owned', persist); result['commands'] = owner.commands; persist()
        def run(label, argv, wall):
            entry, stdout, stderr = owner.run(label, argv, env, P, wall=wall,
                cpu=wall*len(os.sched_getaffinity(0)), rss_cap=48 << 30, text_cap=64 << 20,
                total_cap=256 << 20, file_cap=64 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), stderr.read_text()[-5000:])
            return dict(stdout=stdout.read_text(), stderr=stderr.read_text())
        try:
            subprocess.run(['git','apply',str(patch)],cwd=P,check=True)
            assert all(sha(P / n) == h for n, h in generated['files'].items())
            sources = set(seed['source_pins']) | set(pins['product_inputs']) | set(generated['files'])
            result['source_pins'] = {n:sha(P / n) for n in sorted(sources)}
            result['CPU_outputs'] = {}; persist()
            for sanitized in [False, True]:
                label = 'host-asan-ubsan' if sanitized else 'host'
                exe = output / label
                flags = ['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if sanitized else ['-O2']
                run('compile-' + label, ['/usr/bin/c++','-std=c++17','-UNDEBUG','-Wall','-Wextra','-Werror',
                    '-I' + str(P / 'include'), *flags, str(P/'tools/sycl/kv_copy_groups_host.cpp'), '-o', str(exe)], 90)
                result['CPU_outputs'][label] = run('run-' + label, [str(exe)], 300)
                result['CPU_outputs'][label]['binary_sha256'] = sha(exe)
                assert 'PASS host fixture coverage_cases=' in result['CPU_outputs'][label]['stdout']
                print('CPU PASS ' + label + ' ' + result['CPU_outputs'][label]['stdout'].strip(), flush=True); persist()
            result['host_contracts_passed'] = True; persist()
            run('build-kv-copy-groups', ['cmake','--build',str(build),'--target',
                'strata','kv_copy_groups_host_fixture','sycl_kv_copy_groups_bench','--parallel','2'], 1500)
            result['binaries'] = {n:sha(build / n) for n in [*seed['binaries'], 'kv_copy_groups_host_fixture','sycl_kv_copy_groups_bench']}
            commands = json.loads((build / 'compile_commands.json').read_text())
            result['compile_flag_proof'] = {}
            for name in ['src/kernels/sycl/kv_stream.cpp','tools/sycl/kv_copy_groups_bench.cpp']:
                found = [c for c in commands if c['file'] == str(P / name)]
                assert len(found) == 1 and '-fsycl-id-queries-range=size_t' in found[0]['command']
                result['compile_flag_proof'][name] = found[0]
            result['compile_commands_sha256'] = sha(build / 'compile_commands.json')
            assert all(sha(P / n) == h for n, h in result['source_pins'].items())
            result.update(passed=True, complete=True)
        except BaseException as error: result['error'] = repr(error); raise
        finally:
            result['active'] = False; persist()
            (D / 'kv-copy-groups-build-v2.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(passed=result['passed'], GPU_executed=False, binaries=result.get('binaries'))), flush=True)


if __name__ == '__main__': main()
