#!/usr/bin/env python3
"""Main-owned serial integration of the default-OFF layer26 diagnostic."""
import fcntl
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, sha


def main():
    seed_path = D / 'boundary-service-build-v1.json'
    seed = json.loads(seed_path.read_text())
    closed = json.loads((D / 'decode-format-existing-exact-normal-v1.json').read_text())
    host_path = D / 'phase-audit-host-v1.json'
    host = json.loads(host_path.read_text())
    assert seed['passed'] and closed['passed'] and closed['complete'] and not closed['active']
    assert host['passed'] and host['complete'] and not host['active'] and not host['GPU_executed']
    assert not closed['fresh_faults'] and closed['boot_unchanged'] and not closed['driver_dump_exists']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for command in closed['commands']:
            assert module.closed(command)
            for identity in command['owners']:
                live = module.identity(identity['pid'])
                assert live is None or live['start_ticks'] != identity['start_ticks'] or live['state'] == 'Z'
        assert all(sha(P / n) == h for n, h in seed['source_pins'].items())
        build = Path(seed['build_path'])
        assert all(sha(build / n) == h for n, h in seed['binaries'].items())
        patch = D / 'layer26-phase-audit-v1/source.diff'
        assert sha(patch) == 'c85f9c7689d4081dfbcbd63b849c7a83050b001054912c7e7e9ae34f7e3c63fa'
        template = D / 'layer26-phase-audit-v1/layer26_phase_audit.hpp.in'
        assert sha(template) == host['header_sha256']
        fixture_input = D / 'phase-audit-main-fixture-v1.cpp'
        assert sha(fixture_input) == host['fixture_sha256']
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=P, check=True)
        output = B / 'main-primary-phase-audit-build-root-v1'
        assert not output.exists(); output.mkdir(mode=0o700)
        env = dict(seed['environment'])
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        result = dict(active=True, passed=False, complete=False, GPU_executed=False, commands=[],
                      recipe_sha256=sha(Path(__file__)), seed_receipt_sha256=sha(seed_path),
                      host_receipt_sha256=sha(host_path), applied_patch_sha256=sha(patch),
                      source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
                      build_path=str(build), environment=env, previous_binaries=seed['binaries'],
                      performance_qualified=False, quality_qualified=False, physical_262144_qualified=False,
                      restore_qualified=False, adopted=False,
                      scope='Default-OFF diagnostic source and actual build; not a numerical correction or performance adoption.')
        def persist():
            (output / 'record.json').write_text(json.dumps(result, indent=2) + '\n')
        owner = module.Owner(output / 'owned', persist)
        result['commands'] = owner.commands; persist()
        def run(label, args, wall=90):
            entry, out, err = owner.run(label, args, env, P, wall=wall,
                cpu=wall * len(os.sched_getaffinity(0)), text_cap=64 << 20,
                rss_cap=48 << 30, total_cap=256 << 20, file_cap=64 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), err.read_text()[-5000:])
            return dict(stdout=out.read_text(), stderr=err.read_text())
        try:
            subprocess.run(['git', 'apply', str(patch)], cwd=P, check=True)
            header = 'include/strata/prefill/layer26_phase_audit.hpp'
            assert sha(P / header) == sha(template)
            fixture = P / 'tests/cpu/prefill_layer26_phase_audit.cpp'
            assert not fixture.exists()
            fixture.write_text(fixture_input.read_text().replace(
                '// Proposal fixture, deliberately not built/run by the implementation agent.',
                '// Finite diagnostic contract: exact bytes, bounds, completion and failure lifetime.').replace(
                '#include "layer26-phase-audit-v1/layer26_phase_audit.hpp.in"',
                '#include "strata/prefill/layer26_phase_audit.hpp"'))
            result['source_pins'] = {n: sha(P / n) for n in sorted(set(seed['source_pins']) | {header, str(fixture.relative_to(P))})}
            result['CPU_outputs'] = {}; persist()
            for sanitized in [False, True]:
                label = 'phase' + ('-asan-ubsan' if sanitized else '')
                exe = output / label
                flags = ['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if sanitized else ['-O2']
                run('compile-' + label, ['/usr/bin/c++','-std=c++20','-UNDEBUG','-Wall','-Wextra','-Werror',
                    '-I' + str(P / 'include'), *flags, str(fixture), '-o', str(exe)])
                data = output / (label + '-data'); data.mkdir(mode=0o700)
                result['CPU_outputs'][label] = run('run-' + label, [str(exe), str(data)], 30)
                result['CPU_outputs'][label]['binary_sha256'] = sha(exe)
                print('CPU PASS ' + label, flush=True); persist()
            boundary = output / 'boundary'
            run('compile-boundary', ['/usr/bin/c++','-std=c++20','-UNDEBUG','-O2','-Wall','-Wextra','-Werror',
                '-I' + str(P / 'include'), str(P / 'tests/cpu/prefill_boundary_audit.cpp'), '-o', str(boundary)])
            result['CPU_outputs']['boundary'] = run('run-boundary', [str(boundary), str(output / 'boundary-rows')], 30)
            result['host_contracts_passed'] = True; persist()
            run('build-phase-audit', ['cmake','--build',str(build),'--target','strata','--parallel','2'], 1500)
            result['binaries'] = {n: sha(build / n) for n in seed['binaries']}
            commands = json.loads((build / 'compile_commands.json').read_text())
            found = [x for x in commands if x['file'] == str(P / 'src/prefill/prefill.cpp')]
            assert len(found) == 1 and '-fsycl-id-queries-range=size_t' in found[0]['command']
            result['compile_flag_proof'] = found[0]
            result['compile_commands_sha256'] = sha(build / 'compile_commands.json')
            assert all(sha(P / n) == h for n, h in result['source_pins'].items())
            result.update(passed=True, complete=True)
        except BaseException as error:
            result['error'] = repr(error); raise
        finally:
            result['active'] = False; persist()
            (D / 'phase-audit-build-v1.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(passed=result['passed'], binaries=result.get('binaries'), GPU_executed=False)), flush=True)


if __name__ == '__main__':
    main()
