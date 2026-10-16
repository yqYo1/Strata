#!/usr/bin/env python3
"""Main's serialized CPU qualification; never overlaps a measurement owner."""
import fcntl
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_path = A / 'selective-iq-clean-bench-full-0-v1.json'
        previous = json.loads(previous_path.read_text())
        # The historical producer stores completion timestamps on its commands,
        # not at receipt top level. Preserve that original receipt unchanged.
        require(not previous['active'] and previous.get('commands'), 'full OFF has not closed')
        for c in previous['commands']:
            require(c.get('finished_utc') and c.get('direct_child_reaped') and
                    c.get('session_empty') and not c.get('survivors'), 'earlier owner not retired')
            for identity in c.get('owners', []):
                current = module.identity(identity['pid'])
                require(current is None or current['start_ticks'] != identity['start_ticks'], 'earlier identity still present')
        proposal = A / 'layermajor-wave-census-v2'
        inputs = {}
        for manifest in ('pins.json', 'artifact-sha256.json'):
            inputs[str((proposal / manifest).relative_to(P))] = sha(proposal / manifest)
        for rel, pin in json.loads((proposal / 'pins.json').read_text()).items():
            require(sha(P / rel) == pin, 'original proposal input changed: ' + rel)
            inputs[rel] = pin
        for rel, pin in json.loads((proposal / 'artifact-sha256.json').read_text()).items():
            require(sha(proposal / rel) == pin, 'proposed artifact changed: ' + rel)
            inputs[str((proposal / rel).relative_to(P))] = pin
        for path in [Path(__file__), A / 'wave_census_serialization_fixture_v1.cpp',
                     A / 'check_wave_census_serialization_v1.py',
                     P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py',
                     A / 'owned-stall-stack-probe-v1/owned_stack_probe.py']:
            inputs[str(path.relative_to(P))] = sha(path)
        root = B / 'main-wave-census-host-root-v2'
        receipt = A / 'wave-census-host-v2.json'
        require(not root.exists() and not receipt.exists(), 'fresh host qualification root required')
        root.mkdir(mode=0o700)
        compiler = Path('/usr/bin/c++').resolve()
        record = dict(schema='main-wave-census-host-v2', active=True, passed=False, complete=False,
            GPU_executed=False, product_applied=False, performance_qualified=False, quality_qualified=False,
            physical_262144_qualified=False, restore_qualified=False, source_pins=inputs,
            compiler=str(compiler), compiler_sha256=sha(compiler), commands=[],
            prior_terminal_sha256=sha(previous_path),
            scope='Actual proposed observer header, independent CPU references, normal/ASan/UBSan, serialized fields and error paths. Synthetic geometry is not physical-context qualification.')
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        env = dict(PATH='/usr/bin:/bin', HOME=str(root), LANG='C.UTF-8', LC_ALL='C.UTF-8',
            PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
        def run(label, argv, wall=180):
            c, out, err = owner.run(label, argv, env, P, wall=wall, cpu=wall, rss_cap=2 << 30,
                text_cap=4 << 20, total_cap=256 << 20, file_cap=32 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label + ' failed: ' + err.read_text()[-3000:])
            return c, out, err
        persist()
        try:
            for variant in ('normal', 'asan'):
                flags = ['-O2'] if variant == 'normal' else ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
                for name, source in [('contract', proposal / 'cpu_fixture.cpp'), ('serialization', A / 'wave_census_serialization_fixture_v1.cpp')]:
                    exe = root / (name + '-' + variant)
                    run(name + '-' + variant + '-build', ['/usr/bin/c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-UNDEBUG',
                        *flags, str(source), '-o', str(exe)])
                    c, out, err = run(name + '-' + variant + '-run', [str(exe)])
                    c['binary_sha256'] = sha(exe)
                    if name == 'serialization':
                        _, verified, _ = run('verify-serialization-' + variant,
                            ['/usr/bin/python3', '-B', str(A / 'check_wave_census_serialization_v1.py'), str(out)])
                        record.setdefault('serialization_checks', {})[variant] = json.loads(verified.read_text())
                persist()
            _, out, err = run('validator-tests', ['/usr/bin/python3', '-B', str(proposal / 'test_validator.py')], 60)
            match = re.search(r'Ran (\d+) tests? in .*\n\nOK\s*$', err.read_text())
            require(match and int(match[1]) == 4, 'validator test count')
            record['validator_tests'] = int(match[1])
            require(all(sha(P / rel) == pin for rel, pin in inputs.items()), 'source changed during host checks')
            require(sha(compiler) == record['compiler_sha256'], 'compiler changed during host checks')
            require(record['serialization_checks']['normal'] == record['serialization_checks']['asan'], 'normal/sanitizer serialization differs')
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):
                record.update(passed=False, complete=False)
            persist()
            with receipt.open('x') as stream:
                json.dump(record, stream, indent=2)
                stream.write('\n')
        print(json.dumps(dict(passed=record['passed'], GPU_executed=False)))


if __name__ == '__main__':
    main()
