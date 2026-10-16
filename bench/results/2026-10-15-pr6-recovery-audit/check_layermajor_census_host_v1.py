#!/usr/bin/env python3
"""Main-owned host-only verification of the actual proposed census header."""
import fcntl
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BASE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(HERE / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(PROJECT / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    proposal = HERE / 'layermajor-shape-census-v1'
    root = BASE / 'main-layermajor-census-host-root-v1'
    with (BASE / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = json.loads((HERE / 'parallel-iq-timing-wide-v2.json').read_text())
        require(not prior['active'] and all(c.get('direct_child_reaped') and c.get('session_empty')
                and not c.get('survivors') for c in prior['commands']), 'prior owned job not retired')
        require(not root.exists(), 'fresh host root required')
        root.mkdir(mode=0o700)
        sources = [proposal / 'cpu_fixture.cpp', proposal / 'layermajor_shape_census.hpp.in',
                   HERE / 'layermajor-census-output-io-fixture-v1.cpp']
        pins = {str(p.relative_to(PROJECT)): sha(p) for p in sources}
        record = dict(schema='main-layermajor-census-host-v1', active=True, passed=False,
                      complete=False, GPU_executed=False, product_applied=False, commands=[],
                      source_pins=pins, recipe_sha256=sha(Path(__file__)),
                      prior_terminal_sha256=sha(HERE / 'parallel-iq-timing-wide-v2.json'),
                      prior_GPU_fault='Unresolved. This is CPU-only and does not establish GPU recovery.',
                      scope='Actual header assertions, normal and ASan/UBSan; synthetic 262144 geometry only, no physical model qualification.')

        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        env = dict(PATH='/usr/bin:/bin', HOME=str(root), LANG='C', LC_ALL='C',
                   ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
        try:
            for name, source in [('contract', sources[0]), ('output-io', sources[2])]:
                for variant in ('normal', 'asan'):
                    exe = root / (name + '-' + variant)
                    flags = ['-O2'] if variant == 'normal' else ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
                    argv = ['/usr/bin/c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-UNDEBUG',
                            *flags, '-I' + str(proposal), str(source), '-o', str(exe)]
                    for label, command, wall in [(name+'-'+variant+'-build', argv, 120),
                                                  (name+'-'+variant+'-run', [str(exe)], 120)]:
                        c, out, err = owner.run(label, command, env, PROJECT, wall=wall, cpu=wall,
                                               rss_cap=2 << 30, text_cap=1 << 20, total_cap=64 << 20, file_cap=16 << 20)
                        require(module.closed(c) and c['exit_code'] == 0, label+' failed: '+err.read_text()[-2000:])
                        if label.endswith('-run'):
                            c['binary_sha256'] = sha(exe)
                            c['compact_stdout'] = out.read_text()
            require(all(sha(PROJECT / n) == h for n, h in pins.items()), 'source changed during host check')
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            persist()
            (HERE / 'layermajor-census-host-v1.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=record['passed'], GPU_executed=False)), flush=True)


if __name__ == '__main__':
    main()
