#!/usr/bin/env python3
"""Main-owned CPU normal/sanitized fixtures; never applies product patches."""
import fcntl
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BASE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = BASE / 'main-iq-checkpoint-cpu-contracts-root-v1'
sys.path.insert(0, str(HERE / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(PROJECT / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (BASE / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        diagnostic = json.loads((HERE / 'owned-prefill-stall-model-v1.json').read_text())
        require(diagnostic['complete'] and not diagnostic['active'], 'model diagnostic not terminal/closed')
        require(all(c.get('session_empty') and c.get('direct_child_reaped') for c in diagnostic['commands']), 'model ownership incomplete')
        require(not diagnostic['fresh_faults'] and diagnostic['boot_unchanged'] and not diagnostic['driver_dump_exists'], 'fresh device fault gate')
        require(not ROOT.exists(), 'fresh CPU artifact root required')
        ROOT.mkdir(mode=0o700)
        iq = HERE / 'parallel-iq-dequant-v1'
        ck = HERE / 'layerwise-checkpoint-capture-v2'
        sources = [*iq.glob('*.cpp'), *iq.glob('*.hpp'), *iq.glob('*.py'),
                   iq / 'proposed/include/strata/sycl/iq_parallel_decode.hpp',
                   ck / 'cpu_contract_fixture.cpp', ck / 'proposed/include/strata/program/layerwise_checkpoint.hpp',
                   *ck.glob('fixture_stubs/strata/core/*.hpp'),
                   PROJECT / 'include/strata/artifact/dequant.hpp', PROJECT / 'include/strata/artifact/iq_codebooks.hpp',
                   PROJECT / 'include/strata/kernels/f16_bits.hpp', PROJECT / 'include/strata/sycl/gguf_decode.hpp',
                   PROJECT / 'include/strata/program/conv_cache.hpp', PROJECT / 'include/strata/core/conversation_memory.hpp']
        pins = {str(p.relative_to(PROJECT)): sha(p) for p in sources}
        record = dict(schema='main-candidate-cpu-contracts-v1', active=True, passed=False,
                      GPU_executed=False, product_applied=False, commands=[], source_pins=pins,
                      recipe_sha256=sha(Path(__file__)), closed_diagnostic_sha256=sha(HERE / 'owned-prefill-stall-model-v1.json'),
                      qualification='Host candidate arithmetic/state contracts only. No SYCL compile/device/model/RESTORE/physical256K/performance proof.')

        def persist():
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(ROOT / 'owned', persist)
        record['commands'] = owner.commands
        env = dict(PATH='/usr/bin:/bin', HOME=str(ROOT), LANG='C', LC_ALL='C',
                   ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
        flags = ['-std=c++20', '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror', '-UNDEBUG']
        try:
            for name, folder, source, includes in (
                ('iq', iq, 'cpu_fixture.cpp', [iq / 'proposed/include', PROJECT / 'include']),
                ('checkpoint', ck, 'cpu_contract_fixture.cpp', [ck / 'fixture_stubs', ck / 'proposed/include', PROJECT / 'include']),
            ):
                for variant in ('normal', 'asan'):
                    exe = ROOT / (name + '-' + variant)
                    mode = ['-O2'] if variant == 'normal' else ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
                    argv = ['/usr/bin/c++', *flags, *mode, *['-I' + str(p) for p in includes], str(folder / source), '-o', str(exe)]
                    for label, command, wall in ((name + '-' + variant + '-build', argv, 180),
                                                  (name + '-' + variant + '-run', [str(exe)], 1200 if variant == 'asan' else 600)):
                        entry, out, err = owner.run(label, command, env, PROJECT, wall=wall, cpu=wall,
                                                   rss_cap=4 << 30, text_cap=8 << 20,
                                                   total_cap=128 << 20, file_cap=32 << 20)
                        require(module.closed(entry) and entry['exit_code'] == 0,
                                label + ' failed: ' + err.read_text()[-2000:])
                        if label.endswith('-run'):
                            require('PASS' in out.read_text(), 'fixture did not report PASS')
                            entry['compact_stdout'] = out.read_text()
                            entry['binary_sha256'] = sha(exe)
            require(all(sha(PROJECT / n) == h for n, h in pins.items()), 'candidate source changed during test')
            record['passed'] = True
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            persist()
            (HERE / 'candidate-cpu-contracts-v1.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=record['passed'], GPU_executed=False, product_applied=False)), flush=True)


if __name__ == '__main__':
    main()
