#!/usr/bin/env python3
"""Build proposed public IQ entry points separately; preserve native92 binary."""
import fcntl
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BASE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = BASE / 'main-parallel-iq-component-build-root-v1'
sys.path.insert(0, str(HERE / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(PROJECT / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (BASE / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        diagnostic = json.loads((HERE / 'owned-prefill-stall-model-v1.json').read_text())
        require(diagnostic['complete'] and not diagnostic['active'] and not diagnostic['fresh_faults']
                and diagnostic['boot_unchanged'] and not diagnostic['driver_dump_exists'], 'diagnostic fault/closure gate')
        cpu = json.loads((HERE / 'candidate-cpu-contracts-v1.json').read_text())
        for label in ('iq-normal-build', 'iq-normal-run', 'iq-asan-build', 'iq-asan-run'):
            rows = [c for c in cpu['commands'] if c['label'] == label]
            require(len(rows) == 1 and module.closed(rows[0]) and rows[0]['exit_code'] == 0, label + ' not pass')
        proposal = HERE / 'parallel-iq-dequant-v1'
        pins = json.loads((proposal / 'source-pins.json').read_text())
        require(sha(proposal / 'source.diff') == pins['source_diff_sha256'], 'candidate diff changed')
        for name, pin in pins['changed'].items():
            require(sha(proposal / pin['proposal_body']) == pin['new_sha256'], 'candidate source changed')
            if pin['old_sha256'] is not None:
                require(sha(PROJECT / name) == pin['old_sha256'], 'product source changed/applied')
        build = json.loads((HERE / 'cpu-gu-reuse-build-v3.json').read_text())
        native_build = Path(build['build_path'])
        native = native_build / 'strata'
        require(sha(native) == build['binaries']['strata'], 'baseline native changed')
        archives = [native_build / n for n in ('libstrata_kernels.a', 'libstrata_core.a', 'libstrata_sycl_runtime.a')]
        require(not ROOT.exists(), 'fresh standalone build root required')
        ROOT.mkdir(mode=0o700)
        seed = build['environment']
        env = {k: seed[k] for k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CPLUS_INCLUDE_PATH', 'OMP_NUM_THREADS', 'MKLROOT') if k in seed}
        compiler = Path(env['PATH'].split(':')[0]) / 'clang++'
        sources = [proposal / 'gpu_fixture.cpp', proposal / 'fixture_common.hpp',
                   proposal / 'proposed/src/kernels/sycl/iq.cpp', proposal / 'proposed/include/strata/sycl/iq_parallel_decode.hpp']
        record = dict(schema='main-parallel-iq-component-build-v1', active=True, passed=False, complete=False,
                      GPU_executed=False, product_applied=False, commands=[], environment=env,
                      source_pins={str(p.relative_to(PROJECT)): sha(p) for p in sources},
                      linked_archives={str(p): sha(p) for p in archives}, baseline_binary_sha256=sha(native),
                      compiler=str(compiler), compiler_sha256=sha(compiler), recipe_sha256=sha(Path(__file__)),
                      host_fixture_sha256=sha(HERE / 'candidate-cpu-contracts-v1.json'),
                      scope='Standalone actual candidate IQ public functions plus existing runtime/kernels; no engine source apply/rebuild or performance adoption.')

        def persist():
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(ROOT / 'owned', persist)
        record['commands'] = owner.commands
        exe = ROOT / 'parallel_iq_fixture'
        argv = [str(compiler), '-std=c++20', '-O3', '-DNDEBUG', '-DSTRATA_ENABLE_SYCL=1',
                '-fsycl', '-fsycl-id-queries-range=size_t', '-fno-fast-math', '-ffp-contract=off',
                '-Wall', '-Wextra', '-Werror', '-I' + str(proposal / 'proposed/include'),
                '-I' + str(PROJECT / 'include'), '-I' + str(PROJECT / 'include/strata/sycl/compat'),
                str(proposal / 'gpu_fixture.cpp'), str(proposal / 'proposed/src/kernels/sycl/iq.cpp'),
                *map(str, archives), '-fsycl-allow-device-image-dependencies', '-ldl', '-lpthread', '-o', str(exe)]
        try:
            entry, out, err = owner.run('build-standalone-iq', argv, env, PROJECT, wall=600,
                                        cpu=600 * len(os.sched_getaffinity(0)), rss_cap=32 << 30,
                                        text_cap=64 << 20, total_cap=512 << 20, file_cap=256 << 20)
            require(module.closed(entry) and entry['exit_code'] == 0, 'component compile/link failed: ' + err.read_text()[-3500:])
            require(sha(native) == record['baseline_binary_sha256'], 'baseline native overwritten')
            require(all(sha(Path(p)) == h for p, h in record['linked_archives'].items()), 'linked archive changed')
            record.update(passed=True, complete=True, binary=str(exe), binary_sha256=sha(exe))
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            persist()
            (HERE / 'parallel-iq-component-build-v1.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=record['passed'], GPU_executed=False, product_applied=False)), flush=True)


if __name__ == '__main__':
    main()
