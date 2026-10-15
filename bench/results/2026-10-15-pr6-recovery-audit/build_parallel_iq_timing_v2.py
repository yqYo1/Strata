#!/usr/bin/env python3
"""Main-owned strict same-stack timing fixture build, no GPU/product edits."""
import fcntl
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BASE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = BASE / 'main-parallel-iq-timing-build-root-v2'
sys.path.insert(0, str(HERE / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(PROJECT / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (BASE / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = json.loads((HERE / 'parallel-iq-component-full-v1.json').read_text())
        require(gate['passed'] and gate['complete'] and not gate['active'] and not gate['fresh_faults'], 'full device-array gate')
        old = json.loads((HERE / 'parallel-iq-component-build-v1.json').read_text())
        require(all(sha(PROJECT / n) == h for n, h in old['source_pins'].items()), 'candidate source changed')
        require(all(sha(Path(n)) == h for n, h in old['linked_archives'].items()), 'baseline archive changed')
        require(not ROOT.exists(), 'fresh timing build root required')
        ROOT.mkdir(mode=0o700)
        source = HERE / 'parallel-iq-gemm-timing-v2/timing.cpp'
        iq = HERE / 'parallel-iq-dequant-v1'
        mkl = Path('/opt/intel/oneapi/mkl/2026.1')
        libraries = [mkl / ('lib/' + n) for n in ('libmkl_sycl_blas.so', 'libmkl_intel_ilp64.so', 'libmkl_sequential.so', 'libmkl_core.so')]
        env = old['environment']
        exe = ROOT / 'parallel_iq_timing'
        record = dict(schema='main-parallel-iq-timing-build-v2', active=True, passed=False, complete=False,
                      GPU_executed=False, commands=[], recipe_sha256=sha(Path(__file__)),
                      source_pins={**old['source_pins'], str(source.relative_to(PROJECT)): sha(source)},
                      linked_archives=old['linked_archives'], linked_mkl={str(p): sha(p) for p in libraries},
                      compiler=old['compiler'], compiler_sha256=sha(Path(old['compiler'])),
                      product_applied=False, performance_qualified=False)

        def persist():
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(ROOT / 'owned', persist)
        record['commands'] = owner.commands
        argv = [old['compiler'], '-std=c++20', '-O3', '-DNDEBUG', '-DSTRATA_ENABLE_SYCL=1', '-DMKL_ILP64',
                '-fsycl', '-fsycl-id-queries-range=size_t', '-fno-fast-math', '-ffp-contract=off',
                '-Wall', '-Wextra', '-Werror', '-I' + str(iq / 'proposed/include'),
                '-I' + str(PROJECT / 'include'), '-I' + str(PROJECT / 'include/strata/sycl/compat'),
                '-isystem', str(mkl / 'include'), str(source), str(iq / 'proposed/src/kernels/sycl/iq.cpp'),
                *old['linked_archives'], '-fsycl-allow-device-image-dependencies',
                '-Wl,-Map=' + str(ROOT / 'link.map'), '-Wl,-rpath=' + str(mkl / 'lib'),
                *map(str, libraries), '-lm', '-ldl', '-lpthread', '-o', str(exe)]
        try:
            entry, out, err = owner.run('build-resident-timing', argv, env, PROJECT, wall=600,
                                        cpu=600 * len(os.sched_getaffinity(0)), rss_cap=32 << 30,
                                        text_cap=64 << 20, total_cap=512 << 20, file_cap=256 << 20)
            require(module.closed(entry) and entry['exit_code'] == 0, 'timing build failed: ' + err.read_text()[-3500:])
            link_map = (ROOT / 'link.map').read_text()
            require('iq.cpp.o' not in link_map, 'old IQ archive member unexpectedly selected')
            record.update(passed=True, complete=True, binary=str(exe), binary_sha256=sha(exe),
                          link_map_sha256=sha(ROOT / 'link.map'), old_iq_archive_member_selected=False)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            persist()
            (HERE / 'parallel-iq-timing-build-v2.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=record['passed'], GPU_executed=False)), flush=True)


if __name__ == '__main__':
    main()
