#!/usr/bin/env python3
"""Compile the integrated census and a separately owned timing diagnostic; no GPU."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BASE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(HERE / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(PROJECT / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    root = BASE / 'main-next-diagnostics-build-root-v1'
    with (BASE / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((HERE / 'parallel-iq-timing-wide-v2.json').read_text())
        require(not previous['active'] and all(c.get('direct_child_reaped') and c.get('session_empty')
                for c in previous['commands']), 'previous native not retired')
        host = json.loads((HERE / 'layermajor-census-host-v1.json').read_text())
        require(host['passed'] and host['complete'] and not host['active'], 'census host gate')
        old = json.loads((HERE / 'parallel-iq-component-build-v1.json').read_text())
        require(all(sha(PROJECT / n) == h for n, h in old['source_pins'].items()), 'IQ source changed')
        require(all(sha(Path(n)) == h for n, h in old['linked_archives'].items()), 'baseline library changed')
        require(not root.exists(), 'fresh build root required')
        root.mkdir(mode=0o700)
        timing = HERE / 'iq-timing-progress-probe-v3/timing.cpp'
        files = [PROJECT / n for n in ('src/prefill/prefill.cpp', 'src/prefill/layer_major.inc',
                                      'include/strata/prefill/layermajor_shape_census.hpp')]
        require(sha(files[-1]) == sha(HERE / 'layermajor-shape-census-v1/layermajor_shape_census.hpp.in'),
                'actual product header differs from host-tested header')
        mkl = Path('/opt/intel/oneapi/mkl/2026.1')
        libraries = [mkl / ('lib/' + n) for n in ('libmkl_sycl_blas.so', 'libmkl_intel_ilp64.so', 'libmkl_sequential.so', 'libmkl_core.so')]
        record = dict(schema='main-next-diagnostics-build-v1', active=True, passed=False, complete=False,
                      GPU_executed=False, commands=[], recipe_sha256=sha(Path(__file__)),
                      source_pins={**old['source_pins'], **{str(p.relative_to(PROJECT)): sha(p) for p in [*files, timing]}},
                      linked_archives=old['linked_archives'], linked_mkl={str(p): sha(p) for p in libraries},
                      compiler=old['compiler'], compiler_sha256=sha(Path(old['compiler'])),
                      census_product_applied=True, performance_adopted=False)

        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        baseline = BASE / 'main-primary-native-build-root-v1/build'
        database = baseline / 'compile_commands.json'
        entries = [c for c in json.loads(database.read_text()) if c['file'] == str(files[0])]
        require(len(entries) == 1, 'actual SYCL source resolver')
        compile_command = shlex.split(entries[0]['command'])
        compile_command = [c for c in compile_command if not c.startswith('-DSTRATA_VERSION=')]
        obj = root / 'prefill-census.o'
        compile_command[compile_command.index('-o')+1] = str(obj)
        require(compile_command[0] == old['compiler'], 'same compiler required')
        record['compile_database_sha256'] = sha(database)
        iq = HERE / 'parallel-iq-dequant-v1'
        exe = root / 'iq_timing_progress'
        timing_command = [old['compiler'], '-std=c++20', '-O3', '-DNDEBUG', '-DSTRATA_ENABLE_SYCL=1', '-DMKL_ILP64',
                          '-fsycl', '-fsycl-id-queries-range=size_t', '-fno-fast-math', '-ffp-contract=off',
                          '-Wall', '-Wextra', '-Werror', '-I'+str(iq/'proposed/include'),
                          '-I'+str(PROJECT/'include'), '-I'+str(PROJECT/'include/strata/sycl/compat'),
                          '-isystem', str(mkl/'include'), str(timing), str(iq/'proposed/src/kernels/sycl/iq.cpp'),
                          *old['linked_archives'], '-fsycl-allow-device-image-dependencies',
                          '-Wl,-Map='+str(root/'link.map'), '-Wl,-rpath='+str(mkl/'lib'),
                          *map(str,libraries), '-lm', '-ldl', '-lpthread', '-o', str(exe)]
        try:
            for label, argv, cwd in [('compile-actual-census', compile_command, baseline),
                                      ('build-timing-progress', timing_command, PROJECT)]:
                c, out, err = owner.run(label, argv, old['environment'], cwd, wall=600,
                                       cpu=600*len(os.sched_getaffinity(0)), rss_cap=32<<30,
                                       text_cap=8<<20, total_cap=1<<30, file_cap=512<<20)
                require(module.closed(c) and c['exit_code']==0, label+' failed: '+err.read_text()[-4000:])
            require('iq.cpp.o' not in (root/'link.map').read_text(), 'old IQ archive member selected')
            require(all(sha(PROJECT/n)==h for n,h in record['source_pins'].items()), 'source changed')
            record.update(passed=True, complete=True, census_object=str(obj), census_object_sha256=sha(obj),
                          binary=str(exe), binary_sha256=sha(exe), environment=old['environment'],
                          link_map_sha256=sha(root/'link.map'))
        except BaseException as error:
            record['error']=repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            persist()
            (HERE/'next-diagnostics-build-v1.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(passed=record['passed'],GPU_executed=False)),flush=True)


if __name__=='__main__':
    main()
