#!/usr/bin/env python3
"""Build the changed production prefill object without mutating pinned archives."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from analyze_prefill_window_profile_v2 import B, P, load_owner, require, sha


def main():
    root = B / 'main-prefill-no-workspace-build-root-v1'
    receipt = A / 'prefill-no-workspace-build-v1.json'
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not root.exists() and not receipt.exists(), 'fresh build')
        seed_path = A / 'prefill-head-release-build-v1.json'
        seed = json.loads(seed_path.read_text())
        closed = json.loads((A / 'prefill-head-lifecycle-c4096-v1.json').read_text())
        require(seed['passed'] and seed['complete'] and not seed['active'], 'production build gate')
        require(closed['passed'] and closed['complete'] and not closed['active'] and
                not closed['fresh_faults'] and not closed['driver_dump_exists'], 'closed GPU gate')
        module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for c in closed['commands']:
            require(module.closed(c), 'previous command not closed')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'previous owner present')
        changed = 'src/prefill/prefill.cpp'
        require(all(sha(P/n) == h for n, h in seed['source_pins'].items() if n != changed),
                'unexpected product source change')
        build = Path(seed['build_path'])
        require(all(sha(build/n) == h for n, h in seed['binaries'].items()), 'pinned archives changed')
        compiler = seed['compiler']
        require(sha(Path(compiler)) == seed['compiler_sha256'], 'compiler changed')
        cc = json.loads((build/'compile_commands.json').read_text())
        matches = [c for c in cc if c['file'] == str(P/changed)]
        require(len(matches) == 1, 'actual prefill resolver')
        root.mkdir(mode=0o700)
        obj = root/'prefill-no-workspace.o'
        command = shlex.split(matches[0]['command'])
        command[command.index('-o')+1] = str(obj)
        require(all(x in command for x in ['-DSTRATA_ENABLE_SYCL=1', '-fno-fast-math',
                '-ffp-contract=off', '-fsycl-id-queries-range=size_t', '-Werror']), 'compiler policy')
        libraries = [build/n for n in ['libstrata_prefill.a', 'libstrata_spec.a',
            'libstrata_engine.a', 'libstrata_kernels.a', 'libstrata_core.a',
            'libstrata_sycl_runtime.a', 'libstrata_kernels_cpu.a',
            'ggml/src/libggml-cpu.a', 'ggml/src/libggml-base.a']]
        gen = build/'CMakeFiles/strata.dir/src/program/generate.cpp.o'
        mkl = Path('/opt/intel/oneapi/mkl/2026.1/lib')
        mkl_libs = [mkl/n for n in ['libmkl_sycl_blas.so', 'libmkl_intel_ilp64.so',
                                   'libmkl_sequential.so', 'libmkl_core.so']]
        native = root/'strata'
        link = [compiler, '-O3', '-DNDEBUG', '-std=c++20', '-fsycl',
                '-fsycl-id-queries-range=size_t', '-fno-fast-math', '-ffp-contract=off',
                '-fsycl-allow-device-image-dependencies', str(gen), str(obj),
                *map(str, libraries), '-Wl,-Map='+str(root/'native.map'),
                '-Wl,-rpath='+str(mkl), *map(str, mkl_libs), '-lm', '-ldl', '-lpthread',
                '-o', str(native)]
        r = dict(schema='prefill-no-workspace-build-v1', passed=False, complete=False,
            active=True, GPU_executed=False, commands=[], recipe_sha256=sha(Path(__file__)),
            seed_receipt_sha256=sha(seed_path), compiler=compiler,
            compiler_sha256=sha(Path(compiler)), environment=seed['environment'],
            source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
            source_diff=subprocess.check_output(['git', 'diff', '--', changed], cwd=P, text=True),
            source_pins={n:sha(P/n) for n in seed['source_pins']},
            linked_inputs={str(p):sha(p) for p in [gen, *libraries, *mkl_libs]},
            removed_workspace_payload=32 << 20, removed_carve_bytes=(32 << 20)+256,
            scope='Production SYCL prefill compilation/link only. No arithmetic change; skip unused caller workspace in init/relayout/count. Runtime admission/performance not qualified.')
        def persist():
            (root/'record.json').write_text(json.dumps(r, indent=2)+'\n')
        owner = module.Owner(root/'owned', persist)
        r['commands'] = owner.commands
        persist()
        try:
            for label, argv in [('compile-prefill-no-workspace', command), ('link-native', link)]:
                c, out, err = owner.run(label, argv, seed['environment'], build, wall=600,
                    cpu=600*len(os.sched_getaffinity(0)), rss_cap=32 << 30, text_cap=8 << 20,
                    total_cap=2 << 30, file_cap=512 << 20)
                require(module.closed(c) and c['exit_code'] == 0,
                        label+': '+err.read_text()[-5000:])
                print('PASS '+label, flush=True)
            require('prefill.cpp.o' not in (root/'native.map').read_text(), 'old archive object selected')
            require(all(sha(P/n) == h for n, h in r['source_pins'].items()), 'source changed during build')
            require(all(sha(Path(n)) == h for n, h in r['linked_inputs'].items()), 'linked input changed')
            r.update(passed=True, complete=True, binary=str(native), binary_sha256=sha(native),
                     prefill_object=str(obj), prefill_object_sha256=sha(obj),
                     link_map_sha256=sha(root/'native.map'))
        except BaseException as e:
            r['error'] = repr(e)
            raise
        finally:
            r.update(active=owner.active is not None, finished_utc=module.utc())
            persist()
            receipt.write_text(json.dumps(r, indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'], binary_sha256=r.get('binary_sha256'))), flush=True)


if __name__ == '__main__':
    main()
