#!/usr/bin/env python3
"""Main-only source integration, actual builds and CPU production-planner gates.

Run after the current clean GPU cohort closes. GPU execution is a separate gate.
"""
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
    root = B/'main-prefill-scratch-alias-build-root-v1'
    receipt = A/'prefill-scratch-alias-build-host-v1.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not root.exists() and not receipt.exists(), 'fresh gate')
        seed_path = A/'prefill-no-workspace-build-v1.json'
        seed = json.loads(seed_path.read_text())
        closed_path = A/'prefill-clean-c8192-r0-v1.json'
        closed = json.loads(closed_path.read_text())
        require(seed['passed'] and seed['complete'] and not seed['active'], 'production build')
        require(closed['passed'] and closed['complete'] and not closed['active'] and
                not closed['fresh_faults'] and closed['boot_unchanged'] and
                not closed['driver_dump_exists'], 'closed clean GPU cohort')
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for c in closed['commands']+seed['commands']:
            require(module.closed(c), 'previous command not closed')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'previous owner present')
        require(all(sha(P/n) == h for n, h in seed['source_pins'].items()), 'unchanged reference source')
        require(all(sha(Path(n)) == h for n, h in seed['linked_inputs'].items()), 'unchanged reference libraries')
        proposal = A/'prefill-scratch-alias-v1'
        manifest = json.loads((proposal/'sha256-manifest.json').read_text())
        identities = {item['path']:item for item in manifest['files']}
        require(len(identities) == len(manifest['files']), 'unique proposal artifacts')
        for item in manifest['files']:
            # The committed base revision plus exact patch reconstruct these
            # mirrors. Verify the actual input/output bytes below instead.
            if item['path'].startswith(('base/', 'proposed/')):
                continue
            path = proposal/item['path']
            require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'proposal identity')
        changed = 'src/prefill/prefill.cpp'
        require(sha(P/changed) == identities['base/'+changed]['sha256'] == seed['source_pins'][changed], 'exact proposal base')
        patch = proposal/'source.diff'
        require(sha(patch) == '4b8638e596bb704c98ee3c58ccea9782eda70fd2883daeca9a77e6ff52ed318f', 'reviewed patch')
        review = A/'research-prefill-scratch-alias-review-round688-v1.txt'
        require(sha(review) == '2344a34be3b8c537ebdadb4a2e6a4b9bfd9af97fd86b2d2f96f74d52c62ef960', 'independent review')
        base_receipt = json.loads((A/'prefill-head-release-build-v1.json').read_text())
        build = Path(base_receipt['build_path'])
        compiler = seed['compiler']
        require(sha(Path(compiler)) == seed['compiler_sha256'], 'compiler identity')
        commands = json.loads((build/'compile_commands.json').read_text())
        cc = [c for c in commands if c['file'] == str(P/changed)]
        require(len(cc) == 1, 'actual source resolver')
        libraries = [build/n for n in ['libstrata_prefill.a', 'libstrata_spec.a', 'libstrata_engine.a',
            'libstrata_kernels.a', 'libstrata_core.a', 'libstrata_sycl_runtime.a',
            'libstrata_kernels_cpu.a', 'ggml/src/libggml-cpu.a', 'ggml/src/libggml-base.a']]
        gen = build/'CMakeFiles/strata.dir/src/program/generate.cpp.o'
        mkl = Path('/opt/intel/oneapi/mkl/2026.1/lib')
        mkl_libs = [mkl/n for n in ['libmkl_sycl_blas.so', 'libmkl_intel_ilp64.so',
                                   'libmkl_sequential.so', 'libmkl_core.so']]
        fixture = A/'prefill_scratch_prefix_fixture_v1.cpp'
        root.mkdir(mode=0o700)
        r = dict(schema='prefill-scratch-alias-build-host-v1', passed=False, complete=False,
            active=True, applied=False, GPU_executed=False, commands=[], host_results=[],
            recipe_sha256=sha(Path(__file__)), seed_receipt_sha256=sha(seed_path),
            clean_reference_sha256=sha(closed_path), proposal_sha256=sha(patch),
            independent_review_sha256=sha(review), fixture_sha256=sha(fixture),
            compiler=compiler, compiler_sha256=sha(Path(compiler)), environment=seed['environment'],
            source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
            linked_inputs={str(p):sha(p) for p in [gen, *libraries, *mkl_libs]},
            default_enabled=False, quality_qualified=False, performance_qualified=False,
            restore_qualified=False, physical_262144_qualified=False,
            scope='Actual SYCL product compile/link plus host Alloc/prefix overlap/count/end-canary and immutable strict-flag checks, normal and host sanitizers. No GPU allocations/kernels, model arithmetic, PLE, callback, session, capacity or speed qualification.')
        def persist():
            (root/'record.json').write_text(json.dumps(r, indent=2)+'\n')
        owner = module.Owner(root/'owned', persist)
        r['commands'] = owner.commands
        persist()
        def run(label, argv, env=None, wall=600):
            c, out, err = owner.run(label, argv, env or seed['environment'], P, wall=wall,
                cpu=wall*len(os.sched_getaffinity(0)), rss_cap=24 << 30, text_cap=8 << 20,
                total_cap=2 << 30, file_cap=512 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label+': '+err.read_text()[-5000:])
            print('PASS '+label, flush=True)
            return out.read_text()
        def compile_command(source, obj, sanitized=False):
            argv = shlex.split(cc[0]['command'])
            argv[argv.index('-o')+1] = str(obj)
            argv[argv.index('-c')+1] = str(source)
            argv += ['-I'+str(P/'src/prefill')]
            if sanitized:
                argv += ['-O1', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
            return argv
        def link_command(obj, binary, native=False, sanitized=False):
            argv = [compiler, '-O3', '-DNDEBUG', '-std=c++20', '-fsycl',
                '-fsycl-id-queries-range=size_t', '-fno-fast-math', '-ffp-contract=off',
                '-fsycl-allow-device-image-dependencies']
            if sanitized:
                argv += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
            if native:
                argv += [str(gen)]
            return [*argv, str(obj), *map(str, libraries), '-Wl,-rpath='+str(mkl),
                *map(str, mkl_libs), '-Wl,-Map='+str(binary)+'.map', '-lm', '-ldl', '-lpthread',
                '-o', str(binary)]
        try:
            run('check-reviewed-alias-patch', ['git', 'apply', '--check', str(patch)], wall=60)
            run('apply-reviewed-alias-patch', ['git', 'apply', str(patch)], wall=60)
            r['applied'] = True
            require(sha(P/changed) == identities['proposed/'+changed]['sha256'], 'exact proposal output')
            r['source_pins'] = {n:sha(P/n) for n in seed['source_pins']}
            persist()
            obj = root/'prefill-scratch-alias.o'
            native = root/'strata'
            run('compile-alias-product', compile_command(P/changed, obj))
            run('link-alias-native', link_command(obj, native, native=True))
            require('prefill.cpp.o' not in Path(str(native)+'.map').read_text(), 'old production archive selected')
            r.update(binary=str(native), binary_sha256=sha(native), prefill_object=str(obj),
                prefill_object_sha256=sha(obj), native_link_map_sha256=sha(Path(str(native)+'.map')))
            for sanitized in (False, True):
                label = 'sanitized' if sanitized else 'normal'
                fo = root/('prefix-'+label+'.o')
                exe = root/('prefix-'+label)
                run('compile-prefix-'+label, compile_command(fixture, fo, sanitized))
                run('link-prefix-'+label, link_command(fo, exe, sanitized=sanitized))
                require('prefill.cpp.o' not in Path(str(exe)+'.map').read_text(), 'old fixture archive selected')
                arms = [('1', g, b, 1) for g in ('0', '1') for b in ('0', '1', '2')]
                arms += [(None, '0', '0', 0), ('0', '0', '0', 0)]
                arms += [(v, '0', '0', -1) for v in ('', '01', 'true')]
                for i, (flag, gr, bf, expected) in enumerate(arms):
                    env = dict(seed['environment'], STRATA_GR_UNFUSED=gr, STRATA_PREFILL_BF16X2=bf,
                        STRATA_PREFILL_MMQ='0', ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                        UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
                    env.pop('STRATA_SYCL_PREFILL_SCRATCH_ALIAS', None)
                    if flag is not None:
                        env['STRATA_SYCL_PREFILL_SCRATCH_ALIAS'] = flag
                    output = run('prefix-'+label+'-'+str(i), [str(exe), str(expected)], env, wall=90)
                    marker = 'prefix_flag PASS rejected_before_geometry' if expected < 0 else 'prefix_layout PASS cases=10'
                    require(marker in output, 'complete production prefix checks')
                    r['host_results'].append(dict(sanitized=sanitized, flag=flag, gr_unfused=gr,
                        bf16x2=bf, expected_mode=expected, passed=True, stdout=output,
                        binary_sha256=sha(exe)))
                    persist()
            require(all(sha(P/n) == h for n, h in r['source_pins'].items()), 'source changed during gate')
            require(all(sha(Path(n)) == h for n, h in r['linked_inputs'].items()), 'libraries changed during gate')
            require(sha(fixture) == r['fixture_sha256'], 'fixture changed during gate')
            r.update(passed=True, complete=True)
        except BaseException as e:
            r['error'] = repr(e)
            raise
        finally:
            r.update(active=owner.active is not None, finished_utc=module.utc())
            persist()
            receipt.write_text(json.dumps(r, indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'], host_processes=len(r['host_results']),
                              binary_sha256=r.get('binary_sha256'))), flush=True)


if __name__ == '__main__':
    main()
