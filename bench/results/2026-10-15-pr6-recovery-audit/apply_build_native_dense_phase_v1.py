#!/usr/bin/env python3
"""Main-owned native dense bank integration after all fixed-source clean runs.

No GPU execution. Keep every previous linked binary/archive immutable.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from analyze_prefill_window_profile_v2 import B, P, load_owner, require, sha


def main():
    root = B/'main-native-dense-phase-build-root-v1'
    target = A/'native-dense-phase-build-v1.json'
    proposal = A/'prefill-native-dense-phase-v2'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not root.exists() and not target.exists(), 'fresh build root')
        comparison_path = A/'qsa-device-batch-clean-comparison-v1.json'
        comparison = json.loads(comparison_path.read_text())
        require(comparison['passed'] and comparison['complete'] and not comparison['active']
            and len(comparison['samples']) == 6, 'six independent samples compared')
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for name, digest in comparison['references'].items():
            path = A/name
            require(sha(path) == digest, 'closed sample identity')
            prior = json.loads(path.read_text())
            require(prior['passed'] and prior['complete'] and not prior['active']
                and not prior['fresh_faults'] and prior['boot_unchanged']
                and not prior['driver_dump_exists'], 'healthy preceding sample')
            for c in prior['commands']:
                require(module.closed(c), 'preceding owner closed')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'preceding owner retired')
            require(all(sha(P/n) == h for n,h in prior['source_pins'].items()), 'unchanged measurement source')
        manifest_path = proposal/'proposal-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        patch = proposal/'source.diff'
        require(sha(patch) == manifest['files']['source.diff'] ==
            '03a385d0ef0b771d4aa89df18090fe7c602bd81eaa14c9bfa74a8e092f0caffd', 'reviewed complete patch')
        for name,h in manifest['base_files'].items():
            require(not (P/name).exists() if h is None else sha(P/name) == h, 'exact patch base '+name)
        for name in ('component_fixture.cpp', 'allocator_fault_shim.hpp'):
            require(sha(proposal/name) == manifest['files'][name], 'reviewed fixture '+name)
        host_path = A/'native-dense-phase-host-v2.json'
        host = json.loads(host_path.read_text())
        require(host['passed'] and host['complete'] and not host['active'], 'closed host retirement gate')
        reviews = [A/n for n in ('native-dense-phase-main-review.ja.md',
            'research-native-dense-phase-owner-review-round703-v1.txt',
            'research-native-dense-phase-reader-review-round704-v1.txt')]
        seed = json.loads((B/'main-primary-native-build-root-v1/record.json').read_text())
        require(seed['passed'] and seed['complete'] and not seed['active'], 'actual configure recipe')
        env = dict(seed['environment'])
        compiler = Path(seed['compiler']['path'])
        require(sha(compiler) == seed['compiler']['sha256'], 'unchanged compiler')
        require(not any('TOKEN' in k or 'API_KEY' in k or k == 'LD_PRELOAD'
            or k.startswith(('UNITRACE_', 'PTI_')) for k in env), 'bounded build environment')
        configure = list(next(c['argv'] for c in seed['commands'] if c['label'] == 'configure-primary-native'))
        build = root/'build'
        configure[configure.index('-B')+1] = str(build)
        configure += ['-DSTRATA_PREFILL_MMQ=OFF', '-DSTRATA_BUILD_PREFILL_HOST_TESTS=ON']
        root.mkdir(mode=0o700)
        record = dict(passed=False, complete=False, active=True, applied=False, GPU_executed=False,
            recipe_sha256=sha(Path(__file__)), commands=[], environment=env,
            references={p.name:sha(p) for p in [comparison_path, manifest_path, host_path]+reviews},
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=P, text=True).strip(),
            patch_sha256=sha(patch), compiler=str(compiler), compiler_sha256=sha(compiler),
            configure_argv=configure, build_path=str(build), build_byte_budget=3<<30,
            default_enabled=False, quality_qualified=False, performance_qualified=False,
            physical_model_262144_qualified=False, restore_qualified=False,
            build_consumer='Main actual 300-matrix payload/restore and fault gates, then native generator phase qualification',
            review_point='After component and generator comparisons; retire superseded binaries and duplicate successful logs.',
            scope='Actual SYCL generator and linked native-bank fixture compile/link only; no runtime/VRAM recovery or model proof.')
        def persist():
            (root/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(root/'owned', persist)
        record['commands'] = owner.commands
        persist()
        def run(label, argv, wall=90):
            c,out,err = owner.run(label, argv, env, P, wall=wall,
                cpu=wall*len(os.sched_getaffinity(0)), rss_cap=32<<30,
                text_cap=8<<20, total_cap=3<<30, file_cap=256<<20)
            require(module.closed(c) and c['exit_code'] == 0,
                label+': '+err.read_text()[-4000:]+'\n'+out.read_text()[-2500:])
            print('PASS '+label, flush=True)
        try:
            run('check-native-dense-patch', ['git','apply','--check',str(patch)])
            run('apply-native-dense-patch', ['git','apply',str(patch)])
            record['applied'] = True
            for name,h in manifest['proposed_files'].items():
                require(sha(P/name) == h, 'exact proposed file '+name)
            fixture = P/'tests/sycl/native_dense_bank.cpp'
            shim = P/'tests/sycl/native_dense_bank_allocator_shim.hpp'
            fault = P/'tests/sycl/native_dense_bank_fault_object.cpp'
            require(not any(p.exists() for p in (fixture,shim,fault)), 'fresh manual fixture sources')
            fixture.write_bytes((proposal/'component_fixture.cpp').read_bytes())
            shim.write_bytes((proposal/'allocator_fault_shim.hpp').read_bytes())
            fault.write_text('// Only this manual fault-test translation unit redirects allocation.\n'
                '#include "native_dense_bank_allocator_shim.hpp"\n'
                '#include "../../src/core/native_dense.cpp"\n')
            cmake = P/'cmake/sycl_backend.cmake'
            anchor = '  add_executable(sycl_decode_attention_test tests/sycl/decode_attention.cpp)\n'
            old = cmake.read_text()
            require(old.count(anchor) == 1, 'manual fixture CMake insertion seam')
            extra = '''  # Manual real-model ownership fixtures; Main serially supervises GPU execution.
  add_executable(sycl_native_dense_bank_test EXCLUDE_FROM_ALL tests/sycl/native_dense_bank.cpp)
  target_link_libraries(sycl_native_dense_bank_test PRIVATE strata_engine)
  target_compile_options(sycl_native_dense_bank_test PRIVATE -fno-fast-math -ffp-contract=off)
  add_executable(sycl_native_dense_bank_fault_test EXCLUDE_FROM_ALL
    tests/sycl/native_dense_bank.cpp tests/sycl/native_dense_bank_fault_object.cpp)
  target_link_libraries(sycl_native_dense_bank_fault_test PRIVATE strata_engine)
  target_compile_definitions(sycl_native_dense_bank_fault_test PRIVATE STRATA_DENSE_BANK_ALLOC_FAULT_FIXTURE=1)
  target_compile_options(sycl_native_dense_bank_fault_test PRIVATE -fno-fast-math -ffp-contract=off)
'''
            cmake.write_text(old.replace(anchor, extra+anchor))
            names = subprocess.check_output(['git','ls-files','--','src','include','cmake',
                'CMakeLists.txt','third_party','tests/sycl'], cwd=P, text=True).splitlines()
            names = sorted(set(names) | set(manifest['proposed_files']) |
                {str(p.relative_to(P)) for p in (fixture,shim,fault)})
            record['source_pins'] = {n:sha(P/n) for n in names if (P/n).is_file()}
            persist()
            run('configure-native-dense', configure, 300)
            run('build-native-dense', ['cmake','--build',str(build),'--target','strata',
                'sycl_native_dense_bank_test','sycl_native_dense_bank_fault_test','--parallel','2'], 1800)
            commands = json.loads((build/'compile_commands.json').read_text())
            proof = {}
            for name in ('src/program/generate.cpp','src/core/native_dense.cpp','src/core/verify.cpp',
                'src/core/mtp.cpp','src/core/session.cpp','src/prefill/prefill.cpp',
                'tests/sycl/native_dense_bank.cpp','tests/sycl/native_dense_bank_fault_object.cpp'):
                matches = [c for c in commands if c['file'] == str(P/name)]
                require(matches and all('-fsycl-id-queries-range=size_t' in c['command'] for c in matches),
                    'actual safe SYCL source resolver '+name)
                proof[name] = matches
            record['compile_flag_proof'] = proof
            record['compile_commands_sha256'] = sha(build/'compile_commands.json')
            require(all(sha(P/n) == h for n,h in record['source_pins'].items()), 'source stable during build')
            record['binaries'] = {n:sha(build/n) for n in
                ('strata','sycl_native_dense_bank_test','sycl_native_dense_bank_fault_test')}
            record.update(binary=str(build/'strata'), binary_sha256=record['binaries']['strata'],
                passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            persist()
            with target.open('x') as output:
                json.dump(record, output, indent=2)
                output.write('\n')
        print(json.dumps(dict(passed=record['passed'], GPU_executed=False,
            binary_sha256=record.get('binary_sha256'))), flush=True)


if __name__ == '__main__':
    main()
