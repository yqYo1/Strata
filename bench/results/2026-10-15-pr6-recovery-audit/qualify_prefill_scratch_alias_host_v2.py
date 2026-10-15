#!/usr/bin/env python3
"""Continue the closed v1 gate with host-only ASan/UBSan; preserve v1 failure."""
import fcntl
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from analyze_prefill_window_profile_v2 import B, P, load_owner, require, sha


def main():
    root = B/'main-prefill-scratch-alias-host-root-v2'
    receipt = A/'prefill-scratch-alias-build-host-v2.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not root.exists() and not receipt.exists(), 'fresh host followup')
        prior_path = A/'prefill-scratch-alias-build-host-v1.json'
        prior = json.loads(prior_path.read_text())
        require(not prior['passed'] and not prior['complete'] and not prior['active'] and
                not prior['GPU_executed'] and prior['applied'], 'original failed gate')
        require(prior['commands'][-1]['label'] == 'compile-prefix-sanitized' and
                prior['commands'][-1]['exit_code'] == 1 and
                "only supported for host compilation" in prior['error'], 'known driver-option failure')
        require(all(c['exit_code'] == 0 for c in prior['commands'][:-1]) and
                len(prior['host_results']) == 11 and all(x['passed'] for x in prior['host_results']),
                'actual product build and all normal cases')
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for c in prior['commands']:
            require(module.closed(c), 'previous command not retired')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'previous identity present')
        require(all(sha(P/n) == h for n, h in prior['source_pins'].items()), 'product source unchanged')
        require(all(sha(Path(n)) == h for n, h in prior['linked_inputs'].items()), 'libraries unchanged')
        require(sha(Path(prior['compiler'])) == prior['compiler_sha256'], 'compiler unchanged')
        require(sha(Path(prior['binary'])) == prior['binary_sha256'], 'product executable unchanged')
        fixture = A/'prefill_scratch_prefix_fixture_v1.cpp'
        require(sha(fixture) == prior['fixture_sha256'], 'fixture unchanged')
        root.mkdir(mode=0o700)
        obj = root/'prefix-host-sanitized.o'; exe = root/'prefix-host-sanitized'
        compile_argv = list(prior['commands'][-1]['argv'])
        compile_argv[compile_argv.index('-o')+1] = str(obj)
        compile_argv.insert(compile_argv.index('-fsanitize=address,undefined'), '-Xarch_host')
        link_entry = next(c for c in prior['commands'] if c['label'] == 'link-prefix-normal')
        link_argv = list(link_entry['argv'])
        oldexe = Path(link_argv[link_argv.index('-o')+1])
        require(sha(oldexe) == prior['host_results'][0]['binary_sha256'], 'normal fixture identity')
        link_argv[link_argv.index(str(oldexe)+'.o')] = str(obj)
        link_argv[link_argv.index('-o')+1] = str(exe)
        link_argv[link_argv.index('-Wl,-Map='+str(oldexe)+'.map')] = '-Wl,-Map='+str(exe)+'.map'
        link_argv += ['-Xarch_host', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        r = dict(schema='prefill-scratch-alias-build-host-v2', passed=False, complete=False,
            active=True, GPU_executed=False, commands=[], host_results=[],
            recipe_sha256=sha(Path(__file__)), prior_failed_receipt_sha256=sha(prior_path),
            compiler=prior['compiler'], compiler_sha256=prior['compiler_sha256'],
            binary=prior['binary'], binary_sha256=prior['binary_sha256'],
            source_pins=prior['source_pins'], linked_inputs=prior['linked_inputs'],
            environment=prior['environment'], fixture_sha256=sha(fixture),
            normal_processes_already_passed=11, default_enabled=False,
            references=['https://clang.llvm.org/docs/ClangCommandLineReference.html',
                'https://www.intel.com/content/www/us/en/developer/articles/technical/find-bugs-quickly-using-sanitizers-with-oneapi-compiler.html'],
            scope='Actual product build and normal11 from original FAILED v1; host-only ASan/UBSan of same production-TU fixture and same11 cases. No GPU allocation, model arithmetic, capacity, speed, RESTORE or physical256K qualification.')
        def persist():
            (root/'record.json').write_text(json.dumps(r, indent=2)+'\n')
        owner = module.Owner(root/'owned', persist); r['commands'] = owner.commands; persist()
        def run(label, argv, env, wall=600):
            c, out, err = owner.run(label, argv, env, P, wall=wall, cpu=wall*12,
                rss_cap=24 << 30, text_cap=8 << 20, total_cap=2 << 30, file_cap=512 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label+': '+err.read_text()[-5000:])
            print('PASS '+label, flush=True)
            return out.read_text()
        try:
            run('compile-prefix-host-sanitized', compile_argv, prior['environment'])
            run('link-prefix-host-sanitized', link_argv, prior['environment'])
            require('prefill.cpp.o' not in Path(str(exe)+'.map').read_text(), 'old archive selected')
            symbols = run('host-sanitizer-symbols', ['nm', '-u', str(obj)], prior['environment'], wall=60)
            r['sanitizer_symbols'] = [s.strip() for s in symbols.splitlines() if '__asan_' in s or '__ubsan_' in s]
            require(any('__asan_' in s for s in r['sanitizer_symbols']) and
                    any('__ubsan_' in s for s in r['sanitizer_symbols']), 'actual host instrumentation')
            cases = [c for c in prior['commands'] if c['label'].startswith('prefix-normal-')]
            require(len(cases) == 11, 'all original immutable conditions')
            for i, c in enumerate(cases):
                argv = [str(exe), *c['argv'][1:]]
                output = run('prefix-host-sanitized-'+str(i), argv, c['environment'], wall=90)
                require(('prefix_flag PASS rejected_before_geometry' if argv[1] == '-1' else
                         'prefix_layout PASS cases=10') in output, 'complete same-scope host checks')
                r['host_results'].append(dict(prior_normal_command=c['label'], passed=True,
                    expected_mode=int(argv[1]), stdout=output, binary_sha256=sha(exe)))
                persist()
            require(all(sha(P/n) == h for n, h in r['source_pins'].items()) and
                    sha(fixture) == r['fixture_sha256'], 'source changed during host gate')
            r.update(passed=True, complete=True, total_host_processes=22,
                     sanitized_fixture=str(exe), sanitized_fixture_sha256=sha(exe))
        except BaseException as e:
            r['error'] = repr(e)
            raise
        finally:
            r.update(active=owner.active is not None, finished_utc=module.utc()); persist()
            receipt.write_text(json.dumps(r, indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'], total_host_processes=r.get('total_host_processes'))))


if __name__ == '__main__':
    main()
