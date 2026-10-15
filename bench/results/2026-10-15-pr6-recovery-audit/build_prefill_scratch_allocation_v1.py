#!/usr/bin/env python3
"""Link the allocation fixture to the qualified production alias object."""
import fcntl
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from analyze_prefill_window_profile_v2 import B, P, load_owner, require, sha


def main():
    root = B/'main-prefill-scratch-allocation-build-root-v1'
    target = A/'prefill-scratch-allocation-build-v1.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not root.exists() and not target.exists(), 'fresh fixture build')
        prior_path = A/'prefill-scratch-alias-build-host-v2.json'
        prior = json.loads(prior_path.read_text())
        require(prior['passed'] and prior['complete'] and not prior['active'] and
                prior['total_host_processes'] == 22, 'host qualification')
        partial_path = A/'prefill-scratch-alias-build-host-v1.json'
        partial = json.loads(partial_path.read_text())
        require(sha(partial_path) == prior['prior_failed_receipt_sha256'], 'original build identity')
        require(all(sha(P/n) == h for n, h in prior['source_pins'].items()), 'exact product source')
        require(all(sha(Path(n)) == h for n, h in prior['linked_inputs'].items()), 'exact libraries')
        obj = Path(partial['prefill_object'])
        require(sha(obj) == partial['prefill_object_sha256'], 'actual alias object')
        compiler = prior['compiler']
        require(sha(Path(compiler)) == prior['compiler_sha256'], 'same compiler')
        libraries = [Path(n) for n in prior['linked_inputs'] if Path(n).suffix in ('.a', '.so')]
        source = A/'prefill_scratch_allocation_fixture_v1.cpp'
        root.mkdir(mode=0o700); exe = root/'scratch-allocation'
        command = [compiler, '-O3', '-DNDEBUG', '-std=c++20', '-fsycl',
            '-fsycl-id-queries-range=size_t', '-fno-fast-math', '-ffp-contract=off',
            '-fsycl-allow-device-image-dependencies', '-DSTRATA_ENABLE_SYCL=1',
            '-Wall', '-Wextra', '-Werror', '-I'+str(P/'include'),
            '-I'+str(P/'include/strata/sycl/compat'), str(source), str(obj),
            *map(str, libraries), '-Wl,-rpath=/opt/intel/oneapi/mkl/2026.1/lib',
            '-Wl,-Map='+str(exe)+'.map', '-lm', '-ldl', '-lpthread', '-o', str(exe)]
        r = dict(passed=False, complete=False, active=True, GPU_executed=False,
            commands=[], recipe_sha256=sha(Path(__file__)), host_receipt_sha256=sha(prior_path),
            fixture_sha256=sha(source), source_pins=prior['source_pins'],
            environment=prior['environment'], compiler=compiler,
            compiler_sha256=prior['compiler_sha256'],
            linked_inputs={str(p):sha(p) for p in [obj, *libraries]},
            scope='Actual production Prefill owned/borrowed/relayout fixture build only; no GPU run.')
        def persist():
            (root/'record.json').write_text(json.dumps(r, indent=2)+'\n')
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        owner = module.Owner(root/'owned', persist); r['commands'] = owner.commands; persist()
        try:
            c, out, err = owner.run('link-scratch-allocation', command, prior['environment'], P,
                wall=300, cpu=1800, rss_cap=16 << 30, text_cap=8 << 20,
                total_cap=1 << 30, file_cap=256 << 20)
            require(module.closed(c) and c['exit_code'] == 0, 'fixture link: '+err.read_text()[-5000:])
            require('prefill.cpp.o' not in Path(str(exe)+'.map').read_text(), 'old prefill archive selected')
            require(sha(source) == r['fixture_sha256'], 'fixture changed during link')
            r.update(passed=True, complete=True, binary=str(exe), binary_sha256=sha(exe),
                     link_map_sha256=sha(Path(str(exe)+'.map')))
        except BaseException as e:
            r['error'] = repr(e)
            raise
        finally:
            r.update(active=owner.active is not None, finished_utc=module.utc()); persist()
            target.write_text(json.dumps(r, indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'], binary_sha256=r.get('binary_sha256'))))


if __name__ == '__main__':
    main()
