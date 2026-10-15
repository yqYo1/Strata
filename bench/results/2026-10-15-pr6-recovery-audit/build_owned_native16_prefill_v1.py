#!/usr/bin/env python3
"""Main-owned fresh native16 engine build; retains the earlier FP32 executable."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('v1', 'v2', 'v3'), 'fresh build version')
    version = sys.argv[1]
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = json.loads((A / 'own-native16-ops-correctness-v1.json').read_text())
        require(prior['passed'] and prior['complete'] and not prior['active'] and not prior['fresh_faults'], 'postops GPU fixture closed healthy')
        for c in prior['commands']:
            require(module.closed(c), 'prior closed command')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'prior owner alive')
        known = json.loads((A / 'own-dense-prefill-nll-build-v3.json').read_text())
        baseline = Path(known['binary'])
        require(sha(baseline) == known['binary_sha256'] and sha(Path(known['compiler'])) == known['compiler_sha256'], 'baseline and compiler unchanged')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
        names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, '--',
            'src', 'include', 'cmake', 'CMakeLists.txt', 'tests'], cwd=P, text=True).splitlines()
        pins = {n: sha(P / n) for n in names}
        import hashlib
        for n, h in pins.items():
            require(hashlib.sha256(subprocess.check_output(['git', 'show', commit + ':' + n], cwd=P)).hexdigest() == h, 'committed compiled/test source ' + n)
        root = B / ('main-own-native16-prefill-build-root-' + version)
        target = A / ('own-native16-prefill-build-' + version + '.json')
        require(not root.exists() and not target.exists(), 'fresh build path/receipt')
        root.mkdir(mode=0o700)
        build = root / 'build'
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[],
            source_commit=commit, source_pins=pins, environment=known['environment'], boot_id=boot_id(),
            compiler=known['compiler'], compiler_sha256=known['compiler_sha256'],
            historical_FP32_binary=str(baseline), historical_FP32_binary_sha256=known['binary_sha256'],
            build_path=str(build), recipe_sha256=sha(Path(__file__)), build_budget_bytes=8 << 30,
            scope='Fresh complete native16/FP32 engine plus component/regression fixtures; CPU host fixtures only executed here, no GPU/model/performance qualification')
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands; persist()
        def run(label, argv, wall=1200, cpu=18000, rss=40 << 30):
            c, out, err = owner.run(label, argv, known['environment'], P, wall=wall, cpu=cpu,
                rss_cap=rss, text_cap=64 << 20, total_cap=8 << 30, file_cap=2 << 30)
            require(module.closed(c) and c['exit_code'] == 0, label + ': ' + out.read_text()[-5000:] + err.read_text()[-5000:])
            return out.read_text(), err.read_text()
        try:
            oldcache = Path(known['build_path']) / 'CMakeCache.txt'
            options = {}
            for line in oldcache.read_text().splitlines():
                if '=' in line and ':' in line.split('=', 1)[0]:
                    left, value = line.split('=', 1); key, typ = left.split(':', 1)
                    if key.startswith('STRATA_') and typ in ('BOOL', 'STRING', 'PATH'):
                        options[key] = (typ, value)
            require(options['STRATA_ENABLE_SYCL'][1] == 'ON' and options['STRATA_ENABLE_CUDA'][1] == 'OFF' and options['STRATA_ENABLE_HIP'][1] == 'OFF', 'actual reference SYCL resolver')
            configure = ['cmake', '-S', str(P), '-B', str(build), '-G', 'Ninja',
                '-DCMAKE_CXX_COMPILER=' + known['compiler'], '-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DMKL_DIR=/opt/intel/oneapi/mkl/2026.1/lib/cmake/mkl']
            configure += [f'-D{k}:{t}={v}' for k, (t, v) in sorted(options.items())]
            record['configure_argv'] = configure; persist()
            run('configure', configure, wall=300, cpu=1800, rss=8 << 30)
            targets = ['strata', 'sycl_gemm16_test', 'sycl_native16_prefill_test', 'sycl_runtime_test',
                'sycl_compat_test', 'sycl_gemm_test', 'prefill_layer_major_plan_test', 'dense_prefill_nll_host_test']
            run('build', ['cmake', '--build', str(build), '--target', *targets, '--parallel', '2'])
            for name in ('prefill_layer_major_plan_test', 'dense_prefill_nll_host_test'):
                out, err = run('host-' + name, [str(build / name)], wall=60, cpu=120, rss=2 << 30)
                record.setdefault('host_checks', {})[name] = dict(stdout=out, stderr=err)
            require(all(sha(P / n) == h for n, h in pins.items()) and sha(baseline) == known['binary_sha256'], 'frozen source and historical executable')
            cc = json.loads((build / 'compile_commands.json').read_text())
            record['compiled_native16_commands'] = [v for v in cc if v['file'] in (
                str(P / 'src/prefill/sycl/native16.cpp'), str(P / 'src/prefill/sycl/gemm16.cpp'))]
            require(len(record['compiled_native16_commands']) == 2 and all('-fno-fast-math' in v['command'] for v in record['compiled_native16_commands']), 'actual native16 strict source resolver')
            record.update(passed=True, complete=True, binary=str(build / 'strata'), binary_sha256=sha(build / 'strata'),
                target_pins={n: sha(build / n) for n in targets})
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):
                record.update(passed=False, complete=False)
            persist()
            with target.open('x') as f:
                json.dump(record, f, indent=2); f.write('\n')
        print(json.dumps(dict(passed=record['passed'], complete=record['complete'], GPU_executed=False, version=version)), flush=True)


if __name__ == '__main__':
    main()
