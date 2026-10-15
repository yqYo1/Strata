#!/usr/bin/env python3
"""Main-only bounded isolated build or GPU check; never changes baseline archives."""
import fcntl
import json
from pathlib import Path
import re
import shlex
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('build', 'ops'), 'build or ops')
    mode = sys.argv[1]
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prevpath = A / ('own-residual-quality-fp32-b-v1.json' if mode == 'build' else 'own-native16-ops-build-v1.json')
        prior = json.loads(prevpath.read_text())
        require(prior['passed'] and prior['complete'] and not prior['active'], 'preceding owner passed and closed')
        for c in prior['commands']:
            require(module.closed(c) and c['exit_code'] == 0, 'preceding command closed')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'preceding owned process alive')
        known = json.loads((A / 'own-dense-prefill-nll-build-v3.json').read_text())
        build = Path(known['build_path'])
        require(sha(build / 'strata') == known['binary_sha256'] and sha(Path(known['compiler'])) == known['compiler_sha256'], 'baseline binary and official compiler pins')
        names = ('include/strata/prefill/native16.hpp', 'src/prefill/sycl/native16.cpp',
                 'tests/sycl/native16_prefill.cpp', 'include/strata/kernels/bf16_bits.hpp',
                 'include/strata/kernels/f16_bits.hpp', 'include/strata/sycl/launch.hpp')
        pins = {n: sha(P / n) for n in names}
        libraries = [build / n for n in ('libstrata_kernels.a', 'libstrata_core.a', 'libstrata_sycl_runtime.a')]
        libpins = {str(n): sha(n) for n in libraries}
        runtime = json.loads((A / 'own-qsa-select-model-a1-library-hashes-v1.json').read_text())['files']
        require(all(sha(Path(v['path'])) == v['sha256'] for v in runtime), 'runtime pins')
        root = B / ('main-own-native16-ops-build-root-v1' if mode == 'build' else 'main-own-native16-ops-correctness-root-v1')
        target = A / ('own-native16-ops-build-v1.json' if mode == 'build' else 'own-native16-ops-correctness-v1.json')
        require(not root.exists() and not target.exists(), 'fresh run, never overwrite failed or live evidence')
        root.mkdir(mode=0o700)
        exe = root / 'native16_prefill' if mode == 'build' else Path(prior['binary'])
        if mode == 'ops':
            require(pins == prior['source_pins'] and sha(exe) == prior['binary_sha256'], 'tested source/binary equal build')
        oldenv = json.loads((A / 'own-host-transfer-component-v1.json').read_text())['environment']
        keys = ['PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CPLUS_INCLUDE_PATH',
                'ONEAPI_DEVICE_SELECTOR', 'UR_ADAPTERS_FORCE_LOAD', 'SYCL_CACHE_PERSISTENT',
                'NEOReadDebugKeys', 'EnableDirectSubmission', 'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD',
                'STRATA_SYCL_VERIFY_NATIVE_CAPTURE', 'STRATA_SYCL_PREFILL_SAME_QUEUE']
        env = dict(known['environment']) if mode == 'build' else {k: oldenv[k] for k in keys}
        env.update(XDG_CACHE_HOME=str(root / 'cache'), STRATA_SYCL_PREFILL_NATIVE16='bf16',
                   UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                   UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                   UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[],
            mode=mode, source_pins=pins, library_pins=libpins, runtime_files=runtime,
            compiler=known['compiler'], compiler_sha256=known['compiler_sha256'],
            canonical_native_sha256=known['binary_sha256'], binary=str(exe),
            environment=env, recipe_sha256=sha(Path(__file__)), prior_receipt_sha256=sha(prevpath),
            boot_id=boot_id(), scope='Native BF16 HC/postops independent FP64 oracle, typed producer/consumer IO, guards and input immutability; not GEMM/model-quality/performance/fullcontext/RESTORE qualification',
            raw_budget_bytes=512 << 20, performance_qualified=False, model_quality_qualified=False)
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        hostenv = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        def run(label, argv, current, wall=900, cpu=18000, rss=32 << 30):
            c, out, err = owner.run(label, argv, current, build if mode == 'build' else P,
                wall=wall, cpu=cpu, rss_cap=rss, text_cap=64 << 20, total_cap=1 << 30, file_cap=512 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label + ': ' + err.read_text()[-6000:])
            return out, err
        cursor = None
        try:
            if mode == 'build':
                cc = json.loads((build / 'compile_commands.json').read_text())
                entry = next(v for v in cc if v['file'] == str(P / 'src/prefill/sycl/postops.cpp'))
                objs = []
                for i, name in enumerate(('tests/sycl/native16_prefill.cpp', 'src/prefill/sycl/native16.cpp')):
                    args = shlex.split(entry['command'])
                    require(args[0] == known['compiler'] and '-fno-fast-math' in args, 'strict qualified compilation')
                    obj = root / f'part{i}.o'; objs.append(obj)
                    args[args.index('-o') + 1] = str(obj)
                    args[args.index('-c') + 1] = str(P / name)
                    record.setdefault('actual_compile_commands', []).append(args); persist()
                    run(f'compile-{i}', args, env)
                out, _ = run('link-recipe', ['ninja', '-C', str(build), '-t', 'commands', 'sycl_prompt_k_cellmajor_test'], env, wall=60, cpu=120, rss=2 << 30)
                toks = shlex.split(out.read_text().splitlines()[-1])
                require(toks[:2] == [':', '&&'] and toks[-2:] == ['&&', ':'], 'canonical link envelope')
                args = toks[2:-2]
                old = 'CMakeFiles/sycl_prompt_k_cellmajor_test.dir/tests/sycl/prompt_k_cellmajor.cpp.o'
                require(args.count(old) == 1, 'one old fixture object')
                i = args.index(old); args[i:i + 1] = list(map(str, objs))
                args[args.index('-o') + 1] = str(exe)
                for lib in libraries:
                    args[args.index(lib.name)] = str(lib)
                record['actual_link_command'] = args; persist()
                run('link', args, env)
            else:
                require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'GPU preflight')
                out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], hostenv, wall=60, cpu=120, rss=2 << 30)
                cursor = re.search(r'-- cursor: (\S+)', out.read_text())[1]
                record['GPU_executed'] = True
                for phase in ('smoke', 'full'):
                    current = dict(env, UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if phase == 'smoke' else 'UR_LAYER_PARAMETER_VALIDATION',
                                   UR_LOG_TRACING='level:info;flush:info;output:stderr')
                    out, err = run(phase, [str(exe), *(['--smoke'] if phase == 'smoke' else [])], current,
                                   wall=180, cpu=1800, rss=8 << 30)
                    require(out.read_text() == 'native16_prefill PASS\n', 'oracle fixture pass')
                    require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', err.read_text()), 'runtime failure')
                    record.setdefault('checks', {})[phase] = dict(stdout=out.read_text(), stderr_sha256=sha(err), environment=current)
                    persist()
            require(all(sha(P / n) == h for n, h in pins.items()), 'source frozen during this job')
            require(all(sha(Path(n)) == h for n, h in libpins.items()) and sha(build / 'strata') == known['binary_sha256'], 'baseline binary/archives unchanged')
            record.update(passed=True, complete=True, binary_sha256=sha(exe))
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor and owner.active is None:
                    out, _ = run('journal-final', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'], hostenv, wall=60, cpu=120, rss=2 << 30)
                    rows = [json.loads(s) for s in out.read_text().splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in rows if FAULT.search(str(v.get('MESSAGE', '')))],
                                  boot_unchanged=boot_id() == record['boot_id'], driver_dump_exists=(DEVICE / 'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None, finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or (mode == 'ops' and ('fresh_faults' not in record or record['fresh_faults'] or not record['boot_unchanged'] or record['driver_dump_exists'])):
                    record.update(passed=False, complete=False)
                persist()
                with target.open('x') as f:
                    json.dump(record, f, indent=2); f.write('\n')
        print(json.dumps(dict(mode=mode, passed=record['passed'], GPU_executed=record['GPU_executed'], active=record['active'])), flush=True)


if __name__ == '__main__':
    main()
