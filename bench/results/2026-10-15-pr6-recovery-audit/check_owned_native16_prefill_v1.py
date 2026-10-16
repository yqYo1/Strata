#!/usr/bin/env python3
"""One serialized bounded GPU component/regression cohort from a pinned build."""
import fcntl
import json
from pathlib import Path
import re
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
    require(len(sys.argv) == 3 and sys.argv[1] in ('v1', 'v2', 'v3') and
            sys.argv[2] in ('gemm', 'regression'), 'build version and cohort')
    version, mode = sys.argv[1:]
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bp = A / f'own-native16-prefill-build-{version}.json'
        build = json.loads(bp.read_text())
        require(build['passed'] and build['complete'] and not build['active'], 'engine build gate')
        for c in build['commands']:
            require(module.closed(c) and c['exit_code'] == 0, 'build closed normally')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'build owner present')
        if mode == 'regression':
            prev = json.loads((A / f'own-native16-gemm-correctness-{version}.json').read_text())
            require(prev['passed'] and prev['complete'] and not prev['active'] and not prev['fresh_faults'], 'new GEMM safety/correctness gate')
        require(all(sha(P / n) == h for n, h in build['source_pins'].items()), 'frozen complete source')
        binaries = {n: Path(build['build_path']) / n for n in build['target_pins']}
        require(all(sha(binaries[n]) == h for n, h in build['target_pins'].items()), 'built executable pins')
        require(sha(Path(build['historical_FP32_binary'])) == build['historical_FP32_binary_sha256'], 'historical executable unchanged')
        libs = json.loads((A / 'own-qsa-select-model-a1-library-hashes-v1.json').read_text())['files']
        require(all(sha(Path(v['path'])) == v['sha256'] for v in libs), 'runtime file pins')
        root = B / f'main-own-native16-{mode}-correctness-root-{version}'
        target = A / f'own-native16-{mode}-correctness-{version}.json'
        require(not root.exists() and not target.exists(), 'fresh GPU job')
        root.mkdir(mode=0o700)
        previous = json.loads((A / 'own-native16-ops-correctness-v1.json').read_text())['environment']
        env = dict(previous, XDG_CACHE_HOME=str(root / 'cache'))
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[], checks={},
            mode=mode, source_commit=build['source_commit'], source_pins=build['source_pins'],
            build_receipt_sha256=sha(bp), binary_pins=build['target_pins'], runtime_files=libs,
            environment=env, recipe_sha256=sha(Path(__file__)), boot_id=boot_id(),
            performance_qualified=False, model_quality_qualified=False, physical_262144_qualified=False,
            scope='Typed16 GEMM FP64/special/bounds tests or existing runtime/compat/FP32 GEMM plus typed HC/MoE postops regression. No model inference/speed/quality claim.',
            raw_budget_bytes=1 << 30)
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands; persist()
        hostenv = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        def run(label, argv, current, wall=180, cpu=1800, rss=8 << 30):
            c, out, err = owner.run(label, argv, current, P, wall=wall, cpu=cpu, rss_cap=rss,
                text_cap=64 << 20, total_cap=1 << 30, file_cap=64 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label + ': ' + err.read_text()[-6000:])
            return out, err
        cursor = None
        try:
            require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'GPU preflight')
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], hostenv, wall=60, cpu=120, rss=2 << 30)
            cursor = re.search(r'-- cursor: (\S+)', out.read_text())[1]
            jobs = [('gemm16-first', 'sycl_gemm16_test', ['--smoke'], True)] if mode == 'gemm' else [
                ('runtime', 'sycl_runtime_test', [], False),
                ('compat', 'sycl_compat_test', [], False),
                ('FP32-gemm', 'sycl_gemm_test', [], False),
                ('typed-HC-MoE', 'sycl_native16_prefill_test', [], False),
                ('early-owner-release', 'sycl_allocation_teardown_test', ['--early'], False),
                ('late-static-owner-release', 'sycl_allocation_teardown_test', [], False)]
            # Native graph capture is deliberately not a new test dependency;
            # the allocation change does not require enabling that excluded route.
            for label, name, args, trace in jobs:
                current = dict(env, UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if trace else 'UR_LAYER_PARAMETER_VALIDATION',
                               UR_LOG_TRACING='level:info;flush:info;output:stderr')
                record['GPU_executed'] = True
                out, err = run(label, [str(binaries[name]), *args], current, wall=300 if mode == 'regression' else 180, cpu=3600, rss=12 << 30)
                stdout, stderr = out.read_text(), err.read_text()
                if mode == 'gemm':
                    data = json.loads(stdout)
                    require(data['status'] == 'PASS' and data['numerical_cases'] == 68, 'all typed GEMM cases passed')
                    require('GEMM16 tuple ' in stderr and 'urEnqueueKernelLaunch' in stderr, 'actual tuple and first launch trace')
                    require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', stderr), 'native runtime error')
                else:
                    require('PASS' in stdout or 'passed' in stdout or
                            (name == 'sycl_allocation_teardown_test' and 'main complete' in stdout), 'explicit regression result')
                record['checks'][label] = dict(stdout=stdout, stderr_sha256=sha(err), environment=current,
                    stderr_evidence=[l for l in stderr.splitlines() if l.startswith('GEMM16 device=') or l.startswith('GEMM16 tuple ')])
                persist()
            require(all(sha(P / n) == h for n, h in build['source_pins'].items()), 'source unchanged')
            record.update(passed=True, complete=True)
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
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False, complete=False)
                persist()
                with target.open('x') as f:
                    json.dump(record, f, indent=2); f.write('\n')
        print(json.dumps(dict(mode=mode, version=version, passed=record['passed'], active=record['active'])), flush=True)


if __name__ == '__main__':
    main()
