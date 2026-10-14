#!/usr/bin/env python3
"""Main-owned GEMM descriptor diagnostics and separate clean synthetic samples."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
BUILD = B / 'main-primary-gemm-orientation-build-root-v1'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
ANCHORS = [1, 12, 428, 1490, 1792, 3245, 3855, 4096]
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
NUMBER = r'([\d.eE+-]+)'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    mode = sys.argv[1]
    assert mode in ['validation', 'microbench']
    build = json.loads((BUILD / 'record.json').read_text())
    assert build['passed'] and build['complete'] and not build['active'] and build['host_contracts_passed']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    O = B / ('main-primary-gemm-orientation-' + mode + '-root-v1')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if mode == 'microbench':
            gate = json.loads((B / 'main-primary-gemm-orientation-validation-root-v1/record.json').read_text())
            assert gate['passed'] and gate['complete'] and not gate['active']
            assert gate['binary_sha256'] == build['binary_sha256']
            assert not gate['fresh_faults'] and gate['boot_unchanged'] and not gate['driver_dump_exists']
        assert (DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists()
        assert not O.exists()
        assert all(sha(P / n) == h for n, h in build['source_pins'].items())
        exe = Path(build['build_path']) / 'sycl_gemm_orientation_bench'
        assert sha(exe) == build['binary_sha256']
        ref = P / 'bench/results/2026-10-14-long-generation-benchmark/receipts/main-primary-long-generation-suite-root-v4-upload-fix-v1/record.json.gz'
        env = dict(json.loads(gzip.decompress(ref.read_bytes()))['environment'])
        for name in ['STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT']:
            env.pop(name, None)
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        O.mkdir()
        env['XDG_CACHE_HOME'] = str(O / 'cache')
        if mode == 'validation':
            env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:stderr',
                UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
                ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
                ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOGGING_LEVEL='warning',
                ZEL_LOADER_LOG_CONSOLE='1', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0')
        else:
            assert not any(k in env for k in ['UR_ENABLE_LAYERS', 'ZE_ENABLE_VALIDATION_LAYER',
                                              'STRATA_TRACE', 'LD_PRELOAD', 'MKL_VERBOSE'])
        argv = [str(exe), '--validate' if mode == 'validation' else '--benchmark']
        record = dict(active=True, passed=False, complete=False, commands=[], mode=mode, argv=argv,
            environment=env, build_receipt_sha256=sha(BUILD / 'record.json'), source_revision=build['source_revision'],
            source_pins=build['source_pins'], binary_sha256=sha(exe), recipe_sha256=sha(Path(__file__)),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            limits=dict(wall_seconds=600, raw_text_bytes=64 << 20, job_bytes=1 << 30, RSS_bytes=8 << 30),
            performance_qualified=False, quality_qualified=False, physical_262144_qualified=False,
            restore_qualified=False, adopted=False,
            scope='Synthetic hot FP16 descriptor/wrapper component only; not model or7format rotation/weighted workload. Host submit and whole-queue wall separate, no exclusive service/event assertion.')

        def persist():
            (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(O / 'owned', persist)
        record['commands'] = owner.commands
        persist()

        def utility(label, args):
            entry, out, err = owner.run(label, args, build['environment'], P, wall=30,
                                        text_cap=64 << 20, total_cap=1 << 30)
            assert module.closed(entry) and entry['exit_code'] == 0
            return out.read_text()

        cursor = None
        try:
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before',
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])).group(1)
            record['journal_before_cursor'] = cursor
            entry, out, err = owner.run('gemm-' + mode, argv, env, P, wall=600,
                cpu=600 * len(os.sched_getaffinity(0)), text_cap=64 << 20, rss_cap=8 << 30,
                total_cap=1 << 30, file_cap=64 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (entry.get('error'), err.read_text()[-3000:], out.read_text()[-3000:])
            text, diagnostic = out.read_text(), err.read_text()
            assert 'device=Intel(R) Arc(TM) B570 Graphics' in text
            assert 'DONE mode=' + ('validate' if mode == 'validation' else 'benchmark') in text
            record['stdout_sha256'], record['stderr_sha256'] = sha(out), sha(err)
            record['device_line'] = next(line for line in text.splitlines() if line.startswith('device='))
            warnings = [line for line in diagnostic.splitlines() if re.search(r'(?i)warning|error|FAIL', line)]
            assert len('\n'.join(warnings).encode()) <= 256 << 10
            record['diagnostics'] = warnings
            record['api_trace_summary'] = dict(lines=len(diagnostic.splitlines()), bytes=err.stat().st_size,
                kernel_launch_mentions=diagnostic.count('urEnqueueKernelLaunch'),
                success_mentions=diagnostic.count('UR_RESULT_SUCCESS'), warning_lines=len(warnings),
                scope='Trace mention counts only; not GPU launch/service timing')
            record['input_hash_lines'] = [line for line in text.splitlines() if line.startswith('input_hash_fnv1a64 ')]
            assert len(record['input_hash_lines']) == 16
            if mode == 'validation':
                pattern = r'^PASS role=(GU|Down) ne=(\d+) beta=' + NUMBER + r' compared_outputs=(\d+) orientation_max_abs=' + NUMBER + r' orientation_max_error_over_limit=' + NUMBER + r' wrapper_max_abs=' + NUMBER + r' wrapper_max_error_over_limit=' + NUMBER + '$'
                cases = re.findall(pattern, text, re.M)
                assert len(cases) == 32
                assert [(r, int(n), float(b)) for r, n, b, *_ in cases] == [(r, n, b) for n in ANCHORS for r in ['GU', 'Down'] for b in [0., .25]]
                record['cases'] = [dict(role=r, ne=int(n), beta=float(b), compared_outputs=int(c),
                    orientation_max_abs=float(a), orientation_max_error_over_limit=float(al),
                    wrapper_max_abs=float(w), wrapper_max_error_over_limit=float(wl)) for r, n, b, c, a, al, w, wl in cases]
                record['oracle_lines'] = [line for line in text.splitlines() if line.startswith('oracle ')]
                assert len(record['oracle_lines']) == 96
            else:
                pattern = r'^sample role=(GU|Down) ne=(\d+) N=(\d+) K=(\d+) ld=(\d+) index=(\d+) arm=(\d+) repeats=(\d+) host_submit_ns=(\d+) drain_ns=(\d+) batch_wall_ns=(\d+) wall_ns_per_call=' + NUMBER
                values = re.findall(pattern, text, re.M)
                assert len(values) == 320
                samples = []
                for row in values:
                    r, n, cols, inner, ld, i, a, repeats, submit, drain, wall, _ = row
                    n, cols, inner, ld, i, a, repeats, submit, drain, wall = map(int, [n, cols, inner, ld, i, a, repeats, submit, drain, wall])
                    assert a == (0 if i % 4 in [0, 3] else 1) and repeats == (16 if n <= 12 else 4)
                    assert wall == submit + drain and wall > 0 and cols == (1280 if r == 'GU' else 2560)
                    assert inner == (2560 if r == 'GU' else 640) and ld == cols + 17
                    samples.append(dict(role=r, ne=n, N=cols, K=inner, ld=ld, index=i, arm=a, repeats=repeats,
                                        host_submit_ns=submit, drain_ns=drain, batch_wall_ns=wall,
                                        wall_ns_per_call_exact=wall / repeats))
                assert [(s['ne'], s['role'], s['index']) for s in samples] == [(n, r, i) for n in ANCHORS for r in ['GU', 'Down'] for i in range(20)]
                record['samples'] = samples
                record['first_call_lines'] = [line for line in text.splitlines() if line.startswith('first_call_in_cell ')]
                assert len(record['first_call_lines']) == 32
                assert text.count('post_benchmark_guard_finite PASS ') == 16
            assert all(sha(P / n) == h for n, h in build['source_pins'].items()) and sha(exe) == record['binary_sha256']
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor:
                    raw = utility('journal-after', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    entries = [json.loads(x) for x in raw.splitlines() if x.strip()]
                    record['journal_entries'] = len(entries)
                    record['fresh_faults'] = [x for x in entries if FAULT.search(str(x.get('MESSAGE', '')))]
                    record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                    record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False, complete=False)
            finally:
                record['active'] = False
                persist()
                (D / ('gemm-orientation-' + mode + '-v1.json')).write_text(json.dumps(record, indent=2) + '\n')
        assert record['passed'] and record['complete']
        print(json.dumps(dict(passed=True, mode=mode, fresh_faults=len(record['fresh_faults']),
                              count=len(record.get('cases', record.get('samples', []))))), flush=True)


if __name__ == '__main__':
    main()
