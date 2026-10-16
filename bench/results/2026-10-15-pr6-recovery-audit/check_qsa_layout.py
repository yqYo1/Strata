#!/usr/bin/env python3
"""Serialize first diagnostic correctness and a separate clean component timing job."""
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
BUILD = B / 'main-primary-qsa-layout-build-root-v1'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


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
    O = B / ('main-primary-qsa-layout-' + mode + '-root-v1')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if mode == 'microbench':
            gate = json.loads((B / 'main-primary-qsa-layout-validation-root-v1/record.json').read_text())
            assert gate['passed'] and gate['complete'] and not gate['active']
            assert gate['binary_sha256'] == build['binaries']['sycl_qsa_fallback_layout_test']
            assert not gate['fresh_faults'] and gate['boot_unchanged'] and not gate['driver_dump_exists']
        assert (DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists()
        assert not O.exists()
        assert all(sha(P / n) == h for n, h in build['source_pins'].items())
        exe = Path(build['build_path']) / 'sycl_qsa_fallback_layout_test'
        assert sha(exe) == build['binaries'][exe.name]
        reference = json.loads(gzip.decompress((P / 'bench/results/2026-10-14-long-generation-benchmark/receipts/main-primary-long-generation-suite-root-v4-upload-fix-v1/record.json.gz').read_bytes()))
        env = dict(reference['environment'])
        for name in ['STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT']:
            env.pop(name, None)
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        O.mkdir()
        env['XDG_CACHE_HOME'] = str(O / 'cache')
        argv = [str(exe)]
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
            argv.append('--benchmark')
            assert not any(k in env for k in ['UR_ENABLE_LAYERS', 'ZE_ENABLE_VALIDATION_LAYER',
                'STRATA_TRACE', 'LD_PRELOAD', 'STRATA_PREFILL_TIMING', 'STRATA_DECODE_TIMING'])
        record = dict(active=True, passed=False, complete=False, commands=[], mode=mode,
            argv=argv, environment=env, build_receipt_sha256=sha(BUILD / 'record.json'),
            source_revision=build['source_revision'], source_pins=build['source_pins'],
            binary_sha256=sha(exe), recipe_sha256=sha(Path(__file__)),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            limits=dict(wall_seconds=240 if mode == 'validation' else 180,
                        raw_text_bytes=64 << 20, job_bytes=1 << 30, RSS_bytes=8 << 30),
            performance_qualified=False, quality_qualified=False, physical_262144_qualified=False,
            restore_qualified=False, adopted=False,
            scope='Bounded synthetic fallback Q-local-layout test only. Logical cell262143 mapping is not full-model262144 input. Microbench event envelope includes submission gaps and is not exclusive kernel/whole-model timing.')

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
            wall = record['limits']['wall_seconds']
            entry, out, err = owner.run('qsa-layout-' + mode, argv, env, P, wall=wall,
                cpu=wall * len(os.sched_getaffinity(0)), text_cap=64 << 20, rss_cap=8 << 30,
                total_cap=1 << 30, file_cap=64 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (entry.get('error'), err.read_text()[-3000:], out.read_text()[-3000:])
            text = out.read_text()
            assert 'device=Intel(R) Arc(TM) B570 Graphics' in text
            record['stdout_sha256'] = sha(out)
            record['stderr_sha256'] = sha(err)
            diagnostic = err.read_text()
            warning_lines = [line for line in diagnostic.splitlines()
                             if re.search(r'(?i)warning|error|FAIL', line)]
            assert len('\n'.join(warning_lines).encode()) <= 256 << 10
            record['diagnostics'] = warning_lines
            record['api_trace_summary'] = dict(lines=len(diagnostic.splitlines()),
                kernel_launch_mentions=diagnostic.count('urEnqueueKernelLaunch'),
                success_mentions=diagnostic.count('UR_RESULT_SUCCESS'),
                warning_lines=len(warning_lines), source_bytes=err.stat().st_size,
                source_sha256=sha(err), scope='Host API trace mentions only; not physical launches or device time')
            if mode == 'validation':
                cases = re.findall(r'^PASS format=(\d+) nq=(\d+) cap=(\d+) max_abs=([\d.eE+-]+)$', text, re.M)
                assert len(cases) == 20 and 'PASS host geometry/layout/overflow contracts' in text
                assert [(int(f), int(n), int(c)) for f, n, c, _ in cases] == [
                    (f, n, c) for f in range(4) for n, c in [(1, 63), (17, 64), (32, 65), (17, 129), (1, 129)]]
                record['cases'] = [dict(format=int(f), nq=int(n), cap=int(c), max_absolute_error=float(e))
                                   for f, n, c, e in cases]
                record['evidence_lines'] = text.splitlines()
            else:
                samples = re.findall(r'^device_envelope sample=(\d+) layout=(\d+) repeats=(\d+) ns=(\d+) ns_per_batch=([\d.eE+-]+)', text, re.M)
                assert len(samples) == 40 and [int(s[0]) for s in samples] == list(range(40))
                assert all(int(l) == (0 if int(i) % 4 in [0, 3] else 1) and int(r) == 8 and int(ns) > 0
                           for i, l, r, ns, _ in samples)
                record['samples'] = [dict(sample=int(i), layout=int(l), repeats=int(r), ns=int(ns),
                                          ns_per_batch_exact=int(ns) / int(r)) for i, l, r, ns, _ in samples]
                record['shape'] = dict(format=1, nq=32, cap=2051, all_active_pages_resident=True)
            assert all(sha(P / n) == h for n, h in build['source_pins'].items())
            assert sha(exe) == record['binary_sha256']
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
                (D / ('qsa-layout-' + mode + '-v1.json')).write_text(json.dumps(record, indent=2) + '\n')
        assert record['passed'] and record['complete']
        print(json.dumps(dict(passed=True, mode=mode, fresh_faults=len(record['fresh_faults']),
                              count=len(record.get('cases', record.get('samples', []))))), flush=True)


if __name__ == '__main__':
    main()
