#!/usr/bin/env python3
"""Bounded first VTune visibility check on the qualified standalone GEMM probe."""
import fcntl
import gzip
import importlib.util
import json
import os
import re
from pathlib import Path
from check_gemm_orientation import P, B, D, BUILD, OWNER, DEVICE, FAULT, sha

VTUNE = Path('/opt/intel/oneapi/vtune/latest/bin64/vtune')
O = B / 'main-primary-gemm-vtune-visibility-root-v2'


def main():
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = json.loads((BUILD / 'record.json').read_text())
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert build['passed'] and build['complete'] and not build['active']
        for mode in ['validation', 'microbench']:
            r = json.loads((B / ('main-primary-gemm-orientation-' + mode + '-root-v2') / 'record.json').read_text())
            assert r['passed'] and r['complete'] and not r['active']
            assert r['binary_sha256'] == build['binary_sha256']
            assert not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists']
        assert (DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists()
        assert all(sha(P / n) == h for n, h in build['source_pins'].items())
        exe = Path(build['build_path']) / 'sycl_gemm_orientation_bench'
        assert sha(exe) == build['binary_sha256'] and VTUNE.is_file() and not O.exists()
        reference = P / 'bench/results/2026-10-14-long-generation-benchmark/receipts/main-primary-long-generation-suite-root-v4-upload-fix-v1/record.json.gz'
        env = dict(json.loads(gzip.decompress(reference.read_bytes()))['environment'])
        for k in ['STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT']:
            env.pop(k, None)
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        assert not any(k in env for k in ['UR_ENABLE_LAYERS', 'ZE_ENABLE_VALIDATION_LAYER', 'MKL_VERBOSE', 'LD_PRELOAD'])
        O.mkdir(mode=0o700)
        (O / 'profiler-user-data').mkdir(mode=0o700)
        env['XDG_CACHE_HOME'] = str(O / 'cache')
        record = dict(active=True, passed=False, complete=False, commands=[], environment=env,
            source_pins=build['source_pins'], source_revision=build['source_revision'],
            binary_sha256=sha(exe), profiler_sha256=sha(VTUNE), recipe_sha256=sha(Path(__file__)),
            imported_recipe_sha256=sha(D / 'check_gemm_orientation.py'),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            limits=dict(wall_seconds=300, raw_text_bytes=64 << 20, raw_profiler_MB=32,
                        job_bytes=2 << 30, RSS_bytes=16 << 30,
                        reason='VTune result finalization and binary/symbol metadata may exceed its32MB raw trace budget'),
            raw_profiler_consumer='Determine exact-runtime GPU kernel visibility and produce compact kernel/summary report',
            next_review='Retire raw timeline after report verification, or retain one diagnostic for a distinct unresolved visibility failure',
            performance_qualified=False, quality_qualified=False, adopted=False,
            kernel_visibility_qualified=False,
            scope='First actual VTune collection on validated standalone synthetic probe. Profiling overhead; not model speed, RAM/PCIe limits, graph visibility or causal whole-model bottleneck.')

        def persist():
            (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(O / 'owned', persist)
        record['commands'] = owner.commands
        persist()

        def run(label, argv, wall=60, target_env=None):
            entry, out, err = owner.run(label, argv, target_env or build['environment'], P,
                wall=wall, cpu=wall * len(os.sched_getaffinity(0)), text_cap=64 << 20,
                rss_cap=16 << 30, total_cap=2 << 30, file_cap=256 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), err.read_text()[-4000:], out.read_text()[-1000:])
            return out, err

        cursor = None
        try:
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], 30)
            cursor = re.search(r'-- cursor: (\S+)', out.read_text()).group(1)
            record['journal_before_cursor'] = cursor
            out, _ = run('profiler-version', [str(VTUNE), '-version'])
            record['profiler_version'] = out.read_text().strip()
            result = O / 'vtune-result'
            argv = [str(VTUNE), '-collect', 'xpu-offload', '-knob', 'collect-cpu-sampling=false',
                '-knob', 'enable-user-tasks=false', '-knob', 'enable-characterization-insights=false',
                '-knob', 'collect-memory-bandwidth=false', '-knob', 'collect-programming-api=true',
                '-data-limit=32', '-duration=30', '-finalization-mode=full',
                '-result-dir', str(result), '-user-data-dir', str(O / 'profiler-user-data'),
                '-app-working-dir', str(P), '--', str(exe), '--benchmark']
            out, err = run('vtune-collect', argv, 300, env)
            assert 'DONE mode=benchmark' in out.read_text(), 'target completion not observed'
            record['collection_diagnostics'] = err.read_text().splitlines()
            for report, args in [('summary', ['-report', 'summary']),
                                 ('kernels', ['-report', 'hotspots', '-group-by', 'computing-task'])]:
                out, err = run('vtune-' + report, [str(VTUNE)] + args + [
                    '-result-dir', str(result), '-format', 'csv', '-csv-delimiter', 'comma'], 120, env)
                record[report + '_report'] = out.read_text()
                record[report + '_sha256'] = sha(out)
                record[report + '_diagnostics'] = err.read_text().splitlines()
            assert record['kernels_report'].strip(), 'empty computing-task report'
            assert all(sha(P / n) == h for n, h in build['source_pins'].items()) and sha(exe) == record['binary_sha256']
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor:
                    out, _ = run('journal-after', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'], 30)
                    entries = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
                    record['journal_entries'] = len(entries)
                    record['fresh_faults'] = [x for x in entries if FAULT.search(str(x.get('MESSAGE', '')))]
                    record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                    record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False, complete=False)
            finally:
                record['active'] = False
                persist()
                (D / 'vtune-visibility-v2.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=record['passed'], complete=record['complete'], fresh_faults=len(record['fresh_faults']))), flush=True)


if __name__ == '__main__':
    main()
