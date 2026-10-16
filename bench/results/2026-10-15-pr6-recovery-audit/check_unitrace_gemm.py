#!/usr/bin/env python3
"""Reuse the verified PTI build; collect the current model-stride GEMM validation."""
import fcntl
import gzip
import importlib.util
import json
import os
import re
from pathlib import Path
from check_gemm_orientation import P, B, D, BUILD, OWNER, DEVICE, FAULT, sha

PROFILER = B / 'main-primary-unitrace-build-root-v1/record.json'
WRAPPER = P / 'bench/results/2026-10-14-primary-vtune/runtime_child.py'
O = B / 'main-primary-pr6-gemm-unitrace-root-v1'


def main():
    tool = json.loads(PROFILER.read_text())
    build = json.loads((BUILD / 'record.json').read_text())
    assert tool['passed'] and tool['complete'] and not tool['active']
    assert tool['source_revision'] == '6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033'
    assert build['passed'] and build['complete'] and not build['active']
    provider = Path(tool['build_path'])
    assert all(sha(provider / n) == pin for n, pin in tool['binaries'].items())
    assert all(sha(Path(n)) == pin for n, pin in tool['XPTI_provider_pins'].items())
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for mode in ['validation', 'microbench']:
            g = json.loads((B / ('main-primary-gemm-orientation-' + mode + '-root-v2') / 'record.json').read_text())
            assert g['passed'] and g['complete'] and not g['active']
            assert not g['fresh_faults'] and g['boot_unchanged'] and not g['driver_dump_exists']
        assert (DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists()
        assert all(sha(P / n) == pin for n, pin in build['source_pins'].items())
        exe = Path(build['build_path']) / 'sycl_gemm_orientation_bench'
        assert sha(exe) == build['binary_sha256'] and not O.exists()
        O.mkdir(mode=0o700)
        trace = O / 'trace'
        trace.mkdir(mode=0o700)
        reference = P / 'bench/results/2026-10-14-long-generation-benchmark/receipts/main-primary-long-generation-suite-root-v4-upload-fix-v1/record.json.gz'
        env = dict(json.loads(gzip.decompress(reference.read_bytes()))['environment'])
        for key in ['STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT']:
            env.pop(key, None)
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        assert not any(k in env for k in ['UR_ENABLE_LAYERS', 'ZE_ENABLE_VALIDATION_LAYER', 'MKL_VERBOSE', 'LD_PRELOAD'])
        env['XDG_CACHE_HOME'] = str(O / 'cache')
        record = dict(active=True, passed=False, complete=False, collected=False, commands=[], environment=env,
            source_revision=build['source_revision'], source_pins=build['source_pins'], binary_sha256=sha(exe),
            profiler_binaries=tool['binaries'], profiler_build_receipt_sha256=sha(PROFILER),
            profiler_source_revision=tool['source_revision'], XPTI_provider_pins=tool['XPTI_provider_pins'],
            recipe_sha256=sha(Path(__file__)), wrapper_sha256=sha(WRAPPER),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            kernel_visibility_qualified=False, external_UTC_correlation_qualified=False,
            model_profile_executed=False, performance_qualified=False, quality_qualified=False, adopted=False,
            raw_result_consumer='Main current native PID GPU kernel counts/durations/internal queue order; no external UTC marker attribution',
            review_point='After verified compact per-kernel evidence; retain raw only for a named unresolved mechanism',
            limits=dict(wall_seconds=300, raw_text_bytes=64 << 20, trace_and_job_bytes=256 << 20,
                        file_bytes=64 << 20, RSS_bytes=8 << 30, chrome_events_per_thread=65536),
            scope='Official cached unitrace with current64case GEMM probe; default validate mode, no model/PMU metrics. Previous external UTC shift remains unqualified; use only internal order and event durations.')

        def persist():
            (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(O / 'owned', persist)
        record['commands'] = owner.commands
        persist()

        def run(label, argv, wall=30):
            entry, out, err = owner.run(label, argv, env, P, wall=wall,
                cpu=wall * len(os.sched_getaffinity(0)), text_cap=64 << 20, rss_cap=8 << 30,
                total_cap=256 << 20, file_cap=64 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), err.read_text()[-4000:], out.read_text()[-1200:])
            return out, err

        cursor = None
        try:
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', out.read_text()).group(1)
            record['journal_before_cursor'] = cursor
            out, err = run('unitrace-help', [str(provider / 'unitrace'), '--help'])
            flags = ['--chrome-sycl-logging', '--chrome-call-logging', '--chrome-kernel-logging',
                '--pid', '--tid', '--chrome-event-buffer-size', '65536', '--output-dir-path', str(trace)]
            assert all(flag in out.read_text() + err.read_text() for flag in flags if flag.startswith('--'))
            argv = [str(provider / 'unitrace'), *flags, '/usr/bin/python3', str(WRAPPER), str(O), str(exe)]
            record['collector_argv'] = argv
            persist()
            out, err = run('unitrace-gemm', argv, 300)
            text = out.read_text()
            assert 'DONE mode=validate' in text and len(re.findall(r'^PASS role=', text, re.M)) == 64
            native = json.loads((O / 'runtime-exit.json').read_text())
            assert native['exit_code'] == 0 and native['binary_sha256'] == record['binary_sha256']
            record['real_native_exit'] = native
            record['diagnostics'] = err.read_text().splitlines()
            record['trace_files'] = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
                                     for p in sorted(trace.glob('*.json'))]
            assert record['trace_files'] and all(r['bytes'] <= 64 << 20 for r in record['trace_files'])
            assert all(sha(P / n) == pin for n, pin in build['source_pins'].items())
            assert all(sha(provider / n) == pin for n, pin in tool['binaries'].items())
            record.update(passed=True, complete=True, collected=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor:
                    out, _ = run('journal-after', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    entries = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
                    record['fresh_faults'] = [x for x in entries if FAULT.search(str(x.get('MESSAGE', '')))]
                    record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                    record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False, complete=False)
            finally:
                record['active'] = False
                persist()
                (D / 'unitrace-gemm-collection-v1.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=record['passed'], collected=record['collected'],
                              trace_files=len(record.get('trace_files', [])), fresh_faults=len(record['fresh_faults']))), flush=True)


if __name__ == '__main__':
    main()
