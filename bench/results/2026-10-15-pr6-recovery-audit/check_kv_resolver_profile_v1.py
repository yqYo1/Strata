#!/usr/bin/env python3
"""Main-owned official device-timing capture of the unchanged KV graph fixture."""
import fcntl
import importlib.util
import json
import os
import re
import select
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, DEVICE, FAULT, sha


def main():
    build_path = D / 'kv-copy-groups-build-v2.json'
    profiler_path = B / 'main-primary-unitrace-build-root-v1/record.json'
    clean_path = D / 'kv-copy-groups-graph-benchmark-512-v3.json'
    build, tool, clean = [json.loads(p.read_text()) for p in [build_path, profiler_path, clean_path]]
    assert all(d['passed'] and d['complete'] and not d['active'] for d in [build, tool, clean])
    assert not clean['fresh_faults'] and clean['boot_unchanged'] and not clean['driver_dump_exists']
    provider = Path(tool['build_path'])
    exe = Path(build['build_path']) / 'sycl_kv_copy_groups_bench'
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    assert sha(exe) == build['binaries'][exe.name]
    assert tool['source_revision'] == '6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033'
    assert all(sha(provider / n) == h for n, h in tool['binaries'].items())
    assert all(sha(Path(n)) == h for n, h in tool['XPTI_provider_pins'].items())
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert all(sha(P / n) == h for n, h in build['source_pins'].items())
        assert (DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists()
        output = B / 'main-primary-kv-resolver-profile-root-v1'
        assert not output.exists(); output.mkdir(mode=0o700)
        reports = output / 'provider'; reports.mkdir(mode=0o700)
        env = dict(clean['environment'], XDG_CACHE_HOME=str(output / 'cache'))
        assert not any(k.startswith(('UNITRACE_', 'PTI_')) or k in ['LD_PRELOAD', 'UR_ENABLE_LAYERS', 'ZE_ENABLE_VALIDATION_LAYER', 'MKL_VERBOSE'] for k in env)
        argv = [str(provider / 'unitrace'), '--device-timing', '--result-dir', str(reports),
                str(exe), '--graph-benchmark', '512', '40']
        record = dict(active=True, passed=False, complete=False, collected=False, GPU_executed=True,
            commands=[], environment=env, argv=argv, source_pins=build['source_pins'],
            build_receipt_sha256=sha(build_path), binary_sha256=sha(exe),
            profiler_build_receipt_sha256=sha(profiler_path), profiler_binaries=tool['binaries'],
            profiler_source_revision=tool['source_revision'], XPTI_provider_pins=tool['XPTI_provider_pins'],
            clean_comparison_sha256=sha(clean_path), recipe_sha256=sha(Path(__file__)),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            limits=dict(wall_seconds=1800, RSS_bytes=8 << 30, text_and_provider_bytes=64 << 20, job_bytes=256 << 20, file_bytes=64 << 20),
            execution_contract='Pinned Linux unitrace.cc lines1234/1424: these non-metrics/non-KMD flags directly exec native in the owned leader; no wrapper, CLI session, dynamic pause/resume, or product synchronization changes.',
            adopted=False, performance_qualified=False, model_profile_executed=False,
            kernel_visibility_qualified=False, physical_262144_qualified=False, restore_qualified=False,
            scope='Aggregate GPU kernel durations for complete graph fixture including reset/oracle work. Profiled host times are not clean throughput; per-case kernel attribution and actual model share are unqualified.',
            retention_consumer='Extract and verify exact resolver/copy kernel counts and duration summaries, then retire raw provider outputs.',
            review_point='After compact profile analysis is committed.')
        def persist(): (output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(output / 'owned', persist); record['commands'] = owner.commands; persist()
        def run(label, args, wall=30, interaction=None):
            e, out, err = owner.run(label, args, env, P, wall=wall,
                cpu=wall * len(os.sched_getaffinity(0)), rss_cap=8 << 30, text_cap=64 << 20,
                total_cap=256 << 20, file_cap=64 << 20, interaction=interaction)
            assert module.closed(e) and e['exit_code'] == 0, (label, e.get('error'), err.read_text()[-5000:])
            return out, err
        cursor = None
        try:
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', out.read_text())[1]
            record['journal_before_cursor'] = cursor
            out, err = run('unitrace-help', [str(provider / 'unitrace'), '--help'])
            assert all(s in out.read_text() + err.read_text() for s in ['--device-timing', '--result-dir'])
            def interaction(io, entry):
                io.proc.stdin.close()
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    files = [p for p in reports.rglob('*') if p.is_file()]
                    logs = list((output / 'owned').glob('*.stdout')) + list((output / 'owned').glob('*.stderr'))
                    assert sum(p.stat().st_size for p in files + logs) <= 64 << 20, 'aggregate raw text/provider budget'
                    err = output / 'owned/kv-profile.stderr'
                    if err.exists():
                        assert b'INCOMPLETE owner_hold:' not in err.read_bytes(), 'unknown completion owners held'
                    if select.select([io.proc.stdout], [], [], .02)[0]:
                        io.read(); io.pending.clear()
            print('Official unitrace device timing: unchanged 720-sample KV graph fixture', flush=True)
            out, err = run('kv-profile', argv, 1800, interaction)
            rows = [json.loads(s) for s in out.read_text().splitlines() if s.strip()]
            assert len(rows) == 720 and all(s['oracle_exact'] for s in rows)
            assert 'PASS bounded component scope only' in err.read_text()
            assert not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|INCOMPLETE owner_hold:|^FAIL|terminate called', err.read_text(), re.M)
            record['correctness'] = dict(samples=len(rows), all_oracle_exact=True,
                stdout_sha256=sha(out), stderr_sha256=sha(err), diagnostics=err.read_text())
            record['provider_files'] = []
            for path in sorted(reports.rglob('*')):
                if not path.is_file(): continue
                item = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
                if path.suffix in ['.csv', '.txt'] and path.stat().st_size <= 4 << 20:
                    item['text'] = path.read_text()
                record['provider_files'].append(item)
            assert all(sha(P / n) == h for n, h in build['source_pins'].items())
            assert all(sha(provider / n) == h for n, h in tool['binaries'].items())
            record.update(passed=True, complete=True, collected=True)
        except BaseException as error:
            record['error'] = repr(error); raise
        finally:
            try:
                if cursor:
                    out, _ = run('journal-final', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    entries = [json.loads(s) for s in out.read_text().splitlines() if s.strip()]
                    record['fresh_faults'] = [s for s in entries if FAULT.search(str(s.get('MESSAGE', '')))]
                    record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                    record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False, complete=False)
            finally:
                record['active'] = False; persist()
                (D / 'kv-resolver-profile-v1.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=record['passed'], collected=record['collected'],
            provider_files=len(record.get('provider_files', [])), fresh_faults=len(record['fresh_faults']))), flush=True)


if __name__ == '__main__':
    main()
