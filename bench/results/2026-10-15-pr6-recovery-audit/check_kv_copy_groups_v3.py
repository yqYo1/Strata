#!/usr/bin/env python3
"""Actual bounded-copy component: Main-owned oracle, graph and clean samples."""
import fcntl
import importlib.util
import json
import os
import re
import select
import sys
import time
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, DEVICE, FAULT, sha


def main():
    mode = sys.argv[1]
    assert mode in ['validation','graph-benchmark','direct-benchmark']
    groups = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    assert groups in ([0] if mode == 'validation' else [128,256,512,1024])
    build_path = D / 'kv-copy-groups-build-v2.json'
    build = json.loads(build_path.read_text())
    assert build['passed'] and build['complete'] and not build['active'] and build['host_contracts_passed']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if mode != 'validation':
            gate = json.loads((D / 'kv-copy-groups-validation-v2.json').read_text())
            assert gate['passed'] and gate['complete'] and not gate['active'] and not gate['fresh_faults']
            assert gate['binary_sha256'] == build['binaries']['sycl_kv_copy_groups_bench']
        assert all(sha(P / n) == h for n,h in build['source_pins'].items())
        exe = Path(build['build_path']) / 'sycl_kv_copy_groups_bench'
        assert sha(exe) == build['binaries'][exe.name]
        assert (DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists()
        output = B / ('main-primary-kv-copy-groups-' + mode + (('-' + str(groups)) if groups else '') + '-root-v3')
        assert not output.exists(); output.mkdir(mode=0o700)
        storage = json.loads((D / 'layer-major-storage-small-v1.json').read_text())
        env = dict(storage['environment'], XDG_CACHE_HOME=str(output/'cache'), STRATA_SYCL_KV_COPY_GROUPS='0')
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        for k,v in {'ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD':'1',
                    'NEOReadDebugKeys':'1','EnableDirectSubmission':'0','STRATA_SYCL_VERIFY_NATIVE_CAPTURE':'0'}.items():
            assert env[k] == v
        if mode == 'validation':
            # Finite multi-case diagnostic: progress JSON plus warning/error/parameter validation.
            # Full per-API tracing would dominate the many guarded readbacks.
            env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
                ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
                ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOGGING_LEVEL='warning', ZEL_LOADER_LOG_CONSOLE='1',
                ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0')
        else:
            # Remove the known instrumentation from the small-probe seed before
            # obtaining clean component timings; preserve all safety controls.
            for key in ['UR_ENABLE_LAYERS','UR_LOG_TRACING','UR_LOG_VALIDATION','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO',
                        'ZE_ENABLE_VALIDATION_LAYER','ZE_ENABLE_PARAMETER_VALIDATION','ZEL_ENABLE_LOADER_LOGGING',
                        'ZEL_LOADER_LOGGING_LEVEL','ZEL_LOADER_LOG_CONSOLE','ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT','STRATA_TRACE']:
                env.pop(key,None)
            assert not any(k in env for k in ['UR_ENABLE_LAYERS','ZE_ENABLE_VALIDATION_LAYER','LD_PRELOAD','MKL_VERBOSE'])
        args = ['--validate'] if mode == 'validation' else ['--graph-benchmark' if mode == 'graph-benchmark' else '--benchmark',str(groups),'40']
        record = dict(active=True, passed=False, complete=False, GPU_executed=True, mode=mode,
                      groups=groups, commands=[], samples=[], environment=env, argv=[str(exe),*args],
                      recipe_sha256=sha(Path(__file__)), build_receipt_sha256=sha(build_path),
                      binary_sha256=sha(exe), source_pins=build['source_pins'],
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      limits=dict(wall_seconds=1800,RSS_bytes=8 << 30,text_bytes=64 << 20,job_bytes=256 << 20),
                      adopted=False, performance_qualified=False, model_performance_qualified=False,
                      quality_qualified=False, physical_262144_qualified=False, restore_qualified=False,
                      scope='Production resolve component with explicit geometry. Logical host payload is not PCIe wire bytes. Host whole-queue spans are not exclusive kernel service.')
        def persist(): (output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(output / 'owned', persist); record['commands'] = owner.commands; persist()
        def utility(label, argv):
            entry, stdout, stderr = owner.run(label,argv,build['environment'],P,wall=30,total_cap=256 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0,stderr.read_text()
            return stdout.read_text()
        def journals(label, cursor):
            return [json.loads(x) for x in utility(label,['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if x.strip()]
        cursor = None
        try:
            cursor = re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            record['journal_before_cursor'] = cursor
            label = 'kv-' + mode
            def interaction(io, entry):
                io.proc.stdin.close()
                err = output / 'owned' / (label + '.stderr')
                next_check = 0
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    if time.monotonic() >= next_check:
                        if err.exists() and b'INCOMPLETE owner_hold:' in err.read_bytes():
                            raise RuntimeError('fixture retained unknown-completion owners; terminate owned session')
                        next_check = time.monotonic() + .25
                    if select.select([io.proc.stdout],[],[],.02)[0]:
                        io.read(); io.pending.clear()
            print('KV component starting ' + mode + ' groups=' + str(groups),flush=True)
            entry, stdout, stderr = owner.run(label,[str(exe),*args],env,P,interaction=interaction,wall=1800,
                cpu=1800*len(os.sched_getaffinity(0)),rss_cap=8 << 30,text_cap=64 << 20,total_cap=256 << 20,file_cap=64 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0,(entry.get('error'),stderr.read_text()[-5000:])
            text = stderr.read_text()
            assert 'device=Intel(R) Arc(TM) B570 Graphics' in text and 'PASS bounded component scope only' in text
            assert not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|INCOMPLETE owner_hold:|^FAIL|terminate called',text,re.M)
            record['diagnostics'] = [x for x in text.splitlines() if re.search(r'(?i)warning|error|FAIL',x)]
            record['stdout_sha256'],record['stderr_sha256'] = sha(stdout),sha(stderr)
            records = [json.loads(x) for x in stdout.read_text().splitlines() if x.strip()]
            assert records and all(x['oracle_exact'] for x in records)
            record['samples'] = records
            copies = [x for x in records if x['phase'] == 'kv_stream_resolve_component']
            assert {x['fmt'] for x in copies} == {0,1,2}
            if mode == 'validation':
                replay = [x for x in records if x['phase'] == 'graph_correctness']
                assert len(replay) == 60 and {(x['fmt'],x['captured_group_cap'],x['replay']) for x in replay} == {
                    (f,g,r) for f in [0,1,2] for g in [0,128,256,512,1024] for r in range(4)}
                assert {x['configured_group_cap'] for x in copies} == {0,128,256,512,1024}
                assert any(x['placed_pages'] == 8192 and x['n_slots'] == 8192 for x in copies)
                assert any(x['case'] == 'last_allocated_cell_262143' for x in copies)
                record['graph_replays_checked'] = len(replay)
            else:
                assert len(copies) == len(records) == 720
                assert all(x['sample'] in range(40) and x['timing_mode'] == 'host_clean' and x['bandwidth_GBps'] is None for x in copies)
                assert {x['configured_group_cap'] for x in copies} == {0,groups}
            assert all(sha(P / n) == h for n,h in build['source_pins'].items())
            record.update(passed=True,complete=True)
        except BaseException as error: record['error'] = repr(error); raise
        finally:
            try:
                if cursor:
                    record['fresh_faults'] = [x for x in journals('journal-final',cursor) if FAULT.search(str(x.get('MESSAGE','')))]
                    record['driver_dump_exists'] = (DEVICE/'devcoredump').exists()
                    record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False,complete=False)
            finally:
                record['active'] = False; persist()
                (D / ('kv-copy-groups-' + mode + (('-' + str(groups)) if groups else '') + '-v3.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(passed=record['passed'],mode=mode,samples=len(record['samples']),fresh_faults=len(record['fresh_faults']))),flush=True)


if __name__ == '__main__': main()
