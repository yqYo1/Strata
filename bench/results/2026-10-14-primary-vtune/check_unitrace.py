#!/usr/bin/env python3
"""Main-owned unitrace runtime collection; timeline capability requires review."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import re
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
O = B/'main-primary-unitrace-runtime-root-v1'
BUILD_RECEIPT = B/'main-primary-unitrace-build-root-v1/record.json'
TRANSFER = P/'bench/results/2026-10-14-primary-transfer-ceiling/receipts/main-primary-transfer-ceiling-root-v1/record.json.gz'
NATIVE = P/'bench/results/2026-10-14-primary-model-baseline/receipts/main-primary-watchdog-lifecycle-build-root-v4/record.json.gz'
OWNER = P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    build = json.loads(BUILD_RECEIPT.read_text())
    assert build['passed'] and build['complete'] and not build['active']
    provider = Path(build['build_path'])
    for name, digest in build['binaries'].items():
        assert sha(provider/name) == digest
    transfer = json.loads(gzip.decompress(TRANSFER.read_bytes()))
    native = json.loads(gzip.decompress(NATIVE.read_bytes()))
    executable = Path(native['build_path'])/'sycl_runtime_test'
    assert sha(executable) == native['binaries']['sycl_runtime_test']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists()
        O.mkdir()
        trace = O/'trace'
        trace.mkdir()
        env = dict(transfer['environment'], XDG_CACHE_HOME=str(O/'cache'))
        assert not any(k in env for k in ('UR_ENABLE_LAYERS', 'ZE_ENABLE_VALIDATION_LAYER', 'STRATA_TRACE', 'LD_PRELOAD'))
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        record = dict(active=True, passed=False, complete=False, collected=False, capability_qualified=False,
            model_profile_executed=False, commands=[], environment=env, boot_id=boot, fault_filter=FAULT.pattern,
            runtime_binary_sha256=sha(executable), profiler_binaries=build['binaries'],
            profiler_source_revision=build['source_revision'], build_receipt_sha256=sha(BUILD_RECEIPT),
            recipe_pins={str(x.relative_to(P)):sha(x) for x in (D/'runtime_child.py', Path(__file__))},
            scope='already-qualified64 cross-queue runtime under SYCL/UR/L0 API and kernel trace; no CPU PMU/GPU hardware metrics',
            limits=dict(collector_wall_seconds=180, text_bytes=64<<20, trace_and_job_bytes=128<<20,
                        file_bytes=64<<20, RSS_bytes=4<<30, chrome_events_per_thread=65536),
            raw_result_consumer='Main actual API/GPU task lanes,64-handoff count and clock correlation extraction',
            raw_result_review_point='after verified compact timeline extraction or unsupported capability finding')
        def persist():
            (O/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(O/'owned', persist)
        record['commands'] = owner.commands
        persist()
        cursor = None
        def run(label, argv, wall=30):
            entry, out, err = owner.run(label, argv, env, P, wall=wall, cpu=wall,
                text_cap=64<<20, rss_cap=4<<30, total_cap=128<<20, file_cap=64<<20)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), entry.get('exit_code'), err.read_text()[-3500:])
            return out.read_text(), err.read_text()
        try:
            assert (DEVICE/'driver').resolve().name == 'xe' and not (DEVICE/'devcoredump').exists()
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', out).group(1)
            record['journal_before_cursor'] = cursor
            help_out, help_err = run('unitrace-help', [str(provider/'unitrace'), '--help'])
            flags = ['--chrome-sycl-logging', '--chrome-call-logging', '--chrome-kernel-logging', '--pid', '--tid',
                     '--chrome-event-buffer-size', '65536', '--output-dir-path', str(trace)]
            for flag in flags:
                if flag.startswith('--'):
                    assert flag in help_out+help_err, flag
            argv = [str(provider/'unitrace'), *flags, 'python3', str(D/'runtime_child.py'), str(O), str(executable)]
            record['collector_argv'] = argv
            persist()
            out, err = run('collect-runtime', argv, 180)
            assert 'B570' in out and '64 changing cross-queue handoffs' in out and 'sticky asynchronous errors: PASS' in out
            real_exit = json.loads((O/'runtime-exit.json').read_text())
            assert real_exit['exit_code'] == 0 and real_exit['binary_sha256'] == record['runtime_binary_sha256']
            record['real_runtime_exit'] = real_exit
            files = sorted(trace.rglob('*.json'))
            assert files and all(x.stat().st_size <= 64<<20 for x in files)
            record['trace_files'] = [dict(path=str(x), bytes=x.stat().st_size, sha256=sha(x)) for x in files]
            record['collector_stdout'] = out
            record['collector_stderr'] = err
            record.update(passed=True, complete=True, collected=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            if cursor:
                try:
                    out, err = run('journal-after', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    entries = [json.loads(x) for x in out.splitlines() if x.strip()]
                    record.update(fresh_faults=[x for x in entries if FAULT.search(str(x.get('MESSAGE', '')))],
                        driver_dump_exists=(DEVICE/'devcoredump').exists(), boot_unchanged=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot)
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False, complete=False)
                except BaseException as error:
                    record.update(passed=False, complete=False, fault_check_error=repr(error))
            record['active'] = False
            persist()
        assert record['passed'] and record['complete']
        print(json.dumps(dict(collected=True, capability_qualified=False,
            required_next='validate actual native PID API/device events and expected65-kernel128-copy handoff timeline')), flush=True)


if __name__ == '__main__':
    main()
