#!/usr/bin/env python3
"""Serialized initial checks of the frozen own-fork source, not speed tests."""
import fcntl
import hashlib
import importlib.util
import json
import re
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
BUILD = B / 'main-pre-Xe-1eb-build-root-v2'
O = B / 'main-pre-Xe-1eb-gpu-parity-root-v1'
C = Path('/opt/intel/oneapi/compiler/2026.1')
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    build = json.loads((BUILD / 'record.json').read_text())
    assert build['passed'] and build['complete'] and not build['active']
    assert build['source_revision'] == '1eb89482a4afd20277ae0405780ed4f8eb98eb20'
    source = Path(build['source_path'])
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists()
        for item in build['source_manifest']:
            assert sha(source / item['path']) == item['sha256'], item['path']
        for name, digest in build['binaries'].items():
            assert sha(BUILD / 'build' / name) == digest, name
        O.mkdir()
        env = dict(build['environment'])
        env.update(ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
            UR_ADAPTERS_FORCE_LOAD=str(C / 'lib/libur_adapter_level_zero_v2.so.0'),
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:stderr',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
            UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
            ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
            ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOGGING_LEVEL='trace',
            ZEL_LOADER_LOG_CONSOLE='1', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='1',
            SYCL_CACHE_PERSISTENT='0', XDG_CACHE_HOME=str(O / 'cache'),
            NEOReadDebugKeys='1', EnableDirectSubmission='0',
            UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
            STRATA_WATCHDOG_S='300', STRATA_KV_PREFETCH='0', STRATA_KV_STAGE_OWN='1')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        record = dict(active=True, passed=False, complete=False, commands=[], environment=env,
            build_receipt_sha256=sha(BUILD / 'record.json'), binaries=build['binaries'],
            source_revision=build['source_revision'], source_path=str(source),
            recipe_sha256=sha(Path(__file__)), boot_id=boot, observed=[],
            performance_qualified=False, model_executed=False, physical_262144_qualified=False,
            limits=dict(wall_seconds_per_process=300, cpu_seconds=600,
                log_bytes=64 << 20, RSS_bytes=8 << 30, job_bytes=1 << 30),
            provider_pins={str(f): sha(f) for f in [C / 'lib/libsycl.so.9',
                C / 'lib/libur_loader.so.0', C / 'lib/libur_adapter_level_zero_v2.so.0',
                Path('/usr/lib/x86_64-linux-gnu/libze_loader.so.1')]},
            scope='Frozen pre-Xe source numerical and KV-streaming first checks; no model, throughput, new primary-source adoption or full-context qualification.',
            GPU_reset_or_service_change=False)
        def persist():
            (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(O / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        def run(label, argv, gpu=False):
            entry, out, err = owner.run(label, [str(x) for x in argv],
                env if gpu else build['environment'], P,
                wall=300 if gpu else 30, cpu=600 if gpu else 30,
                text_cap=64 << 20, rss_cap=8 << 30, total_cap=1 << 30,
                file_cap=64 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), entry.get('exit_code'), err.read_text()[-2500:])
            return out.read_text(), err.read_text()
        cursor = None
        def faults(label):
            out, _ = run(label, ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
            entries = [json.loads(line) for line in out.splitlines() if line.strip()]
            check = dict(label=label, entries_scanned=len(entries),
                fault_entries=[x for x in entries if FAULT.search(str(x.get('MESSAGE', '')))],
                driver_dump_exists=(DEVICE / 'devcoredump').exists(),
                boot_unchanged=Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot)
            record.setdefault('fault_checks', []).append(check)
            persist()
            assert not check['fault_entries'] and not check['driver_dump_exists'] and check['boot_unchanged'], check
        try:
            assert (DEVICE / 'driver').resolve().name == 'xe'
            assert not (DEVICE / 'devcoredump').exists()
            before, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', before).group(1)
            record['journal_before_cursor'] = cursor
            for label, exe, args, expected in [
                ('quantize-act', 'quantize_act_parity', ['--selftest'], ['quantize_act_parity OK', 'quantize_act: 0 failures']),
                ('verify-window', 'verify_parity', ['--selftest'], ['verify_parity: PASS']),
                ('qsa-xmx-tails', 'qsa_prompt_attn_parity', ['8192', '129', '1'], ['FAILURES: 0']),
                ('kv-stream-restore', 'kv_stream_parity', [], ['ring restore: identical', '\nPASS\n']),
            ]:
                record['GPU_executed'] = True
                out, err = run(label, [BUILD / 'build' / exe, *args], gpu=True)
                observed = dict(label=label, stdout=out, stderr_bytes=len(err.encode()),
                    stderr_sha256=hashlib.sha256(err.encode()).hexdigest(),
                    UR_success_mentions=err.count('UR_RESULT_SUCCESS'),
                    ZE_success_mentions=err.count('ZE_RESULT_SUCCESS'),
                    UR_errors=sorted(set(re.findall(r'UR_RESULT_ERROR_[A-Z_]+', err))),
                    validation_errors=[line for line in err.splitlines() if '<VALIDATION>[ERROR]' in line],
                    ZE_errors=[line for line in err.splitlines() if '] ERROR (' in line])
                record['observed'].append(observed)
                persist()
                faults('journal-after-' + label)
                assert all(x in out for x in expected), (label, out)
                assert 'urEnqueue' in err and observed['UR_success_mentions'], 'missing API progress'
                assert not observed['UR_errors'] and not observed['validation_errors'] and not observed['ZE_errors'], observed
                print(json.dumps(dict(passed=label, fresh_faults=0)), flush=True)
            for item in build['source_manifest']:
                assert sha(source / item['path']) == item['sha256']
            for name, digest in build['binaries'].items():
                assert sha(BUILD / 'build' / name) == digest
            record.update(passed=True, complete=True, post_source_and_binary_unchanged=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor:
                    faults('journal-final')
            except BaseException as error:
                record.update(passed=False, complete=False, final_fault_error=repr(error))
                raise
            finally:
                record['active'] = False
                persist()
        print(json.dumps(dict(passed=True, model_executed=False, performance_qualified=False)), flush=True)


if __name__ == '__main__':
    main()
