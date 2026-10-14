#!/usr/bin/env python3
"""Main-owned serialized build, lifetime fixture and actual transfer checks."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import re
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
BASE = P / 'bench/results/2026-10-14-primary-model-baseline'
SEED = BASE / 'receipts/main-primary-watchdog-lifecycle-build-root-v4/record.json.gz'
STOCK = B / 'main-primary-stock-bench-accepted-root-v2/record.json'
O = B / 'main-primary-transfer-ceiling-root-v1'
BUILD = B / 'main-primary-native-build-root-v1/build'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
T = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
METADATA = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/native_experts.txt')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_samples(rows, mode, shapes):
    samples = [x for x in rows if x['type'] == 'sample']
    summaries = [x for x in rows if x['type'] == 'bulk_summary']
    assert rows[0]['type'] == 'run' and 'B570' in rows[0]['device']
    assert rows[-1]['type'] == 'complete' and rows[-1]['status'] == 'pass'
    assert rows[-1]['cleanup'] == 'checked'
    assert not any(x['type'] in ('failure', 'cleanup_error') for x in rows)
    assert len(samples) == (12 if mode == 'diagnostic' else 48 + 36 * len(shapes))
    assert len(summaries) == (0 if mode == 'diagnostic' else 12)
    keys = set()
    for x in samples:
        key = tuple(x[k] for k in ('lane', 'allocation', 'direction', 'blob_bytes', 'copies', 'rep', 'burst'))
        assert key not in keys
        keys.add(key)
        assert x['rep'] in (0, 1, 2) and x['status'] == 'pass'
        assert all(x[k] for k in ('verified_full_payload', 'verified_all_corpus_slots', 'verified_guards'))
        assert x['payload_bytes'] == x['blob_bytes'] * x['copies']
        bulk = x['lane'] == 'bulk'
        assert x['timed_api_counts'] == dict(copy=4 if bulk else x['copies'],
            copy_bytes=4*x['payload_bytes'] if bulk else x['payload_bytes'],
            event_record=5 if bulk else 2, event_sync=1, event_elapsed=0)
        assert x['post_completion_api_counts']['event_elapsed'] == (4 if bulk else 1)
        assert len(x['copy_api_ms']) == x['copies']
        for k in ('event_ms', 'host_group_completion_ms', 'event_GBps', 'host_group_GBps'):
            assert math.isfinite(x[k]) and x[k] > 0, (key, k)
        assert abs(x['event_GBps'] - x['payload_bytes']/x['event_ms']/1e6) < 1e-7
        assert abs(x['host_group_GBps'] - x['payload_bytes']*(4 if bulk else 1)/x['host_group_completion_ms']/1e6) < 1e-7
        if bulk:
            assert x['blob_bytes'] == 256 << 20 and x['copies'] == 1
            assert x['allocation_bytes_per_buffer'] == (256 << 20) + 128
        else:
            assert x['blob_bytes'] in shapes and x['copies'] in ((1,) if mode == 'diagnostic' else (1, 8, 32))
            assert x['allocation_bytes_per_buffer'] <= 256 << 20
    return dict(sample_count=len(samples), bulk_summary_count=len(summaries), all_individual_samples=samples,
                all_bulk_summaries=summaries, complete=rows[-1])


def main():
    seed = json.loads(gzip.decompress(SEED.read_bytes()))
    stock = json.loads(STOCK.read_text())
    assert not stock['active'] and stock['passed'] and stock['complete']
    assert not stock['fresh_faults'] and not stock['driver_dump_exists']
    assert seed['passed'] and seed['complete'] and not seed['active']
    old_cmake = subprocess.check_output(['git', '-C', str(P), 'show',
        stock['source_revision']+':cmake/sycl_backend.cmake'])
    assert hashlib.sha256(old_cmake).hexdigest() == seed['source_pins']['cmake/sycl_backend.cmake']
    anchor = b'add_library(strata_sycl_runtime STATIC src/sycl/runtime.cpp src/sycl/compat.cpp)\n'
    addition = (b'add_executable(sycl_transfer_ceiling EXCLUDE_FROM_ALL\n'
        b'  ${PROJECT_SOURCE_DIR}/bench/results/2026-10-14-primary-transfer-ceiling/transfer_ceiling.cpp)\n'
        b'target_link_libraries(sycl_transfer_ceiling PRIVATE strata_sycl_runtime)\n'
        b'target_compile_features(sycl_transfer_ceiling PRIVATE cxx_std_17)\n\n')
    assert old_cmake.count(anchor) == 1
    assert (P/'cmake/sycl_backend.cmake').read_bytes() == old_cmake.replace(anchor, anchor+addition)
    for name, digest in seed['source_pins'].items():
        if name != 'cmake/sycl_backend.cmake':
            assert sha(P/name) == digest, name
    assert sha(BUILD/'strata') == seed['binaries']['strata']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists()
        O.mkdir()
        env = dict(seed['environment'])
        env.update(ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
            UR_ADAPTERS_FORCE_LOAD=str(T/'lib/libur_adapter_level_zero_v2.so.0'),
            NEOReadDebugKeys='1', EnableDirectSubmission='0', UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
            SYCL_CACHE_PERSISTENT='0', XDG_CACHE_HOME=str(O/'cache'))
        debug = dict(env, UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_FULL_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:stderr',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
            UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
            ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
            ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOGGING_LEVEL='trace',
            ZEL_LOADER_LOG_CONSOLE='1', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='1')
        pins = dict(seed['source_pins'])
        pins['cmake/sycl_backend.cmake'] = sha(P/'cmake/sycl_backend.cmake')
        pins.update({str(f.relative_to(P)):sha(f) for f in D.iterdir() if f.is_file()})
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        record = dict(active=True, passed=False, complete=False, commands=[], source_pins=pins,
            source_revision=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip(),
            source_diff=subprocess.check_output(['git', '-C', str(P), 'diff', '--', 'cmake/sycl_backend.cmake'], text=True),
            environment=env, diagnostic_environment=debug, boot_id=boot, fault_filter=FAULT.pattern,
            seed_receipt_sha256=sha(SEED), native_binary_sha256=sha(BUILD/'strata'),
            scope='actual warmed transfer ceiling and real expert-extent copy shape; not model speed or overlap proof',
            limits=dict(text_bytes=64<<20, job_bytes=1<<30, RSS_bytes=4<<30, diagnostic_wall_seconds=360,
                        quiet_wall_seconds=1860, helper_diagnostic_deadline_seconds=300, helper_quiet_deadline_seconds=1800),
            model_payload_read=False, service_or_GPU_reset=False)
        def persist():
            (O/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(O/'owned', persist)
        record['commands'] = owner.commands
        cursor = None
        persist()
        def run(label, argv, wall=120, use=None, expected=0):
            entry, out, err = owner.run(label, argv, use or env, P, wall=wall, cpu=wall,
                text_cap=64<<20, rss_cap=4<<30, total_cap=1<<30)
            assert module.closed(entry) and entry['exit_code'] == expected, (label, entry.get('error'), entry.get('exit_code'), err.read_text()[-2500:])
            return out.read_text(), err.read_text()
        def faults(label):
            out, err = run(label, ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'], 30)
            entries = [json.loads(x) for x in out.splitlines() if x.strip()]
            check = dict(label=label, entries_scanned=len(entries), faults=[x for x in entries if FAULT.search(str(x.get('MESSAGE', '')))],
                driver_dump_exists=(DEVICE/'devcoredump').exists(), boot_unchanged=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,
                journal_stderr=err)
            record.setdefault('fault_checks', []).append(check)
            persist()
            assert not check['faults'] and not check['driver_dump_exists'] and check['boot_unchanged'], check
        try:
            assert (DEVICE/'driver').resolve().name == 'xe' and not (DEVICE/'devcoredump').exists()
            run('configure-helper', ['cmake', '-S', str(P), '-B', str(BUILD)], 180)
            run('build-helper', ['cmake', '--build', str(BUILD), '--target', 'sycl_transfer_ceiling', '--parallel', '2'], 600)
            helper = BUILD/'sycl_transfer_ceiling'
            record['helper_binary_sha256'] = sha(helper)
            compile_commands = json.loads((BUILD/'compile_commands.json').read_text())
            helper_compile = [x for x in compile_commands if Path(x['file']) == D/'transfer_ceiling.cpp']
            assert len(helper_compile) == 1 and '-fsycl-id-queries-range=size_t' in helper_compile[0]['command']
            assert '-fsycl ' in helper_compile[0]['command']
            record['helper_compile_command'] = helper_compile[0]
            record['provider_pins'] = {str(f):sha(f) for f in [T/'bin/clang++', T/'lib/libsycl.so.9',
                T/'lib/libur_loader.so.0', T/'lib/libur_adapter_level_zero_v2.so.0',
                Path('/usr/lib/x86_64-linux-gnu/libze_loader.so.1')]}
            host = O/'failure-guard-host'
            run('compile-failure-guard', [str(T/'bin/clang++'), '-std=c++17', '-O1', '-g', str(D/'failure_guard_host.cpp'), '-o', str(host)])
            for mode, code in [('pending-error', 1), ('closed-error', 7), ('normal', 0)]:
                out, err = run('guard-'+mode, [str(host), mode], 30, expected=code)
                if mode == 'pending-error':
                    assert 'preserving copy buffers' in err and 'host-copy-source-destroyed' not in err and 'outer-catch' not in err
                else:
                    assert 'host-copy-source-destroyed' in err
                    assert ('outer-catch' in err) == (mode == 'closed-error')
            record['host_lifetime_fixture_passed'] = True
            out, _ = run('extract-extents', ['python3', str(D/'extract_extents.py'), str(METADATA), '--extents-output', str(O/'extents.tsv')], 30)
            metadata = json.loads(out)
            assert metadata['metadata_sha256'] == sha(METADATA)
            assert metadata['payload_files_opened'] == 0 and metadata['payload_bytes_read'] == 0
            record['metadata'] = metadata
            record['extent_tsv_sha256'] = sha(O/'extents.tsv')
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], 30)
            cursor = re.search(r'-- cursor: (\S+)', out).group(1)
            record['journal_before_cursor'] = cursor
            for mode, wall, deadline, use in [('diagnostic', 360, 300, debug), ('all', 1860, 1800, env)]:
                persist()
                out, err = run('transfer-'+mode, [str(helper), str(O/'extents.tsv'), '3', str(deadline), mode], wall, use)
                rows = [json.loads(x) for x in out.splitlines()]
                shapes = [metadata['min']] if mode == 'diagnostic' else metadata['representatives']
                evidence = validate_samples(rows, mode, shapes)
                record[mode] = evidence
                if mode == 'diagnostic':
                    assert 'urEnqueueUSMMemcpy' in err and 'UR_RESULT_SUCCESS' in err
                    assert 'ZE_RESULT_SUCCESS' in err
                    errors = sorted(set(re.findall(r'UR_RESULT_ERROR_[A-Z_]+', err)))
                    evidence.update(UR_errors=errors, UR_success_mentions=err.count('UR_RESULT_SUCCESS'), ZE_success_mentions=err.count('ZE_RESULT_SUCCESS'))
                    assert not errors
                else:
                    assert not any(k in use for k in ('UR_ENABLE_LAYERS', 'ZE_ENABLE_VALIDATION_LAYER', 'ZE_ENABLE_PARAMETER_VALIDATION', 'LD_PRELOAD'))
                faults('journal-after-'+mode)
                print(json.dumps(dict(passed=mode, samples=evidence['sample_count'], fresh_faults=0)), flush=True)
            assert sha(BUILD/'strata') == record['native_binary_sha256'], 'native inference executable changed'
            for name, digest in pins.items():
                assert sha(P/name) == digest, name
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            if cursor:
                try:
                    faults('journal-after-failure')
                except Exception as fault_error:
                    record['after_failure_fault_error'] = repr(fault_error)
            raise
        finally:
            record['active'] = False
            persist()
        print(json.dumps(dict(passed=True, model_speed_qualified=False)), flush=True)


if __name__ == '__main__':
    main()
