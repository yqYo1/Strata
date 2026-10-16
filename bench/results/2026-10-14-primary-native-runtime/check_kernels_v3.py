#!/usr/bin/env python3
"""Serial, bounded primary kernel checks; stop on the first failed diagnostic."""
import fcntl
import hashlib
import importlib.util
import json
import re
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
BUILD = B / 'main-primary-native-build-root-v1'
ENGINE = B / 'main-primary-native-engine-build-root-v13'
O = B / 'main-primary-native-kernels-root-v3'
T = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(text):
    """API success and diagnostics are distinct; do not declare all logs clean."""
    validation = [line for line in text.splitlines() if '<VALIDATION>[ERROR]' in line]
    known = '<VALIDATION>[ERROR]: There are no valid references to handle 0'
    ze_errors = [line for line in text.splitlines() if '] ERROR (' in line]
    query_false = [line for line in ze_errors if
                   'ERROR (2013265955) in zeCommandListIsGraphCaptureEnabledExt' in line]
    return dict(UR_errors=sorted(set(re.findall(r'UR_RESULT_ERROR_[A-Z_]+', text))),
                UR_success_mentions=text.count('UR_RESULT_SUCCESS'),
                validation_error_count=len(validation),
                known_optional_null_pool_false_positives=sum(line.strip() == known for line in validation),
                other_validation_errors=[line for line in validation if line.strip() != known],
                ZE_query_false_count=len(query_false),
                other_ZE_errors=[line for line in ze_errors if line not in query_false],
                full_validation_clean=not validation and not ze_errors)


def main():
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = json.loads((ENGINE / 'record.json').read_text())
    assert build['passed'] and build['complete'] and not build['active']
    for path, digest in build['source_pins'].items():
        assert sha(P / path) == digest
    for name, digest in build['binaries'].items():
        assert sha(BUILD / 'build' / name) == digest
    assert not O.exists()
    O.mkdir()
    env = dict(build['environment'])
    env.update(ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
               UR_ADAPTERS_FORCE_LOAD=str(T / 'lib/libur_adapter_level_zero_v2.so.0'),
               UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_FULL_VALIDATION',
               UR_LOG_TRACING='level:info;flush:info;output:stderr',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
               UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
               UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
               ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
               ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOGGING_LEVEL='trace',
               ZEL_LOADER_LOG_CONSOLE='1', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='1',
               SYCL_CACHE_PERSISTENT='0', XDG_CACHE_HOME=str(O / 'cache'),
               NEOReadDebugKeys='1', EnableDirectSubmission='0',
               UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    record = dict(active=True, passed=False, complete=False, GPU_executed=False,
                  model_executed=False, performance_qualified=False, physical_256K_qualified=False,
                  HA_exp_parity_qualified=False, source_pins=build['source_pins'], binaries=build['binaries'],
                  environment=env, commands=[], observed=[], boot_id=boot,
                  reused_runtime_proof=str(B / 'main-primary-native-runtime-root-v1/record.json'),
                  limits=dict(wall_seconds_per_GPU_process=180, text_bytes=64 << 20,
                              job_bytes=1 << 30, rss_bytes=8 << 30),
                  safety='GEMM: small two-queue then realistic single-queue; no known failed large two-queue case',
                  known_diagnostic_reference='research/research-primary-ur-null-pool-validation-round453-v1.txt',
                  consumer='primary compiler/API compatibility and selected kernel correctness before model baseline')
    def persist():
        (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    owner = module.Owner(O / 'owned', persist)
    cursor = None
    def run(label, argv, gpu=False, extra=None):
        use = dict(env if gpu else build['environment'])
        if extra:
            use.update(extra)
        entry, out, err = owner.run(label, [str(a) for a in argv], use, P, wall=180 if gpu else 30,
                                   cpu=180 if gpu else 30, text_cap=64 << 20,
                                   total_cap=1 << 30, rss_cap=8 << 30)
        record['commands'].append(entry)
        assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), entry['exit_code'], err.read_text()[-3000:])
        return out.read_text(), err.read_text()
    def faults(label):
        out, err = run(label, ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
        entries = [json.loads(line) for line in out.splitlines() if line.strip()]
        hits = [j for j in entries if FAULT.search(str(j.get('MESSAGE', '')))]
        result = dict(label=label, entries_scanned=len(entries), fault_entries=hits,
                      driver_dump_exists=(DEVICE / 'devcoredump').exists(),
                      boot_unchanged=Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot,
                      journal_stderr=err)
        record.setdefault('fault_checks', []).append(result)
        persist()
        assert not hits and not result['driver_dump_exists'] and result['boot_unchanged'], result
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            assert (DEVICE / 'driver').resolve().name == 'xe'
            assert not (DEVICE / 'devcoredump').exists(), 'existing dump: do not launch'
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', out).group(1)
            record['journal_before_cursor'] = cursor
            record['prior_failure'] = str(B / 'main-primary-native-kernels-root-v1/record.json')
            record['scope'] = 'Affected Q2 quantizer/hit dispatch after RN division correction; GEMM already passed v2; check only quantizer/dispatch affected by invalid-range guard'
            cases = [
                ('native-single-dispatch', 'sycl_native_single_dispatch_test', {}, ['invalid id PASS']),
            ]
            for label, exe, extra, required in cases:
                print(json.dumps(dict(starting=label)), flush=True)
                record['GPU_executed'] = True
                out, err = run(label, [BUILD / 'build' / exe], gpu=True, extra=extra)
                assert all(token in out for token in required), (label, out)
                diagnostic = classify(err)
                observed = dict(label=label, stdout=out, diagnostic=diagnostic,
                                stderr_sha256=hashlib.sha256(err.encode()).hexdigest(), stderr_bytes=len(err.encode()))
                record['observed'].append(observed)
                faults('journal-after-' + label)
                assert diagnostic['UR_success_mentions'] > 0, 'actual UR API log absent'
                assert not diagnostic['UR_errors'] and not diagnostic['other_validation_errors'] and not diagnostic['other_ZE_errors'], observed
                print(json.dumps(dict(passed=label, fresh_faults=0, diagnostic=diagnostic)), flush=True)
            record.update(passed=True, complete=True)
        except Exception as error:
            record['error'] = repr(error)
            if cursor:
                try:
                    faults('journal-after-failure')
                except Exception as after_error:
                    record['after_failure_fault_error'] = repr(after_error)
            raise
        finally:
            record['active'] = False
            persist()
    print(json.dumps(dict(passed=record['passed'], scope='selected primary kernel fixtures; no HA/model/speed/full-context proof')), flush=True)


if __name__ == '__main__':
    main()
