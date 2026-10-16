#!/usr/bin/env python3
"""Bounded small primary GPU tests with actual API/progress/fault evidence."""
import fcntl
import hashlib
import gzip
import importlib.util
import json
import re
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
BUILD = B / 'main-primary-native-build-root-v1'
O = B / 'main-primary-accepted-mmq-gpu-root-v1'
T = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = json.loads(gzip.decompress((P / 'bench/results/2026-10-14-primary-model-baseline/receipts/main-primary-size-t-range-build-root-v2/record.json.gz').read_bytes()))
    assert build['passed'] and build['complete'] and not build['active']
    for path, digest in build['source_pins'].items():
        assert sha(P / path) == digest
    for path, digest in build['binaries'].items():
        assert sha(BUILD / 'build' / path) == digest
    assert not O.exists()
    O.mkdir()
    env = dict(build['environment'])
    env.update(ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
               UR_ADAPTERS_FORCE_LOAD=str(T / 'lib/libur_adapter_level_zero_v2.so.0'),
               UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_FULL_VALIDATION',
               STRATA_SYCL_MMQ_XMX='1', STRATA_SYCL_MMQ_XMX_TILE='8', STRATA_SYCL_MMQ_XMX_EXACT='1',
               STRATA_SYCL_MMQ_XMX_PACK='0', STRATA_SYCL_MMQ_XMX_COMPACT='1',
               UR_LOG_TRACING='level:info;flush:info;output:stderr',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
               UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
               UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
               ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
               ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOGGING_LEVEL='trace',
               ZEL_LOADER_LOG_CONSOLE='1', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='1',
               SYCL_CACHE_PERSISTENT='0', XDG_CACHE_HOME=str(O / 'cache'), NEOReadDebugKeys='1', EnableDirectSubmission='0', UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    record = dict(active=True, passed=False, complete=False, GPU_executed=False,
                  model_executed=False, performance_qualified=False, physical_256K_qualified=False,
                  source_pins=build['source_pins'], binaries=build['binaries'], commands=[], environment=env,
                  boot_id=boot, fault_filter=FAULT.pattern, recipe_sha256=sha(Path(__file__)),
                  provider_pins={str(f): sha(f) for f in [T / 'lib/libsycl.so.9',
                     T / 'lib/libur_loader.so.0', T / 'lib/libur_adapter_level_zero_v2.so.0',
                     Path('/usr/lib/x86_64-linux-gnu/libze_loader.so.1')]},
                  limits=dict(wall_seconds_per_GPU_process=180, text_bytes=64 << 20,
                              job_bytes=1 << 30, rss_bytes=4 << 30),
                  observed=[], api_debug_sources=['https://oneapi-src.github.io/unified-runtime/core/INTRO.html',
                  'https://github.com/oneapi-src/level-zero/blob/master/README.md'])
    def persist():
        (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    owner = module.Owner(O / 'owned', persist)
    cursor = None
    def run(label, argv, gpu=False):
        use = env if gpu else build['environment']
        entry, out, err = owner.run(label, [str(a) for a in argv], use, P, wall=180 if gpu else 30,
                                    text_cap=64 << 20, total_cap=1 << 30, rss_cap=4 << 30)
        record['commands'].append(entry)
        assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), entry['exit_code'], err.read_text()[-3000:])
        return out.read_text(), err.read_text()
    def faults(label):
        out, err = run(label, ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
        entries = [json.loads(line) for line in out.splitlines() if line.strip()]
        hits = [j for j in entries if FAULT.search(str(j.get('MESSAGE', '')))]
        dump = (DEVICE / 'devcoredump').exists()
        result = dict(label=label, entries_scanned=len(entries), fault_entries=hits,
                      driver_dump_exists=dump, boot_unchanged=Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot,
                      journal_stderr=err)
        record.setdefault('fault_checks', []).append(result)
        persist()
        assert not hits and not dump and result['boot_unchanged'], result
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            assert (DEVICE / 'vendor').read_text().strip() == '0x8086'
            assert (DEVICE / 'device').read_text().strip() == '0xe20c'
            assert (DEVICE / 'driver').resolve().name == 'xe'
            assert not (DEVICE / 'devcoredump').exists(), 'existing GPU dump: do not launch'
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', out).group(1)
            record['journal_before_cursor'] = cursor
            for label, exe, args, required in [
                ('mmq-xmx-exact-compact-debug', 'sycl_prefill_mmq_test', [], ['SYCL quantized prefill PASS']),
            ]:
                record['GPU_executed'] = True
                out, err = run(label, [BUILD / 'build' / exe, *args], gpu=True)
                assert all(token in out for token in required), (label, out)
                assert 'urEnqueue' in err and 'UR_RESULT_SUCCESS' in err, 'UR trace not observed'
                errors = sorted(set(re.findall(r'UR_RESULT_ERROR_[A-Z_]+', err)))
                observed = dict(label=label, stdout=out, stderr_sha256=hashlib.sha256(err.encode()).hexdigest(),
                                stderr_bytes=len(err.encode()), UR_errors=errors,
                                UR_success_mentions=err.count('UR_RESULT_SUCCESS'),
                                ZE_success_mentions=err.count('ZE_RESULT_SUCCESS'),
                                phase_markers=[line for line in err.splitlines() if line.startswith(('runtime:', 'compat:'))])
                record['observed'].append(observed)
                faults('journal-after-' + label)
                other_validation = [line for line in err.splitlines() if '<VALIDATION>[ERROR]' in line and line.strip() != '<VALIDATION>[ERROR]: There are no valid references to handle 0']
                other_ze = [line for line in err.splitlines() if '] ERROR (' in line and 'ERROR (2013265955) in zeCommandListIsGraphCaptureEnabledExt' not in line]
                observed.update(other_validation_errors=other_validation, other_ZE_errors=other_ze, full_validation_clean=not any('<VALIDATION>[ERROR]' in line for line in err.splitlines()))
                assert not errors and not other_validation and not other_ze, observed
                print(json.dumps(dict(passed=label, UR_success_mentions=observed['UR_success_mentions'],
                                      ZE_success_mentions=observed['ZE_success_mentions'], fresh_faults=0)), flush=True)
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
    print(json.dumps(dict(passed=record['passed'], scope='current size_t compiler accepted exact tile8 compact XMX MMQ against preceding reduction; no model or speed proof')), flush=True)

if __name__ == '__main__':
    main()
