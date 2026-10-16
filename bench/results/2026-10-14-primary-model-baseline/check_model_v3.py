#!/usr/bin/env python3
"""Bounded initial primary model/HTTP diagnostic, separate from speed samples."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import re
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = P / 'bench/results/2026-10-14-primary-model-baseline'
O = B / 'main-primary-model-diagnostic-root-v3'
U = Path('/home/yayoi/ghq/github.com/Niko1221/Strata/.worktree/bench-upstream-v0143-pristine-20261012')
T = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
PACK = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
NATIVE = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    final = json.loads(gzip.decompress((P / 'bench/results/2026-10-14-primary-native-runtime/receipts/main-primary-native-engine-build-root-v13/record.json.gz').read_bytes()))
    assert final['passed'] and not final['active']
    for name, digest in final['source_pins'].items():
        assert sha(P / name) == digest, name
    engine = Path(final['build_path']) / 'strata'
    assert sha(engine) == final['binaries']['strata']
    assert not O.exists()
    O.mkdir()
    env = dict(final['environment'])
    env['PYTHONPATH'] = str(P / 'tools') + ':' + str(P)
    env.update(PYTHONUNBUFFERED='1', ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
               UR_ADAPTERS_FORCE_LOAD=str(T / 'lib/libur_adapter_level_zero_v2.so.0'),
               UR_ENABLE_LAYERS='UR_LAYER_FULL_VALIDATION',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
               UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
               UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
               ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
               ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOGGING_LEVEL='warn', ZEL_LOADER_LOG_CONSOLE='1',
               SYCL_CACHE_PERSISTENT='0', XDG_CACHE_HOME=str(O / 'cache'),
               NEOReadDebugKeys='1', EnableDirectSubmission='0', UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
               STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0', STRATA_TRACE='1',
               STRATA_PREFILL_TIMING='1', STRATA_DECODE_TIMING='1',
               STRATA_MEASUREMENT_ENGINE_EXIT=str(O / 'engine-exit.json'))
    args = ['--pack', str(PACK), '--native', str(NATIVE), '--max-context', '262144',
            '--kv', 'int8', '--kv-resident', '32768', '--expert-cache', '128',
            '--expert-profile', str(P / 'data/expert-profile.bin'), '--expert-cache-per-layer',
            '--pool-workers', '5', '--pcie-frac', '0', '--adapt-swaps', '0',
            '--conversation-cache-mib', '0', '--prompt-cache', '0', '--turn-token', '-1',
            '--suffix-draft', '0', '--greedy', '--no-spec-split', '--short-read', '0', '--spec', '4', '--mtp', '/home/yayoi/.local/share/strata-sycl/mtp/rt',
            '--prefill', '4096', '--no-prefill-borrow', '--vram-reserve-mib', '700']
    supported = set(re.findall(r'a == \"(--[^\"]+)\"', (P / 'src/program/generate.cpp').read_text()))
    assert not {a for a in args if a.startswith('--')} - supported, 'unknown primary CLI flag'
    assert Path(args[args.index('--mtp') + 1], 'index.txt').is_file()
    assert int(args[args.index('--spec') + 1]) > 0 and int(args[args.index('--prefill') + 1]) > 0
    config = dict(exe=str(D / 'engine_wrapper.py'), args=args, cwd=str(P),
                  tokenizer=str(PACK / 'tokenizer'), model_name='qwen3.8-flash-next-iq3_s-primary',
                  aliases=['strata'], host='127.0.0.1', log=str(O / 'owned/engine.stderr'),
                  engine_silence_s=180, parallel=1, open_browser=False)
    (O / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (O / 'plan.json').write_text(json.dumps(dict(port=18096, url='http://127.0.0.1:18096'), indent=2) + '\n')
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    record = dict(active=True, passed=False, complete=False, commands=[], environment=env,
                  engine_args=args, source_pins=final['source_pins'], binary_sha256=sha(engine),
                  recipes={str(f.relative_to(P)): sha(f) for f in [D / 'engine_wrapper.py', D / 'run_server_job.py', D / 'check_model_v3.py']},
                  standard_client_source={str(f.relative_to(U)): sha(f) for f in [U / 'bench.sh', U / 'tools/strata_bench.py',
                     U / 'tools/ab_engine.py', U / 'tools/agent_turns_bench.py', U / 'tools/serve_load.py']},
                  model_identity={str(f): dict(bytes=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns)
                                  for f in NATIVE.parent.glob('*.gguf')},
                  pack_metadata_sha256={str(f): sha(f) for f in [PACK / 'index.txt', PACK / 'native_experts.txt',
                     PACK / 'conversions.json', PACK / 'tokenizer/chat_template.jinja', P / 'data/expert-profile.bin']},
                  boot_id=boot, limits=dict(wall_seconds=1200, cpu_seconds=2400, raw_text_bytes=64 << 20,
                      RSS_bytes=88 << 30, total_job_bytes=1 << 30),
                  performance_qualified=False, physical_262144_qualified=False,
                  scope='initial model/HTTP plus >=32K actual long diagnostic; spec4 with required MTP pack, validation/timing overhead; no speed adoption',
                  GPU_reset_or_service_change=False)
    def persist():
        (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    owner = module.Owner(O / 'owned', persist)
    def utility(label, argv):
        entry, out, err = owner.run(label, argv, final['environment'], P, wall=30, cpu=30, total_cap=1 << 30)
        record['commands'].append(entry)
        assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'))
        return out.read_text()
    cursor = None
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            assert (DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists()
            before = utility('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', before).group(1)
            record['journal_before_cursor'] = cursor
            entry, out, err = owner.run('primary-model-diagnostic',
                ['/home/yayoi/.local/share/strata-sycl/venv/bin/python', str(D / 'run_server_job.py'), str(O), 'diagnostic'],
                env, P, wall=1200, cpu=2400, text_cap=64 << 20, rss_cap=88 << 30,
                total_cap=1 << 30, file_cap=64 << 20)
            record['commands'].append(entry)
            assert module.closed(entry) and entry['exit_code'] == 0, (entry.get('error'), entry['exit_code'], err.read_text()[-3000:], out.read_text()[-3000:])
            result = json.loads((O / 'result.json').read_text())
            assert result['passed'] and result['real_engine_exit']['exit_code'] == 0
            record['result'] = result
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            if cursor:
                journal = utility('journal-after', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                entries = [json.loads(line) for line in journal.splitlines() if line.strip()]
                hits = [x for x in entries if FAULT.search(str(x.get('MESSAGE', '')))]
                record['fresh_faults'] = hits
                record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot
                if hits or record['driver_dump_exists'] or not record['boot_unchanged']:
                    record.update(passed=False, complete=False)
            record['active'] = False
            persist()
    assert record['passed'] and record['complete'], record.get('error')
    print(json.dumps(dict(passed=record['passed'], requests=record['result']['requests'], fresh_faults=0)), flush=True)


if __name__ == '__main__':
    main()
