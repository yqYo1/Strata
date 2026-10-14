#!/usr/bin/env python3
"""Pinned old-engine HTTP first check and unchanged Niko stock speed suite."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = P / 'bench/results/2026-10-14-primary-model-baseline'
Q = P / 'bench/results/2026-10-14-pre-Xe-replay'
HERE = Path(__file__).parent
BUILD = B / 'main-pre-Xe-1eb-build-root-v2'
C = Path('/opt/intel/oneapi/compiler/2026.1')
U = Path('/home/yayoi/ghq/github.com/Niko1221/Strata/.worktree/bench-upstream-v0143-pristine-20261012')
PACK = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
NATIVE = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(name):
    archive = Q / 'receipts' / name / 'record.json.gz'
    item = next(x for x in json.loads((Q / 'receipts-index.json').read_text())
                if x['path'] == str(archive.relative_to(Q)))
    data = gzip.decompress(archive.read_bytes())
    assert sha(archive) == item['archive_sha256']
    assert hashlib.sha256(data).hexdigest() == item['decoded_sha256']
    return json.loads(data)


def main():
    mode = sys.argv[1]
    assert mode in ['short4096', 'stock4096', 'stock8192']
    debug = mode.startswith('short')
    chunk = 8192 if mode.endswith('8192') else 4096
    split = chunk == 8192
    O = B / ('main-pre-Xe-1eb-model-' + mode + '-root-v1')
    build = receipt('main-pre-Xe-1eb-build-root-v2')
    assert build['passed'] and build['complete'] and not build['active']
    assert build['source_revision'] == '1eb89482a4afd20277ae0405780ed4f8eb98eb20'
    # Preserve each earlier failed/incomplete attempt; only its qualified components are gates.
    small = receipt('main-pre-Xe-1eb-gpu-parity-root-v1')
    assert not small['active'] and {x['label'] for x in small['observed']} == {'quantize-act', 'verify-window'}
    qsa = receipt('main-pre-Xe-1eb-gpu-parity-root-v2')
    assert not qsa['active'] and {x['label'] for x in qsa['observed']} == {'qsa-fallback-layout1-tails'}
    kv = receipt('main-pre-Xe-1eb-kv-parity-root-v3')
    assert kv['passed'] and kv['complete'] and not kv['active']
    assert kv['binaries'] == build['binaries']
    assert all(not x['fault_entries'] and not x['driver_dump_exists'] and x['boot_unchanged'] for x in kv['fault_checks'])
    if not debug:
        gate = json.loads((B / 'main-pre-Xe-1eb-model-short4096-root-v1/record.json').read_text())
        assert not gate['active'] and gate['passed'] and gate['complete']
        assert not gate['fresh_faults'] and not gate['driver_dump_exists'] and gate['boot_unchanged']
        assert gate['binary_sha256'] == build['binaries']['strata']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists()
        source = Path(build['source_path'])
        for item in build['source_manifest']:
            assert sha(source / item['path']) == item['sha256'], item['path']
        engine = Path(build['build_path']) / 'strata'
        assert sha(engine) == build['binaries']['strata']
        subprocess.run(['git', '-C', str(U), 'diff', '--exit-code', 'HEAD', '--', 'bench.sh',
            'tools/strata_bench.py', 'tools/ab_engine.py', 'tools/agent_turns_bench.py', 'tools/serve_load.py'], check=True)
        O.mkdir()
        env = dict(build['environment'])
        env.update(PYTHONPATH=str(D) + ':' + str(P / 'tools') + ':' + str(P),
            PYTHONUNBUFFERED='1', ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
            UR_ADAPTERS_FORCE_LOAD=str(C / 'lib/libur_adapter_level_zero_v2.so.0'),
            SYCL_CACHE_PERSISTENT='0', XDG_CACHE_HOME=str(O / 'cache'),
            NEOReadDebugKeys='1', EnableDirectSubmission='0', UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
            STRATA_WATCHDOG_S='300', STRATA_PREFILL_COMPACT='2', STRATA_PREFILL_ATTN_LAYOUT='1',
            STRATA_KV_STAGE_OWN='1', STRATA_KV_PREFETCH='0', STRATA_PREFILL_FIRST='0',
            STRATA_PREFILL_LAYER_MAJOR='0', STRATA_PREFILL_RING='8',
            STRATA_MEASUREMENT_ENGINE_EXIT=str(O / 'engine-exit.json'),
            STRATA_MEASUREMENT_ENGINE_START=str(O / 'engine-start.json'),
            STRATA_MEASUREMENT_WRAPPER_DEBUG='1' if debug else '0')
        if debug:
            # Prior tiny tests retained detailed successful API traces. The model's much larger
            # submission count uses flushed validation/warnings and native progress, per policy.
            env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
                ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
                ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOGGING_LEVEL='warning',
                ZEL_LOADER_LOG_CONSOLE='1', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0',
                STRATA_TRACE='1', STRATA_PREFILL_TIMING='1', STRATA_DECODE_TIMING='1')
        else:
            assert not any(k in env for k in ['UR_ENABLE_LAYERS', 'ZE_ENABLE_VALIDATION_LAYER',
                'STRATA_TRACE', 'STRATA_PREFILL_TIMING', 'STRATA_DECODE_TIMING', 'LD_PRELOAD'])
        args = ['--pack', str(PACK), '--native', str(NATIVE), '--max-context', '262144',
            '--kv', 'int8', '--kv-resident', '32768', '--expert-cache', '128',
            '--expert-profile', str(P / 'data/expert-profile.bin'), '--expert-cache-per-layer',
            '--pool-workers', '5', '--pcie-frac', '0', '--adapt-swaps', '0',
            '--conversation-cache-mib', '0', '--prompt-cache', '0', '--turn-token', '-1',
            '--suffix-draft', '0', '--greedy', '--spec-split' if split else '--no-spec-split',
            '--short-read', '0', '--spec', '4', '--mtp', '/home/yayoi/.local/share/strata-sycl/mtp/rt',
            '--prefill', str(chunk), '--no-prefill-borrow', '--vram-reserve-mib', '700']
        supported = set(re.findall(r'a == "(--[^"]+)"', (source / 'sycl/src/program/generate.cpp').read_text()))
        assert not {x for x in args if x.startswith('--')} - supported
        wrapper = HERE / 'engine_wrapper_reference.py'
        config = dict(exe=str(wrapper), args=args, cwd=str(P), tokenizer=str(PACK / 'tokenizer'),
            model_name='qwen3.8-flash-next-iq3_s-primary', aliases=['strata'], host='127.0.0.1',
            log=str(O / 'owned/engine.stderr'), engine_silence_s=480, parallel=1, open_browser=False)
        (O / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (O / 'plan.json').write_text(json.dumps(dict(port=18096, url='http://127.0.0.1:18096',
            expected_long_prompt_tokens=61055, long_soft_deadline_seconds=1500,
            bench_timeout_seconds=6600), indent=2) + '\n')
        job = D / ('run_server_job_v5.py' if debug else 'run_server_job_v4.py')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        record = dict(active=True, passed=False, complete=False, commands=[], environment=env,
            source_revision=build['source_revision'], build_receipt_sha256=sha(Q / 'receipts/main-pre-Xe-1eb-build-root-v2/record.json.gz'),
            source_manifest=build['source_manifest'], source_path=str(source),
            primary_revision=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip(),
            primary_status_at_launch=subprocess.check_output(['git', '-C', str(P), 'status', '--porcelain'], text=True),
            binary_sha256=sha(engine), engine_args=args, mode=mode,
            recipes={str(f.relative_to(P)): sha(f) for f in [Path(__file__), wrapper, job,
                D / 'engine_wrapper_v5.py', D / 'engine_wrapper_v3.py',
                D / 'owned_thread_sampler.py', D / 'bounded_stderr_relay.py']},
            standard_client_source={str(f.relative_to(U)): sha(f) for f in [U / 'bench.sh',
                U / 'tools/strata_bench.py', U / 'tools/ab_engine.py', U / 'tools/agent_turns_bench.py', U / 'tools/serve_load.py']},
            pack_metadata_sha256={str(f): sha(f) for f in [PACK / 'index.txt', PACK / 'native_experts.txt',
                PACK / 'conversions.json', PACK / 'tokenizer/chat_template.jinja', P / 'data/expert-profile.bin']},
            MTP_metadata_sha256={str(f): sha(f) for f in [Path('/home/yayoi/.local/share/strata-sycl/mtp/rt/dense.txt')]},
            model_identity={str(f): dict(bytes=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns) for f in NATIVE.parent.glob('*.gguf')},
            provider_pins={str(f): sha(f) for f in [C / 'lib/libsycl.so.9', C / 'lib/libur_loader.so.0',
                C / 'lib/libur_adapter_level_zero_v2.so.0', Path('/usr/lib/x86_64-linux-gnu/libze_loader.so.1')]},
            boot_id=boot, limits=dict(wall_seconds=900 if debug else 7200, log_bytes=64 << 20,
                RSS_bytes=88 << 30, total_job_bytes=1 << 30, native_watchdog_seconds=300),
            performance_qualified=False, quality_qualified=False, physical_262144_qualified=False,
            source_or_math_patches=[], GPU_reset_or_service_change=False,
            scope='Own-fork1eb reference, not pristine upstream. Same current HTTP server, pinned unchanged Niko client; old coherent oneAPI stack, compact2/layout1. 4096/splitOFF arm matches current flags;8192/splitON is separate historical configuration. No full-input or quality adoption.')
        def persist():
            (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(O / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        def utility(label, argv):
            entry, out, err = owner.run(label, argv, build['environment'], P, wall=30,
                text_cap=64 << 20, total_cap=1 << 30)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'))
            return out.read_text()
        cursor = None
        try:
            assert (DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists()
            before = utility('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', before).group(1)
            record['journal_before_cursor'] = cursor
            entry, out, err = owner.run('reference-model-' + mode,
                ['/home/yayoi/.local/share/strata-sycl/venv/bin/python', str(job), str(O), 'short-diagnostic' if debug else 'stock-bench'],
                env, P, wall=record['limits']['wall_seconds'], cpu=7200, text_cap=64 << 20,
                rss_cap=88 << 30, total_cap=1 << 30, file_cap=64 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (entry.get('error'), entry['exit_code'], err.read_text()[-2000:])
            result = json.loads((O / 'result.json').read_text())
            assert result['passed'] and result['real_engine_exit']['exit_code'] == 0
            record['result'] = result
            if debug:
                sampler = json.loads((O / 'sampler-summary.json').read_text())
                relay = json.loads((O / 'stderr-relay-summary.json').read_text())
                assert sampler['passed'] and sampler['samples'] > 0
                assert relay['passed'] and not relay['error_counts'] and not relay['oversized_lines'] and not relay.get('selected_cap_reached')
                record.update(sampler_summary=sampler, stderr_relay_summary=relay)
            else:
                files = list((O / 'stock-results').glob('*/results.json'))
                assert len(files) == 1
                raw = files[0].read_bytes()
                stock = json.loads(raw)
                assert stock['params']['runs'] == 3 and stock['params']['long'] and not stock['params']['started_server']
                assert not stock['results']['failures']
                assert not any(k in stock['results']['skipped'] for k in ['4K', '16K', '32K', 'agent'])
                rows = stock['results']['prompts']['32K']['requests']
                assert len(rows) == 3 and {x['run'] for x in rows} == {0, 1, 2}
                for row in rows:
                    assert row['ok'] and row['prompt_tokens'] >= 30000 and row['reused_tokens'] == 0 and row['generated_tokens'] > 0
                    assert all(row[k] is not None and math.isfinite(row[k]) and row[k] > 0 for k in ['prompt_tps', 'decode_tps', 'prompt_ms', 'decode_ms'])
                record.update(stock_result=stock, stock_result_path=str(files[0]), stock_result_sha256=hashlib.sha256(raw).hexdigest(),
                    long_phase_samples=rows, prefill_median_tps=statistics.median(x['prompt_tps'] for x in rows),
                    decode_median_tps=statistics.median(x['decode_tps'] for x in rows), performance_qualified=True)
            for item in build['source_manifest']:
                assert sha(source / item['path']) == item['sha256']
            assert sha(engine) == record['binary_sha256']
            record.update(passed=True, complete=True, post_source_and_binary_unchanged=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor:
                    journal = utility('journal-after', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    entries = [json.loads(line) for line in journal.splitlines() if line.strip()]
                    record['fresh_faults'] = [x for x in entries if FAULT.search(str(x.get('MESSAGE', '')))]
                    record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                    record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False, complete=False)
            except BaseException as error:
                record.update(passed=False, complete=False, final_fault_error=repr(error))
                raise
            finally:
                record['active'] = False
                persist()
        assert record['passed'] and record['complete']
        print(json.dumps(dict(passed=True, mode=mode, fresh_faults=0,
            prefill_median_tps=record.get('prefill_median_tps'), decode_median_tps=record.get('decode_median_tps'))), flush=True)


if __name__ == '__main__':
    main()
