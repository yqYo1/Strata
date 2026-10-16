#!/usr/bin/env python3
"""Owned routed-observation off/on numeric pair; not speed or language quality."""
import array
import fcntl
import hashlib
import importlib.util
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE = Path(__file__).parent
D = P / 'bench/results/2026-10-14-primary-model-baseline'
BUILD = B / 'main-primary-routed-fp16-census-build-root-v1'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
T = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
PACK = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
NATIVE = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floats(path):
    raw = path.read_bytes()
    assert raw and len(raw) <= 16 << 20 and len(raw) % 4 == 0
    data = array.array('f')
    data.frombytes(raw)
    assert data.itemsize == 4 and sys.byteorder == 'little'
    assert all(math.isfinite(x) for x in data)
    return data


def residual(path):
    raw = path.read_bytes()
    stride = 8 + 10240 * 4
    assert len(raw) == stride * 3
    positions = [struct.unpack_from('<q', raw, i * stride)[0] for i in range(3)]
    assert positions == [0, 64, 128], positions
    data = array.array('f')
    for i in range(3):
        data.frombytes(raw[i * stride + 8:(i + 1) * stride])
    assert all(math.isfinite(x) for x in data)
    return data, positions


def compare(left, right):
    assert len(left) == len(right)
    errors = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    ref_square = sum(float(x) ** 2 for x in left)
    square = sum(x * x for x in errors)
    return dict(elements=len(left), finite=True, max_abs=max(errors),
        rms=math.sqrt(square / len(errors)), relative_rms=math.sqrt(square / max(ref_square, 1e-30)),
        changed_elements=sum(x != 0 for x in errors), numerical_exact=not any(errors))


def main():
    mode = sys.argv[1]
    assert mode == 'census4096'
    chunk = 4096
    threaded = mode == 'threaded-mmq64'
    O = B / 'main-primary-routed-census129-pair-root-v1'
    build = json.loads((BUILD / 'record.json').read_text())
    assert build['passed'] and build['complete'] and not build['active']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists()
        assert (DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists()
        for name, pin in build['source_pins'].items():
            assert sha(P / name) == pin, name
        engine = Path(build['build_path']) / 'strata'
        assert sha(engine) == build['binaries']['strata']
        O.mkdir()
        tokens = [1000 + ((i * 7) % 512) for i in range(130)]
        token_path = O / 'tokens.txt'
        token_path.write_text(','.join(str(x) for x in tokens) + '\n')
        base_env = dict(build['environment'])
        base_env.update(PYTHONPATH=str(D) + ':' + str(P / 'tools') + ':' + str(P),
            PYTHONUNBUFFERED='1', ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
            UR_ADAPTERS_FORCE_LOAD=str(T / 'lib/libur_adapter_level_zero_v2.so.0'),
            UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
            UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
            ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
            ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOGGING_LEVEL='warning',
            ZEL_LOADER_LOG_CONSOLE='1', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0',
            SYCL_CACHE_PERSISTENT='0', NEOReadDebugKeys='1', EnableDirectSubmission='0',
            UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', STRATA_WATCHDOG_S='300',
            STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0', STRATA_TRACE='1', STRATA_PREFILL_TIMING='1',
            STRATA_SYCL_MMQ_XMX='1', STRATA_SYCL_MMQ_XMX_TILE='8', STRATA_SYCL_MMQ_XMX_EXACT='1',
            STRATA_SYCL_MMQ_XMX_PACK='0', STRATA_SYCL_MMQ_XMX_COMPACT='1',
            STRATA_SYCL_PREFILL_SAME_QUEUE='0' if threaded else '1',
            STRATA_PREFILL_RING='16' if threaded else '8',
            STRATA_PREFILL_MMQ='1' if threaded else '0', STRATA_PREFILL_GROUP_GATHER='1')
        assert 'STRATA_DBG_NAN' not in base_env and 'LD_PRELOAD' not in base_env
        if threaded:
            base_env['STRATA_PREFILL_STREAM_MIN'] = '16'
        record = dict(active=True, passed=False, complete=False, commands=[], mode=mode,
            source_pins=build['source_pins'], binary_sha256=sha(engine), build_receipt_sha256=sha(BUILD / 'record.json'),
            source_revision=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip(),
            source_diff_sha256=build['source_diff_sha256'], recipe_sha256=sha(Path(__file__)),
            tokens=dict(path=str(token_path), sha256=sha(token_path), total=130, batched=129,
                scope='Deterministic valid synthetic token IDs; numerical diagnostic, not language quality'),
            recipes={str(f.relative_to(P)): sha(f) for f in [D / 'engine_wrapper_v5.py',
                D / 'owned_thread_sampler.py', D / 'bounded_stderr_relay.py']},
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            limits=dict(wall_per_arm_seconds=900, raw_text_bytes_per_arm=64 << 20,
                RSS_bytes=88 << 30, job_bytes=1 << 30), arms=[], performance_qualified=False,
            quality_qualified=False, physical_262144_qualified=False, GPU_reset_or_service_change=False,
            scope='Host-only observation OFF/ON, compactFP16=1 in both, exact129 batched synthetic inputs. Same new binary; compare previous compact1 fixture too.')
        def persist():
            (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(O / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        def utility(label, argv):
            entry, out, err = owner.run(label, argv, build['environment'], P,
                wall=30, text_cap=64 << 20, total_cap=1 << 30)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'))
            return out.read_text()
        before = None
        try:
            before = re.search(r'-- cursor: (\S+)', utility('journal-before',
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])).group(1)
            record['journal_before_cursor'] = before
            for value in [0, 1]:
                label = 'observe' + str(value)
                arm = O / label
                arm.mkdir()
                (arm / 'owned').mkdir()
                env = dict(base_env)
                env.update(XDG_CACHE_HOME=str(arm / 'cache'),
                    STRATA_MEASUREMENT_ENGINE_START=str(arm / 'engine-start.json'),
                    STRATA_MEASUREMENT_ENGINE_EXIT=str(arm / 'engine-exit.json'),
                    STRATA_PREFILL_DUMP_R=str(arm / 'prefill-R.bin'),
                    STRATA_DUMP_FIRST_LOGITS=str(arm / 'first-logits.bin'))
                env['STRATA_PREFILL_MOE_COMPACT_FP16'] = '1'
                env['STRATA_PREFILL_ROUTED_FP16_OBSERVE'] = str(value)
                argv = ['/home/yayoi/.local/share/strata-sycl/venv/bin/python', str(D / 'engine_wrapper_v5.py'),
                    '--pack', str(PACK), '--native', str(NATIVE), '--tokens-file', str(token_path),
                    '--max-new', '1', '--max-context', '262144', '--kv', 'int8', '--kv-resident', '32768',
                    '--expert-cache', '128', '--expert-profile', str(P / 'data/expert-profile.bin'),
                    '--expert-cache-per-layer', '--pool-workers', '5', '--pcie-frac', '0',
                    '--adapt-swaps', '0', '--greedy', '--no-spec-split', '--suffix-draft', '0',
                    '--spec', '4', '--mtp', '/home/yayoi/.local/share/strata-sycl/mtp/rt',
                    '--prefill', str(chunk), '--no-prefill-borrow', '--vram-reserve-mib', '700']
                record['arms'].append(dict(label=label, argv=argv, environment=env, passed=False))
                persist()
                entry, out, err = owner.run('model-' + label, argv, env, P, wall=900, cpu=1200,
                    text_cap=64 << 20, rss_cap=88 << 30, total_cap=1 << 30, file_cap=64 << 20)
                assert module.closed(entry) and entry['exit_code'] == 0, (label, entry.get('error'), err.read_text()[-2000:])
                native_exit = json.loads((arm / 'engine-exit.json').read_text())
                relay = json.loads((arm / 'stderr-relay-summary.json').read_text())
                sampler = json.loads((arm / 'sampler-summary.json').read_text())
                assert native_exit['exit_code'] == 0 and native_exit['binary_sha256'] == record['binary_sha256']
                assert relay['passed'] and not relay['error_counts'] and not relay['oversized_lines'] and not relay.get('selected_cap_reached')
                assert sampler['passed'] and sampler['samples'] > 0
                text = err.read_text()
                # Standalone CLI does not emit the old server shutdown marker.
                # Its compat frees wait_all(), and Allocation checked_cleanup fail-stops
                # if completion fails; native normal0 and the complete relay remain gates.
                assert 'Strata SYCL cleanup:' not in text, 'checked allocation completion failed'
                residual(arm / 'prefill-R.bin')
                floats(arm / 'first-logits.bin')
                effective = re.findall(r'FP16 MoE compact requested=(\d+) effective=(\d+), chunk=(\d+), scratch rows=(\d+), accounted=(\d+), shared region=(\d+) bytes', text)
                if not threaded:
                    assert len(effective) == 1 and int(effective[0][1]) == 1 and int(effective[0][2]) == chunk, effective
                branches = re.findall(r'prefill issuer threaded=(\d+) stream_all=(\d+) ring=(\d+) chunk=(\d+) entries=(\d+)', text)
                skips = re.findall(r'prefill skip releases=(\d+) after-open-group=(\d+)', text)
                assert branches and skips
                if threaded:
                    assert any(int(x[0]) == value and int(x[1]) == 1 and int(x[2]) == 16 for x in branches)
                    assert any(int(x[0]) > 0 and int(x[1]) > 0 for x in skips), skips
                else:
                    assert all(int(x[0]) == 0 and int(x[1]) == 0 for x in branches)
                lines = [x.split("strata routed fp16 observation: ", 1)[1] for x in text.splitlines()
                         if x.startswith("strata routed fp16 observation: ")]
                if value == 0:
                    assert not lines, "observation disabled but emitted output"
                    observation = None
                else:
                    assert len(lines) == 1, len(lines)
                    observation = json.loads(lines[0])
                    assert observation['effective'] and observation['run_return'] == 'success'
                    assert observation['input_tokens'] == 129 and observation['chunks'] == 1
                    assert observation['actual_chunk_last'] == 129 and observation['compact_fp16'] == 1
                    hist = observation['hist_ne_count']
                    assert sum(c for ne,c in hist) == 512*observation['layer_chunks']
                    weighted = sum(ne*c for ne,c in hist)
                    assert weighted == observation['gu_rows'] == observation['down_rows']
                    assert weighted == 129*10*observation['layer_chunks']
                    assert observation['gu_calls'] == observation['down_calls'] == observation['active_experts']
                    assert observation['blob_h2d_attempted_host_copy_calls'] == observation['blob_h2d_attempted_direct_pinned_copy_calls']+observation['blob_h2d_attempted_stager_copy_calls']
                    assert observation['pf_timing_enabled'] and len(observation['pf_timeline_ms_by_phase']) == 20
                    assert all(math.isfinite(x) and x >= 0 for x in observation['pf_timeline_ms_by_phase'])
                reference = B/'main-primary-compact4096-pair-root-v2/compact1'
                assert sha(arm/'prefill-R.bin') == sha(reference/'prefill-R.bin'), 'pre-census residual differs'
                assert sha(arm/'first-logits.bin') == sha(reference/'first-logits.bin'), 'pre-census logits differ'
                arm_record = record['arms'][-1]
                arm_record.update(passed=True, native_exit=native_exit, relay=relay, sampler=sampler,
                    routed_observation=observation, prior_native_fixture_bitwise_equal=True,
                    effective_allocation=effective, issuer_branches=branches, skip_counts=skips,
                    residual_sha256=sha(arm / 'prefill-R.bin'), logits_sha256=sha(arm / 'first-logits.bin'),
                    stdout_sha256=sha(out), stderr_sha256=sha(err))
                persist()
                print(json.dumps(dict(completed=label, allocation=effective, issuer=branches, skips=skips)), flush=True)
            left, right = [O / x['label'] for x in record['arms']]
            lr, positions = residual(left / 'prefill-R.bin')
            rr, _ = residual(right / 'prefill-R.bin')
            record['numerical_comparison'] = dict(residual=compare(lr, rr),
                residual_positions=positions, logits=compare(floats(left / 'first-logits.bin'), floats(right / 'first-logits.bin')),
                logits_bitwise_equal=sha(left / 'first-logits.bin') == sha(right / 'first-logits.bin'),
                residual_bitwise_equal=sha(left / 'prefill-R.bin') == sha(right / 'prefill-R.bin'))
            persist()
            assert record['numerical_comparison']['logits_bitwise_equal'] and record['numerical_comparison']['residual_bitwise_equal'], record['numerical_comparison']
            for name, pin in build['source_pins'].items():
                assert sha(P / name) == pin
            assert sha(engine) == record['binary_sha256']
            record.update(passed=True, complete=True, numerical_scope_qualified=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if before:
                    text = utility('journal-after', ['journalctl', '-k', '-b', '--after-cursor', before, '--no-pager', '-o', 'json'])
                    entries = [json.loads(x) for x in text.splitlines() if x.strip()]
                    record['fresh_faults'] = [x for x in entries if FAULT.search(str(x.get('MESSAGE', '')))]
                    record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                    record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False, complete=False)
            finally:
                record['active'] = False
                persist()
        assert record['passed'] and record['complete']
        print(json.dumps(dict(passed=True, mode=mode, comparison=record['numerical_comparison'], fresh_faults=0)), flush=True)


if __name__ == '__main__':
    main()
