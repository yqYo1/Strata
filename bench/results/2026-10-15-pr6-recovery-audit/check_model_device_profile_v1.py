#!/usr/bin/env python3
"""Main-owned official device timing on one real 34K/128-token MTP4 request."""
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, DEVICE, FAULT, sha
from check_decode_format_service_v1 import ledger, format_service


def main():
    build_path = D / 'kv-copy-groups-build-v2.json'
    profiler_path = B / 'main-primary-unitrace-build-root-v1/record.json'
    reference_path = D / 'decode-format-service-normal-v1.json'
    build, tool, reference = [json.loads(p.read_text()) for p in [build_path, profiler_path, reference_path]]
    gate_path = D / 'kv-resolver-profile-v1.json'
    gate = json.loads(gate_path.read_text())
    assert all(d['passed'] and d['complete'] and not d['active'] for d in [build, tool, reference, gate])
    assert gate['collected'] and not gate['fresh_faults'] and not gate['driver_dump_exists'] and gate['boot_unchanged']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    provider = Path(tool['build_path'])
    exe = Path(build['build_path']) / 'strata'
    assert sha(exe) == build['binaries'][exe.name]
    assert tool['source_revision'] == '6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033'
    assert all(sha(provider / n) == h for n, h in tool['binaries'].items())
    assert all(sha(Path(n)) == h for n, h in tool['XPTI_provider_pins'].items())
    natural = json.loads((D / 'natural-layout-long-pair-v2.json').read_text())
    tokens = Path(natural['engine_args'][natural['engine_args'].index('--tokens-file') + 1])
    assert sha(tokens) == natural['input']['tokens_sha256'] == reference['input_sha256']
    ids = [int(x) for x in tokens.read_text().strip().split(',')]
    assert len(ids) == 34603
    args = reference['argv'][1:]
    assert args[args.index('--max-context') + 1] == '262144'
    assert args[args.index('--prefill') + 1] == '4096'
    assert args[args.index('--expert-cache') + 1] == '128'
    assert args[args.index('--spec') + 1] == '4' and '--no-coupled-draft' in args
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert all(sha(P / n) == h for n, h in build['source_pins'].items())
        assert (DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists()
        output = B / 'main-primary-model-device-profile-root-v1'
        assert not output.exists(); output.mkdir(mode=0o700)
        reports = output / 'provider'; reports.mkdir(mode=0o700)
        env = dict(reference['environment'], XDG_CACHE_HOME=str(output / 'cache'),
            STRATA_PREFILL_PHASE_AUDIT='0', STRATA_PREFILL_BOUNDARY_AUDIT='0', STRATA_SYCL_KV_COPY_GROUPS='0')
        for key in ['STRATA_PREFILL_LOGITS', 'STRATA_PREFILL_LOGITS_POSITIONS', 'STRATA_DUMP_FIRST_LOGITS',
                    'STRATA_PREFILL_BOUNDARY_PATH', 'STRATA_PREFILL_BOUNDARY_ROWS', 'STRATA_PREFILL_PHASE_PATH', 'STRATA_PREFILL_PHASE_ROWS', 'UR_LOG_TRACING']:
            env.pop(key, None)
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        assert not any(k.startswith(('UNITRACE_', 'PTI_')) or k in ['LD_PRELOAD', 'MKL_VERBOSE'] for k in env)
        env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
            UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        for key, value in {'ONEAPI_DEVICE_SELECTOR':'level_zero:gpu', 'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD':'1',
                'NEOReadDebugKeys':'1', 'EnableDirectSubmission':'0', 'STRATA_SYCL_PREFILL_SAME_QUEUE':'1',
                'SYCL_CACHE_PERSISTENT':'0', 'STRATA_SYCL_VERIFY_NATIVE_CAPTURE':'0',
                'STRATA_PREFILL_LAYER_MAJOR_R_GPU':'4096', 'STRATA_PREFILL_LAYER_MAJOR_R_DTYPE':'fp32'}.items():
            assert env[key] == value
        argv = [str(provider / 'unitrace'), '--device-timing', '--result-dir', str(reports), str(exe), *args]
        record = dict(active=True, passed=False, complete=False, collected=False, GPU_executed=True,
            model_profile_executed=True, commands=[], environment=env, argv=argv, source_pins=build['source_pins'],
            source_revision='a4d27df2', build_receipt_sha256=sha(build_path), binary_sha256=sha(exe),
            profiler_build_receipt_sha256=sha(profiler_path), profiler_binaries=tool['binaries'],
            profiler_source_revision=tool['source_revision'], component_gate_sha256=sha(gate_path),
            reference_sha256=sha(reference_path), recipe_sha256=sha(Path(__file__)), input_sha256=sha(tokens),
            actual_input_tokens=len(ids), requested_output_tokens=128, requests=[],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            limits=dict(wall_seconds=1800, request_seconds=1500, RSS_bytes=88 << 30, text_and_provider_bytes=64 << 20,
                job_bytes=256 << 20, file_bytes=64 << 20),
            execution_contract=gate['execution_contract'], adopted=False, performance_qualified=False,
            quality_qualified=False, physical_262144_qualified=False, restore_qualified=False,
            scope='Complete startup+prefill+decode device aggregate and actual decode CPU service. No phase-specific device timeline or busy-union claim. No clean-throughput claim; component symbols need source-based attribution.',
            retention_consumer='Extract and verify per-kernel model profile, protocol and joined CPU service, then retire successful raw logs/provider files.',
            review_point='After compact profile analysis is committed.')
        def persist(): (output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(output / 'owned', persist); record['commands'] = owner.commands; persist()
        def utility(label, argv):
            e, out, err = owner.run(label, argv, build['environment'], P, wall=30, total_cap=256 << 20)
            assert module.closed(e) and e['exit_code'] == 0, (label, e.get('error'), err.read_text()[-3000:])
            return out.read_text()
        cursor = None
        try:
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before', ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            record['journal_before_cursor'] = cursor
            def interaction(io, entry):
                original_poll = io.poll
                def bounded_poll():
                    original_poll()
                    files = [p for p in reports.rglob('*') if p.is_file()]
                    logs = list((output/'owned').glob('*.stdout')) + list((output/'owned').glob('*.stderr'))
                    assert sum(p.stat().st_size for p in files + logs) <= 64 << 20, 'aggregate raw text/provider budget'
                io.poll = bounded_poll
                while True:
                    line = io.line(1500)
                    if line.startswith('READY '):
                        assert line == 'READY 262144 stop'
                        print('Profile model READY; issuing 34603-token prompt / 128-token normal MTP4', flush=True)
                        break
                wire = ('GEN 128 temperature=0 seed=42 ' + ','.join(map(str, ids)) + '\n').encode()
                io.send(wire)
                generated, progress = [], []
                deadline = time.monotonic() + 1500
                while True:
                    assert time.monotonic() < deadline, 'request deadline'
                    line = io.line(deadline-time.monotonic())
                    if re.fullmatch(r'T \d+', line):
                        generated.append(int(line[2:])); assert len(generated) <= 128
                    elif line.startswith('DONE '):
                        f = line.split(); assert len(f) == 15
                        done = dict(generated=int(f[1]), prompt=int(f[2]), prompt_ms=float(f[3]), decode_ms=float(f[4]),
                            finish=f[5], accepted=int(f[6]), offered=int(f[7]), reused=int(f[8]), read=int(f[14]))
                        assert done['generated'] == len(generated) == 128 and done['finish'] == 'length'
                        assert done['prompt'] == done['read'] == len(ids) and done['reused'] == 0
                        assert all(math.isfinite(done[k]) and done[k] >= 0 for k in ['prompt_ms', 'decode_ms'])
                        record['requests'].append(dict(done=done, generated_ids=generated, progress=progress,
                            wire_sha256=hashlib.sha256(wire).hexdigest(), wire_bytes=len(wire),
                            reference_ids_equal=generated==reference['requests'][0]['generated_ids']))
                        persist(); print('Profile request DONE '+json.dumps(done), flush=True); break
                    elif re.fullmatch(r'(?:RESUME \d+|REUSED \d+|PP \d+ \d+ [\d.]+ [\d.]+)', line):
                        progress.append(line); assert len(progress) <= 1024
                    else: raise RuntimeError('unexpected protocol ' + line[:200])
                io.send(b'QUIT\n'); io.exit(60)
            print('Official unitrace real-model profile starting', flush=True)
            e, out, err = owner.run('model-profile', argv, env, P, interaction=interaction, wall=1800,
                cpu=1800*len(os.sched_getaffinity(0)), rss_cap=88 << 30, text_cap=64 << 20,
                total_cap=256 << 20, file_cap=64 << 20)
            assert module.closed(e) and e['exit_code'] == 0, (e.get('error'), err.read_text()[-5000:])
            text = err.read_text()
            assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed|stopping with owners live', text)
            blocks = re.findall(r'^strata decode joined begin .*?^strata decode joined close [^\n]*\n?', text, re.M|re.S)
            assert len(blocks) == 1 and len(record['requests']) == 1
            req = record['requests'][0]
            req['ledger'] = ledger(blocks[0], req['done'], 0)
            req['format_service'] = format_service(req['ledger'])
            record['stdout_sha256'], record['stderr_sha256'] = sha(out), sha(err)
            record['diagnostics'] = [s for s in text.splitlines() if s.startswith(('strata mtp:', 'strata prefill:', 'strata trace: layer-major')) or re.search(r'(?i)warning|error|FAIL', s)]
            record['provider_files'] = []
            for path in sorted(reports.rglob('*')):
                if not path.is_file(): continue
                item = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
                if path.suffix in ['.csv', '.txt', '.json'] and path.stat().st_size <= 4 << 20:
                    item['text'] = path.read_text()
                record['provider_files'].append(item)
            assert all(sha(P / n) == h for n, h in build['source_pins'].items())
            assert sha(exe) == build['binaries'][exe.name]
            record.update(passed=True, complete=True, collected=True)
        except BaseException as error:
            record['error'] = repr(error); raise
        finally:
            try:
                if cursor:
                    entries = [json.loads(s) for s in utility('journal-final', ['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record['fresh_faults'] = [s for s in entries if FAULT.search(str(s.get('MESSAGE','')))]
                    record['driver_dump_exists'] = (DEVICE/'devcoredump').exists()
                    record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False, complete=False)
            finally:
                record['active'] = False; persist()
                (D / 'model-device-profile-v1.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=record['passed'], collected=record['collected'],
            provider_files=len(record.get('provider_files', [])), fresh_faults=len(record['fresh_faults']))), flush=True)


if __name__ == '__main__':
    main()
