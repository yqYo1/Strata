#!/usr/bin/env python3
"""Main-owned default-off isolation and layer-wise exact boundary localization."""
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, DEVICE, FAULT, sha

BUILD = D / 'boundary-service-build-v1.json'
EXPERT_BYTES = 707788800


def output_ids(path):
    rows = re.findall(r'^output\s*:\s*([\d ]+)\s*$', path.read_text(), re.M)
    assert len(rows) == 1, str(path)
    return [int(v) for v in rows[0].split()]


def main():
    job = sys.argv[1]
    assert job in ['small', 'long-fixed', 'long-auto']
    mode = 'small' if job == 'small' else 'long'
    build = json.loads(BUILD.read_text())
    assert build['passed'] and build['complete'] and not build['active']
    assert build['host_contracts_passed']
    reference_path = D / ('selected-prefill-' + mode + '-v1.json')
    reference = json.loads(reference_path.read_text())
    storage = json.loads((D / 'layer-major-storage-small-v1.json').read_text())
    assert reference['passed'] and reference['complete'] and not reference['active']
    assert not reference['fresh_faults'] and reference['boot_unchanged'] and not reference['driver_dump_exists']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    O = B / ('main-primary-boundary-audit-' + job + '-root-v1')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if mode == 'long':
            gate = json.loads((D / 'boundary-audit-small-v1.json').read_text())
            assert gate['passed'] and gate['complete'] and not gate['active'] and not gate['fresh_faults']
            assert gate['binary_sha256'] == build['binaries']['strata']
        assert all(sha(P / n) == pin for n, pin in build['source_pins'].items())
        exe = Path(build['build_path']) / 'strata'
        assert sha(exe) == build['binaries']['strata']
        assert not O.exists() and not (DEVICE / 'devcoredump').exists()
        assert (DEVICE / 'driver').resolve().name == 'xe'
        args = list(reference['engine_args'])
        tokens = Path(args[args.index('--tokens-file') + 1])
        assert sha(tokens) == reference['tokens_sha256']
        ids = [int(v) for v in tokens.read_text().strip().split(',')]
        positions = reference['positions']
        assert len(ids) == reference['actual_input_tokens']
        original = Path(args[args.index('--mtp') + 1]) / 'experts.bin'
        assert original.stat().st_size == EXPERT_BYTES
        with original.open('rb') as stream:
            original_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        controls = {}
        for arm in reference['arms']:
            dtype = arm['dtype']
            old_stdout = B / ('main-primary-selected-prefill-' + mode + '-root-v1') / 'owned' / (dtype + '.stdout')
            assert sha(old_stdout) == arm['stdout_sha256']
            controls[dtype] = dict(arm, output_ids=output_ids(old_stdout))
        env = dict(storage['environment'])
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        for key, value in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', EnableDirectSubmission='0',
                               STRATA_SYCL_PREFILL_SAME_QUEUE='1', STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0').items():
            assert env[key] == value
        O.mkdir(mode=0o700)
        r = dict(active=True, passed=False, complete=False, mode=mode, configuration=job, commands=[], arms=[],
                 environment=env, engine_args=args, build_receipt_sha256=sha(BUILD),
                 reference_receipt_sha256=sha(reference_path), binary_sha256=sha(exe),
                 source_pins=build['source_pins'], recipe_sha256=sha(Path(__file__)),
                 tokens_sha256=sha(tokens), actual_input_tokens=len(ids), batched_input_tokens=len(ids)-1,
                 positions=positions, original_expert_bytes=EXPERT_BYTES, original_expert_sha256=original_hash,
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 quality_qualified=False, performance_qualified=False, physical_262144_qualified=False,
                 repeated_serve_qualified=False, coupled_qualified=False, adopted=False,
                 limits=dict(wall_seconds_per_arm=1800, RSS_bytes=88 << 30,
                             text_bytes_per_process=64 << 20, job_bytes=512 << 20),
                 fixture_consumer='Main localization of unresolved residual-placement divergence, per-layer cross-run exact comparison. '
                                  'Review raw artifacts after gate and compact evidence are committed.',
                 fixture_byte_budget=256 << 20,
                 scope='Default-OFF isolation plus exact source/store/next-input checks and per-layer raw rows. Diagnostic synchronization can mask races. Same arithmetic FP32 and FP16 arms separately: first-logits, selected-head rows, output IDs, '
                       'immutable RAM source, actual memory boundaries and normal MTP graph recapture. '
                       'Small input is a lifecycle gate, never a speed comparison; long input is the exact 34603-token history.')
        def persist():
            (O / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
        owner = module.Owner(O / 'owned', persist)
        r['commands'] = owner.commands
        persist()
        def utility(label, argv):
            entry, out, err = owner.run(label, argv, build['environment'], P, wall=30,
                                        text_cap=64 << 20, total_cap=512 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0
            return out.read_text()
        def journals(label, cursor):
            return [json.loads(v) for v in utility(label, ['journalctl', '-k', '-b', '--after-cursor',
                    cursor, '--no-pager', '-o', 'json']).splitlines() if v.strip()]
        cursor = None
        try:
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before',
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])).group(1)
            r['journal_before_cursor'] = cursor
            for dtype, audit_on in ([('fp32', False), ('fp32', True), ('fp16', True)] if job == 'small' else [('fp32', True)]):
                label = dtype + ('-audit' if audit_on else '-off')
                folder = O / label
                folder.mkdir()
                selected = folder / 'selected-logits.bin'
                runenv = dict(env, STRATA_PREFILL_LAYER_MAJOR='1', STRATA_PREFILL_LAYER_MAJOR_R_DTYPE=dtype,
                    STRATA_PREFILL_LAYER_MAJOR_R_GPU='4096' if job == 'long-fixed' else 'auto',
                    STRATA_PREFILL_RELEASE_MTP_EXPERTS='1',
                    STRATA_PREFILL_BOUNDARY_AUDIT='1' if audit_on else '0',
                    STRATA_PREFILL_BOUNDARY_ROWS=','.join(map(str, positions)),
                    STRATA_PREFILL_BOUNDARY_PATH=str(folder / 'boundary.txt'),
                    STRATA_PREFILL_LOGITS=str(selected), STRATA_PREFILL_LOGITS_POSITIONS=','.join(map(str, positions)),
                    STRATA_DUMP_FIRST_LOGITS=str(folder / 'first-logits.bin'), XDG_CACHE_HOME=str(folder / 'cache'))
                print('MTP release ' + job + ': starting ' + dtype, flush=True)
                entry, out, err = owner.run(label, [str(exe), *args], runenv, P, wall=1800,
                    cpu=1800*len(os.sched_getaffinity(0)), text_cap=64 << 20, rss_cap=88 << 30,
                    total_cap=512 << 20, file_cap=64 << 20)
                assert module.closed(entry) and entry['exit_code'] == 0, (dtype, entry.get('error'), err.read_text()[-5000:])
                text = err.read_text()
                assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed|aborting with|stopping with owners live', text)
                control = controls[dtype]
                meta = Path(str(selected) + '.meta')
                partial = Path(str(meta) + '.partial')
                raw = selected.read_bytes()
                vocab, count = struct.unpack_from('ii', raw)
                assert (vocab, count) == (248320, len(positions)) and len(raw) == 8+vocab*count*4
                assert sha(meta) == sha(partial) == control['metadata_sha256']
                first = folder / 'first-logits.bin'
                generated = output_ids(out)
                result = dict(dtype=dtype, audit_on=audit_on, label=label, environment=runenv, generated_ids=generated,
                    first_logits_sha256=sha(first), selected_logits_sha256=sha(selected), metadata_sha256=sha(meta),
                    stdout_sha256=sha(out), stderr_sha256=sha(err),
                    diagnostic_excerpt=[s for s in text.splitlines() if s.startswith(('strata mtp:', 'strata prefill:',
                        'strata generate: prefill', 'strata trace: layer-major'))],
                    fixtures=[dict(path=str(f), bytes=f.stat().st_size, sha256=sha(f)) for f in [selected, meta, partial, first]])
                r['arms'].append(result)
                persist()
                result['same_dtype_first_logits_equal'] = result['first_logits_sha256'] == control['first_logits_sha256']
                result['same_dtype_selected_logits_equal'] = result['selected_logits_sha256'] == control['selected_logits_sha256']
                result['same_dtype_generated_ids_equal'] = generated == control['output_ids']
                result['numerical_passed'] = all(result[k] for k in ['same_dtype_first_logits_equal', 'same_dtype_selected_logits_equal', 'same_dtype_generated_ids_equal'])
                if job != 'long-auto':
                    assert result['numerical_passed']
                if audit_on:
                    boundary = folder / 'boundary.txt'
                    bt = boundary.read_text()
                    lines = [dict(item.split('=', 1) for item in line.split() if '=' in item) for line in bt.splitlines()]
                    output = [line for line in lines if line.get('phase') == 'output_before_store']
                    stores = [line for line in lines if line.get('phase') in ['stored', 'stored_RNE_packed_bits']]
                    inputs = [line for line in lines if line.get('phase') in ['next_input', 'next_input_decoded_expected']]
                    assert len(output) == len(positions)*48 and len(stores) == len(inputs) == len(positions)*47
                    assert bt.endswith('status=complete boundary_memcmp=equal\n')
                    assert all(x['memcmp'] == 'equal' and x['nonfinite'] == '0' for x in stores + inputs)
                    order = [(int(x['layer']), int(x['relative'])) for x in output]
                    assert order == [(layer, row) for layer in range(48) for row in sorted(positions)]
                    raw_rows = Path(str(boundary) + '.rows.f32')
                    assert raw_rows.stat().st_size == len(positions)*48*10240*4
                    result['boundary'] = dict(copy_equal=True, coverage_complete=True, outputs=len(output),
                        stores=len(stores), inputs=len(inputs), output_order=order, text_path=str(boundary),
                        text_bytes=boundary.stat().st_size, text_sha256=sha(boundary),
                        raw_path=str(raw_rows), raw_bytes=raw_rows.stat().st_size, raw_sha256=sha(raw_rows))
                else:
                    assert not (folder / 'boundary.txt').exists()
                source = re.findall(r'expert source original=([01]) bytes=(\d+) source=owned-RAM(?: sha256=([0-9a-f]{64}))?', text)
                assert source == [('1', str(EXPERT_BYTES), original_hash), ('0', str(EXPERT_BYTES), '')]
                marks = re.findall(r'prompt expert phase=([\w-]+) free_bytes=(\d+) total_bytes=(\d+)', text)
                assert [x[0] for x in marks] == ['before-detach', 'after-detach', 'after-prompt-before-restore', 'after-restore']
                assert len({x[2] for x in marks}) == 1
                result['memory_boundaries'] = [dict(phase=s, free_bytes=int(f), total_bytes=int(t)) for s, f, t in marks]
                result['observed_detach_free_delta_bytes'] = int(marks[1][1])-int(marks[0][1])
                result['observed_restore_free_delta_bytes'] = int(marks[2][1])-int(marks[3][1])
                assert result['observed_detach_free_delta_bytes'] >= EXPERT_BYTES-(8 << 20)
                assert abs(result['observed_restore_free_delta_bytes']-EXPERT_BYTES) <= 32 << 20
                restored = re.search(r'experts restored bytes=(\d+), cumulative detach=([\d.]+) ms reload=([\d.]+) ms', text)
                assert restored and int(restored[1]) == EXPERT_BYTES
                result['detach_ms'], result['restore_ms'] = map(float, restored.groups()[1:])
                captures = re.findall(r'expert decode recapture kind=(round|step) coupled=([01]) ok=([01]) ms=([\d.]+) cumulative_ms=([\d.]+)', text)
                assert all(c[1:3] == ('0', '1') and all(math.isfinite(float(v)) and float(v) >= 0 for v in c[3:]) for c in captures)
                if mode == 'small':
                    assert {c[0] for c in captures} == {'round', 'step'}
                result['decode_recaptures'] = [dict(kind=k, coupled=int(c), ok=int(ok), ms=float(ms), cumulative_ms=float(total))
                                               for k, c, ok, ms, total in captures]
                plan = re.search(r'layer-major candidate, tokens=(\d+) chunk=(\d+) residual_dtype=(fp32|fp16) residual_element_bytes=(\d+) residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) residual_host=(\d+) residual_staging=(\d+) layer_cache=(\d+)', text)
                assert plan
                result['residual_plan'] = dict(zip(['tokens', 'chunk', 'dtype', 'element_bytes', 'total_bytes', 'gpu_rows',
                    'gpu_bytes', 'host_bytes', 'staging_bytes', 'cache_bytes'], [v if i == 2 else int(v) for i, v in enumerate(plan.groups())]))
                result['all_residual_in_vram'] = result['residual_plan']['host_bytes'] == 0
                faults = [v for v in journals('journal-after-' + label, cursor) if FAULT.search(str(v.get('MESSAGE', '')))]
                assert not faults and not (DEVICE / 'devcoredump').exists()
                persist()
                print('MTP release ' + job + ': closed ' + dtype + ' restore_ms=' + str(result['restore_ms']) +
                      ' residual_host_bytes=' + str(result['residual_plan']['host_bytes']), flush=True)
            assert all(sha(P / n) == pin for n, pin in r['source_pins'].items())
            assert sha(exe) == r['binary_sha256']
            r.update(passed=all(a['numerical_passed'] for a in r['arms']), complete=True,
                     localization_complete=True, numerical_passed=all(a['numerical_passed'] for a in r['arms']))
        except BaseException as error:
            r['error'] = repr(error)
            raise
        finally:
            try:
                if cursor:
                    r['fresh_faults'] = [v for v in journals('journal-final', cursor) if FAULT.search(str(v.get('MESSAGE', '')))]
                    r['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                    r['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == r['boot_id']
                    if r['fresh_faults'] or r['driver_dump_exists'] or not r['boot_unchanged']:
                        r.update(passed=False, complete=False)
            finally:
                r['active'] = False
                persist()
                (D / ('boundary-audit-' + job + '-v1.json')).write_text(json.dumps(r, indent=2) + '\n')
        print(json.dumps(dict(passed=r['passed'], mode=mode, configuration=job, fresh_faults=len(r['fresh_faults']))), flush=True)


if __name__ == '__main__':
    main()
