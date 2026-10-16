#!/usr/bin/env python3
"""Current natural attention vs qualified native: long numerical/route diagnostic."""
import array
import fcntl
import gzip
import importlib.util
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, DEVICE, FAULT, sha

O = B / 'main-primary-natural-layout-long-pair-root-v2'
PY = '/home/yayoi/.local/share/strata-sycl/venv/bin/python'


def prepare_tokens(root, pack):
    sys.path.insert(0, str(P))
    sys.path.insert(0, str(P / 'tools'))
    import strata_tokenizer
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    from ab_engine import long_prompt
    tpath = pack / 'tokenizer'
    vocab = json.loads((tpath / 'vocab.json').read_text())
    names = [None] * len(vocab)
    for token, index in vocab.items():
        names[index] = token
    tok = Tokenizer(names, (tpath / 'merges.txt').read_text().split('\n'),
                    json.loads((tpath / 'token_type.json').read_text()))
    template = ChatTemplate(tpath / 'chat_template.jinja')
    messages = [dict(role='user', content='[strata natural attention numerical comparison v1]\n' + long_prompt(32768))]
    rendered = template.render(messages, enable_thinking=True)
    ids = tok.encode(rendered, parse_special=True)
    assert 32769 <= len(ids) <= 40961 and all(0 <= i < len(vocab) for i in ids)
    path = root / 'tokens.txt'
    path.write_text(','.join(map(str, ids)) + '\n')
    return path, dict(actual_input_tokens=len(ids), batched_input_tokens=len(ids)-1,
        thinking=True, tokens_sha256=sha(path), canonical_messages_sha256=__import__('hashlib').sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        rendered_sha256=__import__('hashlib').sha256(rendered.encode()).hexdigest(),
        source_pins={str(p): sha(p) for p in [P/'tools/ab_engine.py', Path(strata_tokenizer.__file__),
            P/'serve/frontend.py', *[tpath/n for n in ['vocab.json', 'merges.txt', 'token_type.json', 'chat_template.jinja']]]})


def main():
    build_path = D / 'qsa-layout-build-v1.json'
    build = json.loads(build_path.read_text())
    latest_build_path = D / 'gemm-orientation-build-v2.json'
    latest_build = json.loads(latest_build_path.read_text())
    ref_path = P/'bench/results/2026-10-14-long-generation-benchmark/receipts/main-primary-long-generation-suite-root-v4-upload-fix-v1/record.json.gz'
    reference = json.loads(gzip.decompress(ref_path.read_bytes()))
    assert build['passed'] and build['complete'] and not build['active']
    assert latest_build['passed'] and latest_build['complete'] and not latest_build['active']
    assert reference['passed'] and reference['complete'] and not reference['active']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists() and not (DEVICE/'devcoredump').exists()
        assert (DEVICE/'driver').resolve().name == 'xe'
        # Later builds added only the excluded standalone GEMM CMake target.
        # Its complete source set and unchanged product binary are both pinned.
        assert all(sha(P/n) == pin for n, pin in latest_build['source_pins'].items())
        assert [n for n, pin in build['source_pins'].items() if sha(P/n) != pin] == ['cmake/sycl_backend.cmake']
        current = Path(build['build_path'])/'strata'
        control = Path(build['control_binary']['path'])
        assert sha(current) == build['binaries']['strata']
        assert sha(current) == latest_build['production_binary_sha256']
        assert sha(control) == reference['binary_sha256'] == build['control_binary']['sha256']
        for group in ['pack_metadata_sha256', 'MTP_metadata_sha256']:
            assert all(sha(Path(n)) == pin for n, pin in reference[group].items())
        for n, identity in reference['model_identity'].items():
            info = Path(n).stat()
            assert info.st_size == identity['bytes'] and info.st_mtime_ns == identity['mtime_ns']
        O.mkdir(mode=0o700)
        tokens, identity = prepare_tokens(O, Path(reference['engine_args'][1]))
        env = dict(reference['environment'])
        for key in ['STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT']:
            env.pop(key, None)
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        assert not any(k in env for k in ['LD_PRELOAD', 'STRATA_PREFILL_TIMING', 'MKL_VERBOSE', 'UR_ENABLE_LAYERS'])
        env.update(STRATA_TRACE='1', STRATA_PREFILL_ATTN_LAYOUT='0',
            UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
            UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
            ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
            ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOGGING_LEVEL='warning',
            ZEL_LOADER_LOG_CONSOLE='1', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0')
        args = reference['engine_args'] + ['--tokens-file', str(tokens), '--max-new', '1']
        r = dict(active=True, passed=False, complete=False, commands=[], arms=[], input=identity,
            environment=env, engine_args=args, source_pins=latest_build['source_pins'],
            build_receipt_sha256=sha(build_path), control_suite_archive_sha256=sha(ref_path),
            latest_build_receipt_sha256=sha(latest_build_path),
            source_revision=latest_build['source_revision'],
            recipe_sha256=sha(Path(__file__)), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            performance_qualified=False, quality_qualified=False, physical_262144_qualified=False,
            restore_qualified=False, adopted=False,
            fixture_consumer='Main upcoming current-root layer-major numerical comparison: control/new first logits and every64th final residual; do not use as full-state/task-quality validation',
            fixture_byte_budget=128 << 20, fixture_review='After layer-major matched-input numerical screen closes or this version is superseded',
            limits=dict(wall_seconds_per_arm=1200, CPU_seconds_per_arm=1200*len(os.sched_getaffinity(0)),
                        RSS_bytes=88 << 30, raw_text_bytes=64 << 20, job_bytes=512 << 20),
            scope='Real >=32K thinking-template input, one output token, unchanged natural math expected bitwise parity. Diagnostic validation/dump overhead; no bench.sh speed or long-decode claim.')

        def persist():
            (O/'record.json').write_text(json.dumps(r, indent=2)+'\n')

        owner = module.Owner(O/'owned', persist)
        r['commands'] = owner.commands
        persist()

        def utility(label, argv):
            entry, out, err = owner.run(label, argv, reference['environment'], P, wall=30,
                                       text_cap=64 << 20, total_cap=512 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0
            return out.read_text()

        cursor = None
        try:
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before',
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])).group(1)
            r['journal_before_cursor'] = cursor
            for name, exe in [('control', control), ('natural0', current)]:
                folder = O/name
                folder.mkdir()
                runenv = dict(env, XDG_CACHE_HOME=str(folder/'cache'),
                    STRATA_DUMP_FIRST_LOGITS=str(folder/'first-logits.bin'),
                    STRATA_PREFILL_DUMP_R=str(folder/'prefill-R.bin'))
                print('natural-layout long pair: starting '+name, flush=True)
                entry, out, err = owner.run(name, [str(exe), *args], runenv, P, wall=1200,
                    cpu=r['limits']['CPU_seconds_per_arm'], text_cap=64 << 20, rss_cap=88 << 30,
                    total_cap=512 << 20, file_cap=64 << 20)
                assert module.closed(entry) and entry['exit_code'] == 0, (name, entry.get('error'), err.read_text()[-3000:])
                text = err.read_text()
                assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:', text)
                fs = array.array('f'); fs.frombytes((folder/'first-logits.bin').read_bytes())
                assert len(fs) == 248320 and all(math.isfinite(x) for x in fs)
                raw = (folder/'prefill-R.bin').read_bytes(); stride = 8 + 10240*4
                assert len(raw) % stride == 0
                positions = []
                for offset in range(0, len(raw), stride):
                    positions.append(struct.unpack_from('q', raw, offset)[0])
                    row = array.array('f'); row.frombytes(raw[offset+8:offset+stride])
                    assert all(math.isfinite(x) for x in row)
                assert positions == list(range(0, identity['batched_input_tokens'], 64))
                lines = [line for line in text.splitlines() if line.startswith((
                    'strata prefill:', 'strata generate:', 'strata mtp:', 'strata verify:', 'strata prefill attention route:'))]
                route = [line for line in text.splitlines() if 'matrix_chunks=' in line]
                if name == 'natural0':
                    assert len(route) == 1
                r['arms'].append(dict(name=name, exe=str(exe), binary_sha256=sha(exe),
                    environment=runenv, stdout_sha256=sha(out), stderr_sha256=sha(err),
                    diagnostic_excerpt=lines, route_lines=route,
                    logits_sha256=sha(folder/'first-logits.bin'), residual_sha256=sha(folder/'prefill-R.bin'),
                    residual_positions=positions, finite=True,
                    fixtures=[dict(path=str(f), bytes=f.stat().st_size, sha256=sha(f))
                              for f in [folder/'first-logits.bin', folder/'prefill-R.bin']]))
                faults = [json.loads(line) for line in utility('journal-after-'+name,
                    ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json']).splitlines() if line.strip()]
                assert not any(FAULT.search(str(v.get('MESSAGE', ''))) for v in faults)
                assert not (DEVICE/'devcoredump').exists()
                persist()
                print('natural-layout long pair: closed '+name, flush=True)
            left, right = r['arms']
            r['logits_bitwise_equal'] = left['logits_sha256'] == right['logits_sha256']
            r['sampled_residuals_bitwise_equal'] = left['residual_sha256'] == right['residual_sha256']
            assert r['logits_bitwise_equal'] and r['sampled_residuals_bitwise_equal']
            assert all(sha(P/n) == pin for n, pin in latest_build['source_pins'].items())
            assert sha(current) == build['binaries']['strata'] and sha(control) == reference['binary_sha256']
            r.update(passed=True, complete=True)
        except BaseException as error:
            r['error'] = repr(error)
            raise
        finally:
            try:
                if cursor:
                    entries = [json.loads(line) for line in utility('journal-final',
                        ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json']).splitlines() if line.strip()]
                    r['fresh_faults'] = [v for v in entries if FAULT.search(str(v.get('MESSAGE', '')))]
                    r['driver_dump_exists'] = (DEVICE/'devcoredump').exists()
                    r['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == r['boot_id']
                    if r['fresh_faults'] or r['driver_dump_exists'] or not r['boot_unchanged']:
                        r.update(passed=False, complete=False)
            finally:
                r['active'] = False
                persist()
                (D/'natural-layout-long-pair-v2.json').write_text(json.dumps(r, indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'], logits_bitwise_equal=r['logits_bitwise_equal'],
            residuals_bitwise_equal=r['sampled_residuals_bitwise_equal'], route=r['arms'][1]['route_lines'])), flush=True)


if __name__ == '__main__':
    main()
