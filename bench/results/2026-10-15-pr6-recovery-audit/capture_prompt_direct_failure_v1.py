#!/usr/bin/env python3
"""Preserve closed failure evidence and reconstruct public benchmark inputs offline."""
from pathlib import Path
import fcntl
import hashlib
import json
import subprocess
import sys

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
A = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(P / 'tools'))
    sys.path.insert(0, str(P))
    import strata_bench as bench
    import strata_tokenizer as tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages
    from serve.server import Service
    root = B / 'main-primary-prompt-direct-bench-full-0-root-v1'
    receipt = A / 'prompt-direct-bench-full-0-v1.json'
    original = json.loads(receipt.read_text())
    assert not original['active'] and not original['passed'] and not original['complete']
    assert not original['fresh_faults'] and not original['driver_dump_exists'] and original['boot_unchanged']
    assert all(c['direct_child_reaped'] and c['session_empty'] for c in original['commands'])
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert all(sha(P / n) == h for n, h in original['source_pins'].items())
        assert all(sha(P / n) == h for n, h in original['benchmark_source'].items())
        journals = list((root / 'stock-results').glob('*/requests.jsonl'))
        assert len(journals) == 1
        journal = journals[0]
        verified = bench.verify_request_journal(journal.parent)
        assert verified['valid'] and not verified['complete'] and len(verified['pending']) == 1
        rows = [json.loads(s) for s in journal.read_text().splitlines()]
        results = [r for r in rows if r['kind'] == 'result' and r['phase'] == 'prompt/32K']
        assert len(results) == 3 and [r['sample']['ok'] for r in results] == [True, True, False]
        assert 'exit code 134' in results[-1]['sample']['err']
        agents = [r['sample'] for r in rows if r['kind'] == 'result' and r['phase'] == 'agent']
        assert len(agents) == 18 and all(bench.agent_complete(agents[i:i + 6]) for i in (0, 6, 12))
        # The original journal is already compact (hashes/timings, no prompt/output text).
        for source, name in [(journal, 'prompt-direct-bench-full-0-requests-v1.jsonl'),
                             (journal.parent / 'journal-state.json', 'prompt-direct-bench-full-0-journal-state-v1.json')]:
            target = A / name
            assert not target.exists()
            target.write_bytes(source.read_bytes())
            assert sha(target) == sha(source)
        diagnostic = (root / 'owned/engine.stderr').read_text()
        start = diagnostic.index('strata serve: no progress for 480 s')
        stop_marker = 'strata: no active verify spin waits to release; prefill/device drain not assessed'
        end = diagnostic.index(stop_marker, start) + len(stop_marker)
        excerpt = diagnostic[start:end] + '\n'
        assert len(excerpt.encode()) < 256 << 10
        (A / 'prompt-direct-bench-full-0-stall-v1.txt').write_text(excerpt)

        cal = [r['sample']['prompt_tokens'] for r in rows
               if r['kind'] == 'result' and r['phase'] == 'calibration']
        assert cal == [260, 906]
        suite = bench.Suite('', None, 262144, 3, True, log=lambda *args: None)
        suite.tokens_per_word = (cal[1] - cal[0]) / 600
        suite.overhead = max(0, cal[0] - 200 * suite.tokens_per_word)
        birth = int(subprocess.check_output(['stat', '-c', '%W', str(journal)], text=True))
        expected = results[0]['sample']['request_sha256']
        matches = []
        for nonce in range(birth - 3, birth + 4):
            suite.nonce = str(nonce)
            payload = json.dumps(bench.chat_body(suite.prompt_for(32768, 0), 24, True)).encode()
            if hashlib.sha256(payload).hexdigest() == expected:
                matches.append(nonce)
        assert len(matches) == 1
        suite.nonce = str(matches[0])
        tpath = Path(original['config']['tokenizer'])
        vocab = json.loads((tpath / 'vocab.json').read_text())
        strings = [None] * len(vocab)
        for text, index in vocab.items():
            strings[index] = text
        tok = tokenizer.Tokenizer(strings, (tpath / 'merges.txt').read_text().split('\n'),
                                  json.loads((tpath / 'token_type.json').read_text()))
        template = tpath / 'chat_template.jinja'
        assert template.is_file()
        service = Service(None, tok, ChatTemplate(template))  # no engine, server or GPU initialization
        output = B / 'main-primary-prompt-direct-stall-inputs-root-v1'
        assert not output.exists()
        output.mkdir(mode=0o700)
        reconstructed = []
        for run in range(3):
            body = bench.chat_body(suite.prompt_for(32768, run), 24, True)
            wire = json.dumps(body).encode()
            fingerprint = hashlib.sha256(wire).hexdigest()
            assert fingerprint == results[run]['sample']['request_sha256']
            messages, tools, kwargs = openai_to_messages(body)
            ids = service.encode_prompt(messages, tools, kwargs)
            assert len(ids) == 32089 and all(0 <= value < len(vocab) for value in ids)
            path = output / ('run' + str(run) + '.tokens')
            path.write_text(','.join(map(str, ids)) + '\n')
            reconstructed.append(dict(run=run, request_sha256=fingerprint,
                actual_prompt_tokens=len(ids), tokens_path=str(path), tokens_sha256=sha(path),
                messages_text_sha256=hashlib.sha256(body['messages'][0]['content'].encode()).hexdigest()))
        result = dict(schema='prompt-direct-bench-failure-evidence-v1',
            status='FAILED_BASELINE_PAIR_INVALID_DIRECT1_NOT_RUN', performance_qualified=False,
            original_receipt_sha256=sha(receipt), original_status_preserved=True,
            stop_receipt_sha256=sha(A / 'prompt-direct-bench-stall-stop-v1.json'),
            request_journal_sha256=sha(journal), journal_verification=verified,
            completed_agent_sessions=3, nominal_32k_samples=results,
            stall_excerpt_sha256=sha(A / 'prompt-direct-bench-full-0-stall-v1.txt'),
            native_original_exit_code=-6, wrapper_original_exit_code=134,
            fresh_faults=original['fresh_faults'], driver_dump_exists=False, boot_unchanged=True,
            all_owned_sessions_closed=True, GPU_hang_established=False,
            restarted_native_executed_before_main_stop=True,
            diagnosis='watchdog no heartbeat at layer27/chunk0; actual blocked call/thread stack unavailable',
            benchmark_nonce=matches[0], calibration_prompt_counts=cal,
            tokens_per_word=suite.tokens_per_word, overhead=suite.overhead,
            reconstructed_public_requests=reconstructed,
            tokenizer_template_pins={str(f): sha(f) for f in
                [tpath / 'vocab.json', tpath / 'merges.txt', tpath / 'token_type.json', template]},
            reconstruction_source_pins={name: sha(P / name) for name in
                ['tools/strata_bench.py', 'tools/ab_engine.py', 'tools/strata_tokenizer.py',
                 'serve/server.py', 'serve/frontend.py']}, recipe_sha256=sha(Path(__file__)),
            next='No Direct1 retry or performance adoption. Preserve original binary; host-qualify owned stack capture before diagnostic reproduction.',
            retention_consumer='Three exact token files for current repeated-prefill stall diagnosis; retire after diagnosis or verified replacement.')
        (A / 'prompt-direct-bench-failure-evidence-v1.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(status=result['status'], reconstructed_tokens=[x['actual_prompt_tokens'] for x in reconstructed],
                              nonce=matches[0], fresh_faults=0, owned_sessions_closed=True)))


if __name__ == '__main__':
    main()
