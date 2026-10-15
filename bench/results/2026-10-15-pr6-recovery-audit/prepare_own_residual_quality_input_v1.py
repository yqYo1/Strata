#!/usr/bin/env python3
"""Freeze the official long-prefill workload for a future paired loss comparison."""
import fcntl
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path[:0] = [str(P), str(P / 'tools')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from strata_bench import Suite
        import strata_tokenizer as ST
        from serve.frontend import ChatTemplate
        from serve.server import Service
        tpath = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/tokenizer')
        root = B / 'main-own-residual-quality-input-root-v1'
        target = A / 'own-residual-quality-input-v1.json'
        assert not root.exists() and not target.exists()
        sources = [P / n for n in ('bench.sh', 'tools/strata_bench.py', 'tools/ab_engine.py',
                                    'tools/strata_tokenizer.py', 'serve/server.py', 'serve/frontend.py')]
        sources += [tpath / n for n in ('vocab.json', 'merges.txt', 'token_type.json', 'chat_template.jinja')]
        pins = {str(n): sha(n) for n in sources}
        vocabulary = json.loads((tpath / 'vocab.json').read_text())
        tokens = [None] * len(vocabulary)
        for word, i in vocabulary.items():
            tokens[i] = word
        tokenizer = ST.Tokenizer(tokens, (tpath / 'merges.txt').read_text().split('\n'),
                                 json.loads((tpath / 'token_type.json').read_text()))
        template = ChatTemplate(tpath / 'chat_template.jinja')
        # No Engine, Service constructor, server or GPU is started.
        service = Service.__new__(Service)
        service.tok, service.template, service.effort_end = tokenizer, template, False
        suite = Suite('http://127.0.0.1:1', '', 98304, 1, True,
                      workload_id='residual-quality-v1', prefill_only=True, prefill_tokens=65536)
        messages = suite.prefill_prompt(0)
        rendered = service.render_prompt(messages, None, {'enable_thinking': True})
        ids = service.encode_prompt(messages, None, {'enable_thinking': True})
        assert tokenizer.decode(ids) == rendered, 'lossless tokenizer/template roundtrip'
        assert 65536 <= len(ids) < 98304 and all(0 <= i < 248320 for i in ids)
        assert all(sha(Path(n)) == h for n, h in pins.items())
        root.mkdir(mode=0o700)
        inp = root / 'tokens.txt'
        inp.write_text(','.join(map(str, ids)) + '\n')
        assert inp.stat().st_size < 1 << 20
        record = dict(complete=True, GPU_executed=False, recipe_sha256=sha(Path(__file__)),
                      python_executable=sys.executable, python_version=sys.version,
                      preparation_note='Initial system-python invocation stopped on missing regex before creating artifacts; use the existing Strata venv, with no dependency installation.',
                      source_pins=pins, official_workload_plan=suite.prefill_plan(),
                      messages_sha256=hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                      rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                      template_sha256=sha(tpath / 'chat_template.jinja'), thinking_requested=True,
                      token_file=str(inp), file_sha256=sha(inp), count=len(ids), batched_rows=len(ids)-1,
                      ids_le_u32_sha256=hashlib.sha256(b''.join(struct.pack('<I', i) for i in ids)).hexdigest(),
                      input_bytes=inp.stat().st_size, roundtrip_exact=True,
                      scope='Official reproducible word-list workload for storage-only paired loss comparison. Not a general natural-language/task quality qualification or speed measurement.',
                      retention=dict(owner='Main', consumer='Next matched two-FP32 versus FP16 residual NLL comparison, then placement-only comparison',
                                     byte_budget=1 << 20, next_review='Close or supersede this comparison; retire explicit token file once its last comparator no longer requires it.'),
                      finished_unix=time.time())
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: record[k] for k in ('complete','GPU_executed','count','batched_rows','input_bytes','roundtrip_exact','scope')}))


if __name__ == '__main__':
    main()
