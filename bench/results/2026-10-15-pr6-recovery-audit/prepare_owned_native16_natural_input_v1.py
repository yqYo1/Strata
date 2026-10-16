#!/usr/bin/env python3
"""Freeze a project-document, thinking-on long input for a precision screen."""
import fcntl
import hashlib
import json
from pathlib import Path
import struct
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path[:0] = [str(P), str(P / 'tools')]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import strata_tokenizer as ST
        from serve.frontend import ChatTemplate
        from serve.server import Service
        tpath = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/tokenizer')
        root = B / 'main-own-native16-natural-input-root-v1'
        target = A / 'own-native16-natural-input-v1.json'
        assert not root.exists() and not target.exists()
        corpus = [P / n for n in ('README.md', 'docs/DETAILS.md', 'docs/AI_SETUP.md', 'docs/SYCL_IMPLEMENTATION.md')]
        deps = [P / n for n in ('tools/strata_tokenizer.py', 'serve/server.py', 'serve/frontend.py')]
        deps += [tpath / n for n in ('vocab.json', 'merges.txt', 'token_type.json', 'chat_template.jinja')]
        pins = {str(p): sha(p) for p in [*corpus, *deps]}
        vocab = json.loads((tpath / 'vocab.json').read_text())
        words = [None] * len(vocab)
        for word, i in vocab.items():
            words[i] = word
        tok = ST.Tokenizer(words, (tpath / 'merges.txt').read_text().split('\n'),
                           json.loads((tpath / 'token_type.json').read_text()))
        text = '\n\n'.join(p.read_text() for p in corpus)
        body = tok.encode(text, parse_special=False)
        assert len(body) > 32800, 'substantial natural source, no repeated padding'
        prefix = body[:32800]
        # Avoid truncating inside a UTF-8 sequence or otherwise changing the
        # tokenized corpus through a decode/re-encode round trip.
        while prefix and tok.encode(tok.decode(prefix), parse_special=False) != prefix:
            prefix.pop()
        assert len(prefix) >= 32780
        messages = [dict(role='user', content='Read this technical reference and explain its principal design decisions.\n\n' + tok.decode(prefix))]
        service = Service.__new__(Service)
        service.tok, service.template, service.effort_end = tok, ChatTemplate(tpath / 'chat_template.jinja'), False
        options = {'enable_thinking': True}
        rendered = service.render_prompt(messages, None, options)
        ids = service.encode_prompt(messages, None, options)
        assert tok.decode(ids) == rendered
        assert 32768 <= len(ids) < 35000
        positions = list(range(255, 32700, 512))
        assert len(positions) == 64 and positions[-1] + 1 < len(ids) - 1
        assert all(0 <= i < 248320 for i in ids)
        assert all(sha(Path(p)) == h for p, h in pins.items())
        root.mkdir(mode=0o700)
        inp = root / 'tokens.txt'; inp.write_text(','.join(map(str, ids)) + '\n')
        v = dict(complete=True, GPU_executed=False, source_pins=pins, recipe_sha256=sha(Path(__file__)),
            python=sys.executable, thinking_requested=True, token_file=str(inp), file_sha256=sha(inp),
            count=len(ids), batched_rows=len(ids) - 1, sampled_positions=positions,
            ids_le_u32_sha256=hashlib.sha256(struct.pack('<' + str(len(ids)) + 'I', *ids)).hexdigest(),
            rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(), roundtrip_exact=True,
            raw_corpus_tokens=len(body), selected_corpus_tokens=len(prefix), repeated_padding=False,
            scope='One ordered English/code/document corpus, 64 stratified teacher-forced full-vocabulary positions. Exploratory precision screen, not representative task quality, a statistical acceptance threshold, or a speed benchmark.',
            retention=dict(owner='Main', consumer='Same-revision two-FP32-controls/BF16 sparse long precision screen',
                byte_budget=1 << 20, next_review='At screen closure or supersession; keep reconstruction recipe and source hashes in Git.'))
        with target.open('x') as f:
            json.dump(v, f, indent=2); f.write('\n')
        print(json.dumps({k: v[k] for k in ('complete', 'GPU_executed', 'count', 'batched_rows', 'selected_corpus_tokens')}))


if __name__ == '__main__':
    main()
