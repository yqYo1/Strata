from pathlib import Path
import json, hashlib, shutil, datetime, subprocess

b = Path(__file__).parent
r = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
a = r/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k'
sequence = json.loads((b/'v01402-code32k-sequence/record.json').read_text())
assert sequence['passed'] and not sequence['active']
s = json.loads((b/'v01402-code32k-summary.json').read_text())
assert len(s['rows']) == 6
assert all(x['healthy'] and x['prompt_tokens'] == 32768 and x['generated_tokens'] == 64 for x in s['rows'])
assert len({x['prompt_sha256'] for x in s['rows']}) == 1
assert all(x['repetitions'] == 2 for x in s['clean_means'].values())
a.mkdir()

def copy(src, dest):
    p = a/dest
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(b/src, p)

for p in sorted(b.glob('owned-v01402-code32k-*/record.json')):
    name = p.parent.name
    for file in ['record.json', 'protocol.stdout.raw', 'events.jsonl', 'project-messages.txt']:
        copy(name+'/'+file, 'runs/'+name+'/'+file)
    for q in (p.parent/'probes').glob('*'):
        if q.is_file():
            copy(str(q.relative_to(b)), 'runs/'+name+'/probes/'+q.name)
    for file in ['inferior-argv.json', 'inferior-environment.json']:
        q = p.parent/'debugger'/file
        if q.exists():
            copy(str(q.relative_to(b)), 'runs/'+name+'/debugger/'+file)
for file in ['coding-review-32k-fixture.json', 'coding-review-32k-tokens.txt', 'v01402-code32k-summary.json', 'v01402-code32k-sequence/record.json']:
    copy(file, file)
for file in ['run_owned_v01402_code32k.py', 'run_owned_v01402_code32k_v2.py', 'run_v01402_code32k_sequence.py', 'record_v01402_code32k_measurements.py', 'record_v01402_code32k_measurements_v2.py', 'record_v01402_code32k_measurements_v3.py', 'archive_v01402_code32k.py', 'archive_v01402_code32k_v2.py']:
    copy(file, 'controllers/'+file)
lines = ['case,phase,prompt_tokens,generated_tokens,prompt_ms,decode_ms,prefill_tok_s,decode_tok_s,actual_cache_slots,actual_cache_mib,startup_free_vram_mib']
for x in s['rows']:
    lines.append(','.join(str(x[k]) for k in ['case', 'phase', 'prompt_tokens', 'generated_tokens', 'prompt_ms', 'decode_ms', 'prefill_tok_s', 'decode_tok_s', 'expert_slots', 'expert_cache_mib', 'vram_free_mib']))
(a/'results.csv').write_text('\n'.join(lines)+'\n')
readme = '''# 32K comparison of upstream v0.1.40.2 and the SYCL integration

The unmodified upstream tag and integrated fork each process exactly 32,768
input tokens and generate 64 greedy tokens with normal MTP4. The input is an
expanded real-source code-review prefix, followed by a completed review question
and assistant prefix. `coding-review-32k-fixture.json` records the original fixture,
all suffix IDs, round-trip check and final shared SHA256. Prompt/conversation
caching is disabled; every process reads the entire input once.

Hardware and runtime are the same as the parent integration record: Arc B570 10 GiB,
Ryzen 5 5600X, 128 GiB RAM, installed NEO 26.31.39395.14, oneAPI 2026.1.1, Level Zero V2.
Both request context33024, int8 KV, 8192-token prompt chunks, cache128 per-layer,
five CPU workers and pcie0. Common FIRST0 and RING8 disable the short initial
chunk and set the streaming ring to eight blobs. No layer-major/compact/weight-release
tuning flags or extra phase waits are forced. Record actual allocation because
upstream packs 144 mixed-size cache slots into 281 MiB while the fork preserves
128 explicit-count slots into 325 MiB. These are comparisons of complete
configurations; different residency prevents attributing a difference solely
to the lifetime or launch changes.

The initial logged/validated control for each executable is functional evidence
only. Clean timings have no API tracing, validation, additional waits or logits
dump. Each is a cold first request in a fresh process; the engine's DONE time
includes request JIT/graph preparation and PLE, and excludes model startup.
An external owned GDB/PTY observer remains attached without interrupts in all
successful runs. Clean order is upstream/fork/fork/upstream (ABBA).

| Clean configuration | Repetitions | Prompt tok/s | Decode tok/s | Accepted tuning reference |
| --- | ---: | ---: | ---: | --- |
'''
for mode, label in [('pure', 'Unmodified upstream'), ('patched-default', 'Integrated fork')]:
    x = s['clean_means'][mode]
    readme += f"| {label} | {x['repetitions']} | {x['prefill_tok_s']:.2f} | {x['decode_tok_s']:.2f} | {'yes' if mode in s['accepted_clean_means'] else 'no, output mismatch'} |\n"
readme += '''
All six processes finish normally with 64 finite-logprob tokens, complete owned
cleanup and no new xe fault. Both independent upstream source-status checks
remain clean before and after execution. Per-run protocol, project messages,
health probes and immutable executed controllers are included here. Full diagnostic
API logs remain private; their sizes and SHA256 are in each record. Logged
control rates are excluded from the table and retained in the raw CSV.

The table retains observed elapsed times, including the rejected fork rows.
Only the upstream is currently accepted as a reproducible tuning reference:
its logged control and both clean repetitions match all64 token IDs and every
protocol logprob. The fork's logged control and second clean repetition agree,
but its first clean repetition has a different first logprob and diverges in
IDs at output index34. The first token's logprob is -1.023937 versus -1.051243.
This is a difference within the same binary, input, cache placement and requested
settings, so mixed cache sizing between configurations cannot explain it. It
shows unresolved output reproducibility without identifying the resource or
proving logging caused the difference. The fork times cannot support an accepted
speed comparison or tuning decision before this is resolved. Finite output,
normal exit and no xe fault do not establish model correctness.

These records establish completion of these32K jobs. They do not establish the performance
of larger contexts, repeated/restored conversations, cache release/recreation,
the unresolved earlier process waits, or the PP1000/TG70 target. Full262144-cell
normal-MTP gates remain open. Short37-token preflight timings remain excluded
from performance comparisons.
'''
(a/'README.md').write_text(readme)
meta = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'integration_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=r, text=True).strip(), 'files': {str(p.relative_to(a)): {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(a.rglob('*')) if p.is_file()}}
(a/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
print('archive', a, 'files', len(meta['files']))
