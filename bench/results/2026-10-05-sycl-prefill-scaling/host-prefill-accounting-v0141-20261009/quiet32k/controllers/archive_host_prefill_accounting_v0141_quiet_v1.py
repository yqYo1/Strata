"""Archive the terminal quiet comparison and measured host durations."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

B = Path(__file__).parent
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-server-env-v0141-2026-10-09')
PARENT = W / 'bench/results/2026-10-05-sycl-prefill-scaling/host-prefill-accounting-v0141-20261009'
OUT = PARENT / 'quiet32k'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


summary_path = B / 'host-prefill-accounting-v0141-quiet32k-summary-v1.json'
assert digest(summary_path) == 'a038f17a6eb74b689c000fcadc21f9c4e00dbe76ccb84d5dfccb78071fe90f80'
summary = json.loads(summary_path.read_text())
assert summary['passed'] and not summary['active'] and len(summary['rows']) == 24
assert len(summary['host_reports']) == 8 and not summary['candidate_full_lifecycle_passed'] and not summary['candidate_adopted']
sequence_path = B / 'host-prefill-accounting-v0141-quiet32k-comparison-sequence-v1/record.json'
assert digest(sequence_path) == summary['sequence_sha256']
sequence = json.loads(sequence_path.read_text())
assert not sequence['active'] and sequence['passed']
assert not OUT.exists()
OUT.mkdir(parents=True)


def copy(source, relative):
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert digest(source) == digest(target)


copy(summary_path, 'summary.json')
copy(sequence_path, 'sequence/record.json')
for filename in ['run_owned_host_accounting_v0141_quiet32k_v1.py', 'run_host_prefill_accounting_v0141_quiet_sequence_v1.py',
                 'prepare_owned_host_accounting_v0141_quiet32k_v1.py', 'summarize_host_prefill_accounting_v0141_quiet_v1.py',
                 'archive_host_prefill_accounting_v0141_quiet_v1.py']:
    copy(B / filename, 'controllers/' + filename)
copy(B / 'host-prefill-accounting-v0141-quiet-controller-preparation-v1.json', 'preparation/record.json')
for step in sequence['steps']:
    receipt = Path(step['receipt'])
    assert digest(receipt) == step['receipt_sha256']
    data = json.loads(receipt.read_text())
    assert not data['active'] and data['healthy'] and data['completed'] and data['math_gate_passed']
    assert data['exit_code'] == 0 and not data['new_fault_messages'] and not any(data['cleanup'].values())
    for filename in ['record.json', 'events.jsonl', 'protocol.stdout.raw', 'project-messages.txt']:
        copy(receipt.parent / filename, 'runs/' + receipt.parent.name + '/' + filename)
with (OUT / 'summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary['rows'][0]))
    writer.writeheader()
    writer.writerows(summary['rows'])
g = summary['groups']
table = '\n'.join(f"| {mode} | {g[mode]['first']['pooled_prefill_tok_s']:.3f} | {g[mode]['later']['pooled_prefill_tok_s']:.3f} | {g[mode]['later']['pooled_decode_tok_s']:.3f} |" for mode in ['baseline','hostoff','hoston'])
text = f'''# Quiet32K host accounting comparison

On2026-10-09 JST, Arc B57010 GiB / Ryzen5 5600X /128 GiB RAM ran baseline/off/on/on/off/baseline, four fresh32768-token A/B/A/B inputs and64 visible outputs per process. Context262144, actual8192 chunks,128 cache slots/325 MiB, resident KV32768 and every requested argument/environment value match except the measured binary and explicit host-counter flag. No API logging, validation, profiler, head/state dump, native event query or extra phase wait was enabled.

| Mode | First prefill tokens/s (2 reads) | Later prefill tokens/s (6 reads) | Later decode tokens/s (6 reads) |
| --- | ---: | ---: | ---: |
{table}

The baseline is qualified binary86972697; hostoff and hoston are private instrumented binary494cf4be. Rates are pooled token/time ratios. First process reads are separate from later fresh reads. Hostoff later prefill differs by-0.03235% and hoston by-0.03887% from baseline. These differences are below the per-read prefill spread. The sample does not establish a useful speed change or a small decode regression: baseline decode spans15.637–18.571 tokens/s, off14.991–18.396 and on14.747–18.651. Full medians/ranges and all24 per-read values are in the JSON/CSV.

All24 complete outputs, printed logprobs, MTP counts, finish reasons and repeats match the qualified baseline. Every read has RESUME0 and REUSED0. All six processes exit normally with no new GPU fault, forced cleanup or owned survivor. These quiet runs check visible numerical references; [the separate first diagnostic](../first-diagnostic/README.md) supplies four-fresh first-head and66-part live-state proof for this private binary.

Eight host reports repeat the exact diagnostic transfer counts: A92998 copies/190240998400 bytes and B92979/190199219200. Over the six later reads, host submission totals average1217.629 ms, GPU grouping waits29353.892 ms, CPU grouping47.744 ms, issuer-publication waits0 and RAM-worker copy sums10124.776 ms; prefill wall duration averages72853.172 ms. Submission and grouping CPU work are small compared with wall time. GPU grouping waits include preceding queue work and dependencies; they are not measured PCIe-only waiting. RAM-worker sums may overlap with GPU work and with one another, so these durations cannot be added into a wall-time breakdown.

Exact expert byte counts make transfer reduction and overlap concrete next experiments. The bounded original API log maps the captured expert ring to an engine supporting compute and copy; a separate native copy-only queue candidate is being prepared on another branch. It has no performance or GPU-safety result here.

The accounting binary remains private and unadopted, with its full262144 lifecycle pending. Only the baseline has full-context qualification. Nothing in this archive makes that qualification transferable to modified code. Small terminal receipts, source/runtime/controller hashes and project/protocol logs are preserved byte for byte; large API logs, binaries and state/session tensors remain private.
'''
(OUT / 'README.md').write_text(text)
(PARENT / 'README.md').write_text('''# Host prefill accounting on the updated0.1.41 engine

[The first logged diagnostic](first-diagnostic/README.md) passes four fresh32K full live-state/head/output checks and records about190 GB of expert H2D transfers per input. [The quiet baseline/off/on comparison](quiet32k/README.md) passes24 fresh32K reads and measures449.923 baseline versus449.748 counter-on later prefill tokens/s, a difference of-0.03887%. The host durations expose little CPU grouping/submission time; GPU grouping waits do not distinguish compute from transfer. The counter code and a separately prepared copy-engine experiment remain unadopted. The counter binary has not passed its full262144 lifecycle.
''')
manifest = {str(p.relative_to(OUT)): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT / 'manifest.json').write_text(json.dumps({'files': manifest}, indent=2) + '\n')
print(json.dumps({'archived': True, 'files': len(manifest), 'directory': str(OUT), 'manifest_sha256': digest(OUT/'manifest.json')}, indent=2))
