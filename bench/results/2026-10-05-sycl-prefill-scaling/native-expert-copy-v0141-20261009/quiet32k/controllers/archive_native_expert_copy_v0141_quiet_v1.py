"""Publish only normal terminal quiet evidence; adoption remains separate."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess

B = Path(__file__).parent
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-server-env-v0141-2026-10-09')
PARENT = W / 'bench/results/2026-10-05-sycl-prefill-scaling/native-expert-copy-v0141-20261009'
OUT = PARENT / 'quiet32k'
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

summary_path = B / 'native-expert-copy-v0141-quiet32k-summary-v1.json'
summary = json.loads(summary_path.read_text())
assert summary['passed'] and not summary['active'] and len(summary['rows']) == 24
assert not summary['candidate_full_lifecycle_passed'] and not summary['candidate_adopted']
sequence_path = B / 'native-expert-copy-v0141-quiet32k-comparison-sequence-v1/record.json'
assert digest(sequence_path) == summary['sequence_sha256']
sequence = json.loads(sequence_path.read_text())
assert not sequence['active'] and sequence['passed']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == '98e8f8a278ee4c4bccdd90683435a90f9df1673b'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True).strip()
assert not OUT.exists()
OUT.mkdir(parents=True)

def copy(source, relative):
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert digest(source) == digest(target)

copy(summary_path, 'summary.json')
copy(sequence_path, 'sequence/record.json')
for filename in ('run_owned_native_expert_copy_v0141_quiet32k_v1.py', 'run_native_expert_copy_v0141_quiet_sequence_v1.py',
                 'prepare_owned_native_expert_copy_v0141_quiet32k_v1.py', 'prepare_native_expert_copy_v0141_quiet_sequence_v1.py',
                 'summarize_native_expert_copy_v0141_quiet_v1.py', 'archive_native_expert_copy_v0141_quiet_v1.py'):
    copy(B / filename, 'controllers/' + filename)
for relative, filename in {
    'controller.json': 'native-expert-copy-v0141-quiet-controller-preparation-v1.json',
    'cpu-preflight.json': 'native-expert-copy-v0141-quiet-cpu-preflight-v1.json',
    'sequence.json': 'native-expert-copy-v0141-quiet-sequence-preparation-v1.json',
}.items():
    copy(B / filename, 'preparation/' + relative)
for step in sequence['steps']:
    receipt = Path(step['receipt'])
    assert digest(receipt) == step['receipt_sha256']
    data = json.loads(receipt.read_text())
    assert not data['active'] and data['healthy'] and data['completed'] and data['math_gate_passed']
    assert data['native_copy_queue_startup_gate_passed']
    assert data['exit_code'] == 0 and not data['exit_signal'] and not data['new_fault_messages'] and not any(data['cleanup'].values())
    for filename in ('record.json', 'events.jsonl', 'protocol.stdout.raw', 'project-messages.txt'):
        copy(receipt.parent / filename, 'runs/' + receipt.parent.name + '/' + filename)
with (OUT / 'summary.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary['rows'][0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(summary['rows'])
g, effects = summary['groups'], summary['effects']
table = '\n'.join(f"| {mode} | {g[mode]['first']['pooled_prefill_tok_s']:.3f} | {g[mode]['later']['pooled_prefill_tok_s']:.3f} | {g[mode]['later']['pooled_decode_tok_s']:.3f} |" for mode in ('baseline', 'nativeoff', 'nativeon'))
ranges = '\n'.join(f"| {mode} | {g[mode]['later']['prefill_min']:.3f}–{g[mode]['later']['prefill_max']:.3f} | {g[mode]['later']['decode_min']:.3f}–{g[mode]['later']['decode_max']:.3f} |" for mode in ('baseline', 'nativeoff', 'nativeon'))
pp_gain = effects['nativeon']['pooled_prefill_tok_s_relative_to_baseline_percent']
pp_off = effects['nativeon_vs_nativeoff']['pooled_prefill_tok_s_relative_percent']
tg_change = effects['nativeon']['pooled_decode_tok_s_relative_to_baseline_percent']
text = f'''# Quiet32K native expert-copy comparison

On2026-10-09 JST, Arc B57010 GiB / Ryzen5 5600X /128 GiB RAM ran baseline/off/on/on/off/baseline, four fresh32768-token A/B/A/B inputs and64 visible outputs per process. Context262144, actual8192 chunks,128 cache slots/325 MiB and resident KV32768 match. Every argument/environment value matches except the binary and explicit native-copy flag. No API logs, validation layer, profiler, head/state dumps, event-timestamp queries or extra phase waits were enabled. Existing completion checks and ring barriers remain.

| Mode | First prefill tokens/s (2 reads) | Later prefill tokens/s (6 reads) | Later decode tokens/s (6 reads) |
| --- | ---: | ---: | ---: |
{table}

Baseline is qualified binary86972697. Off/on are uniformly rebuilt binary2721f8ef with `STRATA_PREFILL_COPY_ENGINE=0/1`. Rates are pooled token/time ratios. First process reads are separate from later fresh reads. Native-on later prefill improves by{pp_gain:.3f}% against baseline and{pp_off:.3f}% against candidate-off. The native queue factory is the source difference under test; all114 compiler commands match the baseline after normalizing worktree/build/dependency-output paths.

| Mode | Later prefill per-read range | Later decode per-read range |
| --- | ---: | ---: |
{ranges}

Native-on decode changes by{tg_change:.3f}% against baseline; the per-read spread and sample do not establish a small decode gain or regression. All modes request five logprob alternatives for every output token. Decode time includes their readback and CPU scoring; no-logprob generation is a separate unmeasured condition. Every raw read and process-level pooled rate is preserved in JSON/CSV.

All24 outputs, printed logprobs, MTP counts, finish reasons and repeats match the qualified baseline. Every read has RESUME0/REUSED0. All six processes finish with normalQUIT/exit0, no new GPU fault, no forced cleanup and no owned survivor. These quiet checks concern visible references; [the separate first diagnostic](../first-diagnostic/README.md) supplies four-fresh first-head/all66-live-state proof and a bounded native-queue transfer trace.

The change routes expert copies through a validated copy-only immediate queue while retaining the existing compute queue, math, CPU pool, ring reuse barriers and DMA-event ownership. This throughput comparison does not measure DMA active time or an exclusive PCIe wait fraction. The bounded diagnostic trace confirms captured UR/native expert copies use the imported ordinal1 queue; it is not a trace of every quiet transfer.

Candidate272 is not adopted and its full262144 lifecycle is pending. Baseline869 full qualification does not qualify this different binary. Terminal receipts and exact measured controllers bind binaries, source, flags, runtime, fixtures, protocol and cleanup. Large logs, binaries and tensor/session states remain private.
'''
(OUT / 'README.md').write_text(text)
(PARENT / 'README.md').write_text(f'''# Native expert-copy queue trial

On2026-10-09 JST, Arc B57010 GiB / Ryzen5 5600X /128 GiB RAM completed [four first-use32K head/live-state/output checks](first-diagnostic/README.md) and [24 quiet32K reads](quiet32k/README.md). Native copy-only selection improves later prefill from{g['baseline']['later']['pooled_prefill_tok_s']:.3f} to{g['nativeon']['later']['pooled_prefill_tok_s']:.3f} tokens/s (+{pp_gain:.3f}%). Candidate-off provides a separate uniformly rebuilt control. Decode variation does not establish a small speed change; logprob scoring is included in these timings.

The change is opt-in and retains existing math, CPU work, completion checks and ring barriers. All owned runs finish normally. The candidate remains unadopted until its own full262144 lifecycle qualification; that gate is pending. The first diagnostic archive preserves its historical first-use scope.
''')
root_readme = PARENT.parent / 'README.md'
old = 'The [0.1.41 native-copy first diagnostic](native-expert-copy-v0141-20261009/README.md) confirms four fresh32K numerical/state matches and normal cleanup for an opt-in copy-only immediate queue. A bounded UR/Level Zero trace pairs880 expert copies through the imported queue. The uniform build matches all114 baseline compiler commands. Logged times are excluded; quiet speed comparison and full262144 qualification are pending, and the candidate is not adopted.'
new = f'The [0.1.41 native-copy trial](native-expert-copy-v0141-20261009/README.md) confirms four fresh32K numerical/state matches and24 quiet32K reads. Copy-only selection improves later prefill from{g["baseline"]["later"]["pooled_prefill_tok_s"]:.3f} to{g["nativeon"]["later"]["pooled_prefill_tok_s"]:.3f} tokens/s (+{pp_gain:.3f}%) on Arc B570 / Ryzen5600X /128 GiB RAM. A bounded UR/Level Zero trace pairs880 expert transfers through the imported queue. All114 compiler commands match. Full262144 qualification is pending; the candidate is not adopted.'
s = root_readme.read_text(); assert s.count(old) == 1
root_readme.write_text(s.replace(old, new))
manifest = {str(p.relative_to(OUT)): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT / 'manifest.json').write_text(json.dumps({'files': manifest}, indent=2) + '\n')
print(json.dumps({'archived': True, 'files': len(manifest), 'directory': str(OUT), 'manifest_sha256': digest(OUT/'manifest.json')}, indent=2))
