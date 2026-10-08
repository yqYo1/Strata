"""Archive terminal first-use evidence; never copy live quiet receipts or API logs."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import re
import shutil
import subprocess

B = Path(__file__).parent
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-server-env-v0141-2026-10-09')
OUT = W / 'bench/results/2026-10-05-sycl-prefill-scaling/host-prefill-accounting-v0141-20261009/first-diagnostic'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


assert not OUT.exists()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == '16b1b04ffd77101c4b49543507c06fc71fbf2368'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True).strip()
run = B / 'owned-host-accounting-v0141-code32k-diagnostic-r1'
receipt = run / 'record.json'
assert digest(receipt) == '6605f3c24444b6069ff10ab594e429ee66c6498a8a9f0cc21d6fcd33e388b2ed'
d = json.loads(receipt.read_text())
assert not d['active'] and d['completed'] and d['healthy'] and d['math_gate_passed'] and d['host_accounting_gate_passed']
assert d['exit_code'] == 0 and not d['exit_signal'] and not d['new_fault_messages'] and not any(d['cleanup'].values())
assert not d['performance_eligible'] and not d['full_lifecycle_passed'] and not d['adopted']
assert len(d['requests']) == len(d['host_accounting_reports']) == 4
reports = d['host_accounting_reports']
rows = []
for request, report in zip(d['requests'], reports):
    assert request['math_gate_passed'] and report['tokens'] == 32767 and report['chunks'] == 4
    layers = report['layers']
    assert [x['layer'] for x in layers] == list(range(48))
    assert sum(x['copies'] for x in layers) == report['expert_copies']
    assert sum(x['bytes'] for x in layers) == report['expert_bytes']
    assert all(x['copies'] > 0 and x['bytes'] % x['copies'] == 0 for x in layers)
    blobs = [x['bytes'] // x['copies'] for x in layers]
    rows.append({'request': request['name'], 'expert_copies': report['expert_copies'],
                 'expert_bytes': report['expert_bytes'], 'expert_GB_decimal': report['expert_bytes'] / 1e9,
                 'expert_GiB': report['expert_bytes'] / 2**30,
                 'hypothetical_seconds_at_6GB_s': report['expert_bytes'] / 6e9,
                 'hypothetical_transfer_only_tokens_s_at_6GB_s': 32767 * 6e9 / report['expert_bytes'],
                 'inferred_per_layer_blob_bytes': blobs,
                 'hypothetical_load_all_512_experts_once_bytes': sum(blobs) * 512,
                 'hypothetical_max_single_layer_512_experts_bytes': max(blobs) * 512})
assert reports[0]['layers'] == reports[2]['layers'] and reports[1]['layers'] == reports[3]['layers']
assert all(x['inferred_per_layer_blob_bytes'] == rows[0]['inferred_per_layer_blob_bytes'] for x in rows)

# A bounded prefix includes startup and part of the first prefill. Do not hash
# or scan the multi-gigabyte full API log a second time.
stderr = run / 'debugger/inferior.stderr'
with stderr.open('rb') as f:
    prefix = f.read(32 * 1024 * 1024)
lines = prefix.decode('utf-8', 'replace').splitlines()
cl, urq = '0x4ce3ae8', '0x4cc05c0'
creation = [(i, l) for i, l in enumerate(lines) if 'SUCCESS' in l and 'zeCommandListCreateImmediate(' in l and f'phCommandList={cl})' in l]
urcreation = [(i, l) for i, l in enumerate(lines) if '<--- urQueueCreate(' in l and f'({urq}))' in l]
assert len(creation) == len(urcreation) == 1
assert 'ordinal=0, index=0, flags=2' in creation[0][1] and '.flags = 0}' in urcreation[0][1]
assert urcreation[0][0] - creation[0][0] <= 5
copies = [(i, l) for i, l in enumerate(lines) if 'SUCCESS' in l and 'zeCommandListAppendMemoryCopy(' in l and f'hCommandList={cl},' in l]
urcopies = [(i, l) for i, l in enumerate(lines) if '<--- urEnqueueUSMMemcpy(' in l and f'.hQueue = {urq},' in l]
assert len(copies) == len(urcopies) == 880
size_hist = collections.Counter()
destinations = set()
for (zi, zl), (ui, ul) in zip(copies, urcopies):
    # Other threads' queue/event logs can interleave between append and its UR
    # return. Pair in issuer order and verify all memory operands below.
    assert ui > zi
    size = int(re.search(r'size=(\d+)', zl)[1])
    dst = re.search(r'dstptr=(0x[0-9a-f]+)', zl)[1]
    src = re.search(r'srcptr=(0x[0-9a-f]+)', zl)[1]
    assert f'.pDst = {dst},' in ul and f'.pSrc = {src},' in ul and f'.size = {size},' in ul
    size_hist[size] += 1
    destinations.add(dst)
assert size_hist == {2176000: 487, 1510400: 393} and len(destinations) == 8
group = [(i, l) for i, l in enumerate(lines[:700]) if 'SUCCESS' in l and 'zeDeviceGetCommandQueueGroupProperties(' in l and 'flags=7' in l]
assert len(group) == 1 and 'numQueues=1' in group[0][1]
assert any('main blitter/copy engine is available' in l for l in lines[:700])
trace = {
    'scope': 'Exact paired UR/L0 expert-copy calls in the first 32 MiB of an already completed logged run. No new GPU execution or healthy-process attachment.',
    'original_log': str(stderr), 'original_log_bytes': d['engine_log_bytes'], 'original_log_sha256_from_terminal_receipt': d['engine_log_sha256'],
    'prefix_bytes': len(prefix), 'prefix_sha256': hashlib.sha256(prefix).hexdigest(),
    'command_list': cl, 'ur_queue': urq, 'ordinal': 0, 'index': 0, 'command_queue_flags': 2,
    'ur_queue_flags': 0, 'group0_flags': 7, 'group0_physical_queues': 1,
    'paired_memcpy_count': len(copies), 'paired_bytes': sum(k*v for k,v in size_hist.items()),
    'maximum_interleaved_lines_between_native_append_and_ur_return': max(ui-zi for (zi,_),(ui,_) in zip(copies,urcopies)),
    'copy_size_histogram': dict(size_hist), 'destination_count': len(destinations),
    'interpretation': 'The captured ring copies use ordinal0, which reports COMPUTE|COPY|COOPERATIVE_KERNELS. A main blitter is reported available. The sample does not prove all later transfers use this handle, quantify compute/transfer overlap, or prove native-copy-queue safety/performance.',
    'line_numbers_zero_based': True,
    'evidence': [{'line': i, 'text': l} for i,l in group + creation + urcreation + copies[:8] + urcopies[:8]],
}
analysis = {
    'scope': 'Four fresh 32768 inputs and64 outputs; numerical/state correctness and exact expert H2D counts. Logged times are excluded from speed.',
    'terminal_receipt_sha256': digest(receipt), 'candidate_binary_sha256': d['binary_sha256'],
    'adopted': False, 'full_lifecycle_passed': False, 'rows': rows,
    'limits': ['Counts cover expert copies, not every activation/KV/dense transfer.',
               'The 6 GB/s values are hypothetical lower bounds at the prior approximate measured application bandwidth, not a new bandwidth measurement or measured DMA duration.',
               'RAM-worker sums and GPU-wait durations overlap with other activity; they must not be added to wall time.',
               'Loading each layer once is a byte-count hypothesis, not a validated implementation; KV streaming and full-capacity state ordering require further work.',
               'The candidate has four-fresh 32K numerical/state proof, but has not passed the full262144 lifecycle.']
}
OUT.mkdir(parents=True)
write(OUT / 'expert-byte-analysis.json', analysis)
write(OUT / 'bounded-native-copy-queue-evidence.json', trace)
files = {
    'run/record.json': receipt,
    'run/events.jsonl': run / 'events.jsonl',
    'run/protocol.stdout.raw': run / 'protocol.stdout.raw',
    'run/project-messages.txt': run / 'project-messages.txt',
    'candidate/prefill.cpp': B / 'host-prefill-accounting-v0141-source-v1/prefill.cpp',
    'cpu/record.json': B / 'host-prefill-accounting-v0141-cpu-check-v1/record.json',
    'build/record.json': B / 'host-prefill-accounting-v0141-private-build-v1/record.json',
    'build/plan.json': B / 'host-prefill-accounting-v0141-private-build-plan-v1/record.json',
    'controllers/run_owned_host_accounting_v0141_code32k_v1.py': B / 'run_owned_host_accounting_v0141_code32k_v1.py',
    'controllers/build_host_prefill_accounting_v0141_private_v2.py': B / 'build_host_prefill_accounting_v0141_private_v2.py',
    'controllers/archive_host_prefill_accounting_v0141_first_v1.py': Path(__file__),
    'preparation/first-controller.json': B / 'host-prefill-accounting-v0141-first-controller-preparation-v1.json',
}
for relative, source in files.items():
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert digest(target) == digest(source)
summary = f'''# Private host accounting: first diagnostic

On 2026-10-09 JST, Arc B57010 GiB / Ryzen5 5600X /128 GiB RAM ran four fresh32768-token inputs with64 outputs, context262144, actual8192 chunks and128 expert-cache slots. Private binary `{d['binary_sha256']}` adds CPU counters and steady-clock measurements to the qualified0.1.41 engine. It adds no GPU event queries, markers or waits. This is first-use correctness evidence; API logging and state dumps exclude its durations from speed comparisons.

All four first-head logits and66 live-state parts match the qualified updated baseline under the existing live-state comparator; IDs, logprobs, MTP counts, finish reasons and repeats match. The comparator excludes documented rounded-page padding, not active state or indexer spare state. The run finishes normally with exit0, no new GPU fault, no forced cleanup and no owned survivor. The instrumented engine remains private and unadopted; its full262144 lifecycle is pending.

| Fixture | Expert copies | Expert bytes | Hypothetical seconds at6 GB/s |
| --- | ---: | ---: | ---: |
| A and A repeat | {rows[0]['expert_copies']} | {rows[0]['expert_bytes']} | {rows[0]['hypothetical_seconds_at_6GB_s']:.3f} |
| B and B repeat | {rows[1]['expert_copies']} | {rows[1]['expert_bytes']} | {rows[1]['hypothetical_seconds_at_6GB_s']:.3f} |

These are expert H2D counts for32767 prefilled tokens. They do not cover every transfer. The6 GB/s estimate is the prior approximate application bandwidth, not a new measurement. It gives a transfer-only lower bound near32 s and a transfer-only ceiling near1033 tokens/s at this byte count; actual DMA duration, contention and overlap are not measured here.

A bounded32 MiB prefix of the already completed API log pairs880 UR expert copies with880 native copies,1653299200 bytes across eight ring destinations. The UR queue and native immediate command-list creation are paired; ordinal0 reports compute/copy/cooperative flags7 and one physical queue. The runtime also reports a main blitter available. This supports investigating transfer/compute overlap using a copy-only engine. It does not establish the safety or speed of a new queue. The whole API log was neither recopied nor rescanned.

Code inspection separately confirms that `NativeHead::load` puts `output.weight` in device memory. The host-mapped `NativeEmbed` table is `token_embd.weight`, used for embedding gathers; it is not the output classifier. Moving that table is therefore not assumed to remove a full-table read from each decoded token.

The separate quiet baseline/off/on comparison is still live and is not included in this immutable archive. No speed or full-capacity adoption decision is made from this diagnostic. JSON receipts bind source, runtime, binary, fixtures, comparators and cleanup; the manifest hashes the archived files. Large logs, binaries and state/session tensors remain private.
'''
(OUT / 'README.md').write_text(summary)
manifest = {str(p.relative_to(OUT)): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.rglob('*')) if p.is_file()}
write(OUT / 'manifest.json', {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': manifest})
print(json.dumps({'archived': True, 'directory': str(OUT), 'files': len(manifest), 'manifest_sha256': digest(OUT/'manifest.json'), 'rows': [{k:v for k,v in x.items() if not k.startswith('inferred')} for x in rows]}, indent=2))
