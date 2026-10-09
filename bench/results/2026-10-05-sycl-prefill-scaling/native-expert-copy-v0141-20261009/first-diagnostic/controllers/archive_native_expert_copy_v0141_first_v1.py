"""Archive completed first-use proof without copying live timings or large logs."""
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
N = W.parent / 'perf-sycl-prefill-native-copy-v0141-20261009'
OUT = W / 'bench/results/2026-10-05-sycl-prefill-scaling/native-expert-copy-v0141-20261009/first-diagnostic'
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, d: p.write_text(json.dumps(d, indent=2) + '\n')

assert not OUT.exists()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == 'fe96101e2c3c112d1b7a36caaff66ceda80334ff'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True).strip()
run = B / 'owned-native-expert-copy-v0141-code32k-diagnostic-r1'
receipt = run / 'record.json'
assert digest(receipt) == '83670ee10b75d702c621e0ce029c5be1a4701ac6684d08b4f22de905704ae908'
d = json.loads(receipt.read_text())
assert not d['active'] and d['completed'] and d['healthy'] and d['math_gate_passed']
assert d['native_copy_queue_startup_gate_passed'] and all(x == 1 for x in d['native_copy_queue_ordinals'])
assert d['exit_code'] == 0 and not d['exit_signal'] and not d['new_fault_messages'] and not any(d['cleanup'].values())
assert not d['performance_eligible'] and not d['full_lifecycle_passed'] and not d['adopted']
assert len(d['requests']) == 4 and all(x['math_gate_passed'] for x in d['requests'])
assert all(x['first_head']['finite'] and x['first_head']['floats'] == 248320 and len(x['prefill_state']['parts']) == 66 for x in d['requests'])
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=N, text=True).strip() == d['commit']
assert digest(N / 'sycl/src/prefill/prefill.cpp') == d['candidate_source_sha256']
assert digest(N / 'sycl/include/dpct/device.hpp') == d['candidate_header_sha256']

# Read only the bounded startup/prefill prefix of the terminal diagnostic.
# Reuse the complete log hash that the owned controller already computed.
stderr = run / 'debugger/inferior.stderr'
with stderr.open('rb') as stream:
    prefix = stream.read(32 * 1024**2)
lines = prefix.decode('utf-8', 'replace').splitlines()
cl, urq = '0x4cf4148', '0x4cf82d0'
creation = [(i, l) for i, l in enumerate(lines) if 'SUCCESS' in l and 'zeCommandListCreateImmediate(' in l and f'phCommandList={cl})' in l]
flags = [(i, l) for i, l in enumerate(lines) if 'SUCCESS' in l and 'zeCommandListImmediateGetFlags(' in l and f'hCommandList={cl}, pFlags=2)' in l]
imported = [(i, l) for i, l in enumerate(lines) if '<--- urQueueCreateWithNativeHandle(' in l and f'.hNativeQueue = {cl},' in l and f'({urq}))' in l]
assert len(creation) == len(flags) == len(imported) == 1
assert 'ordinal=1, index=0, flags=2' in creation[0][1]
assert '.isNativeHandleOwned = 1}' in imported[0][1] and 'UR_RESULT_SUCCESS' in imported[0][1]
assert creation[0][0] < flags[0][0] < imported[0][0]
copies = [(i, l) for i, l in enumerate(lines) if 'SUCCESS' in l and 'zeCommandListAppendMemoryCopy(' in l and f'hCommandList={cl},' in l]
urcopies = [(i, l) for i, l in enumerate(lines) if '<--- urEnqueueUSMMemcpy(' in l and f'.hQueue = {urq},' in l]
assert len(copies) == len(urcopies) == 880
histogram = collections.Counter()
destinations, gaps = set(), []
for (zi, zl), (ui, ul) in zip(copies, urcopies):
    size = int(re.search(r'size=(\d+)', zl)[1])
    dst = re.search(r'dstptr=(0x[0-9a-f]+)', zl)[1]
    src = re.search(r'srcptr=(0x[0-9a-f]+)', zl)[1]
    assert ui > zi
    assert f'.pDst = {dst},' in ul and f'.pSrc = {src},' in ul and f'.size = {size},' in ul
    histogram[size] += 1
    destinations.add(dst)
    gaps.append(ui - zi)
assert histogram == {2176000: 487, 1510400: 393} and len(destinations) == 8
trace = {
    'scope': '880 exact paired expert-copy calls in32 MiB of a terminal logged run. No new GPU execution or attachment.',
    'original_log': str(stderr), 'original_log_bytes': d['engine_log_bytes'],
    'original_log_sha256_from_terminal_receipt': d['engine_log_sha256'],
    'prefix_bytes': len(prefix), 'prefix_sha256': hashlib.sha256(prefix).hexdigest(),
    'native_command_list': cl, 'imported_ur_queue': urq, 'ordinal': 1, 'index': 0,
    'immediate_command_queue_flags': 2, 'native_handle_owned_by_runtime': True,
    'paired_memcpy_count': len(copies), 'paired_bytes': sum(k*v for k,v in histogram.items()),
    'copy_size_histogram': dict(histogram), 'destination_count': len(destinations),
    'maximum_interleaved_lines_between_native_append_and_ur_return': max(gaps),
    'line_numbers_zero_based': True,
    'evidence': [{'line': i, 'text': l} for i, l in creation + flags + imported + copies[:8] + urcopies[:8]],
    'limits': ['The factory checks COPY without COMPUTE; this excerpt records ordinal1 and validates native import identity.',
               'The paired sample does not prove all subsequent transfers, DMA duration or overlap with compute.',
               'Logged durations are excluded from performance. Four32K reads do not establish full262144 lifecycle correctness or general hang prevention.'],
}
OUT.mkdir(parents=True)
write(OUT / 'bounded-native-copy-queue-evidence.json', trace)
files = {
    'run/record.json': receipt,
    'run/events.jsonl': run / 'events.jsonl',
    'run/protocol.stdout.raw': run / 'protocol.stdout.raw',
    'run/project-messages.txt': run / 'project-messages.txt',
    'candidate/device.hpp': N / 'sycl/include/dpct/device.hpp',
    'candidate/prefill.cpp': N / 'sycl/src/prefill/prefill.cpp',
    'preparation/source.json': B / 'native-expert-copy-v0141-source-preparation-v1.json',
    'preparation/source-commit.json': B / 'native-expert-copy-v0141-preparation-commit-v1.json',
    'preparation/first-controller.json': B / 'native-expert-copy-v0141-first-controller-preparation-v1.json',
    'preparation/first-cpu-preflight.json': B / 'native-expert-copy-v0141-first-cpu-preflight-v1.json',
    'build/record.json': B / 'native-expert-copy-v0141-private-build-v1/record.json',
    'build/uniform-compiler-flags.json': B / 'native-expert-copy-v0141-uniform-build-flags-v1.json',
    'controllers/prepare_owned_native_expert_copy_v0141_code32k_v1.py': B / 'prepare_owned_native_expert_copy_v0141_code32k_v1.py',
    'controllers/run_owned_native_expert_copy_v0141_code32k_v1.py': B / 'run_owned_native_expert_copy_v0141_code32k_v1.py',
    'controllers/build_native_expert_copy_v0141_v1.py': B / 'build_native_expert_copy_v0141_v1.py',
    'controllers/archive_native_expert_copy_v0141_first_v1.py': Path(__file__),
}
offline = B / 'installed-ur-native-queue-offline-20261009-v1'
for name in ('native-create.asm', 'native-constructor.asm', 'native-deleter.asm'):
    files['installed-runtime/' + name] = offline / name
for relative, source in files.items():
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert digest(source) == digest(target)

(OUT / 'README.md').write_text(f'''# Native expert-copy queue: first diagnostic

On 2026-10-09 JST, Arc B57010 GiB / Ryzen5 5600X /128 GiB RAM ran four fresh32768-token inputs with64 outputs, context262144, actual8192 chunks and128 expert-cache slots. Candidate commit `{d['commit']}` and binary `{d['binary_sha256']}` select a Level Zero copy-only immediate queue when `STRATA_PREFILL_COPY_ENGINE=1`. Unset or0 uses the previous queue. This diagnostic uses flushed UR/Level Zero logs and state dumps; its times are excluded from performance.

All four first-head logits are finite and match the qualified0.1.41 baseline. All66 live prefill-state parts, IDs, logprobs, MTP counts, finish reasons and repeated results match under the existing comparator. Only documented rounded-page padding is excluded. The run finishes normally with exit0, no new GPU fault, no forced cleanup and no owned survivor. This experimental binary has not passed the full262144 lifecycle and is not adopted.

The factory enumerates queue groups and requires COPY without COMPUTE, an in-order immediate command list and the existing context/device. It checks `zeCommandListImmediateGetFlags` and imports the same native command list with runtime ownership. It adds the imported SYCL queue to the existing device queue registry. The compute kernels, CPU pool, ring barriers and retained DMA events are unchanged. Native profiling is rejected for this candidate; it adds no phase waits or event-timestamp queries. The import-error path aborts initialization without retry; an error before runtime adoption can leave an empty command list until process exit. This is a documented prototype limitation.

A bounded32 MiB prefix records native creation at ordinal1/index0/flags2, immediate flags2, and successful UR import of that same command list with ownership1. It pairs880 expert copies across eight ring destinations,1653299200 bytes. Each pair has the same source, destination and size at UR and Level Zero. This confirms the captured expert ring uses the imported queue. It does not measure DMA time or prove compute/transfer overlap or all later transfers.

The fresh uniform build matches all114 baseline compiler commands after normalizing worktree/build and dependency-output paths. The source-preparation proof records preservation of the old header definitions and prefill body outside queue selection. Installed UR v2 library SHA256 is `bfdc0f26bf88bd0bccd66fc25302a4b22559f8fe30e499be18529be3ed610e60`; its source commit is unknown. Archived offline disassembly establishes that its native constructor forwards the supplied command list and its owned deleter destroys it. Current official source is not presented as this installed binary's exact source.

The separate baseline/off/on quiet comparison is pending and is not part of this immutable first-use archive. Large API logs, binaries and state tensors remain private. The manifest binds every archived file; build and preparation receipts keep their historical scope.
''')
manifest = {str(p.relative_to(OUT)): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(OUT.rglob('*')) if p.is_file()}
write(OUT / 'manifest.json', {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': manifest})
print(json.dumps({'archived': True, 'directory': str(OUT), 'files': len(manifest), 'manifest_sha256': digest(OUT/'manifest.json')}, indent=2))
