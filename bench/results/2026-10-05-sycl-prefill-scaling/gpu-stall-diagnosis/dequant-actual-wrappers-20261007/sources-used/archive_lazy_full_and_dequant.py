"""Freeze the terminal lazy-full failure/health and actual-wrapper receipts."""
from pathlib import Path
import array
import hashlib
import json
import math
import shutil
import sys

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent = root / 'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis'
sys.path.insert(0, str(root / 'sycl/tools'))
from owned_gdb import process_identity


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy(source, directory, target):
    destination = directory / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    assert digest(source) == digest(destination)


def save_folder(source, directory, prefix):
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.suffix in ('.bin', '.o', '.a', '.d') or '__pycache__' in path.parts:
            continue
        target = Path(prefix) / path.relative_to(source)
        if path.stat().st_size <= 300000:
            copy(path, directory, target)
        else:
            destination = directory / target.with_name(path.name + '.tail')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 65536))
                destination.write_bytes(stream.read())
            metadata = {'private_path': str(path), 'private_bytes': path.stat().st_size,
                        'private_sha256': digest(path), 'saved_tail': str(destination.relative_to(directory)),
                        'saved_tail_bytes': destination.stat().st_size}
            destination.with_name(destination.name + '.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def manifest(directory):
    files = sorted(p for p in directory.rglob('*') if p.is_file())
    entries = [{'path': str(p.relative_to(directory)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files]
    (directory / 'manifest.json').write_text(json.dumps({'entries': entries}, indent=2) + '\n')
    assert {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} == {e['path'] for e in entries} | {'manifest.json'}
    assert all((directory / e['path']).stat().st_size == e['bytes'] and digest(directory / e['path']) == e['sha256'] for e in entries)
    return {'archive': str(directory), 'files': len(entries) + 1, 'payload_bytes': sum(e['bytes'] for e in entries), 'exact_manifest_checked': True}


full = json.loads((base / 'full-context-verifier-lazy-restore-serve/record.json').read_text())
health = json.loads((base / 'post-full-verifier-lazy-restore-health/record.json').read_text())
dequant = json.loads((base / 'dequant-actual-wrapper-diagnostic-v2/record.json').read_text())
assert not full['active'] and not full['completed'] and not full['healthy']
assert full['active_request'] == 'fills-context-tail-2'
assert not full['new_fault_messages'] and not full['cleanup']['inferior_survived'] and not full['cleanup']['gdb_survived']
assert health['healthy'] and health['boot_id'] == full['boot_id'] == dequant['boot_id']
assert not dequant['active'] and dequant['passed'] and not dequant['new_fault_messages']
for owner in [full['inferior'], full['debugger']] + [job[k] for job in dequant['runs'] for k in ('inferior', 'debugger')]:
    identity = process_identity(owner['pid'])
    assert not identity or identity['start_ticks'] != owner['start_ticks']
for step in health['steps']:
    assert step['exit_code'] == 0 and not step['timed_out'] and not step['still_alive']
    assert not Path('/proc', str(step['pid'])).exists()
first = full['requests'][0]
assert len(full['requests']) == 1 and first['input_tokens'] == 262140 and first['max_new'] == 4
assert len(first['ids']) == 4 and first['last_executed_kv_cell'] == 262143
assert first['draft_releases'] == first['draft_restores'] == 1
head = base / 'full-context-verifier-lazy-restore-serve/fills-context.head.bin'
payload = head.read_bytes()
values = array.array('f')
values.frombytes(payload)
assert len(values) == 248320 and all(map(math.isfinite, values))
assert digest(head) == first['head']['sha256']

full_dir = parent / 'verifier-lazy-full-20261007'
dequant_dir = parent / 'dequant-actual-wrappers-20261007'
assert not full_dir.exists() and not dequant_dir.exists()
full_dir.mkdir(mode=0o755)
dequant_dir.mkdir(mode=0o755)
for name in ['run_full_verifier_lazy_restore_serve.py', 'probe_after_full_verifier_lazy_restore.py',
             'run_full_eager_verifier_serve.py', 'archive_lazy_full_and_dequant.py']:
    copy(base / name, full_dir, Path('sources-used') / name)
save_folder(base / 'full-context-verifier-lazy-restore-serve', full_dir, 'full-run')
save_folder(base / 'post-full-verifier-lazy-restore-health', full_dir, 'health')
points = []
with (base / 'full-context-verifier-lazy-restore-serve/debugger/inferior.stderr').open('rb') as stream:
    for line in stream:
        if line.startswith(b'strata prefill memory:'):
            points.append(line.decode().strip())
(full_dir / 'memory-checkpoints.txt').write_text('\n'.join(points) + '\n')
assessment = {'scope': 'First normal-MTP request consumes all262144 cells and emits four finite outputs; next prefill aborts during64MiB main-cache restoration. Remaining capacity gates fail to complete; no clean timing or prevention claim.',
              'binary_sha256': full['binary_sha256'], 'first_request': first,
              'complete_full_capacity_suite': False, 'first_full_head_compared_to_eager': False,
              'failure_request': full['active_request'], 'crash': 'SIGABRT at CacheLease destructor terminate, captured before continuation',
              'cleanup': full['cleanup'], 'new_fault_messages': full['new_fault_messages'],
              'post_cleanup_exact_word_health': health['healthy'], 'reset_performed': False,
              'temporary_layer_bytes_observed': 1414582272 - 51429376,
              'free_after_second_release': 1244880896,
              'original_return_reason_logged': False,
              'source_review': {'path': str(root / 'sycl/src/prefill/prefill.cpp'), 'sha256': digest(root / 'sycl/src/prefill/prefill.cpp'),
                  'lines': [2110, 2118, 2155, 2158],
                  'inference': 'Second release exposes less free VRAM than the preceding successful whole-layer allocation consumed; open_sized returns before the allocation trace. Destructor restoration then terminates on failure. The specific retained170MB allocation is not identified.'},
              'adopted': False, 'next_comparison_controller': 'sources-used/run_full_eager_verifier_serve.py',
              'next_comparison_full_result_included': False}
(full_dir / 'assessment.json').write_text(json.dumps(assessment, indent=2) + '\n')
(full_dir / 'README.md').write_text('''# Lazy verifier full-context check on 2026-10-07

The private lazy verifier candidate completes its first normal-MTP request
with 262,140 input and four output tokens on B570 10 GiB / Ryzen 5600X /
128 GiB RAM, oneAPI 2026.1.1 and the recorded NEO/Level Zero versions.
The windows [262139,1], [262140,4], [262141,3] execute through KV cell
262143. All four printed logprobs and the complete 248,320-float head are
finite. One verified MTP release/restore pair completes. The separate
eager comparison has not yet validated this full head.

The next 262,142-input request fails before the temporary layer allocation
checkpoint. Main-cache restoration in `CacheLease` destruction then fails
to allocate a 64 MiB physical segment and calls terminate. GDB captures
SIGABRT there. Both owned processes are removed during forced cleanup;
no new xe fault/reset is recorded. The following logged H2D/kernel/D2H
probe passes all three rounds of 16,384 exact words without a reset.

The [memory checkpoints](memory-checkpoints.txt) show 1,414,582,272 bytes
free before the first successful temporary allocation, which consumes
1,363,152,896 bytes. After the second main/MTP release, only
1,244,880,896 bytes are free. Source review supports an insufficient
whole-layer allocation budget, but the original return error is obscured
by failed destructor restoration. The specific allocation retaining the
extra memory is not identified. Lazy capture alone is insufficient for
the complete repeated-capacity test.

The clipped-tail, overflow-refusal and later-valid-request gates do not
complete. No diagnostic duration is accepted as clean throughput, and
the production verifier is unchanged. [Assessment](assessment.json),
terminal [record](full-run/record.json), GDB crash state, bounded log tails
and full private-log hashes preserve the distinction between first-request
success and suite failure. The [next eager controller](sources-used/run_full_eager_verifier_serve.py)
uses the existing eager-window setting and unchanged 3f executable;
its pending full result is not included here. Each manifest covers every
archived file except itself; binaries and full large logs stay private.
''')

for name in ['run_dequant_actual_wrapper_probe_v2.py', 'dequant_actual_wrapper_probe.cpp',
             'archive_lazy_full_and_dequant.py']:
    copy(base / name, dequant_dir, Path('sources-used') / name)
copy(base / 'dequant-actual-wrapper-analysis.json', dequant_dir, Path('analysis.json'))
save_folder(base / 'dequant-actual-wrapper-diagnostic-v2', dequant_dir, 'run')
assert len(dequant['whole_guarded_outputs']) == 144 and all(v['all_bytes_equal'] for v in dequant['whole_guarded_outputs'])
analysis = json.loads((base / 'dequant-actual-wrapper-analysis.json').read_text())
assert [(x['observed_successful_dequant_launches'], x['cooperative_launches']) for x in analysis['runs']] == [(144, 144), (144, 0)]
(dequant_dir / 'README.md').write_text('''# Actual dequant wrapper comparison on 2026-10-07

This follows the [CPU-only preparation](../dequant-launch-properties-20261007/README.md)
on the same B570 host and boot, after the failed full-context process was
removed and a logged GPU health probe passed without reset. The revised
controller records those prerequisites and owns each child through GDB.

Both executables use one shared host fixture object. Their actual engine
kernel objects differ only in the two launch-property declarations. All
144 pairs of guarded FP16 outputs match every byte, totaling 354,511,872
bytes per side. Nine types, seeds 7/23, small/640/641-row shapes, flat/GU
layouts and zero/positive/negative scales produce finite values and intact
guards. Both processes exit normally without force, surviving owners or
new xe faults.

The [kernel-specific API analysis](analysis.json) observes 72 flat and
72 GU successful launches in each process. All 144 original launches use
the cooperative flag; none of the changed launches do. The [complete
receipt](run/record.json) preserves output sizes/hashes, environment,
binary/object/controller provenance and lifecycle checks. API log tails
are bounded; their full private hashes are retained.

These are isolated actual-wrapper checks. The private engine executable
has not run a model, and these logged durations are not clean speed
measurements. Full-model heads/state, repeated performance, capacity and
a causal connection to earlier stalls remain unproven. No production
kernel change is adopted by this evidence.
''')
print(json.dumps([manifest(full_dir), manifest(dequant_dir)], indent=2))
