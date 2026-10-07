"""Archive the real short model/debugger proof without its full API log."""
from pathlib import Path
from collections import Counter, deque
import hashlib
import json
import re
import shutil

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
case = base / 'owned-workspace-reclaim-short'
out = root / 'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/workspace-reclaim-20261007'
record = json.loads((case / 'record.json').read_text())
assert record['completed'] and record['healthy'] and not record['active']
assert len(record['requests']) == 4 and record['release_pairs'] == {'release': 6, 'restore': 6}
assert all(all(r['equality'].values()) for r in record['requests'])
assert not record['expert_wait_batches'] and record['phase_sync_mark_count']>0
assert not record['new_fault_messages'] and not any(record['cleanup'].values())
out.mkdir()

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def copy(source, target):
    target = out / target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return str(target.relative_to(out))

def kernel_subset(source, target):
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.startswith('{')]
    target = out / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        'scope': 'Relevant B570/xe rows; complete host journal remains private',
        'private_source_sha256': digest(source), 'private_source_bytes': source.stat().st_size,
        'private_source_rows': len(rows),
        'selected_rows': [r for r in rows if '0000:05:00.0' in r.get('MESSAGE', '')
                          or re.search(r'\bxe\b', r.get('MESSAGE', ''))]}, indent=2) + '\n')

source = case / 'debugger/inferior.stderr'
h = hashlib.sha256()
entries = returns = lines = 0
errors = Counter()
progress = []
phase_counts=Counter()
phase_last=deque(maxlen=32)
with source.open('rb') as stream:
    for line in stream:
        h.update(line)
        lines += 1
        entries += bool(re.search(rb'\[trace\] ze\w+\(', line))
        returns += b'SUCCESS (ZE_RESULT_SUCCESS) in ' in line
        match = re.search(rb'ERROR \((\d+)\) in (ze\w+)', line)
        if match:
            errors[(int(match[1]), match[2].decode())] += 1
        if line.startswith(b'strata prefill sync:'):
            text=line.decode(errors='replace').rstrip()
            phase_last.append(text)
            match=re.search(r'phase (.*?) done',text)
            assert match
            phase_counts[match[1]]+=1
        elif line.startswith(b'strata '):
            progress.append(line.decode(errors='replace').rstrip())
log_tail = out / 'owned-workspace-reclaim-short/api-log.tail.txt'
log_tail.parent.mkdir(parents=True, exist_ok=True)
with source.open('rb') as stream, log_tail.open('wb') as tail:
    stream.seek(max(0, source.stat().st_size - 65536))
    tail.write(stream.read())
(log_tail.parent / 'api-log.json').write_text(json.dumps({
    'scope': 'Full flushed API log remains private; digest, counts, progress and final 64 KiB archived',
    'private_path': str(source), 'bytes': source.stat().st_size,
    'sha256': h.hexdigest(), 'lines': lines, 'api_entry_lines': entries,
    'successful_result_lines': returns,
    'other_results': [{'result': k[0], 'function': k[1], 'count': v}
                      for k, v in sorted(errors.items())],
    'strata_progress_lines': progress,
    'phase_sync_mark_count':sum(phase_counts.values()),
    'phase_sync_phase_counts':dict(phase_counts),'phase_sync_last_marks':list(phase_last),
    'tail': str(log_tail.relative_to(out))}, indent=2) + '\n')

for path in case.rglob('*'):
    if not path.is_file() or path.suffix == '.bin':
        continue
    relative = path.relative_to(case)
    if relative == Path('debugger/inferior.stderr') or path.name == 'kernel.json':
        continue
    if path.name in ('kernel-after.stdout', 'kernel-gap.stdout'):
        kernel_subset(path, Path('owned-workspace-reclaim-short') / (path.stem + '-subset.json'))
    else:
        copy(path, Path('owned-workspace-reclaim-short') / relative)
for suffix in ['stdout', 'stderr']:
    copy(base / ('owned-workspace-reclaim-short.' + suffix),
         Path('owned-workspace-reclaim-short') / ('controller.' + suffix))

sources = []
for raw, expected in record['source_sha256'].items():
    source = Path(raw)
    assert digest(source) == expected, str(source)
    name = source.relative_to(root) if source.is_relative_to(root) else Path(source.name)
    target = copy(source, Path('sources-used') / name)
    sources.append({'private_source': raw, 'archive': target, 'sha256': expected})
for raw in ['sycl/tools/debug-run.py', 'sycl/src/program/generate.cpp']:
    source = root / raw
    target = copy(source, Path('sources-used') / raw)
    sources.append({'private_source': str(source), 'archive': target, 'sha256': digest(source)})
for source in [base / name for name in ['build_prefill_workspace_reclaim.py', 'run_vram_restore_probe.py', 'vram-restore-probe.cpp', 'run_full_workspace_reclaim_serve.py']]:
    target = copy(source, Path('sources-used') / source.name)
    sources.append({'private_source': str(source), 'archive': target, 'sha256': digest(source)})
target = copy(Path(__file__), Path('sources-used') / Path(__file__).name)
sources.append({'private_source': str(Path(__file__)), 'archive': target, 'sha256': digest(Path(__file__))})
(out / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')

for build in ['prefill-workspace-reclaim-prepared', 'prefill-workspace-reclaim-build', 'vram-restore-probe-build']:
    source = base / build
    if source.exists():
        for path in source.rglob('*'):
            if path.is_file() and path.suffix in ['.json', '.stdout', '.stderr', '.txt']:
                copy(path, Path('build-receipts') / build / path.relative_to(source))

print(out)


for name in ['vram-restore-probe-default-single','vram-restore-probe-default-split']:
    rec=json.loads((base/name/'record.json').read_text())
    assert rec['passed'] and not rec['active'] and rec['result']['restored']
    assert not any(rec['cleanup'].values()) and not rec['new_fault_messages']
    for source in (base/name).rglob('*'):
        if not source.is_file():continue
        relative=source.relative_to(base/name)
        if source.name=='kernel-after.stdout':
            kernel_subset(source,Path(name)/relative.with_name('kernel-after-subset.json'))
        else:
            assert source.stat().st_size<8*1024**2
            copy(source,Path(name)/relative)
copy(base/'allocator-source-audit-20261007/record.json',Path('allocator-source-audit/record.json'))
copy(base/'prefill-workspace-reclaim-prepared/prefill.original.cpp',Path('sources-used/prefill.original.cpp'))
copy(base/'prefill-workspace-reclaim-prepared/prefill.candidate.cpp',Path('sources-used/prefill.candidate.cpp'))
(out/'scope.json').write_text(json.dumps({
 'scope':'32 MiB owned-prefill payload reduction and exact short-model parity; two allocation-pressure probes did not reproduce full-model restoration failure; no full-capacity or throughput claim',
 'candidate_sha256':record['binary_sha256'],
 'accounted_owned_bytes_control_T32':173394432,
 'accounted_owned_bytes_candidate_T32':139840000,
 'accounted_reduction_bytes':33554432,
 'shared_workspace_bytes_both_T32':29428736,
 'private_full_log_bytes':source.stat().st_size if False else (case/'debugger/inferior.stderr').stat().st_size,
 'full_context_candidate_status':'Pending independent controller run; not covered by short parity',
 'untested_pressure_settings':['driver-cache-off','ur-pool-off']},indent=2)+'\n')
