"""Archive the real short model/debugger proof without its 1 GiB API log."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import shutil

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
case = base / 'owned-expert-wait-1-short'
out = root / 'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/expert-wait-one-20261007'
record = json.loads((case / 'record.json').read_text())
assert record['completed'] and record['healthy'] and not record['active']
assert len(record['requests']) == 4 and record['release_pairs'] == {'release': 6, 'restore': 6}
assert all(all(r['equality'].values()) for r in record['requests'])
assert len(record['expert_wait_batches']) == 16677 and all(v[0] == 1 for v in record['expert_wait_batches'])
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
with source.open('rb') as stream:
    for line in stream:
        h.update(line)
        lines += 1
        entries += bool(re.search(rb'\[trace\] ze\w+\(', line))
        returns += b'SUCCESS (ZE_RESULT_SUCCESS) in ' in line
        match = re.search(rb'ERROR \((\d+)\) in (ze\w+)', line)
        if match:
            errors[(int(match[1]), match[2].decode())] += 1
        if line.startswith(b'strata '):
            progress.append(line.decode(errors='replace').rstrip())
log_tail = out / 'owned-expert-wait-1-short/api-log.tail.txt'
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
    'tail': str(log_tail.relative_to(out))}, indent=2) + '\n')

for path in case.rglob('*'):
    if not path.is_file() or path.suffix == '.bin':
        continue
    relative = path.relative_to(case)
    if relative == Path('debugger/inferior.stderr') or path.name == 'kernel.json':
        continue
    if path.name in ('kernel-after.stdout', 'kernel-gap.stdout'):
        kernel_subset(path, Path('owned-expert-wait-1-short') / (path.stem + '-subset.json'))
    else:
        copy(path, Path('owned-expert-wait-1-short') / relative)
for suffix in ['stdout', 'stderr']:
    copy(base / ('owned-expert-wait-1-short.' + suffix),
         Path('owned-expert-wait-1-short') / ('controller.' + suffix))

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
for source in [base / 'build_prefill_expert_wait.py', base / 'run_full_expert_wait_one_serve.py']:
    target = copy(source, Path('sources-used') / source.name)
    sources.append({'private_source': str(source), 'archive': target, 'sha256': digest(source)})
target = copy(Path(__file__), Path('sources-used') / Path(__file__).name)
sources.append({'private_source': str(Path(__file__)), 'archive': target, 'sha256': digest(Path(__file__))})
(out / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')

for build in ['prefill-expert-wait-build']:
    source = base / build
    if source.exists():
        for path in source.rglob('*'):
            if path.is_file() and path.suffix in ['.json', '.stdout', '.stderr', '.txt']:
                copy(path, Path('build-receipts') / build / path.relative_to(source))

print(out)
