"""Preserve real full-cell CLI proof, CPU-only supervision and relink receipts."""
from pathlib import Path
import hashlib
import json
import re
import shutil

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = root / 'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/full-capacity-20261007'
case = base / 'full-context-layer-trace-csr'
record = json.loads((case / 'record.json').read_text())
assert record['healthy'] and record['completed'] and not record['active']
assert record['last_executed_kv_cell'] == 262143
assert record['verify_windows'] == [[262141, 1], [262142, 2]]
assert not record['new_fault_messages'] and not any(record['cleanup'].values())
out.mkdir()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def copy(source, target):
    target = out / target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return str(target.relative_to(out))

def kernel_subset(source, target):
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.startswith('{')]
    target = out / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'scope': 'Relevant xe/B570 rows; full host journal remains private',
        'private_source_sha256': digest(source), 'private_source_bytes': source.stat().st_size,
        'private_source_rows': len(rows), 'selected_rows': [r for r in rows
            if '0000:05:00.0' in r.get('MESSAGE', '') or re.search(r'\bxe\b', r.get('MESSAGE', ''))]}, indent=2) + '\n')

for name in ['record.json', 'metrics.jsonl', 'read_metrics.py']:
    copy(case / name, Path('full-context-layer-trace-csr') / name)
for path in (case / 'debugger').iterdir():
    if path.is_file():
        copy(path, Path('full-context-layer-trace-csr/debugger') / path.name)
for path in (case / 'probes').iterdir():
    if path.name in ['kernel-after.stdout', 'kernel-gap.stdout']:
        kernel_subset(path, Path('full-context-layer-trace-csr') / (path.stem + '-subset.json'))
    elif path.name.startswith(('kernel-before.', 'kernel-after.stderr', 'kernel-gap.stderr', 'embedding-state.')):
        copy(path, Path('full-context-layer-trace-csr/probes') / path.name)
for suffix in ['stdout', 'stderr']:
    copy(base / ('full-context-layer-trace-csr.' + suffix),
         Path('full-context-layer-trace-csr') / ('controller.' + suffix))

sources = []
for raw, expected in record['source_sha256'].items():
    source = Path(raw)
    assert digest(source) == expected, raw
    name = source.relative_to(root) if source.is_relative_to(root) else Path(source.name)
    target = copy(source, Path('sources-used') / name)
    sources.append({'private_source': raw, 'archive': target, 'sha256': expected})
for source in [base / 'run_full_layer_trace_serve.py', base / 'test_full_layer_trace_serve_cpu.py',
               base / 'link_layer_trace_tasks9.py', Path(__file__)]:
    target = copy(source, Path('sources-used') / source.name)
    sources.append({'private_source': str(source), 'archive': target, 'sha256': digest(source)})
(out / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')

for name in ['layer-trace-tasks9-link', 'full-layer-trace-serve-cpu',
             'full-layer-trace-serve-cpu-fixed-fixture', 'full-layer-trace-serve-cpu-final-four']:
    source = base / name
    for path in source.rglob('*'):
        if path.is_file() and path.suffix not in ['.a', '.bin']:
            copy(path, Path(name) / path.relative_to(source))
    for suffix in ['stdout', 'stderr']:
        path = base / (name + '.' + suffix)
        if path.exists():
            copy(path, Path(name) / ('outer.' + suffix))
print(out)
