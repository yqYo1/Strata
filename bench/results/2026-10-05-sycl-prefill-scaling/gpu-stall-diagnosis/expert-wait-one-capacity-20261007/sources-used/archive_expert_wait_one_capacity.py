"""Archive the actual MTP submission stall and reset-free healthy follow-up."""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = root / 'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/expert-wait-one-capacity-20261007'
record = json.loads((base / 'full-context-expert-wait-1-serve/record.json').read_text())
assert not record['active'] and not record['completed']
assert not record['cleanup']['inferior_survived'] and not record['cleanup']['gdb_survived']
assert not record['new_fault_messages']
health = json.loads((base / 'post-expert-wait-one-stall-health/record.json').read_text())
assert health['healthy'] and health['boot_id'] == record['boot_id']
out.mkdir()
original = ast.parse((base / 'archive_full_cli_capacity.py').read_text())
functions = ast.Module(body=[n for n in original.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
exec(compile(functions, str(base / 'archive_full_cli_capacity.py'), 'exec'))

case = base / 'full-context-expert-wait-1-serve'
for name in ['record.json', 'metrics.jsonl', 'events.jsonl', 'protocol.stdout.raw']:
    copy(case / name, Path('full-context-expert-wait-1-serve') / name)
for path in (case / 'debugger').iterdir():
    if path.is_file():
        if path.name == 'inferior.stderr':
            target = out / 'full-context-expert-wait-1-serve/debugger'
            target.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256(); lines = waits = 0; by_layer = {}; chunks = []; other = []; last_wait = None
            with path.open('rb') as stream:
                for line in stream:
                    h.update(line); lines += 1
                    text = line.decode(errors='replace').rstrip()
                    match = re.match(r'strata prefill expert wait: (\d+) experts completed, layer (-?\d+), chunk (-?\d+)', text)
                    if match:
                        values = list(map(int, match.groups())); assert values[0] == 1
                        waits += 1; by_layer[str(values[1])] = by_layer.get(str(values[1]), 0) + 1; last_wait = values
                    elif text.startswith('strata trace: prompt chunk '): chunks.append(text)
                    elif text.startswith('strata '): other.append(text)
            with path.open('rb') as stream, (target / 'log.tail.txt').open('wb') as dest:
                stream.seek(max(0, path.stat().st_size - 65536)); dest.write(stream.read())
            (target / 'log.json').write_text(json.dumps({'scope':'Complete warning/validation/progress log stays private; digest, counts, chunk starts, other Strata progress and final 64 KiB preserved',
                'private_path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest(),'lines':lines,
                'one_expert_wait_lines':waits,'waits_by_layer':by_layer,'last_wait':last_wait,
                'chunk_start_lines':chunks,'other_strata_progress':other},indent=2)+'\n')
        else:
            copy(path, Path('full-context-expert-wait-1-serve/debugger') / path.name)
for path in (case / 'probes').iterdir():
    if path.name in ['kernel-after.stdout', 'kernel-gap.stdout']:
        kernel_subset(path, Path('full-context-expert-wait-1-serve') / (path.stem + '-subset.json'))
    else:
        copy(path, Path('full-context-expert-wait-1-serve/probes') / path.name)

case = base / 'post-expert-wait-one-stall-health'
for path in case.iterdir():
    if path.is_file():
        if path.name == 'health-kernel.stdout':
            raw = path.read_bytes()
            selected = [line.decode(errors='replace') for line in raw.splitlines()
                        if b'0000:05:00.0' in line or re.search(rb'\bxe\b', line)]
            target = out / 'post-expert-wait-one-stall-health/health-kernel-subset.json'
            target.parent.mkdir(exist_ok=True)
            target.write_text(json.dumps({'scope': 'Relevant xe/B570 rows; full host journal stays private',
                'private_source_sha256': digest(path), 'private_source_bytes': len(raw),
                'selected_rows': selected}, indent=2) + '\n')
        else:
            copy(path, Path('post-expert-wait-one-stall-health') / path.name)
for name in ['full-context-expert-wait-1-serve', 'post-expert-wait-one-stall-health']:
    for suffix in ['stdout', 'stderr']:
        copy(base / (name + '.' + suffix), Path(name) / ('controller.' + suffix))
sources = []
for raw, expected in record['source_sha256'].items():
    source = Path(raw)
    assert digest(source) == expected, raw
    name = source.relative_to(root) if source.is_relative_to(root) else Path(source.name)
    target = copy(source, Path('sources-used') / name)
    sources.append({'private_source': raw, 'archive': target, 'sha256': expected})
for source in [base / 'probe_after_expert_wait_one_stall.py', base / 'archive_full_cli_capacity.py', Path(__file__)]:
    target = copy(source, Path('sources-used') / source.name)
    sources.append({'private_source': str(source), 'archive': target, 'sha256': digest(source)})
(out / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
print(out)
