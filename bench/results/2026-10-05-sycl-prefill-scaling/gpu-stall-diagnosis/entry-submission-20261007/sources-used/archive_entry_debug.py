"""Archive bounded diagnostic receipts; retain large API logs privately."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import shutil

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = root/'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/entry-submission-20261007'
out.mkdir()

def copy(source, target):
    target = out/target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)

def subset(source, target):
    raw = source.read_bytes()
    rows = [json.loads(x) for x in raw.splitlines() if x]
    selected = [x for x in rows if '0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))]
    target = out/target
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({'scope':'Relevant xe/B570 rows only; full kernel journal remains private',
        'private_source_bytes':len(raw),'private_source_sha256':hashlib.sha256(raw).hexdigest(),
        'private_source_rows':len(rows),'selected_rows':selected},indent=2)+'\n')

def api_log(source, target):
    digest = hashlib.sha256(); errors = Counter(); lines = entries = 0; strata = []
    with source.open('rb') as f:
        for line in f:
            digest.update(line); lines += 1
            entries += bool(re.search(rb'\[trace\] ze\w+\(',line))
            match = re.search(rb'ERROR \((\d+)\) in (ze\w+)',line)
            if match: errors[(int(match[1]),match[2].decode())] += 1
            if line.startswith(b'strata '): strata.append(line.decode(errors='replace').rstrip())
    target = out/target; target.parent.mkdir(parents=True,exist_ok=True)
    tail = target.with_suffix('.tail.txt')
    with source.open('rb') as f:
        f.seek(max(0,source.stat().st_size-65536)); tail.write_bytes(f.read())
    target.write_text(json.dumps({'scope':'Full flushed API log remains private; digest, counts and last 64 KiB archived',
        'private_path':str(source),'bytes':source.stat().st_size,'sha256':digest.hexdigest(),
        'lines':lines,'api_entry_lines':entries,'tail':str(tail.relative_to(out)),
        'result_counts':[{'result':k[0],'function':k[1],'count':v} for k,v in sorted(errors.items())],
        'strata_progress_lines':strata},indent=2)+'\n')

def model_case(name):
    source = base/name; d = json.loads((source/'record.json').read_text())
    assert not d['active'] and not d['cleanup']['inferior_survived'] and not d['cleanup']['gdb_survived']
    for n in ['record.json','metrics.jsonl','read_metrics.py']:
        copy(source/n,Path(name)/n)
    for p in (source/'debugger').iterdir():
        if p.name == 'engine-and-gdb.stderr':
            api_log(p,Path(name)/'api-log.json')
        elif p.is_file():
            copy(p,Path(name)/'debugger'/p.name)
    for p in (source/'probes').iterdir():
        if p.name == 'kernel-after.stdout':
            subset(p,Path(name)/'kernel-subset.json')
        elif p.name.startswith(('kernel-before.','kernel-after.stderr','embedding-state.')):
            copy(p,Path(name)/'probes'/p.name)
    for suffix in ['stdout','stderr']:
        p = base/(name+'.'+suffix)
        if p.exists():copy(p,Path(name)/('controller.'+suffix))

for name in ['full-context-entry-capacity','full-context-entry-csr']:
    model_case(name)
for name in ['api-entry-profile-health','post-entry-stall-health','post-entry-csr-health','owned-gdb-pty-cpu','owned-gdb-pty-cpu-fixed-path','owned-gdb-with-tty-cli-cpu']:
    source = base/name
    for p in source.rglob('*'):
        if not p.is_file() or p.suffix not in ['.json','.jsonl','.txt','.stdout','.stderr','.raw','.py','.c']:
            continue
        if 'kernel' in p.name and p.suffix == '.stdout':
            raw = p.read_bytes()
            if raw.lstrip().startswith(b'{') or not raw.strip():
                subset(p,Path(name)/(p.stem+'-subset.json'))
                continue
        copy(p,Path(name)/p.relative_to(source))
    for suffix in ['stdout','stderr']:
        p = base/(name+'.'+suffix)
        if p.exists(): copy(p,Path(name)/('controller.'+suffix))

for name in ['run_full_entry_capacity.py','run_full_entry_csr.py','read-csr.gdb','prepare_entry_csr_debug.py','probe_after_entry_stall.py','probe_after_entry_csr.py','owned_gdb-before-tty.py']:
    copy(base/name,Path('sources-used')/name)
for name in ['owned_gdb.py','test_owned_gdb.py','test_owned_gdb_protocol.py','recover-xe.sh']:
    copy(root/'sycl/tools'/name,Path('sources-used')/name)
for name in ['receipts.json','csr-receipts.json','installed-module-receipt.json','xe-exec-ioctl-installed.disassembly.txt']:
    copy(base/'submit-stall-sources'/name,Path('upstream-source-receipts')/name)
copy(Path(__file__),Path('sources-used')/Path(__file__).name)
print(out)
