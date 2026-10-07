"""Archive completed full-prefill diagnosis without its 1.48 GB phase log."""
from pathlib import Path
from collections import Counter, deque
import ast
import hashlib
import json
import re
import shutil

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=root/'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/phase-sync-capacity-20261007'
case=base/'full-context-phase-sync-serve'
record=json.loads((case/'record.json').read_text())
health=json.loads((base/'post-phase-full-failure-health/record.json').read_text())
assert not record['active'] and not record['completed'] and not record['requests']
assert not record['new_fault_messages'] and not record['cleanup']['inferior_survived'] and not record['cleanup']['gdb_survived']
assert health['healthy'] and health['boot_id']==record['boot_id']
assert len(record['snapshots'])==1 and 'SIGABRT' in record['snapshots'][0]['stop']
out.mkdir()
original=ast.parse((base/'archive_full_cli_capacity.py').read_text())
functions=ast.Module(body=[n for n in original.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
exec(compile(functions,str(base/'archive_full_cli_capacity.py'),'exec'))

log=case/'debugger/inferior.stderr'
sha=hashlib.sha256();lines=0;phases=Counter();last=deque(maxlen=32);ranges=Counter();last_chunk=None
with log.open('rb') as stream,(out/'strata-progress.txt').open('w') as progress:
    for row in stream:
        sha.update(row);lines+=1
        if row.startswith(b'strata prefill sync:'):
            text=row.decode(errors='replace').rstrip();last.append(text)
            match=re.search(r'phase (.*?) done',text);assert match
            phases[match[1]]+=1
        else:
            if row.startswith(b'strata ') or b'cannot restore decode' in row or b'terminate called' in row:
                progress.write(row.decode(errors='replace'))
            match=re.search(rb'strata trace: prompt chunk (\d+) of (\d+), layers \[(\d+), (\d+)\)',row)
            if match:
                last_chunk=list(map(int,match.groups()));ranges[(last_chunk[2],last_chunk[3])]+=1
assert sorted(ranges)==[(i,i+1) for i in range(48)] and last_chunk==[261120,262139,47,48]
with log.open('rb') as stream,(out/'phase-log.tail.txt').open('wb') as tail:
    stream.seek(max(0,log.stat().st_size-65536));tail.write(stream.read())
(out/'phase-log.json').write_text(json.dumps({
    'scope':'Private complete phase log; digest/counts, non-phase progress and bounded tail archived. Printed phase names refer to the next phase.',
    'private_path':str(log),'bytes':log.stat().st_size,'sha256':sha.hexdigest(),'lines':lines,
    'phase_mark_count':sum(phases.values()),'phase_counts':dict(phases),'last_marks':list(last),
    'layer_chunk_counts':[{'range':list(k),'chunks':v} for k,v in sorted(ranges.items())],
    'last_chunk':last_chunk,'whole_prefill_completed_before_restore':True,
    'generation_completed':False,'tail':'phase-log.tail.txt','progress':'strata-progress.txt'},indent=2)+'\n')
for path in case.rglob('*'):
    if not path.is_file() or path==log or path.suffix=='.bin':continue
    relative=path.relative_to(case)
    if path.name in ('kernel-after.stdout','kernel-gap.stdout'):
        kernel_subset(path,Path(case.name)/(path.stem+'-subset.json'))
    else:copy(path,Path(case.name)/relative)
case=base/'post-phase-full-failure-health'
for path in case.iterdir():
    if not path.is_file():continue
    if path.name=='health-kernel.stdout':
        raw=path.read_bytes()
        target=out/case.name/'health-kernel-subset.json';target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps({'scope':'Relevant B570/xe rows; whole journal remains private',
            'private_source_sha256':digest(path),'private_source_bytes':len(raw),
            'selected_rows':[v.decode(errors='replace') for v in raw.splitlines() if b'0000:05:00.0' in v or re.search(rb'\bxe\b',v)]},indent=2)+'\n')
    else:copy(path,Path(case.name)/path.name)
for name in ('full-context-phase-sync-serve','post-phase-full-failure-health'):
    for suffix in ('stdout','stderr'):
        path=base/(name+'.'+suffix)
        if path.exists():copy(path,Path(name)/('controller.'+suffix))
sources=[]
for raw,expected in record['source_sha256'].items():
    source=Path(raw);assert digest(source)==expected,raw
    name=source.relative_to(root) if source.is_relative_to(root) else Path(source.name)
    target=copy(source,Path('sources-used')/name)
    sources.append({'private_source':raw,'archive':target,'sha256':expected})
for source in (base/'probe_after_phase_full_failure.py',base/'archive_full_cli_capacity.py',Path(__file__),root/'sycl/src/core/mtp.cpp',root/'sycl/src/core/expert_cache.cpp'):
    target=copy(source,Path('sources-used')/(source.relative_to(root) if source.is_relative_to(root) else Path(source.name)))
    sources.append({'private_source':str(source),'archive':target,'sha256':digest(source)})
(out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
print(json.dumps({'archive':str(out),'phase_mark_count':sum(phases.values()),'phase_log_sha256':sha.hexdigest(),'last_chunk':last_chunk}))
