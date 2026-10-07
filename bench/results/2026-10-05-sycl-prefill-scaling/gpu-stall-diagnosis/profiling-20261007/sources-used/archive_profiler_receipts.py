"""Archive bounded, hash-checked profiler receipts and the completed 256K failure."""
from pathlib import Path
import hashlib
import json
import re
import shutil

b=Path(__file__).parent
r=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=r/'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/profiling-20261007'
out.mkdir(mode=0o755)

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def copy(p,target):
    q=out/target;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    return str(q.relative_to(out))

def bounded(p,target):
    q=out/target;q.parent.mkdir(parents=True,exist_ok=True)
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-65536));q.write_bytes(f.read())
    metadata=dict(private_path=str(p),private_bytes=p.stat().st_size,private_sha256=digest(p),
                  saved_tail=str(q.relative_to(out)),saved_tail_bytes=q.stat().st_size)
    (q.parent/(q.name+'.metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata

sources=[]
for name in ['profile_supervisor.py','vtune_smoke.py','build_unitrace.py',
             'build_unitrace_system_cc.py','unitrace_smoke.py',
             'run_unitrace_model_short.py','run_unitrace_model_2048.py',
             'run_unitrace_model_2048_suffix.py','analyze_unitrace_short.py',
             'analyze_unitrace_2048.py','probe_after_workspace_full_failure.py',
             'run_full_workspace_reclaim_serve.py','archive_profiler_receipts.py']:
    p=b/name;sources.append(dict(private_path=str(p),archive=copy(p,Path('sources-used')/name),sha256=digest(p)))
for name in ['sycl/src/prefill/prefill.cpp','sycl/src/program/generate.cpp',
             'sycl/tools/owned_gdb.py','sycl/tools/recover-xe.sh','sycl/tools/debug-run.py']:
    p=r/name;sources.append(dict(private_path=str(p),archive=copy(p,Path('sources-used')/name),sha256=digest(p)))
(out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')

for name in ['vtune-preflight','vtune-smoke','unitrace-build','unitrace-build-system-cc',
             'unitrace-smoke','unitrace-model-short-profile','unitrace-model-short-clean',
             'unitrace-model-2048-clean','unitrace-model-2048-suffix-clean',
             'unitrace-model-2048-suffix-profile','post-workspace-full-health']:
    folder=b/name
    if not folder.exists():continue
    for p in folder.rglob('*'):
        if not p.is_file():continue
        local=p.relative_to(folder)
        if any(x in local.parts for x in ['build','__pycache__']):continue
        if p.suffix in ['.bin','.so']:continue
        if p.name.startswith('strata-') and p.suffix=='.json':
            metadata=dict(private_path=str(p),bytes=p.stat().st_size,sha256=digest(p),
                          retained='Full private Chrome/Perfetto-compatible timeline; not copied into git')
            target=out/name/'timeline.json';target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(metadata,indent=2)+'\n');continue
        target=Path(name)/local
        if p.stat().st_size<=300000:copy(p,target)
        else:bounded(p,target.with_name(local.name+'.tail'))
for name in ['unitrace-short-analysis.json','unitrace-2048-analysis.json']:copy(b/name,Path(name))

full=b/'full-context-workspace-reclaim-serve'
record=json.loads((full/'record.json').read_text())
assert not record['active'] and not record['completed'] and not record['new_fault_messages']
assert not record['cleanup']['inferior_survived'] and not record['cleanup']['gdb_survived']
for key in ['inferior','debugger']:assert not Path('/proc',str(record[key]['pid'])).exists()
copy(full/'record.json',Path('full-context-workspace-reclaim-serve/record.json'))
for name in ['protocol.stdout.raw','debugger/failure.mi.txt','debugger/failure.csr.mi.txt',
             'debugger/inferior-argv.json','debugger/engine-and-gdb.stderr']:
    p=full/name
    if p.stat().st_size<=300000:copy(p,Path('full-context-workspace-reclaim-serve')/name)
    else:bounded(p,Path('full-context-workspace-reclaim-serve')/(name+'.tail'))
meta=bounded(full/'debugger/inferior.stderr',Path('full-context-workspace-reclaim-serve/inferior.stderr.tail'))
tail=(out/meta['saved_tail']).read_text(errors='replace')
memory=re.findall(r'strata prefill memory: ([^;\n]+); free=(\d+) total=(\d+) bytes',tail)
marks=re.findall(r'strata prefill sync: mark (\d+) phase ([^\n]+)',tail)
assessment=dict(scope='Completed 256K diagnostic failure after full prefill, before any generated token or capacity case; not GPU hang or throughput',
    last_sync_mark=int(marks[-1][0]),memory_tail=[dict(phase=phase,free_bytes=int(f),total_bytes=int(t)) for phase,f,t in memory],
    log=meta,baseline_source_guard='Original c7ba0822 source guard is already frozen in the workspace-reclaim archive; actual executable source is bdc0dea4 in the candidate build',
    failed_segment_bytes=8388608,elapsed_seconds=record['elapsed_seconds'],
    no_new_xe_faults=True,owned_processes_gone=True,
    post_cleanup_health=json.loads((b/'post-workspace-full-health/record.json').read_text())['healthy'])
(out/'full-context-result.json').write_text(json.dumps(assessment,indent=2)+'\n')

header_build=b/'unitrace-build-system-cc/build'
provenance=dict(repository='https://github.com/intel/pti-gpu',commit='6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033',
    level_zero_headers_commit='d3b3efb',level_zero_headers_tag='v1.32.0',
    compute_runtime_headers_commit='050536ff3b6f830199ec99cff8ffcac3d804c40a',
    note='Headers are private build dependencies. No loader, driver or system package replaced.',
    header_sha256={str(p.relative_to(header_build)):digest(p) for p in (header_build/'level_zero').rglob('*.h') if '.git' not in p.parts})
(out/'profiler-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

print(json.dumps(dict(archive=str(out),files=len(list(out.rglob('*'))),full_context=assessment),indent=2))
