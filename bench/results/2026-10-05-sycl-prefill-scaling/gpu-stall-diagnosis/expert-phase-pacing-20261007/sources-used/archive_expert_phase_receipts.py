"""Archive bounded, hash-checked private phase-pacing correctness and failed clean-repeat receipts."""
from pathlib import Path
import hashlib
import json
import re
import shutil

b=Path(__file__).parent
r=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=r/'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/expert-phase-pacing-20261007'
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


for name in ['build_expert_phase_pacing.py','run_owned_expert_phase_pacing_16_short.py','run_expert_phase_pacing_2048_quiet.py','run_expert_phase_baseline_repeat_debug.py','run_expert_phase_baseline_repeat_sync.py','probe_after_baseline_repeat_debug.py','profile_supervisor.py','archive_expert_phase_receipts.py']:
    copy(b/name,Path('sources-used')/name)
copy(b/'expert-phase-pacing-build/prefill.cpp',Path('sources-used/prefill.candidate.cpp'))
for name in ['expert-phase-pacing-prepared','expert-phase-pacing-build','owned-expert-phase-pacing-16-short','expert-phase-pacing-2048-quiet-b1','expert-phase-baseline-repeat-debug','post-baseline-repeat-debug-health','expert-phase-baseline-repeat-sync']:
    folder=b/name
    if (folder/'record.json').exists():
        record=json.loads((folder/'record.json').read_text());assert not record.get('active')
    for p in folder.rglob('*'):
        if not p.is_file() or p.suffix in ['.o','.a','.d','.bin'] or '__pycache__' in p.parts:continue
        target=Path(name)/p.relative_to(folder)
        if p.stat().st_size<=300000:copy(p,target)
        else:bounded(p,target.with_name(p.name+'.tail'))
failed=json.loads((b/'expert-phase-pacing-2048-quiet-b1/record.json').read_text())
assert not failed['passed'] and not failed['job']['survivors'] and not failed['new_fault_messages']
for item in failed['job']['owned']:assert not Path('/proc',str(item['pid'])).exists()
summary=dict(scope='Actual expert-phase pacing 16 short PASS, quiet batch1 repeat mismatch, original-control API-logged stall and phase-log repeat PASS; candidate not adopted',
 short=json.loads((b/'owned-expert-phase-pacing-16-short/record.json').read_text())['expert_phase_wait_schedule'],
 quiet_batch1=[dict(name=x['name'],done=x['done'],equality=x['equality'],head_sha256=x['head_sha256']) for x in failed['requests']],
 original_repeat=json.loads((b/'expert-phase-baseline-repeat-sync/record.json').read_text())['passed'],
 logged_original_stall_exit=json.loads((b/'expert-phase-baseline-repeat-debug/record.json').read_text())['job']['exit_code'],
 post_stall_health=json.loads((b/'post-baseline-repeat-debug-health/record.json').read_text())['healthy'],
 adopted=False,phase_batch16_quiet_2k_run=False,full_capacity_verified=False)
(out/'assessment.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(archive=str(out),files=len([p for p in out.rglob('*') if p.is_file()])),indent=2))
