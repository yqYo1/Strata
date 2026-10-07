"""Archive bounded, hash-checked private lazy verifier-graph restore build, short check and allocated-capacity comparison."""
from pathlib import Path
import hashlib
import json
import re
import shutil

b=Path(__file__).parent
r=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=r/'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/verifier-lazy-restore-20261007'
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


for name in ['build_verifier_lazy_restore.py','build_verifier_lazy_restore_v2.py','run_owned_verifier_lazy_restore_short.py','run_large_kv_short_verifier_lazy_restore.py','run_large_kv_short_eager_control.py','run_full_verifier_lazy_restore_serve.py','archive_lazy_restore_receipts.py']:
    copy(b/name,Path('sources-used')/name)
copy(b/'verifier-lazy-restore-build-v2/verify.cpp',Path('sources-used/verify.candidate.cpp'))
copy(r/'sycl/src/core/verify.cpp',Path('sources-used/verify.original.cpp'))
import difflib
old=(r/'sycl/src/core/verify.cpp').read_text().splitlines(keepends=True)
new=(b/'verifier-lazy-restore-build-v2/verify.cpp').read_text().splitlines(keepends=True)
(out/'candidate.diff').write_text(''.join(difflib.unified_diff(old,new,fromfile='verify.original.cpp',tofile='verify.candidate.cpp')))
for name in ['verifier-lazy-restore-build','verifier-lazy-restore-build-v2','owned-verifier-lazy-restore-short','large-kv-short-verifier-lazy-restore','large-kv-short-eager-control']:
    folder=b/name
    record=json.loads((folder/'record.json').read_text());assert not record.get('active')
    if 'healthy' in record:assert record['healthy'] and not record['new_fault_messages'] and not any(record['cleanup'].values())
    for p in folder.rglob('*'):
        if not p.is_file() or p.suffix in ['.o','.a','.d','.bin'] or '__pycache__' in p.parts:continue
        target=Path(name)/p.relative_to(folder)
        if p.stat().st_size<=300000:copy(p,target)
        else:bounded(p,target.with_name(p.name+'.tail'))
records=[json.loads((b/name/'record.json').read_text()) for name in ['owned-verifier-lazy-restore-short','large-kv-short-verifier-lazy-restore','large-kv-short-eager-control']]
for record in records:
    for key in ['inferior','debugger']:assert not Path('/proc',str(record[key]['pid'])).exists()
assert all(records[-1]['requests'][0]['same_configuration_comparison'].values())
summary=dict(scope='Private lazy verifier restore; short heads PASS; 262144-cell allocation with 2048 consumed input only, normal-MTP finite outputs exactly match unchanged-binary eager verifier at identical settings; not full occupancy or clean throughput',
    binary_sha256=records[0]['binary_sha256'],
    short_four_heads_exact=all(all(x['equality'].values()) for x in records[0]['requests']),
    short_scope='Original context128 check exercises MTP release; main-cache release flag was absent, so modified main-cache callback is exercised only by the large-KV checks',
    lazy=records[1]['requests'][0],eager=records[2]['requests'][0],
    adopted=False,full_occupancy_completed=False,
    full_occupancy_controller='sources-used/run_full_verifier_lazy_restore_serve.py',
    memory_log_label='Private prefill retains the old graphs rebuilt phase label; candidate callback actually defers capture until Verifier::run()',
    production_sources_unchanged=True)
(out/'assessment.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(archive=str(out),files=len([p for p in out.rglob('*') if p.is_file()])),indent=2))
