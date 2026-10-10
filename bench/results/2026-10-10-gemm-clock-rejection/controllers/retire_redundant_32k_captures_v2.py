"""Retire only closed successful candidate captures already compared in Git."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess
B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
OUT=W/'bench/results/2026-10-10-gemm-clock-rejection/redundant-capture-retirement.json'
def ident(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h,allocated_bytes=p.stat().st_blocks*512)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not OUT.exists()
 head=subprocess.run(['git','rev-parse','HEAD'],cwd=W,capture_output=True,text=True,check=True).stdout.strip()
 assert subprocess.run(['git','status','--porcelain'],cwd=W,capture_output=True,text=True,check=True).stdout==''
 subset=B/'upstream-v0141-tuned-fourfresh-numerical-subset-20261009-v1.json';ref=json.loads(subset.read_text())['requests'][0]
 canonical=[Path(ref[k]['file']) for k in ['first_head','prefill_state']];assert all(p.is_file() for p in canonical)
 fixture_identity=ident(subset);rows=[]
 for name,arch in [('owned-prefill-route-census-v3-code32k-diagnostic-r1','2026-10-10-actual32k-route-capacity'),('owned-prefill-route-census-v3-off-code32k-parity-r1','2026-10-10-actual32k-route-capacity'),('owned-prefill-gemm-only-code32k-diagnostic-r1','2026-10-10-gemm-clock-rejection')]:
  P=B/name;rp=P/'record.json';r=json.loads(rp.read_text());assert not r['active'] and r['math_gate_passed'] and r['exit_code']==0 and not r['new_fault_messages'] and not any(r['cleanup'].values())
  for key in ['inferior','debugger']:
   own=r[key]
   try:s=Path(f"/proc/{own['pid']}/stat").read_text();t=int(s[s.rfind(')')+2:].split()[19])
   except FileNotFoundError:continue
   assert t!=own['start_ticks'], 'owned process still present'
  archived=(W/'bench/results'/arch/('model/record.json' if 'gemm-only' in name else 'models/'+name+'/record.json'))
  assert ident(rp)['sha256']==ident(archived)['sha256']
  req=r['requests'][0];assert req['comparison']['first_head_equal'] and req['comparison']['all66_live_state_parts_equal']
  for field in ['first_head','prefill_state']:
   p=Path(req[field]['file']);assert p.parent==P and p not in canonical
   i=ident(p)
   if field=='first_head':assert i['sha256']==req[field]['sha256']
   else:
    # Recheck every serialized part hash and length preserved in the comparison receipt.
    import struct
    with p.open('rb') as f:
     for part in req[field]['parts']:
      assert struct.unpack('=Q',f.read(8))[0]==part['bytes'] and f.tell()==part['offset']
      h=hashlib.sha256();left=part['bytes']
      while left:
       data=f.read(min(left,1<<20));assert data;left-=len(data);h.update(data)
      assert h.hexdigest()==part['sha256']
     assert not f.read(1)
   rows.append(dict(original=i,retained_comparison=str(archived.relative_to(W)),retained_record_sha256=ident(archived)['sha256'],reason='Duplicate successful candidate capture: all66 live states and head match canonical; future fixed12K and span qualification use canonical references/census, not this capture. GEMM timestamp failure has no numerical divergence.',named_consumers=[]))
 # Source identity of unused links is durable. Current v4 failure reproducer and
 # latest GEMM-only candidate/library stay; remove only never-executed dead outputs.
 for name in ['prefill-route-census-linked-root-v3']:
  P=B/name;r=json.loads((P/'record.json').read_text());assert not r['active'] and not r['model_executed']
  archived=W/'bench/results/2026-10-10-actual32k-route-capacity/receipts'/name/'record.json';assert ident(P/'record.json')['sha256']==ident(archived)['sha256']
  p=P/'strata';i=ident(p)
  if 'binary_identity' in r:assert i['sha256']==r['binary_identity']['sha256']
  rows.append(dict(original=i,retained_comparison=str(archived.relative_to(W)),reason='Unused partial FSIZE-failed or superseded v2-header link; no GPU/model execution and no named current consumer',named_consumers=[]))
 manifest=dict(prior_plan_outcome={'controller':'retire_redundant_32k_captures_v1.py','passed':False,'reason':'Unused FSIZE-failed v2 strata was already absent at identity precheck; no unlink performed by v1','missing_path':str(B/'prefill-route-census-linked-root-v2/strata'),'deletion_count':0},created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),durable_evidence_parent_commit=head,canonical_reference_subset=fixture_identity,canonical_files_preserved=list(map(str,canonical)),current_controller_consumers=[str(B/'run_owned_fixed12k_chunk_major_code32k_v1.py'),str(B/'qualify-observed-gemm-span-v1/root_recipe.txt')],live_GPU_job=False,deletions=rows,logical_bytes_removed=sum(x['original']['bytes'] for x in rows),allocated_file_bytes_removed=sum(x['original']['allocated_bytes'] for x in rows),physical_free_space_gain_claimed=False,reason='Retain compact complete numerical comparisons, original failure/normal statuses, current reference/census and unresolved failure diagnostics; ZFS snapshots can retain extents.')
 # Verify every identity again immediately before unlink. No original receipt is edited.
 for row in rows:
  i=row['original'];assert ident(i['path'])==i;Path(i['path']).unlink()
 OUT.write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({k:manifest[k] for k in ['logical_bytes_removed','allocated_file_bytes_removed','physical_free_space_gain_claimed']}),flush=True)
