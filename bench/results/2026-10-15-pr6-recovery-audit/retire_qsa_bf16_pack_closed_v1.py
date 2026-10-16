#!/usr/bin/env python3
"""Retire closed raw logs/duplicate receipts after compact evidence is committed."""
import fcntl,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 dst=A/'retention-qsa-bf16-pack-closed-v1.json';assert not dst.exists()
 ownerpath=P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py';assert sha(ownerpath)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
 spec=importlib.util.spec_from_file_location('pack_cleanup_owner',ownerpath);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 names=['own-qsa-bf16-packed-build-v1.json','own-qsa-bf16-packed-fixture-build-v1.json','own-qsa-bf16-packed-bench-build-v1.json','own-qsa-bf16-packed-bench-build-v2.json','own-qsa-bf16-packed-smoke-v1.json','own-qsa-bf16-packed-component-v1.json','own-qsa-bf16-packed-bench-smoke-base-v1.json','own-qsa-bf16-packed-bench-smoke-packed-v1.json','own-qsa-bf16-packed-diagnostic-build-v1.json','own-qsa-bf16-packed-diagnostic-base-v1.json','own-qsa-bf16-packed-diagnostic-packed-v1.json','own-qsa-bf16-numerical-probe-build-v1.json','own-qsa-bf16-numerical-probe-fixture-build-v1.json','own-qsa-bf16-numerical-probe-base-v1.json','own-qsa-bf16-prefix-build-v1.json','own-qsa-bf16-prefix-build-v2.json','own-qsa-bf16-prefix-fixture-build-v1.json','own-qsa-bf16-prefix-smoke-v1.json','own-qsa-bf16-prefix-component-v1.json','own-qsa-bf16-prefix-engine-build-v1.json','own-qsa-bf16-prefix-engine-smoke-v1.json','own-qsa-bf16-prefix-engine-component-v1.json','own-qsa-bf16-prefix-model-smoke-v1.json']
 names += [f'own-qsa-bf16-packed-bench-{arm}-32768-{n}-v1.json' for arm in ['base','packed'] for n in range(3)]
 rec=dict(schema='strata-closed-pack-retention-v1',deleted=[],logical_bytes=0,allocated_file_bytes=0,receipt_pins={},original_statuses_unchanged=True,current_engine_and_component_binaries_retained=True,scope='Successful raw API/build/journal/protocol logs, duplicate closed root record.json and completed synthetic model tensors. No active run or final fixture. No model/source/service/driver state changes. Failed original numerical status preserved; compact representative retained in Git.',replacements=['qsa-bf16-packed-32k-summary-v1.json','qsa-bf16-256k-numerical-attribution-v1.json','qsa-bf16-packed-256k-numerical-stop-v1.json','qsa-bf16-prefix-boundary-v1.json'],actual_available_space_reclaimed='Not inferred from logical/allocated bytes; filesystem snapshots may retain extents.')
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  paths={}
  for name in names:
   p=A/name;v=json.loads(p.read_text());assert v['complete'] and v['ownership_closed'] and not v['active'];assert not v.get('fresh_faults')
   original=subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(P))],cwd=P);assert hashlib.sha256(original).hexdigest()==sha(p)
   rec['receipt_pins'][str(p)]=sha(p)
   for c in v['commands']:
    assert m.closed(c) and c['exit_code']==0
    for old in c.get('owners',[])+[c['direct_identity']]:
     now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
   # Receipt roots are fixed by the owned logger directory, not string guesses.
   root=None
   for candidate in B.glob('main-*'):
    rp=candidate/'record.json'
    if rp.is_file() and rp.stat().st_size==p.stat().st_size and sha(rp)==sha(p):
     assert root is None;root=candidate
   assert root is not None,name
   for q in (root/'owned').iterdir():
    if q.is_file() and q.suffix in ['.stdout','.stderr']:paths[q]=dict(replacement=str(p),reason='Useful progress/errors, command/env/identity/exit and fault findings already committed; full successful API/build/journal repetition has no remaining consumer.')
   paths[root/'record.json']=dict(replacement=str(p),reason='Byte-identical duplicate closed controller receipt; committed Git report is canonical.')
   if name=='own-qsa-bf16-prefix-model-smoke-v1.json':
    for arm in ['existing','bf16','bf16-prefix']:
     for q in (root/arm).iterdir():
      assert q.is_file();paths[q]=dict(replacement=str(p),reason='All128 NLLs, selected full-vocabulary oracle checks, cross-arm maxima and tensor SHA256 recorded; synthetic mechanics comparison closed, no current RESTORE or numerical-failure consumer.')
    for q in [root/'tokens.txt',root/'expert-profile.bin']:
     if q.is_file():paths[q]=dict(replacement=str(p),reason='Closed synthetic mechanics private input/profile copy; external user profile preserved and hash checked.')
  # Unused first kernel build was superseded before any native launch.
  old=B/'main-qsa-bf16-prefix-build-v1/qsa_bf16_prefix.o'
  if old.is_file():paths[old]=dict(replacement=str(A/'own-qsa-bf16-prefix-build-v1.json'),reason='Unused compiled-only object superseded by zero-query-corrected v2 before first native check; qualified v2 and linked fixtures retained.')
  # Do not remove any file named as a frozen input to the live engine/next run.
  frozen=json.loads((A/'own-qsa-bf16-prefix-engine-build-v1.json').read_text())['frozen_files']
  for q in paths:
   assert str(q) not in frozen and q.is_relative_to(B) and not q.is_symlink(),str(q)
   st=q.stat();rec['deleted'].append(dict(path=str(q),bytes=st.st_size,allocated_file_bytes=st.st_blocks*512,sha256=sha(q),**paths[q]));rec['logical_bytes']+=st.st_size;rec['allocated_file_bytes']+=st.st_blocks*512
  for row in rec['deleted']:
   q=Path(row['path']);assert sha(q)==row['sha256'] and q.stat().st_size==row['bytes'];q.unlink()
  for name in names:assert sha(A/name)==rec['receipt_pins'][str(A/name)]
  dst.write_text(json.dumps(rec,indent=2)+'\n')
 print(json.dumps(dict(manifest=str(dst),files=len(rec['deleted']),logical_bytes=rec['logical_bytes'],allocated_file_bytes=rec['allocated_file_bytes'])))
if __name__=='__main__':main()
