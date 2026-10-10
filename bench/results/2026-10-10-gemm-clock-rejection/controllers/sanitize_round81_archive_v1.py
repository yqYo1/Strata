"""Remove inherited credentials from an unpublished archive, preserving raw receipts."""
from pathlib import Path
import copy, datetime, fcntl, hashlib, json, subprocess
B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A=W/'bench/results/2026-10-10-gemm-clock-rejection'
RAW=B/'prefill-gemm-only-contract-root-v1/record.json'
DEST=A/'qualification/prefill-gemm-only-contract-root-v1/record.json'
ALLOW={'PATH','LANG','LANGUAGE','LC_ALL','LC_CTYPE','LD_LIBRARY_PATH','CPATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','TMPDIR','TMP','TEMP','CXX','OMP_NUM_THREADS','MKL_NUM_THREADS','STRATA_PREFILL_GEMM_SERVICE_TIMING','STRATA_PREFILL_SERVICE_TIMING','STRATA_PREFILL_TRANSFER_TIMING'}
def ident(p):
 with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert ident(RAW)['sha256']=='2c45015582fbe9f5139ef55c383d92430a5076f63940a085037664e95886fff3'
 assert ident(DEST)['sha256']==ident(RAW)['sha256']
 original=json.loads(RAW.read_text()); redacted=copy.deepcopy(original)
 assert original['passed'] and original['complete'] and not original['active']
 changes=[]; secret_values=set()
 def walk(v,path=''):
  if isinstance(v,dict):
   for k,x in list(v.items()):
    p=path+'/'+k
    if k in {'environment','environment_base','env'} and isinstance(x,dict):
     removed=sorted(set(x)-ALLOW)
     for name in removed:
      val=x[name]
      if isinstance(val,str) and len(val)>=12 and any(t in name.upper() for t in ['TOKEN','SECRET','PASSWORD','API_KEY']):secret_values.add(val.encode())
     v[k]={name:val for name,val in x.items() if name in ALLOW}
     changes.append(dict(json_path=p,removed_key_names=removed,retained_key_names=sorted(v[k])))
    else:walk(x,p)
  elif isinstance(v,list):
   for i,x in enumerate(v):walk(x,path+'/'+str(i))
 walk(redacted)
 assert len(changes)==15
 redacted['archive_redaction']={'policy':'Execution-relevant environment allowlist; original private execution receipt remains immutable. No result, command, identity, flags, or validation outcome changed.','original_identity':ident(RAW),'redaction_manifest':'environment-redaction.json'}
 write(DEST,redacted)
 invp=A/'archive-inventory.json';inv=json.loads(invp.read_text());found=0
 for entry in inv['files']:
  if entry['archive']==str(DEST.relative_to(W)):
   assert entry['original']==ident(RAW)
   entry.update(byte_identical=False,archive_identity=ident(DEST),transformation='Recursive execution-environment allowlist; see environment-redaction.json')
   found+=1
 assert found==1
 write(invp,inv)
 manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='GitHub push protection rejected inherited Hugging Face credential in unpushed commit; do not bypass. Other inherited credentials also omitted by allowlist.',original_private_receipt=ident(RAW),redacted_archive=ident(DEST),raw_receipt_changed=False,result_status_changed=False,environment_allowlist=sorted(ALLOW),environment_maps=changes,removed_values_recorded=False,source_controller_v1_unsafe_for_new_archival_execution=True,future_controller='run_prefill_gemm_only_contract_v2.py')
 write(A/'environment-redaction.json',manifest)
 # Scan tracked and newly added textual/binary content for the original private secret values,
 # without printing any values or making assumptions about GitHub's supported patterns.
 paths=subprocess.check_output(['git','ls-files','-z'],cwd=W).split(b'\0')
 violations=[]
 for rel in paths:
  if not rel:continue
  p=W/rel.decode();data=p.read_bytes()
  if any(value in data for value in secret_values):violations.append(rel.decode())
 assert not violations, 'Inherited credential still present in tracked files; no values displayed'
 assert ident(RAW)['sha256']==manifest['original_private_receipt']['sha256']
 print(json.dumps(dict(redacted_environment_maps=len(changes),tracked_files_scanned=len(paths)-1,original_private_unchanged=True,archive_credentials_absent=True)))
