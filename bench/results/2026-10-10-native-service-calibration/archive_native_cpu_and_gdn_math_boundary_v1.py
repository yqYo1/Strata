"""Root compact evidence archive; no tensor copies, cleanup, GPU, tests or builds."""
from pathlib import Path
import csv,datetime,fcntl,hashlib,io,json,shutil
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROLE=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-role-plan-v0141-20261010')
ORACLE=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-gdn-wy-equation-oracle-20261010')
RA=ROLE/'bench/results/2026-10-10-native-service-calibration'
OA=ORACLE/'bench/results/2026-10-10-gdn-wy-equation-oracle'
items=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy(src,target):
 src=Path(src);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
 assert not target.exists(),str(target)
 shutil.copyfile(src,target);assert sha(src)==sha(target)
 items.append(dict(original_path=str(src),original_sha256=sha(src),archived_path=str(target),archived_sha256=sha(target),bytes=target.stat().st_size))
def closed(p):
 d=json.loads(Path(p).read_text());assert not d['active'] and not d['cleanup'] and not d['survivors']
 return d

def main():
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not OA.exists();OA.mkdir(parents=True)
  for n in ('run_native_service_calibration_cpu_v1.py','run_native_service_calibration_cpu_v2.py','run_native_service_calibration_cpu_v3.py','run_native_service_calibration_cpu_v4.py','summarize_native_service_calibration_cpu_v1.py','summarize_native_service_calibration_cpu_v2.py','summarize_native_service_calibration_cpu_v3.py','build_native_service_calibration_cpu_v3.py','build_native_service_calibration_cpu_v4.py','native-service-calibration-affinity-invalidation-v1.json','native-service-calibration-initial-launch-admission-v1.json'):
   copy(B/n,RA/n)
  # One canonical real selected-role identity list, not payload bytes.
  selection=None;selection_path=RA/'selected-real-weight-role-identities.json'
  for rep in (2,3):
   name=f'native-service-calibration-22-20-nt1-correctness-tasks6-batch6-r{rep}'
   src=B/name/'record.json';d=closed(src);assert d['passed'] and d['complete']
   selected=d.pop('selected_extent_sha256');assert len(selected)==1152
   if selection is None:
    selection=selected;selection_path.write_text(json.dumps(selection,indent=2)+'\n')
   else:assert selected==selection,'real-weight role identities changed'
   target=RA/name/'compact-record.json';target.parent.mkdir(parents=True)
   result=dict(original_record_path=str(src),original_record_sha256=sha(src),original_status_preserved=True,
               selected_extent_field_archived_separately=dict(path=str(selection_path.relative_to(RA)),sha256=sha(selection_path),count=1152),record_except_deduplicated_field=d)
   target.write_text(json.dumps(result,indent=2)+'\n')
   items.append(dict(original_path=str(src),original_sha256=sha(src),archived_path=str(target),archived_sha256=sha(target),bytes=target.stat().st_size,transform='selected role identity field deduplicated; original status kept'))
  failed=B/'native-service-calibration-22-20-nt1-correctness-tasks6-batch6-r1'
  assert not closed(failed/'record.json')['passed']
  for n in ('record.json','selection-error.txt'):copy(failed/n,RA/failed.name/n)
  co=B/'native-service-calibration-cohort-22-20-nt1-v2'
  for f in co.iterdir():
   if f.is_file():copy(f,RA/co.name/f.name)
  for task,rep in ((6,1),(0,1),(0,2),(6,2),(6,3),(0,3),(0,4),(6,4)):
   name=f'native-service-calibration-22-20-nt1-timing-tasks{task}-batch6-r{rep}'
   src=B/name;d=closed(src/'record.json');assert d['passed'] and d['complete']
   copy(src/'record.json',RA/name/'record.json')
   raw=src/'uninstrumented-rounds/stdout.csv';allrows=list(csv.reader(raw.read_text().splitlines()))
   keep=[r for r in allrows if r and r[0] not in ('EXTENT','ID','REFERENCE')]
   assert sum(r[0]=='ROUND' and r[1]!='cohort' for r in keep)==250
   assert sum(r[0]=='WARMUP' for r in keep)==30
   out=RA/name/'individual-rounds.csv';stream=io.StringIO(newline='');csv.writer(stream,lineterminator='\n').writerows(keep);out.write_text(stream.getvalue())
   items.append(dict(original_path=str(raw),original_sha256=sha(raw),archived_path=str(out),archived_sha256=sha(out),bytes=out.stat().st_size,transform='retain every warmup/round and env/meta/placement/result; remove duplicate per-ID extent/reference lines already verified by closed receipt/cohort/role identities',timing_interpretation_valid=rep>=3))
  for version in (1,2,3):
   src=B/f'native-service-calibration-22-20-nt1-summary-v{version}'
   for f in src.iterdir():
    if f.is_file():copy(f,RA/src.name/f.name)
  for version in (3,4):
   src=B/f'native-service-calibration-cpu-build-v{version}';d=closed(src/'record.json');assert d['passed'] and d['complete']
   for f in src.iterdir():
    if f.is_file() and f.name in ('record.json','compile_commands.json','target-commands.stdout','direct-imports.stdout','compiler-c.stdout','compiler-cxx.stdout'):
     copy(f,RA/src.name/f.name)
  for n in ('run_gdn_wy_equation_oracle_cpu_v1.py','run_gdn_wy_equation_oracle_cpu_v2.py'):copy(B/n,OA/n)
  for ver in (1,2):
   src=B/f'gdn-wy-equation-oracle-cpu-validation-v{ver}';d=closed(src/'record.json')
   assert d['passed']==(ver==2)
   copy(src/'record.json',OA/src.name/'record.json')
   if ver==1:copy(src/'asan-ubsan-build.stdout',OA/src.name/'asan-ubsan-build.stdout')
   else:
    for n in ('compiler.stdout','release-compile_commands.json','asan-ubsan-compile_commands.json','release-imports.stdout','asan-ubsan-imports.stdout'):
     copy(src/n,OA/src.name/n)
    for label in ('release','asan-ubsan'):
     raw=src/(label+'-fixtures.stdout');canonical=OA/'all100-case-results.csv'
     if label=='release':copy(raw,canonical)
     else:assert sha(raw)==sha(canonical)
  for n in ('implementation-gdn-wy-equation-oracle-v1.txt','gdn-wy-equation-oracle-independent-review-round85.txt'):
   copy(B/'research-20261009'/n,OA/n)
  now=datetime.datetime.now(datetime.timezone.utc).isoformat()
  for archive in (RA,OA):
   selected_items=[i for i in items if Path(i['archived_path']).is_relative_to(archive)]
   (archive/'root-compact-archive-manifest-v2.json').write_text(json.dumps(dict(created_utc=now,controller_sha256=sha(__file__),items=selected_items,raw_deleted=False,model_payload_copied=False),indent=2)+'\n')
  copy(__file__,RA/Path(__file__).name)
  print(json.dumps(dict(archived_files=len(items),bytes=sum(i['bytes'] for i in items),role_archive=str(RA),oracle_archive=str(OA),raw_deleted=False)))

if __name__=='__main__':main()
