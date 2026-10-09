"""Root-owned actual native-role CPU smoke; traced samples are not performance."""
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-role-plan-v0141-20261010')
PACK=Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
NATIVE=Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
BUILD=B/'native-service-host-cpu-build-v1/record.json'
TRACE=B/'owned-cache-route-pairs-v0141-code32k-tasks6-diagnostic-r6/debugger/inferior.stderr'
EXPECTED={
 BUILD:'c8227c12ea5701da507bf87ed54d7a20e3b4286c50c75b01837e6409294ff92c',
 TRACE:'3309c390f8e5f047321ca84bcdc6d182e9c13ca1b8db178c79a88b108e13ab3f',
 PACK/'native_experts.txt':'d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d',
 PACK/'conversions.json':'51df3cd6ffd0d38c2da8e2d3a95a604b4a96c20b06fc639a9bf477282a42b86a',
}


def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def members(pgid):
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdecimal():continue
  try:
   words=(p/'stat').read_text().rsplit(')',1)[1].split()
   if int(words[2])==pgid and words[0]!='Z':result.append(dict(pid=int(p.name),start_ticks=int(words[19]),rss_bytes=int(words[21])*os.sysconf('SC_PAGE_SIZE')))
  except (FileNotFoundError,ProcessLookupError):pass
 return result


def main():
 assert __debug__
 with (B/'owned-v0141-measurement.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for path,pin in EXPECTED.items():assert sha(path)==pin,str(path)
  build=json.loads(BUILD.read_text());assert build['passed'] and not build['active'] and not build['cleanup'] and not build['survivors']
  binary=Path(build['binary']['path']);assert sha(binary)==build['binary']['sha256']
  for path,pin in build['pins'].items():assert sha(path)==pin,path
  layers={}
  for line in (PACK/'native_experts.txt').read_text().splitlines():
   if not line or line.startswith('#'):continue
   fields=line.split();layers[int(fields[0])]=dict(gu=int(fields[1]),down=int(fields[2]),blob_bytes=int(fields[4]))
  assert set(layers)==set(range(48))
  eligible={}
  for line in TRACE.read_text().splitlines():
   fields=line.split()
   if fields[:2]!=['CACHE_ROUTE_PAIRS_V1','PAIR']:continue
   assert len(fields)==9
   layer,expert,n,hits,refused,offloaded,j=map(int,fields[2:])
   nt1,nt2=2*j-n,n-j
   if refused>0 and nt1>0 and nt2>0:
    eligible.setdefault((layers[layer]['gu'],layers[layer]['down']),{})[(layer,expert)]=dict(entries=n,callback_pairs=j,NT1=nt1,NT2=nt2,refused=refused)
  selected=[]
  # Fixed source-ID ordering, not service times; two IDs per available GU/Down cell.
  for cell,ids in sorted(eligible.items()):
   assert cell[0] in (18,21,22,23) and cell[1] in (20,42)
   for pair in sorted(ids)[:2]:selected.append(dict(layer=pair[0],expert=pair[1],GU=cell[0],Down=cell[1],route=ids[pair]))
  assert 1<=len(selected)<=16
  out=B/'native-service-host-actual-cpu-smoke-v1';out.mkdir(mode=0o700)
  cohort=out/'cohort.tsv';cohort.write_text('layer\texpert\n'+''.join(f"{x['layer']}\t{x['expert']}\n" for x in selected));cohort.chmod(0o600)
  conversions=json.loads((PACK/'conversions.json').read_text())
  model_identity=[]
  for entry in conversions['source_shards']:
   path=NATIVE.parent/entry['name'];s=path.stat();assert path.is_file() and s.st_size==entry['size']
   model_identity.append(dict(path=str(path),bytes=s.st_size,dev=s.st_dev,ino=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns))
  env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_DEBUG='libs')
  trace=out/'system-call-trace.txt'
  argv=['/usr/bin/strace','-f','-yy','-s','512','-e','trace=open,openat,openat2,close,close_range,mmap,ioctl,execve','-o',str(trace),str(binary),str(PACK),str(NATIVE),str(cohort)]
  record=dict(active=True,complete=False,passed=False,scope='Traced controlled hot-cohort actual native CPU payload correctness smoke; no engine or GPU inference',controller_sha256=sha(__file__),build_receipt_sha256=EXPECTED[BUILD],binary=build['binary'],input_pins={str(p):h for p,h in EXPECTED.items()},model_identity=model_identity,whole_model_payload_hashed=False,cohort=dict(path=str(cohort),sha256=sha(cohort),selected=selected,selection='first two sorted eligible IDs per GU/Down cell with both nonresident NT1 and NT2 route exposure; never timing-selected',full_blob_bytes=sum(layers[x['layer']]['blob_bytes'] for x in selected)),argv=argv,cwd=str(W),environment=env,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),deadline_seconds=180,text_budget_bytes=16*1024**2,limits=dict(AS_each_bytes=96*1024**3,AS_reason='full model mmap is virtual; selected blobs bounded separately',RSS_group_poll_bytes=2*1024**3,CPU_each_soft_seconds=120,CPU_each_hard_seconds=121,FSIZE_each_bytes=16*1024**2,NOFILE=256,CORE_bytes=0),peak_group_rss_bytes=0,cleanup=[],survivors=[],gpu_work_submitted=False,inference_run=False,performance_eligible=False,adopted=False,full_lifecycle_passed=False,independent_NT2_Q2_arithmetic_certified=False,retained_trace=dict(owner='/root',byte_budget=16*1024**2,consumer='runtime no-device access and numerical failure diagnosis',review_point='after compact syscall/numeric evidence is committed and actual untraced calibration starts'))
  rp=out/'record.json';start=time.monotonic();proc=None
  def save():rp.write_text(json.dumps(record,indent=2)+'\n')
  save();stdout=out/'samples.csv';stderr=out/'pool-diag.txt'
  try:
   with stdout.open('wb') as so,stderr.open('wb') as se:
    def limits():
     for kind,pair in ((resource.RLIMIT_AS,(96*1024**3,96*1024**3)),(resource.RLIMIT_CPU,(120,121)),(resource.RLIMIT_FSIZE,(16*1024**2,16*1024**2)),(resource.RLIMIT_NOFILE,(256,256)),(resource.RLIMIT_CORE,(0,0))):resource.setrlimit(kind,pair)
    proc=subprocess.Popen(argv,cwd=W,env=env,stdout=so,stderr=se,preexec_fn=limits,start_new_session=True)
    record['owner']=dict(pid=proc.pid,pgid=proc.pid,start_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]));save()
    while proc.poll() is None:
     assert time.monotonic()-start<180,'CPU smoke wall deadline'
     assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<=16*1024**2,'aggregate diagnostic text budget'
     rss=sum(x['rss_bytes'] for x in members(proc.pid));record['peak_group_rss_bytes']=max(record['peak_group_rss_bytes'],rss)
     assert rss<=2*1024**3,'CPU smoke group RSS cap'
     time.sleep(.05)
   record['exit_code']=proc.returncode
   assert not members(proc.pid),'owned descendants survived'
   system=trace.read_text();record['device_path_mentions']=[s for s in system.splitlines() if re.search(r'/dev/(dri|nvidia|kfd)',s)]
   record['device_runtime_library_mentions']=[s for s in system.splitlines() if re.search(r'lib(?:sycl|ur_|ze_loader|mkl|igc|intelocl)',s,re.I)]
   assert not record['device_path_mentions'] and not record['device_runtime_library_mentions'],'device access/runtime loader observed'
   loader=stderr.read_text();objects=sorted(set(re.findall(r'calling init: (.+)',loader)))
   assert objects and 'transferring control: '+str(binary) in loader,'dynamic loader record present'
   assert not any(re.search(r'lib(?:sycl|ur_|ze_loader|mkl|igc|intelocl)',p,re.I) for p in objects),'unexpected loaded runtime'
   record['observed_loader_initialization_objects']=objects
   record['runtime_audit_scope']='strace followed exec-to-exit and all threads, decoded fd paths; no GPU device open/ioctl/runtime-library path; LD_DEBUG initialization object set captured; not an API interposition claim'
   assert proc.returncode==0,'actual native service child failed'
   rows=list(csv.reader(stdout.read_text().splitlines()));assert rows[-1]==['RESULT','pass','partition_and_repeat_bitexact_only'],'harness completion'
   samples=[x for x in rows if x[:1]==['SAMPLE'] and x[1]!='layer'];assert len(samples)==len(selected)*2*2*5,'all individual samples'
   seen=set()
   for row in samples:
    assert len(row)==22,'sample columns'
    layer,expert,gu,down,nt,tasks,workers,repeat=map(int,row[1:9]);key=(layer,expert,nt,tasks,repeat)
    assert key not in seen and nt in (1,2) and tasks in (0,6) and workers==5 and 0<=repeat<5,'sample cell geometry';seen.add(key)
    values=list(map(float,row[9:18]));assert all(math.isfinite(v) and v>=0 for v in values),'finite phase samples'
    assert math.isfinite(float(row[19])) and math.isfinite(float(row[20])),'finite independent delta characterization'
   assert {(int(x[1]),int(x[2])) for x in samples}=={(x['layer'],x['expert']) for x in selected},'exact selected IDs'
   extents=[x for x in rows if x[:1]==['EXTENT']];assert len(extents)==3*len(selected),'all selected role extents'
   extent_pins=[]
   for row in extents:
    assert len(row)==9
    layer,expert,role=map(int,row[1:4]);path=Path(row[4]);offset,length=int(row[6]),int(row[7]);assert path.parent==NATIVE.parent and 0<=offset and 0<length<=4*1024**2
    with path.open('rb') as f:f.seek(offset);data=f.read(length)
    assert len(data)==length,'selected extent read';extent_pins.append(dict(layer=layer,expert=expert,role=role,path=str(path),offset=offset,bytes=length,sha256=hashlib.sha256(data).hexdigest()))
   for original in model_identity:
    s=Path(original['path']).stat();assert all(original[k]==v for k,v in dict(bytes=s.st_size,dev=s.st_dev,ino=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns).items()),'model shard metadata changed'
   for path,pin in EXPECTED.items():assert sha(path)==pin,'input changed'
   for path,pin in build['pins'].items():assert sha(path)==pin,'harness changed'
   assert sha(binary)==build['binary']['sha256'],'binary changed'
   record.update(complete=True,passed=True,actual_weight_service_measured=True,individual_samples=len(samples),selected_extent_sha256=extent_pins,correctness_gate_scope='finite writes/quantized meaningful bytes/NT1 GGML GU+Down20/direct-pool partition/repeat exact; NT2 and Q2 independent deltas characterized only')
  except BaseException as e:record['error']=type(e).__name__+': '+str(e)
  finally:
   if proc is not None:
    current=members(proc.pid)
    if proc.poll() is None or current:
     record['cleanup'].append(dict(signal='TERM',pgid=proc.pid,members=current))
     try:os.killpg(proc.pid,signal.SIGTERM)
     except ProcessLookupError:pass
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:pass
     if proc.poll() is None or members(proc.pid):
      record['cleanup'].append(dict(signal='KILL',pgid=proc.pid))
      try:os.killpg(proc.pid,signal.SIGKILL)
      except ProcessLookupError:pass
      proc.wait(timeout=5)
    record['survivors']=members(proc.pid)
   record.update(active=False,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (stdout,stderr,trace,cohort) if p.exists()});save()
  print(json.dumps(dict(record=str(rp),sha256=sha(rp),passed=record['passed'],error=record.get('error'),elapsed_seconds=record['elapsed_seconds'])))
  return 0 if record['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
