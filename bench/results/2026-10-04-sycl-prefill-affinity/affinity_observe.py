from pathlib import Path
import subprocess,os,time,json
exe='/tmp/strata-sycl-goal-prefill-event-measured-aot'
env=dict(os.environ,PERF_EXE=exe,STRATA_SYCL_PREFILL_SAME_QUEUE='1')
for k in ('STRATA_PREFILL_RING','STRATA_PREFILL_ISSUER','STRATA_PREFILL_DEVICE_TRACE'):env.pop(k,None)
p=subprocess.Popen(['python3','/tmp/strata-sycl-goal-prefill-event-profile.py','4k','affinity-observe'],env=env)
samples=[]
while p.poll() is None:
 try:children=Path(f'/proc/{p.pid}/task/{p.pid}/children').read_text().split()
 except OSError:children=[]
 for pid in children:
  try:
   args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
   if args[0].decode()!=exe:continue
   tasks=[]
   for d in Path(f'/proc/{pid}/task').iterdir():
    try:
     status=d.joinpath('status').read_text();allowed=next(x.split(':',1)[1].strip() for x in status.splitlines() if x.startswith('Cpus_allowed_list:'))
     v=d.joinpath('stat').read_text().rsplit(')',1)[1].split()
     tasks.append(dict(tid=int(d.name),affinity=allowed,state=v[0],utime=int(v[11]),stime=int(v[12]),start=int(v[19]),processor=int(v[36])))
    except (OSError,StopIteration):pass
   samples.append(dict(time=time.time(),pid=int(pid),threads=tasks))
  except OSError:pass
 time.sleep(1)
Path('/tmp/strata-sycl-goal-prefill-affinity-observe.json').write_text(json.dumps(dict(exit_code=p.returncode,samples=samples),indent=2)+'\n')
print('Affinity snapshots:',len(samples),'return',p.returncode,flush=True)
raise SystemExit(p.returncode)
