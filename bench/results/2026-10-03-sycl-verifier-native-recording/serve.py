import os,json,subprocess,threading,queue,time
from pathlib import Path
flags=json.load(open('/tmp/strata-sycl-mtp-serve.json'))['flags'].copy()
for key,val in [('--max-context','512'),('--expert-cache','auto'),('--adapt-swaps','64')]:flags[flags.index(key)+1]=val
flags+=['--stats'];env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='level_zero:gpu');env.pop('STRATA_SYCL_ADAPT_SYNC',None);env.pop('STRATA_SYCL_VERIFY_LOGITS',None);env.pop('STRATA_TRACE',None)
mode=os.environ.get('STRATA_SYCL_VERIFY_NATIVE_CAPTURE','0');env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE']=mode
trial=os.environ.get('STRATA_BENCH_TRIAL','1')
prefix=f'/tmp/strata-sycl-verifier-native-{mode}'+(f'-trial{trial}' if trial!='1' else '')
err=open(prefix+'.stderr.log','w');stdout=open(prefix+'.stdout.log','w');p=subprocess.Popen(['./build-sycl/strata']+flags,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1);q=queue.Queue()
def reader():
 for line in p.stdout:stdout.write(line);stdout.flush();q.put(line.strip())
 q.put(None)
threading.Thread(target=reader,daemon=True).start()
def read(timeout=180):
 line=q.get(timeout=timeout)
 if line is None:raise RuntimeError(f'engine ended {p.poll()}')
 if line.startswith('ERR'):raise RuntimeError(line)
 return line
start=time.monotonic()
while True:
 ready=read()
 if ready.startswith('READY '):break
result=dict(flags=flags,ready=ready,startup_seconds=time.monotonic()-start,runs=[],environment=dict(STRATA_SYCL_VERIFY_NATIVE_CAPTURE=mode,STRATA_SYCL_ADAPT_SYNC='unset: default completed refill boundary'))
def drm_info():
 rows=[]
 for fd in Path(f'/proc/{p.pid}/fdinfo').glob('*'):
  try:text=fd.read_text()
  except OSError:continue
  if 'drm-driver:' in text:rows.append(text)
 return rows
def run(prompt,new,cancel=False):
 ids=Path('/tmp/strata-sycl-'+('writing' if prompt=='writing' else 'profile-coding')+'-tokens.txt').read_text().strip();ids=','.join(ids.replace(',',' ').split())
 p.stdin.write(f'GEN {new} {ids}\n');p.stdin.flush();start=time.monotonic();tokens=[];sent=False
 while True:
  line=read(timeout=60)
  if line.startswith('T '):
   tokens.append(int(line.split()[1]))
   if cancel and not sent:p.stdin.write('STOP\n');p.stdin.flush();sent=True
  if line.startswith('DONE '):break
 fields=line.split();assert int(fields[1])==len(tokens)
 if cancel:assert fields[5]=='cancel'
 else:assert len(tokens)==new and fields[5]=='length'
 row=dict(prompt=prompt,requested_tokens=new,output_ids=tokens,done=line,wall_seconds=time.monotonic()-start,decode_ms=float(fields[4]),decode_tok_s=int(fields[1])*1000/float(fields[4]),prompt_ms=float(fields[3]),reused_prompt_tokens=int(fields[8]),cache_hits=int(fields[9]),cache_lookups=int(fields[10]),cancelled=cancel)
 row['drm_fdinfo']=drm_info();result['runs'].append(row);Path(prefix+'.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in row.items() if k not in ['output_ids','drm_fdinfo']},flush=True)
 return row
try:
 for prompt in ['writing','coding']:
  for trial in range(1,3):run(prompt,128)
 run('writing',128,cancel=True)
 last=run('writing',8);assert last['output_ids']==json.load(open('bench/results/2026-10-03-sycl-q2-xmx/run.json'))['output_ids'][:8]
 result['cancel_recovery_known_eight_ids']=True;p.stdin.write('QUIT\n');p.stdin.flush();p.wait(timeout=60);assert p.returncode==0
 result['exit_code']=p.returncode;Path(prefix+'.json').write_text(json.dumps(result,indent=2)+'\n');print('default completed refill boundary: repeat requests and cancel recovery passed',flush=True)
finally:
 if p.poll() is None:p.terminate();p.wait(timeout=60)
 err.close();stdout.close()
