import json,os,subprocess,re,hashlib
from pathlib import Path
ref=json.load(open('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json'));env=dict(os.environ,**ref['env']);env['SYCL_CACHE_PERSISTENT']='0'
exe=Path('/tmp/strata-upstream-arc-f16-pad-probe');rows=[]
for profiling in [0,1]:
 log=Path('/tmp/strata-upstream-arc-f16-pad-profiling-'+str(profiling)+'.log')
 with log.open('w') as f:subprocess.run([str(exe)],env=dict(env,STRATA_PROBE_QUEUE_PROFILING=str(profiling)),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
 current=[]
 for line in log.read_text().splitlines():
  m=re.fullmatch(r'(PASS|FAIL) N=(\d+) K=(\d+) T=(\d+) padded=(\d+) multiple=(\d+) unequal=(\d+) copied_unequal=(\d+) guards=(\d+) padding_nonzero=(\d+) finite=(\d+) ref_L1=([.\d+e-]+) pad_L1=([.\d+e-]+) original=([.\d]+)ms padded=([.\d]+)ms copied=([.\d]+)ms',line)
  if not m:continue
  assert m[1]=='PASS' and m[9]=='0' and m[10]=='0' and m[11]=='1'
  r=dict(queue_profiling=profiling,N=int(m[2]),K=int(m[3]),T=int(m[4]),padded=int(m[5]),multiple=int(m[6]),unequal=int(m[7]),copied_unequal=int(m[8]),original_ms=float(m[14]),padded_ms=float(m[15]),copied_ms=float(m[16]),original_L1_error=float(m[12]),padded_L1_error=float(m[13]),finite=True,guards_pass=True,padding_zero=True)
  current.append(r);rows.append(r)
 assert len(current)==72
 print('queue_profiling',profiling,'cases',len(current),'exact',sum(r['unequal']==0 and r['copied_unequal']==0 for r in current),'copied faster',sum(r['copied_ms']<r['original_ms'] for r in current),flush=True)
 for r in current:
  if r['N']==2560 and r['T'] in [65,129,159] and r['multiple'] in [64,256]:print(r,flush=True)
Path('/tmp/strata-upstream-arc-f16-pad-report.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path('/tmp/strata-upstream-arc-f16-pad-probe.cpp').read_bytes()).hexdigest(),runs=rows),indent=2)+'\n')
