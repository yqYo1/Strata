import json,os,subprocess,time
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
shard=Path.home()/'.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf'
fixture=root/'bench/results/2026-10-05-sycl-prefill-scaling/coding-context-tokens.txt'
exe=Path('/tmp/strata-upstream-arc-ple-read-probe');out=Path('/tmp/strata-upstream-arc-ple-probe');out.mkdir(exist_ok=True)
records=[]
for mode,threads in [('direct',16),('mmap',16),('direct',64)]:
 label=f'{mode}-{threads}';log=out/(label+'.log');args=[str(exe),str(shard),str(fixture),'2048,4096,8087',mode,'2'];env=dict(os.environ,STRATA_IO_THREADS=str(threads));start=time.monotonic()
 try:
  with log.open('w') as f:r=subprocess.run(args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
  rc=r.returncode
 except subprocess.TimeoutExpired:rc=124
 lines=log.read_text().splitlines();rows=[json.loads(l) for l in lines if l.startswith('{')];records.append({'label':label,'args':args,'exit_code':rc,'wall_seconds':time.monotonic()-start,'rows':rows,'log':log.name});(out/'run.json').write_text(json.dumps({'runs':records},indent=2)+'\n')
 print(label,'exit',rc,'rows',rows,flush=True)
 if rc:raise SystemExit(rc)
