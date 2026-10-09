import hashlib,json,os,re,subprocess,time
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
base=Path.home()/'.local/share/strata-sycl'
exe=Path('/tmp/strata-upstream-arc-host-staging-aot')
shard=base/'models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
args=['--pack',str(base/'packs/qwen3.8-flash-next-iq3_s'),'--native',str(shard),'--tokens-file','bench/results/2026-10-05-sycl-upstream-arc/writing-tokens.txt','--max-new','128','--max-context','512','--spec','4','--spec-min-p','0.9','--mtp',str(base/'mtp/rt'),'--suffix-draft','0','--prefill','1024','--expert-cache','auto','--expert-profile','data/expert-profile.bin','--expert-cache-per-layer','--pool-workers','5','--pcie-frac','0','--greedy','--check-logits','--stats','--dump-final-r','/tmp/strata-upstream-arc-split-dma-single-residual.bin']
log=Path('/tmp/strata-upstream-arc-baseline-writing.log')
start=time.monotonic()
try:
 with log.open('w') as f:r=subprocess.run([str(exe)]+args,cwd=root,env=os.environ,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 rc=r.returncode
except subprocess.TimeoutExpired:rc=124
s=log.read_text()
record={'exe':str(exe),'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'args':args,'env':{k:v for k,v in os.environ.items() if k.startswith(('STRATA_','SYCL_','ONEAPI_','NEO_')) or k=='LD_LIBRARY_PATH'},'exit_code':rc,'wall_seconds':time.monotonic()-start,'log':s}
log.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
print('exit',rc,'seconds',record['wall_seconds'],flush=True)
print('\n'.join(s.splitlines()[-40:]),flush=True)
raise SystemExit(rc)
