import os,json,subprocess
from pathlib import Path
cfg=json.loads((Path.home()/'.local/share/strata-sycl/serve-config-iq3_s.json').read_text())
a=cfg['args'].copy(); a.remove('--serve'); del a[a.index('--tokens'):a.index('--tokens')+2]
for k,v in [('--max-context','2048'),('--prefill',os.environ.get('PERF_PREFILL','64')),('--expert-cache','1649'),('--adapt-swaps','0'),('--pool-workers',os.environ.get('PERF_WORKERS','4'))]:a[a.index(k)+1]=v
a+=['--tokens-file','/tmp/strata-sycl-functional-context-v4-tokens.txt','--check-logits','--stats']
env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='level_zero:gpu')
if os.environ.get('PERF_TIMING','1')=='1':env['STRATA_PREFILL_TIMING']='1'
else:env.pop('STRATA_PREFILL_TIMING',None)
with open('/tmp/strata-sycl-perf-'+os.environ.get('PERF_NAME','baseline')+'-profile.log','w') as f: subprocess.run([os.environ.get('PERF_EXE','/tmp/strata-sycl-perf-baseline-8e41083')]+a,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
