import os,json,subprocess,sys
from pathlib import Path
ref=json.load(open('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json'));env=dict(os.environ,**ref['env']);env['SYCL_CACHE_PERSISTENT']='0'
exe='/tmp/strata-upstream-arc-attn-order-probe'
for i,variant in enumerate([3,4,4,3],1):
 log=Path('/tmp/strata-upstream-arc-attn-order-v'+str(variant)+'-r'+str(i)+'.log')
 with log.open('w') as f:subprocess.run([exe,'8192','513','5','128',str(variant)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
 rows=[s for s in log.read_text().splitlines() if s.startswith('PASS')]
 assert len(rows)==8,rows
 print('variant',variant,'run',i,'all8casespass',flush=True)
 for row in rows:print(row,flush=True)
