import os,json,subprocess,re
from pathlib import Path
p=json.load(open('/tmp/strata-sycl-verifier-native-check.json'))['mtp64'];flags=p['flags'].copy();flags[flags.index('--spec-min-p')+1]='0.9'
dump='/tmp/strata-sycl-mtp-floor-static64-windows.bin';log='/tmp/strata-sycl-mtp-floor-static64.log';Path(dump).unlink(missing_ok=True)
env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='level_zero:gpu',STRATA_SYCL_VERIFY_LOGITS=dump);env.pop('STRATA_SYCL_VERIFY_NATIVE_CAPTURE',None);env.pop('STRATA_SYCL_ADAPT_SYNC',None);env.pop('STRATA_TRACE',None)
with open(log,'w') as f:subprocess.run(['./build-sycl/strata']+flags,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
ids=list(map(int,re.search(r'^output  : (.*)$',Path(log).read_text(),re.M)[1].split()));reference_ids=json.load(open('bench/results/2026-10-03-sycl-q2-xmx/run.json'))['output_ids'][:64];assert ids==reference_ids
import sys,numpy as np
sys.path.insert(0,'tools/sycl');from compare_verify_logits import read
start=36
prompt=list(map(int,Path('/tmp/strata-sycl-writing-tokens.txt').read_text().replace(',',' ').split()))
def committed(path,output):
 rows={}
 for pos,tokens,values in read(path):
  top=values.argmax(axis=1);accepted=0
  while accepted<len(tokens)-1 and tokens[accepted+1]==top[accepted]:accepted+=1
  for j in range(accepted+1):
   index=pos+j-start
   if index>=len(output):continue
   assert index>=0 and int(top[j])==output[index]
   assert int(tokens[j])==(prompt[-1] if index==0 else output[index-1])
   assert pos+j not in rows;rows[pos+j]=values[j]
 assert sorted(rows)==list(range(start,start+len(output)))
 return rows
base='/tmp/strata-sycl-verify-overlap-after-64-windows.bin';a=committed(base,reference_ids);b=committed(dump,ids)
unequal=[pos for pos in a if not np.array_equal(a[pos].view('<u4'),b[pos].view('<u4'))]
maximum=max(float(np.max(np.abs(a[pos].astype('float64')-b[pos]))) for pos in a)
result=dict(flags=flags,log=log,reference=base,candidate=dump,output_ids=ids,generated_ids_equal=True,committed_rows=len(a),bitwise_equal=not unequal,unequal_positions=unequal,absolute_error_max=maximum,scope='Different probability floors with adaptive changes disabled and fixed 2048 profile slots; only committed-token rows share the same input history. Rejected draft rows are excluded because the proposal windows differ.')
Path('/tmp/strata-sycl-mtp-floor-static-check.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['flags','output_ids']},flush=True)
