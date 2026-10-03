import os,json,subprocess,re
from pathlib import Path
base=json.load(open('/tmp/strata-sycl-adapt-boundary-check.json'))['runs'][0]
flags=base['flags'];dump='/tmp/strata-sycl-verifier-native-default-adaptive64-windows.bin';Path(dump).unlink(missing_ok=True)
log='/tmp/strata-sycl-verifier-native-default-adaptive64.log'
env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='level_zero:gpu',STRATA_SYCL_VERIFY_LOGITS=dump);env.pop('STRATA_SYCL_VERIFY_NATIVE_CAPTURE',None);env.pop('STRATA_SYCL_ADAPT_SYNC',None);env.pop('STRATA_TRACE',None)
with open(log,'w') as f:subprocess.run(['./build-sycl/strata']+flags,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
ids=list(map(int,re.search(r'^output  : (.*)$',Path(log).read_text(),re.M)[1].split()));assert ids==base['output_ids']
comparison=json.loads(subprocess.check_output(['/home/yayoi/.local/share/strata-sycl/venv/bin/python','tools/sycl/compare_verify_logits.py',base['dump'],dump,'--require-exact'],text=True))
p=json.load(open('/tmp/strata-sycl-verifier-native-check.json'));p['default_adaptive64']=dict(flags=flags,log=log,comparison=comparison,output_ids=ids,environment='Default short-window recording and completed adaptive boundary: both control variables unset');Path('/tmp/strata-sycl-verifier-native-check.json').write_text(json.dumps(p,indent=2)+'\n')
print('Default adaptive64: all full-window logits bitwise equal',flush=True)
