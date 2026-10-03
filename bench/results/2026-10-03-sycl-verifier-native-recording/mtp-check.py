import json,os,subprocess,re
from pathlib import Path
base=json.load(open('/tmp/strata-sycl-mtp-serve.json'))['flags'].copy();base.remove('--serve');i=base.index('--tokens');base[i:i+2]=['--tokens-file','/tmp/strata-sycl-writing-tokens.txt']
for k,v in [('--max-new','64'),('--max-context','512')]:base[base.index(k)+1]=v
base+=['--stats','--check-logits','--stop-eos'];dump='/tmp/strata-sycl-verifier-native-mtp64-windows.bin';log='/tmp/strata-sycl-verifier-native-mtp64.log';Path(dump).unlink(missing_ok=True)
env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='level_zero:gpu',STRATA_SYCL_VERIFY_NATIVE_CAPTURE='1',STRATA_SYCL_VERIFY_LOGITS=dump)
with open(log,'w') as f:subprocess.run(['./build-sycl/strata']+base,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
ref='/tmp/strata-sycl-verify-overlap-after-64-windows.bin'
comparison=subprocess.check_output(['/home/yayoi/.local/share/strata-sycl/venv/bin/python','tools/sycl/compare_verify_logits.py',ref,dump,'--require-exact'],text=True)
p={};p['mtp64']=dict(flags=base,log=log,comparison=json.loads(comparison));Path('/tmp/strata-sycl-verifier-native-check.json').write_text(json.dumps(p,indent=2)+'\n');print('native MTP64 all window logits exact',flush=True)
