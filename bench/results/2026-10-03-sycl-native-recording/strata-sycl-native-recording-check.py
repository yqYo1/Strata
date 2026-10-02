import json,os,subprocess,re
from pathlib import Path
p=json.load(open('bench/results/2026-10-03-sycl-q4k-esimd/run.json'));base=p['model_flags']+['--tokens-file','/tmp/strata-sycl-writing-tokens.txt']+p['common_flags']
for k,v in [('--max-new','32'),('--expert-cache','3038'),('--ple-io','direct')]:base[base.index(k)+1]=v
log='/tmp/strata-sycl-native-recording-cold32.log';dump='/tmp/strata-sycl-native-recording-cold32.bin';flags=base+['--dump-logits',dump]
env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='level_zero:gpu',STRATA_SYCL_NATIVE_RECORDING='1');env.pop('STRATA_TRACE',None);env.pop('STRATA_SYCL_VERIFY_LOGITS',None)
with open(log,'w') as f:subprocess.run(['./build-sycl/strata']+flags,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
comparison=subprocess.check_output(['/home/yayoi/.local/share/strata-sycl/venv/bin/python','tools/sycl/compare_logits.py','/tmp/strata-sycl-post-route-cold32.bin',dump,'--require-exact'],text=True)
result=dict(cold32=dict(flags=flags,log=log,comparison=json.loads(comparison)));Path('/tmp/strata-sycl-native-recording-check.json').write_text(json.dumps(result,indent=2)+'\n');print('native ordinary32 full logits exact',flush=True)
