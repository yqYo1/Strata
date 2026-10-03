import json,os,subprocess,re
from pathlib import Path
base=json.load(open('/tmp/strata-sycl-mtp-serve.json'))['flags'].copy();base.remove('--serve')
i=base.index('--tokens');base[i:i+2]=['--tokens-file','/tmp/strata-sycl-writing-tokens.txt'];base+=['--stats','--check-logits','--stop-eos']
ids=json.load(open('bench/results/2026-10-03-sycl-q5-head-batch/run.json'))['output_ids']
Path('/tmp/strata-sycl-verify-overlap-oracle-128.txt').write_text(','.join(map(str,ids))+'\n')
env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='level_zero:gpu');env.pop('STRATA_TRACE',None)
for label in ['split8','position0']:
 args=base.copy()
 if label=='split8':
  for flag in ['--mtp','--spec-min-p']:
   i=args.index(flag);del args[i:i+2]
  args[args.index('--spec')+1]='8';args[args.index('--max-new')+1]='32'
  args+=['--spec-split','--spec-oracle','/tmp/strata-sycl-verify-overlap-oracle-128.txt','--spec-corrupt','3']
 else:
  i=args.index('--tokens-file');args[i:i+2]=['--tokens','1']
  args[args.index('--prefill')+1]='0';args[args.index('--max-new')+1]='8'
 dump=f'/tmp/strata-sycl-verifier-native-{label}-windows.bin';Path(dump).unlink(missing_ok=True)
 e=dict(env,STRATA_SYCL_VERIFY_NATIVE_CAPTURE='1',STRATA_SYCL_VERIFY_LOGITS=dump)
 log=f'/tmp/strata-sycl-verifier-native-{label}.log'
 with open(log,'w') as f:subprocess.run(['./build-sycl/strata']+args,env=e,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
 output=list(map(int,re.search(r'^output  : (.*)$',Path(log).read_text(),re.M)[1].split()))
 if label=='split8':assert output==ids[:32]
 ref=f'/tmp/strata-sycl-verify-overlap-{label}-after-windows.bin'
 comparison=subprocess.check_output(['/home/yayoi/.local/share/strata-sycl/venv/bin/python','tools/sycl/compare_verify_logits.py',ref,dump,'--require-exact'],text=True)
 p=json.load(open('/tmp/strata-sycl-verifier-native-check.json'));p[label]=dict(flags=args,log=log,comparison=json.loads(comparison),output_ids=output);Path('/tmp/strata-sycl-verifier-native-check.json').write_text(json.dumps(p,indent=2)+'\n')
 print(label,'all window logits bitwise equal',flush=True)

args=base.copy();args[args.index('--max-new')+1]='8'
log='/tmp/strata-sycl-verifier-native-withheld.log'
with open(log,'w') as f:proc=subprocess.run(['./build-sycl/strata']+args,env=dict(env,STRATA_SYCL_VERIFY_NATIVE_CAPTURE='1',STRATA_TEST_VERIFY_STALL='1'),stdout=f,stderr=subprocess.STDOUT,timeout=240)
text=Path(log).read_text();assert proc.returncode==1 and 'host results withheld' in text and 'GPU finished' in text
p=json.load(open('/tmp/strata-sycl-verifier-native-check.json'));p['withheld']=dict(flags=args,log=log,exit_code=proc.returncode,gpu_drained_before_error=True);Path('/tmp/strata-sycl-verifier-native-check.json').write_text(json.dumps(p,indent=2)+'\n')
print('Withheld host results fail after draining native GPU graphs',flush=True)
