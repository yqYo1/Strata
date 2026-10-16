import json,os,subprocess,re
from pathlib import Path
base=json.load(open('bench/results/2026-10-03-sycl-verify-overlap/run.json'))['runs'][0]['flags'].copy()
for flag,value in [('--expert-cache','auto'),('--adapt-swaps','64'),('--max-new','64')]:base[base.index(flag)+1]=value
results=[]
for trial in [1,2]:
 dump=f'/tmp/strata-sycl-adapt-boundary-{trial}-64-windows.bin';log=f'/tmp/strata-sycl-adapt-boundary-{trial}-64.log';Path(dump).unlink(missing_ok=True)
 env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='level_zero:gpu',STRATA_SYCL_ADAPT_SYNC='1',STRATA_SYCL_VERIFY_LOGITS=dump);env.pop('STRATA_TRACE',None)
 with open(log,'w') as f:subprocess.run(['./build-sycl/strata']+base,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
 text=Path(log).read_text();ids=list(map(int,re.search(r'^output\s*:\s*(.*)$',text,re.M)[1].split()));results.append(dict(trial=trial,flags=base,log=log,dump=dump,output_ids=ids));print('adaptive deterministic',trial,'64 tokens',flush=True)
assert results[0]['output_ids']==results[1]['output_ids']
comparison=subprocess.check_output(['/home/yayoi/.local/share/strata-sycl/venv/bin/python','tools/sycl/compare_verify_logits.py',results[0]['dump'],results[1]['dump'],'--require-exact'],text=True)
Path('/tmp/strata-sycl-adapt-boundary-check.json').write_text(json.dumps(dict(runs=results,comparison=json.loads(comparison)),indent=2)+'\n');print('identical adaptive settings: all verification window bits exact',flush=True)
