import os,json,subprocess
from pathlib import Path
rows=[]
for floor in ['0.5','0.7','0.9','0.99']:
 subprocess.run(['python3','/tmp/strata-sycl-mtp-floor-serve.py'],env=dict(os.environ,STRATA_BENCH_FLOOR=floor,STRATA_DECODE_TIMING='1'),check=True,timeout=360)
 row=json.load(open(f'/tmp/strata-sycl-mtp-floor-{floor}-trial1.json'));row['floor']=floor;assert row['exit_code']==0
 rows.append(row);Path('/tmp/strata-sycl-mtp-floor-screen.json').write_text(json.dumps(dict(runs=rows),indent=2)+'\n')
 print('floor',floor,[round(r['decode_tok_s'],2) for r in row['runs'][:4]],flush=True)
