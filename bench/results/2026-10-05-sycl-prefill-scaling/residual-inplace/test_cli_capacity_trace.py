"""Reject incomplete CLI capacity witnesses using CPU-only protocol stubs."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
out = Path(sys.argv[1]); out.mkdir(); out.chmod(0o700)
stub = out / 'fake-engine'
stub.write_text('''#!/usr/bin/python3
import array,json,os,sys
from pathlib import Path
a=sys.argv;get=lambda k:a[a.index(k)+1]
c=int(get('--max-context'));mode=os.environ['CAPACITY_STUB_MODE']
def head():
 p=Path(os.environ['STRATA_DUMP_FIRST_LOGITS']);p.write_bytes((array.array('f',[.5])*248320).tobytes())
def windows(n,new):
 w=[(n-1,1),(n,min(4,c-n))]
 if mode=='missing':w=[]
 if mode=='short':w[-1]=(n,1)
 if mode=='overrun':w[-1]=(n,3)
 if mode=='wrong-start':w[0]=(n,1)
 for p,t in w:print(f'strata trace: window {p} {t}',file=sys.stderr,flush=True)
if '--serve' in a:
 print('READY stub',flush=True)
 for line in sys.stdin:
  if line.startswith('QUIT'):break
  words=line.split();new=int(words[1]);n=len(words[-1].split(','))
  if n+new>c:print('ERR prompt exceeds context',flush=True);continue
  head();windows(n,new);print('REUSED 0',flush=True)
  for i in range(new):print(f'T {100+i}\\nLP -0.5 {100+i}:-0.5',flush=True)
  print(f'DONE {new} {n} 1.0 1.0 length 0 0 0 0 0 0 0 0.0 {n} 0',flush=True)
else:
 n=len(Path(get('--tokens-file')).read_text().split());new=int(get('--max-new'))
 if n+new>c:print('positive --max-new and --max-context must fit the prompt and generation');sys.exit(2)
 head();windows(n,new);print(f'prefill {n-1} tokens');print('output : '+' '.join(str(100+i) for i in range(new)))
''')
stub.chmod(0o755)
controller = Path(__file__).with_name('check_full_context.py')
record = {'scope': 'CPU stub only: verifies trace/cardinality parsing, not physical GPU capacity',
          'controller_sha256': hashlib.sha256(controller.read_bytes()).hexdigest(),
          'cases': [], 'passed': False}
(out/'controller.py').write_bytes(controller.read_bytes())
cases = [('valid',64,'cli',False,True),('valid',262144,'cli',True,True),
         ('missing',64,'cli',False,False),('short',64,'cli',False,False),
         ('overrun',64,'cli',False,False),('wrong-start',64,'cli',False,False),
         ('valid',64,'boundary',True,True),('valid',64,'serve',True,True),
         ('valid',262144,'serve',True,True)]
try:
 for i,(mode,c,stage,diag,expected) in enumerate(cases):
  d=out/f'{i}-{stage}-{mode}-{c}';d.mkdir()
  (d/'coding-context-256k-tokens.txt').write_text(' '.join(['1']*c)+'\n')
  (d/'reference.json').write_text(json.dumps({'runs':[{'args':['--pack','stub','--native','stub','--expert-profile','stub']}],'env':{}}))
  envfile=d/'environment.json';envfile.write_text(json.dumps({'PATH':'/usr/bin:/bin','LD_LIBRARY_PATH':'/usr/lib/x86_64-linux-gnu','CAPACITY_STUB_MODE':mode}))
  argv=['/usr/bin/python3',str(controller),'--stage',stage,'--context',str(c),'--recovery',str(d),'--executable',str(stub),'--environment-file',str(envfile),'--job-timeout','5','--protocol-timeout','5','--shutdown-timeout','.1']
  if diag:argv.append('--diagnostics')
  p=subprocess.run(argv,cwd=root,env=dict(os.environ,PATH='/usr/bin:/bin'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=12)
  (d/'controller.stdout').write_bytes(p.stdout);(d/'controller.stderr').write_bytes(p.stderr)
  summary=next(d.glob('*/summary.json'));r=json.loads(summary.read_text())
  case={'mode':mode,'context':c,'stage':stage,'diagnostics':diag,'expected':expected,'exit_code':p.returncode,'completed':r['completed'],'summary':str(summary),'passed':False};record['cases'].append(case)
  assert r['completed']==expected and (p.returncode==0)==expected,case
  assert all(not x['still_alive'] for x in r['processes']),case
  assert r['env']['STRATA_TRACE']=='1'
  if diag:assert r['env']['ZEL_LOADER_LOGGING_LEVEL']=='trace' and r['env']['UR_ENABLE_LAYERS']=='UR_LAYER_TRACING'
  if expected and stage=='cli':
   assert r['runs'][0]['last_executed_kv_cell']==c-1
   assert r['runs'][0]['verify_windows']==[[c-3,1],[c-2,2]]
  case['passed']=True
  (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
 record['passed']=True
except BaseException as error:
 record['error']=repr(error);raise
finally:(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
