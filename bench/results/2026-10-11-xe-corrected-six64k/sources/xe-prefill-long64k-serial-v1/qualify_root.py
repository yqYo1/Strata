import ast,fcntl,hashlib,json,math,re
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');HERE=Path(__file__).resolve().parent;p=HERE/'run_root.py'
class M:
 @staticmethod
 def require(ok,message):
  if not ok:raise AssertionError(message)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);env=dict(m=M,math=math,re=re);tree=ast.parse(p.read_text())
 nodes=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in ['validate','runtime_findings','validate_trace'] or isinstance(v,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='RUNTIME_ERROR' for x in v.targets)]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
 lines=['RESUME 0','REUSED 0']+['PP '+str(v)+' 65536 18000 227' for v in list(range(4096,65536,4096))+[65535]]+['T 24','LP -0.1 24:-0.1 1:-4 2:-5 3:-6 4:-7','DONE 1 65536 38000 220 length 0 0 0 0 0 0 0 0 65536 0'];pos=env['validate'](lines);cases=1
 for bad in [[v for v in lines if not v.startswith('PP 65535')],[v.replace('PP 4096 65536','PP 4096 65535') for v in lines],[v for v in lines if not v.startswith('LP ')],[v.replace('DONE 1 ','DONE 2 ') for v in lines]]:
  try:env['validate'](bad)
  except AssertionError:cases+=1
  else:raise AssertionError('expected protocol refusal')
 trace=''.join('strata trace: prompt chunk '+str(v)+' of 65535\n' for v in range(0,65535,4096));env['validate_trace'](trace);cases+=1
 for bad in [trace.replace('4096','4095'),trace+'UR_RESULT_ERROR_OUT_OF_RESOURCES\n',trace+'ZE_RESULT_ERROR_DEVICE_LOST\n',trace+'[ERROR] bad operation\n',trace+'GPU completion unconfirmed\n']:
  try:env['validate_trace'](bad)
  except AssertionError:cases+=1
  else:raise AssertionError('expected trace refusal')
 assert not env['runtime_findings']('ZE_RESULT_NOT_READY\n');cases+=1
 (HERE/'cpu-qualification.json').write_text(json.dumps(dict(passed=True,gpu_executed=False,model_executed=False,controller_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),qualifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_and_trace_groups=cases,positive_scope=pos),indent=2)+'\n');print('controller qualification passed',cases)
