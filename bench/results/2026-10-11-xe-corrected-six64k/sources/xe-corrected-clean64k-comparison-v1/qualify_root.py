import ast,fcntl,hashlib,json,math,re
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');HERE=Path(__file__).resolve().parent
class M:
 @staticmethod
 def require(ok,message):
  if not ok:raise AssertionError(message)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import protocol
 tree=ast.parse((HERE/'run_root.py').read_text());nodes=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in ['validate','runtime_findings'] or isinstance(v,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='RUNTIME_ERROR' for x in v.targets)]
 env=dict(m=M,math=math,re=re);exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-controller','exec'),env)
 def lines(count,reason='length'):
  return ['RESUME 0','REUSED 0']+['PP '+str(v)+' 65536 18000 227' for v in list(range(4096,65536,4096))+[65535]]+sum(([f'T {i}',f'LP -0.1 {i}:-0.1 1:-4 2:-5 3:-6 4:-7'] for i in range(count)),[])+[f'DONE {count} 65536 38000 220 {reason} 0 0 0 0 0 0 0 0 65536 0']
 good=lines(64);v=env['validate'](good);assert v['decode_performance_eligible'] and v['generated_tokens']==64
 early=env['validate'](lines(3,'stop'));assert early['fresh_complete_finite'] and not early['decode_performance_eligible'] and early['decode_tps'] is None
 cases=2
 for bad in [[x for x in good if not x.startswith('PP 65535')],[x for i,x in enumerate(good) if i!=good.index(next(v for v in good if v.startswith('LP ')))],[x.replace('DONE 64 ','DONE 63 ') for x in good],[x.replace('PP 4096 65536','PP 4096 65535') for x in good],[x.replace('RESUME 0','RESUME 1') for x in good],[x.replace('LP -0.1 ','LP nan ') for x in good]]:
  try:env['validate'](bad)
  except RuntimeError:cases+=1
  else:raise AssertionError('expected negative protocol refusal')
 for bad in ['UR_RESULT_ERROR_OUT_OF_RESOURCES','ZE_RESULT_ERROR_DEVICE_LOST','[ERROR] command failed','GPU completion unconfirmed']:
  assert env['runtime_findings'](bad);cases+=1
 assert not env['runtime_findings']('ZE_RESULT_NOT_READY');cases+=1
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha(HERE/'direct_owner.py')=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686'
 r=dict(passed=True,gpu_executed=False,model_executed=False,controller_sha256=sha(HERE/'run_root.py'),protocol_sha256=sha(HERE/'protocol.py'),owner_sha256=sha(HERE/'direct_owner.py'),qualifier_sha256=sha(Path(__file__)),protocol_and_error_groups=cases,positive=v,early_stop=early,scope='Actual common protocol positive/negative tests and runtime error classification. Existing owner bytes unchanged and prior 5 ownership cases retained; no new GPU safety claim.')
 (HERE/'cpu-qualification.json').write_text(json.dumps(r,indent=2)+'\n');print('common64K controller qualification PASS',cases)
