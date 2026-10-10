from pathlib import Path
import hashlib,json,fcntl,types,sys,os,math
B=Path(__file__).parent
repeat=int(sys.argv[1]);assert repeat in (1,2,3)
def ident(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 buildpath=B/'cpu-fma-capacity-build-v1/record.json';build=json.loads(buildpath.read_text());assert build['passed'] and build['complete'] and not build['active']
 src=Path(build['source']['path']);binary=Path(build['binary']['path']);assert ident(src)=={k:build['source'][k] for k in ('bytes','sha256')};assert ident(binary)=={k:build['binary'][k] for k in ('bytes','sha256')}
 parent=B/'run_gdn_gate_factor_probe_v2.py';assert ident(parent)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 source=parent.read_text().split('\ndef parse_probe(',1)[0];m=types.ModuleType('fmacapacity');exec(compile(source,str(parent),'exec'),m.__dict__)
 out=B/f'cpu-fma-capacity-v1-r{repeat}';assert not out.exists();out.mkdir();m.W=out;o=m.Owner(out)
 env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8');args=[str(binary),'0.99999904632568359375','0.00000095367431640625'];r=dict(active=True,complete=False,passed=False,repeat=repeat,source=build['source'],binary=build['binary'],build_receipt=dict(path=str(buildpath),**ident(buildpath)),controller=ident(Path(__file__)),parent_owner=ident(parent),commands=o.commands,environment=env,argv=args,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cpu_policy={str(p):p.read_text().strip() for name in ['scaling_governor','scaling_min_freq','scaling_max_freq','energy_performance_preference'] for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/'+name)},gpu_work_submitted=False,model_inference=False,adopted=False,scope='Register-only FP32 AVX2 FMA attained capacity; distinct from packed quantized dot instruction/DRAM capacity;1/6 pinned physicalcores,1warm+7measure perprocess percorecount')
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 o.persist=save
 try:
  command,stdout,stderr=o.run('fma',[*args],env,wall=120);r['output_rows']=[json.loads(x) for x in stdout.read_text().splitlines()];r['stderr']=ident(stderr);save();assert m.completed(command) and command['exit_code']==0
  rows=r['output_rows'];assert len(rows)==17 and rows[-1]['success'];samples=rows[:-1];assert {(x['threads'],x['sample']) for x in samples}=={(c,s) for c in (1,6) for s in range(-1,7)}
  for x in samples:
   assert x['full_reference_pass'] and x['affinity_pass'] and x['iterations_per_thread']==64000000 and x['independent_accumulators']==12 and x['FP32_lanes']==8 and x['operations_per_fma']==2
   assert math.isfinite(x['seconds']) and x['seconds']>0 and math.isclose(x['GFLOPs'],64000000*12*8*2*x['threads']/x['seconds']/1e9,rel_tol=1e-12)
  assert ident(binary)=={k:build['binary'][k] for k in ('bytes','sha256')};assert ident(src)=={k:build['source'][k] for k in ('bytes','sha256')};r.update(passed=True,complete=True)
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:r['active']=o.active is not None;save();print(json.dumps({k:r.get(k) for k in ['repeat','passed','complete','active','error']}))
 if not r['passed']:raise SystemExit(1)
