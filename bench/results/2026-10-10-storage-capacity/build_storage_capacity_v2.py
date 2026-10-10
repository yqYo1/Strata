from pathlib import Path
import fcntl,hashlib,json,subprocess,types,traceback
B=Path(__file__).parent
out=B/'storage-capacity-v2-build';src=B/'storage_capacity_v2.cpp';parent=B/'run_gdn_gate_factor_probe_v2.py'
def ident(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not out.exists();assert ident(src)['sha256']=='5d32e4fb69adae2c85cdc48967132a7297bea8e8e2e8d7f5adc866f2fec280e9';assert ident(parent)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f';out.mkdir()
 source=parent.read_text().split('\ndef parse_probe(',1)[0];m=types.ModuleType('storagebuild');exec(compile(source,str(parent),'exec'),m.__dict__);m.W=out;o=m.Owner(out);binary=out/'storage-capacity'
 r=dict(active=True,complete=False,passed=False,source=dict(path=str(src),**ident(src)),controller=ident(Path(__file__)),parent_owner=ident(parent),commands=o.commands,compiler=subprocess.check_output(['/usr/bin/g++','--version'],text=True).splitlines()[0],gpu_work_submitted=False,model_inference=False,adopted=False,source_review='Rootfull source-only handoff+source read; v1 originalkept unchanged. v2 aligns sequentialstart to1MiB toavoid unnecessarycrossrecordboundary vs4KiB start beforeanyfixture run.',memory_scope='Sourceownedbuffers/stacks/metadata<256MiB; separate runtimeRSS bound.')
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 o.persist=save;env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
 try:
  c,_,_=o.run('compile',['/usr/bin/g++','-std=c++20','-O3','-pthread','-Wall','-Wextra','-Wpedantic',str(src),'-o',str(binary)],env,wall=120);assert m.completed(c) and c['exit_code']==0,c
  d,so,_=o.run('dependencies',['/usr/bin/readelf','-d',str(binary)],env,wall=30);assert m.completed(d) and d['exit_code']==0;dependencies=so.read_text();assert not any(x in dependencies for x in ['libsycl','libze_loader','libur_adapter','libcuda','libhip']);r['dependencies']=dependencies
  assert ident(src)=={k:r['source'][k] for k in ('bytes','sha256')};r['binary']=dict(path=str(binary),**ident(binary));r.update(passed=True,complete=True)
 except BaseException as e:r['error']=type(e).__name__+': '+str(e);r['traceback']=traceback.format_exc()
 finally:r['active']=o.active is not None;save();print(json.dumps({k:r.get(k) for k in ['passed','complete','active','error','binary']}))
 if not r['passed']:raise SystemExit(1)
