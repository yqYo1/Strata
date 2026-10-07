"""Build without imposing the GPU runtime's library path on Nix build tools."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = base/'prefill-expert-wait-build';out.mkdir(mode=0o700)
env = dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
env.update(MKLROOT='/opt/intel/oneapi/mkl/2026.1',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
def objects():
    return {str(p.relative_to(root/'build-sycl-upstream-jit')):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root/'build-sycl-upstream-jit').rglob('*.o') if p.is_file()}
argv = ['/usr/bin/ninja','-C','build-sycl-upstream-jit','strata']
record = {'scope':'Build-only default-off direct-FP16 expert submission backpressure; only prefill translation unit changes; no GPU execution or arithmetic change',
          'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,
          'source_sha256':hashlib.sha256((root/'sycl/src/prefill/prefill.cpp').read_bytes()).hexdigest(),
          'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'build_environment':{'PATH':env['PATH'],'LD_LIBRARY_PATH':None,'MKLROOT':env['MKLROOT'],'LIBRARY_PATH':env['LIBRARY_PATH']},
          'objects_before_build':objects()}
try:
    dry=subprocess.run(argv[:3]+['-n','strata'],cwd=root,env=env,capture_output=True,text=True,timeout=5)
    record['dry_run_stdout']=dry.stdout
    assert dry.returncode==0 and 'src/prefill/prefill.cpp.o' in dry.stdout and 'Linking CXX executable strata' in dry.stdout
    assert sum('Building CXX object' in line for line in dry.stdout.splitlines())==1
    assert 'STRATA_NATIVE_POOL_TASK_FACTOR:STRING=0' in (root/'build-sycl-upstream-jit/CMakeCache.txt').read_text()
    record['native_pool_task_factor']=0
    with (out/'build.stdout').open('wb') as stdout,(out/'build.stderr').open('wb') as stderr:
        run=subprocess.run(argv,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=180)
    record['exit_code']=run.returncode
    record['objects_after_build']=objects()
    record['changed_objects']=[k for k,v in record['objects_before_build'].items() if record['objects_after_build'].get(k)!=v]
    assert run.returncode==0 and len(record['changed_objects'])==1 and record['changed_objects'][0].endswith('src/prefill/prefill.cpp.o')
    assert set(record['objects_before_build'])==set(record['objects_after_build'])
    target=base/'strata-prefill-expert-wait-candidate';assert not target.exists()
    shutil.copyfile(root/'build-sycl-upstream-jit/strata',target);target.chmod(0o700)
    record['binary_path']=str(target);record['binary_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    record['passed']=True
except BaseException as error:
    record['error']=repr(error)
    raise
finally:
    record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if not k.startswith('objects_')},indent=2))
