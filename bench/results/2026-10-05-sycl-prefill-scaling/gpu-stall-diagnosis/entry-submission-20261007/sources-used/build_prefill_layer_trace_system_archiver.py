"""Build the diagnostic-only prefill layer-range trace and freeze its executable."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = base/'prefill-layer-trace-build-system-archiver';out.mkdir(mode=0o700)
source = root/'sycl/src/prefill/prefill.cpp'
env = dict(os.environ, PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',
           LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib:/opt/intel/oneapi/umf/1.1/lib:/opt/intel/oneapi/mkl/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_PRELOAD',None)
argv = ['/usr/bin/ninja','-C','build-sycl-upstream-jit','strata']
record = {'scope':'Build-only host-archiver correction and addition of zero-based half-open layer range to prefill chunk start trace; arithmetic unchanged; no GPU execution',
          'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'argv':argv,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'build_environment':{k:env[k] for k in ['PATH','LD_LIBRARY_PATH']}}
try:
    cache=(root/'build-sycl-upstream-jit/CMakeCache.txt').read_text()
    assert 'STRATA_NATIVE_POOL_TASK_FACTOR:STRING=0' in cache
    record['native_pool_task_factor']=0
    def objects():
        return {str(p.relative_to(root/'build-sycl-upstream-jit')):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root/'build-sycl-upstream-jit').rglob('*.o') if p.is_file()}
    record['objects_before_configure'] = objects()
    configured = ['/usr/bin/cmake','-S','sycl','-B','build-sycl-upstream-jit','-DCMAKE_AR:FILEPATH=/usr/bin/ar','-DCMAKE_RANLIB:FILEPATH=/usr/bin/ranlib']
    record['configure_argv'] = configured
    with (out/'configure.stdout').open('wb') as stdout,(out/'configure.stderr').open('wb') as stderr:
        result=subprocess.run(configured,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=60)
    record['configure_exit_code']=result.returncode
    assert result.returncode==0
    with (out/'build.stdout').open('wb') as stdout,(out/'build.stderr').open('wb') as stderr:
        run=subprocess.run(argv,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=180)
    record['exit_code']=run.returncode
    record['objects_after_build']=objects()
    record['object_bytes_unchanged_during_archiver_reconfiguration']=record['objects_before_configure']==record['objects_after_build']
    assert record['object_bytes_unchanged_during_archiver_reconfiguration']
    assert run.returncode==0
    target=base/'strata-prefill-layer-trace-candidate'
    assert not target.exists()
    shutil.copyfile(root/'build-sycl-upstream-jit/strata',target)
    target.chmod(0o700)
    record['binary_path']=str(target)
    record['binary_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    record['passed']=True
except BaseException as error:
    record['error']=repr(error)
    raise
finally:
    record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
