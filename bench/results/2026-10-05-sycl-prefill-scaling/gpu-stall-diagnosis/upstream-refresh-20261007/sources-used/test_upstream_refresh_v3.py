"""Offline upstream-refresh build and existing host tests; no GPU invocation."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import time

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=base/'upstream-refresh-20261007'
mode=sys.argv[1]
assert mode in ('build','serve','setup')
env=dict(os.environ, PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',
         MKLROOT='/opt/intel/oneapi/mkl/2026.1',
         LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
record=dict(mode=mode,active=True,passed=False,steps=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            scope='Existing host tests or isolated SYCL compilation only; no GPU model/parity/performance/capacity proof')
start=time.monotonic()
def save():
    record['elapsed_seconds']=time.monotonic()-start
    (out/(mode+'-record.json')).write_text(json.dumps(record,indent=2)+'\n')
def run(label,argv,timeout):
    item=dict(label=label,argv=argv);record['steps'].append(item);save();began=time.monotonic()
    with (out/(label+'.stdout')).open('wb') as stdout,(out/(label+'.stderr')).open('wb') as stderr:
        result=subprocess.run(argv,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=timeout)
    item.update(exit_code=result.returncode,elapsed_seconds=time.monotonic()-began);save()
    print(label,'exit',result.returncode,'seconds',round(item['elapsed_seconds'],3),flush=True)
    return result.returncode==0
try:
    if mode=='build':
        build=root/'build-sycl-refresh-20261007'
        ok=run('configure',['/usr/bin/cmake','-S',str(root/'sycl'),'-B',str(build),'-G','Ninja',
            '-DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx',
            '-DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx',
            '-DCMAKE_BUILD_TYPE=Release','-DSTRATA_GGML_DIR=/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned',
            '-DSTRATA_SYCL_PARITY=ON','-DSTRATA_IQ2S_GCC=ON',
            '-DSTRATA_IQ_FIXTURE_DIR=/tmp/strata-upstream-arc-iq-fixtures',
            '-DSTRATA_PLE_FIXTURE_DIR=/tmp/strata-upstream-arc-ple-fixtures'],120)
        record['passed']=ok and run('build',['/usr/bin/cmake','--build',str(build),'--parallel','2','--','-k','0'],1800)
    elif mode=='serve':
        record['passed']=run('serve-host-tests',[str(out/'test-venv/bin/python'),'-m','unittest','discover','-s','serve','-t','.','-p','test_*.py'],900)
    else:
        failed=[]
        files=sorted((root/'tools').glob('test_setup_*.py'))
        for path in files:
            if not run(path.stem,['/usr/bin/python3',str(path)],180):failed.append(path.name)
        record['failed_scripts']=failed;record['passed']=bool(files) and not failed
except BaseException as e:
    record['error']=repr(e)
finally:
    record['active']=False;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(mode,'PASS' if record['passed'] else 'FAIL',flush=True)
sys.exit(0 if record['passed'] else 1)
