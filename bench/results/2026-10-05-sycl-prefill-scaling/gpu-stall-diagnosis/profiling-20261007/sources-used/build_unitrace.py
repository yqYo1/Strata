"""Build a private Intel PTI/unitrace without installing system components."""
from pathlib import Path
import hashlib
import json
import os
from profile_supervisor import run

b=Path(__file__).parent
source=b/'pti-gpu-profiler-source'
out=b/'unitrace-build';out.mkdir(mode=0o700)
build=out/'build'
env=dict(os.environ)
compiler=Path('/opt/intel/oneapi/compiler/2026.1')
env['LD_LIBRARY_PATH']=str(compiler/'lib')+':/usr/lib/x86_64-linux-gnu'
args=['/usr/bin/cmake','-S',str(source/'tools/unitrace'),'-B',str(build),'-G','Ninja',
      '-DCMAKE_BUILD_TYPE=Release','-DBUILD_WITH_MPI=OFF','-DBUILD_WITH_ITT=0',
      '-DBUILD_WITH_OMP=0','-DBUILD_WITH_OPENCL=0','-DBUILD_WITH_PERFETTO=OFF',
      '-DBUILD_WITH_L0=1','-DBUILD_WITH_XPTI=1','-DONEAPI_COMPILER_HOME='+str(compiler),
      '-DXptifw_INCLUDE_DIR='+str(compiler/'include'),
      '-DXptifw_LIBRARY='+str(compiler/'lib/libxptifw.so')]
record={'scope':'Private unitrace build; Level Zero and XPTI enabled; no metric collection, package installation or GPU workload',
        'source_commit':'6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033','passed':False}
try:
    record['configure']=run(args,out/'configure',env,seconds=180)
    assert record['configure']['exit_code']==0
    record['build']=run(['/usr/bin/cmake','--build',str(build),'--parallel','4'],out/'compile',env,seconds=600)
    assert record['build']['exit_code']==0
    record['binaries']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [build/'unitrace',build/'libunitrace_tool.so']}
    record['passed']=True
finally:
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['configure','build']},indent=2))
