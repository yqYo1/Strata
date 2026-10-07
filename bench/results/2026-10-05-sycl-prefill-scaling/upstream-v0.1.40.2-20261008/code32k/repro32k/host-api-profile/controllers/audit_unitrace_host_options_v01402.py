"""Check the pinned collector's actual option derivation without GPU work."""
from pathlib import Path
import datetime,hashlib,json,subprocess,time

base=Path(__file__).parent
source=base/'pti-gpu-profiler-source'
out=base/'unitrace-host-options-v01402';out.mkdir(mode=0o700)
record={'active':True,'passed':False,'scope':'CPU-only check of the pinned unitrace collector options and generated callback guards. No GPU workload, profiler activation or global/system change. Host timing/call trace must not enable kernel tracing or metrics; option-free collection intentionally differs.','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':[]}
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,argv):
 start=time.monotonic()
 with (out/(label+'.stdout')).open('wb') as stdout,(out/(label+'.stderr')).open('wb') as stderr:
  result=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=60)
 record['steps'].append({'label':label,'argv':list(map(str,argv)),'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start});save();assert result.returncode==0
save()
try:
 record['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
 assert record['source_commit']=='6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033'
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True)
 build=json.loads((base/'unitrace-build-system-cc/record.json').read_text());assert build['passed'] and build['source_commit']==record['source_commit']
 files=['tools/unitrace/src/collector_options.h','tools/unitrace/scripts/gen_tracing_callbacks.py','tools/unitrace/src/levelzero/ze_collector.h','tools/unitrace/src/unitrace.cc','tools/unitrace/src/tracer.cc']
 record['source_sha256']={name:digest(source/name) for name in files}
 generator=(source/files[1]).read_text();collector=(source/files[2]).read_text()
 assert 'if (collector->options_.kernel_tracing)' in generator
 assert 'profiling_desc->flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;' in collector
 record['observation']='Generated callback invokes the event-pool/append instrumentation only when kernel_tracing is set. Its event-pool callback adds timestamp/host-visible flags. The chosen host-only options leave kernel_tracing false; this is source/option evidence, not proof of model/runtime correctness.'
 cpp=out/'options.cpp'
 cpp.write_text('''#include <cstdint>
#include <iostream>
#include "collector_options.h"
int main() {
    CollectorOptions host;
    host.host_timing = 1;
    host.chrome_call_logging = 1;
    host.log_to_file = 1;
    host.output_dir_path = 1;
    host.DeriveFlags();
    std::cout << "host-only api=" << host.api_tracing
              << " kernel=" << host.kernel_tracing
              << " device=" << host.device_timing
              << " metric=" << host.metric_query << "\\n";
    if (!host.api_tracing || host.kernel_tracing || host.device_timing ||
        host.device_timeline || host.kernel_submission || host.chrome_kernel_logging ||
        host.chrome_device_logging || host.metric_query || host.metric_stream ||
        host.stall_sampling) return 1;
    CollectorOptions bare;
    bare.DeriveFlags();
    std::cout << "option-free api=" << bare.api_tracing
              << " kernel=" << bare.kernel_tracing
              << " device=" << bare.device_timing << "\\n";
    if (!bare.host_timing || !bare.device_timing || !bare.kernel_tracing) return 2;
}
''')
 run('compile',['/usr/bin/g++','-O2','-std=c++17','-I'+str(source/'tools/unitrace/src'),str(cpp),'-o',str(out/'options')])
 run('options',[str(out/'options')])
 record['profiler_sha256']={str(p):digest(p) for p in [base/'unitrace-build-system-cc/build/unitrace',base/'unitrace-build-system-cc/build/libunitrace_tool.so']}
 record['controller_sha256']=digest(Path(__file__))
 record['passed']=True
except BaseException as error:record['error']=repr(error);raise
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps(record,indent=2))
