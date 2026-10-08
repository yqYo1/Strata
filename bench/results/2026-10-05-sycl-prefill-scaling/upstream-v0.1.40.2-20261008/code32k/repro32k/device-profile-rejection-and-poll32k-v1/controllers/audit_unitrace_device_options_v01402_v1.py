"""Check the pinned GPU timestamp collector options and ownership before use."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import time

base=Path(__file__).parent
source=base/'pti-gpu-profiler-source'
out=base/'unitrace-device-options-v01402-v1'
out.mkdir(mode=0o700)
record={'active':True,'passed':False,'scope':'CPU-only pinned unitrace timestamp option/source review. No GPU run and no proof of model correctness. No metrics, KMD, sampling, forked target or pause/resume transitions selected.','steps':[],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,argv):
    started=time.monotonic()
    with (out/(label+'.stdout')).open('wb') as stdout,(out/(label+'.stderr')).open('wb') as stderr:
        done=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=60)
    record['steps'].append({'label':label,'argv':argv,'exit_code':done.returncode,'elapsed_seconds':time.monotonic()-started});save()
    assert done.returncode==0
save()
try:
    record['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert record['source_commit']=='6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True)
    prior=json.loads((base/'unitrace-host-options-v01402/record.json').read_text())
    assert prior['passed'] and not prior['active'] and prior['source_commit']==record['source_commit']
    files=list(prior['source_sha256'])+['tools/unitrace/src/levelzero/ze_event_cache.h']
    record['source_sha256']={name:digest(source/name) for name in files}
    assert all(record['source_sha256'][p]==sha for p,sha in prior['source_sha256'].items())
    record['profiler_sha256']=prior['profiler_sha256']
    assert all(digest(Path(p))==sha for p,sha in record['profiler_sha256'].items())
    collector=(source/'tools/unitrace/src/levelzero/ze_collector.h').read_text()
    cache=(source/'tools/unitrace/src/levelzero/ze_event_cache.h').read_text()
    launcher=(source/'tools/unitrace/src/unitrace.cc').read_text()
    assert 'OnEnterCommandListAppendLaunchKernelWithArguments' in collector
    assert 'OnExitCommandListAppendLaunchKernelWithArguments' in collector
    assert 'ZE_STRUCTURE_TYPE_COUNTER_BASED_EVENT_POOL_EXP_DESC' in collector
    assert 'ze_instance_data.in_order_counter_event_ = signal_event;' in collector
    assert 'ZE_FUNC(zeCommandListAppendSignalEvent)(command_list, ze_instance_data.in_order_counter_event_)' in collector
    assert 'if (info == event_info_map_.end()) {\n      return;' in cache
    assert 'if (utils::GetEnv("UNITRACE_KernelMetrics") == "1" || !utils::GetEnv("UNITRACE_ChromeKmdLogging").empty())' in launcher
    assert 'execvp(app_args[0], app_args.data())' in launcher
    record['source_observations']=[
        'Timestamp instrumentation includes the modern LaunchKernelWithArguments API used by this installed UR adapter.',
        'Counter-based event pools bypass timestamp flag rewriting; in-order app counter signals are replaced for profiling by a cached timestamp event and the original counter is signaled afterward on the same command list.',
        'ReleaseEvent returns for events outside the profiler-owned cache. Enter/exit callbacks retain the enter-time instrumentation decision.',
        'No metrics/KMD means the launcher uses execvp without the forked-supervisor branch; existing owned host-only run validated the same wrapper PID/executable transition.',
        'These source facts do not establish absence of profiler defects. All32K heads, live state, outputs and restore continuation must pass; timings with logging/instrumentation are excluded from speed comparisons.'
    ]
    cpp=out/'options.cpp'
    cpp.write_text('''#include <cstdint>
#include <iostream>
#include "collector_options.h"
int main() {
    CollectorOptions o;
    o.device_timing=1; o.chrome_device_logging=1;
    o.log_to_file=1; o.output_dir_path=1; o.DeriveFlags();
    std::cout << "device=" << o.device_timing << " kernel=" << o.kernel_tracing
              << " api=" << o.api_tracing << " metrics=" << o.metric_query << "\\n";
    return !(o.device_timing && o.chrome_device_logging && o.kernel_tracing &&
             !o.api_tracing && !o.host_timing && !o.chrome_call_logging &&
             !o.metric_query && !o.metric_stream && !o.stall_sampling &&
             !o.conditional_collection);
}
''')
    run('compile',['/usr/bin/g++','-O2','-std=c++17','-I'+str(source/'tools/unitrace/src'),str(cpp),'-o',str(out/'options')])
    run('options',[str(out/'options')])
    record['controller_sha256']=digest(Path(__file__))
    record['passed']=True
except BaseException as error:
    record['error']=repr(error)
    raise
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({'passed':record['passed'],'record':str(out/'record.json'),'source_commit':record.get('source_commit')},indent=2))
