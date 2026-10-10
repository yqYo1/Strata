"""Compact actual native kernel/profiling observations from the closed diagnostic."""
from pathlib import Path
from collections import Counter, defaultdict
import fcntl, hashlib, json, re
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
RUN=B/'iq4nl-host-wall-runtime-qualify-v1'
def ident(p):
    with Path(p).open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(bytes=Path(p).stat().st_size,sha256=h)
def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not json.loads((RUN/'record.json').read_text())['active']
        trace_path=RUN/'iq4nl-host-wall.stderr';trace=trace_path.read_text()
        handles={};launches=Counter();groups=defaultdict(Counter);sizes={};profiles=Counter()
        for line in trace.splitlines():
            if 'SUCCESS (ZE_RESULT_SUCCESS) in zeKernelCreate(' in line:
                name=re.search(r'pKernelName="([^"]+)"',line)[1]
                handle=re.search(r'phKernel=(0x[0-9a-f]+)',line)[1]
                assert handle not in handles;handles[handle]=name
            if 'SUCCESS (ZE_RESULT_SUCCESS) in zeKernelSetGroupSize(' in line:
                handle=re.search(r'hKernel=(0x[0-9a-f]+)',line)[1]
                sizes[handle]=tuple(int(re.search(key+r'=(\d+)',line)[1]) for key in ['groupSizeX','groupSizeY','groupSizeZ'])
            if 'SUCCESS (ZE_RESULT_SUCCESS) in zeCommandListAppendLaunchKernel' in line:
                handle=re.search(r'hKernel=(0x[0-9a-f]+)',line)[1];name=handles[handle]
                grid=tuple(int(re.search(key+r'=(\d+)',line)[1]) for key in ['groupCountX','groupCountY','groupCountZ'])
                assert handle in sizes;launches[name]+=1;groups[name][str(grid+sizes[handle])]+=1
            if '<--- urEventGetProfilingInfo(' in line:
                assert line.endswith('-> UR_RESULT_SUCCESS;')
                profiles[re.search(r'\.propName = (UR_PROFILING_INFO_\w+)',line)[1]]+=1
        assert len(handles)==2 and sorted(launches.values())==[384,384]
        for name,count in launches.items():
            assert groups[name]=={'(6400, 1, 1, 32, 1, 1)':384}
        ur=Counter(re.findall(r'-> (UR_RESULT_\w+);',trace))
        ze=Counter(re.findall(r'\((ZE_RESULT_\w+)\) in ',trace))
        assert set(ur)=={'UR_RESULT_SUCCESS'} and set(ze)=={'ZE_RESULT_SUCCESS'}
        assert not profiles, 'unexpected event profiling query in host-wall protocol'
        out=B/'iq4nl-host-wall-qualification-trace-summary-v1.json';assert not out.exists()
        summary=dict(original_log=dict(path=str(trace_path),**ident(trace_path)),controller=ident(__file__),
                     native_create_handles=handles,native_successful_launch_counts=dict(launches),
                     groups_by_kernel={k:dict(v) for k,v in groups.items()},UR_results=dict(ur),ZE_results=dict(ze),
                     successful_profiling_queries=dict(profiles),
                     successful_native_kernel_timestamp_queries=len(re.findall(r'SUCCESS \(ZE_RESULT_SUCCESS\) in zeEventQueryKernelTimestamp\(',trace)),
                     scope='Actual synthetic diagnostic launches and profiling queries; no whole-model route/performance/native ISA claim')
        out.write_text(json.dumps(summary,indent=2)+'\n');print('summary',out,'launches',list(launches.values()),'profiles',dict(profiles),'native_ts',summary['successful_native_kernel_timestamp_queries'])
if __name__=='__main__':main()
