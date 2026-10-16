#!/usr/bin/env python3
"""Attribute visible GPU gaps to recorded API spans without calling them causes."""
import collections
import fcntl
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,merged,length,overlap,require,sha


def main():
    target=A/'prefill-window-gap-api-summary-v1.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not target.exists(),'fresh summary')
        closed=json.loads((A/'prefill-window-interval-summary-host-v2.json').read_text())
        require(closed['passed'],'independent timing crosscheck')
        summary=A/'prefill-window-long-interval-summary-v2.json'
        d=json.loads(summary.read_text())
        trace=Path(d['trace']['path'])
        gpu=[]; api=collections.defaultdict(list); counts=collections.Counter()
        digest=hashlib.sha256();events=0
        origin=d['relative_origin_trace_timestamp_us']
        with trace.open('rb') as f:
            for raw in f:
                digest.update(raw)
                line=raw.strip().rstrip(b',')
                if line==b'{ "traceEvents":[' or not line.startswith(b'{'):
                    continue
                e=json.loads(line);events+=1
                if e['ph']!='X': continue
                start=e['ts']-origin; end=start+e['dur']
                if e['cat']=='gpu_op': gpu.append((start,end))
                else:
                    require(e['cat']=='cpu_op','known timed category')
                    api[e['name']].append((start,end));counts[e['name']]+=1
        require(digest.hexdigest()==d['trace']['sha256'] and events==d['trace']['events'],
                'complete raw trace identity/count')
        gpu=merged(gpu)
        require(abs(length(gpu)-d['GPU_busy_union_us'])<0.001,'independent GPU interval union')
        gaps=[(gpu[i-1][1],gpu[i][0]) for i in range(1,len(gpu)) if gpu[i][0]>gpu[i-1][1]]
        require(abs(length(gaps)-d['GPU_gap_within_span_us'])<0.001,'independent GPU gap union')
        rows=[]
        for name,intervals in api.items():
            union=merged(intervals)
            rows.append(dict(name=name,calls=counts[name],union_us=length(union),
                overlap_visible_GPU_us=overlap(union,gpu),overlap_GPU_gap_us=overlap(union,gaps)))
        cpu=merged(x for v in api.values() for x in v)
        gap_api=overlap(cpu,gaps)
        r=dict(passed=True,GPU_executed=False,recipe_sha256=sha(Path(__file__)),
            summary_sha256=sha(summary),trace=d['trace'],
            GPU_gap_union_us=length(gaps),GPU_gap_overlapping_recorded_API_us=gap_api,
            GPU_gap_without_recorded_API_us=length(gaps)-gap_api,
            API_spans=sorted(rows,key=lambda x:x['overlap_GPU_gap_us'],reverse=True),
            scope='Within-trace intersection only, prefill+restore+one output. An overlapping API can include provider/profiler overhead and is not itself proof of a production bottleneck. Unrecorded host work, SYCL work and instrumentation cannot be separated by this trace.',
            clean_performance_qualified=False)
        with target.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=True,GPU_gap_seconds=length(gaps)/1e6,
            gap_with_API_seconds=gap_api/1e6,gap_without_API_seconds=(length(gaps)-gap_api)/1e6)))


if __name__=='__main__':main()
