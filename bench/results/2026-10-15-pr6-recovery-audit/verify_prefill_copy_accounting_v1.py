#!/usr/bin/env python3
"""Main source/histogram check; application copy lengths are not wire bytes."""
import collections
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import sha,require


def main():
    path=A/'prefill-window-long-interval-summary-v2.json'
    d=json.loads(path.read_text())
    capacity=A/'prefill-capacity-metadata-v2.json'
    c=json.loads(capacity.read_text())
    require(d['passed'] and c['passed'],'closed source and profile accounting')
    n=d['actual_input_tokens']-1
    require(n==34602,'last prompt token is outside the batched pass')
    chunk=4096; gpu_rows=4096; row=10240*4
    host_chunks=[min(chunk,n-i) for i in range(gpu_rows,n,chunk)]
    residual=collections.Counter()
    for rows in host_chunks:residual[rows*row]+=47
    expert=collections.Counter()
    for layer in c['layers']:expert[layer['expert_bytes']]+=512
    kv=collections.Counter()
    for prefix in range(chunk,n,chunk):
        kv[prefix*512]+=12*2
        kv[prefix*16]+=12*2
    ple=collections.Counter()
    for i in range(0,n,chunk):ple[min(chunk,n-i)*2560*4]+=1
    prefix=collections.Counter({gpu_rows*row:47*2})
    owners={'zeCommandListAppendMemoryCopy(M2D)':{'expert':expert,'host_residual_reload':residual},
            'zeCommandListAppendMemoryCopy(D2M)':{'host_residual_store':residual},
            'zeCommandListAppendMemoryCopy(H2D)':{'KV_prefix_stage':kv,'PLE_rows':ple},
            'zeCommandListAppendMemoryCopy(D2D)':{'GPU_residual_store_reload':prefix}}
    rows=[]
    for operation in d['GPU_operations']:
        if operation['role']!='provider_memory_copy':continue
        name=operation['name']
        require(operation['distinct_sizes']<=32,'complete histogram, not truncated')
        actual=collections.Counter({x['bytes']:x['calls'] for x in operation['size_histogram_top32']})
        require(sum(actual.values())==operation['calls'] and
                sum(size*count for size,count in actual.items())==operation['provider_requested_bytes'],
                'histogram conserves complete requested bytes and calls')
        assigned={}
        for owner,expected in owners[name].items():
            require(all(actual[size]>=count for size,count in expected.items()),'source lengths not found: '+owner)
            actual.subtract(expected)
            assigned[owner]=dict(calls=sum(expected.values()),bytes=sum(size*count for size,count in expected.items()),
                histogram=[dict(bytes=size,calls=count) for size,count in sorted(expected.items())])
        leftover=[dict(bytes=size,calls=count) for size,count in sorted(actual.items()) if count]
        if name.endswith('(H2D)'):require(not leftover,'KV and PLE account every H2D byte')
        rows.append(dict(operation=name,provider_requested_bytes=operation['provider_requested_bytes'],
            classified=assigned,unassigned_calls=sum(x['calls'] for x in leftover),
            unassigned_bytes=sum(x['bytes']*x['calls'] for x in leftover),unassigned_histogram=leftover))
    result=dict(passed=True,GPU_executed=False,recipe_sha256=sha(Path(__file__)),
        profile_summary_sha256=sha(path),capacity_metadata_sha256=sha(capacity),
        actual_prompt_tokens=d['actual_input_tokens'],batched_rows=n,GPU_residual_rows=gpu_rows,
        host_residual_rows=n-gpu_rows,host_chunk_rows=host_chunks,
        per_host_direction_bytes=sum(size*count for size,count in residual.items()),
        copy_groups=rows,unassigned_requested_bytes=sum(x['unassigned_bytes'] for x in rows),
        source_pins={str(p):sha(p) for p in [A.parents[2]/'src/prefill/prefill.cpp',A.parents[2]/'src/prefill/layer_major.inc',A.parents[2]/'src/core/expert_cache.cpp',A.parents[2]/'src/kernels/sycl/kv_stream.cpp']},
        corrections_to_R681_v1=[
            'Residual storage uses batched34602 rows, host30506 rows, final1834-row tail. R681 text says34603/30507/1835 but its byte figures correspond to these corrected counts.',
            'The remaining354324480B H2D histogram is exactly the9 PLE uploads at prefill.cpp:1764; code pins, pinned source owner and all histogram lengths agree.',
            'Append host-mirror writes are kernel stores and remain outside this copy-API ledger; whole-window token count is not a pure-prefill kernel write counter.'
        ],scope='Source-derived owner paths and exact complete copy-size distributions; not intercepted addresses or measured PCIe wire bytes. Remaining copies stay unclassified.',
        performance_qualified=False)
    with (A/'prefill-copy-accounting-main-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,batched_rows=n,host_rows=n-gpu_rows,
        host_residual_bytes_each_direction=result['per_host_direction_bytes'],
        unassigned_bytes=result['unassigned_requested_bytes'])))


if __name__=='__main__':main()
