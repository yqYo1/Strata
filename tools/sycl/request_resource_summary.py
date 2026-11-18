#!/usr/bin/env python3
"""Summarize copy and kernel activity in closed, owned PTI request windows.

Main-authored under the repository LICENSE. The resource reduction comes from
the owned d3557debdf2 analysis. Copy sizes are API payload bytes. PTI's M
denotes ZE_MEMORY_TYPE_UNKNOWN.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import request_profile_reader as reader

COPY = re.compile(r'^zeCommandListAppendMemoryCopy\(([HDMS]2[HDMS])\)\[(\d+)\]$')
def summarize(trace, receipt, original=None):
    check = reader.summarize(trace, receipt)
    if original is not None:
        reader.require(all(check[k] == original[k] for k in check), 'original request summary mismatch')
    output = []
    for request, prior in zip(receipt['requests'], original['requests']):
        low, high = request['begin_raw_ns'], request['end_raw_ns']
        groups = defaultdict(lambda: dict(calls=0, service_ns=0, requested_API_bytes=0))
        kernels = defaultdict(lambda: dict(calls=0, service_ns=0))
        intervals = dict(copy=[], kernel=[], other_command=[])
        crossing = 0
        for event in trace['traceEvents']:
            if event.get('cat') != 'gpu_op': continue
            start = reader.nanoseconds(event['ts']); stop = start + reader.nanoseconds(event['dur'])
            if stop <= low or start >= high: continue
            name = event['name']; match = COPY.fullmatch(name)
            kind = 'copy' if match else ('other_command' if name.startswith('zeCommandList') else 'kernel')
            intervals[kind].append((max(start, low), min(stop, high)))
            if start < low or stop > high:
                crossing += 1
                continue
            if match:
                value = groups[match[1]]; value['requested_API_bytes'] += int(match[2])
            elif kind == 'other_command':
                value = groups[name]
            else:
                # Keep template parameters; combine launch geometries only.
                value = kernels[re.sub(r'\[SIMD[^\n]*$', '', name).rstrip()]
            value['calls'] += 1; value['service_ns'] += stop-start
        ranked = sorted(kernels.items(), key=lambda item: (-item[1]['service_ns'], item[0]))
        top, tail = ranked[:12], ranked[12:]
        counts = sum(x['calls'] for x in groups.values()) + sum(x['calls'] for x in kernels.values())
        service = sum(x['service_ns'] for x in groups.values()) + sum(x['service_ns'] for x in kernels.values())
        reader.require(counts == prior['contained_GPU_events'] and service == prior['contained_GPU_service_ns']
                       and crossing == prior['boundary_crossing_GPU_events'], 'conservation failure')
        unions = {kind: reader.union_ns(value) for kind, value in intervals.items()}
        all_union = reader.union_ns([v for values in intervals.values() for v in values])
        reader.require(all_union == prior['observed_GPU_interval_union_ns'], 'union mismatch')
        output.append(dict(request_id=request['request_id'], sample_scope=request['sample_scope'],
            request_sha256=request['request_sha256'], actual_read_tokens=request['actual_read_tokens'],
            generated_tokens=request['generated_tokens'], window_ns=high-low,
            contained_GPU_events=counts, contained_GPU_service_ns=service,
            boundary_crossing_GPU_events=crossing, copy_and_other_commands=dict(sorted(groups.items())),
            top_kernel_families=[dict(name=name, **value) for name,value in top],
            kernel_tail=dict(families=len(tail), calls=sum(v['calls'] for _,v in tail),
                             service_ns=sum(v['service_ns'] for _,v in tail)),
            interval_union_ns=unions, all_interval_union_ns=all_union,
            copy_kernel_overlap_ns=unions['copy']+unions['kernel']-
                reader.union_ns(intervals['copy']+intervals['kernel'])))
    return dict(requests=output, totals_match_request_reader=True,
        physical_bus_bytes_qualified=False, hardware_saturation_qualified=False,
        phase_attribution_qualified=False, clean_speed_qualified=False)


if __name__ == '__main__':
    reader.require(len(sys.argv) == 4, 'trace.json receipt.json output.json required')
    trace, receipt, output = map(Path, sys.argv[1:])
    contract = reader.load_json(receipt)
    reader.require(trace.stat().st_size <= reader.MAX_BYTES, 'trace byte limit')
    with trace.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    reader.require(digest == contract['trace_sha256'], 'closed trace identity differs')
    summary = summarize(reader.load_json(trace), contract)
    summary['trace_sha256'] = digest
    with output.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
