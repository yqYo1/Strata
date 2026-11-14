#!/usr/bin/env python3
"""Reduce a closed SYCL capture to kernel service and copy/compute overlap.

Main-authored; repository LICENSE. The capture's original model/clock/flow gates
remain authoritative. API payloads and observed intervals are not physical bus
traffic, hardware idle, saturation, or a new clean throughput measurement.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def union_us(intervals):
    total, end = 0.0, None
    for begin, finish in sorted(intervals):
        if end is None or begin > end:
            total += finish - begin
            end = finish
        elif finish > end:
            total += finish - end
            end = finish
    return total


def reduce_window(events, begin, end, kernel):
    if not all(math.isfinite(v) for v in (begin, end)) or end <= begin:
        raise ValueError('invalid qualified client window')
    kernels, copies, selected = [], [], []
    for event in events:
        if event.get('ph') != 'X' or event.get('cat') != 'gpu_op':
            continue
        start, duration = event['ts'], event['dur']
        if not all(math.isfinite(v) for v in (start, duration)) or duration < 0:
            raise ValueError('invalid GPU event clock')
        if not (start < end and start + duration > begin):
            continue
        interval = max(begin, start), min(end, start + duration)
        name = event['name']
        if not name.startswith('zeCommandList'):
            kernels.append(interval)
        if name.startswith('zeCommandListAppendMemoryCopy'):
            copies.append(interval)
        if kernel in name:
            selected.append(interval)
    return dict(selected_kernel_calls=len(selected),
        selected_kernel_service_s=sum(b-a for a,b in selected)/1e6,
        kernel_union_s=union_us(kernels)/1e6,
        copy_union_s=union_us(copies)/1e6,
        observed_copy_kernel_overlap_s=max(0.0,
            (union_us(kernels)+union_us(copies)-union_us(kernels+copies))/1e6))


def selfcheck():
    event = lambda name,start,duration:dict(ph='X',cat='gpu_op',name=name,ts=start,dur=duration)
    events = [event('kernel',0,6),event('kernel',3,5),
        event('zeCommandListAppendMemoryCopy(M2D)',5,5),event('irrelevant',20,3)]
    value = reduce_window(events,1,9,'kernel')
    assert value['selected_kernel_calls'] == 2
    assert value['selected_kernel_service_s'] == 10/1e6
    assert value['kernel_union_s'] == 7/1e6
    assert value['copy_union_s'] == 4/1e6
    assert value['observed_copy_kernel_overlap_s'] == 3/1e6
    assert union_us([(0,2),(2,4),(1,3),(7,8)]) == 5
    assert union_us([]) == 0
    for a,b in ((0,0),(2,1),(0,float('nan'))):
        try:reduce_window([],a,b,'kernel')
        except ValueError:pass
        else:raise AssertionError('invalid window accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--kernel',required=True)
    parser.add_argument('--expected-calls',type=int,required=True)
    args = parser.parse_args()
    selfcheck()
    row = json.loads(args.record.read_text())
    if not (row['complete'] and row['passed'] and row['ownership_closed']
            and not row['active'] and row['decision']['observer_cost_qualified']):
        raise ValueError('closed qualified observer bracket required')
    observations = [p['resource_observation'] for p in row['processes']
        if p.get('resource_observation',{}).get('complete')]
    if len(observations) != 1 or args.expected_calls <= 0 or not args.kernel:
        raise ValueError('one complete capture and explicit kernel/count required')
    observed, = observations
    path = Path(observed['timeline']['path'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != observed['timeline']['sha256']:
        raise ValueError('original timeline hash differs')
    events = json.loads(raw)['traceEvents']
    windows = []
    for window in observed['windows']:
        clock = window['clock']
        value = reduce_window(events,clock['begin_raw_ns']/1000,
            clock['end_raw_ns']/1000,args.kernel)
        if value['selected_kernel_calls'] != args.expected_calls:
            raise ValueError('actual selected kernel count differs')
        windows.append(value)
    print(json.dumps(dict(timeline_sha256=observed['timeline']['sha256'],
        kernel_name_contains=args.kernel,windows=windows,
        scope='Post-capture derivation from the original qualified client windows;'
            'copy union is observed GPU copy service,not physical bus traffic or saturation.'),
        indent=2))


if __name__ == '__main__':
    main()
