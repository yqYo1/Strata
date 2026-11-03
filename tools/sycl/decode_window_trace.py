"""Reduce a bounded UnitTrace window; repository LICENSE, Main-authored.

The caller must qualify the collector, request, clock, losses and capture bounds.
This reducer checks observed identities and clips service; it cannot establish
complete device coverage, CPU time, wire traffic or hardware saturation.
"""
from collections import Counter
import hashlib
import math
import re


def require(value, message):
    if not value:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def identifier(value):
    require(type(value) in (int, str) and str(value), 'missing work identity')
    return str(value)


def union_us(intervals):
    end, total = None, 0.
    for start, finish in sorted(intervals):
        require(number(start) and number(finish) and start <= finish, 'invalid interval')
        total += finish - start if end is None or start > end else max(0., finish - end)
        end = finish if end is None else max(end, finish)
    return total


def kind(name):
    for prefix, label in (
        ('zeCommandListAppendMemoryCopy', 'copy'),
        ('zeCommandListAppendMemoryFill', 'fill'),
        ('zeCommandListAppendBarrier', 'barrier'),
        ('zeCommandList', 'other_L0'),
    ):
        if name.startswith(prefix):
            return label
    return 'kernel'


def top_services(rows):
    result = []
    for name, row in sorted(rows.items(), key=lambda item: item[1]['clipped_sum_us'], reverse=True)[:12]:
        result.append(dict(name=name[:512], name_sha256=hashlib.sha256(name.encode()).hexdigest(),
                           name_truncated=len(name) > 512, **row))
    return result


def summarize(events, *, native_pid, begin_ns, end_ns, capture_begin_ns, capture_end_ns):
    """Admit native ID joins, then report observed service in an inner window.

    Nanosecond bounds use the collector's qualified CLOCK_MONOTONIC_RAW domain;
    trace timestamps/durations are microseconds. Graph flow sources may precede
    the window because their timestamp describes capture, not replay. Join by
    instance ID only. Duplicate GPU IDs (including unqualified multi-tile rows)
    are refused instead of being silently counted twice.
    """
    require(type(events) is list and type(native_pid) is int and native_pid > 0,
            'invalid trace/process identity')
    bounds = (capture_begin_ns, begin_ns, end_ns, capture_end_ns)
    require(all(type(value) is int and value > 0 for value in bounds) and
            capture_begin_ns <= begin_ns < end_ns <= capture_end_ns, 'invalid capture/window bounds')
    begin, end = begin_ns / 1000, end_ns / 1000
    require(all(type(event) is dict for event in events), 'invalid trace event')
    sources = Counter()
    labels = {}
    for event in events:
        if event.get('ph') == 's' and event.get('pid') == native_pid and \
                str(event.get('cat', '')).startswith('Flow_H2D_'):
            sources[identifier(event.get('id'))] += 1
        if event.get('ph') == 'M' and event.get('name') == 'process_name':
            label = event.get('args', {}).get('name')
            pid = event.get('pid')
            require(type(label) is str and pid is not None and
                    (pid not in labels or labels[pid] == label), 'ambiguous process label')
            labels[pid] = label

    spans = {'gpu': [], 'kernel': [], 'copy': [], 'host': []}
    groups, apis, seen = {}, {}, set()
    counts, outside, partial = Counter(), Counter(), Counter()
    contained_payload, boundary_payload = Counter(), Counter()
    for event in events:
        category = event.get('cat')
        if event.get('ph') != 'X' or category not in ('gpu_op', 'cpu_op'):
            continue
        start, duration = event.get('ts'), event.get('dur')
        name = event.get('name')
        require(number(start) and number(duration) and duration >= 0 and
                type(name) is str and name, 'invalid service/time/name')
        finish = start + duration
        require(number(finish), 'overflowed trace timestamp')
        if finish <= begin or start >= end:
            outside[category] += 1
            continue
        clipped = (max(begin, start), min(end, finish))
        if category == 'cpu_op':
            require(event.get('pid') == native_pid, 'foreign host API process')
            spans['host'].append(clipped)
            row = apis.setdefault(name, dict(count=0, clipped_sum_us=0.))
            row['count'] += 1
            row['clipped_sum_us'] += clipped[1] - clipped[0]
            continue
        work_id = identifier(event.get('args', {}).get('id'))
        require(work_id not in seen and sources[work_id] == 1, 'duplicate or missing/ambiguous native GPU source')
        seen.add(work_id)
        label = labels.get(event.get('pid'), '')
        require(label.startswith('DEVICE<') and '] 0:5:0:0 #' in label, 'unexpected GPU PCI lane')
        operation = kind(name)
        require(operation != 'kernel' or duration > 0, 'nonpositive kernel service')
        counts[operation] += 1
        spans['gpu'].append(clipped)
        if operation in ('kernel', 'copy'):
            spans[operation].append(clipped)
        crossing = start < begin or finish > end
        if crossing:
            partial[operation] += 1
        if operation in ('copy', 'fill'):
            match = re.search(r'\[(\d+)\]$', name)
            require(match is not None, 'copy/fill requested payload unavailable')
            target = boundary_payload if crossing else contained_payload
            target[operation] += int(match[1])
        normalized = name.split('[SIMD', 1)[0]
        row = groups.setdefault(normalized, dict(kind=operation, count=0, clipped_sum_us=0.))
        require(row['kind'] == operation, 'ambiguous normalized operation')
        row['count'] += 1
        row['clipped_sum_us'] += clipped[1] - clipped[0]

    require(counts['kernel'] > 0 and spans['host'], 'GPU kernels/host APIs missing from window')
    wall = end - begin
    gpu_union = union_us(spans['gpu'])
    require(0 < gpu_union <= wall, 'invalid observed GPU union')
    return dict(window_wall_us=wall, observed_gpu_interval_union_us=gpu_union,
        observed_kernel_interval_union_us=union_us(spans['kernel']),
        observed_copy_interval_union_us=union_us(spans['copy']),
        observed_host_api_interval_union_us=union_us(spans['host']),
        unobserved_gpu_event_gap_us=max(0., wall - gpu_union),
        operation_counts=dict(counts), outside_window_events=dict(outside),
        boundary_crossing_counts=dict(partial),
        wholly_contained_requested_payload_bytes=dict(contained_payload),
        crossing_command_requested_bytes_not_assigned_to_window=dict(boundary_payload),
        largest_observed_device_services=top_services(groups),
        largest_observed_host_apis=top_services(apis),
        included_gpu_ids_unique=True, included_native_flow_joins_unique=True,
        complete_device_coverage_qualified=False, physical_saturation_qualified=False,
        limits='Observed clipped service only;omitted or lost commands remain unknown. '
               'Graph flows join by instance ID,not source timestamp or row order. '
               'Service/API sums may overlap;API wall is not CPU time. '
               'Gaps between observed events are not proof of device idle. '
               'Requested payload is not PCIe/VRAM traffic;crossing commands are not prorated. '
               'Caller must separately qualify source/runtime,request identity,clocks,boundaries,loss and observer cost.')
