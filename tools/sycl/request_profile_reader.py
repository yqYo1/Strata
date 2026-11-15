#!/usr/bin/env python3
"""Read owned request windows from a non-counter PTI trace.

Main-authored under the repository LICENSE. Schema reference: Intel PTI
4b6d742, unitimer.h and chromelogger.h (MIT, inspected independently).
UNITRACE_SystemTime=1 makes the emitted microseconds CLOCK_MONOTONIC_RAW;
the controller must pin that setting/source and supply actual RAW bounds.
This reader verifies intervals, not collector admission or a drained prefill
boundary. Until that separate handshake is qualified, coverage is whole request.
"""
import argparse
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re

SCHEMA = 'strata-request-profile-v1'
PTI_REVISION = '4b6d74294cdc72766c00a9211fe2e4261ee94244'
CLOCK = 'CLOCK_MONOTONIC_RAW'
# Three actual long requests can emit hundreds of thousands of device events.
# Controllers must predeclare this finite allowance and supervise their output.
MAX_BYTES, MAX_EVENTS = 256 << 20, 1_000_000
HOST_CHECK = dict(command=['/home/yayoi/.local/share/strata-sycl/venv/bin/python3',
    '-B', '-m', 'unittest', 'tools.test_request_profile_reader'],
    wall_seconds=90, CPU_seconds=60, RSS_bytes=1 << 30,
    text_bytes=128 << 10, total_bytes=4 << 20, GPU_executed=False)


def require(value, message):
    if not value: raise ValueError(message)


def nanoseconds(value):
    require(type(value) in (int, float, Decimal), 'numeric trace microseconds required')
    try: result = Decimal(str(value)) * 1000
    except InvalidOperation as error: raise ValueError('invalid trace time') from error
    require(result.is_finite() and result >= 0, 'finite nonnegative trace time required')
    # UniTimer emits a double via std::to_string; its decimal microseconds can
    # contain sub-nanosecond rounding even though the input timer is integral.
    return int(result.to_integral_value())


def raw_bound(value):
    require(type(value) is int and 0 <= value < 1 << 64, 'uint64 RAW nanoseconds required')
    return value


def union_ns(intervals):
    total, end = 0, None
    for begin, stop in sorted(intervals):
        if end is None or begin >= end: total += stop - begin
        elif stop > end: total += stop - end
        end = stop if end is None else max(stop, end)
    return total


def summarize(trace, receipt):
    require(receipt.get('schema') == SCHEMA and receipt.get('clock_domain') == CLOCK
            and receipt.get('pti_revision') == PTI_REVISION
            and receipt.get('unitrace_system_time') == '1'
            and receipt.get('level_zero_only') is True,
            'pinned non-counter PTI RAW clock contract required')
    require(type(receipt.get('native_exit_code')) is int and receipt['native_exit_code'] == 0
            and receipt.get('ownership_closed') is True
            and receipt.get('fresh_faults') == [] and receipt.get('boot_unchanged') is True,
            'closed normal owned native execution without fresh faults required')
    source = receipt.get('source', {})
    require(re.fullmatch('[0-9a-f]{40}', source.get('native_source_commit', ''))
            and re.fullmatch('[0-9a-f]{64}', source.get('binary_sha256', '')),
            'exact native source and executable identity required')
    lifetime = receipt.get('native_lifetime', {})
    begin, end = raw_bound(lifetime.get('begin_raw_ns')), raw_bound(lifetime.get('end_raw_ns'))
    require(begin < end, 'positive native lifetime required')
    requests = receipt.get('requests')
    require(type(requests) is list and 0 < len(requests) <= 16, 'bounded actual requests required')
    previous, identities, windows = begin, set(), []
    for request in requests:
        low = raw_bound(request.get('begin_raw_ns')); high = raw_bound(request.get('end_raw_ns'))
        identity = request.get('request_id')
        require(type(identity) is str and identity and identity not in identities
                and previous <= low < high <= end, 'unique ordered nonoverlapping request windows required')
        identities.add(identity); previous = high
        read = request.get('actual_read_tokens')
        require(type(read) is int and read >= 32768 and read == request.get('expected_read_tokens')
                and type(request.get('actual_reused_tokens')) is int
                and request['actual_reused_tokens'] == 0 and request.get('thinking') is True
                and type(request.get('generated_tokens')) is int and request['generated_tokens'] > 0
                and re.fullmatch('[0-9a-f]{64}', request.get('request_sha256', '')),
                'actual fresh long thinking input and wire identity required')
        require(request.get('sample_scope') in ('first_long', 'warmed'), 'explicit sample scope required')
        windows.append((low, high))
    require(type(trace) is dict and type(trace.get('traceEvents')) is list
            and len(trace['traceEvents']) <= MAX_EVENTS, 'bounded Chrome trace event list required')
    events = []
    for event in trace['traceEvents']:
        require(type(event) is dict, 'trace event object required')
        if event.get('cat') != 'gpu_op': continue
        require(event.get('ph') == 'X' and type(event.get('name')) is str and event['name']
                and not event['name'].startswith('Tile #'), 'single-tile complete GPU events required')
        low, duration = nanoseconds(event.get('ts')), nanoseconds(event.get('dur'))
        high = low + duration
        require(begin <= low <= high <= end, 'GPU event lies outside pinned RAW native lifetime')
        events.append((low, high, event['name']))
    require(events and any(high > low for low, high, _ in events), 'nonempty timed GPU work required')
    output = []
    assigned = set()
    for request, (low, high) in zip(requests, windows):
        intervals, inside, crossing, groups = [], [], [], defaultdict(lambda: [0, 0])
        for i, (start, stop, name) in enumerate(events):
            if stop <= low or start >= high: continue
            assigned.add(i); intervals.append((max(start, low), min(stop, high)))
            if start < low or stop > high: crossing.append(i); continue
            inside.append(i); groups[name][0] += 1; groups[name][1] += stop - start
        service = sum(events[i][1] - events[i][0] for i in inside)
        require(intervals, 'no GPU activity observed in an actual long request window')
        top = sorted(groups.items(), key=lambda item: (-item[1][1], item[0]))[:20]
        top_names = {name for name, _ in top}
        output.append(dict(request_id=request['request_id'], request_sha256=request['request_sha256'],
            actual_read_tokens=request['actual_read_tokens'], generated_tokens=request['generated_tokens'],
            thinking=True, sample_scope=request['sample_scope'],
            window_ns=high-low, contained_GPU_events=len(inside), boundary_crossing_GPU_events=len(crossing),
            contained_GPU_service_ns=service, observed_GPU_interval_union_ns=union_ns(intervals),
            top_operations=[dict(name=name, calls=value[0], service_ns=value[1]) for name, value in top],
            other_operation_calls=sum(value[0] for name, value in groups.items() if name not in top_names),
            other_operation_service_ns=sum(value[1] for name, value in groups.items() if name not in top_names)))
    rounding = max(1, math.ceil(math.ulp(end / 1000) * 500 + .001))
    return dict(schema=SCHEMA, source=source, clock_domain=CLOCK, trace_unit='microseconds',
        time_conversion='emitted_decimal_microseconds_rounded_to_nearest_nanosecond',
        source_double_timestamp_rounding_bound_ns=rounding,
        coverage='whole_request_external_bounds', requests=output, total_GPU_events=len(events),
        outside_request_GPU_events=len(events)-len(assigned),
        runtime_admission_qualified=False, observer_overhead_qualified=False,
        phase_attribution_qualified=False, native_request_drain_qualified=False,
        physical_bus_bytes_qualified=False, hardware_saturation_qualified=False,
        clean_speed_qualified=False,
        limits='Service sums,overlapping observed intervals and wall time differ. Boundary-crossing '
               'work is reported separately. Counts and time do not establish physical traffic, '
               'hardware utilization,collector overhead or prefill critical-path attribution.')


def load_json(path):
    require(path.stat().st_size <= MAX_BYTES, 'finite input byte budget exceeded')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key'); result[key] = value
        return result
    def constant(value): raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(path.read_text(), parse_float=Decimal, parse_constant=constant, object_pairs_hook=pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = summarize(load_json(args.trace), load_json(args.receipt))
    with args.out.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
