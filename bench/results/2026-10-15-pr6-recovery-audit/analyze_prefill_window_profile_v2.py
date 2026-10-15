#!/usr/bin/env python3
"""Bounded streaming summary of the closed, native-process unitrace window.

No raw event/flow identifiers are emitted. Sums and unions stay distinct. The
window includes restore and one generated token; no external clock alignment
or pure-prefill critical-path attribution is inferred.
"""
import collections
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def merged(intervals):
    out = []
    for start, end in sorted(intervals):
        require(math.isfinite(start) and math.isfinite(end) and end >= start,
                'finite ordered interval')
        if out and start <= out[-1][1]:
            if end > out[-1][1]:
                out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def length(intervals):
    return sum(end - start for start, end in intervals)


def overlap(a, b):
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def role(name):
    if name.startswith('zeCommandListAppendMemoryCopy'):
        return 'provider_memory_copy'
    if name.startswith('zeCommandListAppendMemoryFill'):
        return 'provider_memory_fill'
    for label, token in [
        ('prefill_selected_int8_attention', 'launch<1, false, false>(float const*, strata::kernels::QsaAttnPools'),
        ('prefill_GU_dequant', 'iq_dequant_gu_f16('),
        ('generic_f16_dequant_multiple_consumers', '::dequant<unsigned short>('),
        ('QSA_block_scores_shared', '::qsa_block_scores('),
        ('GDN_recurrence_shared', '::recurrence('),
        ('KV_resolver_shared', 'kv_stream_resolve_with_copy_groups')]:
        if token in name:
            return label
    return 'other_compute'


def collect(receipt, target):
    d = json.loads(receipt.read_text())
    require(d['passed'] and d['complete'] and not d['active'] and d['provider_window_valid'],
            'complete closed profile')
    require(not d['fresh_faults'] and not d['driver_dump_exists'] and d['boot_unchanged']
            and d['session_released'], 'fault/session closure')
    owner = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    for c in d['commands'] + d['control_commands']:
        require(owner.closed(c), 'supervised command closure')
        for old in c['owners']:
            now = owner.identity(old['pid'])
            require(now is None or now['start_ticks'] != old['start_ticks'], 'run still active')
    trace = Path(d['native_trace']['path'])
    require(trace.stat().st_size < 2 << 30, 'trace file budget')
    digest = hashlib.sha256()
    counts = collections.Counter()
    groups = {'compute': [], 'copy': [], 'fill': []}
    lanes = collections.defaultdict(list)
    cpus = collections.defaultdict(list)
    waits = []
    kernel = {}
    api = {}
    metadata = []
    origin = None
    span_first = float('inf')
    span_last = float('-inf')
    events = 0
    with trace.open('rb') as f:
        require(f.readline() == b'{ "traceEvents":[\n', 'known unitrace line encoding')
        digest.update(b'{ "traceEvents":[\n')
        for raw in f:
            digest.update(raw)
            line = raw.strip().rstrip(b',')
            if not line.startswith(b'{'):
                require(line in (b']}', b']', b'}', b''), 'known trace footer')
                continue
            e = json.loads(line)
            events += 1
            cat = e.get('cat')
            if cat and cat.startswith('Flow_H2D_'):
                cat = 'Flow_H2D'
            elif cat and cat.startswith('Flow_D2H_'):
                cat = 'Flow_D2H'
            counts[(e['ph'], cat)] += 1
            if e['ph'] == 'M':
                metadata.append(e)
            if e['ph'] != 'X':
                continue
            require(cat in ('cpu_op', 'gpu_op'), 'known timed event category')
            if origin is None:
                origin = e['ts']
            start, dur = e['ts'] - origin, e['dur']
            end = start + dur
            require(math.isfinite(dur) and dur >= 0, 'finite nonnegative duration')
            span_first, span_last = min(span_first, start), max(span_last, end)
            name = e['name']
            key = (e['pid'], e['tid'])
            if cat == 'gpu_op':
                r = role(name)
                group = 'copy' if r == 'provider_memory_copy' else 'fill' if r == 'provider_memory_fill' else 'compute'
                interval = (start, end)
                groups[group].append(interval)
                lanes[key].append(interval)
                # Allocation/copy sizes would otherwise create one category per
                # operation. Preserve exact byte totals and a bounded histogram.
                match = re.search(r'\[(\d+)\]$', name)
                byte_count = int(match[1]) if match and group != 'compute' else None
                normalized = name[:match.start()] if byte_count is not None else name
                geometry = re.search(r'\[SIMD\d+ .*\]$', normalized) if group == 'compute' else None
                geometry_label = normalized[geometry.start():] if geometry else None
                if geometry:
                    normalized = normalized[:geometry.start()]
                row = kernel.setdefault(normalized, dict(name=normalized, role=r, calls=0,
                    duration_us=0.0, min_us=float('inf'), max_us=0.0,
                    first_us=start, last_us=end, provider_requested_bytes=0,
                    sized_calls=0, sizes=collections.Counter(), geometries=collections.Counter()))
                row['calls'] += 1
                if geometry_label is not None:
                    row['geometries'][geometry_label] += 1
                row['duration_us'] += dur
                row['min_us'], row['max_us'] = min(row['min_us'], dur), max(row['max_us'], dur)
                row['first_us'], row['last_us'] = min(row['first_us'], start), max(row['last_us'], end)
                if byte_count is not None:
                    row['provider_requested_bytes'] += byte_count
                    row['sized_calls'] += 1
                    row['sizes'][byte_count] += 1
            else:
                cpus[key].append((start, end))
                row = api.setdefault(name, dict(name=name, calls=0, duration_us=0.0,
                    min_us=float('inf'), max_us=0.0))
                row['calls'] += 1
                row['duration_us'] += dur
                row['min_us'], row['max_us'] = min(row['min_us'], dur), max(row['max_us'], dur)
                if 'synchronize' in name.lower() or 'wait' in name.lower():
                    waits.append((start, end))
            require(len(kernel) < 10000 and len(api) < 1000, 'bounded operation categories')
    require(digest.hexdigest() == d['native_trace']['sha256'], 'full raw trace SHA')
    expected = collections.Counter({(x['phase'],x['category']):x['count']
                                    for x in d['native_trace']['event_counts']})
    require(events == d['native_trace']['events'] and counts == expected, 'all raw events conserved')
    unions = {k: merged(v) for k,v in groups.items()}
    gpu = merged(x for v in groups.values() for x in v)
    cpu = merged(x for v in cpus.values() for x in v)
    wait = merged(waits)
    gpu_span = gpu[-1][1] - gpu[0][0]
    require(length(gpu) <= gpu_span + 0.01, 'GPU interval bound')
    for row in kernel.values():
        sizes = row.pop('sizes')
        geometries = row.pop('geometries')
        row['distinct_launch_geometries'] = len(geometries)
        row['launch_geometries_top32'] = [dict(geometry=g,calls=c) for g,c in geometries.most_common(32)]
        row['size_histogram_top32'] = [dict(bytes=n,calls=c) for n,c in sizes.most_common(32)]
        row['distinct_sizes'] = len(sizes)
    role_rows = {}
    for row in kernel.values():
        rr = role_rows.setdefault(row['role'], dict(calls=0,duration_us=0.0))
        rr['calls'] += row['calls']; rr['duration_us'] += row['duration_us']
    # Keep only the largest gaps, never a second full timeline.
    gaps = sorted([(gpu[i][0]-gpu[i-1][1],gpu[i-1][1],gpu[i][0])
                   for i in range(1,len(gpu))], reverse=True)[:32]
    result = dict(schema='prefill-window-interval-summary-v2', passed=True,
        recipe_sha256=sha(Path(__file__)), receipt=str(receipt), receipt_sha256=sha(receipt),
        trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=digest.hexdigest(),events=events),
        event_counts=[dict(phase=k[0],category=k[1],count=v) for k,v in counts.items()],
        actual_input_tokens=d['actual_input_tokens'], done=d['done'], metadata=metadata,
        relative_origin_trace_timestamp_us=origin,
        timed_event_span_us=span_last-span_first, gpu_span_us=gpu_span,
        GPU_service_sum_us=sum(x['duration_us'] for x in kernel.values()),
        GPU_busy_union_us=length(gpu), GPU_gap_within_span_us=gpu_span-length(gpu),
        GPU_group_union_us={k:length(v) for k,v in unions.items()},
        GPU_compute_copy_overlap_us=overlap(unions['compute'],unions['copy']),
        GPU_lanes=[dict(pid=k[0],tid=k[1],calls=len(v),union_us=length(merged(v))) for k,v in lanes.items()],
        CPU_API_service_sum_us=sum(x['duration_us'] for x in api.values()),
        CPU_API_union_us=length(cpu), CPU_wait_union_us=length(wait),
        CPU_wait_overlap_GPU_busy_us=overlap(wait,gpu),
        CPU_wait_without_visible_GPU_us=length(wait)-overlap(wait,gpu),
        GPU_gaps_top32=[dict(duration_us=v,start_us=s,end_us=e) for v,s,e in gaps],
        role_service=role_rows,
        GPU_operations=sorted(kernel.values(),key=lambda x:x['duration_us'],reverse=True),
        CPU_APIs=sorted(api.values(),key=lambda x:x['duration_us'],reverse=True),
        limitations=[
            d['scope'],
            'Provider timestamps only; no external monotonic/UTC phase alignment is used.',
            'Device service sum, busy union and request wall are different quantities.',
            'Launch geometries are grouped by complete base kernel name, preserving counts, service and top32 geometries; raw trace SHA pins the complete distribution.',
            'A host wait overlapping useful GPU work is not wholly removable overhead.',
            'Copy names/size brackets describe the provider operation, not measured PCIe wire bytes; endpoint semantics need provider-source verification.',
            'Flow_H2D/D2H are causal categories, not transfer payload direction or bytes.',
            'Unobserved GPU gaps include submission, CPU work, instrumentation and scheduling; no single cause is inferred.',
            'Clock normalization has provider precision; gaps and overlap are diagnostic rather than cycle-exact.'
        ], performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False)
    with target.open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    require(target.stat().st_size < 2 << 20,'compact summary budget')
    print(json.dumps(dict(passed=True,mode=d['mode'],events=events,
        GPU_service_seconds=result['GPU_service_sum_us']/1e6,
        GPU_busy_seconds=result['GPU_busy_union_us']/1e6,
        GPU_gap_seconds=result['GPU_gap_within_span_us']/1e6,
        summary_bytes=target.stat().st_size)),flush=True)


def main():
    mode=sys.argv[1]
    require(mode in ('component','long'),'known closed experiment')
    receipt=A/('prefill-window-profile-long-v4.json' if mode=='long'
               else 'prefill-window-profile-component-v3-compact.json')
    target=A/f'prefill-window-{mode}-interval-summary-v2.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not target.exists(),'fresh evidence')
        collect(receipt,target)


if __name__=='__main__':
    main()
