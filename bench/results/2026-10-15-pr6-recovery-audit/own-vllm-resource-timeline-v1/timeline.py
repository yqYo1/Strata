#!/usr/bin/env python3
"""Strict offline reader for pinned PTI text DeviceTimeline. Repository LICENSE.
No runtime imports, subprocesses, environment changes or device access.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

LINE = re.compile(r'Thread ([1-9][0-9]*) Device ([1-9][0-9]*) : "(.+)" \[ns\] ([0-9]+) \(append\) ([0-9]+) \(submit\) ([0-9]+) \(start\) ([0-9]+) \(end\)')
SHAPE = re.compile(r'\[SIMD(?:[1-9][0-9]*|_ANY) \{([0-9]+); ([0-9]+); ([0-9]+)\} \{([0-9]+); ([0-9]+); ([0-9]+)\}\]$')
COPY = re.compile(r'zeCommandListAppendMemoryCopy(?:FromContext)?\(([HDMS])2([HDMS])\)\[([1-9][0-9]*)\]')
QSA_PREFIX = 'strata::kernels::(anonymous namespace)::launch<1, true, false, 0, false>(float const*, strata::kernels::QsaAttnPools,'
MAX_RAW = 256 << 20


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def base_name(name):
    return SHAPE.sub('', name)


def split(raw):
    require(isinstance(raw, str) and raw.endswith('\n') and '\x00' not in raw, 'complete UTF-8 newline-terminated text required')
    events, rest = [], []
    summary_started = False
    for n, line in enumerate(raw.splitlines(), 1):
        if line.startswith('=== '):
            summary_started = True
        if line.startswith('Thread '):
            require(len(line) <= 65536 and len(events) < 1000000, 'finite row length/count budget')
            require(not summary_started, 'timeline row after summary; incomplete/unknown flush schema')
            m = LINE.fullmatch(line)
            require(m is not None, 'malformed timeline row at line ' + str(n))
            thread, device, name, append, submit, start, end = m.groups()
            a, q, s, e = map(int, (append, submit, start, end))
            require(0 < a <= q <= s <= e < (1 << 64), 'negative/nonfinite/reversed/overflow interval')
            events.append(dict(line=n, thread=int(thread), device=int(device), name=name,
                               append=a, submit=q, start=s, end=e))
        else:
            rest.append(line)
    require(events and summary_started, 'missing timeline or summary completion')
    require(len({e['device'] for e in events}) == 1, 'one device handle required; multi-device scopes unqualified')
    # Text does not emit an instance id or engine. Do not deduplicate even exact
    # duplicate rows. No per-tile controls are permitted by the launch allowlist.
    return events, '\n'.join(rest) + '\n'


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        require(type(a) is int and type(b) is int and 0 <= a <= b, 'integer interval')
        if a == b:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([a, b])
    return out


def length(intervals):
    return sum(b - a for a, b in intervals)


def complement(intervals, window):
    a, b = window
    out, cursor = [], a
    for s, e in union(intervals):
        require(a <= s <= e <= b, 'interval outside declared window')
        if cursor < s:
            out.append([cursor, s])
        cursor = e
    if cursor < b:
        out.append([cursor, b])
    return out


def measure(events, window, label):
    a, b = window
    require(type(a) is int and type(b) is int and 0 < a < b, 'finite measured window')
    clips = [(e, max(a, e['start']), min(b, e['end'])) for e in events if e['start'] < b and e['end'] > a]
    groups = {k: union((s, t) for e, s, t in clips if e['kind'] == k) for k in ('compute', 'copy', 'other')}
    busy = union((s, t) for _, s, t in clips)
    c, m = length(groups['compute']), length(groups['copy'])
    cm = union(groups['compute'] + groups['copy'])
    both = c + m - length(cm)
    return dict(label=label, window_ns=[a, b], duration_ns=b-a,
                event_service_sum_ns=sum(t-s for _, s, t in clips), busy_union_ns=length(busy),
                compute_union_ns=c, copy_union_ns=m, other_union_ns=length(groups['other']),
                compute_copy_intersection_ns=both, copy_only_ns=m-both, compute_only_ns=c-both,
                neither_compute_nor_copy_ns=b-a-length(cm),
                no_observed_command_ns=b-a-length(busy), command_gap_count=len(complement(busy, window)),
                largest_command_gaps_ns=sorted(complement(busy, window), key=lambda x: x[1]-x[0], reverse=True)[:32],
                caveat='No-observed-command gaps are not proof of hardware idle; other includes barriers/fills/unknown commands. Neither may contain other work. Service sums overlap. Window clips service intervals; payload totals are not prorated.')


def reconcile(events, summary):
    expected = defaultdict(lambda: dict(calls=0, time_ns=0))
    for row in summary['device_rows']:
        expected[row['name']]['calls'] += row['calls']
        expected[row['name']]['time_ns'] += row['time_ns']
    actual = defaultdict(list)
    for e in events:
        actual[e['name']].append(e['end']-e['start'])
    require(set(actual) == set(expected), 'timeline/summary name coverage mismatch')
    checks = []
    for name in sorted(expected):
        v, xs = expected[name], actual[name]
        require(len(xs) == v['calls'] and sum(xs) == v['time_ns'], 'exact count/service-sum reconciliation: ' + name)
        rows = [r for r in summary['device_rows'] if r['name'] == name]
        require(min(xs) == min(r['min_ns'] for r in rows) and max(xs) == max(r['max_ns'] for r in rows), 'exact min/max reconciliation')
        checks.append(dict(name=name, calls=len(xs), time_ns=sum(xs), min_ns=min(xs), max_ns=max(xs), difference_ns=0))
    require(sum(e['end']-e['start'] for e in events) == summary['totals_ns']['Device'], 'exact device total')
    return dict(tolerance_ns=0, rationale='PrintCommandCompleted and CollectKernelCommandTimeStats receive the same uint64 start/end values in one LogCommandCompleted call; no Chrome floating-point conversion. Timer hardware granularity remains unmeasured.', rows=checks)


def qsa_windows(events, suite):
    request_samples = suite['results']['request_samples']
    require(len(request_samples) == 2 and [s['phase'] for s in request_samples] == ['warm-up/chat', 'prefill-only'], 'one short warmup then one long request')
    warm = request_samples[0]['sample']
    require(warm['ok'] and warm['stream_complete'] and warm['prompt_tokens'] == warm['read_tokens'] == 74 and warm['reused_tokens'] == 0, 'actual warmup74 shape identity')
    qs = sorted((e for e in events if base_name(e['name']).startswith(QSA_PREFIX)), key=lambda e: (e['append'], e['line']))
    require(len(qs) == 72 and len({e['append'] for e in qs}) == 72, 'exact72 ordered QSA dispatches; no timestamp tie waiver')
    require(len({base_name(e['name']) for e in qs}) == 1, 'single exact QSA kernel symbol')
    expected_queries = [73]*12 + ([8192]*4 + [6493])*12
    samples = []
    for i, (e, queries) in enumerate(zip(qs, expected_queries)):
        m = SHAPE.search(e['name'])
        require(m is not None and list(map(int, m.groups())) == [2*queries, 1, 1, 128, 1, 1], 'QSA sequence/shape mismatch; do not classify by duration')
        samples.append(dict(timeline_line=e['line'], sequence=i, queries=queries,
                            phase='short-warmup' if i < 12 else 'long-input',
                            qsa_layer_ordinal=i if i < 12 else (i-12)//5,
                            chunk_ordinal=0 if i < 12 else (i-12)%5,
                            append_ns=e['append'], submit_ns=e['submit'], start_ns=e['start'], end_ns=e['end']))
    long = qs[12:]
    return dict(proved=True, long_dispatch_count=60, warm_dispatch_count=12, samples=samples,
                mapping='Source: prompt_attention.cpp launch uses groups=queries*2, WG128; prefill.cpp dispatches once per QSA layer/chunk; exact sole request39262/batched39261 and 48-layer five-chunk trace are independently required by controller. QSA layer ordinals only, not invented physical layer IDs.',
                full_prefill_boundaries_proved=False,
                limitation='QSA append/start/end excludes earlier embedding, GDN, projections, later MoE/head and host work. No exact complete long-prefill host/device boundary in this collector; no full-prefill utilization fraction is emitted.',
                attention_envelope=measure(events, [min(e['start'] for e in long), max(e['end'] for e in long)], 'long-input QSA attention envelope: includes intervening commands; NOT full prefill'),
                long_QSA_service_sum_ns=sum(e['end']-e['start'] for e in long),
                warm_QSA_service_sum_ns=sum(e['end']-e['start'] for e in qs[:12]))


def extract(events, summary, raw_path, samples_path, suite=None):
    checks = reconcile(events, summary)
    properties = {p['name'] for p in summary['kernel_properties']}
    payloads, categories = Counter(), Counter()
    names = sorted({e['name'] for e in events})
    ids = {n:i for i,n in enumerate(names)}
    for e in events:
        name = e['name']
        e['kind'], e['payload_bytes'], e['direction'] = 'other', None, None
        if name.startswith('zeCommandListAppend') and ('MemoryCopy' in name or 'ImageCopy' in name):
            m = COPY.fullmatch(name)
            require(m is not None, 'missing/unqualified required copy-byte evidence: ' + name)
            e.update(kind='copy', payload_bytes=int(m[3]), direction=m[1]+'2'+m[2])
            payloads[e['direction']] += e['payload_bytes']
        elif base_name(name) in properties:
            require(SHAPE.search(name), 'required compute shape missing with verbose enabled')
            e['kind'] = 'compute'
        categories[e['kind']] += 1
    require(categories['compute'] and categories['copy'], 'compute and byte-qualified copy evidence required even in tiny gate')
    window = [min(e['append'] for e in events), max(e['end'] for e in events)]
    data = dict(schema='strata-unitrace-text-individual-samples-v1', names=names,
                columns=['raw_line','name_index','thread','device','append_ns','submit_ns','start_ns','end_ns'],
                rows=[[e['line'],ids[e['name']],e['thread'],e['device'],e['append'],e['submit'],e['start'],e['end']] for e in events])
    with Path(samples_path).open('x') as f:
        json.dump(data, f, allow_nan=False, separators=(',', ':')); f.write('\n')
    sample_pin = dict(path=str(samples_path), bytes=Path(samples_path).stat().st_size, sha256=sha(samples_path))
    unknowns = Counter(e['name'] for e in events if e['kind'] == 'other')
    result = dict(schema='strata-unitrace-resource-intervals-v1', proved=True,
                  raw=dict(path=str(raw_path), bytes=Path(raw_path).stat().st_size, sha256=sha(raw_path)),
                  parser_source_sha256=sha(__file__), individual_samples=sample_pin,
                  schema_representatives={k: next({x:y for x,y in e.items()} for e in events if e['kind']==k) for k in categories},
                  counts=dict(categories), zero_duration_row_count=sum(e['start']==e['end'] for e in events), unknown_or_other_categories=dict(unknowns),
                  reconciliation=checks,
                  whole_native_command_envelope=measure(events, window, 'All collected native command envelope, first append through last completed command; includes loading/warmup/generation/teardown commands, excludes unobserved leading/trailing native lifetime'),
                  requested_copy_payload_bytes_by_allocation_type=dict(payloads),
                  allocation_types=dict(H='HOST or HOST_IMPORTED', D='DEVICE', M='UNKNOWN/unregistered', S='SHARED'),
                  payload_caveat='Requested copy bytes only; no physical PCIe/bus/RAM/VRAM traffic. D2D may use GPU compute engines; M is allocation-query UNKNOWN, not a measured memory placement.',
                  append_to_submit_sum_ns=sum(e['submit']-e['append'] for e in events),
                  submit_to_start_sum_ns=sum(e['start']-e['submit'] for e in events),
                  gap_caveat='Collector host-device correlated timestamps; immediate command lists have append==submit by construction. These delays are dependency/queue/host submission evidence, not CPU compute duration or scheduling causality.',
                  deduplication='NONE: no identity in text; every emitted row preserved. Per-tile and Chrome controls forbidden. Exact summary reconciliation required.',
                  clock='PTI GetHostTime: zeDeviceGetGlobalTimestamps host ns plus wrap-corrected device ticks converted to ns. No external wall/CPU clock bridge asserted.',
                  unavailable=['PMU utilization','XMX occupancy','cache misses','RAM/VRAM physical bandwidth','SLM conflicts','barrier stalls','unobserved GPU owners','full prefill boundary','independently measured resource capacities'])
    if suite is not None:
        result['source_identified_QSA'] = qsa_windows(events, suite)
    return result
