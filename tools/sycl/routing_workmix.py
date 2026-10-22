#!/usr/bin/env python3
"""Decode the audited synchronous serve callback stream, never infer its frames.

The caller must bind exact native sources, one unsplit verifier, immutable cache,
request gates and the complete STRATA_TRACE/census. A route file alone has none
of those guarantees. Counts below are logical work, not CPU service or bus bytes.
"""
from collections import Counter
import re
import struct

from measurement_guards import require

RECORD = struct.Struct('<12i10f')
LAYERS, EXPERTS, K = 48, 512, 10


def request_windows(text, samples, context=131072):
    footers = list(re.finditer(r'^strata serve: prompt .*$', text, re.M))
    require(len(footers) == len(samples), 'routing request/footer count')
    result, previous = [], 0
    for footer, sample in zip(footers, samples):
        segment = text[previous:footer.start()]; previous = footer.end()
        begin = list(re.finditer(r'^strata trace: request (\d+) ([01])$', segment, re.M))
        done = list(re.finditer(r'^strata trace: prompt done \(slots refilled\) -1 -1$', segment, re.M))
        windows = list(re.finditer(r'^strata trace: window (\d+) (\d+)$', segment, re.M))
        require(len(begin) == len(done) == 1 and windows and
                len(re.findall(r'^strata trace: request ',segment,re.M)) == 1 and
                len(re.findall(r'^strata trace: window ',segment,re.M)) == len(windows) and
                int(begin[0][1]) == sample['prompt_tokens'] and begin[0][2] == '0' and
                begin[0].end() < done[0].start() < windows[0].start(),
                'routing request/prefill/window framing missing or interleaved')
        pairs = [(int(m[1]), int(m[2])) for m in windows]
        require(pairs[0] == (sample['prompt_tokens']-1, 1) and
                all(t == 4 for _, t in pairs[1:]) and
                all(0 <= p < p+t <= context for p,t in pairs) and
                all(1 <= p-q <= t for (q,t),(p,_) in zip(pairs,pairs[1:])) and
                not re.search(r'^strata trace: read \d+ tokens \(windows\)',segment,re.M),
                'unsupported routing callback schedule or prompt route')
        result.append([t for _, t in pairs])
    return result


def decode(stream, windows, membership, formats, expected):
    """Read exactly the externally framed callbacks and reconcile each provider.

    Each window calls layers0..47 once with T rows because split=false. Expert
    jobs deduplicate across rows within that callback. The existing writer emits
    zeros for its placeholder weights; these are not actual routing weights.
    """
    require(len(membership) == len(formats) == LAYERS and windows and
            windows[0] == 1 and all(t == 4 for t in windows[1:]), 'unsupported routing geometry')
    counts = [[[0]*4 for _ in range(EXPERTS)] for _ in range(LAYERS)]
    bins = {}
    for f in formats:
        bins.setdefault(f,dict(calls=0,CPU=0,VRAM=0,jobs_NT=[0]*8))
    for tokens in windows:
        for layer in range(LAYERS):
            routed = Counter()
            for _ in range(tokens):
                raw = stream.read(RECORD.size)
                require(len(raw) == RECORD.size, 'partial routing callback')
                values = RECORD.unpack(raw); ids = values[2:12]
                require(values[:2] == (layer,K) and len(set(ids)) == K and
                        all(0 <= e < EXPERTS for e in ids) and raw[-40:] == bytes(40),
                        'routing layer/order/IDs/placeholder mismatch')
                routed.update(ids)
            b = bins[formats[layer]]; b['calls'] += 1
            for expert,nt in routed.items():
                require(1 <= nt <= tokens, 'routing expert NT invalid')
                counts[layer][expert][nt-1] += 1
                if membership[layer] & (1 << expert):
                    b['VRAM'] += nt
                else:
                    b['CPU'] += nt; b['jobs_NT'][nt-1] += 1
    jobs = [sum(b['jobs_NT'][n] for b in bins.values()) for n in range(8)]
    summary = dict(windows=len(windows),rows=sum(windows),cpu_jobs=sum(jobs),
        cpu_entries=sum(b['CPU'] for b in bins.values()),
        vram_entries=sum(b['VRAM'] for b in bins.values()),jobs_NT=jobs,
        logical_job_blob_bytes=sum(sum(b['jobs_NT'])*f[2] for f,b in bins.items()))
    require(all(summary[k] == expected[k] for k in summary), 'routing/census work mismatch')
    require(len(bins) == len(expected['format_bins']), 'routing/census format count')
    for row in expected['format_bins']:
        b = bins[(row['gu'],row['down'],row['blob_bytes'])]
        require(all(b[k] == row[k] for k in b), 'routing/census format work mismatch')
    return counts,summary


def encode_counts(counts):
    require(len(counts) == LAYERS and all(len(l) == EXPERTS for l in counts) and
            all(len(row) == 4 and all(type(v) is int and 0 <= v < 1<<32 for v in row)
                for layer in counts for row in layer), 'invalid expert counts')
    # Versioned little-endian header followed by layer/expert/NT1..4 uint32s.
    return struct.pack('<4s4I',b'SRWC',1,LAYERS,EXPERTS,4) + b''.join(
        struct.pack('<4I',*row) for layer in counts for row in layer)
