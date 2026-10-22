#!/usr/bin/env python3
"""Decode the audited synchronous serve callback stream, never infer its frames.

The caller must bind exact native sources, one unsplit verifier, immutable cache,
request gates and the complete STRATA_TRACE/census. A route file alone has none
of those guarantees. Counts below are logical work, not CPU service or bus bytes.
"""
from collections import Counter
import hashlib
from pathlib import Path
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


def read_counts(raw):
    require(len(raw) == 393236 and struct.unpack_from('<4s4I',raw) == (b'SRWC',1,48,512,4),
            'unsupported/incomplete expert-window matrix')
    rows = list(struct.iter_unpack('<4I',raw[20:]))
    return [[list(v) for v in rows[l*512:(l+1)*512]] for l in range(48)]


def ranked_experts(counts, profile, by):
    import cpu_work_census as census
    require(by in ('windows','entries'),'unknown cache score')
    census.expected_membership(profile)  # Validate the complete header/reverse map.
    raw = Path(profile).read_bytes(); order = [[] for _ in range(48)]
    for l,e in struct.iter_unpack('<HH',raw[24:24+48*512*4]):order[l].append(e)
    result = []
    for l in range(48):
        def score(e):return sum(c*(n+1 if by=='entries' else 1) for n,c in enumerate(counts[l][e]))
        # Stable sort retains original layer-local rank for equal observations.
        result.append(sorted(order[l],key=lambda e:-score(e)))
    return result


def ranked_membership(counts, profile, by):
    import cpu_work_census as census
    quotas = [m.bit_count() for m in census.expected_membership(profile)]
    return [sum(1<<e for e in order[:cap]) for order,cap in
        zip(ranked_experts(counts,profile,by),quotas)]


def window_profile(counts, profile):
    """Change expert order only, preserving all original layer labels and bytes.

    The native per-layer loader keeps its original quotas/physical slot order.
    The complete reverse map is regenerated, including uncached experts.
    """
    # Also rejects malformed dimensions/counts before indexing or sorting them.
    encode_counts(counts)
    ranked = [iter(row) for row in ranked_experts(counts,profile,'windows')]
    raw = Path(profile).read_bytes()
    pairs = [(layer,next(ranked[layer])) for layer,_ in
        struct.iter_unpack('<HH',raw[24:24+48*512*4])]
    reverse = [0]*(48*512)
    for index,(layer,expert) in enumerate(pairs):reverse[layer*512+expert] = index
    return (raw[:24] + b''.join(struct.pack('<HH',*pair) for pair in pairs) +
        struct.pack('<'+'i'*(48*512),*reverse))


def counterfactual(counts, membership, formats):
    jobs, cpu, gpu, weight_bytes = [0]*8,0,0,0
    for l in range(48):
        for e,row in enumerate(counts[l]):
            entries = sum((n+1)*c for n,c in enumerate(row))
            if membership[l] & (1<<e):gpu += entries
            else:
                cpu += entries;weight_bytes += sum(row)*formats[l][2]
                for n,c in enumerate(row):jobs[n] += c
    return dict(cpu_jobs=sum(jobs),cpu_entries=cpu,vram_entries=gpu,jobs_NT=jobs,
        logical_job_blob_bytes=weight_bytes,
        membership_sha256=hashlib.sha256(b''.join(m.to_bytes(64,'little') for m in membership)).hexdigest())


def analyze(record, profile, manifest):
    """Fixed-route counterfactual only; changed residency may change future IDs.

    Compare original ranks, token-entry frequency and unique-window frequency
    with the same per-layer quotas. Only the latter may propose a future model
    plan, and only for a >=5% logical-byte reduction. That is not a speed gate.
    """
    import cpu_work_census as census
    require(record['passed'] and record['complete'] and record['ownership_closed'] and
        not record['active'],'closed qualified routing record required')
    for name,path in (('profile',profile),('manifest',manifest)):
        pin=record['rank_inputs'][name]
        require(Path(path).resolve() == Path(pin['path']).resolve() and
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == pin['sha256'],'ranking input changed: '+name)
    descriptor = record['per_expert_counts'];raw = Path(descriptor['path']).read_bytes()
    require(len(raw) == descriptor['bytes'] and hashlib.sha256(raw).hexdigest() == descriptor['sha256'],
        'expert count matrix changed')
    counts, formats = read_counts(raw),census.manifest_formats(manifest)
    expected = record['census']
    require(all(sum((n+1)*c for row in layer for n,c in enumerate(row)) == expected['rows']*10
        for layer in counts),'matrix layer/row conservation differs')
    masks = census.expected_membership(profile); base = counterfactual(counts,masks,formats)
    require(base['logical_job_blob_bytes'] > 0 and all(base[k] == expected[k] for k in base),'matrix/native census differs')
    values = {}
    for mode in ('entries','windows'):
        chosen = ranked_membership(counts,profile,mode)
        require([m.bit_count() for m in chosen] == [m.bit_count() for m in masks], 'cache quotas changed')
        values[mode] = counterfactual(counts,chosen,formats)
        values[mode]['logical_bytes_ratio'] = values[mode]['logical_job_blob_bytes']/base['logical_job_blob_bytes']
        values[mode]['replaced_pairs'] = sum((a&~b).bit_count() for a,b in zip(masks,chosen))
    require(values['windows']['logical_job_blob_bytes'] <= values['entries']['logical_job_blob_bytes'] and
        values['windows']['logical_job_blob_bytes'] <= base['logical_job_blob_bytes'],
        'window ranking violates fixed-quota minimum')
    sizes = [(z+255)//256*256 for _,_,z in formats]
    slots = [l for l,m in enumerate(masks) for _ in range(m.bit_count())]
    allocation = sum(sizes[l] for l in slots);prefix = sum(sizes[l] for l in slots[:128])
    return dict(kind='fixed-observed-route counterfactual;not model performance',baseline=base,rankings=values,
        unchanged_geometry=dict(slots=len(slots),allocated_bytes=allocation,prefix128_bytes=prefix,
            maximum_loan_bytes=allocation-prefix,actual_candidate_admission_qualified=False),
        decision=dict(can_propose_new_model_plan=values['windows']['logical_bytes_ratio']<=.95,
            prospective_logical_ratio_max=.95,clean_speed_qualified=False,adoption_qualified=False),
        limits='Training request only;no per-expert CPU service or physical traffic measured. Changed CPU/GPU residency may alter future routes/output. Any pilot needs a new committed plan,actual admission,phase preservation and both MTP workloads;adoption also requires independent workloads/agent/task/full physical262144/session/RESTORE.')
