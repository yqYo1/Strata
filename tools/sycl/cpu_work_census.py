#!/usr/bin/env python3
"""Read bounded native CPU-job summaries; counts are logical work, not bus bytes."""
import hashlib
import math
from pathlib import Path
import re
import struct

import measurement_guards as guards

PREFIX = 'strata cpu work: '
BEGIN = {'request', 'phase', 'prompt_tokens', 'layers', 'experts', 'cache'}
CACHE = {'request', 'layer', 'bin', 'resident_bits'}
BIN = {'request', 'bin', 'gu', 'down', 'H', 'FF', 'blob_bytes', 'gu_act', 'down_act',
       'gu_row', 'down_row', 'act_bytes', 'h_bytes', 'calls', 'pool_ns', 'CPU', 'VRAM', 'PCIe', 'peer', 'jobs_NT'}
END = {'request', 'phase', 'valid', 'windows', 'rows', 'callbacks', 'k', 'cpu_jobs',
       'CPU', 'VRAM', 'PCIe', 'peer', 'pcie_jobs', 'error'}


def integer(value):
    guards.require(re.fullmatch(r'\d+', value) is not None, 'census noninteger')
    n = int(value)
    guards.require(n <= (1 << 64) - 1, 'census uint64 overflow')
    return n


def expected_membership(profile, slots=1256):
    """Exact per-layer ranked admission, including the last layer's remainder."""
    raw = Path(profile).read_bytes()
    guards.require(len(raw) >= 24 and raw[:4] == b'STRP', 'profile header missing')
    version, layers, experts, _, count = struct.unpack_from('<5I', raw, 4)
    guards.require((version, layers, experts, count) == (1, 48, 512, 48 * 512) and
        len(raw) == 24 + count * 8 and 0 < slots <= count, 'unsupported/incomplete profile')
    pairs = list(struct.iter_unpack('<HH', raw[24:24 + count * 4]))
    guards.require(len(set(pairs)) == count and all(l < 48 and e < 512 for l, e in pairs), 'profile pairs invalid')
    # Verify the supplied reverse map too; a partial valid ranked prefix is not
    # evidence that this is the same complete input the native loader admitted.
    table = struct.unpack_from('<' + 'i' * count, raw, 24 + count * 4)
    guards.require(all(table[l * 512 + e] == i for i, (l, e) in enumerate(pairs)), 'profile reverse map differs')
    caps = [slots // 48] * 47 + [slots // 48 + slots % 48]
    selected = [set() for _ in range(48)]
    for l, e in pairs:
        if len(selected[l]) < caps[l]: selected[l].add(e)
    guards.require([len(s) for s in selected] == caps, 'profile admission incomplete')
    return [sum(1 << e for e in row) for row in selected]


def manifest_formats(path):
    result = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        fields = line.split()
        guards.require(len(fields) >= 5 and int(fields[0]) == len(result), 'native manifest layer order')
        result.append(tuple(int(fields[i]) for i in (1, 2, 4)))
    guards.require(len(result) == 48, 'complete native manifest required')
    return result


def rows(text):
    result = []
    for line in text.splitlines():
        if not line.startswith(PREFIX): continue
        fields = line[len(PREFIX):].split()
        guards.require(fields and all(x.count('=') == 1 for x in fields[1:]), 'malformed census row')
        values = dict(x.split('=', 1) for x in fields[1:])
        expected = {'begin': BEGIN, 'cache': CACHE, 'bin': BIN, 'end': END}.get(fields[0])
        guards.require(expected is not None and set(values) == expected and len(values) == len(fields) - 1,
            'unknown/duplicate/missing census fields')
        result.append((fields[0], values))
    return result


def extract(text, bindings, formats, membership):
    """Bindings must come from the same request's verified native/client clocks.

    The native state machine verifies every window/layer and final job map.
    This reader independently conserves their aggregate framing, format/NT,
    static residency, existing footer counts and existing pool wall clocks.
    """
    guards.require(len(formats) == len(membership) == 48, 'census model geometry')
    records = rows(text); offset = 0; result = []
    for request, binding in enumerate(bindings, 1):
        guards.require(offset < len(records) and records[offset][0] == 'begin', 'missing census request start')
        begin = records[offset][1]; offset += 1
        guards.require(begin == dict(request=str(request), phase='decode', prompt_tokens=str(binding['prompt_tokens']),
            layers='48', experts='512', cache='restored_decode_snapshot_immutable'), 'census request binding differs')
        layer_bins = []
        for l in range(48):
            guards.require(offset < len(records) and records[offset][0] == 'cache', 'missing census cache row')
            row = records[offset][1]; offset += 1
            guards.require(integer(row['request']) == request and integer(row['layer']) == l and
                re.fullmatch('[0-9a-f]{128}', row['resident_bits']) is not None, 'census cache framing')
            # Each 64-bit word prints most-significant nibble first; words run
            # in increasing expert index, so int(the whole string,16) is wrong.
            bits = sum(int(row['resident_bits'][i:i+16], 16) << (4*i) for i in range(0, 128, 16))
            guards.require(bits == membership[l], 'actual restored cache membership differs')
            layer_bins.append(integer(row['bin']))
        bins = []
        while offset < len(records) and records[offset][0] == 'bin':
            row = records[offset][1]; offset += 1
            guards.require(integer(row['request']) == request and integer(row['bin']) == len(bins), 'census bin identity')
            counts = [integer(x) for x in row['jobs_NT'].split(',')]
            guards.require(len(counts) == 8 and not any(counts[4:]), 'actual spec4 CPU NT range differs')
            values = {k: integer(v) for k, v in row.items() if k != 'jobs_NT'}
            layers = [l for l, b in enumerate(layer_bins) if b == len(bins)]
            guards.require(layers and values['H'] == 2560 and values['FF'] == 640 and
                all(formats[l] == (values['gu'], values['down'], values['blob_bytes']) for l in layers) and
                values['calls'] == binding['windows'] * len(layers) and
                values['CPU'] == sum((n+1)*count for n, count in enumerate(counts)) and
                values['CPU'] + values['VRAM'] == binding['rows'] * 10 * len(layers) and
                values['PCIe'] == values['peer'] == 0, 'format/NT/route/callback conservation differs')
            bins.append(dict(values, jobs_NT=counts, layer_count=len(layers)))
        guards.require(set(layer_bins) == set(range(len(bins))) and offset < len(records) and
            records[offset][0] == 'end', 'missing census bins/end')
        end = records[offset][1]; offset += 1
        guards.require(end['phase'] == 'decode' and end['error'] == 'none', 'native census invalid/partial')
        summary = {k: integer(v) for k, v in end.items() if k not in ('phase', 'error')}
        jobs = [sum(b['jobs_NT'][n] for b in bins) for n in range(8)]
        guards.require(summary == dict(request=request, valid=1, windows=binding['windows'], rows=binding['rows'],
            callbacks=binding['windows']*48, k=10, cpu_jobs=binding['cpu_jobs'], CPU=binding['cpu_entries'],
            VRAM=binding['rows']*48*10-binding['cpu_entries'], PCIe=0, peer=0, pcie_jobs=0) and
            sum(jobs) == summary['cpu_jobs'] and sum(b['CPU'] for b in bins) == summary['CPU'] and
            sum(b['VRAM'] for b in bins) == summary['VRAM'], 'census/native-footer conservation differs')
        pool_ms = sum(b['pool_ns'] for b in bins)/1e6
        # The old clock is rounded to .01ms/window; each integer-ns duration
        # loses <1ns. Neither number is process CPU time or useful arithmetic.
        tolerance = .005*binding['windows'] + summary['callbacks']/1e6 + 1e-8
        guards.require(math.isfinite(binding['pool_ms']) and abs(pool_ms-binding['pool_ms']) <= tolerance,
            'census pool clocks differ from existing host scope')
        encoded = b''.join(bits.to_bytes(64, 'little') for bits in membership)
        result.append(dict(request=request, windows=summary['windows'], rows=summary['rows'],
            cpu_jobs=summary['cpu_jobs'], cpu_entries=summary['CPU'], vram_entries=summary['VRAM'],
            jobs_NT=jobs, NT1_fraction=jobs[0]/sum(jobs) if sum(jobs) else 0,
            format_bins=bins, pool_ms=pool_ms, pool_clock_rounding_bound_ms=tolerance,
            membership_sha256=hashlib.sha256(encoded).hexdigest(),
            logical_job_blob_bytes=sum(sum(b['jobs_NT'])*b['blob_bytes'] for b in bins),
            limits='Actual target-verifier jobs/entries, excluding separate draft. Blob visits are logical,not physical RAM/PCIe traffic. Pool wall includes scheduling/waits. Diagnostic only; no clean speed or hardware saturation qualification.'))
    guards.require(offset == len(records), 'unbound trailing census records')
    return result
