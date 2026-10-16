#!/usr/bin/env python3
"""Independently check the actual proposed C++ serializer's synthetic output."""
from collections import Counter
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A / 'layermajor-wave-census-v2'))
from validate_wave_census import validate_wave_fields


def need(ok, why):
    if not ok:
        raise RuntimeError(why)


def expected_layer(n, chunk, layer, rotate):
    chunks = []
    for ordinal, off in enumerate(range(0, n, chunk)):
        t = min(chunk, n - off)
        ids = {(layer * 7 + ordinal * 13 + k * 31) % 512 for k in range(10)} if rotate else set(range(10))
        need(len(ids) == 10, 'reference route uniqueness')
        chunks.append((t, {e: t for e in ids}))
    exact_hist = Counter()
    for t, ids in chunks:
        exact_hist[0] += 512 - len(ids)
        exact_hist[t] += len(ids)
    waves = []
    for q in (2, 4):
        unions, all_rows, duplicate, partial_chunks, partial_tokens = [], [], 0, 0, 0
        for begin in range(0, len(chunks), q):
            group = chunks[begin:begin + q]
            rows = Counter()
            incidences = 0
            for t, ids in group:
                rows.update(ids)
                incidences += len(ids)
            unions.append(len(rows))
            all_rows.extend(rows.values())
            duplicate += incidences - len(rows)
            if len(group) < q:
                partial_chunks, partial_tokens = len(group), sum(t for t, _ in group)
        hist = Counter(x.bit_length() - 1 for x in all_rows)
        waves.append(dict(width_chunks=q, wave_count=len(unions),
            partial_wave_count=int(bool(partial_chunks)), partial_chunks=partial_chunks,
            partial_tokens=partial_tokens, union_sum=sum(unions), union_min=min(unions), union_max=max(unions),
            active_incidences=10 * len(chunks), duplicate_incidences=duplicate,
            expert_rows=10 * n, max_wave_ne=max(all_rows), expert_wave_rows_hist=[hist[i] for i in range(16)]))
    return [[ne, count, count if ne else 0, count if ne else 0] for ne, count in sorted(exact_hist.items())], waves


def check(path):
    need(path.is_file() and path.stat().st_size <= 8 << 20, 'bounded serializer output required')
    lines = path.read_text().splitlines()
    need(len(lines) == 6 and lines[0].startswith('wave-host-meta: '), 'exact output cases/metadata')
    meta = json.loads(lines[0].split(': ', 1)[1])
    need(type(meta['owner_bytes']) is int and 0 < meta['owner_bytes'] <= 4 << 20, 'actual owner size')
    need(meta['output_cap_bytes'] == 1 << 20, 'original output budget')
    results = []
    shapes = [(1, 0, 1, False), (13, 17, 3, True), (32769, 0, 8192, True),
              (262144, 0, 8192, False), (262127, 17, 8192, True)]
    for line, (n, position, chunk, rotate) in zip(lines[1:], shapes):
        need(line.startswith('strata lm census: ') and len(line.encode()) + 1 <= 1 << 20, 'complete bounded census line')
        d = json.loads(line.split(': ', 1)[1])
        need((d['tokens'], d['position'], d['configured_chunk']) == (n, position, chunk), 'actual fixture geometry')
        need(d['issuer_exact_ne_verified'] is True and d['hardware_cache_residency_measured'] is False, 'diagnostic scope')
        validate_wave_fields(d)
        for layer, row in enumerate(d['layers']):
            hist, waves = expected_layer(n, chunk, layer, rotate)
            need(row['histogram'] == hist and row['waves'] == waves, 'independent reference mismatch')
            need(row['gu_rows'] == row['down_rows'] == 10 * n, 'actual serialized old row counters')
        results.append(dict(tokens=n, position=position, chunk=chunk, rotate=rotate,
                            serialized_bytes=len(line.encode()) + 1, layers_checked=48, passed=True))
    return dict(passed=True, metadata=meta, cases=results,
        scope='CPU synthetic routes and serialization, including absolute 262144 boundary; no physical model, GPU, quality or RESTORE result.')


if __name__ == '__main__':
    need(len(sys.argv) == 2, 'usage: check_wave_census_serialization_v1.py OUTPUT')
    print(json.dumps(check(Path(sys.argv[1])), sort_keys=True))
