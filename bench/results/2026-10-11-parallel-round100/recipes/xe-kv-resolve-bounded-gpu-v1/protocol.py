"""Finite production KV probe receipt parsing only, no Runtime/GPU operations."""
import json
from pathlib import Path

EXPECTED = json.loads(Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/xe-kv-resolve-bounded-source-v2/manifest.json').read_text())['expected_success_counts']

def require(value, message):
    if not value:
        raise ValueError(message)

def fields(line):
    out = {}
    for part in line.split()[1:]:
        key, value = part.split('=', 1)
        require(key not in out, 'duplicate field')
        out[key] = value if key == 'format' else int(value)
    return out

def validate(stdout, stderr):
    prefixes = ('FORMAT_BEGIN ', 'FORMAT_ALLOCATED ', 'RESOLVE_BEGIN ', 'RESOLVE_DONE ', 'FORMAT_PASS ', 'FINAL_COUNTS ', 'PASS ')
    lines = [x for x in stdout.splitlines() if x.startswith(prefixes)]
    require(len(lines) == 33, 'exact 33 progress/result lines')
    at = 0
    done = []
    formats = {}
    allocated = {}
    for name in ['fp16', 'int8']:
        require(lines[at] == 'FORMAT_BEGIN format=%s slots=8192 blocks=8256 nq=2,3,4,2,3,4 resolves=6' % name, 'sequential format start')
        at += 1
        require(lines[at].startswith('FORMAT_ALLOCATED '), 'allocation evidence')
        info = fields(lines[at]);at += 1
        require(info['format'] == name and 0 < info['usm_bytes'] <= info['live_buffer_bytes_upper'] < info['limit'] == 67108864, 'probe-only live payload bound')
        allocated[name] = info
        for i, misses in enumerate([36, 8, 8, 0, 4, 8]):
            nq = 2 + i % 3
            require(lines[at] == 'RESOLVE_BEGIN format=%s pass=%d nq=%d expected_misses=%d' % (name, i+1, nq, misses), 'finite resolve start')
            at += 1
            require(lines[at].startswith('RESOLVE_DONE '), 'checked resolve completion')
            values = fields(lines[at]);at += 1
            require(values == dict(format=name, **{'pass':i+1}, nq=nq, misses=misses, map_entries=16448, pool_compares=16384 if name == 'fp16' else 32768, oracle_bytes=16777216 if name == 'fp16' else 8650752, selected_cells=nq*512), 'each exact completed map/byte/count oracle')
            done.append(values)
        require(lines[at].startswith('FORMAT_PASS '), 'format terminal drain/release receipt')
        values = fields(lines[at]);at += 1
        require(values == dict(format=name, **EXPECTED[name]), 'per-format precise checked ledger')
        formats[name] = values
    for name in ['fp16', 'int8']:
        require(lines[at].startswith('FINAL_COUNTS '), 'final format ledger')
        require(fields(lines[at]) == formats[name], 'final and format ledgers equal');at += 1
    expected_total = dict(formats=2, resets=2, resolves=12, uploads=38, readbacks=72, direct_copies=110, direct_copy_bytes=179147776, drains=16, released=30, map_entries=197376, selected_cells=18432, pool_compares=294912, oracle_bytes=152567808, lookups=4608, missed_pages=128, production_copy_runs=384, production_copy_bytes=198656)
    require(fields(lines[at]) == expected_total and lines[at].startswith('PASS '), 'all exact terminal counts');at += 1
    require(at == len(lines), 'finite trace exhausted')
    for error in ['FORMAT_FAIL', 'FAIL:', 'FAIL before', 'GPU completion unconfirmed', 'UR_RESULT_ERROR', 'ZE_RESULT_ERROR', 'terminate called']:
        require(error not in stdout + stderr, 'no unclassified failure ' + error)
    return dict(resolve_results=done, formats=formats, allocations=allocated, totals=expected_total, complete_pool_byte_oracle=True, map_inverse_unique_verified=True, scope='FP16+INT8 copying/resolution only; no attention/fullcontext/RESTORE/globalmodel/performance/hang-causality proof')
