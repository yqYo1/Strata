"""Root-owned, CPU-only localization of an immutable failed full-context run."""
import fcntl
import hashlib
import json
import time
from pathlib import Path

import numpy as np

BASE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
RUN = BASE / 'owned-repeat-capture-v0141-two-full-diagnostic-r3'
REFERENCE = BASE / 'owned-upstream-v0141-integrated-full256k-diagnostic-r2/record.json'
REFERENCE_SHA = '168d7896cbe3a60dfc604a08b6fe310f16d823a17d13d8a013260bfc7facb0a4'
RECEIPT_SHA = 'cc0e49f0dde0642549a5b6ff7b9c2a8d03d7a32d60501dbae69bfeac306e48e3'
OUTPUT = BASE / 'repeat-capture-first-full-localization-v1.json'
start = time.monotonic()
lock = (BASE / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not OUTPUT.exists()
record_path = RUN / 'record.json'
assert hashlib.sha256(record_path.read_bytes()).hexdigest() == RECEIPT_SHA
record = json.loads(record_path.read_text())
assert not record['active'] and not record['completed'] and not record['healthy']
assert record['error'] == "ValueError('five-GEN capture request history incomplete or ledger differs')"
assert not record['diagnostic_sequence_completed'] and not record['capture_diagnostic_complete']
assert record['exit_code'] == 0 and record['exit_signal'] is None
assert record['capture_manifest']['structural_complete'] and record['capture_manifest']['record_count'] == 345
assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA
reference = json.loads(REFERENCE.read_text())
assert not reference['active'] and reference['completed'] and reference['healthy'] and reference['math_gate_passed']
assert not record['new_fault_messages'] and not any(record['cleanup'].values())
for owner in ['inferior', 'debugger']:
    assert not Path('/proc', str(record[owner]['pid'])).exists()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        name = (proc / 'comm').read_text().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    assert name not in ['strata', 'strata-xe-health', 'gdb', 'vtune', 'ninja'], (proc.name, name)
first = next(q for q in reference['requests'] if q['name'] == 'full256k-first')
repeat = next(q for q in record['requests'] if q['name'] == 'full256k-first')
assert first['math_gate_passed'] and not repeat['math_gate_passed']
assert first['input_tokens'] == repeat['input_tokens'] == 262140
result = {'scope': 'CPU-only byte-onset localization; no model/GPU execution or correctness relaxation',
          'receipt_sha256': RECEIPT_SHA, 'reference_receipt_sha256': REFERENCE_SHA, 'comparison': 'baseline first full vs D instrumented first full, not a repeated-request pair',
          'original_completed': False, 'original_healthy': False, 'original_request_math_gate_passed': False, 'source_controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'parts': [], 'gpu_executed': False, 'complete': False}
from compare_live_prefill_state_v01402_v2 import padding_ranges

for pa, pb in zip(first['prefill_state']['parts'], repeat['prefill_state']['parts']):
    assert time.monotonic() - start < 240
    assert pa['index'] == pb['index'] and pa['bytes'] == pb['bytes']
    idx, size = pa['index'], pa['bytes']
    detail = {'part': idx, 'bytes': size, 'first_sha256': pa['sha256'], 'repeat_sha256': pb['sha256']}
    if pa['sha256'] == pb['sha256']:
        detail.update(live_equal=True, different_live_bytes=0)
        result['parts'].append(detail)
        continue
    a = np.memmap(first['prefill_state']['file'], dtype=np.uint8, mode='r', offset=pa['offset'], shape=(size,))
    b = np.memmap(repeat['prefill_state']['file'], dtype=np.uint8, mode='r', offset=pb['offset'], shape=(size,))
    ha, hb = hashlib.sha256(), hashlib.sha256()
    first_diff = last_diff = None
    raw_count = live_count = 0
    ranges = padding_ranges(idx, size, 262139)
    groups = {}
    group_bytes = size // 36 if idx == 1 else size // 12 if idx in [3, 4, 5] else None
    for offset in range(0, size, 4 * 1024 * 1024):
        assert time.monotonic() - start < 240
        end = min(offset + 4 * 1024 * 1024, size)
        av, bv = a[offset:end], b[offset:end]
        ha.update(av); hb.update(bv)
        different = av != bv
        raw_count += int(np.count_nonzero(different))
        for left, right in ranges:
            if right > offset and left < end:
                different[max(0, left - offset):min(end - offset, right - offset)] = False
        n = int(np.count_nonzero(different))
        live_count += n
        if not n:
            continue
        at = np.flatnonzero(different) + offset
        if first_diff is None:
            first_diff = int(at[0])
        last_diff = int(at[-1])
        if group_bytes:
            keys, counts = np.unique(at // group_bytes, return_counts=True)
            for k, count in zip(keys, counts):
                key = str(int(k)); groups[key] = groups.get(key, 0) + int(count)
    assert ha.hexdigest() == pa['sha256'] and hb.hexdigest() == pb['sha256']
    detail.update(live_equal=not live_count, different_raw_bytes=raw_count, different_live_bytes=live_count,
                  first_live_different_byte=first_diff, last_live_different_byte=last_diff, ignored_padding=ranges)
    if groups:
        detail.update(group_bytes=group_bytes, different_bytes_by_ordinal=groups)
    if first_diff is not None and idx >= 6:
        kind = (idx - 6) % 5
        detail.update(qsa_ordinal=(idx - 6) // 5, kind=['k_int8', 'v_int8', 'k_scale_fp16', 'v_scale_fp16', 'indexer_pooled_fp32'][kind])
        for label, at in [('first', first_diff), ('last', last_diff)]:
            if kind == 4:
                detail[label + '_different_pool_block'] = at // 512
                detail[label + '_different_pool_component'] = at % 512 // 4
            else:
                row_bytes = 256 if kind < 2 else 8
                page, within = divmod(at, 2 * 4 * row_bytes)
                head, within = divmod(within, 4 * row_bytes)
                cell, component = divmod(within, row_bytes)
                detail[label + '_different_cell'] = page * 4 + cell
                detail[label + '_different_head'] = head
                detail[label + '_different_row_byte'] = component
        lo, hi = max(0, first_diff - 8), min(size, first_diff + 9)
        detail['first_byte_window'] = {'offset': lo, 'first': a[lo:hi].tolist(), 'repeat': b[lo:hi].tolist()}
    result['parts'].append(detail)
    del a, b

for p in [first['first_head'], repeat['first_head']]:
    assert hashlib.sha256(Path(p['file']).read_bytes()).hexdigest() == p['sha256']
a = np.fromfile(first['first_head']['file'], dtype=np.float32)
b = np.fromfile(repeat['first_head']['file'], dtype=np.float32)
result['head_difference'] = {'finite': bool(np.isfinite(a).all() and np.isfinite(b).all()),
                             'different_values': int(np.count_nonzero(a != b)),
                             'max_absolute_difference': float(np.max(np.abs(a - b))),
                             'mean_absolute_difference': float(np.mean(np.abs(a - b))),
                             'first_different_index': int(np.flatnonzero(a != b)[0])}
assert hashlib.sha256(record_path.read_bytes()).hexdigest() == RECEIPT_SHA
assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA
result.update(complete=True, elapsed_seconds=time.monotonic() - start)
OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'complete': True, 'gpu_executed': False, 'elapsed_seconds': result['elapsed_seconds'],
                  'first_different_qsa': next(p for p in result['parts'] if p.get('qsa_ordinal') is not None and not p['live_equal']),
                  'gdn_ordinal_difference_counts': result['parts'][1].get('different_bytes_by_ordinal'),
                  'head_difference': result['head_difference']}))
