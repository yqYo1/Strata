#!/usr/bin/env python3
"""Exact selected phase comparison with numeric magnitudes, never a quality gate."""
import fcntl
import importlib.util
import json
import math
import struct
from pathlib import Path
from check_gemm_orientation import B, D, sha


def values(body, dtype):
    if dtype == 'bf16':
        return [struct.unpack('<f', struct.pack('<I', u[0] << 16))[0]
                for u in struct.iter_unpack('<H', body)]
    code = {'fp16': 'e', 'fp32': 'f', 'int32': 'i'}[dtype]
    return [v[0] for v in struct.iter_unpack('<' + code, body)]


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = [D / ('phase-audit-long-' + arm + '-v1.json') for arm in ['fixed', 'auto']]
        receipts = [json.loads(p.read_text()) for p in paths]
        for d in receipts:
            assert d['complete'] and not d['active'] and not d['fresh_faults']
            assert d['boot_unchanged'] and not d['driver_dump_exists']
        assert receipts[0]['source_pins'] == receipts[1]['source_pins']
        assert receipts[0]['binary_sha256'] == receipts[1]['binary_sha256']
        spec = importlib.util.spec_from_file_location('phase', D / 'compare_layer26_phase_v2.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        captures = [module.read(d['arms'][0]['phase']['text_path']) for d in receipts]
        left, right = captures
        assert left[0] == right[0] and list(left[1]) == list(right[1])
        assert left[0]['byteorder'] == 'little'
        results = []
        for key, (meta, a) in left[1].items():
            other, b = right[1][key]; assert meta == other
            row = dict(record=int(meta['record']), row=key[0], tensor=key[1],
                       dtype=meta['dtype'], bytes=len(a), equal=a == b)
            if a != b:
                row['first_different_byte'] = next(i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y)
                va, vb = values(a, meta['dtype']), values(b, meta['dtype'])
                row['different_values'] = sum(x != y for x, y in zip(va, vb, strict=True))
                row['nonfinite_fixed'] = sum(not math.isfinite(x) for x in va)
                row['nonfinite_auto'] = sum(not math.isfinite(x) for x in vb)
                finite = [(x,y) for x,y in zip(va,vb,strict=True) if math.isfinite(x) and math.isfinite(y)]
                row['max_abs_finite'] = max((abs(x-y) for x,y in finite), default=None)
                norm = sum(x*x for x,y in finite)
                row['relative_l2_finite'] = math.sqrt(sum((x-y)**2 for x,y in finite) / norm) if norm else None
                if meta['dtype'] == 'int32':
                    row['fixed_values'] = va; row['auto_values'] = vb
            results.append(row)
        different = [r for r in results if not r['equal']]
        out = dict(GPU_executed=False, complete=True, numerical_equal=not different,
                   first_selected_operation_difference=different[0] if different else None,
                   results=results, receipt_sha256={p.name:sha(p) for p in paths},
                   binary_sha256=receipts[0]['binary_sha256'], captures=[x[2] for x in captures],
                   adopted=False, performance_qualified=False, quality_qualified=False,
                   scope='Same-binary selected layer26 coarse boundaries only. Matching selected inputs does not establish full recurrent-input equality. Diagnostic drains can mask races.')
        (D / 'layer26-phase-pair-v1.json').write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps({k: out[k] for k in ['complete','numerical_equal','first_selected_operation_difference']}))


if __name__ == '__main__': main()
