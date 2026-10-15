#!/usr/bin/env python3
"""Closed exact raw controls; preserve old failed binary's distinct result."""
import fcntl
import json
from pathlib import Path
from check_gemm_orientation import B, D, sha


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = ['phase-audit-long-fixed-v1.json','phase-audit-long-auto-v1.json',
                 'boundary-audit-phase-off-long-auto-v2.json','boundary-audit-long-fixed-v1.json',
                 'boundary-audit-long-auto-v1.json']
        receipts = [json.loads((D / n).read_text()) for n in names]
        for d in receipts:
            assert d['complete'] and not d['active'] and not d['fresh_faults']
        comparisons = []
        for left, right in [(0,1),(1,2),(3,0),(4,2)]:
            a = Path(receipts[left]['arms'][0]['boundary']['raw_path'])
            b = Path(receipts[right]['arms'][0]['boundary']['raw_path'])
            raw_a, raw_b = a.read_bytes(), b.read_bytes()
            assert len(raw_a) == len(raw_b) == 37355520
            differences = []
            for i in range(912):
                start = i * 40960
                if raw_a[start:start+40960] != raw_b[start:start+40960]:
                    differences.append(dict(layer=i//19,row=receipts[left]['positions'][i%19]))
            comparisons.append(dict(left=names[left], right=names[right], equal=raw_a == raw_b,
                bytes_compared=len(raw_a), different_selected_rows=len(differences),
                first_difference=differences[0] if differences else None,
                left_sha256=sha(a), right_sha256=sha(b)))
        out = dict(complete=True, GPU_executed=False, comparisons=comparisons,
                   recipe_sha256=sha(Path(__file__)),
                   receipt_sha256={n:sha(D / n) for n in names}, old_failure_resolved=False,
                   new_same_binary_failure_reproduced=False, adopted=False,
                   scope='Exact raw19selectedrows x48layers only. Different binaries/diagnostics do not establish a causal correction.')
        (D / 'phase-boundary-controls-v1.json').write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps(dict(complete=True, comparisons=[c['equal'] for c in comparisons])))


if __name__ == '__main__': main()
