#!/usr/bin/env python3
"""Reweight an existing routed census; no benchmark or new workload is run."""
from pathlib import Path
import gzip
import hashlib
import json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    source = here.parent / ('2026-10-14-primary-routed-fp16-census/receipts/'
                           'main-primary-long-routed-phase-root-v3/record.json.gz')
    result = json.loads(gzip.decompress(source.read_bytes()))
    assert result['passed'] and result['complete'] and not result['active']
    assert not result['fresh_faults'] and not result['driver_dump_exists']
    data = result['routed_observation']
    assert data['effective'] and data['run_return'] == 'success'
    hist = data['hist_ne_count']
    assert len({n for n, count in hist}) == len(hist)
    assert all(isinstance(n, int) and isinstance(count, int) and n >= 0 and count >= 0
               for n, count in hist)
    calls = sum(count for n, count in hist if n > 0)
    rows = sum(n * count for n, count in hist)
    assert calls == data['gu_calls'] == data['down_calls']
    assert rows == data['gu_rows'] == data['down_rows']
    gu_bytes, down_bytes = 2 * 640 * 2560 * 2, 2560 * 640 * 2
    bands = []
    for limit in (16, 32, 64, 128):
        c = sum(count for n, count in hist if 0 < n <= limit)
        r = sum(n * count for n, count in hist if 0 < n <= limit)
        bands.append(dict(max_ne=limit, expert_calls_per_role=c, rows_per_role=r,
            call_fraction=c / calls, row_fraction=r / rows,
            logical_fp16_weight_outputs_bytes=c * (gu_bytes + down_bytes)))
    output = dict(schema='dequant-shape-coverage-v1',
        status='EXISTING_CENSUS_OFFLINE_REWEIGHT_ONLY',
        source=str(source.relative_to(here.parent.parent.parent)), source_sha256=sha(source),
        recipe_sha256=sha(Path(__file__)),
        census_binary_sha256=result['binary_sha256'],
        input_batched_tokens=data['input_tokens'], chunks=data['chunks'],
        expert_calls_per_role=calls, rows_per_role=rows,
        expanded_gu_bytes_per_call=gu_bytes, expanded_down_bytes_per_call=down_bytes,
        all_logical_fp16_weight_outputs_bytes=calls * (gu_bytes + down_bytes),
        threshold_coverage=bands,
        current_layer_major_coverage_qualified=False, per_format_ne_joint_histogram_available=False,
        performance_qualified=False,
        limitations=[
            'Historical chunk-major census differs from current layer-major/profile source and workload.',
            'Global ne histogram and per-layer formats do not establish a format x ne joint histogram.',
            'Logical outputs are not physical DRAM traffic, device service or achievable speed savings.',
            'Hypothetical direct tiles need actual type/shape admission; custom Down42 needs its own support.',
            'Generic dequant profile also includes other projections, not only expert Down.',
        ])
    (here / 'dequant-shape-coverage-v1.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(calls=calls, rows=rows, coverage=bands)))


if __name__ == '__main__':
    main()
