#!/usr/bin/env python3
"""Read the pinned small pack table; derive widths without opening model data."""
import datetime
import fcntl
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = A / 'layer-upload-width-census-v1.json'
    if destination.exists():
        raise RuntimeError('Preserve the original receipt; use a fresh version.')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current_path = A / 'own-qsa-bf16-prefix-tilepack-clean-bf16-prefix-optimized-r0-v2.json'
        if sha(current_path) != 'f279fd81aef8cab999a7122a23e5e5bc6cf94b4b088a8bae6afabe2a354361ba':
            raise RuntimeError('Current closed observation changed.')
        current = json.loads(current_path.read_text())
        if not (current['passed'] and current['complete'] and current['ownership_closed'] and not current['active']):
            raise RuntimeError('Closed current observation required.')
        metadata, = [Path(n) for n in current['pack_metadata_sha256']
                     if Path(n).name == 'native_experts.txt']
        if metadata.stat().st_size > 65536 or sha(metadata) != current['pack_metadata_sha256'][str(metadata)]:
            raise RuntimeError('Small pinned metadata required.')
        text = metadata.read_text()
        version = re.search(r'^# strata native experts v(\d+):', text)
        header = re.search(r'\(n_expert (\d+), total (\d+);', text)
        if not version or int(version[1]) != 3 or not header or int(header[1]) != 512:
            raise RuntimeError('Expected the measured v3, 512-expert pack.')
        rows = []
        offset = 0
        for line in text.splitlines():
            if not line or line.startswith('#'):
                continue
            values = line.split()
            layer, gu, down, start, width = map(int, values[:5])
            if layer != len(rows) or start != offset or width <= 0 or len(values) not in (8, 9):
                raise RuntimeError('Unexpected metadata row or byte conservation.')
            rows.append(dict(layer=layer, gu_type=gu, down_type=down,
                             pack_offset=start, blob_bytes=width))
            offset += 512 * width
        if len(rows) != 48 or offset != int(header[2]):
            raise RuntimeError('Expected all 48 layers and the exact pack payload.')
        maximum = max(row['blob_bytes'] for row in rows)
        stride = (maximum + 255) // 256 * 256
        budget_path = A / 'native16-chunk-budget-v1.json'
        if sha(budget_path) != '45446a13ed369a6f22166dee96ffc8d636876e319a4f6c9fe77d7580ce106ef5':
            raise RuntimeError('Reviewed arena budget changed.')
        budget = json.loads(budget_path.read_text())
        if stride * 512 != budget['observation']['cache_bytes']:
            raise RuntimeError('Metadata stride differs from observed cache bytes.')
        for row in rows:
            row.update(stride=stride, exact_payload_bytes=512 * row['blob_bytes'],
                       padding_bytes=512 * (stride - row['blob_bytes']),
                       exact_contiguous_bulk_eligible=row['blob_bytes'] == stride)
        summary_path = A / 'qsa-bf16-prefix-profile-summary-v1.json'
        prior = json.loads(summary_path.read_text())
        if not prior['passed'] or not prior['complete'] or prior['coarse']['rows'][-1]['upload_bytes'] != offset:
            raise RuntimeError('Metadata total differs from the prior closed profile.')
        result = dict(schema='strata-layer-upload-width-census-v1', passed=True, complete=True,
            recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            recipe_sha256=sha(__file__),
            execution_tree=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
            inputs={str(f):sha(f) for f in (current_path, metadata, budget_path, summary_path)},
            source_pins={n:sha(P/n) for n in ('src/prefill/layer_major.inc',
                'src/core/expert_cache.cpp', 'src/kernels/cpu/expert_layout.cpp',
                'include/strata/kernels/cpu/expert_layout.hpp', 'src/sycl/compat.cpp')},
            layers=rows, experts_per_layer=512, stride=stride,
            exact_total_payload_bytes=offset, full_stride_payload_bytes=48*512*stride,
            width_histogram=dict(sorted(Counter(r['blob_bytes'] for r in rows).items())),
            equal_width_layers=[r['layer'] for r in rows if r['exact_contiguous_bulk_eligible']],
            limits='Metadata/source census only. The same table was pinned by the closed current model observation; native_fmt was not re-executed. No model tensor, CPU/GPU engine, compiler or native copy was run. Equality establishes payload contiguity, not native command count, pointer eligibility, completion time or speed.',
            GPU_executed=False, native_compiler_executed=False, performance_adopted=False)
        destination.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k:result[k] for k in ('equal_width_layers', 'width_histogram',
            'exact_total_payload_bytes', 'full_stride_payload_bytes')}))


if __name__ == '__main__':
    main()
