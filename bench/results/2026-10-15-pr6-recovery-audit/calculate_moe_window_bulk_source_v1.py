#!/usr/bin/env python3
"""Source arithmetic for one window-wide source-index upload, not a fit test."""
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
base = A / 'moe_window_contract_v1.py'
assert hashlib.sha256(base.read_bytes()).hexdigest() == 'e85ec84e890b2d6bdb2823b8da02ae2f1091e4e4af89b2d52e6da244bbc7f64a'
spec = importlib.util.spec_from_file_location('moe_base', base)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)

with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = A / 'moe-window-bulk-source-arithmetic-v1.json'
    assert not out.exists()
    previous = module.memory_plan()
    old = previous['region']
    views, used = module.carve([
        (name, module.WINDOW * module.TOPK * 4 if name == 'expert_src_i32' else span.payload)
        for name, span in old.items()], module.REGION)
    assert views['expert_src_i32'].payload == 655360
    assert used == 998443520 and module.REGION - used == 182255104
    assert used - previous['region_used'] == 622592
    for name, span in old.items():
        if name != 'expert_src_i32':
            assert views[name] == span
    record = dict(schema='strata-moe-window-bulk-source-arithmetic-v1',
        status='source_arithmetic_only', GPU_executed=False, runtime_fit_proven=False,
        recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        region={name:asdict(span) for name,span in views.items()},
        region_used_bytes=used, region_capacity_bytes=module.REGION,
        region_arithmetic_spare_bytes=module.REGION-used, extra_source_reservation_bytes=622592,
        maximum_source_upload_bytes=655360,
        decision='Use one full-window source upload and gather from device_source+job.global_row. '
                 'This is a true W*10 owner, not an offset into the old 8K job buffer. '
                 'Keep each GEMM/gather output job <=8192 rows. Keep host source and device view '
                 'alive through checked window drain.',
        API_count_only='One source H2D submission per window instead of one per expert job; '
                       'same total source-index bytes. Actual driver commands, staging, latency '
                       'and throughput remain unmeasured.',
        preserved='Original source-rebased32KiB contract and all host receipts remain immutable; '
                  'this prospective engine integration choice does not retrospectively change their scope.')
    out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('status','region_used_bytes',
        'region_arithmetic_spare_bytes','extra_source_reservation_bytes')}))
