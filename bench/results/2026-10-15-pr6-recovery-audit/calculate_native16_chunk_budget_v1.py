#!/usr/bin/env python3
"""Source arithmetic at a pinned, closed 39K observation; no runtime probing."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess

A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
N, HC, D, LR, K, NE = 2560, 4, 10240, 320, 10, 512


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def take(payload):
    return (payload + 256 + 255) // 256 * 256


def layout(tokens, blob_stride):
    prefix_payloads = [tokens * count * element for count, element in
                       [(N, 4), (D, 4), (HC, 4), (D, 2), (LR, 4),
                        (LR, 2), (D, 4), (HC, 4), (N, 2), (N, 2)]]
    gdn = sum(take(tokens * count * element) for count, element in
              [(10240, 4), (6144, 4), (96, 4), (48, 4),
               (48, 4), (10240, 4), (6144, 4), (6144, 2)])
    # Conservative QSA upper bound for the same model through max_cells262144:
    # nine token-dependent takes plus sel_batch256 and attn_batch32 scratch.
    cap, max_blocks = 2051, 262144 // 4 + 2
    qsa_payloads = [tokens * width * element for width, element in
                    [(512, 4), (512, 4), (12288, 4), (6144, 4),
                     (128, 4), (512, 4), (6144, 4), (6144, 2), (cap, 4)]]
    qsa_payloads += [256 * max_blocks * 4,
                     32 * (((cap + 63) // 64) * 24 * (256 + 2) + 64) * 4]
    qsa = sum(map(take, qsa_payloads))
    moe = sum(take(tokens * count * element) for count, element in
              [(NE, 4), (K, 4), (K, 4), (K, 4), (K, 4), (N, 2),
               (1280, 4), (640, 2), (K * N, 4), (640, 4), (640, 4),
               (640, 2), (N, 4), (1, 4)])
    assert gdn >= max(qsa, moe), 'shared-region family changed'
    entries = dict(gemm_scratch=take((32 << 20) * 2),
                   prefix=sum(map(take, prefix_payloads)),
                   steps=take(tokens * 4 * 4), shared_region=take(gdn),
                   dequant_ring=2 * (take(1280 * 2560 * 2) + take(2560 * 640 * 2)),
                   upload_ring=8 * (blob_stride + 256), ple_norm=take(D * 4))
    return dict(chunk=tokens, accounted=sum(entries.values()),
                direct_allocation_families=entries, tok_dev=tokens * 4,
                gdn_inner=gdn, qsa262144_inner_upper_bound=qsa, moe_inner=moe,
                largest_single_allocation=take(gdn))


def main():
    out = A/'native16-chunk-budget-v1.json'
    assert not out.exists()
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = A/'own-qsa-bf16-prefix-tilepack-clean-bf16-prefix-optimized-r0-v2.json'
        observation = json.loads(receipt.read_text())
        assert observation['passed'] and observation['complete'] and observation['ownership_closed'] and not observation['active']
        assert not observation['fresh_faults'] and not observation['driver_dump_exists']
        log = Path(observation['config']['log'])
        assert log.stat().st_size < 16 << 20
        assert sha(log) == observation['native_log_sha256']
        text = log.read_text()
        free = int(re.search(r'phase=after-detach free_bytes=(\d+)', text)[1])
        accounted = int(re.search(r'chunk=8192,.*?accounted=(\d+)', text)[1])
        placement = observation['placement_lines'][0]
        fields = {key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', placement)}
        assert fields['tokens'] == 39261 and fields['residual_gpu_rows'] == 32768
        prefix = int(re.search(r'transient_bytes=(\d+)', text)[1])
        stride = fields['layer_cache'] // NE
        assert stride * NE == fields['layer_cache'] and stride % 256 == 0
        small, large = (layout(tokens, stride) for tokens in (8192, 16384))
        assert small['accounted'] == accounted == 2311569920
        delta = large['accounted'] - small['accounted'] + large['tok_dev'] - small['tok_dev']
        rows = []
        for chunk, increment in ((8192, 0), (16384, delta)):
            for gpu_rows in (32768, fields['tokens']):
                residual = gpu_rows * D * 2
                need = fields['layer_cache'] + prefix + residual + (512 << 20)
                margin = free - increment - need
                rows.append(dict(chunk=chunk, GPU_rows=gpu_rows, residual_bytes=residual,
                                 extra_prefill_allocation_delta=increment,
                                 known_request_owners_plus_headroom=need,
                                 arithmetic_margin_bytes=margin,
                                 arithmetic_margin_MiB=margin / (1 << 20)))
        sources = ['src/prefill/prefill.cpp', 'src/prefill/layer_major.inc',
                   'src/prefill/sycl/qsa_native16.cpp', 'src/program/generate.cpp',
                   'include/strata/kernels/qsa.hpp', 'src/kernels/sycl/decode_attention.cpp',
                   'src/sycl/compat.cpp']
        result = dict(schema='strata-native16-chunk-budget-v1', passed=True, complete=True,
                      recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      recipe_sha256=sha(__file__), source_commit=subprocess.check_output(
                          ['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
                      source_pins={name: sha(P/name) for name in sources},
                      observation=dict(path=str(receipt), sha256=sha(receipt),
                                       source_commit=observation['source_commit'],
                                       binary_sha256=observation['binary_sha256'],
                                       native_log_sha256=sha(log), free_after_detach=free,
                                       cache_bytes=fields['layer_cache'], prefix_bytes=prefix,
                                       blob_rounded_stride=stride, accounted=accounted),
                      layouts=[small, large], scenarios=rows,
                      conditions='Same39K/context65536 Mode0/native16BF16/alias1/no-prefill-borrow/compactMoE/MMQ0/ring8; no BF16x2 or unfused HC. Delta includes separate tok_dev; no mode1 identity/staging. Ring stride inferred from the observed512-expert layer cache; Alloc ring adds its own guard.',
                      ordering='generate.cpp prepares/enters native/canonical banks before its run lambda; MTP after-detach snapshot is inside that lambda immediately before prompt.run. The separate layer cache, BF16 prefix and residual owners are still additional and must all be subtracted.',
                      decision='Do not infer16K admission from free4.9GB alone. After the complete known owner budget,16K/GPU32768 has only about51MiB arithmetic spare above the existing512MiB headroom;16K/all39261 is negative. Keep current queued profile first, then existing64K placement/QSA decisions. No16K source widening or arena change made.',
                      limits=['This is source arithmetic against a prior closed observation, not a current free-VRAM query, complete allocation trace or future fit/performance proof.',
                              '16K is still rejected by layer-major m.T/m.T_max and QSA16 shape guards; optional profiling/audit guards also require review. No kernel was run at16K.',
                              'Actual allocator granularity/fragmentation, maximum single allocation, library-private workspace and new runtime headroom require independent admission/execution.',
                              'The QSA upper-bound family assumes max_cells<=262144 and is below GDN for both sizes. Larger-context persistent KV owners must not reuse this39K/context65536 free snapshot.',
                              'The GDN output16-only store proposal keeps the arena reservation unchanged and cannot be counted as freed memory.'],
                      GPU_executed=False, native_compiler_executed=False, model_execution=False,
                      physical_admission_qualified=False, performance_adopted=False,
                      physical262144_model_qualified=False, RESTORE_qualified=False)
        out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(output=str(out), accounted_matches_current=True,
                              extra_C16_bytes=delta, scenarios=rows)))


if __name__ == '__main__':
    main()
