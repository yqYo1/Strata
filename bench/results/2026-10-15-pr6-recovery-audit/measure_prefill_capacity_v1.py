#!/usr/bin/env python3
"""Read actual model directories and reproduce prefill allocation arithmetic.

Reads GGUF headers, never tensor payloads. No GPU inference/allocation. The
calculated one-layer-only architecture is NOT the current serve owner layout.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
PACK = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
sys.path.insert(0, str(P/'tools'))
from gguf_reader import GGUFFile
sys.path.insert(0, str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id

GIB = 1 << 30
MIB = 1 << 20
N, D, HC, LR, HV, ZV, C, K, NE = 2560, 10240, 4, 320, 48, 6144, 10240, 10, 512


def aligned(n, a):
    return (n+a-1)//a*a


def alloc(n):
    # Actual Prefill::Alloc::take: at least 256 guard bytes per allocation.
    return aligned(n+256, 256)


def workspace(t, context=262144, streamed=False):
    cap = 2051
    gdn = sum(map(alloc, [t*x for x in (C*4, ZV*4, 2*HV*4, HV*4, HV*4, C*4, ZV*4, ZV*2)]))
    attn_scalars = ((cap+63)//64)*24*(256+2)+64
    qsa = sum(map(alloc, [t*x for x in (512*4, 512*4, 12288*4, ZV*4, 128*4, 512*4, ZV*4, ZV*2, cap*4)] +
                  [256*(context//4+2)*4, 32*attn_scalars*4]))
    moe = sum(map(alloc, [t*x for x in (NE*4, K*4, K*4, K*4, K*4, N*2, 1280*4, 640*2,
                    K*N*4, 640*4, 640*4, 640*2, N*4, 4)]))
    region = max(gdn, qsa, moe)
    parts = [
        ('dense_dequant_FP16_32M_elements', 64*MIB), ('gemm_reserved_workspace', 32*MIB),
        ('embedding_chunk', t*N*4), ('live_R_FP32_chunk', t*D*4),
        ('grs', t*HC*4), ('xn16', t*D*2), ('lo', t*LR*4), ('lo16', t*LR*2),
        ('gated', t*D*4), ('inject', t*HC*4), ('mixed', t*N*4),
        ('mixed_bf16', t*N*2), ('mixed_fp16', t*N*2), ('block_output', t*N*4),
        ('steps', t*4*4), ('shared_max_GDN_QSA_MoE', region),
    ]
    for i in range(2):
        parts += [(f'expert_GU_FP16_{i}', 1280*2560*2), (f'expert_Down_FP16_{i}', 2560*640*2)]
    for i in range(8):
        parts.append((f'legacy_ring_{i}', 2662400))
    parts += [('PLE_embedding_chunk', t*N*4), ('PLE_norm', D*4)]
    if streamed:
        rows = aligned(context, 4)*2
        parts += [('KV_stage_K8', rows*256), ('KV_stage_V8', rows*256),
                  ('KV_stage_K_scale', rows*4*2), ('KV_stage_V_scale', rows*4*2)]
    return dict(chunk=t, context=context, kv_mode=1 if streamed else 0,
        carve_accounted_bytes=sum(alloc(n) for _, n in parts),
        separate_token_buffer_bytes=t*4,
        separate_identity_table_bytes=aligned(context, 4)//4*4 if streamed else 0,
        shared_regions=dict(GDN=gdn, QSA=qsa, MoE=moe, maximum=region),
        parts=[dict(name=k, payload_bytes=n, allocation_bytes=alloc(n)) for k, n in parts])


def qsa_state(context, fmt, layers=12):
    pages = (context+3)//4
    per_token = dict(K8V8=1056, K8V4=816)[fmt]
    pools = pages*4*per_token+64
    # Mirrors qsa_state_bytes(..., with_rope=false), including returned slack.
    one = pools+4*16+pages*4+3*128*4+128*4+(context//4+2)*128*4+16+16+16+24*4
    one = aligned(one, 16)+256
    return layers*one+context*256


def calculate():
    baseline_path = A/'prefill-only-clean-iq1-c4096-v1.json'
    baseline = json.loads(baseline_path.read_text())
    owner = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    require(baseline['passed'] and baseline['complete'] and not baseline['active'] and
            not baseline['fresh_faults'] and baseline['boot_unchanged'] and
            not baseline['driver_dump_exists'] and baseline['boot_id'] == boot_id(), 'closed baseline')
    require(all(owner.closed(c) for c in baseline['commands']), 'closed owned commands')
    for command in baseline['commands']:
        for old in command['owners']:
            now = owner.identity(old['pid'])
            require(now is None or now['start_ticks'] != old['start_ticks'], 'baseline process still alive')
    # Verify source-derived count against actual allocation announcement, not
    # bytes_needed (that API conservatively counts an unused xn buffer).
    native_log = B/'main-prefill-only-clean-iq1-c4096-root-v1/owned/engine.stderr'
    text = native_log.read_text()
    require(sha(native_log) == baseline['native_log_sha256'], 'closed allocation log identity')
    observed = re.search(r'chunk=4096, scratch rows=4096, accounted=(\d+), shared region=(\d+) bytes', text)
    require(observed, 'actual allocation announcement')
    check = workspace(4096, streamed=True)
    require(check['carve_accounted_bytes'] == int(observed[1]) == 1662831872 and
            check['shared_regions']['maximum'] == int(observed[2]) == 607128320, 'count reproduction')
    totals = {int(x) for x in re.findall(r'total_bytes=(\d+)', text)}
    require(totals == {10666115072}, 'actual device budget observation')
    total = totals.pop()
    index, expert_path = PACK/'index.txt', PACK/'native_experts.txt'
    rows = {}
    for line in index.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        f = line.split()
        require(len(f) >= 10 and f[0] not in rows, 'index row')
        rows[f[0]] = dict(name=f[0], kind=int(f[2]), bytes=int(f[6]),
                          shape=[int(f[7]), int(f[8])], code_bits=int(f[9]))
    experts = {}
    for line in expert_path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        f = line.split()
        experts[int(f[0])] = dict(gu_type=int(f[1]), down_type=int(f[2]), expert_bytes=int(f[4]))
    require(set(experts) == set(range(48)) and max(e['expert_bytes'] for e in experts.values()) == 2662400,
            'expert metadata')
    suffixes = ('.attn_qkv.weight', '.attn_gate.weight', '.ssm_out.weight', '.attn_q.weight',
        '.attn_k.weight', '.attn_v.weight', '.attn_output.weight', '.ffn_gate_shexp.weight',
        '.ffn_up_shexp.weight', '.ffn_down_shexp.weight')
    supported = {2, 6, 7, 8, 11, 12, 13, 14, 18, 20, 21, 22, 23, 42, 16, 17, 29}
    native_tensors, headers, table_tensors = {}, [], []
    for filename, expected in baseline['model_identity'].items():
        path = Path(filename)
        s = path.stat()
        require(s.st_size == expected['bytes'] and s.st_mtime_ns == expected['mtime_ns'], 'model identity')
        gg = GGUFFile(path)
        with path.open('rb') as f:
            header_sha = hashlib.sha256(f.read(gg.data_start)).hexdigest()
        headers.append(dict(path=str(path), file_bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                            header_bytes=gg.data_start, header_sha256=header_sha, tensor_count=len(gg.tensors)))
        for tensor in gg.tensors:
            if tensor.name in ('token_embd.weight', 'blk.1.ple_embd.weight') or 'ple_' in tensor.name:
                table_tensors.append(dict(name=tensor.name, shape=tensor.shape, type=tensor.type_name,
                                           bytes=tensor.expected_bytes()))
            eligible = tensor.name.startswith('blk.') and (tensor.name.endswith(suffixes) or
                (tensor.name == 'blk.1.ple_key.weight' and tensor.type_id in {42, 18, 23, 8}))
            if eligible and tensor.type_id in supported and len(tensor.shape) == 2:
                require(tensor.name not in native_tensors, 'duplicate native tensor')
                native_tensors[tensor.name] = dict(name=tensor.name, shape=tensor.shape, type=tensor.type_name,
                                                  bytes=tensor.expected_bytes())
    require(len(native_tensors) == 300 and all(v['bytes'] for v in native_tensors.values()), 'actual native matrix list')
    skip = set(native_tensors) | {'output.weight', 'token_embd.weight'}
    if rows.get('blk.1.ple_key.weight', {}).get('code_bits') == 0:
        skip.discard('blk.1.ple_key.weight')
    canonical_pool = sum(aligned(r['bytes'], 64) for n, r in rows.items() if n not in skip)
    layer_data = []
    for layer in range(48):
        prefix = f'blk.{layer}.'
        canonical = [dict(r, allocated_bytes=aligned(r['bytes'], 64)) for n, r in rows.items() if n.startswith(prefix) and n not in skip]
        native = [v for n, v in native_tensors.items() if n.startswith(prefix)]
        cb = sum(v['allocated_bytes'] for v in canonical)
        nb = sum(v['bytes'] for v in native)
        expert_bytes = aligned(experts[layer]['expert_bytes'], 256)*512
        layer_data.append(dict(layer=layer, kind='QSA' if layer % 4 == 3 else 'GDN',
            canonical_bytes=cb, native_bytes=nb, packed_experts_bytes=expert_bytes,
            total_bytes=cb+nb+expert_bytes, canonical=canonical, native=native, **experts[layer]))
    maximum = max(layer_data, key=lambda v: v['total_bytes'])
    # Keep actual packed-layer allocation strategy: uniform max expert stride,
    # plus the maximum current-layer dense payload, even if on different layers.
    layer_reservation = max(v['packed_experts_bytes'] for v in layer_data)+max(v['canonical_bytes']+v['native_bytes'] for v in layer_data)
    gdn_states = 36*(128*48*128+10240*(4-1))*4
    ple_history = 2*10240*4
    small_fixed = 16*MIB
    reserve = 512*MIB
    cases = []
    for t in (4096, 8192):
        work = workspace(t)
        fixed = layer_reservation+work['carve_accounted_bytes']+work['separate_token_buffer_bytes']+gdn_states+ple_history+small_fixed+reserve
        budget = total-fixed
        for fmt in ('K8V8', 'K8V4'):
            def needed(c):
                return qsa_state(c, fmt)+c*D*2
            lo, hi = 0, 262144//4
            while lo < hi:
                mid = (lo+hi+1)//2
                if needed(mid*4) <= budget:
                    lo = mid
                else:
                    hi = mid-1
            full = qsa_state(262144, fmt)
            hidden_bytes = max(0, budget-full)
            cases.append(dict(chunk=t, format=fmt, total_device_bytes=total,
                active_layer_reservation_bytes=layer_reservation, workspace=work,
                recurrent_GDN_bytes=gdn_states, PLE_history_bytes=ple_history,
                small_owner_allowance_bytes=small_fixed, runtime_headroom_bytes=reserve,
                fixed_before_KV_hidden_bytes=fixed, available_for_KV_hidden_bytes=budget,
                max_context_with_all_hidden_rows=lo*4,
                bytes_at_max_context=needed(lo*4), next_page_bytes=needed((lo+1)*4),
                full_262144_QSA_bytes=full, hidden_bytes_with_full_KV=hidden_bytes,
                hidden_rows_with_full_KV=min(262144, hidden_bytes//(D*2)),
                full_context_shortfall_bytes=max(0, needed(262144)-budget),
                one_active_QSA_full_context_bytes=qsa_state(262144, fmt, layers=1),
                all_hidden_with_one_active_QSA_fits=qsa_state(262144, fmt, layers=1)+262144*D*2 <= budget,
                one_active_QSA_remaining_bytes=budget-qsa_state(262144, fmt, layers=1)-262144*D*2))
    pins = {str(n): sha(n) for n in [Path(__file__), P/'tools/gguf_reader.py', index, expert_path,
        P/'src/core/native_dense.cpp', P/'src/core/weights.cpp', P/'src/core/layer.cpp',
        P/'src/core/session.cpp', P/'src/prefill/prefill.cpp', P/'include/strata/core/layout.hpp',
        P/'src/kernels/sycl/decode_attention.cpp', P/'include/strata/kernels/kv_q8.hpp',
        P/'include/strata/kernels/kv_q4.hpp']}
    return dict(schema='prefill-capacity-metadata-and-source-v1', complete=True, passed=True, active=False,
        GPU_executed=False, GPU_allocation_probe_executed=False, physical_admission_qualified=False,
        quality_qualified=False, performance_qualified=False, boot_id=boot_id(), finished_utc=owner.utc(),
        baseline_sha256=sha(baseline_path), source_pins=pins, model_headers=headers,
        observed_device_total_bytes=total, observed_streamed_workspace=check,
        actual_native_matrix_count=len(native_tensors), actual_native_weight_bytes=sum(v['bytes'] for v in native_tensors.values()),
        actual_canonical_pool_bytes=canonical_pool, mapped_table_metadata=table_tensors,
        largest_layer=maximum, layer_reservation_bytes=layer_reservation, layers=layer_data,
        storage_table=[dict(context=c, FP16_hidden_bytes=c*D*2,
            K8V8_QSA_bytes=qsa_state(c, 'K8V8'), K8V4_QSA_bytes=qsa_state(c, 'K8V4')) for c in (32768, 65536, 262144)],
        cases=cases, assumptions=[
            'Hypothetical dedicated long-prefill layout: only active layer weights resident; no decode cache/output head/MTP full-draft owners.',
            'Actual packed expert formats and canonical/native dense formats retained; no all-expert FP16 decoded weight bank.',
            'A16 means FP16 inter-layer residual storage only; current FP32 working tensors and accumulators retained.',
            'All 12 QSA KV/indexer states and one shared RoPE resident for main cases; active-QSA-only case is separately labeled.',
            'Full-resident KV needs no full-context per-active-layer take_stage allocation.',
            'Current ring8/DQ2/GEMM reserved buffers kept; 16MiB small-owner allowance and512MiB runtime headroom are planning assumptions.',
            'K8V4 storage scenario is not a working configured-SYCL route; FP16 storage quality and maximum context physical execution remain unqualified.'
        ])


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = calculate()
        target = A/'prefill-capacity-metadata-v1.json'
        with target.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(dict(passed=True, largest_layer={k: result['largest_layer'][k] for k in
            ('layer', 'kind', 'canonical_bytes', 'native_bytes', 'packed_experts_bytes', 'total_bytes')},
            layer_reservation_bytes=result['layer_reservation_bytes'], cases=[{k: c[k] for k in
            ('chunk', 'format', 'available_for_KV_hidden_bytes', 'max_context_with_all_hidden_rows',
             'full_262144_QSA_bytes', 'hidden_rows_with_full_KV', 'full_context_shortfall_bytes',
             'all_hidden_with_one_active_QSA_fits', 'one_active_QSA_remaining_bytes')} for c in result['cases']])), flush=True)
