"""Prepare a private SYCL port of the upstream exact-order12-head reduction.

Source-only preparation. Do not build or run while full DD5 qualification is active.
"""
from pathlib import Path
import datetime, hashlib, json, re

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
original = root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp'
s = original.read_text()
function_start = s.index('__device__ __forceinline__ float reduce12(const float (&part)[16], int lane) {')
function_end = s.index('\n}', function_start)+2
cuda_helper = s[function_start:function_end]
helper = cuda_helper.replace('__device__ __forceinline__ float reduce12', '__dpct_inline__ float reduce12_sg32')
helper = helper.replace('int lane) {', 'int lane) {\n    const auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();')
helper, count = re.subn(r'__shfl_xor_sync\(0xffffffffu, ([^\n]*?), (16|8|4|2|1)\)',
                        r'sycl::permute_group_by_xor(sg, \1, \2)', helper)
assert count == 5 and '__shfl_xor_sync' not in helper
helper = '''// Upstream pre75 reduce-scatter, confined to the SG32/transposed prompt arm.
// All32 subgroup items execute each XOR exchange; masks16/8/4/2/1 are in range.
// Keep the existing eight-product expression, pairing tree, scales,64-cell split and merge.
'''+helper+'\n\n'
marker = 'template <int SG = 32>\n__dpct_inline__ float warp_max(float v) {'
assert s.count(marker) == 1
s = s.replace(marker, helper+marker)
marker = '        float upper_k[8];\n        if constexpr (SG == 16) load8<KV_MODE>(p, false, srow[c], (lane + 16) * 8, upper_k);\n'
assert s.count(marker) == 1
s = s.replace(marker, marker+'        float qk_part[16] = {}; // four unused heads are neutral; SG16 keeps its original reduction\n')
old = '''            s = warp_sum<SG>(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
'''
new = '''            if constexpr (SG == 32 && TRANSPOSE_Q) {
                qk_part[h] = s;
            } else {
                s = warp_sum<SG>(s);
                if (lane == 0) sp[h][c] = s * scale;
            }
        }
        if constexpr (SG == 32 && TRANSPOSE_Q) {
            // The valid-cell branch and this template condition are subgroup-uniform.
            const int sg_lane = (int) sycl::ext::oneapi::this_work_item::get_sub_group().get_local_linear_id();
            const float score = reduce12_sg32(qk_part, sg_lane);
            const int h = sg_lane >> 1;
            if ((sg_lane & 1) == 0 && h < G) sp[h][c] = score * scale;
        }
    }
'''
assert s.count(old) == 1
s = s.replace(old, new)
assert s.count('#define DPCT_PROFILING_ENABLED') == 1
assert s.count('reduce12_sg32') == 2 and s.count('sycl::permute_group_by_xor(sg,') == 5
out = base/'qsa-reduce12-sg32-v01402-source-v2'
out.mkdir(mode=0o700)
target = out/'qsa_decode_attn.dp.cpp'
target.write_text(s)
(out/'upstream-reduce12-exact-body.txt').write_text(cuda_helper+'\n')
record = {'active': False, 'prepared': True, 'compiled': False, 'cpu_float_identity_tested': False,
          'gpu_tested': False, 'adopted': False, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'original_source': str(original), 'original_source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
          'candidate_source': str(target), 'candidate_source_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
          'caller_uses_actual_subgroup_local_linear_id': True,
          'upstream_reduce12_sha256': hashlib.sha256(cuda_helper.encode()).hexdigest(),
          'scope': 'Private source-only port of existing upstream experimental pre75 reduce12. SG32/transposed prompt arm only; preserve dot operands/expression,64-cell partials, scale and external merge. No engine binary or active GPU process changed.',
          'mechanism': 'Reduce twelve lane dot partials by the existing XOR16/8/4/2/1 pairing tree, distributed across head lanes. Sixteen subgroup exchanges per lane instead of twelve independent five-exchange sums. This instruction count is source-derived, not a speed measurement.',
          'sycl_primary_reference': 'https://github.khronos.org/SYCL_Reference/iface/group-algorithms-library.html',
          'convergence_primary_reference': 'https://github.khronos.org/SYCL_Reference/iface/group-functions.html',
          'guards': ['Only SG32&&TRANSPOSE_Q uses the new collective; launch_chunk_variant forces SG32.',
                     'All subgroup items evaluate all exchanges. Masks and loops are identical; lane-dependent output stores happen after the reduction.',
                     'Masked/past-end cell branch reads one shared srow[c] for a whole subgroup; it skips the collective uniformly.',
                     'No new barrier, queue/event change, global memory layout, vector-pointer alias, cache policy or hidden state.',
                     'Do not assert bit equality for arbitrary NaN payloads or nonfinite intermediate values.'],
          'pending': ['Independent CPU FP32 tree identity check with finite, cancellation, signed-zero and subnormal cases.',
                      'Compile/actual dependency and single archive-member/link review after active full DD5 job exits.',
                      'First logged>=32768 head/used-state/IDs/logprobs/MTP/disk continuation gate.',
                      'Qualified quiet>=32768 ABBA with first and later full reads separate, then mandatory physical256K gate before adoption.']}
(out/'record.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({'prepared': True, 'compiled': False, 'gpu_tested': False, 'candidate_source_sha256': record['candidate_source_sha256']}, indent=2))
