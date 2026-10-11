from pathlib import Path
import hashlib, json, difflib
root=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/xe-native-dequant-group-api-source-v1')
hdr='''
/// Optional Xe-only native-to-FP16 group APIs; no caller is switched automatically.
constexpr int kIqDequantGroup = 16;
struct IqDequantGroup {
    const void* gate[kIqDequantGroup]{};
    const void* up[kIqDequantGroup]{};
    const void* down[kIqDequantGroup]{};
    size_t gate_bytes[kIqDequantGroup]{}, up_bytes[kIqDequantGroup]{}, down_bytes[kIqDequantGroup]{};
    int first = 0, n = 0;   // active slots [first, first+n), also the destination slots
};
/// The pointer arrays are host metadata copied by value; active addresses already reside on this stream's device.
/// `stride` is FP16 elements between expert slots; `dst_bytes` is the full accessible destination capacity.
/// Sources are at least 4-byte aligned, destination 16-byte aligned (stride multiple 8), and source/destination ranges must not overlap.
/// False: refused before submission. True: zero work or normal submission (null stream also waits, as the old APIs).
/// Exceptions from Runtime, submission or the null-stream wait propagate; true with a stream is not completion.
/// n==0 accepts legal slot bounds without inspecting dimensions, pointers, capacities, type or stream; no launch.
/// Nonempty calls require positive dimensions, n_embd multiple 256, supported IQ types, and checked capacities.
/// GU writes [2*n_ff,n_embd] interleaved per slot; Down writes flat [n_embd,n_ff] per slot (total multiple 256).
/// Caller owns device accessibility, actual allocation capacities, stream lifetime and ordering until completion.
bool iq_dequant_gu_group_f16(int ggml_type, const IqDequantGroup& group, int64_t n_ff, int64_t n_embd,
                             uint16_t* dst, size_t stride, size_t dst_bytes, void* stream);
bool iq_dequant_down_group_f16(int ggml_type, const IqDequantGroup& group, int64_t n_ff, int64_t n_embd,
                               uint16_t* dst, size_t stride, size_t dst_bytes, void* stream);
'''
code='''
namespace {

// All group checks precede queue submission; no allocation, queue or synchronization is introduced.
bool dequant_group_mul(size_t a, size_t b, size_t& out) {
    const size_t limit = std::min((size_t) std::numeric_limits<int64_t>::max(),
                                  (size_t) std::numeric_limits<ptrdiff_t>::max());
    if (b != 0 && a > limit / b) return false;
    out = a * b;
    return true;
}
bool dequant_group_span(const void* p, size_t bytes, size_t alignment) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    return p && a % alignment == 0 && bytes <= (size_t) std::numeric_limits<ptrdiff_t>::max() &&
           bytes <= std::numeric_limits<uintptr_t>::max() - a;
}
bool dequant_group_bounds(const IqDequantGroup& g) {
    return g.first >= 0 && g.first <= kIqDequantGroup && g.n >= 0 && g.n <= kIqDequantGroup - g.first;
}
bool dequant_group_check(int t, const IqDequantGroup& g, int64_t n_ff, int64_t n_embd,
                         uint16_t* dst, size_t stride, size_t dst_bytes, bool gu,
                         size_t& values, size_t& global) {
    if (!is_iq(t) || n_ff <= 0 || n_embd <= 0 || n_embd % 256 != 0 ||
        (uint64_t) n_ff > std::numeric_limits<size_t>::max() ||
        (uint64_t) n_embd > std::numeric_limits<size_t>::max()) return false;
    size_t matrix = 0;
    if (!dequant_group_mul((size_t) n_ff, (size_t) n_embd, matrix) ||
        !dequant_group_mul(matrix, gu ? 2 : 1, values) || stride < values || stride % 8 != 0) return false;
    const size_t nsb = matrix / QK_K;
    // Exactly the old flat / GU launch: eight 256-value superblocks per 256-work-item group.
    const size_t groups = nsb / 8 + (nsb % 8 != 0);
    size_t total = 0;
    if (!dequant_group_mul(groups, 256, global) ||
        !dequant_group_mul(global, (size_t) g.n * (gu ? 2 : 1), total)) return false;
    size_t last = 0, end = 0, bytes = 0, source_bytes = 0;
    if (!dequant_group_mul((size_t) (g.first + g.n - 1), stride, last)) return false;
    const size_t limit = std::min((size_t) std::numeric_limits<int64_t>::max(),
                                  (size_t) std::numeric_limits<ptrdiff_t>::max());
    if (values > limit - last) return false;
    end = last + values;
    if (!dequant_group_mul(end, sizeof(uint16_t), bytes) || bytes > dst_bytes ||
        !dequant_group_span(dst, bytes, alignof(sycl::vec<sycl::half, 8>)) ||
        !dequant_group_mul(nsb, iq_row_bytes(t, QK_K), source_bytes)) return false;
    const uintptr_t lo = reinterpret_cast<uintptr_t>(dst), hi = lo + bytes;
    for (int e = g.first; e < g.first + g.n; ++e) {
        const void* sources[2] = {gu ? g.gate[e] : g.down[e], g.up[e]};
        const size_t capacities[2] = {gu ? g.gate_bytes[e] : g.down_bytes[e], g.up_bytes[e]};
        for (int p = 0; p < (gu ? 2 : 1); ++p) {
            if (source_bytes > capacities[p] || !dequant_group_span(sources[p], source_bytes, 4)) return false;
            const uintptr_t a = reinterpret_cast<uintptr_t>(sources[p]);
            if (a < hi && lo < a + source_bytes) return false;
        }
    }
    return true;
}

}  // namespace

bool iq_dequant_gu_group_f16(int t, const IqDequantGroup& group, int64_t n_ff, int64_t n_embd,
                             uint16_t* dst, size_t stride, size_t dst_bytes, void* stream) {
    if (!dequant_group_bounds(group)) return false;
    if (group.n == 0) return true;
    size_t values = 0, global = 0;
    if (!dequant_group_check(t, group, n_ff, n_embd, dst, stride, dst_bytes, true, values, global)) return false;
    auto& q = queue_for(stream);
    if (q.get_device().get_info<sycl::info::device::max_work_group_size>() < 256 ||
        q.get_device().get_info<sycl::info::device::max_work_item_sizes<3>>()[2] < 256) return false;
    const IqDequantGroup g = group;   // addresses copied by value, never an array read from host memory on device
    const int64_t per_row = n_embd / 256, nsb = n_ff * per_row;
    auto* y = reinterpret_cast<sycl::half*>(dst);
    const auto e = q.parallel_for(
        sycl::nd_range<3>({(size_t) g.n, 2, global}, {1, 1, 256}), [=](sycl::nd_item<3> it) {
            const int expert = g.first + (int) it.get_group(0);
            const int64_t i = (int64_t) (it.get_global_id(2) / 32);
            if (i >= nsb) return;
            const int parity = (int) it.get_group(1);
            const int64_t r = i / per_row, c = i % per_row;
            dq_dispatch(t, parity ? g.up[expert] : g.gate[expert], i,
                        ToGlobal<sycl::half>{y + (size_t) expert * stride + ((2 * r + parity) * per_row + c) * QK_K},
                        (int) (it.get_local_id(2) % 32));
        });
    sync_if_needed(stream, e, "iq_dequant_gu_group_f16");
    return true;
}

bool iq_dequant_down_group_f16(int t, const IqDequantGroup& group, int64_t n_ff, int64_t n_embd,
                               uint16_t* dst, size_t stride, size_t dst_bytes, void* stream) {
    if (!dequant_group_bounds(group)) return false;
    if (group.n == 0) return true;
    size_t values = 0, global = 0;
    if (!dequant_group_check(t, group, n_ff, n_embd, dst, stride, dst_bytes, false, values, global)) return false;
    auto& q = queue_for(stream);
    if (q.get_device().get_info<sycl::info::device::max_work_group_size>() < 256 ||
        q.get_device().get_info<sycl::info::device::max_work_item_sizes<2>>()[1] < 256) return false;
    const IqDequantGroup g = group;
    const int64_t nsb = (int64_t) (values / QK_K);
    auto* y = reinterpret_cast<sycl::half*>(dst);
    const auto e = q.parallel_for(
        sycl::nd_range<2>({(size_t) g.n, global}, {1, 256}), [=](sycl::nd_item<2> it) {
            const int expert = g.first + (int) it.get_group(0);
            const int64_t i = (int64_t) (it.get_global_id(1) / 32);
            if (i >= nsb) return;
            dq_dispatch(t, g.down[expert], i,
                        ToGlobal<sycl::half>{y + (size_t) expert * stride + i * QK_K},
                        (int) (it.get_local_id(1) % 32));
        });
    sync_if_needed(stream, e, "iq_dequant_down_group_f16");
    return true;
}
'''
files=['src/kernels/xe/iq_kernels.cpp','include/strata/kernels/iq_kernels.hpp']
manifest={'base_commit':'9c0b291d85adecb6c1dc489c3e9965b82226b799','qualification':'source-only, unbuilt and unexecuted','files':[]}
patch=''
for f in files:
    original=(root/'base'/f).read_text()
    if f.endswith('.cpp'):
        replacement=original.replace('#include <cstring>\n','#include <cstring>\n#include <limits>\n',1)
        anchor='\nbool iq_gemm_grouped_ok('
        replacement=replacement.replace(anchor,code+anchor,1)
        recovered=replacement.replace(code,'',1).replace('#include <limits>\n','',1)
    else:
        anchor='\n/// Whether iq_gemm_grouped_f16'
        replacement=original.replace(anchor,hdr+anchor,1)
        recovered=replacement.replace(hdr,'',1)
    if recovered!=original: raise RuntimeError('insertion removal did not recover original')
    target=root/'replacement'/f
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(replacement)
    patch+=''.join(difflib.unified_diff(original.splitlines(True),replacement.splitlines(True),fromfile='a/'+f,tofile='b/'+f))
    manifest['files'].append({'path':f,'base_sha256':hashlib.sha256(original.encode()).hexdigest(),'replacement_sha256':hashlib.sha256(replacement.encode()).hexdigest(),'base_bytes':len(original.encode()),'replacement_bytes':len(replacement.encode()),'insertion_removal_recovered_exact_original':True})
(root/'candidate.patch').write_text(patch)
manifest['patch_sha256']=hashlib.sha256(patch.encode()).hexdigest()
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
