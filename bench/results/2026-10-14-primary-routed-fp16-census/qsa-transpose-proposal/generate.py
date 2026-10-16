#!/usr/bin/env python3
"""Generate an unapplied proposal; never write outside this artifact directory."""
from pathlib import Path
import difflib
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BASE = {
    "src/kernels/sycl/decode_attention.cpp": "2ce2a61f6beb1acbf9c673da346e887b386fffedba10e7df2c157aedab8ac05d",
    "include/strata/kernels/qsa_decode_attn.hpp": "6fa4b817f835c43783774d5946d14938f2867d7af16abb4db0f4b75d4622a915",
}

HEADER = r'''
#if defined(STRATA_ENABLE_SYCL)
/// Allocated storage extents, not selection capacity or currently written cells.
/// The caller owns these buffers and must retain them until the queue completes.
struct QsaPrefillAttnStorage {
    int64_t logical_cells = 0;  ///< physical context cells (262144 at the full boundary)
    int64_t table_pages = 0;    ///< allocated logical page-table entries, including any spare
    int64_t pool_pages = 0;     ///< allocated physical pages of each supplied INT8 K/V pool
};

/// Prefill fallback experiment only: old own-fork FP32 query-local transpose.
/// False means unsupported geometry/format/storage/capability; nothing enqueued.
/// Invalid aligned/disjoint span contracts throw before enqueue, as in decode.
/// Requires INT8 K/V, 24 Q heads, 2 KV heads, D256. Scratch layout is unchanged.
bool qsa_prefill_attn_transpose_batch(
    const float* q, const QsaAttnPools& pools, const int32_t* ids,
    const int32_t* steps, int64_t cap, const QsaShapes& s,
    const QsaPrefillAttnStorage& storage, float* scratch, float* attn,
    int64_t n_q, void* stream);
#endif
'''

ENTRY = r'''
// No environment reader here: Main wires the default-OFF prefill fallback hook.
bool qsa_prefill_attn_transpose_batch(
    const float *query, const QsaAttnPools &p, const int32_t *ids,
    const int32_t *steps, int64_t cap, const QsaShapes &s,
    const QsaPrefillAttnStorage &storage, float *scratch, float *attn,
    int64_t nq, void *stream) {
  // No host read of device IDs/steps and no submissions before all admission
  // and span checks. The storage descriptor is an allocation-owner contract.
  if (nq <= 0 || nq > 65535 || cap <= 0 || cap > INT_MAX ||
      s.head_dim != HD || s.n_head_kv != 2 || s.n_head != 24 ||
      s.page_size <= 0 || s.page_size > INT_MAX ||
      storage.logical_cells <= 0 || storage.logical_cells > INT_MAX ||
      storage.table_pages <= 0 || storage.table_pages > INT_MAX ||
      storage.pool_pages <= 0 || storage.pool_pages > INT_MAX ||
      storage.table_pages <
          (storage.logical_cells + s.page_size - 1) / s.page_size ||
      !p.k_q || !p.v_q || !p.k_scale || !p.v_scale || !p.page_table ||
      p.k_pool || p.v_pool || p.k_q4 || p.v_q4)
    return false;

  auto &queue = sycl_backend::queue_for(stream);
  // The same reported-device checks used by prompt admission, with this
  // kernel's actual subgroup, work-group and FP32 local-array requirements.
  // Do not infer SG32 support from a matrix SG16 acceptance/decline.
  try {
    const auto device = queue.get_device();
    const auto subgroups =
        device.get_info<sycl::info::device::sub_group_sizes>();
    constexpr size_t local_bytes =
        sizeof(float) * (G * HD + G * CHUNK) + sizeof(int64_t) * CHUNK;
    if (std::find(subgroups.begin(), subgroups.end(), 32) == subgroups.end() ||
        device.get_info<sycl::info::device::local_mem_size>() < local_bytes ||
        device.get_info<sycl::info::device::max_work_group_size>() < WG ||
        device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0] < WG)
      return false;
  } catch (const sycl::exception &) {
    return false;
  }

  // Every product has a checked signed-64 and size_t byte bound. Checking the
  // full allocated pool/table extents also excludes scratch/output aliasing
  // a later physical page, rather than validating only the first element.
  const auto count = [](int64_t a, int64_t b) -> size_t {
    if (a < 0 || b < 0 ||
        (b && uint64_t(a) > uint64_t(SIZE_MAX) / uint64_t(b)))
      throw std::invalid_argument("prefill transpose extent overflow");
    return sycl_backend::checked_count(a, b);
  };
  size_t n, sn, code_bytes, scale_bytes, output_bytes, scratch_bytes,
      ids_bytes, steps_bytes, table_bytes;
  try {
    n = count(count(nq, s.n_head), HD);
    sn = count(nq, qsa_decode_attn_scratch_floats(cap, s));
    code_bytes = count(count(storage.pool_pages, s.n_head_kv),
                       count(s.page_size, HD));
    scale_bytes = count(count(storage.pool_pages, s.n_head_kv),
                        count(s.page_size, (HD / 64) * 2));
    const size_t groups = count(nq, count((cap + CHUNK - 1) / CHUNK,
                                        s.n_head_kv));
    (void)count(groups, WG); // ND-range, before launch
    output_bytes = count(n, sizeof(float));
    scratch_bytes = count(sn, sizeof(float));
    ids_bytes = count(count(nq, cap), sizeof(int32_t));
    steps_bytes = count(count(nq, kStepCount), sizeof(int32_t));
    table_bytes = count(storage.table_pages, sizeof(int32_t));
  } catch (const std::invalid_argument &) {
    return false; // unsupported extents; existing fallback still owns its checks
  }
  sycl_backend::validate_spans(
      {{scratch, scratch_bytes}, {attn, output_bytes}},
      {{query, output_bytes}, {ids, ids_bytes}, {steps, steps_bytes},
       {p.page_table, table_bytes}});
  sycl_backend::validate_spans(
      {{scratch, scratch_bytes}, {attn, output_bytes}},
      {{p.k_q, code_bytes}, {p.v_q, code_bytes}}, 1);
  sycl_backend::validate_spans(
      {{scratch, scratch_bytes}, {attn, output_bytes}},
      {{p.k_scale, scale_bytes}, {p.v_scale, scale_bytes}}, 2);
  launch_prefill_transpose(query, p, ids, steps, cap, s, storage, scratch,
                          attn, nq, stream);
  return true;
}
'''

def sha(data):
    return hashlib.sha256(data).hexdigest()

def generate():
    original = {}
    for rel, expected in BASE.items():
        data = (ROOT / rel).read_bytes()
        if sha(data) != expected:
            raise SystemExit(f"base changed: {rel}; re-review before regeneration")
        original[rel] = data.decode()
    src = original["src/kernels/sycl/decode_attention.cpp"]
    # Copy the current FP32 kernel/merge into a new, distinct launch entrypoint.
    # Existing decode launch and its public functions remain byte-identical.
    begin = src.index("template <int Format>\nvoid launch(")
    end = src.index("\n} // namespace\nuint64_t qsa_decode_attn_scratch_floats", begin)
    launch = src[begin:end]
    launch = launch.replace("template <int Format>\nvoid launch(",
                            "void launch_prefill_transpose(", 1)
    launch = launch.replace("int64_t cap, QsaShapes s, float *scratch,",
                            "int64_t cap, QsaShapes s,\n            QsaPrefillAttnStorage storage, float *scratch,", 1)
    launch = launch.replace("load<Format>", "load<1>")
    launch = launch.replace(
        "sq[i] = query[(batch * s.n_head + kvh * G) * HD + i];",
        "sq[(i / HD) * HD + (i % 8) * 32 + (i % HD) / 8] =\n"
        "                query[(batch * s.n_head + kvh * G) * HD + i];", 1)
    launch = launch.replace(
        "const int page = cell < 0 ? -1 : p.page_table[cell / s.page_size];\n"
        "              if (page >= 0)\n"
        "                row = (int64_t(page) * s.n_head_kv + kvh) * s.page_size +\n"
        "                      cell % s.page_size;",
        "// Bound the logical cell before page-table lookup; negative IDs\n"
        "              // and negative/nonresident pages stay masked. Ring/staging\n"
        "              // lookup uses the supplied table without remapping.\n"
        "              const int page =\n"
        "                  cell < 0 || int64_t(cell) >= storage.logical_cells\n"
        "                      ? -1\n"
        "                      : p.page_table[size_t(cell) / size_t(s.page_size)];\n"
        "              if (page >= 0 && int64_t(page) < storage.pool_pages)\n"
        "                row = int64_t((size_t(page) * size_t(s.n_head_kv) + kvh) *\n"
        "                              size_t(s.page_size) + size_t(cell) %\n"
        "                              size_t(s.page_size));", 1)
    launch = launch.replace("sq[h * HD + lane * 8]", "sq[h * HD + lane]", 1)
    launch = launch.replace("sq[h * HD + lane * 8 + j]", "sq[h * HD + j * 32 + lane]", 1)
    proposed = dict(original)
    proposed["src/kernels/sycl/decode_attention.cpp"] = (
        src.replace("#include <climits>", "#include <algorithm>\n#include <climits>", 1)
        .replace("\n} // namespace\nuint64_t qsa_decode_attn_scratch_floats",
                 "\n// Own-fork old layout1 mapping; current FP32 math and merge.\n" +
                 launch + "\n} // namespace\nuint64_t qsa_decode_attn_scratch_floats", 1)
        .replace("\n} // namespace strata::kernels", ENTRY + "\n} // namespace strata::kernels", 1))
    header_rel = "include/strata/kernels/qsa_decode_attn.hpp"
    proposed[header_rel] = original[header_rel].replace(
        "\n}  // namespace strata::kernels", HEADER + "\n}  // namespace strata::kernels", 1)
    patch = ""
    for rel in BASE:
        patch += "".join(difflib.unified_diff(
            original[rel].splitlines(keepends=True),
            proposed[rel].splitlines(keepends=True),
            fromfile="a/" + rel, tofile="b/" + rel))
    patch_bytes = patch.encode()
    (OUT / "unified.patch").write_bytes(patch_bytes)
    evidence = {
        "status": "unapplied-unbuilt-untested-unmeasured",
        "base_sha256": BASE,
        "proposed_sha256": {rel: sha(text.encode()) for rel, text in proposed.items()},
        "patch_sha256": sha(patch_bytes),
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "writes": ["unified.patch", "identity.json"],
        "existing_decode_changed": False,
        "prefill_hook_in_patch": False,
    }
    (OUT / "identity.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"patch_sha256": evidence["patch_sha256"], "bytes": len(patch_bytes)}))

if __name__ == "__main__":
    generate()
