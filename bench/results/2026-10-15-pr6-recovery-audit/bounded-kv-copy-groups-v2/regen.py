#!/usr/bin/env python3
"""Regenerate NON-APPLIED source.diff under this proposal directory only.
No build, test, Git, runtime, cleanup, or product write. Refuse changed inputs.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def candidate(text):
    text = text.replace('#include "strata/kernels/kv_stream.hpp"\n',
                        '#include "strata/kernels/kv_stream.hpp"\n#include "strata/kernels/kv_stream_copy_groups.hpp"\n', 1)
    text = text.replace('#include <climits>\n', '#include <climits>\n#include <cstdlib>\n', 1)
    signature = '''void kv_stream_resolve(const KvStreamMap &m, const QsaAttnPools &slots,
                       const KvHostPools &host, int fmt, const int32_t *ids,
                       const int32_t *steps, int64_t nq, int64_t cap,
                       const QsaShapes &s, void *stream) {
'''
    wrapper = signature + '''  if (nq <= 0)
    return;
  // The process environment is fixed before runtime initialization. The
  // component fixture supplies the same policy explicitly without changing it.
  const uint32_t group_cap = kv_stream_copy_group_cap_from_text(
      std::getenv("STRATA_SYCL_KV_COPY_GROUPS"));
  kv_stream_resolve_with_copy_groups(m, slots, host, fmt, ids, steps, nq, cap,
                                     s, stream, group_cap);
}
void kv_stream_resolve_with_copy_groups(
    const KvStreamMap &m, const QsaAttnPools &slots, const KvHostPools &host,
    int fmt, const int32_t *ids, const int32_t *steps, int64_t nq, int64_t cap,
    const QsaShapes &s, void *stream, uint32_t copy_group_cap) {
'''
    assert text.count(signature) == 1
    text = text.replace(signature, wrapper, 1)
    start = text.index('void kv_stream_resolve_with_copy_groups(')
    early = text.index('  if (nq <= 0)\n    return;\n', start)
    at = early + len('  if (nq <= 0)\n    return;\n')
    text = text[:at] + '  validate_kv_stream_copy_group_cap(copy_group_cap);\n' + text[at:]
    q = text.index('  auto &q = queue_for(stream);\n', start)
    at = q + len('  auto &q = queue_for(stream);\n')
    text = text[:at] + '  const size_t copy_groups = std::min(size_t(m.n_slots), size_t(copy_group_cap));\n' + text[at:]
    old = '''  auto event = q.parallel_for(
      sycl::nd_range<1>(size_t(m.n_slots) * 128, 128),
      [=](sycl::nd_item<1> it) {
        const size_t k = it.get_group_linear_id(), t = it.get_local_linear_id();
        if (k >= size_t(m.ctl[2]))
          return;
        const size_t b = m.miss_block[k], slot = m.miss_slot[k];
        for (int a = 0; a < r.n; ++a)
          for (size_t i = t; i < r.len[a]; i += 128)
            r.dst[a][slot * r.len[a] + i] = r.src[a][b * r.len[a] + i];
      });
'''
    new = '''  sycl::event event;
  if (!copy_groups) {
    // Legacy default: one workgroup per residency slot, including all hits.
    event = q.parallel_for(
        sycl::nd_range<1>(size_t(m.n_slots) * 128, 128),
        [=](sycl::nd_item<1> it) {
          const size_t k = it.get_group_linear_id(), t = it.get_local_linear_id();
          if (k >= size_t(m.ctl[2]))
            return;
          const size_t b = m.miss_block[k], slot = m.miss_slot[k];
          for (int a = 0; a < r.n; ++a)
            for (size_t i = t; i < r.len[a]; i += 128)
              r.dst[a][slot * r.len[a] + i] = r.src[a][b * r.len[a] + i];
        });
  } else {
    // The same in-order queue publishes ctl/miss lists before this kernel.
    // Every placement k belongs to exactly one k % copy_groups workgroup.
    // Placements have distinct destination slots, so no barrier is needed.
    event = q.parallel_for(
        sycl::nd_range<1>(copy_groups * 128, 128),
        [=](sycl::nd_item<1> it) {
          const size_t t = it.get_local_linear_id();
          for (size_t k = it.get_group_linear_id(); k < size_t(m.ctl[2]);
               k += copy_groups) {
            const size_t b = m.miss_block[k], slot = m.miss_slot[k];
            for (int a = 0; a < r.n; ++a)
              for (size_t i = t; i < r.len[a]; i += 128)
                r.dst[a][slot * r.len[a] + i] = r.src[a][b * r.len[a] + i];
          }
        });
  }
'''
    assert text.count(old) == 1
    return text.replace(old, new, 1)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=D.parents[3])
    args = ap.parse_args()
    root = args.root.resolve()
    pins = json.loads((D / 'pins.json').read_text())
    for name, expected in pins['v1_files'].items():
        if sha(D.parent / 'bounded-kv-copy-groups-v1' / name) != expected:
            raise SystemExit('Preserved v1 file changed: ' + name)
    for name, expected in pins['product_inputs'].items():
        if sha(root / name) != expected:
            raise SystemExit('Pinned product input changed: ' + name)
    kv = 'src/kernels/sycl/kv_stream.cpp'
    cm = 'cmake/sycl_backend.cmake'
    kh = 'include/strata/kernels/kv_stream.hpp'
    header = (root / kh).read_text()
    declaration = '''void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream);
'''
    explicit = '''
/// Explicit copy geometry for component fixtures/benchmarks. Shares the full
/// validation, residency policy and copy implementation with kv_stream_resolve.
/// 0 preserves the legacy launch; supported opt-ins are 128/256/512/1024.
/// Does not read or modify the process environment. Geometry is fixed at capture.
void kv_stream_resolve_with_copy_groups(
    const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
    const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
    void* stream, uint32_t copy_group_cap);
'''
    assert header.count(declaration) == 1
    texts = {kv: candidate((root / kv).read_text()), kh: header.replace(declaration, declaration + explicit, 1)}
    cmake = (root / cm).read_text()
    anchor = 'target_compile_options(strata_kernels PRIVATE -ffp-contract=off)\n'
    targets = '''
# Manual component targets: no automatic GPU execution in CTest.
add_executable(kv_copy_groups_host_fixture EXCLUDE_FROM_ALL tools/sycl/kv_copy_groups_host.cpp)
target_compile_features(kv_copy_groups_host_fixture PRIVATE cxx_std_17)
target_include_directories(kv_copy_groups_host_fixture PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_executable(sycl_kv_copy_groups_bench EXCLUDE_FROM_ALL tools/sycl/kv_copy_groups_bench.cpp)
target_link_libraries(sycl_kv_copy_groups_bench PRIVATE strata_kernels)
target_compile_options(sycl_kv_copy_groups_bench PRIVATE -fno-fast-math -ffp-contract=off)
'''
    assert cmake.count(anchor) == 1
    texts[cm] = cmake.replace(anchor, anchor + targets, 1)
    for name, text in texts.items():
        p = D / 'files' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    proposed = {p.relative_to(D / 'files').as_posix(): p for p in sorted((D / 'files').rglob('*')) if p.is_file()}
    diff = []
    for name, p in proposed.items():
        before = (root / name).read_text() if name in texts else ''
        if name not in texts and (root / name).exists():
            raise SystemExit('Proposed new path already exists: ' + name)
        diff.extend(difflib.unified_diff(before.splitlines(True), p.read_text().splitlines(True),
                    fromfile='a/' + name if before else '/dev/null', tofile='b/' + name))
    (D / 'source.diff').write_text(''.join(diff))
    result = dict(schema=1, state='UNAPPLIED_UNBUILT_UNTESTED',
                  source_diff_sha256=sha(D/'source.diff'),
                  files={name:sha(p) for name,p in proposed.items()},
                  product_written=False, build_run=False, tests_run=False, GPU_run=False)
    (D / 'generated.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['source_diff_sha256'])

if __name__ == '__main__':
    main()
