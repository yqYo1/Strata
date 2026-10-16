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
    text = text.replace('#include <climits>\n', '#include <climits>\n#include <cstdlib>\n#include <cstring>\n', 1)
    anchor = 'using Spans = std::vector<Span>;\n'
    helper = '''// Opt-in launch geometry, fixed while a graph is captured. No device state
// is read on the host. Changing this setting does not rewrite captured graphs.
size_t bounded_copy_groups(int64_t n_slots) {
  const char *value = std::getenv("STRATA_SYCL_KV_COPY_GROUPS");
  if (!value || !*value || std::strcmp(value, "0") == 0)
    return 0; // Preserve the legacy kernel and launch by default.
  size_t cap = 0;
  if (std::strcmp(value, "128") == 0) cap = 128;
  else if (std::strcmp(value, "256") == 0) cap = 256;
  else if (std::strcmp(value, "512") == 0) cap = 512;
  else if (std::strcmp(value, "1024") == 0) cap = 1024;
  else throw std::invalid_argument(
      "STRATA_SYCL_KV_COPY_GROUPS must be 0, 128, 256, 512, or 1024");
  return std::min(size_t(n_slots), cap);
}
'''
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + helper, 1)
    start = text.index('void kv_stream_resolve(')
    q = text.index('  auto &q = queue_for(stream);\n', start)
    at = q + len('  auto &q = queue_for(stream);\n')
    text = text[:at] + '  const size_t copy_groups = bounded_copy_groups(m.n_slots);\n' + text[at:]
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
    for name, expected in pins['product_inputs'].items():
        if sha(root / name) != expected:
            raise SystemExit('Pinned product input changed: ' + name)
    kv = 'src/kernels/sycl/kv_stream.cpp'
    cm = 'cmake/sycl_backend.cmake'
    texts = {kv: candidate((root / kv).read_text())}
    cmake = (root / cm).read_text()
    anchor = 'target_compile_options(strata_kernels PRIVATE -ffp-contract=off)\n'
    targets = '''
# Manual component targets: no automatic GPU execution in CTest.
add_executable(kv_copy_groups_host_fixture EXCLUDE_FROM_ALL tools/sycl/kv_copy_groups_host.cpp)
target_compile_features(kv_copy_groups_host_fixture PRIVATE cxx_std_17)
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
