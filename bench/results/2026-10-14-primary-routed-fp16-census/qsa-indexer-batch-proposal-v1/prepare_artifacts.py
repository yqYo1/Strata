#!/usr/bin/env python3
"""Generate only this source-pinned proposal; do not apply, build, or test it."""
import difflib
import hashlib
import json
from pathlib import Path
import textwrap

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
CPP = 'src/kernels/sycl/native_qsa_indexer.cpp'
API = 'include/strata/kernels/native_qsa_indexer.hpp'
PLAN = 'include/strata/kernels/qsa_indexer_batch_plan.hpp'
PIN = 'd87abcf8aece04bc1566f4310e0dee546b7c80e0b711d75d118551b606d38685'
original = (ROOT / CPP).read_text()
if hashlib.sha256(original.encode()).hexdigest() != PIN:
    raise SystemExit('source pin mismatch: regenerate only after review')

planner = '''#pragma once
#include <cstdint>

namespace strata::kernels::detail {
// Host/device integer planning only. Caller validates 0 <= p0, n <= capacity-p0,
// capacity <= INT32_MAX before use; no pointer, allocation, or queue ownership.
struct QsaIndexerBatchPlan {
  int64_t p0;
  int64_t end;
  int64_t first_block;
  int64_t completed;
  bool initialize_dead;

  // Completed block rows before p0 are the existing tail slots. All other
  // sources are raw[input_row,128], rounded to F16 by the consuming kernel.
  constexpr int64_t input_row(int64_t block, int row) const {
    const int64_t pos = 4 * block + row;
    return pos < p0 ? -1 : pos - p0;
  }

  // Last serial write to tail slot 0..2. -1 retains the old slot unchanged,
  // including when a short continuation never touches that slot.
  constexpr int64_t tail_input_row(int slot) const {
    if (end <= slot)
      return -1;
    const int64_t last = end - 1 - (end - 1 - slot) % 4;
    return last < p0 ? -1 : last - p0;
  }

  constexpr int64_t last_completed_block() const {
    return first_block + completed - 1;
  }
  constexpr int64_t spare_block() const {
    return first_block + completed;
  }
};

constexpr QsaIndexerBatchPlan qsa_indexer_batch_plan(int64_t p0, int64_t n) {
  return {p0, p0 + n, p0 / 4, (p0 + n) / 4 - p0 / 4,
          p0 == 0 && n > 0};
}
} // namespace strata::kernels::detail
'''

# Move the existing exact reduction/normalization/rotation body; do not maintain
# a second numerical oracle. No statement changes except the non-writing lanes'
# void return becomes a dummy float return after the same two local barriers.
start = original.index('          auto sg = item.get_sub_group();')
finish = original.index('          b.pooled[size_t(block) * 128 + d] = y;', start)
math = textwrap.dedent(original[start:finish])
math = math.replace('if (d >= 128)\n  return;', 'if (d >= 128)\n  return 0.f;')
helper = '''// Shared by the serial and completed-block kernels: same 256 lanes, subgroup
// shuffle tree, local partials, barriers, RMSNorm and analytic/table IMRoPE.
inline float normalize_rotate(sycl::nd_item<1> item, float mean, int pos,
                              int base, const float *gamma, float eps,
                              sycl::local_accessor<float, 1> values,
                              sycl::local_accessor<float, 1> partials,
                              const Angles &a) {
  const int d = item.get_local_linear_id();
''' + textwrap.indent(math, '  ') + '  return y;\n}\n'
proposed = original[:start] + '''          const float y = normalize_rotate(item, mean, pos, base, gamma, eps,
                                           values, partials, a);
          if (d >= 128)
            return;
          const int block = pos / 4, rp = pos == 0 ? 0 : base + 4 * block;
''' + original[finish:]
append_at = proposed.index('void append(sycl::queue &q')
proposed = proposed[:append_at] + helper + proposed[append_at:]
proposed = proposed.replace('#include "strata/kernels/native_qsa_indexer.hpp"',
    '#include "strata/kernels/native_qsa_indexer.hpp"\n#include "strata/kernels/qsa_indexer_batch_plan.hpp"')
proposed = proposed.replace('#include <cmath>', '#include <cmath>\n#include <cstdlib>')

batch = '''// One in-order queue and at most three submissions regardless of n. Bulk
// work writes completed keys only: intermediate spare writes would race with
// the next work-group's completed key. Final metadata runs after all groups.
void append_batch(sycl::queue &q, const float *raw, int64_t n, int64_t p0,
                  int base, const float *gamma, float eps, QsaIndexerBuffers b,
                  int capacity, Angles a) {
  const auto plan = detail::qsa_indexer_batch_plan(p0, n);
  if (plan.initialize_dead)
    append(q, raw, nullptr, 0, base, gamma, eps, b, capacity, a);
  if (plan.completed) {
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 1> values(128, h), partials(8, h);
      h.parallel_for(
          sycl::nd_range<1>(size_t(plan.completed) * 256, 256),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const int d = item.get_local_linear_id();
            const int block = int(plan.first_block + item.get_group_linear_id());
            float mean = 0;
            if (d < 128) {
              const int64_t first = plan.input_row(block, 0);
              float sum = first < 0 ? b.tail[d]
                                    : float(sycl::half(raw[size_t(first) * 128 + d]));
              for (int j = 1; j < 4; ++j) {
                const int64_t row = plan.input_row(block, j);
                sum += row < 0 ? b.tail[j * 128 + d]
                               : float(sycl::half(raw[size_t(row) * 128 + d]));
              }
              mean = sycl::fma(.25f, sum, 0.f);
            }
            const float y = normalize_rotate(item, mean, 4 * block + 3, base,
                                             gamma, eps, values, partials, a);
            if (d < 128)
              b.pooled[size_t(block) * 128 + d] = y;
          });
    });
  }
  // Ordered after bulk's reads of the preexisting partial block. Reproduce the
  // last serial write to each slot, even for aligned ends (retained full-block
  // rows) and tiny chunks with untouched tail slots. No new waits or events.
  q.parallel_for(sycl::range<1>(128), [=](sycl::id<1> id) {
    const int d = int(id[0]);
    for (int slot = 0; slot < 3; ++slot) {
      const int64_t row = plan.tail_input_row(slot);
      if (row >= 0)
        b.tail[slot * 128 + d] = float(sycl::half(raw[size_t(row) * 128 + d]));
    }
    if (plan.completed) {
      b.pooled[size_t(plan.spare_block()) * 128 + d] = b.dead[d];
      if (d == 0)
        *b.block_pos = base + 4 * int(plan.last_completed_block());
    }
  });
}
'''
pos = proposed.index('} // namespace\nvoid native_qsa_indexer_set_enabled')
proposed = proposed[:pos] + batch + proposed[pos:]
begin = proposed.index('  // Baseline: preserve the exact single-cell state transition')
end = proposed.index('  for (int64_t i = 0; i < n; ++i)', begin)
proposed = proposed[:begin] + '''  // Only the exact value "1" enables this uncaptured host-position path.
  // Leave the existing serial transition as the default and comparison arm.
  const char *batch_flag = std::getenv("STRATA_SYCL_QSA_INDEXER_BATCH");
  if (batch_flag && batch_flag[0] == '1' && batch_flag[1] == '\\0') {
    append_batch(q, raw, n, p0, base, gamma, eps, b, int(capacity), a);
    return;
  }
''' + proposed[end:]

api = (ROOT / API).read_text()
new_api = api.replace('// Host-side positions (not for a captured graph).', '''// Host-side positions (not for a captured graph). SYCL uses the serial baseline
// by default; exact STRATA_SYCL_QSA_INDEXER_BATCH=1 opts into at most three
// ordered submissions with shared serial normalization/rotation arithmetic.
// The switch does not affect the device-position single append or its captures.''')
contents = {CPP: proposed, API: new_api, PLAN: planner}
patch = ''
for name, value in contents.items():
    target = OUT / 'proposed' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value)
    old = (ROOT / name).read_text() if (ROOT / name).exists() else ''
    patch += ''.join(difflib.unified_diff(old.splitlines(True), value.splitlines(True),
        fromfile='a/' + name if old else '/dev/null', tofile='b/' + name))
(OUT / 'unified.patch').write_text(patch)

source_names = [CPP, API, 'include/strata/kernels/qsa.hpp',
    'include/strata/kernels/mrope.hpp', 'include/strata/kernels/rope_scaling.hpp',
    'include/strata/sycl/launch.hpp', 'src/prefill/prefill.cpp',
    'src/core/layer.cpp', 'src/kernels/qsa_parity.cpp', 'cmake/sycl_backend.cmake']
provenance = {
    'status': 'proposal_only_not_applied_not_built_not_tested',
    'date': '2026-10-15 JST',
    'origin': 'current root SYCL source only; no PR6/history/donor code',
    'base_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in source_names},
    'proposed_sha256': {name: hashlib.sha256(value.encode()).hexdigest()
                        for name, value in contents.items()},
    'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
    'new_file': PLAN,
    'independent_host_fixture': 'fixture/coverage.cpp includes the exact proposed planning header',
    'validation': {'source_review': 'performed', 'patch_apply_check': 'deferred',
                   'CPU_build_test': 'deferred', 'SYCL_build': 'deferred',
                   'GPU_correctness': 'deferred', 'runtime_performance': 'deferred'},
    'retention': {'owner': 'Main', 'consumer': 'serial post-benchmark proposal review; PR6 recovery has priority',
                  'byte_budget': 262144, 'next_review': 'after active benchmark closes and PR6 recovery review',
                  'raw_captures': 'none', 'deletions': 'none'},
}
(OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print('Prepared proposal only:', OUT)
print('unified.patch SHA256:', provenance['patch_sha256'])
