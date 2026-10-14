#!/usr/bin/env python3
"""Fabricate this proposal only; never apply, import, build or run product code.

The source pin is mandatory. Outputs stay beside this script. CPU fixture
generation extracts the real unchanged host validator, not a second validator.
"""
from pathlib import Path
import difflib
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SOURCE = "src/kernels/sycl/qsa_select.cpp"
BASE = "45084756ca5e8e36085dae46581049c81366a45cdfde23e8c6f38defbba0c809"
HEADER = "include/strata/kernels/qsa_score_grid_stride.hpp"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def put(relative, data):
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)

base_bytes = (ROOT / SOURCE).read_bytes()
if sha(base_bytes) != BASE:
    raise SystemExit("Source pin changed; Main must review/rebase before generating")
if (ROOT / HEADER).exists():
    raise SystemExit("Proposed header already exists in product; review instead of overwriting")
base = base_bytes.decode()
header = '''#pragma once

#include <cstdint>

namespace strata::kernels::detail {

// Host policy only: missing, empty and every spelling except exactly "1" are off.
inline bool score_grid_stride_enabled(const char *value) {
  return value && value[0] == '1' && value[1] == 0;
}

// Construct only with the original positive bound returned by validate().
// Capacity/active bound and launch group count have different meanings.
struct ScoreGridStride {
  const int64_t active_bound;
  const int64_t group_count;

  constexpr explicit ScoreGridStride(int64_t bound)
      : active_bound(bound), group_count(bound < 4096 ? bound : 4096) {}

  constexpr int64_t query_for_group(int64_t group) const {
    return group / group_count;
  }
  constexpr int64_t first_block(int64_t group) const {
    return group % group_count;
  }

  template <typename Function>
  void for_each_block(int64_t n, int64_t nb, int64_t first,
                      Function function) const {
    // Read n/nb from the device step, just as in the original score kernel.
    // nb >= group_count would incorrectly reject long-context queries.
    if (n < 0 || nb != n / 4 || nb >= active_bound)
      return;
    for (int64_t b = first; b <= nb; b += group_count)
      function(b);
  }
};

} // namespace strata::kernels::detail
'''
body_start = base.index('        const int lane = it.get_local_linear_id();')
body_end = base.index('\n      });', body_start)
body = base[body_start:body_end]
# Only indentation changes; all FP statements and their order are reused verbatim.
indented_body = '\n'.join('    ' + line for line in body.splitlines())
branch = '''  // Process-latched opt-in; captured launches retain this fixed policy/shape.
  static const bool grid_stride = detail::score_grid_stride_enabled(
      std::getenv("STRATA_SYCL_QSA_SCORE_GRID_STRIDE"));
  if (grid_stride) {
    const detail::ScoreGridStride grid{launched};
    auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>(size_t(nq * grid.group_count) * 32, 32),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const int64_t qi = grid.query_for_group(it.get_group_linear_id()),
                        first = grid.first_block(it.get_group_linear_id());
          const auto st = steps + qi * kStepCount;
          const int64_t n = st[kStepNKv], nb = st[kStepNBid];
          grid.for_each_block(n, nb, first, [&](int64_t b) {
''' + indented_body + '''
          });
        });
    finish(stream, event);
    return;
  }
'''
proposed = base.replace('#include "strata/kernels/qsa_select.hpp"',
                        '#include "strata/kernels/qsa_select.hpp"\n#include "strata/kernels/qsa_score_grid_stride.hpp"', 1)
proposed = proposed.replace('#include <climits>', '#include <climits>\n#include <cstdlib>', 1)
anchor = '  auto event = queue_for(stream).parallel_for('
proposed = proposed.replace(anchor, branch + anchor, 1)
put("proposed/" + SOURCE, proposed)
put("proposed/" + HEADER, header)
patch = ''.join(difflib.unified_diff(base.splitlines(True), proposed.splitlines(True),
                                  fromfile='a/' + SOURCE, tofile='b/' + SOURCE))
patch += ''.join(difflib.unified_diff([], header.splitlines(True),
                                   fromfile='/dev/null', tofile='b/' + HEADER))
put("proposal.patch", patch)
validator = base[base.index('int64_t validate('):base.index('\nuint32_t order(')]
put("fixture/extracted_validate.hpp", '''#pragma once
#include "strata/kernels/qsa.hpp"
#include <climits>
#include <stdexcept>
namespace fixture_source {
using strata::kernels::QsaShapes;
// Byte-derived from the pinned qsa_select.cpp host validator; not handwritten.
''' + validator + '\n} // namespace fixture_source\n')
identities = {}
for name in [SOURCE, "include/strata/kernels/qsa_select.hpp", "include/strata/kernels/qsa.hpp",
             "include/strata/sycl/launch.hpp", "cmake/sycl_backend.cmake", "src/core/verify.cpp",
             "src/core/layer.cpp", "AGENTS.md", "docs/ARTIFACT_RETENTION.md",
             "bench/results/2026-10-14-primary-routed-fp16-census/research-index.json",
             "bench/results/2026-10-14-primary-routed-fp16-census/research/research-long-context-qsa-indexer-budget-round550-v1.txt",
             "bench/results/2026-10-14-primary-routed-fp16-census/research-round549-550-review.ja.md"]:
    identities[name] = sha((ROOT / name).read_bytes())
put("provenance.json", json.dumps({
    "status": "proposal-only; unbuilt; fixtures uncompiled/unrun; unapplied; unadopted",
    "scope": "root SYCL qsa_block_scores only; no Git state read or mutation",
    "base_sha256": identities,
    "proposed_sha256": {SOURCE: sha(proposed.encode()), HEADER: sha(header.encode())},
    "patch_sha256": sha(patch.encode()),
    "policy": {"env": "STRATA_SYCL_QSA_SCORE_GRID_STRIDE", "enabled_value": "1",
               "default": "off", "latch": "first valid qsa_block_scores call per process",
               "work_groups_per_query": "min(original validated active bound, 4096)",
               "device_validity_bound": "original validated active bound"},
    "generator": "artifact fabrication only; no product import/compilation/execution",
    "fixture": "actual shared traversal plus byte-extracted actual host validator; independent baseline enumeration; CPU-only; unrun",
    "prior_artifacts": "Stager affinity/error and MTP ledger proposals left untouched",
    "owner_and_next_review": "Main; after active benchmark and requested PR6 audit boundary; review source pins then serial CPU/SYCL correctness and performance",
}, indent=2) + '\n')
print("Fabricated proposal sources/patch/validator/provenance only; no tests or builds run.")
