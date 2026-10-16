#pragma once
#include "strata/kernels/qsa.hpp"
#include <climits>
#include <stdexcept>
namespace fixture_source {
using strata::kernels::QsaShapes;
// Byte-derived from the pinned qsa_select.cpp host validator; not handwritten.
int64_t validate(int64_t nq, int64_t blocks, int64_t active,
                 const QsaShapes &s) {
  if (nq <= 0 || nq > 65535 || blocks <= 0 || blocks > INT_MAX ||
      s.idx_dim != 128 || s.idx_n_head != 4 || s.idx_block != 4 ||
      s.idx_top_k < 0 || s.idx_top_k > INT_MAX - 3 ||
      (active != -1 && (active <= 0 || active > blocks)))
    throw std::invalid_argument("invalid SYCL block indexer geometry/capacity");
  return active == -1 ? blocks : active;
}
} // namespace fixture_source
