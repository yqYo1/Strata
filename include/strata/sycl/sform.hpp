#pragma once
#include "strata/kernels/s_gemv.hpp"
#include "strata/sycl/launch.hpp"
#include <climits>

namespace strata::sycl_backend {
inline void validate_sform(const kernels::SForm &f, int64_t ni, int64_t no) {
  if (ni <= 0 || ni > INT_MAX || no <= 0 || no > INT_MAX ||
      (f.code_bits != 2 && f.code_bits != 4 && f.code_bits != 8) ||
      f.group_elems <= 0 || (f.group_elems & (f.group_elems - 1)) ||
      f.group_elems % 4 || ni % f.group_elems ||
      (f.codebook != kernels::Codebook::Affine &&
       f.codebook != kernels::Codebook::Iq4Nl) ||
      (f.codebook == kernels::Codebook::Iq4Nl && f.code_bits != 4) ||
      (f.act_kind != 0 && f.act_kind != 1) || f.code_bias < -256 ||
      f.code_bias > 256)
    throw std::invalid_argument("invalid SYCL canonical weight geometry/form");
}
inline float canonical_value(const uint8_t *codes, const float *scales,
                             const float *offset, size_t row, int64_t col,
                             int64_t ni, kernels::SForm f) {
  constexpr int table[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                             1,    13,   25,  38,  53,  69,  89,  113};
  const int per = 8 / f.code_bits;
  const int code =
      (codes[row * (ni / per) + col / per] >> ((col % per) * f.code_bits)) &
      ((1 << f.code_bits) - 1);
  const int value =
      f.codebook == kernels::Codebook::Iq4Nl ? table[code] : code + f.code_bias;
  const size_t group = row * (ni / f.group_elems) + col / f.group_elems;
  return sycl::fma(float(value), scales[group],
                   f.has_offset ? offset[group] : 0.f);
}
} // namespace strata::sycl_backend
