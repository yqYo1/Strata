#pragma once
#include "strata/artifact/dequant.hpp"

namespace strata::sycl_backend {
struct BlockFormat {
  int width, bytes;
};
inline BlockFormat block_format(int type) {
  switch (type) {
  case 2:
    return {32, 18};
  case 6:
    return {32, 22};
  case 7:
    return {32, 24};
  case 8:
    return {32, 34};
  case 11:
    return {256, 110};
  case 12:
    return {256, 144};
  case 13:
    return {256, 176};
  case 14:
    return {256, 210};
  case 18:
    return {256, 98};
  case 16:
    return {256, 66};
  case 17:
    return {256, 74};
  case 29:
    return {256, 56};
  case 20:
    return {32, 18};
  case 21:
    return {256, 110};
  case 22:
    return {256, 82};
  case 23:
    return {256, 136};
  case 30:
    return {1, 2};
  case 42:
    return {64, 18};
  default:
    return {0, 0};
  }
}
// Use the repository's scalar GGUF definitions for transfer-time
// dequantization. Dot-product kernels keep their separate packed arithmetic and
// are cross-checked against these definitions. This path has no matrix-multiply
// performance claim.
inline void decode_block(int type, const uint8_t *src, float *out) {
  using namespace strata;
  switch (type) {
  case 2:
    dequantize_q4_0(src, out);
    break;
  case 6:
    dequantize_q5_0(src, out);
    break;
  case 7:
    dequantize_q5_1(src, out);
    break;
  case 8:
    dequantize_q8_0(src, out);
    break;
  case 11:
    dequantize_q3_K(src, out);
    break;
  case 12:
    dequantize_q4_K(src, out);
    break;
  case 13:
    dequantize_q5_K(src, out);
    break;
  case 14:
    dequantize_q6_K(src, out);
    break;
  case 18:
    dequantize_iq3_xxs(src, out);
    break;
  case 16:
    dequantize_iq2_xxs(src, out);
    break;
  case 17:
    dequantize_iq2_xs(src, out);
    break;
  case 29:
    dequantize_iq1_m(src, out);
    break;
  case 20:
    dequantize_iq4_nl(src, out);
    break;
  case 21:
    dequantize_iq3_s(src, out);
    break;
  case 22:
    dequantize_iq2_s(src, out);
    break;
  case 23:
    dequantize_iq4_xs(src, out);
    break;
  case 30:
    dequantize_bf16(src, out, 1);
    break;
  case 42:
    dequantize_q2_0(src, out);
    break;
  }
}
} // namespace strata::sycl_backend
