#pragma once
#include "strata/sycl/gguf_decode.hpp"
#include "strata/sycl/iq_parallel_decode.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <array>
#include <stdexcept>
#include <vector>

namespace fixture {
inline constexpr std::array<int, 6> types{18, 22, 21, 23, 20, 42};
inline constexpr std::array<uint16_t, 18> edge_scales{
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
    0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfc01,
    0x7e00, 0xfe00};
inline float element(int type, const uint8_t *p, unsigned j) {
  using namespace strata::sycl_backend;
  switch (type) {
  case 18: return parallel_iq_element<18>(p, j);
  case 20: return parallel_iq_element<20>(p, j);
  case 21: return parallel_iq_element<21>(p, j);
  case 22: return parallel_iq_element<22>(p, j);
  case 23: return parallel_iq_element<23>(p, j);
  case 42: return parallel_iq_element<42>(p, j);
  default: throw std::invalid_argument("fixture element type");
  }
}
inline void put16(uint8_t *p, uint16_t v) {
  p[0] = uint8_t(v); p[1] = uint8_t(v >> 8);
}
inline void put32(uint8_t *p, uint32_t v) {
  for (unsigned b = 0; b < 4; ++b) p[b] = uint8_t(v >> (8 * b));
}
// Deterministic coverage corpus, not model weights. seq=0..1023 visits every
// admitted grid index, every stored sign byte, parity input and scale field.
// Multiple cells/groups deliberately differ to expose group/byte-stride errors.
inline void make_block(int type, uint8_t *p, unsigned seq, uint16_t scale) {
  const auto f = strata::sycl_backend::block_format(type);
  uint32_t state = 0x9e3779b9u ^ (seq * 0x85ebca6bu) ^ unsigned(type);
  for (int b = 0; b < f.bytes; ++b) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    p[b] = uint8_t(state);
  }
  put16(p, scale);
  if (type == 18) {
    for (unsigned g = 0; g < 8; ++g) {
      for (unsigned w = 0; w < 8; ++w)
        p[2 + g * 8 + w] = uint8_t(seq + g * 8 + w);
      uint32_t aux = ((seq + g) & 15u) << 28;
      for (unsigned pair = 0; pair < 4; ++pair)
        aux |= ((seq + g * 4 + pair) & 127u) << (7 * pair);
      put32(p + 66 + 4 * g, aux);
    }
  } else if (type == 21) {
    for (unsigned g = 0; g < 8; ++g) {
      p[66 + g] = 0;
      for (unsigned w = 0; w < 8; ++w) {
        const unsigned index = (seq + g * 8 + w) & 511u;
        p[2 + g * 8 + w] = uint8_t(index);
        p[66 + g] |= uint8_t((index >> 8) << w);
      }
      for (unsigned b = 0; b < 4; ++b)
        p[74 + g * 4 + b] = uint8_t(seq + g * 4 + b);
    }
    for (unsigned b = 0; b < 4; ++b)
      p[106 + b] = uint8_t(((seq + 2*b) & 15) | (((seq + 2*b+1) & 15) << 4));
  } else if (type == 22) {
    for (unsigned g = 0; g < 8; ++g) {
      p[66 + g] = 0;
      for (unsigned c = 0; c < 4; ++c) {
        const unsigned index = (seq + g * 4 + c) & 1023u;
        p[2 + g * 4 + c] = uint8_t(index);
        p[66 + g] |= uint8_t((index >> 8) << (2 * c));
        p[34 + g * 4 + c] = uint8_t(seq + g * 4 + c);
      }
      p[74 + g] = uint8_t(((seq + g) & 15) | (((seq + g + 1) & 15) << 4));
    }
  } else if (type == 23) {
    uint16_t hi = 0;
    for (unsigned g = 0; g < 8; ++g) {
      const unsigned s = (seq + g) & 63u;
      hi |= uint16_t((s >> 4) << (2 * g));
      if (!(g % 2)) p[4 + g / 2] = uint8_t(s & 15);
      else p[4 + g / 2] |= uint8_t((s & 15) << 4);
      for (unsigned b = 0; b < 16; ++b)
        p[8 + g * 16 + b] = uint8_t(((seq + b) & 15) |
                                         (((seq + b + 7) & 15) << 4));
    }
    put16(p + 2, hi);
  } else if (type == 20) {
    for (unsigned b = 0; b < 16; ++b)
      p[2+b] = uint8_t(((seq + b) & 15) | (((seq + b + 7) & 15) << 4));
  } else if (type == 42) {
    for (unsigned b = 0; b < 16; ++b) p[2+b] = uint8_t(seq + b);
  }
}
inline std::vector<uint16_t> oracle(int type, const uint8_t *p, size_t count) {
  const auto f = strata::sycl_backend::block_format(type);
  std::vector<uint16_t> out(count);
  for (size_t i = 0; i < count; i += size_t(f.width)) {
    float values[256];
    strata::sycl_backend::decode_block(type, p + i / f.width * f.bytes, values);
    for (int j = 0; j < f.width; ++j)
      out[i+j] = strata::kernels::f16_from_f32(values[j]);
  }
  return out;
}
} // namespace fixture
