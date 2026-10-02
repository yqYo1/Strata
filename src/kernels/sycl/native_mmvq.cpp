// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/sycl/launch.hpp"
#include <atomic>
#include <sycl/ext/intel/math.hpp>

namespace strata::kernels {
void native_q5_head_esimd(const void *, const void *, float *, int, int, void *);
void native_q3_k_esimd(const void *, const void *, float *, int, int, int, void *);
void native_q4_k_esimd(const void *, const void *, float *, int, int, int, void *);
void native_q2_0_esimd(const void *, const void *, float *, int, int, int, void *);
void native_iq4_xs_esimd(const void *, const void *, float *, int, int, int, void *);
void native_q2_grouped_esimd(bool, const NativeExpertLayout &,
                            const unsigned long long *, const int32_t *,
                            const int32_t *, const int32_t *, const int32_t *,
                            int, int, const void *, float *, float *, float *, void *);
namespace {
using namespace sycl_backend;
// On-disk half values are read through the shared bit converter, including
// subnormals.
struct half {
  uint16_t bits;
  operator float() const { return f32_from_f16(bits); }
};
struct alignas(4) half2 {
  half x, y;
};
struct float2 {
  float x, y;
};
struct int2 {
  int x, y;
};
float2 half_pair_float(half2 h) { return {float(h.x), float(h.y)}; }
float half_low(half2 h) { return h.x; }
float half_float(half h) { return h; }
int packed_dot(int a, int b, int acc) {
  for (int i = 0; i < 4; ++i)
    acc += int(int8_t(uint32_t(a) >> (8 * i))) *
           int(int8_t(uint32_t(b) >> (8 * i)));
  return acc;
}
uint32_t byte_permute(uint32_t a, uint32_t b, uint32_t selector) {
  const uint64_t joined = uint64_t(a) | (uint64_t(b) << 32);
  uint32_t result = 0;
  for (int i = 0; i < 4; ++i)
    result |= uint32_t((joined >> (8 * ((selector >> (4 * i)) & 7))) & 255)
              << (8 * i);
  return result;
}
int subtract_saturating_bytes(int a, int b) {
  uint32_t result = 0;
  for (int i = 0; i < 4; ++i) {
    const int v = int(int8_t(uint32_t(a) >> (8 * i))) -
                  int(int8_t(uint32_t(b) >> (8 * i)));
    result |= uint32_t(uint8_t(sycl::clamp(v, -128, 127))) << (8 * i);
  }
  return int(result);
}
constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;

struct Q5KBlock {
  half2 dm;
  uint8_t scales[12];
  uint8_t qh[32];
  uint8_t qs[128];
};
struct Q81Block {
  half2 ds;
  int8_t qs[32];
};
struct Q20Block {
  half d;
  uint8_t qs[16];
};
struct Q3KBlock {
  uint8_t hmask[32];
  uint8_t qs[64];
  uint8_t scales[12];
  half d;
};
struct IQ4XSBlock {
  half d;
  uint16_t scales_h;
  uint8_t scales_l[4];
  uint8_t qs[128];
};
struct Q4KBlock {
  half2 dm;
  uint8_t scales[12];
  uint8_t qs[128];
};
struct Q6KBlock {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  half d;
};
struct Q40Block {
  half d;
  uint8_t qs[16];
};
struct Q50Block {
  half d;
  uint8_t qh[4];
  uint8_t qs[16];
};
struct Q80Block {
  half d;
  int8_t qs[32];
};
struct IQ4NLBlock {
  half d;
  uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 &&
              offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 &&
              offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 &&
              offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 &&
              offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 &&
              offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 &&
              offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 &&
              offsetof(IQ4NLBlock, qs) == 2);

inline int load_int_b2(const void *ptr, int i32) {
  const auto *x = static_cast<const uint16_t *>(ptr);
  int value = x[2 * i32] << 0;
  value |= x[2 * i32 + 1] << 16;
  return value;
}
inline float q5_q8_dot_impl(const int *__restrict__ vl,
                            const int *__restrict__ vh,
                            const int *__restrict__ u,
                            const uint8_t *__restrict__ sc,
                            const uint8_t *__restrict__ m, const half2 &dm5,
                            const float *__restrict__ d8) {
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
    const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
    const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
    const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
    const int v0i = vl0i | vh0i;
    const int v1i = vl1i | vh1i;
    const int dot1 =
        packed_dot(v0i, u[2 * i], packed_dot(v1i, u[2 * i + 1], 0));
    const int dot2 = packed_dot(0x01010101, u[2 * i],
                                packed_dot(0x01010101, u[2 * i + 1], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm5f = half_pair_float(dm5);
  return dm5f.x * sumf_d - dm5f.y * sumf_m;
}
inline float q3_q8_dot_impl(int vl, int vh, const int *__restrict__ u,
                            const uint8_t *__restrict__ scales,
                            int scale_offset, float d3,
                            const float *__restrict__ d8) {
  float sumf = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int isc = scale_offset + 2 * i;
    const int isc_low = isc % 8;
    const int sc_shift_low = 4 * (isc / 8);
    const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
    const int isc_high = isc % 4;
    const int sc_shift_high = 2 * (isc / 4);
    const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
    const int sc = (sc_low | sc_high) - 32;
    const int vil = (vl >> (2 * i)) & 0x03030303;
    const int vih = ((vh >> i) << 2) & 0x04040404;
    const int vi = subtract_saturating_bytes(vil, vih);
    sumf += d8[i] * (packed_dot(vi, u[i], 0) * sc);
  }
  return d3 * sumf;
}
inline float q4_q8_dot_impl(const int *__restrict__ v,
                            const int *__restrict__ u,
                            const uint8_t *__restrict__ sc,
                            const uint8_t *__restrict__ m, const half2 &dm4,
                            const float *__restrict__ d8) {
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
    const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
    const int dot1 =
        packed_dot(v1i, u[2 * i + 1], packed_dot(v0i, u[2 * i], 0));
    const int dot2 = packed_dot(0x01010101, u[2 * i + 1],
                                packed_dot(0x01010101, u[2 * i], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm4f = half_pair_float(dm4);
  return dm4f.x * sumf_d - dm4f.y * sumf_m;
}
inline float q6_q8_dot_impl(int vl, int vh, const int *__restrict__ u,
                            const int8_t *__restrict__ scales, float d,
                            const float *__restrict__ d8) {
  float sumf = 0.0f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int sc = scales[4 * i];
    const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
    const int vi = subtract_saturating_bytes(vil | vih, 0x20202020);
    sumf += d8[i] * (packed_dot(vi, u[i], 0) * sc);
  }
  return d * sumf;
}

int2 iq4_table_lookup(int packed) {
  constexpr int8_t values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                 1,    13,   25,  38,  53,  69,  89,  113};
  uint32_t low = 0, high = 0;
  for (int i = 0; i < 4; ++i) {
    const unsigned byte = uint32_t(packed) >> (8 * i);
    low |= uint32_t(uint8_t(values[byte & 15])) << (8 * i);
    high |= uint32_t(uint8_t(values[(byte >> 4) & 15])) << (8 * i);
  }
  return {int(low), int(high)};
}
inline float small_q8_dot(const Q40Block *__restrict__ w,
                          const Q81Block *__restrict__ x, int iqs) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int v = load_int_b2(w->qs, iqs + i);
    const int vi0 = (v >> 0) & 0x0f0f0f0f;
    const int vi1 = (v >> 4) & 0x0f0f0f0f;
    sumi = packed_dot(vi0, reinterpret_cast<const int *>(x->qs)[iqs + i], sumi);
    sumi = packed_dot(vi1, reinterpret_cast<const int *>(x->qs)[iqs + i + 4],
                      sumi);
  }
  const float2 ds = half_pair_float(x->ds);
  const float d = w->d;
  return d * (sumi * ds.x - 4 * ds.y);
}
inline float small_q8_dot(const Q50Block *__restrict__ w,
                          const Q81Block *__restrict__ x, int iqs) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int vl = load_int_b2(w->qs, iqs + i);
    const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
    int vi0 = (vl >> 0) & 0x0f0f0f0f;
    vi0 |= (vh << 4) & 0x00000010;
    vi0 |= (vh << 11) & 0x00001000;
    vi0 |= (vh << 18) & 0x00100000;
    vi0 |= (vh << 25) & 0x10000000;
    sumi = packed_dot(vi0, reinterpret_cast<const int *>(x->qs)[iqs + i], sumi);
    int vi1 = (vl >> 4) & 0x0f0f0f0f;
    vi1 |= (vh >> 12) & 0x00000010;
    vi1 |= (vh >> 5) & 0x00001000;
    vi1 |= (vh << 2) & 0x00100000;
    vi1 |= (vh << 9) & 0x10000000;
    sumi = packed_dot(vi1, reinterpret_cast<const int *>(x->qs)[iqs + i + 4],
                      sumi);
  }
  const float2 ds = half_pair_float(x->ds);
  const float d = w->d;
  return d * (sumi * ds.x - 8 * ds.y);
}
inline float small_q8_dot(const Q80Block *__restrict__ w,
                          const Q81Block *__restrict__ x, int iqs) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int v = load_int_b2(w->qs, iqs + i);
    const int u = reinterpret_cast<const int *>(x->qs)[iqs + i];
    sumi = packed_dot(v, u, sumi);
  }
  const float d0 = w->d;
  const float d1 = half_low(x->ds);
  return d0 * d1 * float(sumi);
}
inline float small_q8_dot(const IQ4NLBlock *__restrict__ w,
                          const Q81Block *__restrict__ x, int iqs) {
  const int *q8 = reinterpret_cast<const int *>(x->qs) + iqs;
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int2 v = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
    sumi = packed_dot(v.x, q8[i], sumi);
    sumi = packed_dot(v.y, q8[i + 4], sumi);
  }
  const float d = half_float(w->d) * half_low(x->ds);
  return d * sumi;
}
struct Q5KTraits {
  using Block = Q5KBlock;
  static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K,
                       BPI = VDR * WARPS * WARP / QI;
  static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
  struct W {
    int vl[2], vh[2];
    uint16_t aux[2];
    half2 dm;
    int bq8_offset;
  };
  static W load(const Block *__restrict__ bq5, int iqs) {
    W r;
    r.bq8_offset = 2 * ((iqs / 2) / 4);
    const int *ql = reinterpret_cast<const int *>(bq5->qs + 16 * r.bq8_offset +
                                                  4 * ((iqs / 2) % 4));
    const int *qh =
        reinterpret_cast<const int *>(bq5->qh + 4 * ((iqs / 2) % 4));
    r.vl[0] = ql[0];
    r.vl[1] = ql[4];
    r.vh[0] = qh[0] >> r.bq8_offset;
    r.vh[1] = qh[4] >> r.bq8_offset;
    const uint16_t *scales = reinterpret_cast<const uint16_t *>(bq5->scales);
    const int j = r.bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                        ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                        ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    r.dm = bq5->dm;
    return r;
  }
  static float apply(const W &r, const Q81Block *__restrict__ bq8, int iqs) {
    int u[4];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const Q81Block *bq8i = bq8 + r.bq8_offset + i;
      d8[i] = half_low(bq8i->ds);
      const int *q8 = reinterpret_cast<const int *>(bq8i->qs) + ((iqs / 2) % 4);
      u[2 * i] = q8[0];
      u[2 * i + 1] = q8[4];
    }
    const uint8_t *sc = reinterpret_cast<const uint8_t *>(r.aux);
    return q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
  }
};
struct Q4KTraits {
  using Block = Q4KBlock;
  static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K,
                       BPI = VDR * WARPS * WARP / QI;
  static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
  struct W {
    int v[2];
    uint16_t aux[2];
    half2 dm;
    int bq8_offset;
  };
  static W load(const Block *__restrict__ bq4, int iqs) {
    W r;
    r.bq8_offset = 2 * ((iqs / 2) / 4);
    const int *ql = reinterpret_cast<const int *>(bq4->qs + 16 * r.bq8_offset +
                                                  4 * ((iqs / 2) % 4));
    r.v[0] = ql[0];
    r.v[1] = ql[4];
    const uint16_t *scales = reinterpret_cast<const uint16_t *>(bq4->scales);
    const int j = r.bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                        ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                        ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    r.dm = bq4->dm;
    return r;
  }
  static float apply(const W &r, const Q81Block *__restrict__ bq8, int iqs) {
    int u[4];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const Q81Block *bq8i = bq8 + r.bq8_offset + i;
      d8[i] = half_low(bq8i->ds);
      const int *q8 = reinterpret_cast<const int *>(bq8i->qs) + ((iqs / 2) % 4);
      u[2 * i] = q8[0];
      u[2 * i + 1] = q8[4];
    }
    const uint8_t *sc = reinterpret_cast<const uint8_t *>(r.aux);
    return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
  }
};
struct Q20Traits {
  using Block = Q20Block;
  static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
  static int kqs(int tid) { return tid % 2; }
  struct W {
    int qx[4], qy[4];
    float d2;
  };
  static W load(const Block *__restrict__ w, int iqs) {
    W r;
    r.d2 = w->d;
    const int16_t *qs = reinterpret_cast<const int16_t *>(w->qs) + iqs * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int q = qs[j];
      const int qe = byte_permute(0x020100ff, 0x020100ff, q >> 0);
      const int qo = byte_permute(0x020100ff, 0x020100ff, q >> 2);
      r.qx[j] = byte_permute(qe, qo, 0x5140);
      r.qy[j] = byte_permute(qe, qo, 0x7362);
    }
    return r;
  }
  static float apply(const W &r, const Q81Block *__restrict__ x, int iqs) {
    const Q81Block *chunk = x + iqs;
    const int *q8 = reinterpret_cast<const int *>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      sumi = packed_dot(q8[j * 2], r.qx[j], sumi);
      sumi = packed_dot(q8[j * 2 + 1], r.qy[j], sumi);
    }
    const float d8 = half_low(chunk->ds);
    return r.d2 * d8 * sumi;
  }
};
struct Q3KTraits {
  using Block = Q3KBlock;
  static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
  static int kqs(int tid) { return tid % 16; }
  struct W {
    int vl, vh;
    float d;
    const uint8_t *scales;
    int scale_offset, bq8_offset;
  };
  static W load(const Block *__restrict__ w, int iqs) {
    W r;
    r.bq8_offset = 4 * (iqs / 8);
    r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    r.d = w->d;
    r.vl = load_int_b2(w->qs, iqs);
    r.vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
    r.scales = w->scales;
    return r;
  }
  static float apply(const W &r, const Q81Block *__restrict__ x, int iqs) {
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      u[i] = reinterpret_cast<const int *>(x[r.bq8_offset + i].qs)[iqs % 8];
      d8[i] = half_low(x[r.bq8_offset + i].ds);
    }
    return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
  }
};
struct Q6KTraits {
  using Block = Q6KBlock;
  static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
  static int kqs(int tid) { return tid % 32; }
  struct W {
    int vl, vh;
    float d;
    const int8_t *scales;
    int bq8_offset;
  };
  static W load(const Block *__restrict__ w, int iqs) {
    W r;
    r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    r.vl = load_int_b2(w->ql, iqs);
    r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    r.scales = w->scales + scale_offset;
    r.d = w->d;
    return r;
  }
  static float apply(const W &r, const Q81Block *__restrict__ x, int iqs) {
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      u[i] = reinterpret_cast<const int *>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
      d8[i] = half_low(x[r.bq8_offset + 2 * i].ds);
    }
    return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
  }
};
struct IQ4XSTraits {
  using Block = IQ4XSBlock;
  static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
  static int kqs(int tid) { return 4 * (tid % 8); }
  struct W {
    int2 v[4];
    int ls;
    float dw;
  };
  static W load(const Block *__restrict__ w, int iqs) {
    W r;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      r.v[j] = iq4_table_lookup(reinterpret_cast<const int *>(w->qs)[iqs + j]);
    r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
           (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    r.dw = half_float(w->d);
    return r;
  }
  static float apply(const W &r, const Q81Block *__restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int u0 = reinterpret_cast<const int *>(x[iqs / 4].qs)[j];
      const int u1 = reinterpret_cast<const int *>(x[iqs / 4].qs)[j + 4];
      sumi = packed_dot(r.v[j].x, u0, sumi);
      sumi = packed_dot(r.v[j].y, u1, sumi);
    }
    sumi *= r.ls - 32;
    const float d = r.dw * half_low(x[iqs / 4].ds);
    return d * sumi;
  }
};
// The four 32-element formats: `load` keeps the block pointer (their decode is
// a few integer ops) and `apply` is the unchanged small_q8_dot. Their
// per-column cost is small; the win above is for the K and IQ formats.
template <typename Weight, int Qi> struct SmallTraits {
  using Block = Weight;
  static constexpr int DIV = 32, T = Qi / 2, KBY = 1,
                       BPI = 2 * WARPS * WARP / Qi;
  static int kqs(int tid) { return 2 * (tid % (Qi / 2)); }
  struct W {
    const Weight *w;
  };
  static W load(const Block *__restrict__ w, int) { return W{w}; }
  static float apply(const W &r, const Q81Block *__restrict__ x, int k) {
    return small_q8_dot(r.w, x, k);
  }
};

std::atomic<bool> multi_exact{true};
void validate_shape(int n_in, int ncols, int width = 32) {
  if (n_in <= 0 || n_in % width || ncols < 1 || ncols > 8)
    throw std::invalid_argument(
        "invalid SYCL native MMVQ block width/column count");
}
void validate_stream(void *stream) {
  if (!stream)
    throw std::invalid_argument(
        "SYCL native MMVQ requires an explicit in-order queue");
  (void)queue_for(stream);
}
struct Format {
  int width, bytes;
};
Format format(int type) {
  switch (type) {
  case 2:
    return {32, 18};
  case 6:
    return {32, 22};
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
  case 20:
    return {32, 18};
  case 23:
    return {256, 136};
  case 42:
    return {64, 18};
  default:
    throw std::invalid_argument("unsupported SYCL native MMVQ type");
  }
}

// Preserve the native thread-to-block mapping and ascending cross-subgroup
// partial sum, followed by XOR reduction. Each column uses the same arithmetic
// regardless of the total column count when multi_exact is enabled.
template <typename F, int ROWS, int NW>
void launch(const void *weights, const void *activations, float *y, int n_in,
            int n_out, int ncols, void *stream) {
  const auto *w = static_cast<const typename F::Block *>(weights);
  const auto *x = static_cast<const Q81Block *>(activations);
  const size_t groups = (size_t(n_out) + ROWS - 1) / ROWS;
  queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> partial((NW - 1) * ncols * ROWS * 32, h);
    h.parallel_for(
        sycl::nd_range<1>(groups * NW * 32, NW * 32),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
          const int tid = int(item.get_local_linear_id());
          const int lane = tid % 32, warp = tid / 32;
          const size_t row0 = item.get_group_linear_id() * ROWS;
          const int blocks = n_in / F::DIV;
          const size_t stride = size_t(n_in) / 32;
          float tmp[8][ROWS] = {};
          for (int kbx = tid / F::T; kbx < blocks; kbx += F::BPI * NW / 4) {
            const int kqs = F::kqs(tid);
            for (int i = 0; i < ROWS; ++i)
              if (row0 + i < size_t(n_out)) {
                const auto weight = F::load(w + (row0 + i) * blocks + kbx, kqs);
                for (int j = 0; j < ncols; ++j)
                  tmp[j][i] +=
                      F::apply(weight, x + j * stride + kbx * F::KBY, kqs);
              }
          }
          if (warp > 0)
            for (int j = 0; j < ncols; ++j)
              for (int i = 0; i < ROWS; ++i)
                partial[(((warp - 1) * ncols + j) * ROWS + i) * 32 + lane] =
                    tmp[j][i];
          item.barrier(sycl::access::fence_space::local_space);
          if (warp == 0) {
            const auto group = item.get_sub_group();
            for (int j = 0; j < ncols; ++j)
              for (int i = 0; i < ROWS; ++i) {
                for (int k = 0; k < NW - 1; ++k)
                  tmp[j][i] +=
                      partial[((k * ncols + j) * ROWS + i) * 32 + lane];
                for (int offset = 16; offset; offset >>= 1)
                  tmp[j][i] +=
                      sycl::permute_group_by_xor(group, tmp[j][i], offset);
                if (lane == i && row0 + i < size_t(n_out))
                  y[size_t(j) * n_out + row0 + i] = tmp[j][i];
              }
          }
        });
  });
}
template <typename F>
void dispatch(const void *w, const void *x, float *y, int n_in, int n_out,
              int ncols, void *stream) {
  if (ncols > 1 && !multi_exact.load(std::memory_order_relaxed)) {
    if (ncols <= 4)
      launch<F, 2, 4>(w, x, y, n_in, n_out, ncols, stream);
    else
      launch<F, 2, 2>(w, x, y, n_in, n_out, ncols, stream);
  } else if (n_in / F::DIV < F::BPI)
    launch<F, 4, 4>(w, x, y, n_in, n_out, ncols, stream);
  else
    launch<F, 1, 4>(w, x, y, n_in, n_out, ncols, stream);
}
template <typename F, bool Down>
void grouped_projection(const NativeExpertLayout &L,
                        const unsigned long long *pointers,
                        const int32_t *starts, const int32_t *ng,
                        const int32_t *dest, const int32_t *tokens, int groups,
                        int entries, const Q81Block *x, float *gate, float *up,
                        float *out, int tile, sycl::queue &q) {
  const int rows = Down ? int(L.n_embd) : int(L.n_ff * 2);
  const int ni = Down ? int(L.n_ff) : int(L.n_embd);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> partial(3 * 8 * 32, h);
    h.parallel_for(
        sycl::nd_range<2>({size_t(groups), size_t(rows) * 128}, {1, 128}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
          const int g = int(it.get_group(0)), row = int(it.get_group(1));
          const int count = *ng;
          if (count < 0 || count > groups || g >= count)
            return;
          const int begin = starts[g], end = starts[g + 1];
          if (begin < 0 || end < begin || end > entries || !pointers[g])
            return;
          const auto blob = reinterpret_cast<const uint8_t *>(pointers[g]);
          const auto weights = reinterpret_cast<const typename F::Block *>(
              blob + (Down ? L.down_off : 0));
          const int tid = int(it.get_local_linear_id()), lane = tid % 32,
                    warp = tid / 32;
          const int blocks = ni / F::DIV;
          for (int first = begin; first < end; first += tile) {
            const int used = sycl::min(tile, end - first);
            float sum[8] = {};
            for (int b = tid / F::T; b < blocks; b += F::BPI) {
              const int kqs = F::kqs(tid);
              const auto weight =
                  F::load(weights + size_t(row) * blocks + b, kqs);
              for (int j = 0; j < used; ++j) {
                const int input = Down ? first + j : tokens[first + j];
                if (input >= 0 && input < entries)
                  sum[j] += F::apply(
                      weight, x + size_t(input) * (ni / 32) + b * F::KBY, kqs);
              }
            }
            if (warp)
              for (int j = 0; j < used; ++j)
                partial[((warp - 1) * 8 + j) * 32 + lane] = sum[j];
            it.barrier(sycl::access::fence_space::local_space);
            if (!warp)
              for (int j = 0; j < used; ++j) {
                for (int w = 0; w < 3; ++w)
                  sum[j] += partial[(w * 8 + j) * 32 + lane];
                for (int offset = 16; offset; offset >>= 1)
                  sum[j] += sycl::permute_group_by_xor(it.get_sub_group(),
                                                       sum[j], offset);
                if (!lane) {
                  if constexpr (Down) {
                    const int dst = dest[first + j];
                    if (dst >= 0 && dst < entries)
                      out[size_t(dst) * L.n_embd + row] = sum[j];
                  } else {
                    float *dst = row < L.n_ff ? gate : up;
                    dst[size_t(first + j) * L.n_ff + row % L.n_ff] = sum[j];
                  }
                }
              }
            // A later entry tile reuses the cross-subgroup partials.
            it.barrier(sycl::access::fence_space::local_space);
          }
        });
  });
}
template <bool Down>
void grouped_dispatch(int type, const NativeExpertLayout &L,
                      const unsigned long long *ptr, const int32_t *start,
                      const int32_t *count, const int32_t *dst,
                      const int32_t *tok, int groups, int entries,
                      const Q81Block *x, float *gate, float *up, float *out,
                      int tile, sycl::queue &q) {
#define GROUPED(TYPE, ...)                                                     \
  case TYPE:                                                                   \
    grouped_projection<__VA_ARGS__, Down>(L, ptr, start, count, dst, tok,      \
                                          groups, entries, x, gate, up, out,   \
                                          tile, q);                            \
    break
  switch (type) {
    GROUPED(2, SmallTraits<Q40Block, 4>);
    GROUPED(6, SmallTraits<Q50Block, 4>);
    GROUPED(8, SmallTraits<Q80Block, 8>);
    GROUPED(11, Q3KTraits);
    GROUPED(12, Q4KTraits);
    GROUPED(13, Q5KTraits);
    GROUPED(14, Q6KTraits);
    GROUPED(20, SmallTraits<IQ4NLBlock, 4>);
    GROUPED(23, IQ4XSTraits);
    case 42:
      if (iq_old_kernels())
        grouped_projection<Q20Traits, Down>(L, ptr, start, count, dst, tok,
                                              groups, entries, x, gate, up,
                                              out, tile, q);
      else
        native_q2_grouped_esimd(Down, L, ptr, start, count, dst, tok, groups,
                                entries, x, gate, up, out, &q);
      break;
  }
#undef GROUPED
}
void composition(int type, const void *w, const float *x, void *scratch,
                 float *y, int n_in, int n_out, int ncols, void *stream) {
  const auto wb = native_mmvq_weight_bytes(type, n_in, n_out);
  const auto xb = native_q8_1_bytes(n_in, ncols);
  validate_stream(stream);
  validate_spans(
      {{scratch, xb}, {y, checked_count(checked_count(ncols, n_out), 4)}},
      {{w, wb}, {x, checked_count(checked_count(ncols, n_in), 4)}});
  native_quantize_q8_1(x, scratch, n_in, ncols, stream);
  native_mmvq(type, w, scratch, y, n_in, n_out, ncols, stream);
}
} // namespace

void native_mmvq_set_multi_exact(bool exact) {
  multi_exact.store(exact, std::memory_order_relaxed);
}
bool native_mmvq_multi_exact() {
  return multi_exact.load(std::memory_order_relaxed);
}
size_t native_q8_1_bytes(int n_in, int ncols) {
  validate_shape(n_in, ncols);
  return checked_count(checked_count(ncols, n_in / 32), 36);
}
void native_quantize_q8_1(const float *x, void *scratch, int n_in, int ncols,
                          void *stream) {
  const auto bytes = native_q8_1_bytes(n_in, ncols);
  const auto total = checked_count(n_in, ncols);
  validate_stream(stream);
  validate_spans({{scratch, bytes}}, {{x, checked_count(total, 4)}});
  auto *out = static_cast<Q81Block *>(scratch);
  queue_for(stream).parallel_for(
      sycl::nd_range<1>((total + 255) / 256 * 256, 256),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const size_t i = item.get_global_linear_id();
        if (i >= total)
          return; // Only whole subgroups return.
        const auto group = item.get_sub_group();
        const float v = x[i];
        float maximum = sycl::fabs(v), sum = v;
        for (int offset = 16; offset; offset >>= 1) {
          maximum = sycl::fmax(
              maximum, sycl::permute_group_by_xor(group, maximum, offset));
          sum += sycl::permute_group_by_xor(group, sum, offset);
        }
        const float d = sycl::ext::intel::math::fdiv_rn(maximum, 127.f);
        const int q =
            maximum == 0
                ? 0
                : int(sycl::round(sycl::ext::intel::math::fdiv_rn(v, d)));
        out[i / 32].qs[i % 32] = int8_t(q);
        if (!(i % 32))
          out[i / 32].ds = {{f16_from_f32(d)}, {f16_from_f32(sum)}};
      });
}
bool native_mmvq_supported(int type) noexcept {
  return type == 2 || type == 6 || type == 8 || type == 11 || type == 12 ||
         type == 13 || type == 14 || type == 20 || type == 23 || type == 42;
}
size_t native_mmvq_weight_bytes(int type, int n_in, int n_out) {
  const auto f = format(type);
  validate_shape(n_in, 1, f.width);
  if (n_out <= 0)
    throw std::invalid_argument("SYCL MMVQ requires positive output width");
  return checked_count(checked_count(n_out, n_in / f.width), f.bytes);
}
void native_mmvq(int type, const void *w, const void *x, float *y, int n_in,
                 int n_out, int ncols, void *stream) {
  const auto wb = native_mmvq_weight_bytes(type, n_in, n_out);
  const auto xb = native_q8_1_bytes(n_in, ncols);
  validate_stream(stream);
  validate_spans({{y, checked_count(checked_count(n_out, ncols), 4)}},
                 {{w, wb}, {x, xb}});
  switch (type) {
  case 2:
    dispatch<SmallTraits<Q40Block, 4>>(w, x, y, n_in, n_out, ncols, stream);
    break;
  case 6:
    dispatch<SmallTraits<Q50Block, 4>>(w, x, y, n_in, n_out, ncols, stream);
    break;
  case 8:
    dispatch<SmallTraits<Q80Block, 8>>(w, x, y, n_in, n_out, ncols, stream);
    break;
  case 11:
    if (ncols == 1 || multi_exact.load(std::memory_order_relaxed)) {
      native_q3_k_esimd(w, x, y, n_in, n_out, ncols, stream);
      break;
    }
    dispatch<Q3KTraits>(w, x, y, n_in, n_out, ncols, stream);
    break;
  case 12:
    if (!iq_old_kernels() &&
        (ncols == 1 || multi_exact.load(std::memory_order_relaxed))) {
      native_q4_k_esimd(w, x, y, n_in, n_out, ncols, stream);
      break;
    }
    dispatch<Q4KTraits>(w, x, y, n_in, n_out, ncols, stream);
    break;
  case 13:
    if (n_in == 2560 && n_out >= 4096 &&
        (ncols == 1 || multi_exact.load(std::memory_order_relaxed))) {
      native_q5_head_esimd(w, x, y, n_out, ncols, stream);
      break;
    }
    dispatch<Q5KTraits>(w, x, y, n_in, n_out, ncols, stream);
    break;
  case 14:
    dispatch<Q6KTraits>(w, x, y, n_in, n_out, ncols, stream);
    break;
  case 20:
    dispatch<SmallTraits<IQ4NLBlock, 4>>(w, x, y, n_in, n_out, ncols, stream);
    break;
  case 23:
    // Batched columns reuse decoded weights more efficiently in SPMD.
    if (!iq_old_kernels() && ncols == 1) {
      native_iq4_xs_esimd(w, x, y, n_in, n_out, ncols, stream);
      break;
    }
    dispatch<IQ4XSTraits>(w, x, y, n_in, n_out, ncols, stream);
    break;
  case 42:
    if (!iq_old_kernels() &&
        (ncols == 1 || multi_exact.load(std::memory_order_relaxed))) {
      native_q2_0_esimd(w, x, y, n_in, n_out, ncols, stream);
      break;
    }
    dispatch<Q20Traits>(w, x, y, n_in, n_out, ncols, stream);
    break;
  }
}
#define STRATA_SYCL_MMVQ(NAME, TYPE)                                           \
  void native_##NAME##_mmvq(const void *w, const void *x, float *y, int ni,    \
                            int no, int nc, void *s) {                         \
    native_mmvq(TYPE, w, x, y, ni, no, nc, s);                                 \
  }                                                                            \
  void native_##NAME##_f32(const void *w, const float *x, void *scratch,       \
                           float *y, int ni, int no, int nc, void *s) {        \
    composition(TYPE, w, x, scratch, y, ni, no, nc, s);                        \
  }
STRATA_SYCL_MMVQ(q4_0, 2)
STRATA_SYCL_MMVQ(q5_0, 6)
STRATA_SYCL_MMVQ(q8_0, 8)
STRATA_SYCL_MMVQ(q3_k, 11)
STRATA_SYCL_MMVQ(q4_k, 12)
STRATA_SYCL_MMVQ(q5_k, 13)
STRATA_SYCL_MMVQ(q6_k, 14)
STRATA_SYCL_MMVQ(iq4_nl, 20)
STRATA_SYCL_MMVQ(iq4_xs, 23)
STRATA_SYCL_MMVQ(q2_0, 42)
#undef STRATA_SYCL_MMVQ
bool native_expert_supported(int gu, int down, int64_t n, int64_t f) noexcept {
  if (!native_mmvq_supported(gu) || !native_mmvq_supported(down) || n <= 0 ||
      f <= 0 || n > INT_MAX || f > INT_MAX / 2)
    return false;
  return n % format(gu).width == 0 && f % format(down).width == 0 &&
         n % 256 == 0;
}
NativeExpertLayout native_expert_layout(int gu, int down, int64_t n,
                                        int64_t f) {
  if (!native_expert_supported(gu, down, n, f))
    throw std::invalid_argument("unsupported SYCL native expert layout");
  NativeExpertLayout L;
  L.gu_type = gu;
  L.d_type = down;
  L.n_embd = n;
  L.n_ff = f;
  L.gu_row = native_mmvq_weight_bytes(gu, n, 1);
  L.d_row = native_mmvq_weight_bytes(down, f, 1);
  L.up_off = checked_count(f, L.gu_row);
  L.down_off = 2 * L.up_off;
  const size_t downbytes = checked_count(n, L.d_row);
  if (L.up_off > SIZE_MAX / 2 || downbytes > SIZE_MAX - L.down_off)
    throw std::invalid_argument("native expert blob overflow");
  L.bytes = L.down_off + downbytes;
  return L;
}
size_t native_expert_scratch_bytes(int64_t cap, int64_t f) {
  if (cap <= 0 || cap > INT_MAX || f <= 0 || f > INT_MAX || f % 32 ||
      checked_count(cap, f) > INT_MAX)
    throw std::invalid_argument("invalid native expert scratch dimensions");
  const size_t bytes = checked_count(cap, f) * 4;
  return 3 * ((bytes + 255) & ~size_t(255)) +
         ((checked_count(cap, f / 32) * 36 + 255) & ~size_t(255));
}
void native_expert_grouped(const NativeExpertLayout &L,
                           const unsigned long long *ptr, const int32_t *start,
                           const int32_t *count, const int32_t *dst,
                           const int32_t *tok, int64_t groups, int64_t entries,
                           const void *x, void *scratch, float *out,
                           void *stream) {
  if (groups <= 0 || groups > INT_MAX || entries <= 0 || entries > INT_MAX ||
      groups > entries)
    throw std::invalid_argument("invalid native expert group capacities");
  const auto expected =
      native_expert_layout(L.gu_type, L.d_type, L.n_embd, L.n_ff);
  if (expected.gu_row != L.gu_row || expected.d_row != L.d_row ||
      expected.up_off != L.up_off || expected.down_off != L.down_off ||
      expected.bytes != L.bytes)
    throw std::invalid_argument("inconsistent native expert blob layout");
  const size_t bytes = native_expert_scratch_bytes(entries, L.n_ff),
               plane =
                   (checked_count(entries, L.n_ff) * 4 + 255) & ~size_t(255);
  validate_spans(
      {{scratch, bytes}, {out, checked_count(entries, L.n_embd) * 4}},
      {{ptr, size_t(groups) * 8},
       {start, size_t(groups + 1) * 4},
       {count, 4},
       {dst, size_t(entries) * 4},
       {tok, size_t(entries) * 4},
       {x, native_q8_1_bytes(L.n_embd)}});
  if (reinterpret_cast<uintptr_t>(ptr) % 8)
    throw std::invalid_argument(
        "native expert pointers require 8-byte alignment");
  // Blob and activation row extents are owned by the caller; they are reached
  // through device metadata, never read back or reallocated here.
  auto &q = queue_for(stream);
  auto gate = static_cast<float *>(scratch),
       up = reinterpret_cast<float *>(static_cast<uint8_t *>(scratch) + plane),
       h = reinterpret_cast<float *>(static_cast<uint8_t *>(scratch) +
                                     2 * plane);
  auto hq =
      reinterpret_cast<Q81Block *>(static_cast<uint8_t *>(scratch) + 3 * plane);
  q.memset(scratch, 0, 3 * plane);
  const int tile = iq_old_kernels() ? 1 : 8;
  grouped_dispatch<false>(
      L.gu_type, L, ptr, start, count, dst, tok, int(groups), int(entries),
      static_cast<const Q81Block *>(x), gate, up, nullptr, tile, q);
  q.parallel_for(sycl::range<1>(size_t(entries) * L.n_ff), [=](sycl::id<1> i) {
    const float v = gate[i];
    h[i] = (v / (1.f + sycl::exp(-v))) * up[i];
  });
  native_quantize_q8_1(h, hq, int(entries * L.n_ff), 1, &q);
  grouped_dispatch<true>(L.d_type, L, ptr, start, count, dst, tok, int(groups),
                         int(entries), hq, nullptr, nullptr, out, tile, q);
  if (!stream)
    q.wait_and_throw();
}
} // namespace strata::kernels
