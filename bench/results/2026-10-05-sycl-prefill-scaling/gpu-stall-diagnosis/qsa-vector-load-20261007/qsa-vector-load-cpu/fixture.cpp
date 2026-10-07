#include <sycl/sycl.hpp>
#include "strata/kernels/qsa_decode_attn.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
using strata::kernels::QsaAttnPools;
constexpr int HD = 256, KV_Q8_GROUP = 64;

namespace original {
inline void load8_f16(const QsaAttnPools &p, bool value, long long row,
                               int d0, float *out) {
    const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
    const sycl::uint4 raw = *reinterpret_cast<const sycl::uint4 *>(base);
    const sycl::half2 *h2 = reinterpret_cast<const sycl::half2 *>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const sycl::float2 f =
            h2[j].template convert<float, sycl::rounding_mode::automatic>();
        out[2 * j] = f.x();
        out[2 * j + 1] = f.y();
    }
}
inline void load8_q8(const QsaAttnPools &p, bool value, long long row,
                              int d0, float *out) {
    const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
    const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
    const float sc = sycl::vec<sycl::half, 1>(
                         sycl::bit_cast<sycl::half, unsigned short>(sbits))
                         .convert<float, sycl::rounding_mode::automatic>()[0];
    const sycl::uint2 raw = *reinterpret_cast<const sycl::uint2 *>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}

}

namespace candidate {
inline void load8_f16(const QsaAttnPools &p, bool value, long long row,
                               int d0, float *out) {
    const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
    sycl::vec<uint16_t, 8> bits;
    bits.load(0, base);
    const sycl::half8 halves = bits.as<sycl::half8>();
    const sycl::half2 h2[] = {halves.template swizzle<0, 1>(), halves.template swizzle<2, 3>(),
                             halves.template swizzle<4, 5>(), halves.template swizzle<6, 7>()};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const sycl::float2 f =
            h2[j].template convert<float, sycl::rounding_mode::automatic>();
        out[2 * j] = f.x();
        out[2 * j + 1] = f.y();
    }
}
inline void load8_q8(const QsaAttnPools &p, bool value, long long row,
                              int d0, float *out) {
    const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
    const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
    const float sc = sycl::vec<sycl::half, 1>(
                         sycl::bit_cast<sycl::half, unsigned short>(sbits))
                         .convert<float, sycl::rounding_mode::automatic>()[0];
    sycl::vec<int8_t, 8> c;
    c.load(0, codes);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}

}

float ieee_half(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 31, mantissa = h & 1023;
    if (exponent == 31) return std::bit_cast<float>(sign | 0x7f800000 | (mantissa << 13));
    if (exponent) return std::bit_cast<float>(sign | ((exponent + 112) << 23) | (mantissa << 13));
    if (!mantissa) return std::bit_cast<float>(sign);
    unsigned bit = 0;
    for (unsigned m = mantissa; m > 1; m >>= 1) ++bit;
    return std::bit_cast<float>(sign | ((103 + bit) << 23) | ((mantissa - (1u << bit)) << (23 - bit)));
}
bool exact_or_nan(float value, float reference) {
    return std::isnan(reference) ? std::isnan(value)
           : std::bit_cast<uint32_t>(value) == std::bit_cast<uint32_t>(reference);
}
int main() {
    alignas(64) std::array<uint16_t, 3 * HD> k16{}, v16{};
    alignas(64) std::array<int8_t, 3 * HD> k8{}, v8{};
    std::array<uint16_t, 3 * HD / KV_Q8_GROUP> ks{}, vs{};
    QsaAttnPools pools;
    pools.k_pool=k16.data(); pools.v_pool=v16.data();
    pools.k_q=k8.data(); pools.v_q=v8.data(); pools.k_scale=ks.data(); pools.v_scale=vs.data();
    uint64_t mismatches=0, comparisons=0, finite_bits=0, nan_classes=0;
    const int8_t codes[8]={-128, -127, -1, 0, 1, 2, 126, 127};
    for (unsigned pattern=0; pattern<65536; ++pattern) {
        const int row=pattern%3, d0=((pattern/3)%32)*8;
        for (int j=0; j<8; ++j) {
            k16[row*HD+d0+j]=uint16_t(pattern + 8191*j);
            v16[row*HD+d0+j]=uint16_t(pattern + 8191*j) ^ 0x8000;
            k8[row*HD+d0+j]=codes[(j+pattern)%8];
            v8[row*HD+d0+j]=codes[(7-j+pattern)%8];
        }
        ks[row*(HD/KV_Q8_GROUP)+d0/KV_Q8_GROUP]=uint16_t(pattern);
        vs[row*(HD/KV_Q8_GROUP)+d0/KV_Q8_GROUP]=uint16_t(pattern)^0x8000;
        for (bool value : {false,true}) {
            float old16[8], new16[8], old8[8], new8[8];
            original::load8_f16(pools,value,row,d0,old16);
            candidate::load8_f16(pools,value,row,d0,new16);
            original::load8_q8(pools,value,row,d0,old8);
            candidate::load8_q8(pools,value,row,d0,new8);
            for (int j=0; j<8; ++j) {
                const float reference16=ieee_half((value?v16:k16)[row*HD+d0+j]);
                const float reference8=float((value?v8:k8)[row*HD+d0+j]) *
                    ieee_half((value?vs:ks)[row*(HD/KV_Q8_GROUP)+d0/KV_Q8_GROUP]);
                mismatches += !exact_or_nan(old16[j],reference16);
                mismatches += !exact_or_nan(new16[j],reference16);
                mismatches += !exact_or_nan(old8[j],reference8);
                mismatches += !exact_or_nan(new8[j],reference8);
                mismatches += !exact_or_nan(old16[j],new16[j]);
                mismatches += !exact_or_nan(old8[j],new8[j]);
                comparisons += 6;
                finite_bits += !std::isnan(reference16) + !std::isnan(reference8);
                nan_classes += std::isnan(reference16) + std::isnan(reference8);
            }
        }
    }
    std::printf("{\"patterns\":65536,\"comparisons\":%llu,\"non_nan_bit_checks\":%llu,"
                "\"nan_class_checks\":%llu,\"mismatches\":%llu}\n",
                (unsigned long long)comparisons,(unsigned long long)finite_bits,
                (unsigned long long)nan_classes,(unsigned long long)mismatches);
    return mismatches ? 1 : 0;
}
