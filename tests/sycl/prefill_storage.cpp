#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl/runtime.hpp"
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata;
namespace p = strata::prefill;
std::shared_ptr<sycl_backend::Runtime> rt;
template <class T> struct Buffer {
  sycl_backend::Allocation a;
  size_t n;
  Buffer(size_t n,
         sycl_backend::MemoryKind kind = sycl_backend::MemoryKind::Device)
      : a(rt, n * sizeof(T), kind), n(n) {}
  T *data() { return a.as<T>(); }
  void set(const std::vector<T> &v) {
    if (v.size() != n)
      throw std::runtime_error("size");
    rt->wait(rt->compute().memcpy(data(), v.data(), n * sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> v(n);
    rt->wait(rt->compute().memcpy(v.data(), data(), n * sizeof(T)));
    return v;
  }
};
void need(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
void kv(bool quant) {
  constexpr int T = 9, P = 4, physical = 2 * 2 * P * 256,
                logical = 3 * 2 * P * 256;
  std::vector<float> k(T * 512), v(T * 512);
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = std::sin(float(i) * .017f);
    v[i] = std::cos(float(i) * .023f) * .7f;
  }
  Buffer<float> K(k.size()), V(v.size());
  K.set(k);
  V.set(v);
  Buffer<int32_t> TABLE(3);
  TABLE.set({1, -1, 0});
  Buffer<uint16_t> KP(physical), VP(physical),
      HK(logical, sycl_backend::MemoryKind::Host),
      HV(logical, sycl_backend::MemoryKind::Host), SK(logical), SV(logical);
  Buffer<int8_t> KQ(physical), VQ(physical),
      HKQ(logical, sycl_backend::MemoryKind::Host),
      HVQ(logical, sycl_backend::MemoryKind::Host), SKQ(logical), SVQ(logical);
  Buffer<uint16_t> KS(physical / 64), VS(physical / 64), HKS(logical / 64),
      HVS(logical / 64), SKS(logical / 64), SVS(logical / 64);
  std::vector<uint16_t> pk(physical, 0x5a5a), pv = pk, hk(logical, 0x5a5a),
                                              hv = hk;
  std::vector<int8_t> qk(physical, 123), qv = qk, hqk(logical, 123), hqv = hqk;
  std::vector<uint16_t> sk(physical / 64, 0xffff),
      sv = sk, hsk(logical / 64, 0xffff), hsv = hsk;
  KP.set(pk);
  VP.set(pv);
  HK.set(hk);
  HV.set(hv);
  SK.set(hk);
  SV.set(hv);
  KQ.set(qk);
  VQ.set(qv);
  HKQ.set(hqk);
  HVQ.set(hqv);
  SKQ.set(hqk);
  SVQ.set(hqv);
  KS.set(sk);
  VS.set(sv);
  HKS.set(hsk);
  HVS.set(hsv);
  SKS.set(hsk);
  SVS.set(hsv);
  kernels::KvHostPools host{}, stage{};
  if (quant) {
    host.k_q = HKQ.data();
    host.v_q = HVQ.data();
    host.k_scale = HKS.data();
    host.v_scale = HVS.data();
    stage.k_q = SKQ.data();
    stage.v_q = SVQ.data();
    stage.k_scale = SKS.data();
    stage.v_scale = SVS.data();
  } else {
    host.k_pool = HK.data();
    host.v_pool = HV.data();
    stage.k_pool = SK.data();
    stage.v_pool = SV.data();
  }
  p::kv_append(K.data(), V.data(), T, 3, TABLE.data(), P,
               quant ? nullptr : KP.data(), quant ? nullptr : VP.data(),
               KQ.data(), VQ.data(), KS.data(), VS.data(), &rt->compute(),
               &host, &stage);
  for (int t = 0; t < T; ++t)
    for (int head = 0; head < 2; ++head)
      for (int group = 0; group < 4; ++group)
        for (int value = 0; value < 2; ++value) {
          const auto &x = value ? v : k;
          int pos = 3 + t, page = pos / P,
              phys = page == 0   ? 1
                     : page == 1 ? -1
                                 : 0;
          size_t lr = (page * 2 + head) * P + pos % P,
                 pr = (phys * 2 + head) * P + pos % P;
          float max = 0;
          for (int d = 0; d < 64; ++d)
            max = std::max(max,
                           std::abs(x[t * 512 + head * 256 + group * 64 + d]));
          auto sb = kernels::f16_from_f32(max / 127);
          float sf = kernels::f32_from_f16(sb);
          if (quant) {
            (value ? hsv : hsk)[lr * 4 + group] = sb;
            if (phys >= 0)
              (value ? sv : sk)[pr * 4 + group] = sb;
          }
          for (int d = 0; d < 64; ++d) {
            float f = x[t * 512 + head * 256 + group * 64 + d];
            size_t a = lr * 256 + group * 64 + d, b = pr * 256 + group * 64 + d;
            if (quant) {
              int8_t q = sf > 0 ? int8_t(std::clamp(int(std::nearbyint(f / sf)),
                                                    -127, 127))
                                : 0;
              (value ? hqv : hqk)[a] = q;
              if (phys >= 0)
                (value ? qv : qk)[b] = q;
            } else {
              auto bits = kernels::f16_from_f32(f);
              (value ? hv : hk)[a] = bits;
              if (phys >= 0)
                (value ? pv : pk)[b] = bits;
            }
          }
        }
  if (quant) {
    need(KQ.get() == qk && VQ.get() == qv && KS.get() == sk && VS.get() == sv,
         "INT8 physical KV");
    need(HKQ.get() == hqk && HVQ.get() == hqv && HKS.get() == hsk &&
             HVS.get() == hsv,
         "INT8 host KV");
    need(SKQ.get() == hqk && SVQ.get() == hqv && SKS.get() == hsk &&
             SVS.get() == hsv,
         "INT8 staging KV");
  } else {
    need(KP.get() == pk && VP.get() == pv, "FP16 physical KV");
    need(HK.get() == hk && HV.get() == hv && SK.get() == hk && SV.get() == hv,
         "FP16 host/staging KV");
  }
}
void rope() {
  constexpr int T = 3, H = 2, D = 256, LD = H * D + 7;
  std::vector<float> x(T * LD, 42), flat(T * H * D);
  std::vector<int32_t> pos(T * H);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      pos[t * H + h] = t + 31;
      for (int d = 0; d < D; ++d)
        x[t * LD + h * D + d] = flat[(t * H + h) * D + d] =
            std::sin(float((t * H + h) * D + d) * .04f);
    }
  Buffer<float> X(x.size()), F(flat.size());
  Buffer<int32_t> P(pos.size());
  P.set(pos);
  for (int mode = 0; mode < 3; ++mode) {
    kernels::RopeScaling sc;
    if (mode == 1) {
      sc.type = kernels::RopeScalingType::Linear;
      sc.factor = 4;
    }
    if (mode == 2) {
      sc.type = kernels::RopeScalingType::YaRN;
      sc.factor = 4;
      sc.orig_ctx = 4096;
      sc.ext_factor = 1;
    }
    X.set(x);
    F.set(flat);
    p::rope(X.data(), T, H, D, LD, 31, sc, &rt->compute());
    kernels::native_rope_apply(F.data(), F.data(), T * H, D, 64, sc, P.data(),
                               &rt->compute());
    auto a = X.get(), b = F.get();
    for (int t = 0; t < T; ++t) {
      for (int d = 0; d < H * D; ++d)
        need(a[t * LD + d] == b[t * H * D + d], "prompt RoPE parity");
      for (int d = H * D; d < LD; ++d)
        need(a[t * LD + d] == 42, "RoPE stride padding");
    }
  }
}
void blob() {
  constexpr size_t gu_codes = 1280 * 640, dc = 2560 * 160, sc = gu_codes + dc,
                   ds = sc + 1280 * 40 * 2, total = ds + 2560 * 10 * 2;
  std::vector<uint8_t> b(total);
  for (size_t i = 0; i < sc; ++i)
    b[i] = uint8_t(i * 17 + 31);
  for (size_t i = sc; i < total; i += 2) {
    auto u = kernels::f16_from_f32(float((i / 2) % 17 + 1) / 128);
    b[i] = uint8_t(u);
    b[i + 1] = uint8_t(u >> 8);
  }
  Buffer<uint8_t> B(total);
  B.set(b);
  Buffer<uint16_t> GU(1280 * 2560), DW(2560 * 640);
  for (bool half : {false, true}) {
    if (half)
      p::blob_dequant_f16(B.data(), GU.data(), DW.data(), &rt->compute());
    else
      p::blob_dequant(B.data(), GU.data(), DW.data(), &rt->compute());
    auto g = GU.get(), d = DW.get();
    for (int role = 0; role < 2; ++role) {
      const auto &o = role ? d : g;
      size_t width = role ? 640 : 2560, off = role ? gu_codes : 0,
             scaleoff = role ? ds : sc;
      for (size_t i = 0; i < o.size(); ++i) {
        size_t row = i / width, col = i % width,
               si = scaleoff + (row * (width / 64) + col / 64) * 2;
        float scale =
            kernels::f32_from_f16(uint16_t(b[si]) | (uint16_t(b[si + 1]) << 8));
        int code = (b[off + i / 4] >> (2 * (i % 4))) & 3;
        float v = (code - 1) * scale;
        uint16_t want = half ? kernels::f16_from_f32(v)
                             : uint16_t(std::bit_cast<uint32_t>(v) >> 16);
        need(o[i] == want, "canonical expert expansion");
      }
    }
  }
}
int main() {
  try {
    rt = sycl_backend::runtime_for();
    kv(false);
    kv(true);
    rope();
    blob();
    std::cout << "SYCL prompt storage: paged KV host/staging, scaled RoPE and "
                 "canonical experts passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
