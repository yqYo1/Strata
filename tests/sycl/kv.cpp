#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;
template <class T> struct Buffer {
  sycl_backend::Allocation mem;
  Buffer(size_t n, bool host = false)
      : mem(runtime, n * sizeof(T),
            host ? sycl_backend::MemoryKind::Host
                 : sycl_backend::MemoryKind::Device) {}
  T *data() { return mem.as<T>(); }
  void put(const std::vector<T> &v) {
    runtime->wait(
        runtime->compute().memcpy(data(), v.data(), v.size() * sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> v(mem.size() / sizeof(T));
    runtime->wait(runtime->compute().memcpy(v.data(), data(), mem.size()));
    return v;
  }
};
void check(bool ok, const char *msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
uint16_t half(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
float decode(uint16_t x) { return float(std::bit_cast<_Float16>(x)); }
struct Reference {
  std::vector<uint8_t> bytes;
  std::vector<uint16_t> scales, values;
};
Reference encode(const float *x, int dim, int fmt) {
  Reference r;
  r.values.resize(dim);
  if (fmt == kKvF16) {
    r.bytes.resize(dim * 2);
    for (int i = 0; i < dim; ++i) {
      const uint16_t b = half(x[i]);
      r.values[i] = b;
      std::memcpy(r.bytes.data() + i * 2, &b, 2);
    }
  } else if (fmt == kKvInt8) {
    r.bytes.resize(dim);
    r.scales.resize(dim / 64);
    for (int g = 0; g < dim / 64; ++g) {
      float a = 0;
      for (int j = 0; j < 64; ++j)
        a = std::max(a, std::abs(x[g * 64 + j]));
      const uint16_t b = half(a / 127.f);
      r.scales[g] = b;
      const float s = decode(b);
      for (int j = 0; j < 64; ++j) {
        const int c = s > 0 ? std::clamp(int(std::nearbyint(x[g * 64 + j] / s)),
                                         -127, 127)
                            : 0;
        r.bytes[g * 64 + j] = uint8_t(int8_t(c));
        r.values[g * 64 + j] = half(float(c) * s);
      }
    }
  } else {
    r.bytes.resize(dim / 32 * 18);
    for (int g = 0; g < dim / 32; ++g) {
      float m = x[g * 32];
      for (int j = 1; j < 32; ++j) {
        float v = x[g * 32 + j];
        if (std::abs(v) > std::abs(m) || (std::abs(v) == std::abs(m) && v > m))
          m = v;
      }
      float d = m / (-8.f), inv = d != 0 ? 1.f / d : 0;
      uint16_t b = half(d);
      std::memcpy(r.bytes.data() + g * 18, &b, 2);
      for (int j = 0; j < 16; ++j) {
        int lo = std::clamp(int(x[g * 32 + j] * inv + 8.5f), 0, 15),
            hi = std::clamp(int(x[g * 32 + j + 16] * inv + 8.5f), 0, 15);
        r.bytes[g * 18 + 2 + j] = uint8_t(lo | (hi << 4));
        r.values[g * 32 + j] = half(float(lo - 8) * decode(b));
        r.values[g * 32 + j + 16] = half(float(hi - 8) * decode(b));
      }
    }
  }
  return r;
}
void storage(int fmt, int page_size, int dim) {
  constexpr int cells = 13, heads = 2, guard = 32;
  const int pages = (cells + page_size - 1) / page_size, slots = pages;
  const int rowbytes = fmt == kKvF16    ? dim * 2
                       : fmt == kKvInt8 ? dim
                                        : dim / 32 * 18;
  const int poolbytes = slots * heads * page_size * rowbytes,
            scalecount = slots * heads * page_size * (dim / 64);
  QsaShapes s = qsa_real_shapes();
  s.n_head_kv = heads;
  s.head_dim = dim;
  s.page_size = page_size;
  std::vector<int32_t> table(pages);
  for (int b = 0; b < pages; ++b)
    table[b] = pages - 1 - b;
  // An absent last page exercises the authoritative host copy without a VRAM
  // write.
  table.back() = -1;
  std::vector<float> keys(cells * heads * dim), vals(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    keys[i] = float(std::sin(i * .11) * (.01 + (i % 11)));
    vals[i] = float(std::cos(i * .073) * (.02 + (i % 17)));
  }
  // Zero groups, opposite-sign maximum ties and half-scale underflow.
  for (int j = 0; j < 32; ++j)
    keys[j] = 0;
  if (dim >= 256) {
    for (int j = 0; j < 64; ++j)
      keys[128 + j] = float(j - 32) + .5f;
    keys[128] = 127.f;
  }
  keys[32] = 8;
  keys[33] = -8;
  for (int j = 64; j < std::min(dim, 128); ++j)
    keys[j] = 1e-10f;
  Buffer<uint8_t> kp(poolbytes + guard), vp(poolbytes + guard),
      hk(poolbytes + guard, true), hv(poolbytes + guard, true),
      sk(poolbytes + guard), sv(poolbytes + guard);
  Buffer<uint16_t> ks(scalecount + guard), vs(scalecount + guard),
      hks(scalecount + guard, true), hvs(scalecount + guard, true);
  kp.put(std::vector<uint8_t>(poolbytes + guard, 0xa5));
  vp.put(std::vector<uint8_t>(poolbytes + guard, 0xa5));
  hk.put(std::vector<uint8_t>(poolbytes + guard, 0xa5));
  hv.put(std::vector<uint8_t>(poolbytes + guard, 0xa5));
  sk.put(std::vector<uint8_t>(poolbytes + guard, 0xa5));
  sv.put(std::vector<uint8_t>(poolbytes + guard, 0xa5));
  ks.put(std::vector<uint16_t>(scalecount + guard, 0xa5a5));
  vs.put(std::vector<uint16_t>(scalecount + guard, 0xa5a5));
  hks.put(std::vector<uint16_t>(scalecount + guard, 0xa5a5));
  hvs.put(std::vector<uint16_t>(scalecount + guard, 0xa5a5));
  Buffer<float> x(keys.size()), y(vals.size());
  x.put(keys);
  y.put(vals);
  Buffer<int32_t> dt(table.size()), step(kStepCount);
  dt.put(table);
  KvHostPools host, stage;
  if (fmt == kKvF16) {
    host.k_pool = reinterpret_cast<uint16_t *>(hk.data());
    host.v_pool = reinterpret_cast<uint16_t *>(hv.data());
  } else if (fmt == kKvInt8) {
    host.k_q = reinterpret_cast<int8_t *>(hk.data());
    host.v_q = reinterpret_cast<int8_t *>(hv.data());
    host.k_scale = hks.data();
    host.v_scale = hvs.data();
  } else {
    host.k_q4 = hk.data();
    host.v_q4 = hv.data();
    stage.k_q4 = sk.data();
    stage.v_q4 = sv.data();
  }
  for (int t = 0; t < cells; ++t) {
    std::vector<int32_t> st(kStepCount);
    qsa_step_fill(st.data(), t, s);
    step.put(st);
    check(st[kStepNKv] == t + 1 && st[kStepNBid] == (t + 1) / 4 &&
              st[kStepWidth] == t + 1,
          "step counts");
    const float *a = x.data() + t * heads * dim;
    const float *b = y.data() + t * heads * dim;
    if (fmt == kKvF16)
      kv_append_step(reinterpret_cast<uint16_t *>(kp.data()),
                     reinterpret_cast<uint16_t *>(vp.data()), dt.data(),
                     step.data(), a, b, s, &runtime->compute(), &host);
    if (fmt == kKvInt8)
      kv_append_q8_step(reinterpret_cast<int8_t *>(kp.data()),
                        reinterpret_cast<int8_t *>(vp.data()), ks.data(),
                        vs.data(), dt.data(), step.data(), a, b, s,
                        &runtime->compute(), &host);
    if (fmt == kKvQ4)
      kv_append_q4_step(kp.data(), vp.data(), dt.data(), step.data(), a, b, s,
                        &runtime->compute(), &host);
  }
  const auto kb = kp.get(), vb = vp.get(), hkb = hk.get(), hvb = hv.get();
  const auto ksb = ks.get(), vsb = vs.get(), hksb = hks.get(), hvsb = hvs.get();
  std::vector<uint8_t> ek(poolbytes + guard, 0xa5), ev = ek, ehk = ek, ehv = ek;
  std::vector<uint16_t> eks(scalecount + guard, 0xa5a5), evs = eks, ehks = eks,
                                                         ehvs = eks;
  std::vector<uint16_t> decodedk(keys.size()), decodedv(vals.size());
  for (int t = 0; t < cells; ++t)
    for (int h = 0; h < heads; ++h) {
      const auto kr = encode(keys.data() + (t * heads + h) * dim, dim, fmt),
                 vr = encode(vals.data() + (t * heads + h) * dim, dim, fmt);
      const int row = ((t / page_size) * heads + h) * page_size + t % page_size;
      std::copy(kr.bytes.begin(), kr.bytes.end(), ehk.begin() + row * rowbytes);
      std::copy(vr.bytes.begin(), vr.bytes.end(), ehv.begin() + row * rowbytes);
      std::copy(kr.scales.begin(), kr.scales.end(),
                ehks.begin() + row * (dim / 64));
      std::copy(vr.scales.begin(), vr.scales.end(),
                ehvs.begin() + row * (dim / 64));
      if (table[t / page_size] >= 0) {
        int pr = (table[t / page_size] * heads + h) * page_size + t % page_size;
        std::copy(kr.bytes.begin(), kr.bytes.end(), ek.begin() + pr * rowbytes);
        std::copy(vr.bytes.begin(), vr.bytes.end(), ev.begin() + pr * rowbytes);
        std::copy(kr.scales.begin(), kr.scales.end(),
                  eks.begin() + pr * (dim / 64));
        std::copy(vr.scales.begin(), vr.scales.end(),
                  evs.begin() + pr * (dim / 64));
      }
      std::copy(kr.values.begin(), kr.values.end(),
                decodedk.begin() + (t * heads + h) * dim);
      std::copy(vr.values.begin(), vr.values.end(),
                decodedv.begin() + (t * heads + h) * dim);
    }
  check(kb == ek && vb == ev, "resident encoded bytes/canaries");
  check(hkb == ehk && hvb == ehv, "host encoded bytes/canaries");
  if (fmt == kKvInt8) {
    check(ksb == eks && vsb == evs, "resident scales");
    check(hksb == ehks && hvsb == ehvs, "host scales");
  }
  if (fmt == kKvQ4) {
    kv_append_q4(kp.data(), vp.data(), dt.data(), 0, cells, x.data(), y.data(),
                 s, &runtime->compute(), &host, &stage);
    check(kp.get() == kb && vp.get() == vb, "batch/step equality");
    check(sk.get() == ehk && sv.get() == ehv, "prompt stage bytes");
  }
  // Gather every cell, including the absent page, and leave excess capacity
  // untouched.
  std::vector<int32_t> ids = {12, 0, 7, 3, 11};
  const int cap = 9;
  ids.resize(cap, 0);
  Buffer<int32_t> di(cap);
  di.put(ids);
  Buffer<uint16_t> outk(cap * heads * dim + guard),
      outv(cap * heads * dim + guard);
  outk.put(std::vector<uint16_t>(cap * heads * dim + guard, 0xa5a5));
  outv.put(std::vector<uint16_t>(cap * heads * dim + guard, 0xa5a5));
  std::vector<int32_t> st = {12, 13, 3, 5};
  step.put(st);
  auto gather = [&] {
    if (fmt == kKvF16)
      kv_gather_step(reinterpret_cast<uint16_t *>(kp.data()),
                     reinterpret_cast<uint16_t *>(vp.data()), dt.data(),
                     di.data(), step.data(), cap, s, outk.data(), outv.data(),
                     &runtime->compute());
    if (fmt == kKvInt8)
      kv_gather_q8_step(reinterpret_cast<int8_t *>(kp.data()),
                        reinterpret_cast<int8_t *>(vp.data()), ks.data(),
                        vs.data(), dt.data(), di.data(), step.data(), cap, s,
                        outk.data(), outv.data(), &runtime->compute());
    if (fmt == kKvQ4)
      kv_gather_q4_step(kp.data(), vp.data(), dt.data(), di.data(), step.data(),
                        cap, s, outk.data(), outv.data(), &runtime->compute());
  };
  gather();
  const auto ok = outk.get(), ov = outv.get();
  for (int i = 0; i < cap * heads * dim + guard; ++i) {
    int cell = i / (heads * dim);
    uint16_t kr = 0xa5a5, vr = 0xa5a5;
    if (cell < 5) {
      const int t = ids[cell];
      int d = i % (heads * dim);
      kr = table[t / page_size] < 0 ? 0x7e00 : decodedk[t * heads * dim + d];
      vr = table[t / page_size] < 0 ? 0x7e00 : decodedv[t * heads * dim + d];
    }
    check(ok[i] == kr && ov[i] == vr, "gather decoded values/guards");
  }
  st[kStepWidth] = 0;
  step.put(st);
  gather();
  check(outk.get() == ok && outv.get() == ov, "empty gather changes output");
  std::cout << "KV format=" << fmt << " page=" << page_size << " dim=" << dim
            << " passed\n";
}
void hadamard() {
  constexpr int rows = 7;
  std::vector<float> x(rows * 256 + 16, 1234.f), ref = x;
  for (int i = 0; i < rows * 256; ++i)
    x[i] = float(std::sin(i * .017));
  for (int r = 0; r < rows; ++r) {
    for (int d = 0; d < 256; ++d)
      ref[r * 256 + d] = x[r * 256 + d] / 16.f;
    for (int h = 1; h < 256; h *= 2)
      for (int i = 0; i < 256; i += h * 2)
        for (int j = 0; j < h; ++j) {
          float a = ref[r * 256 + i + j], b = ref[r * 256 + i + j + h];
          ref[r * 256 + i + j] = a + b;
          ref[r * 256 + i + j + h] = a - b;
        }
  }
  Buffer<float> b(x.size()), separate(x.size());
  separate.put(std::vector<float>(x.size(), 1234.f));
  b.put(x);
  fwht256_cuda(b.data(), separate.data(), rows, &runtime->compute());
  check(separate.get() == ref, "FWHT separate output");
  fwht256_inplace_cuda(b.data(), rows, &runtime->compute());
  check(b.get() == ref, "FWHT exact butterfly reference");
  fwht256_inplace_cuda(b.data(), rows, &runtime->compute());
  auto y = b.get();
  for (size_t i = 0; i < x.size(); ++i)
    check(std::abs(x[i] - y[i]) < 4e-7, "FWHT inverse/guards");
}
} // namespace
int main() {
  try {
    std::fesetround(FE_TONEAREST);
    runtime = sycl_backend::runtime_for();
    hadamard();
    for (int fmt : {kKvF16, kKvInt8, kKvQ4})
      for (int page : {1, 4, 16}) {
        storage(fmt, page, 256);
        if (fmt != kKvQ4)
          storage(fmt, page, fmt == kKvF16 ? 68 : 128);
      }
    Buffer<float> x(512), y(512);
    Buffer<uint16_t> k(512), v(512);
    Buffer<int32_t> table(1), ids(1);
    table.put({0});
    ids.put({0});
    x.put(std::vector<float>(512, 1.25f));
    y.put(std::vector<float>(512, -.75f));
    auto shape = qsa_real_shapes();
    shape.page_size = 1;
    kv_append(k.data(), v.data(), table.data(), 0, x.data(), y.data(), shape,
              nullptr);
    Buffer<uint16_t> ko(512), vo(512);
    kv_gather(k.data(), v.data(), table.data(), ids.data(), 1, shape, ko.data(),
              vo.data(), nullptr);
    check(ko.get() == std::vector<uint16_t>(512, half(1.25f)) &&
              vo.get() == std::vector<uint16_t>(512, half(-.75f)),
          "scalar KV/null stream");
    auto reject = [](auto f) {
      bool caught = false;
      try {
        f();
      } catch (const std::invalid_argument &) {
        caught = true;
      }
      check(caught, "invalid KV accepted");
    };
    reject([&] {
      kv_append(k.data(), v.data(), table.data(), -1, x.data(), y.data(), shape,
                nullptr);
    });
    reject([&] {
      kv_append_step(k.data(), v.data(), table.data(), nullptr, x.data(),
                     y.data(), shape, nullptr);
    });
    reject([&] { fwht256_cuda(x.data(), x.data() + 1, 1, nullptr); });
    reject([&] {
      int32_t step[4];
      qsa_step_fill(step, INT32_MAX, shape);
    });
    shape.head_dim = 65;
    reject([&] {
      kv_gather(k.data(), v.data(), table.data(), ids.data(), 1, shape,
                ko.data(), vo.data(), nullptr);
    });
    runtime->wait();
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
