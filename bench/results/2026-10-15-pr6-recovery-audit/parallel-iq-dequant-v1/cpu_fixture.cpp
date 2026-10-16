// Standalone host oracle. Build/run only by Main after both live benchmarks close.
#include "fixture_common.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace {
uint64_t blocks = 0, elements = 0, nan_bits = 0;
void require(bool ok, const char *msg) { if (!ok) throw std::runtime_error(msg); }
void check(int type, unsigned seq, uint16_t scale, unsigned offset) {
  const auto f = strata::sycl_backend::block_format(type);
  std::vector<uint8_t> src(size_t(f.bytes) + offset + 16, 0xa5);
  uint8_t *p = src.data() + offset;
  fixture::make_block(type, p, seq, scale);
  const auto original = src;
  std::vector<float> ref(size_t(f.width) + 2, 123456.f);
  std::vector<uint16_t> dst(size_t(f.width) + 2, 0xa55a);
  strata::sycl_backend::decode_block(type, p, ref.data() + 1);
  for (unsigned j = 0; j < unsigned(f.width); ++j) {
    const float v = fixture::element(type, p, j);
    const uint16_t a = strata::kernels::f16_from_f32(ref[1+j]);
    const uint16_t b = strata::kernels::f16_from_f32(v);
    dst[1+j] = b;
    if (a != b) {
      std::cerr << "type=" << type << " seq=" << seq << " scale=" << scale
                << " j=" << j << " ref=" << a << " got=" << b << '\n';
      throw std::runtime_error("half bits differ (NaNs included)");
    }
    // Also catch FP32 reassociation differences that currently round alike.
    uint32_t x, y;
    std::memcpy(&x, &ref[1+j], 4); std::memcpy(&y, &v, 4);
    const bool nan = (x & 0x7fffffff) > 0x7f800000;
    require(nan || x == y, "finite/Inf FP32 bits differ");
    nan_bits += nan;
    ++elements;
  }
  require(src == original, "source changed");
  require(dst.front() == 0xa55a && dst.back() == 0xa55a, "half guards");
  require(ref.front() == 123456.f && ref.back() == 123456.f, "oracle guards");
  ++blocks;
}
void layout(int type, size_t rows, size_t width) {
  const auto f = strata::sycl_backend::block_format(type);
  const size_t rowbytes = width / f.width * f.bytes;
  std::vector<uint8_t> g(rows * rowbytes), u(rows * rowbytes);
  for (size_t b = 0; b < rows * width / f.width; ++b) {
    fixture::make_block(type, g.data() + b*f.bytes, unsigned(b), 0x3a01);
    fixture::make_block(type, u.data() + b*f.bytes, unsigned(b+337), 0xb901);
  }
  // Independent oracle concatenates complete decoded gate/up rows.
  std::vector<uint16_t> expected;
  for (size_t r = 0; r < rows; ++r) {
    for (auto *src : {g.data(), u.data()}) {
      auto row = fixture::oracle(type, src+r*rowbytes, width);
      expected.insert(expected.end(), row.begin(), row.end());
    }
  }
  std::vector<uint16_t> got(expected.size()+2, 0xa55a);
  for (size_t i = 0; i < expected.size(); ++i) {
    const size_t row = i / width, col = i % width;
    const uint8_t *base = row % 2 ? u.data() : g.data();
    const uint8_t *p = base + row/2*rowbytes + col/f.width*f.bytes;
    got[1+i] = strata::kernels::f16_from_f32(
        fixture::element(type, p, unsigned(col%f.width)));
  }
  require(std::equal(expected.begin(), expected.end(), got.begin()+1), "GU layout");
  require(got.front()==0xa55a && got.back()==0xa55a, "GU guards");
}
}
int main() {
  try {
    for (int type : fixture::types) {
      // All grid indices at every cell, signs including high bit, every stored
      // scale, all four host byte offsets. Edge scales cross with all indices.
      for (unsigned seq = 0; seq < 1024; ++seq)
        for (uint16_t scale : fixture::edge_scales)
          check(type, seq, scale, seq % 4);
      // Every possible fp16 d bit pattern, including all payloads/signs of NaN.
      for (unsigned scale = 0; scale < 65536; ++scale)
        check(type, scale * 37u, uint16_t(scale), scale % 4);
      for (size_t rows : {1u, 2u, 3u})
        for (size_t width : {256u, 512u, 2560u}) layout(type, rows, width);
      std::cout << "type=" << type << " complete\n";
    }
    std::cout << "PASS blocks=" << blocks << " elements=" << elements
              << " nan_half_comparisons=" << nan_bits << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n'; return 1;
  }
}
