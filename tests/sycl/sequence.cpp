#include "strata/kernels/verify_kernels.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata::kernels;
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
void require(bool v, const char *s) {
  if (!v)
    throw std::runtime_error(s);
}
template <class T> struct B {
  T *p{};
  size_t n;
  bool host;
  B(size_t N, bool h = false) : n(N), host(h) {
    check(h ? cudaMallocHost(&p, n * sizeof(T))
            : cudaMalloc(&p, n * sizeof(T)));
  }
  ~B() {
    if (host)
      cudaFreeHost(p);
    else
      cudaFree(p);
  }
  void put(const std::vector<T> &v) {
    require(v.size() == n, "upload size");
    check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> get() {
    std::vector<T> v(n);
    check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
int main() {
  try {
    cudaStream_t st{};
    check(cudaStreamCreate(&st));
    B<float> x(14), r(42), out(42);
    std::vector<float> xv(14), rv(42);
    for (int i = 0; i < 14; ++i)
      xv[i] = i * .5f;
    for (int i = 0; i < 42; ++i)
      rv[i] = float(i) * .25f;
    x.put(xv);
    r.put(rv);
    broadcast_streams(x.p, out.p, 7, 3, 2, st);
    auto got = out.get();
    for (int i = 0; i < 42; ++i)
      require(got[i] == xv[(i / 21) * 7 + i % 7], "broadcast");
    add_streams_broadcast(r.p, x.p, out.p, 7, 3, 2, st);
    got = out.get();
    for (int i = 0; i < 42; ++i)
      require(got[i] == rv[i] + xv[(i / 21) * 7 + i % 7], "add broadcast");
    B<int> idx(1), ids(6), slot(6), dst(6), cnt(1);
    ids.put({2, 1, 2, 3, 1, 3});
    ident_hits(ids.p, 6, slot.p, dst.p, cnt.p, st);
    require(slot.get() == ids.get() &&
                dst.get() == std::vector<int>({0, 1, 2, 3, 4, 5}) &&
                cnt.get()[0] == 6,
            "identity hits");
    B<float> copied(7);
    idx.put({1});
    copy_indexed(copied.p, r.p, 13, idx.p, 7, st);
    got = copied.get();
    for (int i = 0; i < 7; ++i)
      require(got[i] == rv[13 + i], "indexed copy");
    idx.put({-1});
    copy_indexed(copied.p, r.p, 13, idx.p, 7, st);
    require(copied.get() == got, "negative index");
    B<int> cells(3), steps(12), sel(24);
    cells.put({1, 9, 0});
    sel.put(std::vector<int>(24, -77));
    dense_steps(cells.p, 3, steps.p, st);
    window_ids(steps.p, 3, 5, sel.p, 8, st);
    auto rec = steps.get(), picked = sel.get();
    for (int row = 0; row < 3; ++row) {
      int end = std::vector<int>{2, 10, 1}[row], w = std::min(end, 5);
      require(rec[row * 4 + 3] == w && rec[row * 4 + 1] == end,
              "window record");
      for (int j = 0; j < 8; ++j)
        require(picked[row * 8 + j] == (j < w ? end - w + j : -77),
                "window ids");
    }
    B<unsigned> skip(1);
    skip.put({9});
    copy_or_zero_from_mapped(out.p, r.p, 42, skip.p, 9, st);
    for (float f : out.get())
      require(f == 0, "conditional zero");
    copy_i32_from_mapped_unless(slot.p, ids.p, 6, skip.p, 9, st);
    require(slot.get() == ids.get(), "conditional retained");
    slot.put(std::vector<int>(6, -8));
    skip.put({0});
    copy_i32_from_mapped_unless(slot.p, ids.p, 6, skip.p, 9, st);
    require(slot.get() == ids.get(), "conditional copy");
    copy_or_zero_from_mapped(out.p, r.p, 42, skip.p, 9, st);
    require(out.get() == rv, "conditional floats");
    B<int> table(4);
    table.put({13, 15, 17, 19});
    map_ids(slot.p, table.p, 6, st);
    require(slot.get() == std::vector<int>({17, 15, 17, 19, 15, 19}),
            "token map");
    B<unsigned char> rows(1 + 4 * 7), gather(1 + 3 * 7 + 3);
    std::vector<unsigned char> rb(rows.n);
    for (size_t i = 0; i < rb.size(); ++i)
      rb[i] = i + 10;
    rows.put(rb);
    gather.put(std::vector<unsigned char>(gather.n, 0xA5));
    B<int> take(3);
    take.put({2, 0, -1});
    gather_rows(rows.p + 1, 7, take.p, 3, gather.p + 1, st);
    auto gb = gather.get();
    require(gb[0] == 0xA5 && gb.back() == 0xA5, "gather guard");
    for (int i = 0; i < 21; ++i)
      require(gb[1 + i] ==
                  (i >= 14 ? 0 : rb[1 + (i / 7 == 0 ? 14 : 0) + i % 7]),
              "byte gather");
    B<unsigned char> blobs(64, true), fetched(80);
    for (int i = 0; i < 64; ++i)
      blobs.p[i] = i + 1;
    B<unsigned long long> ptr(4);
    ptr.put({(unsigned long long)(blobs.p + 32), (unsigned long long)blobs.p, 0,
             0});
    cnt.put({2});
    fetched.put(std::vector<unsigned char>(80, 0xA5));
    fetch_blobs(ptr.p, cnt.p, fetched.p, 16, 4, st);
    auto fb = fetched.get();
    for (int i = 0; i < 80; ++i)
      require(fb[i] == (i < 16   ? i + 33
                        : i < 32 ? i - 15
                                 : 0xA5),
              "bounded fetch");
    rebase_ptrs(ptr.p, cnt.p, fetched.p, 16, st);
    auto pp = ptr.get();
    require(pp[0] == (unsigned long long)fetched.p &&
                pp[1] == (unsigned long long)(fetched.p + 16) && pp[2] == 0,
            "rebased pointers");
    B<int> res(4), plan(74);
    B<unsigned long long> offsets(3);
    res.put({-1, 2, 0, 1});
    offsets.put({0, 16, 48});
    plan.put(std::vector<int>(74, -77));
    resident_plan(ids.p, 6, 3, res.p, 4, blobs.p, offsets.p, 16, plan.p, 8,
                  skip.p, 9, st);
    auto pl = plan.get();
    require(pl[0] == 3 && pl[1] == 6 && pl[2] == 0 && skip.get()[0] == 9,
            "resident counts");
    require(std::vector<int>(pl.begin() + 4, pl.begin() + 8) ==
                std::vector<int>({0, 2, 4, 6}),
            "resident starts");
    require(std::vector<int>(pl.begin() + 13, pl.begin() + 19) ==
                std::vector<int>({0, 2, 1, 4, 3, 5}),
            "resident destinations");
    require(std::vector<int>(pl.begin() + 21, pl.begin() + 27) ==
                std::vector<int>({0, 0, 0, 1, 1, 1}),
            "resident tokens");
    for (int g = 0; g < 3; ++g) {
      uint64_t p =
          uint32_t(pl[30 + 2 * g]) | (uint64_t(uint32_t(pl[31 + 2 * g])) << 32);
      require(p == (uint64_t)(blobs.p + std::vector<int>{0, 48, 16}[g]),
              "resident pointer");
    }
    require(pl[62] == 6 && pl[63] == -77, "resident terminal start");
    res.put({-1, -1, 0, 1});
    resident_plan(ids.p, 6, 3, res.p, 4, blobs.p, offsets.p, 16, plan.p, 8,
                  skip.p, 9, st);
    require(skip.get()[0] == 0 && plan.get() == pl,
            "resident miss must not publish plan");
    B<int> tok(1), tokout(5), selectids(3);
    B<float> selectprobs(3), probout(5);
    idx.put({2});
    selectids.put({18, 21, 42});
    selectprobs.put({.1f, .2f, .3f});
    tokout.put(std::vector<int>(5, -7));
    probout.put(std::vector<float>(5, -7));
    mtp_select(r.p, 7, selectids.p, idx.p, copied.p, tok.p, tokout.p, 2, st,
               selectprobs.p, probout.p);
    got = copied.get();
    for (int i = 0; i < 7; ++i)
      require(got[i] == rv[14 + i], "draft residual selection");
    require(tok.get()[0] == 42 && tokout.get()[2] == 42 &&
                probout.get()[2] == .3f && tokout.get()[3] == -7,
            "draft token selection");
    B<float> logits(3 * 269), probs(3);
    std::vector<float> l(logits.n);
    for (size_t i = 0; i < l.size(); ++i)
      l[i] = std::sin(i * .1f);
    logits.put(l);
    selectids.put({0, 268, -1});
    row_top_prob(logits.p, 3, 269, selectids.p, probs.p, st);
    auto pr = probs.get();
    for (int row = 0; row < 2; ++row) {
      double sum = 0;
      int id = row ? 268 : 0;
      for (int i = 0; i < 269; ++i)
        sum += std::exp(double(l[row * 269 + i]) - l[row * 269 + id]);
      require(std::abs(pr[row] - 1 / sum) < 1e-7, "row probability");
    }
    require(pr[2] == 0, "invalid probability index");
    for (int bits : {2, 4, 8}) {
      constexpr int N = 13, G = 4;
      int cb = (N * bits + 7) / 8;
      B<unsigned char> codes(1 + 4 * cb);
      B<float> sc(4 * G), of(4 * G), emb(4 * N + 3);
      B<int> tokens(4);
      std::vector<unsigned char> cc(codes.n, 0);
      std::vector<float> scales(16), offs(16);
      for (int i = 0; i < 16; ++i) {
        scales[i] = .125f * (i + 1);
        offs[i] = -.25f * i;
      }
      for (int row = 0; row < 4; ++row)
        for (int i = 0; i < N; ++i)
          cc[1 + row * cb + i / (8 / bits)] |= uint8_t(
              ((i + row) & ((1 << bits) - 1)) << ((i % (8 / bits)) * bits));
      codes.put(cc);
      sc.put(scales);
      of.put(offs);
      tokens.put({2, 0, -1, 3});
      emb.put(std::vector<float>(emb.n, 42));
      embedding_gather_dev(codes.p + 1, sc.p, of.p, tokens.p, 4, N, bits, -1, G,
                           cb, G, emb.p, st);
      auto e = emb.get();
      for (int t = 0; t < 4; ++t)
        for (int i = 0; i < N; ++i) {
          int row = std::vector<int>{2, 0, -1, 3}[t];
          float ref = row < 0 ? 0
                              : float(((i + row) & ((1 << bits) - 1)) - 1) *
                                        scales[row * G + i / G] +
                                    offs[row * G + i / G];
          require(e[t * N + i] == ref, "device embedding");
        }
      require(e.back() == 42, "embedding canary");
    }
    bool rejected = false;
    try {
      wait_flag_ge(skip.p, 1, st);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "polling must reject");
    check(cudaStreamDestroy(st));
    std::cout << "SYCL sequence: records, copies, resident plans, token maps "
                 "and embeddings passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
