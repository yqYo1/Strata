// Main-authored MoE Down epilogue qualification; repository LICENSE (MIT).
#include "native16.hpp"
#include "strata/prefill/native16.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
namespace xe = strata::experimental::vllm_xe2_native16;
using Routes = strata::prefill::detail::MoeWindowRoutes;
constexpr int N = 2560, K = 640;
constexpr size_t Pad = 64;
std::shared_ptr<sb::Runtime> runtime;
void need(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
template <class T> T marker();
template <> float marker<float>() { return -913.25f; }
template <> uint16_t marker<uint16_t>() { return 0xa55a; }
template <> int32_t marker<int32_t>() { return 0x5aa55aa5; }
template <> uint32_t marker<uint32_t>() { return 0x5aa55aa5; }
void wait(sycl::queue &q, sycl::event event) {
  try { event.wait_and_throw(); runtime->check(); }
  catch (...) {
    // Keep all caller-owned host and device destinations alive during drain.
    try { q.wait_and_throw(); runtime->check(); } catch (...) { std::_Exit(86); }
    throw;
  }
}
template <class T> struct Buffer {
  sb::Allocation owner;
  size_t n;
  sycl::queue &q;
  Buffer(size_t count, sycl::queue &queue)
      : owner(runtime, (count + 2 * Pad) * sizeof(T), sb::MemoryKind::Device), n(count), q(queue) {
    std::vector<T> values(n + 2 * Pad, marker<T>());
    wait(q, q.memcpy(owner.data(), values.data(), values.size() * sizeof(T)));
  }
  T *data() const { return owner.as<T>() + Pad; }
  void set(const std::vector<T> &values) {
    need(values.size() == n, "upload extent");
    wait(q, q.memcpy(data(), values.data(), n * sizeof(T)));
  }
  std::vector<T> get() const {
    std::vector<T> all(n + 2 * Pad);
    wait(q, q.memcpy(all.data(), owner.data(), all.size() * sizeof(T)));
    for (size_t i = 0; i < Pad; ++i)
      need(all[i] == marker<T>() && all[n + Pad + i] == marker<T>(), "buffer guard changed");
    return std::vector<T>(all.begin() + Pad, all.begin() + Pad + n);
  }
};
template <class T> void equal(const Buffer<T> &a, const Buffer<T> &b) {
  auto x = a.get(), y = b.get();
  need(x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0,
       "separate/fused bitwise mismatch");
}
uint16_t half(float value) { return sycl::bit_cast<uint16_t>(sycl::half(value)); }
std::shared_ptr<const Routes> plan(int tokens, bool irregular) {
  std::vector<int32_t> ids(size_t(tokens) * 10);
  for (int t = 0; t < tokens; ++t)
    for (int s = 0; s < 10; ++s) ids[size_t(t) * 10 + s] = s ? s : (irregular && t % 2 ? 10 : 0);
  return Routes::build_accum(tokens, ids.data(), ids.size());
}
struct Case {
  std::shared_ptr<const Routes> routes;
  sycl::queue &q;
  xe::Gemm gemm;
  Buffer<uint16_t> x, w, down, capture;
  Buffer<int32_t> source, original;
  Buffer<float> weights, reference, candidate;
  Buffer<uint32_t> reference_status, candidate_status;
  std::vector<int32_t> host_source, host_original;
  std::vector<float> initial;
  bool edge_inputs;
  Case(int tokens, bool irregular, bool edges, sycl::queue &queue)
      : routes(plan(tokens, irregular)), q(queue), gemm(*runtime, q),
        x(size_t(std::min(tokens, 8192)) * K, q), w(size_t(N) * K, q),
        down(size_t(std::min(tokens, 8192)) * N, q), capture(down.n, q),
        source(size_t(tokens) * 10, q), original(source.n, q), weights(source.n, q),
        reference(size_t(tokens) * N, q), candidate(reference.n, q),
        reference_status(1, q), candidate_status(1, q),
        host_source(routes->source_rows()), host_original(routes->original_slots()), initial(reference.n), edge_inputs(edges) {
    std::vector<uint16_t> hx(x.n), hw(w.n);
    for (size_t i = 0; i < hx.size(); ++i)
      hx[i] = half(edges ? (i % K == 0 ? 1.f : 0.f) : float(int((i * 13 + i / K) % 17) - 8) / 32.f);
    for (size_t i = 0; i < hw.size(); ++i) {
      const int col = int(i / K), k = int(i % K);
      // Both BF16 halfway parities/signs, finite extremes and FP16 subnormals.
      // BF16 subnormal outputs are unreachable from these finite FP16 products.
      constexpr std::array<uint16_t, 12> edge{
          0x3c04, 0x3c0c, 0xbc04, 0xbc0c, 0x0001, 0x8001,
          0x0000, 0x8000, 0x7bff, 0xfbff, 0x7c00, 0x7e01};
      hw[i] = edges ? (k == 0 ? edge[size_t(col) % edge.size()] : 0) :
          half(float(int((i * 7 + i / K) % 23) - 11) / 64.f);
    }
    x.set(hx); w.set(hw); source.set(host_source); original.set(host_original);
    std::vector<float> hwts(weights.n);
    constexpr std::array<float, 8> values{1.f, -.5f, .125f, 0.f, -0.f, 1.f / 3.f, 0x1p-126f, 16.f};
    for (size_t i = 0; i < hwts.size(); ++i) hwts[i] = values[i % values.size()];
    weights.set(hwts);
    for (size_t i = 0; i < initial.size(); ++i) initial[i] = float(int(i % 19) - 9) / 16.f;
    reset();
  }
  void reset() {
    reference.set(initial); candidate.set(initial);
    reference_status.set({0}); candidate_status.set({0});
  }
  xe::DownScatterProblem problem(size_t job, bool observe = false) {
    return {routes.get(), job, x.data(), w.data(), source.data(), original.data(), weights.data(),
            candidate.data(), candidate_status.data(), observe ? capture.data() : nullptr};
  }
  sycl::event fused(size_t job, xe::Tile tile, bool observe) {
    return gemm.down_scatter(problem(job, observe), tile);
  }
  void separate(size_t index, xe::Tile tile) {
    const auto &job = routes->jobs().at(index);
    xe::Problem p;
    p.x = x.data(); p.w = w.data(); p.y = down.data(); p.t = job.rows; p.n = N; p.k = K;
    (void)gemm.beta0(p, tile);
    strata::prefill::native16::moe_accumulate(*routes, index, down.data(), source.data(), original.data(),
        weights.data(), reference.data(), reference_status.data(), &q);
  }
  void check(size_t index, xe::Tile tile, uint32_t expected = 0) {
    separate(index, tile);
    wait(q, fused(index, tile, true));
    equal(reference, candidate); equal(reference_status, candidate_status);
    need(reference_status.get().at(0) == expected, "unexpected route status");
    auto a = down.get(), b = capture.get();
    const int job_rows = routes->jobs().at(index).rows;
    const auto cells = size_t(job_rows) * N;
    need(std::memcmp(a.data(), b.data(), cells * 2) == 0, "BF16 rounding/cell mapping differs");
    if (!edge_inputs && job_rows <= 3) {
      // Independent exact integer dot product and host RNE BF16 encoder.
      // Every finite test product is a multiple of2^-11; these sums are
      // exactly representable in FP32, independently of DPAS grouping.
      for (int r = 0; r < job_rows; ++r) for (int col = 0; col < N; ++col) {
        int dot = 0;
        for (int k = 0; k < K; ++k)
          dot += (int((size_t(r * K + k) * 13 + r) % 17) - 8) *
                 (int((size_t(col * K + k) * 7 + col) % 23) - 11);
        const uint32_t bits = std::bit_cast<uint32_t>(float(dot) / 2048.f);
        const uint16_t expected = uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        need(a[size_t(r) * N + size_t(col)] == expected, "dense DPAS independent integer oracle mismatch");
      }
    }
    // Also qualify the actual no-capture kernel, including its queue ordering.
    separate(index, tile);
    wait(q, fused(index, tile, false));
    equal(reference, candidate); equal(reference_status, candidate_status);
  }
};

void ownership() {
  for (auto tile : {xe::Tile::compact_16x64, xe::Tile::wide_128x128})
    for (int rows : {1, 3, 17, 33, 129, 2334, 5963, 8191, 8192}) {
      const auto r = xe::inspect_down_ownership(rows, tile);
      need(r.cells == size_t(rows) * N && !r.missing && !r.multiple, "tile ownership is not bijective");
      std::cout << "{\"kind\":\"ownership\",\"tile\":" << int(tile) << ",\"rows\":" << rows
                << ",\"cells\":" << r.cells << ",\"masked\":" << r.masked << ",\"status\":\"PASS\"}\n";
    }
}
void admissions(Case &c) {
  int rejected = 0;
  auto rejects = [&](xe::DownScatterProblem p, xe::Tile t = xe::Tile::wide_128x128) {
    try { (void)c.gemm.down_scatter(p, t); }
    catch (const std::invalid_argument &) { ++rejected; return; }
    catch (const std::out_of_range &) { ++rejected; return; }
    throw std::runtime_error("invalid admission submitted work");
  };
  auto p = c.problem(0); p.plan = nullptr; rejects(p);
  p = c.problem(c.routes->jobs().size()); rejects(p);
  p = c.problem(0); rejects(p, static_cast<xe::Tile>(99));
  p = c.problem(0); p.x += 1; rejects(p);
  p = c.problem(0); p.x = reinterpret_cast<uint16_t *>(c.initial.data()); rejects(p);
  p = c.problem(0); p.accumulator = reinterpret_cast<float *>(c.x.data()); rejects(p);
  Buffer<uint16_t> tiny(1, c.q); p = c.problem(0); p.x = tiny.data(); rejects(p);
  p = c.problem(0, true); p.rounded_diagnostic = reinterpret_cast<uint16_t *>(c.candidate.data()); rejects(p);
  std::vector<int32_t> duplicate(10, 0);
  try { (void)Routes::build_accum(1, duplicate.data(), duplicate.size()); }
  catch (const std::invalid_argument &) { ++rejected; }
  need(rejected == 9, "admission checks incomplete");
  std::cout << "{\"kind\":\"admissions\",\"rejected\":" << rejected << ",\"status\":\"PASS\"}\n";
}
void correctness(int tokens, bool irregular, bool edges, xe::Tile tile, sycl::queue &q) {
  Case c(tokens, irregular, edges, q);
  std::vector<size_t> jobs{0, std::min(size_t(1), c.routes->jobs().size() - 1), c.routes->jobs().size() - 1};
  if (tokens > 8192 && !irregular) jobs.push_back(2); // split job plus next expert, overlapping tokens
  std::sort(jobs.begin(), jobs.end()); jobs.erase(std::unique(jobs.begin(), jobs.end()), jobs.end());
  for (auto job : jobs) c.check(job, tile);
  std::cout << "{\"kind\":\"correctness\",\"tokens\":" << tokens << ",\"irregular\":" << irregular
            << ",\"edges\":" << edges << ",\"tile\":" << int(tile) << ",\"jobs\":" << jobs.size()
            << ",\"status\":\"PASS\"}\n";
}
class Marker;
sycl::event mark(sycl::queue &q) { return q.single_task<Marker>([] {}); }
void benchmark(int process, sycl::queue &q) {
  for (int rows : {64, 160, 2334, 5963, 8192}) for (bool irregular : {false, true}) {
    const auto tile = rows < 128 ? xe::Tile::compact_16x64 : xe::Tile::wide_128x128;
    Case c(rows * (irregular ? 2 : 1), irregular, false, q);
    need(c.routes->jobs().front().rows == rows, "timing job shape");
    c.check(0, tile);
    constexpr int repeat = 16;
    for (int round = -1; round < 6; ++round) for (int order = 0; order < 2; ++order) {
      const int arm = (process + round + 3 + order) % 2;
      c.reset();
      const auto start = mark(q);
      const auto wall = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; ++i)
        if (arm) (void)c.fused(0, tile, false); else c.separate(0, tile);
      const auto end = mark(q); wait(q, end);
      const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall).count() / repeat;
      const double queue_ms = double(end.get_profiling_info<sycl::info::event_profiling::command_start>() -
          start.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1e6 / repeat;
      if (arm) {
        for (int i = 0; i < repeat; ++i) c.separate(0, tile);
      } else {
        for (int i = 0; i < repeat; ++i) (void)c.fused(0, tile, false);
      }
      q.wait_and_throw(); runtime->check(); equal(c.reference, c.candidate);
      need(c.reference_status.get()[0] == 0 && c.candidate_status.get()[0] == 0, "timed route status");
      need(std::isfinite(wall_ms) && wall_ms > 0 && std::isfinite(queue_ms) && queue_ms > 0, "invalid timing");
      std::cout << "{\"kind\":\"sample\",\"process\":" << process << ",\"rows\":" << rows
                << ",\"irregular\":" << irregular << ",\"round\":" << round << ",\"order\":" << order
                << ",\"arm\":" << arm << ",\"repeat\":" << repeat << ",\"wall_ms\":" << wall_ms
                << ",\"queue_interval_ms\":" << queue_ms << ",\"status\":\"PASS\"}\n";
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);
  try {
    need(argc >= 2, "use --ownership|--smoke|--validate|--bench PROCESS");
    const std::string mode = argv[1];
    if (mode == "--ownership") ownership();
    else {
      runtime = sb::runtime_for();
      auto q = runtime->make_queue(true);
      if (mode == "--smoke") {
        correctness(3, false, false, xe::Tile::wide_128x128, q);
        correctness(3, true, true, xe::Tile::compact_16x64, q);
        Case c(3, false, false, q); admissions(c);
        for (int kind = 0; kind < 4; ++kind) {
          c.reset(); auto s = c.host_source, o = c.host_original;
          const auto at = size_t(c.routes->jobs()[0].global_row);
          if (kind == 0) o[at] = -1;
          if (kind == 1) o[at] = int(o.size());
          if (kind == 2) s[at] = -1;
          if (kind == 3) s[at] = c.routes->rows();
          c.source.set(s); c.original.set(o);
          c.check(0, xe::Tile::wide_128x128, kind < 2 ? 1 : 2);
          std::cout << "{\"kind\":\"invalid_map\",\"case\":" << kind << ",\"status\":\"PASS\"}\n";
        }
      } else if (mode == "--validate") {
        for (auto tile : {xe::Tile::compact_16x64, xe::Tile::wide_128x128}) {
          for (int tokens : {1, 17, 33, 129, 2334, 5963, 8191, 8192, 8193})
            correctness(tokens, false, false, tile, q);
          correctness(16384, true, false, tile, q);
          correctness(17, true, true, tile, q);
        }
      } else if (mode == "--bench") { need(argc == 3, "bench process required"); benchmark(std::stoi(argv[2]), q); }
      else throw std::runtime_error("invalid mode");
      q.wait_and_throw(); runtime->check(); runtime->wait();
    }
    std::cout << "fixture_status=PASS\n"; return 0;
  } catch (const std::exception &e) {
    if (runtime) { try { runtime->wait(); } catch (...) { std::_Exit(86); } }
    std::cerr << "fixture_status=FAIL " << e.what() << '\n'; return 1;
  }
}
