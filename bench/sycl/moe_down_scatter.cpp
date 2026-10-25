// Main-authored MoE Down epilogue qualification; repository LICENSE (MIT).
#include "native16.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/native16.hpp"
#include "strata/sycl/command_profile.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
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
void drain(sycl::queue &q) noexcept {
  try { q.wait_and_throw(); runtime->check(); }
  catch (...) {
    std::cerr << "CHECKED_DRAIN_FAILED: Down fixture queue; retaining owners until exit\n";
    std::_Exit(86);
  }
}
void wait(sycl::queue &q, sycl::event event) {
  try { event.wait_and_throw(); runtime->check(); }
  catch (...) {
    // Keep all caller-owned host and device destinations alive during drain.
    drain(q);
    throw;
  }
}
template <class Submit> void submit_wait(sycl::queue &q, Submit submit) {
  // Catch synchronous submission failures while the caller's host copy buffer
  // is still alive, including during Buffer construction.
  try { wait(q, submit()); } catch (...) { drain(q); throw; }
}
template <class T> struct Buffer {
  sb::Allocation owner;
  size_t n;
  sycl::queue &q;
  Buffer(size_t count, sycl::queue &queue)
      : owner(runtime, (count + 2 * Pad) * sizeof(T), sb::MemoryKind::Device), n(count), q(queue) {
    std::vector<T> values(n + 2 * Pad, marker<T>());
    submit_wait(q, [&] { return q.memcpy(owner.data(), values.data(), values.size() * sizeof(T)); });
  }
  // Runtime::wait() drains its two built-in queues, not this borrowed queue.
  // Also protect members unwound from an incomplete Case constructor.
  ~Buffer() { drain(q); }
  T *data() const { return owner.as<T>() + Pad; }
  void set(const std::vector<T> &values) {
    need(values.size() == n, "upload extent");
    submit_wait(q, [&] { return q.memcpy(data(), values.data(), n * sizeof(T)); });
  }
  std::vector<T> get() const {
    std::vector<T> all(n + 2 * Pad);
    submit_wait(q, [&] { return q.memcpy(all.data(), owner.data(), all.size() * sizeof(T)); });
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
template <class T> bool same_bytes(const std::vector<T> &a, const std::vector<T> &b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
uint16_t half(float value) { return sycl::bit_cast<uint16_t>(sycl::half(value)); }
std::shared_ptr<const Routes> plan(int tokens, bool irregular, int target_rows = 0) {
  std::vector<int32_t> ids(size_t(tokens) * 10);
  need(target_rows >= 0 && target_rows <= std::min(tokens, 8192), "target row span");
  for (int t = 0; t < tokens; ++t) {
    // A fixed model-sized window with either a contiguous prefix or exactly
    // target_rows evenly distributed unique tokens. Other slots stay valid.
    const bool selected = target_rows ? (irregular ?
        int64_t(t + 1) * target_rows / tokens != int64_t(t) * target_rows / tokens :
        t < target_rows) : !(irregular && t % 2);
    for (int slot = 0; slot < 10; ++slot)
      ids[size_t(t) * 10 + slot] = slot ? slot : (selected ? 0 : 10);
  }
  return Routes::build_accum(tokens, ids.data(), ids.size());
}
struct Case {
  struct Inputs {
    std::vector<uint16_t> x, w;
    std::vector<int32_t> source, original;
    std::vector<float> weights;
  };
  std::shared_ptr<const Routes> routes;
  sycl::queue &q;
  xe::Gemm gemm;
  strata::prefill::Gemm routed;
  bool routed_initialized = false;
  Buffer<uint16_t> x, w, down, capture;
  Buffer<int32_t> source, original;
  Buffer<float> weights, reference, candidate;
  Buffer<uint32_t> reference_status, candidate_status;
  std::vector<int32_t> host_source, host_original;
  std::vector<float> initial;
  bool edge_inputs;
  Case(int tokens, bool irregular, bool edges, sycl::queue &queue, int target_rows = 0)
      : routes(plan(tokens, irregular, target_rows)), q(queue), gemm(*runtime, q),
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
  // Run before host routes and USM members are destroyed, including when
  // a later submission throws before an event reaches wait().
  ~Case() { drain(q); }
  void reset() {
    reference.set(initial); candidate.set(initial);
    reference_status.set({0}); candidate_status.set({0});
  }
  Inputs input_snapshot() const { return {x.get(), w.get(), source.get(), original.get(), weights.get()}; }
  void inputs_unchanged(const Inputs &before) const {
    const auto after = input_snapshot();
    need(same_bytes(before.x, after.x) && same_bytes(before.w, after.w) &&
         same_bytes(before.source, after.source) && same_bytes(before.original, after.original) &&
         same_bytes(before.weights, after.weights), "input bits changed");
  }
  xe::DownScatterProblem problem(size_t job, bool observe = false) {
    return {routes.get(), job, x.data(), w.data(), source.data(), original.data(), weights.data(),
            candidate.data(), candidate_status.data(), observe ? capture.data() : nullptr};
  }
  sycl::event fused(size_t job, xe::Tile tile, bool observe, xe::ScatterLayout mapping) {
    return gemm.down_scatter(problem(job, observe), tile, mapping);
  }
  void init_routed() {
    if (!routed_initialized) {
      std::string error;
      if (!routed.init_external(&q, nullptr, 0, nullptr, 0, error))
        throw std::runtime_error(error);
      routed_initialized = true;
    }
  }
  bool try_selected(size_t index) {
    init_routed();
    return routed.try_down_scatter16(*routes, index, x.data(), w.data(), source.data(),
        original.data(), weights.data(), candidate.data(), candidate_status.data());
  }
  void selected_fallback(size_t index) {
    const auto &job = routes->jobs().at(index);
    routed.f16_out16(x.data(), w.data(), down.data(), job.rows, N, K, true);
    strata::prefill::native16::moe_accumulate(*routes, index, down.data(), source.data(),
        original.data(), weights.data(), candidate.data(), candidate_status.data(), &q);
  }
  bool selected_submit(size_t index) {
    const bool selected = try_selected(index);
    if (!selected) selected_fallback(index);
    return selected;
  }
  void check_selected(bool enabled, uint32_t expected_mask = 0) {
    const auto inputs = input_snapshot();
    const auto tile = routes->jobs().front().rows < 128 ?
        xe::Tile::compact_16x64 : xe::Tile::wide_128x128;
    separate(0, tile); drain(q);
    const auto down_before = down.get(), capture_before = capture.get();
    const bool selected = try_selected(0); drain(q);
    need(selected == (enabled && routes->jobs().front().rows >= 2334), "wrong selected route");
    need(same_bytes(down_before, down.get()) && same_bytes(capture_before, capture.get()),
         "try route wrote a Down/diagnostic tensor");
    if (!selected) {
      need(same_bytes(initial, candidate.get()) && candidate_status.get()[0] == 0,
           "false route modified outputs");
      selected_fallback(0); drain(q);
    }
    equal(reference, candidate); equal(reference_status, candidate_status);
    need(candidate_status.get()[0] == expected_mask, "selected route error mask");
    inputs_unchanged(inputs);
    std::cout << "{\"kind\":\"selected_route\",\"rows\":" << routes->jobs().front().rows
              << ",\"window_rows\":" << routes->rows() << ",\"source_first\":" << host_source.front()
              << ",\"source_last\":" << host_source[size_t(routes->jobs().front().rows - 1)]
              << ",\"enabled\":" << enabled << ",\"selected\":" << selected
              << ",\"expected_mask\":" << expected_mask
              << ",\"inputs_immutable\":true,\"false_route_no_write\":true,\"status\":\"PASS\"}\n";
  }
  void check_profile() {
    namespace cp = sb::command_profile;
    init_routed();
    const auto inputs=input_snapshot();
    separate(0,xe::Tile::wide_128x128);drain(q);
    const auto before_down=down.get(),before_capture=capture.get();
    const auto& job=routes->jobs().front();
    bool selected=false;
    {
      std::unique_ptr<cp::Trace> trace;
      if (cp::requested()) trace.reset(new cp::Trace(&q,1));
      struct Retirement { sycl::queue& queue; ~Retirement() { drain(queue); } } retirement{q};
      auto frame=trace ? trace->begin(0,0) : cp::Frame{};
      frame.consider(job.expert,job.global_row,job.rows);
      {
        auto tag=frame.tag(cp::Kind::down,job.expert,job.global_row,job.rows,N,K);
        selected=try_selected(0);
        if (!selected) routed.f16_out16(x.data(),w.data(),down.data(),job.rows,N,K,true);
      }
      if (!selected) {
        auto tag=frame.tag(cp::Kind::accumulate,job.expert,job.global_row,job.rows,N);
        strata::prefill::native16::moe_accumulate(*routes,0,down.data(),source.data(),
            original.data(),weights.data(),candidate.data(),candidate_status.data(),&q);
      }
      drain(q);frame.after_checked_drain();
    }
    need(selected==(job.rows>=2334),"profile changed production Down dispatch");
    equal(reference,candidate);equal(reference_status,candidate_status);
    need(candidate_status.get()[0]==0,"profiled Down status");
    need(same_bytes(before_down,down.get()) && same_bytes(before_capture,capture.get()),"profiled Down wrote separate output");
    inputs_unchanged(inputs);
    std::cout<<"{\"kind\":\"profiled_engine_down\",\"rows\":"<<job.rows
      <<",\"selected\":"<<selected<<",\"expected_records\":"<<(selected ? 1 : 2)
      <<",\"guards\":true,\"inputs_immutable\":true,\"status\":\"PASS\"}\n";
  }
  void separate(size_t index, xe::Tile tile) {
    try {
      const auto &job = routes->jobs().at(index);
      xe::Problem p;
      p.x = x.data(); p.w = w.data(); p.y = down.data(); p.t = job.rows; p.n = N; p.k = K;
      (void)gemm.beta0(p, tile);
      strata::prefill::native16::moe_accumulate(*routes, index, down.data(), source.data(), original.data(),
          weights.data(), reference.data(), reference_status.data(), &q);
    } catch (...) { drain(q); throw; }
  }
  void check(size_t index, xe::Tile tile, xe::ScatterLayout mapping, uint32_t expected = 0) {
    const auto inputs = input_snapshot();
    down.set(std::vector<uint16_t>(down.n, marker<uint16_t>()));
    capture.set(std::vector<uint16_t>(capture.n, marker<uint16_t>()));
    separate(index, tile);
    wait(q, fused(index, tile, true, mapping));
    equal(reference, candidate); equal(reference_status, candidate_status);
    need(reference_status.get().at(0) == expected, "unexpected route status");
    auto a = down.get(), b = capture.get();
    const int job_rows = routes->jobs().at(index).rows;
    const auto cells = size_t(job_rows) * N;
    need(std::memcmp(a.data(), b.data(), cells * 2) == 0, "BF16 rounding/cell mapping differs");
    for (size_t i = cells; i < a.size(); ++i)
      need(a[i] == marker<uint16_t>() && b[i] == marker<uint16_t>(), "logical M tail overwritten");
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
    wait(q, fused(index, tile, false, mapping));
    equal(reference, candidate); equal(reference_status, candidate_status);
    need(same_bytes(b, capture.get()), "no-capture kernel wrote diagnostic storage");
    inputs_unchanged(inputs);
  }
};

void ownership() {
  for (auto mapping : {xe::ScatterLayout::matrix, xe::ScatterLayout::output_copy,
                       xe::ScatterLayout::matrix_row_cache})
   for (auto tile : {xe::Tile::compact_16x64, xe::Tile::wide_128x128})
    for (int rows : {1, 3, 17, 33, 129, 2334, 5963, 8191, 8192}) {
      const auto r = xe::inspect_down_ownership(rows, tile, mapping);
      need(r.cells == size_t(rows) * N && !r.missing && !r.multiple, "tile ownership is not bijective");
      need(!r.metadata_missing && !r.metadata_multiple && !r.metadata_outside_tile &&
           (mapping != xe::ScatterLayout::matrix_row_cache || r.metadata_slots > 0),
           "local row metadata ownership or coordinate bounds");
      std::cout << "{\"kind\":\"ownership\",\"tile\":" << int(tile) << ",\"rows\":" << rows
                << ",\"mapping\":" << int(mapping) << ",\"cells\":" << r.cells << ",\"masked\":" << r.masked
                << ",\"metadata_slots\":" << r.metadata_slots << ",\"metadata_missing\":" << r.metadata_missing
                << ",\"metadata_multiple\":" << r.metadata_multiple << ",\"metadata_outside_tile\":" << r.metadata_outside_tile
                << ",\"status\":\"PASS\"}\n";
    }
}
void admissions(Case &c, xe::ScatterLayout mapping) {
  int rejected = 0;
  auto rejects = [&](xe::DownScatterProblem p, xe::Tile t = xe::Tile::wide_128x128) {
    try { (void)c.gemm.down_scatter(p, t, mapping); }
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
  const auto unaligned = [](auto *pointer) {
    return reinterpret_cast<decltype(pointer)>(reinterpret_cast<uintptr_t>(pointer) + 1);
  };
  p = c.problem(0); p.source = unaligned(p.source); rejects(p);
  p = c.problem(0); p.original = unaligned(p.original); rejects(p);
  p = c.problem(0); p.job_weights = unaligned(p.job_weights); rejects(p);
  p = c.problem(0); p.accumulator = unaligned(p.accumulator); rejects(p);
  p = c.problem(0); p.status = unaligned(p.status); rejects(p);
  p = c.problem(0, true); p.rounded_diagnostic = unaligned(p.rounded_diagnostic); rejects(p);
  std::vector<int32_t> duplicate(10, 0);
  try { (void)Routes::build_accum(1, duplicate.data(), duplicate.size()); }
  catch (const std::invalid_argument &) { ++rejected; }
  try { (void)c.gemm.down_scatter(c.problem(0), xe::Tile::wide_128x128,
                                  static_cast<xe::ScatterLayout>(99)); }
  catch (const std::invalid_argument &) { ++rejected; }
  need(rejected == 16, "admission checks incomplete");
  std::cout << "{\"kind\":\"admissions\",\"rejected\":" << rejected << ",\"mapping\":" << int(mapping) << ",\"status\":\"PASS\"}\n";
}
void correctness(int tokens, bool irregular, bool edges, xe::Tile tile, sycl::queue &q,
                 xe::ScatterLayout mapping, int target_rows = 0) {
  Case c(tokens, irregular, edges, q, target_rows);
  std::vector<size_t> jobs{0, std::min(size_t(1), c.routes->jobs().size() - 1), c.routes->jobs().size() - 1};
  if (tokens > 8192 && !irregular) jobs.push_back(2); // split job plus next expert, overlapping tokens
  std::sort(jobs.begin(), jobs.end()); jobs.erase(std::unique(jobs.begin(), jobs.end()), jobs.end());
  for (auto job : jobs) c.check(job, tile, mapping);
  std::cout << "{\"kind\":\"correctness\",\"tokens\":" << tokens << ",\"irregular\":" << irregular
            << ",\"target_rows\":" << target_rows << ",\"mapping\":" << int(mapping)
            << ",\"edges\":" << edges << ",\"tile\":" << int(tile) << ",\"jobs\":" << jobs.size()
            << ",\"inputs_immutable\":true,\"capture_and_no_capture\":true,\"status\":\"PASS\"}\n";
}
void exception_unwind(sycl::queue &q, xe::ScatterLayout mapping) {
  sycl::event pending;
  bool submitted = false, rejected = false;
  try {
    Case c(3, false, false, q);
    pending = c.fused(0, xe::Tile::wide_128x128, false, mapping);
    submitted = true;
    auto invalid = c.problem(0);
    invalid.plan = nullptr;
    (void)c.gemm.down_scatter(invalid, xe::Tile::wide_128x128, mapping);
  } catch (const std::invalid_argument &) { rejected = true; }
  need(submitted && rejected, "synchronous admission exception not exercised");
  need(pending.get_info<sycl::info::event::command_execution_status>() ==
           sycl::info::event_command_status::complete,
       "borrowed queue still pending after Case destruction");
  runtime->check();
  std::cout << "{\"kind\":\"exception_unwind\",\"mapping\":" << int(mapping)
            << ",\"queue_complete\":true,\"status\":\"PASS\"}\n";
}
class Marker;
sycl::event mark(sycl::queue &q) { return q.single_task<Marker>([] {}); }
void benchmark(int process, sycl::queue &q, int public_arm = -1) {
  for (int rows : {64, 160, 2333, 2334, 2335, 5963, 8192}) for (bool irregular : {false, true}) {
    const auto tile = rows < 128 ? xe::Tile::compact_16x64 : xe::Tile::wide_128x128;
    Case c(78451, irregular, false, q, rows);
    need(c.routes->jobs().front().rows == rows, "timing job shape");
    c.check(0, tile, xe::ScatterLayout::matrix);
    c.check(0, tile, xe::ScatterLayout::matrix_row_cache);
    c.reset(); c.check_selected(public_arm != 0);
    const auto inputs = c.input_snapshot();
    const int repeat = public_arm >= 0 && rows < 256 ? 256 : 16;
    for (int round = -1; round < 6; ++round)
      for (int order = 0; order < (public_arm >= 0 ? 1 : 2); ++order) {
      const int arm = public_arm >= 0 ? public_arm : (process + round + 2 + order) % 2;
      c.reset();
      int fused_dispatches = 0;
      const auto start = mark(q);
      const auto wall = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; ++i)
        if (public_arm >= 0 || arm) fused_dispatches += c.selected_submit(0); else c.separate(0, tile);
      const auto end = mark(q); wait(q, end);
      const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall).count() / repeat;
      const auto begin_ns = start.get_profiling_info<sycl::info::event_profiling::command_end>();
      const auto end_ns = end.get_profiling_info<sycl::info::event_profiling::command_start>();
      need(end_ns > begin_ns, "unordered timing markers");
      const double queue_batch_ms = double(end_ns - begin_ns) / 1e6;
      const double queue_ms = queue_batch_ms / repeat;
      need(fused_dispatches == (arm && rows >= 2334 ? repeat : 0), "timed dispatch differs from selected arm");
      if (public_arm >= 0 || arm) {
        for (int i = 0; i < repeat; ++i) c.separate(0, tile);
      } else {
        for (int i = 0; i < repeat; ++i) (void)c.selected_submit(0);
      }
      q.wait_and_throw(); runtime->check(); equal(c.reference, c.candidate);
      c.inputs_unchanged(inputs);
      need(c.reference_status.get()[0] == 0 && c.candidate_status.get()[0] == 0, "timed route status");
      need(std::isfinite(wall_ms) && wall_ms > 0 && std::isfinite(queue_ms) && queue_ms > 0, "invalid timing");
      std::cout << "{\"kind\":\"sample\",\"process\":" << process << ",\"rows\":" << rows
                << ",\"window_rows\":" << c.routes->rows() << ",\"accumulator_bytes\":" << c.reference.n * sizeof(float)
                << ",\"source_first\":" << c.host_source.front() << ",\"source_last\":" << c.host_source[size_t(rows - 1)]
                << ",\"irregular\":" << irregular << ",\"round\":" << round << ",\"order\":" << order
                << ",\"arm\":" << arm << ",\"fused_route\":" << (fused_dispatches != 0)
                << ",\"public_dispatch\":" << (public_arm >= 0)
                << ",\"observed_fused_dispatches\":" << fused_dispatches
                << ",\"repeat\":" << repeat << ",\"wall_ms\":" << wall_ms
                << ",\"queue_batch_ms\":" << queue_batch_ms
                << ",\"queue_interval_ms\":" << queue_ms << ",\"inputs_immutable\":true,\"status\":\"PASS\"}\n";
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cout << std::setprecision(12);
  try {
    need(argc >= 2, "use --ownership|--smoke|--validate|--bench PROCESS");
    const std::string mode = argv[1];
    if (mode == "--ownership") ownership();
    else {
      runtime = sb::runtime_for();
      auto q = runtime->make_queue(true);
      if (mode == "--profile-engine-hook") {
        need(argc==2,"profile hook takes no timing parameters");
        for (int rows : {2333,2334,8192}) { Case c(rows,false,false,q);c.check_profile(); }
        Case irregular(16384,true,false,q,2334);irregular.check_profile();
      } else if (mode == "--smoke") {
        for (auto mapping : {xe::ScatterLayout::matrix, xe::ScatterLayout::output_copy,
                             xe::ScatterLayout::matrix_row_cache}) {
          correctness(3, false, false, xe::Tile::wide_128x128, q, mapping);
          correctness(3, true, true, xe::Tile::compact_16x64, q, mapping);
          Case c(3, false, false, q); admissions(c, mapping);
          for (int kind = 0; kind < 7; ++kind) {
            c.reset(); auto src = c.host_source, original = c.host_original;
            const auto at = size_t(c.routes->jobs()[0].global_row);
            if (kind == 0) original[at] = -1;
            if (kind == 1) original[at] = int(original.size());
            if (kind == 2) src[at] = -1;
            if (kind == 3) src[at] = c.routes->rows();
            if (kind == 4) { original[at] = -1; src[at] = -1; }
            if (kind == 5) src[at] = (src[at] + 1) % c.routes->rows();
            if (kind == 6) { original[at] = -1; src[at + 1] = -1; }
            c.source.set(src); c.original.set(original);
            const uint32_t expected = kind == 6 ? 3 : (kind < 2 || kind == 4) ? 1 : 2;
            c.check(0, xe::Tile::wide_128x128, mapping, expected);
            std::cout << "{\"kind\":\"invalid_map\",\"case\":" << kind
                      << ",\"mapping\":" << int(mapping) << ",\"expected_mask\":" << expected
                      << ",\"inputs_immutable\":true,\"capture_and_no_capture\":true,\"status\":\"PASS\"}\n";
          }
          exception_unwind(q, mapping);
        }
      } else if (mode == "--validate") {
        for (auto mapping : {xe::ScatterLayout::matrix, xe::ScatterLayout::output_copy,
                             xe::ScatterLayout::matrix_row_cache})
          for (auto tile : {xe::Tile::compact_16x64, xe::Tile::wide_128x128}) {
            for (int tokens : {1, 17, 33, 129, 2334, 5963, 8191, 8192, 8193})
              correctness(tokens, false, false, tile, q, mapping);
            correctness(16384, true, false, tile, q, mapping);
            correctness(17, true, true, tile, q, mapping);
            for (bool irregular : {false, true})
              correctness(78451, irregular, false, tile, q, mapping, 17);
          }
      } else if (mode == "--route-smoke" || mode == "--route-validate") {
        need(argc == 3 && (std::string(argv[2]) == "0" || std::string(argv[2]) == "1"), "route enabled argument");
        const bool enabled = std::string(argv[2]) == "1";
        if (mode == "--route-smoke") {
          for (int rows : {2333, 2334, 2335}) {
            Case c(rows, false, false, q); c.check_selected(enabled);
          }
          Case c(2334, false, false, q);
          for (uint32_t mask : {1u, 2u, 3u}) {
            c.reset(); auto src = c.host_source, original = c.host_original;
            if (mask & 1) original[0] = -1;
            if (mask & 2) src[mask == 3 ? 1 : 0] = -1;
            c.source.set(src); c.original.set(original); c.check_selected(enabled, mask);
          }
        } else {
          for (int rows : {1, 17, 160, 2333, 2334, 2335, 8191, 8192})
            for (bool irregular : {false, true}) {
              Case c(2 * rows + 17, irregular, false, q, rows); c.check_selected(enabled);
            }
        }
      } else if (mode == "--bench-public") {
        need(argc == 4, "public bench process and arm required");
        const int process = std::stoi(argv[2]), arm = std::stoi(argv[3]);
        need(process >= 0 && process < 3 && (arm == 0 || arm == 1), "public bench process/arm");
        const char *flag = std::getenv("STRATA_SYCL_PREFILL_DOWN_SCATTER");
        need(flag && std::string(flag) == (arm ? "large" : "0"), "public arm must match startup environment");
        benchmark(process, q, arm);
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
