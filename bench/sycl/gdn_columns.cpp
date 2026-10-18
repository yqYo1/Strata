// Main-authored exact-math GDN scheduling fixture; repository LICENSE.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
namespace p = strata::prefill;
constexpr int C = 10240, HV = 48, V = HV * 128, ST = 128 * V;
constexpr size_t Pad = 32;
constexpr std::array<int, 5> Arms{0, 1, 2, 4, 8};
using Clock = std::chrono::steady_clock;
std::shared_ptr<sb::Runtime> rt;
void need(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template <class T> T sentinel();
template <> float sentinel<float>() { return -731.25f; }
template <> uint16_t sentinel<uint16_t>() { return 0x5a5a; }
template <class T> struct Buffer {
  sb::Allocation owner;
  size_t n;
  explicit Buffer(size_t count)
      : owner(rt, (count + 2 * Pad) * sizeof(T), sb::MemoryKind::Device), n(count) {
    reset();
  }
  T *data() { return owner.as<T>() + Pad; }
  void reset() {
    std::vector<T> values(n + 2 * Pad, sentinel<T>());
    rt->wait(rt->compute().memcpy(owner.data(), values.data(), values.size() * sizeof(T)));
  }
  void set(const std::vector<T> &input) {
    need(input.size() == n, "fixture input extent");
    rt->wait(rt->compute().memcpy(data(), input.data(), n * sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> values(n + 2 * Pad);
    rt->wait(rt->compute().memcpy(values.data(), owner.data(), values.size() * sizeof(T)));
    for (size_t i = 0; i < Pad; ++i)
      need(values[i] == sentinel<T>() && values[Pad + n + i] == sentinel<T>(), "fixture guard changed");
    return values;
  }
  void immutable(const std::vector<T> &input) {
    auto values = get();
    need(input.size() == n && std::memcmp(values.data() + Pad, input.data(), n * sizeof(T)) == 0,
         "read-only fixture input changed");
  }
};
template <class T> void equal(Buffer<T> &a, Buffer<T> &b) {
  auto x = a.get(), y = b.get();
  need(x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0,
       "partitioned GDN differs bit for bit");
}
struct Trial {
  Buffer<float> state, y;
  Buffer<uint16_t> y16;
  explicit Trial(int tokens) : state(ST), y(size_t(tokens) * V), y16(size_t(tokens) * V) {}
};
struct Case {
  int capacity;
  std::vector<float> h, gate, beta, z, gamma, initial;
  Buffer<float> H, G, B, Z, N;
  std::vector<std::unique_ptr<Trial>> trials;
  explicit Case(int tokens, int arms = 2)
      : capacity(tokens), h(size_t(tokens) * C), gate(size_t(tokens) * HV), beta(gate.size()),
        z(size_t(tokens) * V), gamma(128), initial(ST), H(h.size()), G(gate.size()),
        B(beta.size()), Z(z.size()), N(gamma.size()) {
    for (size_t i = 0; i < h.size(); ++i)
      h[i] = float(int((i * 37 + i / 101 + 17) % 127) - 63) / 512.f;
    for (size_t i = 0; i < gate.size(); ++i) {
      gate[i] = -float((i * 7 + i / HV) % 31) / 32.f;
      beta[i] = float((i * 13 + i / HV) % 17) / 16.f;
    }
    for (size_t i = 0; i < z.size(); ++i)
      z[i] = float(int((i * 19 + i / 131) % 257) - 128) / 8.f;
    for (size_t i = 0; i < gamma.size(); ++i) gamma[i] = .5f + float(i % 17) / 32.f;
    for (size_t i = 0; i < initial.size(); ++i)
      initial[i] = float(int((i * 23 + i / 19) % 61) - 30) / 1024.f;
    H.set(h); G.set(gate); B.set(beta); Z.set(z); N.set(gamma);
    for (int i = 0; i < arms; ++i) {
      trials.push_back(std::make_unique<Trial>(tokens));
      trials.back()->state.set(initial);
    }
  }
  void run(Trial &trial, int partitions, int tokens, int offset = 0) {
    need(tokens > 0 && offset >= 0 && offset + tokens <= capacity, "fixture run extent");
    if (partitions)
      p::gdn_recurrence_partitioned(trial.state.data(), H.data() + size_t(offset) * C,
          G.data() + size_t(offset) * HV, B.data() + size_t(offset) * HV, Z.data() + size_t(offset) * V,
          N.data(), 1e-6f, trial.y.data() + size_t(offset) * V, trial.y16.data() + size_t(offset) * V,
          tokens, partitions, &rt->compute());
    else
      p::gdn_recurrence(trial.state.data(), H.data() + size_t(offset) * C,
          G.data() + size_t(offset) * HV, B.data() + size_t(offset) * HV, Z.data() + size_t(offset) * V,
          N.data(), 1e-6f, trial.y.data() + size_t(offset) * V, trial.y16.data() + size_t(offset) * V,
          tokens, &rt->compute());
  }
  void compare(Trial &a, Trial &b, int active_tokens = -1) {
    if (active_tokens < 0) active_tokens = capacity;
    equal(a.state, b.state); equal(a.y, b.y); equal(a.y16, b.y16);
    auto values = b.y.get();
    auto image = b.y16.get();
    for (size_t i = 0; i < size_t(active_tokens) * V; ++i)
      need(std::isfinite(values[Pad + i]) &&
           image[Pad + i] == strata::kernels::f16_from_f32(values[Pad + i]), "finite output or FP16 image");
    for (size_t i = size_t(active_tokens) * V; i < size_t(capacity) * V; ++i)
      need(values[Pad + i] == sentinel<float>() && image[Pad + i] == sentinel<uint16_t>(),
           "output tail beyond requested tokens changed");
    auto state = b.state.get();
    need(std::all_of(state.begin(), state.end(), [](float v) { return std::isfinite(v); }), "nonfinite state");
  }
  void immutable() { H.immutable(h); G.immutable(gate); B.immutable(beta); Z.immutable(z); N.immutable(gamma); }
};
void correctness(int tokens, bool split) {
  Case c(tokens);
  c.run(*c.trials[0], 0, tokens);
  for (int partitions : {1, 2, 4, 8}) {
    auto &candidate = *c.trials[1];
    candidate.state.set(c.initial); candidate.y.reset(); candidate.y16.reset();
    if (split) {
      c.run(candidate, partitions, 5);
      c.run(candidate, partitions, tokens - 5, 5);
    } else c.run(candidate, partitions, tokens);
    rt->wait(); c.compare(*c.trials[0], candidate);
    std::cout << "{\"kind\":\"validation\",\"tokens\":" << tokens
              << ",\"partitions\":" << partitions << ",\"split\":" << (split ? "true" : "false")
              << ",\"exact_output_state_fp16\":true,\"guards\":true,\"status\":\"PASS\"}\n";
  }
  c.immutable();
}
void rejection() {
  Case c(1); auto &v = *c.trials[1]; int rejected = 0;
  for (int mode = 0; mode < 6; ++mode) {
    try {
      float host = 0;
      p::gdn_recurrence_partitioned(v.state.data(), mode == 5 ? &host : c.H.data(),
          c.G.data(), c.B.data(), c.Z.data(), c.N.data(), mode == 3 ? NAN : 1e-6f,
          mode == 4 ? v.state.data() : v.y.data(), v.y16.data(), mode == 1 ? 0 : mode == 2 ? 8193 : 1,
          mode == 0 ? 3 : 4, &rt->compute());
    } catch (const std::invalid_argument &) { ++rejected; }
      catch (const std::out_of_range &) { ++rejected; }
  }
  need(rejected == 6, "invalid partition/extent/epsilon/alias/host pointer accepted");
  std::cout << "{\"kind\":\"rejection\",\"cases\":6,\"status\":\"PASS\"}\n";
}
void context(int partitions) {
  Case c(8192); auto &reference = *c.trials[0], &candidate = *c.trials[1];
  int total = 0;
  for (int chunk = 0; chunk < 32; ++chunk) {
    const int tokens = chunk == 31 ? 8191 : 8192;
    if (chunk == 31) {
      reference.y.reset(); reference.y16.reset(); candidate.y.reset(); candidate.y16.reset();
    }
    c.run(reference, 0, tokens); c.run(candidate, partitions, tokens);
    rt->wait(); c.compare(reference, candidate, tokens); total += tokens;
    if (chunk == 15) {
      auto snapshot = candidate.state.get();
      candidate.state.set(c.initial);
      candidate.state.set(std::vector<float>(snapshot.begin() + Pad, snapshot.end() - Pad));
      equal(reference.state, candidate.state);
    }
  }
  need(total == 262143, "context extent");
  // Check both the unchanged decode consumer and partitioned prefill at the
  // final physical position, starting from the same saved262143-token carry.
  const auto snapshot = candidate.state.get();
  c.run(reference, 0, 1, 8191);
  const size_t row = 8191;
  strata::kernels::fused_gdn_step_norm(candidate.state.data(), c.H.data() + row * C,
      c.H.data() + row * C + 16 * 128, c.H.data() + row * C + 32 * 128,
      c.G.data() + row * HV, c.B.data() + row * HV, c.Z.data() + row * V,
      c.N.data(), 1e-6f, candidate.y.data() + row * V, 16, HV, &rt->compute());
  rt->wait(); equal(reference.state, candidate.state); equal(reference.y, candidate.y);
  candidate.state.set(std::vector<float>(snapshot.begin() + Pad, snapshot.end() - Pad));
  c.run(candidate, partitions, 1, 8191);
  rt->wait(); c.compare(reference, candidate);
  c.immutable();
  std::cout << "{\"kind\":\"context\",\"tokens\":262144,\"partitions\":" << partitions
            << ",\"state_restore\":true,\"existing_decode_tail\":true,\"status\":\"PASS\"}\n";
}
void benchmark(int process) {
  Case c(8192, 5); auto &queue = rt->compute();
  struct Sample { int round, partitions; sycl::event begin, end; double submit_ms; };
  std::vector<Sample> samples;
  for (int round = -1; round < 6; ++round)
    for (int slot = 0; slot < 5; ++slot) {
      const int arm = (slot + round + 1 + process) % 5;
      auto begin = queue.ext_oneapi_submit_barrier();
      const auto start = Clock::now();
      c.run(*c.trials[arm], Arms[arm], 8192);
      const double submit = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
      auto end = queue.ext_oneapi_submit_barrier();
      samples.push_back({round, Arms[arm], begin, end, submit});
    }
  rt->wait();
  for (int arm = 1; arm < 5; ++arm) c.compare(*c.trials[0], *c.trials[arm]);
  c.immutable();
  for (auto &s : samples) {
    need(s.begin.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete &&
         s.end.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete,
         "incomplete marker");
    const auto start = s.begin.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto end = s.end.get_profiling_info<sycl::info::event_profiling::command_start>();
    need(end > start && std::isfinite(s.submit_ms) && s.submit_ms >= 0, "invalid timing interval");
    std::cout << "{\"kind\":\"sample\",\"process\":" << process << ",\"round\":" << s.round
              << ",\"partitions\":" << s.partitions << ",\"queue_interval_ms\":" << double(end - start) / 1e6
              << ",\"host_submit_ms\":" << s.submit_ms << ",\"status\":\"PASS\"}\n";
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::cout << std::setprecision(12);
    need(argc >= 2, "use --smoke|--validate|--bench PROCESS|--context PARTITIONS");
    rt = sb::runtime_for();
    const std::string mode = argv[1];
    if (mode == "--smoke" && argc == 2) { correctness(17, false); correctness(17, true); rejection(); }
    else if (mode == "--validate" && argc == 2)
      for (int tokens : {1, 33, 65, 129, 8191, 8192}) correctness(tokens, false);
    else if (mode == "--bench" && argc == 3) {
      const int process = std::stoi(argv[2]); need(process >= 0 && process < 3, "process ordinal"); benchmark(process);
    } else if (mode == "--context" && argc == 3) {
      const int partitions = std::stoi(argv[2]);
      need(partitions == 1 || partitions == 2 || partitions == 4 || partitions == 8, "context partitions");
      context(partitions);
    } else throw std::invalid_argument("invalid fixture arguments");
    rt->wait(); std::cout << "fixture_status=PASS\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
