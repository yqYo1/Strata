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
#include <cstdlib>
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
void admission(const char *phase) {
  const char *setting = std::getenv("STRATA_TEST_MAIN_GATES");
  if (setting && std::string(setting) == "1") {
    std::cout << "COMPONENT_READY " << phase << '\n' << std::flush;
    need(std::cin.get() == 'X', "Main component admission gate");
  }
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
  void run(Trial &trial, int partitions, int tokens, int offset = 0, bool barriers3 = false) {
    need(tokens > 0 && offset >= 0 && offset + tokens <= capacity, "fixture run extent");
    if (barriers3)
      p::gdn_recurrence_barriers3(trial.state.data(), H.data() + size_t(offset) * C,
          G.data() + size_t(offset) * HV, B.data() + size_t(offset) * HV, Z.data() + size_t(offset) * V,
          N.data(), 1e-6f, trial.y.data() + size_t(offset) * V, trial.y16.data() + size_t(offset) * V,
          tokens, &rt->compute());
    else if (partitions)
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
void barriers3_correctness(int tokens, bool split) {
  Case c(tokens, 3);
  c.run(*c.trials[0], 0, tokens);
  for (int arm = 1; arm < 3; ++arm) {
    if (split) {
      c.run(*c.trials[arm], 4, 5, 0, arm == 2);
      c.run(*c.trials[arm], 4, tokens - 5, 5, arm == 2);
    } else c.run(*c.trials[arm], 4, tokens, 0, arm == 2);
  }
  rt->wait();
  c.compare(*c.trials[0], *c.trials[1]);
  c.compare(*c.trials[0], *c.trials[2]);
  c.immutable();
  std::cout << "{\"kind\":\"validation\",\"schedule\":\"barriers3\",\"tokens\":" << tokens
            << ",\"split\":" << (split ? "true" : "false")
            << ",\"exact_vs_fused_and_columns4\":true,\"guards\":true,\"status\":\"PASS\"}\n";
}
void barriers3_rejection() {
  Case c(1); auto &v = *c.trials[1]; int rejected = 0;
  for (int mode = 0; mode < 7; ++mode) {
    try {
      float host = 0;
      p::gdn_recurrence_barriers3(v.state.data(), mode == 5 ? &host : mode == 6 ? nullptr : c.H.data(),
          c.G.data(), c.B.data(), c.Z.data(), c.N.data(), mode == 3 ? NAN : 1e-6f,
          mode == 4 ? v.state.data() : v.y.data(), v.y16.data(),
          mode == 0 ? -1 : mode == 1 ? 0 : mode == 2 ? 8193 : 1, &rt->compute());
    } catch (const std::invalid_argument &) { ++rejected; }
      catch (const std::out_of_range &) { ++rejected; }
    rt->wait();
    equal(c.trials[0]->state, v.state);
    equal(c.trials[0]->y, v.y);
    equal(c.trials[0]->y16, v.y16);
  }
  need(rejected == 7, "invalid barriers3 geometry/epsilon/alias/host/null pointer accepted");
  c.immutable();
  std::cout << "{\"kind\":\"rejection\",\"schedule\":\"barriers3\",\"cases\":7,"
               "\"no_write\":true,\"status\":\"PASS\"}\n";
}
void context(int partitions, bool barriers3 = false) {
  Case c(8192); auto &reference = *c.trials[0], &candidate = *c.trials[1];
  int total = 0;
  for (int chunk = 0; chunk < 32; ++chunk) {
    const int tokens = chunk == 31 ? 8191 : 8192;
    if (chunk == 31) {
      reference.y.reset(); reference.y16.reset(); candidate.y.reset(); candidate.y16.reset();
    }
    c.run(reference, 0, tokens); c.run(candidate, partitions, tokens, 0, barriers3);
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
  c.run(candidate, partitions, 1, 8191, barriers3);
  rt->wait(); c.compare(reference, candidate);
  c.immutable();
  std::cout << "{\"kind\":\"context\",\"tokens\":262144,\"partitions\":" << partitions
            << (barriers3 ? ",\"schedule\":\"barriers3\"" : "")
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
void barriers3_benchmark(int process) {
  constexpr std::array<const char *, 3> names{"fused", "columns4", "barriers3"};
  Case c(8192, 3); auto &queue = rt->compute();
  const auto resolution = queue.get_device().get_info<sycl::info::device::profiling_timer_resolution>();
  need(resolution > 0, "unknown device profiling timer resolution");
  for (int control = 0; control < 8; ++control) {
    auto begin = queue.ext_oneapi_submit_barrier();
    auto end = queue.ext_oneapi_submit_barrier();
    rt->wait(end);
    const auto a = begin.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto b = end.get_profiling_info<sycl::info::event_profiling::command_start>();
    need(b >= a, "invalid empty marker clock");
    std::cout << "{\"kind\":\"timer_control\",\"process\":" << process << ",\"control\":" << control
              << ",\"resolution_ns\":" << resolution << ",\"empty_marker_gap_ns\":" << b - a
              << ",\"status\":\"PASS\"}\n";
  }
  struct Sample { int round, arm; sycl::event begin, end; double submit_ms; };
  std::vector<Sample> samples;
  for (int round = -1; round < 6; ++round)
    for (int slot = 0; slot < 3; ++slot) {
      const int arm = (slot + round + 1 + process) % 3;
      auto begin = queue.ext_oneapi_submit_barrier();
      const auto start = Clock::now();
      c.run(*c.trials[arm], arm == 0 ? 0 : 4, 8192, 0, arm == 2);
      const double submit = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
      auto end = queue.ext_oneapi_submit_barrier();
      samples.push_back({round, arm, begin, end, submit});
    }
  rt->wait();
  c.compare(*c.trials[0], *c.trials[1]); c.compare(*c.trials[0], *c.trials[2]); c.immutable();
  std::cout << "{\"kind\":\"train_validation\",\"process\":" << process
            << ",\"tokens_per_call\":8192,\"updates_per_arm\":7,"
               "\"final_state_output_fp16_exact\":true,\"guards\":true,\"inputs_immutable\":true,"
               "\"status\":\"PASS\"}\n";
  for (auto &s : samples) {
    need(s.begin.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete &&
         s.end.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete,
         "incomplete barriers3 marker");
    const auto start = s.begin.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto end = s.end.get_profiling_info<sycl::info::event_profiling::command_start>();
    need(end > start && end - start >= 1000000 && double(end - start) / double(resolution) >= 1000 &&
         std::isfinite(s.submit_ms) && s.submit_ms >= 0, "invalid/sub-millisecond/coarse barriers3 interval");
    std::cout << "{\"kind\":\"sample\",\"process\":" << process << ",\"round\":" << s.round
              << ",\"schedule\":\"" << names[s.arm] << "\",\"queue_interval_ms\":" << double(end - start) / 1e6
              << ",\"host_submit_ms\":" << s.submit_ms << ",\"status\":\"PASS\"}\n";
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::cout << std::setprecision(12);
    need(argc >= 2, "use --smoke|--validate|--bench PROCESS|--context PARTITIONS|"
                    "--barriers3-smoke|--barriers3-validate|--barriers3-context|--barriers3-bench PROCESS");
    admission("before-runtime");
    rt = sb::runtime_for();
    admission("runtime-ready");
    const std::string mode = argv[1];
    if (mode == "--smoke" && argc == 2) { correctness(17, false); correctness(17, true); rejection(); }
    else if (mode == "--barriers3-smoke" && argc == 2) {
      barriers3_correctness(17, false); barriers3_correctness(17, true); barriers3_rejection();
    } else if (mode == "--barriers3-validate" && argc == 2) {
      for (int tokens : {1, 33, 65, 129, 8191, 8192}) barriers3_correctness(tokens, false);
    } else if (mode == "--barriers3-context" && argc == 2) context(4, true);
    else if (mode == "--barriers3-bench" && argc == 3) {
      const int process = std::stoi(argv[2]); need(process >= 0 && process < 3, "process ordinal");
      barriers3_benchmark(process);
    }
    else if (mode == "--validate" && argc == 2)
      for (int tokens : {1, 33, 65, 129, 8191, 8192}) correctness(tokens, false);
    else if (mode == "--bench" && argc == 3) {
      const int process = std::stoi(argv[2]); need(process >= 0 && process < 3, "process ordinal"); benchmark(process);
    } else if (mode == "--context" && argc == 3) {
      const int partitions = std::stoi(argv[2]);
      need(partitions == 1 || partitions == 2 || partitions == 4 || partitions == 8, "context partitions");
      context(partitions);
    } else throw std::invalid_argument("invalid fixture arguments");
    rt->wait();
    admission("before-exit");
    std::cout << "fixture_status=PASS\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
