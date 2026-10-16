// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors
// Own component diagnostic, adapted from tests/sycl/prompt_k_cellmajor.cpp.
// Query/physical-replica separation is an own source-only controlled diagnostic.
// See the accompanying proposal manifest for the immutable source/provenance.
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

using namespace strata;
using namespace strata::kernels;
namespace {
constexpr int Heads = 24, KvHeads = 2, Dim = 256, Page = 4, Guard = 16;
constexpr size_t QueryStride = Heads * Dim;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
std::shared_ptr<sycl_backend::Runtime> runtime;
void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
size_t checked_add(size_t a, size_t b) {
  require(b <= std::numeric_limits<size_t>::max() - a, "size addition overflow");
  return a + b;
}
size_t checked_mul(size_t a, size_t b) {
  require(a == 0 || b <= std::numeric_limits<size_t>::max() / a, "size product overflow");
  return a * b;
}
template<class T> size_t guarded_count(size_t count) {
  const size_t n = checked_add(count, 2 * Guard);
  (void)checked_mul(n, sizeof(T));
  return n;
}
void main_gate(const char *phase) {
  std::cout << "QSA_RESOURCE_READY " << phase << '\n';
  char byte = 0;
  ssize_t count;
  do { count = ::read(STDIN_FILENO, &byte, 1); } while (count < 0 && errno == EINTR);
  require(count == 1 && byte == 'X', "Main inspection gate not released");
}
// Every drain guard precedes destruction of asynchronous device AND host owners.
// On unknown completion, terminate without unwinding those owners.
struct Drain {
  ~Drain() noexcept {
    try { runtime->wait(); }
    catch (...) { sycl_backend::cleanup_fail_stop("QSA diagnostic drain failed"); }
  }
};
template<class T> struct Buffer {
  size_t count;
  sycl_backend::Allocation allocation;
  explicit Buffer(size_t n)
      : count(n), allocation(runtime, checked_mul(guarded_count<T>(n), sizeof(T)),
                            sycl_backend::MemoryKind::Device) {}
  T *raw() { return allocation.as<T>(); }
  T *data() { return raw() + Guard; }
  static T canary() { return T(73); }
  void put(const std::vector<T> &input) {
    require(input.size() == count, "buffer upload extent");
    std::vector<T> guarded(guarded_count<T>(count), canary());
    std::copy(input.begin(), input.end(), guarded.begin() + Guard);
    Drain drain; // Includes a partially accepted memcpy followed by an exception.
    runtime->wait(runtime->compute().memcpy(raw(), guarded.data(), allocation.size()));
  }
  std::vector<T> get() {
    std::vector<T> result(guarded_count<T>(count));
    Drain drain; // result must stay live through a failed event wait as well.
    runtime->wait(runtime->compute().memcpy(result.data(), raw(), allocation.size()));
    for (int i = 0; i < Guard; ++i)
      require(result[i] == canary() && result[Guard + count + i] == canary(),
              "buffer head/tail guard");
    return result;
  }
  void unchanged(const std::vector<T> &expected) {
    const auto actual = get();
    require(expected.size() == count &&
                std::memcmp(actual.data() + Guard, expected.data(), checked_mul(count, sizeof(T))) == 0,
            "input buffer changed");
  }
};
template<class T> bool exact(const std::vector<T> &a, const std::vector<T> &b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), checked_mul(a.size(), sizeof(T))) == 0;
}
uint16_t half(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
float decode(uint16_t x) { return float(std::bit_cast<_Float16>(x)); }
// Stable byte fingerprint for logs, explicitly NOT a cryptographic hash.
template<class T> uint64_t fingerprint(const std::vector<T> &v) {
  const auto *p = reinterpret_cast<const unsigned char *>(v.data());
  uint64_t hash = UINT64_C(14695981039346656037);
  const size_t bytes = checked_mul(v.size(), sizeof(T));
  for (size_t i = 0; i < bytes; ++i)
    hash = (hash ^ p[i]) * UINT64_C(1099511628211);
  return hash;
}
long integer(const char *s) {
  char *end = nullptr;
  errno = 0;
  const long n = std::strtol(s, &end, 10);
  require(!errno && end != s && *end == '\0', "invalid integer");
  return n;
}
struct Options {
  bool bench = false, replicated = false;
  int nq = 3, grain = 3, repetitions = 1, warmups = 2;
  std::string dump;
};
Options options(int argc, char **argv) {
  require(argc == 2 || argc == 14,
          "usage: fixture --smoke | --bench --nq 128|256|8192 --reuse shared|replicated "
          "--grain 1|8|32|whole (128/256) or 512|whole (8192) --repeats 1..20 --warmups 2..5 --dump PATH");
  Options o;
  if (argc == 2) {
    require(std::string(argv[1]) == "--smoke", "expected --smoke");
    return o;
  }
  o.bench = true;
  require(std::string(argv[1]) == "--bench" && std::string(argv[2]) == "--nq" &&
              std::string(argv[4]) == "--reuse" && std::string(argv[6]) == "--grain" &&
              std::string(argv[8]) == "--repeats" && std::string(argv[10]) == "--warmups" &&
              std::string(argv[12]) == "--dump", "benchmark argument order");
  const long nq = integer(argv[3]), reps = integer(argv[9]), warm = integer(argv[11]);
  require(nq == 128 || nq == 256 || nq == 8192, "nq must be 128, 256 or 8192");
  require(reps >= 1 && reps <= 20 && warm >= 2 && warm <= 5, "repeat/warmup bounds");
  o.nq = int(nq); o.repetitions = int(reps); o.warmups = int(warm);
  const std::string reuse(argv[5]), grain(argv[7]);
  require(reuse == "shared" || reuse == "replicated", "invalid reuse arm");
  o.replicated = reuse == "replicated";
  const long g = grain == "whole" ? nq : integer(argv[7]);
  require(nq == 8192 ? (g == 512 || g == nq)
                      : (g == 1 || g == 8 || g == 32 || g == nq),
          "invalid grain for nq: 8192 permits only 512 or whole");
  o.grain = int(g); o.dump = argv[13];
  require(!o.dump.empty(), "empty dump path");
  return o;
}
void reference_environment() {
  // Read ONCE before runtime creation. Never setenv/unsetenv in this executable.
  auto value = [](const char *key) {
    const char *v = std::getenv(key);
    return std::string(v ? v : "<unset>");
  };
  require(value("STRATA_SYCL_PROMPT_DIRECT") == "1", "requires pre-exec PROMPT_DIRECT=1");
  require(value("STRATA_SYCL_PROMPT_K_CELL_MAJOR") == "0", "requires pre-exec K_CELL_MAJOR=0");
  require(value("STRATA_SYCL_PROMPT_SOFTMAX_SUBGROUP") == "0", "requires pre-exec SOFTMAX_SUBGROUP=0");
  require(value("STRATA_SYCL_PROMPT_NATIVE_KV_MIN") == "0", "requires pre-exec NATIVE_KV_MIN=0");
  std::cout << "reference Mode=1 Direct=1 Native=false KMapping=0 Softmax=false "
               "controls=argv_once environment_mutations=0\n";
}
struct Frame {
  const int nq, nreplica, cap, cells, pages, padded_cells;
  const size_t base_rows, rows;
  std::vector<int8_t> k, v;
  std::vector<uint16_t> ks, vs;
  std::vector<int32_t> table, shared_ids, replicated_ids, steps;
  std::vector<float> query;
  Frame(int n, bool bench, int scenario)
      : nq(n), nreplica(n == 8192 ? 128 : n),
        cap(bench ? 2051 : 67), cells(bench ? 2113 : 79),
        pages((cells + Page - 1) / Page), padded_cells(pages * Page),
        base_rows(checked_mul(size_t(padded_cells), KvHeads)),
        rows(checked_mul(base_rows, size_t(nreplica))),
        k(checked_mul(rows, Dim)), v(k.size()), ks(checked_mul(rows, 4)), vs(ks.size()),
        table(checked_mul(size_t(nreplica), pages)), shared_ids(checked_mul(size_t(nq), cap)),
        replicated_ids(shared_ids.size()), steps(checked_mul(size_t(nq), kStepCount)),
        query(checked_mul(size_t(nq), QueryStride)) {
    for (size_t i = 0; i < base_rows * Dim; ++i) {
      k[i] = int8_t(int(i * 7 % 255) - 127);
      v[i] = int8_t(int(i * 11 % 255) - 127);
    }
    for (size_t i = 0; i < base_rows * 4; ++i) {
      ks[i] = half(.007f + float(i % 7) * .0001f);
      vs[i] = half(.008f + float(i % 11) * .0002f);
    }
    for (int b = 0; b < nreplica; ++b) {
      if (b) {
        std::memcpy(k.data() + size_t(b) * base_rows * Dim, k.data(), base_rows * Dim);
        std::memcpy(v.data() + size_t(b) * base_rows * Dim, v.data(), base_rows * Dim);
        std::memcpy(ks.data() + size_t(b) * base_rows * 4, ks.data(), base_rows * 4 * sizeof(uint16_t));
        std::memcpy(vs.data() + size_t(b) * base_rows * 4, vs.data(), base_rows * 4 * sizeof(uint16_t));
      }
      for (int p = 0; p < pages; ++p)
        table[b * pages + p] = scenario == 1 && p % 3 == 0 ? -1 : b * pages + pages - 1 - p;
    }
    for (int b = 0; b < nq; ++b) {
      // The prompt entrypoint consumes only kStepWidth: no ID-based RoPE,
      // positional bias, causal mask, sorting, deduplication or index selection.
      steps[b * kStepCount + kStepPos] = 4096 + b;
      steps[b * kStepCount + kStepNKv] = cells;
      steps[b * kStepCount + kStepNBid] = 0;
      steps[b * kStepCount + kStepWidth] = scenario == 1 ? (b == 0 ? 13 : b == 1 ? 31 : cap)
          : scenario == 2 ? (b == 0 ? 0 : b == 1 ? -1 : cap + 1) : cap;
      for (int j = 0; j < cap; ++j) {
        int c = (j * 17) % cells; // Same ordered cohort for every query.
        if (scenario == 1 && j % 11 == 5) c = -1;
        if (scenario == 1 && j == 6) c = shared_ids[b * cap + 4]; // Explicit duplicate.
        shared_ids[b * cap + j] = c;
        replicated_ids[b * cap + j] = c < 0 ? c : (b % nreplica) * padded_cells + c;
      }
    }
    for (size_t i = 0; i < query.size(); ++i) query[i] = float(std::sin(i * .019));
  }
  int count(int b) const {
    const int n = steps[b * kStepCount + kStepWidth];
    return n >= 0 && n <= cap ? n : 0;
  }
  int64_t row(int cell, int h) const {
    if (cell < 0) return -1;
    require(h >= 0 && h < KvHeads, "host KV head bounds");
    require(size_t(cell / Page) < table.size(), "host ID table bounds");
    const int p = table[cell / Page];
    if (p < 0) return -1;
    require(size_t(p) < checked_mul(size_t(nreplica), pages), "host physical page bounds");
    const int64_t r = (int64_t(p) * KvHeads + h) * Page + cell % Page;
    require(r >= 0 && size_t(r) < rows, "host physical row bounds");
    return r;
  }
  void proof(int scenario) const {
    // Byte-prove every physical row and scale, including padding/unselected rows.
    // Prove the complete logical->physical table, not just selected addresses.
    for (int b = 0; b < nreplica; ++b) {
      const size_t code_offset = checked_mul(checked_mul(size_t(b), base_rows), Dim);
      const size_t scale_offset = checked_mul(checked_mul(size_t(b), base_rows), 4);
      const size_t code_bytes = checked_mul(base_rows, Dim);
      const size_t scale_bytes = checked_mul(checked_mul(base_rows, 4), sizeof(uint16_t));
      require(std::memcmp(k.data(), k.data() + code_offset, code_bytes) == 0 &&
                  std::memcmp(v.data(), v.data() + code_offset, code_bytes) == 0 &&
                  std::memcmp(ks.data(), ks.data() + scale_offset, scale_bytes) == 0 &&
                  std::memcmp(vs.data(), vs.data() + scale_offset, scale_bytes) == 0,
              "full physical replica K/V/scales identity");
      for (int p = 0; p < pages; ++p) {
        const int expected = scenario == 1 && p % 3 == 0 ? -1 : b * pages + pages - 1 - p;
        require(table[size_t(b) * pages + p] == expected, "complete page table identity");
        require(expected == -1 || (expected >= 0 &&
                    size_t(expected) < checked_mul(size_t(nreplica), pages)),
                "complete page table bounds");
      }
    }
    for (float x : query) require(std::isfinite(x), "host query finite");
    for (uint16_t x : ks) require(std::isfinite(decode(x)), "host K scale finite");
    for (uint16_t x : vs) require(std::isfinite(decode(x)), "host V scale finite");
    size_t selected = 0, resident = 0, missing = 0, duplicates = 0;
    std::vector<bool> seen_shared(rows), seen_replicated(rows);
    for (int b = 0; b < nq; ++b) {
      std::vector<bool> seen_cell(cells);
      for (int j = 0; j < count(b); ++j) {
        ++selected;
        const int c = shared_ids[b * cap + j], rc = replicated_ids[b * cap + j];
        require(c == -1 || (c >= 0 && c < cells), "host logical cell bounds");
        require(c < 0 ? rc == c : rc == (b % nreplica) * padded_cells + c, "ID translation proof");
        if (c >= 0) { duplicates += seen_cell[c]; seen_cell[c] = true; }
        for (int h = 0; h < KvHeads; ++h) {
          const int64_t a = row(c, h), r = row(rc, h);
          require((a < 0) == (r < 0), "missing page equivalence");
          if (a < 0) { if (h == 0) ++missing; continue; }
          require(r == int64_t(size_t(b % nreplica) * base_rows) + a, "replica physical mapping proof");
          require(std::memcmp(k.data() + a * Dim, k.data() + r * Dim, Dim) == 0 &&
                      std::memcmp(v.data() + a * Dim, v.data() + r * Dim, Dim) == 0 &&
                      std::memcmp(ks.data() + a * 4, ks.data() + r * 4, 8) == 0 &&
                      std::memcmp(vs.data() + a * 4, vs.data() + r * 4, 8) == 0,
                  "selected K/V/scales bit identity");
          seen_shared[a] = true; seen_replicated[r] = true;
          if (h == 0) ++resident;
        }
      }
    }
    const size_t us = std::count(seen_shared.begin(), seen_shared.end(), true);
    const size_t ur = std::count(seen_replicated.begin(), seen_replicated.end(), true);
    if (cap == 2051)
      require(selected == size_t(nq) * cap && resident == selected && missing == 0 && duplicates == 0 &&
                  us == size_t(cap) * KvHeads && ur == checked_mul(us, size_t(nreplica)),
              "benchmark uniform resident unique cohort contract");
    std::cout << "host_model_proof=PASS selected_slots=" << selected << " resident_slots=" << resident
              << " missing_slots=" << missing << " duplicate_slots=" << duplicates
              << " logical_selected_rows=" << resident * KvHeads
              << " unique_shared_rows=" << us << " unique_replicated_rows=" << ur
              << " shared_selected_bytes=" << us * (2 * Dim + 2 * 4 * sizeof(uint16_t))
              << " replicated_selected_bytes=" << ur * (2 * Dim + 2 * 4 * sizeof(uint16_t))
              << " logical_reuse_shared=" << (us ? double(resident * KvHeads) / us : 0)
              << " logical_reuse_replicated=" << (ur ? double(resident * KvHeads) / ur : 0) << '\n';
  }
  double oracle(const std::vector<float> &output, bool sampled) const {
    double worst = 0;
    for (int b = 0; b < nq; ++b) {
      if (sampled && b != 0 && b != 1 && b != 2 && b != nq - 1) continue;
      for (int h = 0; h < Heads; ++h) {
        std::vector<double> weights(count(b));
        double maximum = -INFINITY, sum = 0;
        for (int j = 0; j < count(b); ++j) {
          const auto r = row(shared_ids[b * cap + j], h / 12);
          double dot = 0;
          if (r >= 0) for (int d = 0; d < Dim; ++d)
            dot += double(query[(b * Heads + h) * Dim + d]) *
                   (float(k[r * Dim + d]) * decode(ks[r * 4 + d / 64]));
          weights[j] = r < 0 ? -INFINITY : dot / 16.;
          maximum = std::max(maximum, weights[j]);
        }
        for (auto &w : weights) { w = std::isfinite(w) ? std::exp(w - maximum) : 0.; sum += w; }
        for (int d = 0; d < Dim; ++d) {
          double expected = 0;
          for (int j = 0; j < count(b); ++j) {
            const auto r = row(shared_ids[b * cap + j], h / 12);
            if (r >= 0) expected += weights[j] *
                (float(v[r * Dim + d]) * decode(vs[r * 4 + d / 64]));
          }
          if (sum) expected /= sum;
          const float actual = output[Guard + (b * Heads + h) * Dim + d];
          const double error = std::abs(actual - expected);
          worst = std::max(worst, error);
          require(std::isfinite(actual) && error < 1e-5 * (1 + std::abs(expected)), "FP64 oracle");
        }
      }
    }
    return worst;
  }
};
void finite(const std::vector<float> &v, size_t n) {
  for (size_t i = 0; i < n; ++i) require(std::isfinite(v[Guard + i]), "nonfinite output");
}
double cpu_seconds() {
  timespec t{};
  require(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) == 0, "thread CPU clock unavailable");
  return double(t.tv_sec) + double(t.tv_nsec) * 1e-9;
}
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
// Exclusive creation keeps the canonical frame intact. Failed partial files are
// left for Main's recorded retention review; no overwrite/unlink is attempted.
struct DumpFile {
  int fd;
  explicit DumpFile(const std::string &path)
      : fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600)) {
    require(fd >= 0, "dump exclusive open (path must not exist)");
  }
  ~DumpFile() { if (fd >= 0) ::close(fd); }
  void write(const void *data, size_t bytes) {
    const auto *p = static_cast<const char *>(data);
    while (bytes) {
      const size_t chunk = std::min(bytes, size_t(1024 * 1024));
      const ssize_t done = ::write(fd, p, chunk);
      if (done < 0 && errno == EINTR) continue;
      require(done > 0, "dump write");
      p += done; bytes -= size_t(done);
    }
  }
  void close_checked() {
    const int owned = fd;
    fd = -1; // Linux close errors must not be retried against a reused descriptor.
    require(::close(owned) == 0, "dump close");
  }
};
void write_dump(const std::string &path, const Frame &f, const std::vector<float> &y) {
  // One initial frame only: max 192 MiB payload + 24 bytes, budget 193 MiB.
  const char magic[8] = {'Q','R','C','V','1','L','E','\0'};
  require(std::endian::native == std::endian::little, "dump requires little endian");
  const uint32_t nq = f.nq, cap = f.cap;
  const uint64_t count = f.query.size();
  const size_t payload = checked_mul(f.query.size(), sizeof(float));
  require(checked_add(24, payload) <= size_t(193) * 1024 * 1024, "dump budget");
  require(y.size() == guarded_count<float>(f.query.size()), "dump output extent");
  DumpFile out(path);
  out.write(magic, 8); out.write(&nq, 4); out.write(&cap, 4); out.write(&count, 8);
  out.write(y.data() + Guard, payload);
  out.close_checked(); // All writes and the final close must succeed.
}
void run(const Options &o, int scenario) {
  Frame f(o.nq, o.bench, scenario);
  f.proof(scenario); // Actual selected byte comparisons, not a construction-only assertion.
  std::cout << "raw_widths=";
  if (scenario == 0) std::cout << f.cap << "_for_all_" << f.nq;
  else for (int b = 0; b < f.nq; ++b)
    std::cout << (b ? "," : "") << f.steps[b * kStepCount + kStepWidth];
  std::cout << '\n';
  std::cout << "case=" << scenario << " nq=" << f.nq << " cap=" << f.cap
            << " width=" << (scenario == 0 ? "uniform" : "smoke_variable")
            << " nreplica=" << f.nreplica
            << " pages_per_replica=" << f.pages << " physical_pages=" << f.pages * f.nreplica
            << " reuse=" << (o.replicated ? "replicated" : "shared") << " grain=" << o.grain
            << " calls_per_work=" << (f.nq + o.grain - 1) / o.grain
            << " workgroups_total=" << size_t(f.nq) * KvHeads
            << " workgroups_per_full_call=" << size_t(o.grain) * KvHeads
            << " warmups=" << o.warmups << " repetitions=" << o.repetitions
            << " fingerprints=FNV1a64_native_bytes"
            << " k=" << fingerprint(f.k) << " v=" << fingerprint(f.v)
            << " ks=" << fingerprint(f.ks) << " vs=" << fingerprint(f.vs)
            << " query=" << fingerprint(f.query) << " table=" << fingerprint(f.table)
            << " shared_ids=" << fingerprint(f.shared_ids) << " replicated_ids=" << fingerprint(f.replicated_ids)
            << " steps=" << fingerprint(f.steps) << '\n';
  // Both arms allocate and fully initialize ALL replicas and BOTH ID arrays.
  Buffer<int8_t> k(f.k.size()), v(f.v.size());
  Buffer<uint16_t> ks(f.ks.size()), vs(f.vs.size());
  Buffer<int32_t> table(f.table.size()), si(f.shared_ids.size()), ri(f.replicated_ids.size()), steps(f.steps.size());
  Buffer<float> q(f.query.size()), out(f.query.size());
  Drain drain; // Declared after every allocation; covers partial submit throws.
  size_t device_bytes = 0, host_frame_bytes = 0;
  auto account = [&](const auto &buffer) {
    device_bytes = checked_add(device_bytes, buffer.allocation.size());
  };
  account(k); account(v); account(ks); account(vs); account(table);
  account(si); account(ri); account(steps); account(q); account(out);
  auto host_account = [&](const auto &vector) {
    host_frame_bytes = checked_add(host_frame_bytes,
        checked_mul(vector.size(), sizeof(vector[0])));
  };
  host_account(f.k); host_account(f.v); host_account(f.ks); host_account(f.vs);
  host_account(f.table); host_account(f.shared_ids); host_account(f.replicated_ids);
  host_account(f.steps); host_account(f.query);
  require(device_bytes < size_t(1024) * 1024 * 1024, "device allocation budget exceeds 1 GiB");
  std::cout << "device_bytes=" << device_bytes << " host_frame_payload_bytes=" << host_frame_bytes
            << " footprint_and_initialization=identical_across_arms\n";
  k.put(f.k); v.put(f.v); ks.put(f.ks); vs.put(f.vs); table.put(f.table);
  si.put(f.shared_ids); ri.put(f.replicated_ids); steps.put(f.steps); q.put(f.query);
  out.put(std::vector<float>(f.query.size(), 1234.f));
  QsaAttnPools pools{};
  pools.k_q = k.data(); pools.v_q = v.data(); pools.k_scale = ks.data();
  pools.v_scale = vs.data(); pools.page_table = table.data();
  auto shape = qsa_real_shapes(); shape.page_size = Page;
  auto issue = [&](bool replicated, int grain) {
    for (int b = 0; b < f.nq; b += grain)
      require(qsa_prompt_attn_batch(q.data() + size_t(b) * QueryStride, pools,
                  (replicated ? ri.data() : si.data()) + size_t(b) * f.cap,
                  steps.data() + size_t(b) * kStepCount, f.cap, shape,
                  out.data() + size_t(b) * QueryStride, std::min(grain, f.nq - b),
                  &runtime->compute()), "prompt matrix entrypoint unavailable; no fallback");
  };
  // Shared/whole is a correctness reference only; all times below concern argv arm.
  issue(false, f.nq); runtime->wait();
  const auto reference = out.get(); finite(reference, f.query.size());
  const double worst = f.oracle(reference, o.bench);
  if (o.bench) {
    out.put(std::vector<float>(f.query.size(), 1234.f));
    issue(o.replicated, o.grain); runtime->wait();
  } else {
    // Three finite cases x two address arms x three grains = 18 comparisons.
    for (bool replicated : {false, true}) for (int grain : {1, 2, f.nq}) {
      out.put(std::vector<float>(f.query.size(), 1234.f));
      issue(replicated, grain); runtime->wait();
      require(exact(reference, out.get()), "smoke cross-arm/grain output bits");
    }
  }
  const auto initial = out.get(); finite(initial, f.query.size());
  require(exact(reference, initial), "initial full cross-arm output bits");
  if (o.bench) write_dump(o.dump, f, initial);
  if (o.bench) {
    for (int w = 0; w < o.warmups; ++w) { issue(o.replicated, o.grain); runtime->wait(); }
    // No allocation, transfer, oracle, dump or logging inside any measured window.
    // No readback between repeats: only the final repeated frame is validated.
    for (int r = 0; r < o.repetitions; ++r) {
      const double c0 = cpu_seconds(); const auto t0 = Clock::now();
      issue(o.replicated, o.grain);
      const auto t1 = Clock::now(); const double c1 = cpu_seconds();
      runtime->wait();
      const double c2 = cpu_seconds(); const auto t2 = Clock::now();
      // Emit outside the measured window, retaining completed samples even if
      // a later repeat fails. Qualification remains pending the final checks.
      std::cout << "sample=" << r << " enqueue_wall_ms=" << milliseconds(t0, t1)
                << " wait_wall_ms=" << milliseconds(t1, t2) << " whole_wall_ms=" << milliseconds(t0, t2)
                << " enqueue_thread_cpu_ms=" << (c1-c0)*1000 << " wait_thread_cpu_ms=" << (c2-c1)*1000
                << " whole_thread_cpu_ms=" << (c2-c0)*1000 << " validation=PENDING_FINAL_CHECKS\n";
    }
    const auto final = out.get(); finite(final, f.query.size());
    require(exact(initial, final), "final repeated full output bits");
    std::cout << "final_output_fnv1a64_with_guards=" << fingerprint(final) << '\n';
  }
  // Full post-drain input checks, including guards and unused replica backing.
  k.unchanged(f.k); v.unchanged(f.v); ks.unchanged(f.ks); vs.unchanged(f.vs);
  table.unchanged(f.table); si.unchanged(f.shared_ids); ri.unchanged(f.replicated_ids);
  steps.unchanged(f.steps); q.unchanged(f.query);
  require(exact(reference, out.get()), "post-input-check output bits/guards");
  std::cout << "case=" << scenario << " status=PASS host_model_identity=PASS full_finite=PASS"
            << " input_and_guards=PASS cross_arm_bits=PASS oracle="
            << (o.bench ? "FP64_queries_0_1_2_last" : "FP64_full_small") << " max_abs=" << worst
            << " gpu_events=NOT_EXPOSED wall_times=HOST_ONLY cpu_times=ISSUER_THREAD"
            << " per_repeat_numerical=FINAL_FRAME_ONLY independent_processes=EXTERNAL_REQUIRED"
            << " physical262144=NOT_TESTED RESTORE=NOT_TESTED model=NOT_TESTED"
            << " physical_traffic=NOT_MEASURED occupancy=NOT_MEASURED utilization=NOT_MEASURED\n";
}
} // namespace
int main(int argc, char **argv) {
  try {
    const Options o = options(argc, argv); // All arm controls fixed before runtime.
    reference_environment();
    const char *gate_control = std::getenv("STRATA_TEST_MAIN_GATES");
    require(!gate_control || std::string(gate_control) == "0" ||
                std::string(gate_control) == "1", "invalid Main gate control");
    const bool gates = gate_control && std::string(gate_control) == "1";
    std::cout << std::setprecision(12) << std::unitbuf;
    if (gates) main_gate("before-runtime");
    runtime = sycl_backend::runtime_for();
    require(runtime->compute().has_property<sycl::property::queue::in_order>(), "compute queue not in-order");
    std::cout << "device=" << runtime->device().get_info<sycl::info::device::name>()
              << " driver=" << runtime->device().get_info<sycl::info::device::driver_version>() << '\n';
    if (gates) main_gate("runtime-ready");
    if (o.bench) run(o, 0);
    else for (int scenario = 0; scenario < 3; ++scenario) run(o, scenario);
    runtime->wait();
    if (gates) main_gate("after-cleanup");
    std::cout << "fixture_status=PASS performance_adopted=false\n";
    return 0;
  } catch (const std::exception &e) {
    if (runtime) {
      try { runtime->wait(); }
      catch (...) { sycl_backend::cleanup_fail_stop("QSA diagnostic final failure drain"); }
    }
    std::cerr << "fixture_status=FAIL reason=" << e.what() << '\n';
    return 1;
  }
}
