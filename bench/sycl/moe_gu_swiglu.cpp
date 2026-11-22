// Main-authored GU/SwiGLU qualification; repository LICENSE (MIT) applies.
#include "native16.hpp"
#include "strata/prefill/native16.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/sycl/command_profile.hpp"
#include "strata/sycl/cpu_wait_profile.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
namespace xe = strata::experimental::vllm_xe2_native16;
constexpr int N = 1280, K = 2560, H = 640;
constexpr size_t Pad = 64;
constexpr uint16_t Sentinel = 0xa55a;
std::shared_ptr<sb::Runtime> runtime;
void need(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
void drain(sycl::queue &q) noexcept {
  try { q.wait_and_throw(); runtime->check(); }
  catch (...) { std::_Exit(86); }
}
void wait(sycl::queue &q, sycl::event e) {
  try {
    sb::cpu_wait_profile::blocking(sb::cpu_wait_profile::event,[&] { e.wait_and_throw(); });
    runtime->check();
  } catch (...) { drain(q); throw; }
}
template <class Submit> void submit_wait(sycl::queue &q,Submit submit) {
  // Keep local host vectors alive through synchronous enqueue failures too.
  try { wait(q,submit()); } catch (...) { drain(q); throw; }
}
struct Buffer {
  sb::Allocation owner;
  size_t n;
  sycl::queue &q;
  Buffer(size_t count, sycl::queue &queue)
      : owner(runtime, (count + 2 * Pad) * 2, sb::MemoryKind::Device), n(count), q(queue) {
    std::vector<uint16_t> all(n + 2 * Pad, Sentinel);
    submit_wait(q,[&] { return q.memcpy(owner.data(), all.data(), all.size() * 2); });
  }
  ~Buffer() { drain(q); }
  uint16_t *data() const { return owner.as<uint16_t>() + Pad; }
  void set(const std::vector<uint16_t> &v) {
    need(v.size() == n, "upload extent");
    submit_wait(q,[&] { return q.memcpy(data(), v.data(), n * 2); });
  }
  std::vector<uint16_t> get() const {
    std::vector<uint16_t> all(n + 2 * Pad);
    submit_wait(q,[&] { return q.memcpy(all.data(), owner.data(), all.size() * 2); });
    for (size_t i = 0; i < Pad; ++i)
      need(all[i] == Sentinel && all[n + Pad + i] == Sentinel, "buffer guard changed");
    return {all.begin() + Pad, all.begin() + Pad + n};
  }
  void reset() { set(std::vector<uint16_t>(n, Sentinel)); }
};
uint16_t half(float v) { return sycl::bit_cast<uint16_t>(sycl::half(v)); }
float widen(uint16_t v) { return float(sycl::bit_cast<sycl::half>(v)); }
bool nan16(uint16_t v) { return (v & 0x7c00) == 0x7c00 && (v & 0x3ff); }
int ordered(uint16_t v) { return 0x8000 + (v & 0x8000 ? -int(v & 0x7fff) : int(v)); }
int compare_hidden(const Buffer &a, const Buffer &b) {
  const auto x = a.get(), y = b.get(); int max_ulp = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    if (nan16(x[i])) need(nan16(y[i]), "hidden NaN classification mismatch");
    else {
      need(!nan16(y[i]) && std::isfinite(widen(x[i])) && std::isfinite(widen(y[i])),
           "hidden finite saturation mismatch");
      max_ulp = std::max(max_ulp, std::abs(ordered(x[i]) - ordered(y[i])));
    }
  }
  need(max_ulp <= 1, "hidden exceeds prospective one-FP16-ULP bound");
  return max_ulp;
}
void ownership() {
  for (auto layout : {xe::GuLayout::matrix, xe::GuLayout::output_copy})
    for (auto tile : {xe::Tile::compact_16x64, xe::Tile::wide_128x128})
      for (int rows : {1, 3, 17, 33, 129, 2334, 5963, 8191, 8192}) {
        const auto p = xe::inspect_gu_pair_ownership(rows, tile, layout);
        need(p.gu_values == size_t(rows) * N && p.hidden_values == size_t(rows) * H &&
             !p.gu_missing && !p.gu_multiple && !p.hidden_missing && !p.hidden_multiple &&
             p.checked_pairs == p.hidden_values, "GU ownership incomplete");
        std::cout << "case gu_ownership rows=" << rows
          << " tile=" << (tile == xe::Tile::compact_16x64 ? "compact" : "wide")
          << " layout=" << (layout == xe::GuLayout::matrix ? "matrix" : "output_copy")
          << " gu_values=" << p.gu_values << " hidden_values=" << p.hidden_values
          << " masked=" << p.masked << " checked_pairs=" << p.checked_pairs
          << " partner_mismatch=" << p.partner_mismatch
          << " packed_missing=" << p.packed_missing << " packed_multiple=" << p.packed_multiple
          << " packed_orientation_mismatch=" << p.packed_orientation_mismatch
          << " lane_xor1_same_index_eligible=" << (p.partner_mismatch == 0) << " status=PASS\n";
      }
}
class Topology;
void topology(sycl::queue &q) {
  Buffer checked(384, q); auto *p = checked.data();
  wait(q, q.parallel_for<Topology>(sycl::nd_range<1>(384, 128), [=](sycl::nd_item<1> item)
      [[sycl::reqd_sub_group_size(16)]] {
    auto sg = item.get_sub_group();
    const auto id = item.get_local_linear_id();
    p[item.get_global_linear_id()] = uint16_t(sg.get_local_linear_range() == 16 &&
        sg.get_local_linear_id() == id % 16 && sg.get_group_linear_id() == id / 16);
  }));
  for (auto value : checked.get()) need(value == 1, "TLA/local/subgroup lane mapping mismatch");
  std::cout << "{\"kind\":\"topology\",\"work_items\":384,\"status\":\"PASS\"}\n";
}

#ifdef STRATA_PREFILL_GU_A_PROBE
void prefill_gu_a_probe(sycl::queue &q, xe::Tile tile, int rows, bool invalid) {
  constexpr int map_offset = 7, meta_pad = 16;
  const bool wide = tile == xe::Tile::wide_128x128;
  const int bank_rows = rows * 2 + 7;
  const auto layout = xe::prefill_gu_a_layout(rows, tile);
  const size_t words = layout.size();
  const size_t status_index = meta_pad + map_offset + rows + meta_pad;
  std::vector<int32_t> meta(status_index + 1 + meta_pad, -777);
  for (int r = 0; r < rows; ++r) {
    // Exercise sorted expert rows, then a reversed map with both corrupt endpoints.
    const int mapped = invalid ? 2 * (rows - 1 - r) + 1 : 2 * r + 3;
    meta[meta_pad + map_offset + r] = mapped;
  }
  if (invalid) {
    meta[meta_pad + map_offset] = -1;
    meta[meta_pad + map_offset + rows - 1] = bank_rows;
  }
  const size_t bank_values = size_t(bank_rows) * K;
  std::vector<uint16_t> bank_bits(bank_values + 2, Sentinel), dense_bits(size_t(rows) * K);
  Buffer bank(bank_bits.size(), q), dense(dense_bits.size(), q);
  sb::Allocation metadata(runtime, meta.size() * sizeof(int32_t), sb::MemoryKind::Device);
  sb::Allocation capture(runtime, (words + 2 * Pad) * sizeof(xe::PrefillGuAWord), sb::MemoryKind::Device);
  std::vector<xe::PrefillGuAWord> captured(words + 2 * Pad);
  std::vector<int32_t> observed_meta(meta.size());
  const xe::PrefillGuAWord poison{UINT32_MAX, Sentinel, Sentinel};
  auto *bank_input = bank.data() + 2;
  need(uintptr_t(bank_input) % 64 == 4, "mapped input lost 4-byte offset");
  xe::PrefillGuAProbe p{rows, bank_rows, map_offset, dense.data(), bank_input,
      metadata.as<int32_t>() + meta_pad, capture.as<xe::PrefillGuAWord>() + Pad,
      metadata.as<uint32_t>() + status_index, words};

  // The two FP16 bit tags jointly identify the entire bank cell, including
  // NaN payloads. Compare both registers with an independently packed bank.
  for (int high : {0, 1}) {
    for (size_t i = 0; i < bank_values; ++i)
      bank_bits[2 + i] = high ? uint16_t((i >> 16) ^ 0xbefd) : uint16_t(i * 40503 + 123);
    for (int r = 0; r < rows; ++r) {
      const int from = meta[meta_pad + map_offset + r];
      auto out = dense_bits.begin() + size_t(r) * K;
      if (from < 0 || from >= bank_rows) std::fill_n(out, K, uint16_t(0));
      else std::copy_n(bank_bits.begin() + 2 + size_t(from) * K, K, out);
    }
    bank.set(bank_bits);
    dense.set(dense_bits);
    std::fill(captured.begin(), captured.end(), poison);
    meta[status_index] = 32;
    submit_wait(q, [&] { return q.memcpy(metadata.data(), meta.data(), meta.size() * sizeof(int32_t)); });
    submit_wait(q, [&] { return q.memcpy(capture.data(), captured.data(), captured.size() * sizeof(poison)); });
    submit_wait(q, [&] { return xe::probe_prefill_gu_a(*runtime, q, tile, p); });
    submit_wait(q, [&] { return q.memcpy(captured.data(), capture.data(), captured.size() * sizeof(poison)); });
    submit_wait(q, [&] { return q.memcpy(observed_meta.data(), metadata.data(), meta.size() * sizeof(int32_t)); });
    meta[status_index] = invalid ? 36 : 32;
    need(observed_meta == meta, "prefill GU A map, sticky status or guard changed");
    const auto guard_ok = [&](const auto &v) {
      return v.cell == poison.cell && v.dense_bits == poison.dense_bits && v.mapped_bits == poison.mapped_bits;
    };
    need(std::all_of(captured.begin(), captured.begin() + Pad, guard_ok) &&
         std::all_of(captured.end() - Pad, captured.end(), guard_ok), "prefill GU A capture guard");
    for (size_t i = 0; i < words; ++i) {
      const auto &v = captured[Pad + i];
      need(v.cell == layout[i], "prefill GU A device lane differs from host layout");
      const int row = int(v.cell / K), col = int(v.cell % K);
      const uint16_t expected = row < rows ? dense_bits[size_t(row) * K + col] : 0;
      need(v.dense_bits == expected && v.mapped_bits == expected, "prefill GU A register-bit mismatch");
    }
    need(bank.get() == bank_bits && dense.get() == dense_bits, "prefill GU A source or guard changed");
  }
  std::cout << "prefill_gu_a rows=" << rows << " tile=" << (wide ? "wide" : "compact")
            << " invalid=" << invalid << " checked_words=" << 2 * words
            << " block2d_reorder=exact mapped=exact coverage=full guards=PASS\n";
}
#endif
struct Case {
  int rows;
  sycl::queue &q;
  xe::Gemm gemm;
  Buffer x, w, gu, capture, reference, candidate;
  std::vector<uint16_t> hx, hw;
  bool edges;
  Case(int r, bool edge_inputs, sycl::queue &queue)
      : rows(r), q(queue), gemm(*runtime, q), x(size_t(r)*K,q), w(size_t(N)*K,q),
        gu(size_t(r)*N,q), capture(gu.n,q), reference(size_t(r)*H,q), candidate(reference.n,q),
        hx(x.n), hw(w.n), edges(edge_inputs) {
    for (size_t i=0;i<hx.size();++i)
      hx[i]=half(edges ? (i%K == 0 ? 1.f : 0.f) : float(int((i*13+i/K)%17)-8)/32.f);
    constexpr std::array<uint16_t,24> values{
      0x3c04,0x3c0c,0xbc04,0xbc0c,0x7bff,0x7bff,0x0001,0x0001,
      0x0000,0x8000,0x7c00,0x3c00,0xfc00,0x3c00,0x7e01,0x3c00,
      0x3c00,0x7e01,0x4c00,0x7bff,0xcc00,0x7bff,0x7bff,0xfbff};
    for (size_t i=0;i<hw.size();++i)
      hw[i]=edges ? (i%K == 0 ? values[(i/K)%values.size()] : 0) :
          half(float(int((i*7+i/K)%23)-11)/64.f);
    x.set(hx); w.set(hw);
  }
  ~Case() { drain(q); }
  xe::GuSwiGLUProblem problem(bool observe=false) {
    return {x.data(),w.data(),candidate.data(),rows,observe ? capture.data() : nullptr};
  }
  sycl::event fused(xe::Tile tile, bool observe=false) { return gemm.gu_swiglu(problem(observe),tile); }
  void separate(xe::Tile tile) {
    try {
      xe::Problem p; p.x=x.data(); p.w=w.data(); p.y=gu.data(); p.t=rows; p.n=N; p.k=K;
      (void)gemm.beta0(p,tile);
      strata::prefill::native16::swiglu(gu.data(),reference.data(),rows,&q);
    } catch (...) { drain(q); throw; }
  }
  int check(xe::Tile tile) {
    candidate.reset(); capture.reset();
    separate(tile); wait(q,fused(tile,true));
    auto a=gu.get(),b=capture.get();
    need(a==b,"BF16 capture differs from dense DPAS output");
    if (!edges && rows<=3) {
      for (int r=0;r<rows;++r) for (int c=0;c<N;++c) {
        int dot=0;
        for (int k=0;k<K;++k)
          dot+=(int((size_t(r*K+k)*13+r)%17)-8)*(int((size_t(c*K+k)*7+c)%23)-11);
        const uint32_t bits=std::bit_cast<uint32_t>(float(dot)/2048.f);
        need(a[size_t(r)*N+c]==uint16_t((bits+0x7fff+((bits>>16)&1))>>16),"independent integer dot oracle");
      }
    }
    int ulp=compare_hidden(reference,candidate);
    candidate.reset(); capture.reset(); wait(q,fused(tile));
    ulp=std::max(ulp,compare_hidden(reference,candidate));
    for (auto v:capture.get()) need(v==Sentinel,"null-capture wrote diagnostic");
    need(x.get()==hx && w.get()==hw,"input changed");
    std::cout << "{\"kind\":\"correctness\",\"rows\":" << rows << ",\"tile\":" << int(tile)
      << ",\"edges\":" << edges << ",\"max_half_ulp\":" << ulp
      << ",\"BF16_capture_exact\":true,\"guards\":true,\"inputs_immutable\":true,\"status\":\"PASS\"}\n";
    return ulp;
  }
  void check_profile(bool fusion_enabled=true) {
    namespace cp = sb::command_profile;
    strata::prefill::Gemm routed;
    std::string error;
    if (!routed.init_external(&q,nullptr,0,nullptr,0,error)) throw std::runtime_error(error);
    candidate.reset();capture.reset();separate(xe::Tile::wide_128x128);drain(q);
    bool selected=false;
    {
      std::unique_ptr<cp::Trace> trace;
      if (cp::requested()) trace.reset(new cp::Trace(&q,1));
      // Must drain before Trace or any Case-owned device/host storage unwinds.
      struct Retirement { sycl::queue& queue; ~Retirement() { drain(queue); } } retirement{q};
      auto frame=trace ? trace->begin(0,0) : cp::Frame{};
      frame.consider(0,0,rows);
      {
        auto tag=frame.tag(cp::Kind::gu,0,0,rows,N,K);
        selected=routed.try_gu_swiglu16(x.data(),w.data(),candidate.data(),rows);
        if (!selected) routed.f16_out16(x.data(),w.data(),gu.data(),rows,N,K,true);
      }
      if (!selected) {
        auto tag=frame.tag(cp::Kind::swiglu,0,0,rows,H);
        strata::prefill::native16::swiglu(gu.data(),candidate.data(),rows,&q);
      }
      drain(q);frame.after_checked_drain();
    }
    need(selected==(fusion_enabled && rows>=2334),"profile changed production GU dispatch");
    const int ulp=compare_hidden(reference,candidate);
    need(x.get()==hx && w.get()==hw,"profiled GU changed inputs");
    for (auto v:capture.get()) need(v==Sentinel,"profiled GU wrote diagnostic storage");
    std::cout<<"{\"kind\":\"profiled_engine_gu\",\"rows\":"<<rows<<",\"selected\":"<<selected
      <<",\"expected_records\":"<<(selected ? 1 : 2)<<",\"max_half_ulp\":"<<ulp
      <<",\"guards\":true,\"inputs_immutable\":true,\"status\":\"PASS\"}\n";
  }
};
#ifdef STRATA_PREFILL_GU_A_PROBE
using Routes = strata::prefill::detail::MoeWindowRoutes;

struct MappedGuInputs {
  std::shared_ptr<const Routes> plan;
  size_t job_index;
  int primary_rows;
  std::vector<int32_t> metadata;
};

MappedGuInputs mapped_gu_inputs(int rows, bool split, bool clustered, bool invalid) {
  const int bank_rows = 2 * rows + 7;
  std::vector<int32_t> ids(size_t(bank_rows) * Routes::top_k);
  for (int row = 0; row < bank_rows; ++row) {
    ids[size_t(row) * Routes::top_k] = 0;
    for (int slot = 1; slot < Routes::top_k; ++slot)
      ids[size_t(row) * Routes::top_k + slot] = 15 + slot;
  }
  for (int row = 0; row < rows; ++row) {
    const int token = clustered ? 32 * (row / 16) + row % 16 + 3 : 2 * row + 3;
    ids[size_t(token) * Routes::top_k + 1] = 1;
  }
  auto plan = Routes::build_split_accum(bank_rows, ids.data(), ids.size());
  const auto &jobs = plan->jobs();
  const auto found = std::find_if(jobs.begin(), jobs.end(), [](const auto &job) { return job.expert == 1; });
  need(found != jobs.end() && found->rows == rows && found->global_row > 0, "mapped GU fixture job");
  const size_t job_index = size_t(found - jobs.begin());
  std::vector<int32_t> metadata(ids.size() + 2 * Pad, -0x55aa);
  std::copy(plan->source_rows().begin(), plan->source_rows().end(), metadata.begin() + Pad);
  if (invalid) {
    need(rows >= 2, "mapped GU invalid endpoints require distinct rows");
    metadata[Pad + found->global_row] = -1;
    metadata[Pad + found->global_row + rows - 1] = bank_rows;
  }
  return {std::move(plan), job_index, split ? rows + 3 : bank_rows, std::move(metadata)};
}

struct MappedGuCase {
  MappedGuInputs inputs;
  Case dense;
  Buffer primary;
  std::unique_ptr<Buffer> tail;
  sb::Allocation metadata, sticky;
  static constexpr uint32_t status_seed = 32, status_guard = 0xb3c4d5e6;

  MappedGuCase(int rows, bool split, bool clustered, bool invalid, sycl::queue &q)
      : inputs(mapped_gu_inputs(rows, split, clustered, invalid)), dense(rows, false, q),
        primary(size_t(inputs.primary_rows) * K, q),
        tail(split ? std::make_unique<Buffer>(size_t(inputs.plan->rows() - inputs.primary_rows) * K, q)
                   : nullptr),
        metadata(runtime, inputs.metadata.size() * sizeof(int32_t), sb::MemoryKind::Device),
        sticky(runtime, 3 * sizeof(uint32_t), sb::MemoryKind::Device) {
    primary.set(bank_values(0, primary.n));
    if (tail) tail->set(bank_values(inputs.primary_rows, tail->n));
    submit_wait(q, [&] {
      return q.memcpy(metadata.data(), inputs.metadata.data(), inputs.metadata.size() * sizeof(int32_t));
    });
    reset_status();
  }

  ~MappedGuCase() { drain(dense.q); }

  static uint16_t input_value(size_t row, size_t col) {
    // Finite dyadic inputs with a row-dependent pattern in every K tile.
    const uint32_t mixed = uint32_t((row + 1) * 257 + col) * 0x9e3779b9u;
    return half(float(int(mixed >> 27) - 16) / 64.f);
  }
  static std::vector<uint16_t> bank_values(int first_row, size_t count) {
    std::vector<uint16_t> v(count);
    for (size_t i = 0; i < count; ++i) v[i] = input_value(first_row + i / K, i % K);
    return v;
  }
  const int32_t *source() const { return metadata.as<int32_t>() + Pad; }
  uint32_t *status() const { return sticky.as<uint32_t>() + 1; }
  void reset_status() {
    const std::array<uint32_t, 3> values{status_guard, status_seed, status_guard};
    submit_wait(dense.q, [&] { return dense.q.memcpy(sticky.data(), values.data(), sizeof(values)); });
  }
  void check_status(bool invalid) {
    std::array<uint32_t, 3> values{};
    submit_wait(dense.q, [&] { return dense.q.memcpy(values.data(), sticky.data(), sizeof(values)); });
    need(values == std::array<uint32_t, 3>{status_guard, status_seed | (invalid ? 4u : 0u), status_guard},
         "mapped GU sticky status or guard mismatch");
  }
  sycl::event gathered(xe::Tile tile, bool capture) {
    strata::prefill::native16::moe_gather_mixed(*inputs.plan, inputs.job_index, primary.data(),
        tail ? tail->data() : nullptr, inputs.primary_rows, source(), dense.x.data(), status(), &dense.q);
    return dense.fused(tile, capture);
  }
  xe::PrefillMappedGuProblem problem(bool capture) {
    return {inputs.plan.get(), inputs.job_index, primary.data(), tail ? tail->data() : nullptr,
        inputs.primary_rows, source(), dense.w.data(), dense.reference.data(), status(),
        capture ? dense.gu.data() : nullptr};
  }
  sycl::event mapped(xe::Tile tile, bool capture) {
    return xe::prefill_mapped_gu_swiglu(*runtime, dense.q, problem(capture), tile);
  }
  void check(xe::Tile tile, bool invalid) {
    auto &q = dense.q;
    reset_status();
    submit_wait(q, [&] { return gathered(tile, true); });
    check_status(invalid);
    const auto operand = dense.x.get();
    const auto &job = inputs.plan->jobs()[inputs.job_index];
    for (int i = 0; i < job.rows; ++i) {
      const int row = inputs.metadata[Pad + job.global_row + i];
      for (int col = 0; col < K; ++col)
        need(operand[size_t(i) * K + col] ==
             (row < 0 || row >= inputs.plan->rows() ? 0 : input_value(row, col)), "GU gather oracle");
    }
    const auto ref = dense.candidate.get();
    reset_status();
    submit_wait(q, [&] { return mapped(tile, true); });
    check_status(invalid);
    need(dense.gu.get() == dense.capture.get(), "mapped GU BF16 bits differ from gathered GU");
    need(dense.reference.get() == ref, "mapped GU FP16 output differs from gathered GU");

    dense.gu.reset();
    dense.capture.reset();
    dense.reference.reset();
    dense.candidate.reset();
    reset_status();
    submit_wait(q, [&] { return gathered(tile, false); });
    check_status(invalid);
    reset_status();
    submit_wait(q, [&] { return mapped(tile, false); });
    check_status(invalid);
    need(dense.candidate.get() == ref && dense.reference.get() == ref, "mapped GU null capture changed output");
    for (auto *out : {&dense.gu, &dense.capture})
      for (auto word : out->get()) need(word == Sentinel, "mapped GU null capture wrote diagnostics");
    std::vector<int32_t> observed(inputs.metadata.size());
    submit_wait(q, [&] { return q.memcpy(observed.data(), metadata.data(), observed.size() * sizeof(int32_t)); });
    need(observed == inputs.metadata && primary.get() == bank_values(0, primary.n) &&
         dense.w.get() == dense.hw && dense.x.get() == operand, "mapped GU input or guard changed");
    if (tail) need(tail->get() == bank_values(inputs.primary_rows, tail->n), "mapped GU tail or guard changed");
  }
};

void check_mapped_gu(sycl::queue &q, xe::Tile tile, int rows, bool split, bool clustered, bool invalid) {
  MappedGuCase c(rows, split, clustered, invalid, q);
  c.check(tile, invalid);
  std::cout << "prefill_mapped_gu rows=" << rows << " tile=" << (tile == xe::Tile::wide_128x128 ? "wide" : "compact")
            << " split=" << split << " clustered=" << clustered << " invalid=" << invalid
            << " gather=exact BF16_capture=exact FP16_output=exact null_capture=exact guards=PASS\n";
}
#endif

void admissions(Case &c) {
  int count=0;
  auto reject=[&](xe::GuSwiGLUProblem p, xe::Tile tile=xe::Tile::wide_128x128) {
    try { wait(c.q,c.gemm.gu_swiglu(p,tile)); }
    catch (const std::invalid_argument &) { ++count; return; }
    throw std::runtime_error("invalid GU admission submitted work");
  };
  auto p=c.problem(); p.rows=0; reject(p);
  p=c.problem(); p.rows=8193; reject(p);
  reject(c.problem(),static_cast<xe::Tile>(99));
  p=c.problem(); p.x=nullptr; reject(p);
  p=c.problem(); p.x=c.hx.data(); reject(p);
  p=c.problem(); ++p.x; reject(p);
  p=c.problem(); p.y=const_cast<uint16_t*>(p.x); reject(p);
  p=c.problem(); p.y=reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(p.y)+1); reject(p);
  Buffer tiny(1,c.q); p=c.problem(); p.x=tiny.data(); reject(p);
  p=c.problem(); p.y=tiny.data(); reject(p);
  p=c.problem(true); p.rounded_diagnostic=tiny.data(); reject(p);
  p=c.problem(true); p.rounded_diagnostic=p.y; reject(p);
  p=c.problem(true); p.rounded_diagnostic=const_cast<uint16_t*>(p.w); reject(p);
  need(count==13,"admission case count");
  std::cout << "{\"kind\":\"admissions\",\"rejected\":" << count << ",\"status\":\"PASS\"}\n";
}
class Marker;
sycl::event mark(sycl::queue &q) { return q.single_task<Marker>([] {}); }
// Exercise the public production dispatcher, including its refused-path cost.
// The process environment selects an arm before SYCL initializes; never mutate it.
struct PublicGU {
  Case &c;
  strata::prefill::Gemm routed;
  bool expected;
  PublicGU(Case &input, bool fusion) : c(input), expected(fusion && c.rows>=2334) {
    std::string error;
    if (!routed.init_external(&c.q,nullptr,0,nullptr,0,error)) throw std::runtime_error(error);
  }
  ~PublicGU() { drain(c.q); }
  bool call() {
    try {
      const bool selected=routed.try_gu_swiglu16(c.x.data(),c.w.data(),c.candidate.data(),c.rows);
      if (!selected) {
        routed.f16_out16(c.x.data(),c.w.data(),c.gu.data(),c.rows,N,K,true);
        strata::prefill::native16::swiglu(c.gu.data(),c.candidate.data(),c.rows,&c.q);
      }
      return selected;
    } catch (...) { drain(c.q); throw; }
  }
};
void public_settings(int arm) {
  const auto matches=[](const char *key,const char *wanted) {
    const char *value=std::getenv(key);
    return value && std::string(value)==wanted;
  };
  need(arm==0 || arm==1,"public arm must be0 or1");
  need(matches("STRATA_SYCL_PREFILL_NATIVE16","bf16") &&
       matches("STRATA_SYCL_PREFILL_NATIVE16_BACKEND","vllm") &&
       matches("STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE","auto") &&
       matches("STRATA_SYCL_PREFILL_GU_SWIGLU",arm ? "large" : "0"),
       "public mode requires matching immutable bf16/vllm/auto/GU startup settings");
}
void benchmark(int process,sycl::queue &q) {
  for (int rows:{64,160,2334,5963,8192}) {
    const auto tile=rows<128 ? xe::Tile::compact_16x64 : xe::Tile::wide_128x128;
    Case c(rows,false,q); c.check(tile);
    constexpr int repeat=16;
    for (int round=-1;round<5;++round) for (int order=0;order<2;++order) {
      const int arm=(process+round+2+order)%2;
      c.candidate.reset(); c.reference.reset();
      const auto start=mark(q); const auto wall=std::chrono::steady_clock::now();
      for (int i=0;i<repeat;++i) if (arm) (void)c.fused(tile); else c.separate(tile);
      const auto end=mark(q); wait(q,end);
      const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall).count()/repeat;
      const auto a=start.get_profiling_info<sycl::info::event_profiling::command_end>();
      const auto b=end.get_profiling_info<sycl::info::event_profiling::command_start>();
      need(b>a && wall_ms>0 && std::isfinite(wall_ms),"invalid timing");
      const double interval_ms=double(b-a)/1e6/repeat;
      if (arm) c.separate(tile); else (void)c.fused(tile);
      drain(q); const int ulp=compare_hidden(c.reference,c.candidate);
      need(c.x.get()==c.hx && c.w.get()==c.hw,"timed inputs changed");
      (void)c.gu.get(); (void)c.capture.get();
      std::cout << "{\"kind\":\"sample\",\"process\":" << process << ",\"rows\":" << rows
        << ",\"round\":" << round << ",\"order\":" << order << ",\"arm\":" << arm
        << ",\"repeat\":" << repeat << ",\"wall_ms\":" << wall_ms
        << ",\"queue_interval_ms\":" << interval_ms << ",\"max_half_ulp\":" << ulp
        << ",\"status\":\"PASS\"}\n";
    }
  }
}
} // namespace
int main(int argc,char **argv) {
  std::cout << std::unitbuf;
  std::cout.precision(17); // preserve nanosecond CPU partitions in longer batches
  try {
    need(argc>=2,"use --ownership|--smoke|--validate|--bench PROCESS");
    const std::string mode=argv[1];
    if (mode=="--ownership") {
      ownership(); std::cout << "fixture_status=PASS mode=ownership GPU_submissions=0\n"; return 0;
    }
#ifdef STRATA_PREFILL_GU_A_PROBE
    if (mode=="--prefill-mapped-a-layout") {
      need(argc == 2, "prefill GU A layout takes no arguments");
      for (auto tile : {xe::Tile::compact_16x64, xe::Tile::wide_128x128}) {
        const bool wide = tile == xe::Tile::wide_128x128;
        const int block = wide ? 128 : 16;
        for (int rows : {1, 3, 16, 17, 128, 129, 211, 2334, 6640, 8191, 8192}) {
          const auto cells = xe::prefill_gu_a_layout(rows, tile);
          const size_t valid = std::count_if(cells.begin(), cells.end(),
              [rows](uint32_t cell) { return cell / K < uint32_t(rows); });
          const size_t copies = 96 * (wide ? 2 : 4);
          need(cells.size() == size_t((rows + block - 1) / block * block) * copies &&
               valid == rows * copies, "prefill GU A valid/masked coverage");
          std::cout << "prefill_gu_a_layout rows=" << rows << " tile=" << (wide ? "wide" : "compact")
                    << " valid=" << valid << " masked=" << cells.size() - valid << '\n';
        }
      }
      std::cout << "fixture_status=PASS mode=" << mode << " GPU_submissions=0\n";
      return 0;
    }
    if (mode=="--prefill-mapped-a-smoke" || mode=="--prefill-mapped-a-validate") {
      need(argc == 2, "prefill GU A probe takes no arguments");
      runtime = sb::runtime_for();
      auto q = runtime->make_queue(true);
      topology(q);
      const bool smoke = mode == "--prefill-mapped-a-smoke";
      for (auto tile : {xe::Tile::compact_16x64, xe::Tile::wide_128x128}) {
        if (smoke) prefill_gu_a_probe(q, tile, 3, false);
        else for (int rows : {16, 17, 128, 129, 211, 2334})
          for (bool invalid : {false, true}) prefill_gu_a_probe(q, tile, rows, invalid);
      }
      std::cout << "fixture_status=PASS mode=" << mode << '\n';
      return 0;
    }
    if (mode == "--prefill-mapped-gu-smoke" || mode == "--prefill-mapped-gu-validate") {
      need(argc == 2, "mapped GU verification takes no arguments");
      runtime = sb::runtime_for();
      auto q = runtime->make_queue(true);
      topology(q);
      if (mode == "--prefill-mapped-gu-smoke") {
        check_mapped_gu(q, xe::Tile::compact_16x64, 3, true, false, false);
        check_mapped_gu(q, xe::Tile::wide_128x128, 3, true, false, true);
      } else {
        check_mapped_gu(q, xe::Tile::compact_16x64, 1, false, false, false);
        check_mapped_gu(q, xe::Tile::compact_16x64, 17, true, false, true);
        check_mapped_gu(q, xe::Tile::wide_128x128, 129, true, true, true);
        for (int rows : {2334, 5963, 8192})
          for (bool clustered : {false, true})
            check_mapped_gu(q, xe::Tile::wide_128x128, rows, true, clustered, false);
        check_mapped_gu(q, xe::Tile::wide_128x128, 8191, true, false, true);
      }
      std::cout << "fixture_status=PASS mode=" << mode << '\n';
      return 0;
    }
#endif
    const bool public_profile=mode=="--profile-public";
    const bool public_smoke=mode=="--smoke-public";
    int public_arm=0;
    if (public_profile || public_smoke) {
      need(argc==3,"public mode needs ARM");
      public_arm=std::stoi(argv[2]);public_settings(public_arm);
    }
    need(mode=="--smoke" || mode=="--validate" || mode=="--bench" || mode=="--profile-engine-hook" ||
         public_profile || public_smoke,"invalid mode");
    for (auto tile:{xe::Tile::compact_16x64,xe::Tile::wide_128x128}) {
      const auto p=xe::inspect_gu_pair_ownership(8192,tile,xe::GuLayout::matrix);
      need(!p.partner_mismatch && !p.packed_missing && !p.packed_multiple && !p.packed_orientation_mismatch,
           "packed partner ownership gate");
    }
    runtime=sb::runtime_for();
    auto q=runtime->make_queue(true);
    topology(q);
    if (public_profile || public_smoke) {
      for (int rows:{3,160,2333,2334}) { Case c(rows,false,q);c.check_profile(public_arm!=0); }
    } else if (mode=="--profile-engine-hook") {
      need(argc==2,"profile hook takes no timing parameters");
      for (int rows:{2333,2334,8192}) { Case c(rows,false,q);c.check_profile(); }
    } else if (mode=="--smoke") {
      for (auto tile:{xe::Tile::compact_16x64,xe::Tile::wide_128x128}) {
        Case c(3,false,q); c.check(tile); Case edge(3,true,q); edge.check(tile);
      }
      Case c(3,false,q); admissions(c);
    } else if (mode=="--validate") {
      for (auto tile:{xe::Tile::compact_16x64,xe::Tile::wide_128x128}) {
        for (int rows:{1,17,33,129,2334,5963,8191,8192}) { Case c(rows,false,q); c.check(tile); }
        Case edge(17,true,q); edge.check(tile);
      }
    } else { need(argc==3,"bench process required"); const int p=std::stoi(argv[2]);
      need(p>=0 && p<3,"invalid process ordinal"); benchmark(p,q); }
    q.wait_and_throw(); runtime->check(); runtime->wait();
    std::cout << "fixture_status=PASS mode=" << mode.substr(2) << '\n'; return 0;
  } catch (const std::exception &e) {
    if (runtime) { try { runtime->wait(); } catch (...) { std::_Exit(86); } }
    std::cerr << "fixture_status=FAIL " << e.what() << '\n'; return 1;
  }
}
