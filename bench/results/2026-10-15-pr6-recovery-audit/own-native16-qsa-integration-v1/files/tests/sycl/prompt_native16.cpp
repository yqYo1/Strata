// SPDX-License-Identifier: MIT
// Own BF16 fixture adapted from the explicitly pinned independent FP64 oracle
// in own-prompt-heads8-v2/files/tests/sycl/prompt_heads8.cpp. No head8 kernel used.
// Source-only proposal: Main owns compilation, tracing, GPU runs and retention.
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;
std::ofstream dump;
uint64_t dump_offset = 0;
int repetitions = 0, benchmark_nq = 0;
constexpr int cells = 2113, cap = 2051, heads = 24, kvheads = 2, dim = 256;
void check(bool value, const char *what) {
  if (!value) throw std::runtime_error(what);
}
void checked_drain() noexcept {
  sycl_backend::checked_cleanup([] { runtime->wait(); }, [] {},
      "R787 BF16 fixture drain failed");
}
struct Drain { ~Drain() noexcept { checked_drain(); } };
template <class T> struct Buffer {
  sycl_backend::Allocation allocation;
  explicit Buffer(size_t count)
      : allocation(runtime, count * sizeof(T), sycl_backend::MemoryKind::Device) {}
  T *data() { return allocation.as<T>(); }
  void put(const std::vector<T> &values) {
    check(values.size() * sizeof(T) == allocation.size(), "full fixture upload size");
    try { runtime->wait(runtime->compute().memcpy(data(), values.data(), allocation.size())); }
    catch (...) { checked_drain(); throw; }
  }
  std::vector<T> get() {
    std::vector<T> values(allocation.size() / sizeof(T));
    try { runtime->wait(runtime->compute().memcpy(values.data(), data(), allocation.size())); }
    catch (...) { checked_drain(); throw; }
    return values;
  }
};
// Host oracle conversions are independent of production conversion helpers.
uint16_t bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if (std::isnan(value)) return uint16_t((bits >> 16) | 0x40u);
  uint32_t upper = bits >> 16, lower = bits & 0xffffu;
  if (lower > 0x8000u || (lower == 0x8000u && (upper & 1u))) ++upper;
  return uint16_t(upper);
}
float expand(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits) << 16); }
uint16_t fp16(float value) { return std::bit_cast<uint16_t>(_Float16(value)); }
float expand_half(uint16_t bits) { return float(std::bit_cast<_Float16>(bits)); }
double bf16_spacing(float value) {
  const uint16_t bits = bf16(std::fabs(value));
  if (bits >= 0x7f7f) return std::numeric_limits<double>::infinity();
  return double(expand(uint16_t(bits + 1))) - double(expand(bits));
}
template <class T> uint64_t hash(const std::vector<T> &values) {
  uint64_t h = 14695981039346656037ULL;
  const auto *bytes = reinterpret_cast<const unsigned char *>(values.data());
  for (size_t i = 0; i < values.size() * sizeof(T); ++i) h = (h ^ bytes[i]) * 1099511628211ULL;
  return h;
}
std::string hex(uint64_t h) {
  std::ostringstream text; text << std::hex << std::setfill('0') << std::setw(16) << h;
  return text.str();
}
void attention(const uint16_t *query, const QsaAttnPools &p,
               const int32_t *ids, const int32_t *steps, const QsaShapes &shape,
               uint16_t *out, int nq, const QsaPromptBf16Bounds &bounds) {
  check(qsa_prompt_attn_bf16_batch(query,p,ids,steps,cap,shape,out,nq,bounds,
                                  &runtime->compute()), "BF16 Mode1 matrix attention unavailable");
}
// Profiles cover distinct rounding/range mechanisms; none changes process env.
void run(int page_size, bool masked, int nq, int width, const std::string &profile,
         bool extended = false) {
  auto shape = qsa_real_shapes(); shape.page_size = page_size;
  const int pages = (cells + page_size - 1) / page_size;
  const int rows = pages * page_size * kvheads;
  const QsaPromptBf16Bounds bounds{pages,pages};
  std::vector<int32_t> table(pages);
  for (int i = 0; i < pages; ++i) table[i] = pages - 1 - i;
  if (masked) for (int i = 0; i < pages; i += 3) table[i] = -1;
  std::vector<uint16_t> ks(rows*4), vs(ks.size());
  std::vector<int8_t> k8(rows*dim), v8(k8.size());
  for (size_t i = 0; i < k8.size(); ++i) {
    k8[i] = int8_t(int(i*7%255)-127);
    v8[i] = int8_t(int(i*11%255)-127);
  }
  for (size_t i = 0; i < ks.size(); ++i) {
    ks[i] = fp16(.007f+float(i%7)*.0001f);
    vs[i] = fp16(.008f+float(i%11)*.0002f);
  }
  std::vector<uint16_t> query(size_t(nq)*heads*dim);
  for (size_t i = 0; i < query.size(); ++i) {
    float amplitude = (i/dim)%12 < 8 ? 4.f : 0x1p-80f;
    if (profile == "range") amplitude = (i/dim)%2 ? 0x1p-110f : 0x1p110f;
    query[i] = bf16(float(std::sin(i*.019))*amplitude);
  }
  if (profile == "zero") {
    std::fill(query.begin(),query.end(),bf16(0.f));
    std::fill(vs.begin(),vs.end(),fp16(0.f));
  } else if (profile == "smallp") {
    // Exp arguments span ~[-64,0]. V groups include minimum subnormal, zero,
    // maximum finite and negative finite scales; no reciprocal overflow allowed.
    std::fill(query.begin(),query.end(),bf16(0.f));
    for (int h = 0; h < nq*heads; ++h) query[h*dim] = bf16(4.f);
    std::fill(ks.begin(),ks.end(),fp16(1.f));
    for (size_t i = 0; i < vs.size(); ++i) {
      const float values[] = {0x1p-24f,0.f,65504.f,-0x1p-24f};
      vs[i] = fp16(values[i%4]);
    }
  } else if (profile == "positive_infinity_score" || profile == "negative_infinity_score") {
    // Finite operands overflow FP32 scores uniformly, so the FP64 reference
    // remains a uniform distribution. Explicitly exercise inf-inf avoidance.
    const uint16_t q = profile[0] == 'p' ? uint16_t(0x7f7f) : uint16_t(0xff7f);
    std::fill(query.begin(),query.end(),q);
    std::fill(k8.begin(),k8.end(),int8_t(127));
    std::fill(ks.begin(),ks.end(),fp16(65504.f));
  }
  std::vector<int32_t> ids(size_t(nq)*cap), steps(size_t(nq)*kStepCount);
  for (int b = 0; b < nq; ++b) {
    const int position = cells - 1 - (b%3)*37;
    steps[b*kStepCount+kStepPos] = position;
    steps[b*kStepCount+kStepNKv] = position+1;
    steps[b*kStepCount+kStepNBid] = (position+1)/4;
    steps[b*kStepCount+kStepWidth] = width;
    for (int i = 0; i < cap; ++i) {
      const int cell = (i*17+b*3)%cells;
      // Causal masking belongs to selected IDs. The kernel must not infer it
      // from a different step field, union rows, or expose a future cache cell.
      ids[b*cap+i] = cell <= position ? cell : -1;
    }
    ids[b*cap+5] = -1;
  }
  const uint16_t guard = bf16(1234.f);
  const std::vector<uint16_t> sentinel(query.size()+32,guard);
  Buffer<uint16_t> dks(ks.size()), dvs(vs.size()), q(query.size()),
      out(query.size()+32), single(query.size()+32);
  Buffer<int8_t> dk8(k8.size()), dv8(v8.size());
  Buffer<int32_t> dt(table.size()), di(ids.size()), ds(steps.size());
  Drain drain; // After ALL host/device owners, before the first upload.
  dks.put(ks); dvs.put(vs); dk8.put(k8); dv8.put(v8);
  q.put(query); dt.put(table); di.put(ids); ds.put(steps);
  QsaAttnPools p;
  p.page_table = dt.data(); p.k_q = dk8.data(); p.v_q = dv8.data();
  p.k_scale = dks.data(); p.v_scale = dvs.data();
  out.put(sentinel); single.put(sentinel);
  auto unchanged = [&] {
    check(q.get()==query && di.get()==ids && ds.get()==steps && dt.get()==table &&
          dk8.get()==k8 && dv8.get()==v8 && dks.get()==ks && dvs.get()==vs,
          "input immutability, including current table/step/scales");
  };
  auto checked = [&](Buffer<uint16_t> &buffer, bool zero = false, bool finite = true) {
    auto all = buffer.get();
    for (int i = 0; i < 16; ++i) {
      check(all[i]==guard,"BF16 output prefix guard");
      check(all[all.size()-16+i]==guard,"BF16 output suffix guard");
    }
    for (size_t i = 16; i < all.size()-16; ++i) {
      if (zero) check(expand(all[i])==0.f,"expected zero output");
      if (finite) check(std::isfinite(expand(all[i])),"full BF16 output finite");
    }
    return all;
  };
  auto invoke = [&] { attention(q.data(),p,di.data(),ds.data(),shape,out.data()+16,nq,bounds); };
  std::cout << "case_start nq=" << nq << " page=" << page_size << " masked=" << masked
            << " width=" << width << " profile=" << profile << "\n";
  invoke();
  auto y = checked(out);
  if (dump.is_open()) {
    const size_t bytes = query.size()*sizeof(uint16_t);
    dump.write(reinterpret_cast<const char*>(y.data()+16),bytes);
    check(bool(dump),"BF16 output dump write");
    std::cout << "dump offset=" << dump_offset << " bytes=" << bytes
              << " dtype=bf16 shape=" << nq << ",24,256 output_guarded_fnv1a64=" << hex(hash(y)) << "\n";
    dump_offset += bytes;
  }
  if (repetitions) {
    // Each sample measures the full submission-to-completion dependency, with
    // clean immutable env. Separate Main-owned processes establish repeats.
    runtime->wait();
    for (int sample = 0; sample < repetitions; ++sample) {
      const auto begin = std::chrono::steady_clock::now(); invoke(); runtime->wait();
      const double ms = std::chrono::duration<double,std::milli>(
          std::chrono::steady_clock::now()-begin).count();
      std::cout << "sample index=" << sample << " nq=" << nq << " cap=" << cap
                << " width=" << width << " wall_ms=" << std::setprecision(10) << ms << "\n";
    }
    check(checked(out)==y,"repeat BF16 bit identity");
  } else {
    for (int b = 0; b < nq; ++b)
      attention(q.data()+b*heads*dim,p,di.data()+b*cap,ds.data()+b*kStepCount,
                shape,single.data()+16+b*heads*dim,1,bounds);
    check(checked(single)==y,"batch/single BF16 bit identity");
  }
  // Independent scalar FP64 reference: exact BF16 queries and stored INT8/FP16
  // cache, no production helpers and no reproduction of XMX/online recurrence.
  auto load = [&](bool value,int row,int d) {
    return double((value?v8:k8)[row*dim+d]) *
           double(expand_half((value?vs:ks)[row*4+d/64]));
  };
  double worst_raw = 0, worst_rounded = 0, worst_budget_ratio = 0;
  for (int b = 0; b < nq; ++b) {
    if (benchmark_nq && b!=0 && b!=1 && b!=2 && b!=nq-1) continue;
    const int count = width >= 0 && width <= cap ? width : 0;
    for (int h = 0; h < heads; ++h) {
      std::vector<double> weights(count,0.);
      std::vector<int> selected_rows(count,-1);
      double maximum = -INFINITY, sum = 0;
      for (int j = 0; j < count; ++j) {
        const int cell = ids[b*cap+j];
        const int page = cell < 0 ? -1 : table[cell/page_size];
        if (page < 0) { weights[j] = -INFINITY; continue; }
        const int row = (page*kvheads+h/12)*page_size+cell%page_size;
        selected_rows[j] = row;
        double dot = 0;
        for (int d = 0; d < dim; ++d)
          dot += double(expand(query[(b*heads+h)*dim+d])) * load(false,row,d);
        weights[j] = dot/16.; maximum = std::max(maximum,weights[j]);
      }
      for (auto &w : weights) { w = std::isfinite(w) ? std::exp(w-maximum) : 0.; sum += w; }
      for (int d = 0; d < dim; ++d) {
        double reference = 0, magnitude = 0;
        for (int j = 0; j < count; ++j) if (selected_rows[j]>=0) {
          const double term = weights[j]*load(true,selected_rows[j],d);
          reference += term; magnitude += std::abs(term);
        }
        if (sum) { reference /= sum; magnitude /= sum; }
        const float actual = expand(y[16+(b*heads+h)*dim+d]);
        const float rounded = expand(bf16(float(reference)));
        const double raw = std::abs(double(actual)-reference);
        const double error = std::abs(double(actual)-double(rounded));
        // Predetermined component screen: one-part probability rounding plus
        // two BF16 output steps, with a tiny absolute floor for zero/subnormal.
        // It is not an end-to-end model quality or full-context acceptance gate.
        const double budget = .001*magnitude + 2.*bf16_spacing(float(reference)) + 1e-12;
        worst_raw = std::max(worst_raw,raw); worst_rounded = std::max(worst_rounded,error);
        worst_budget_ratio = std::max(worst_budget_ratio,error/budget);
        if (!(std::isfinite(actual) && error <= budget)) {
          std::cerr << "oracle_error b=" << b << " head=" << h << " d=" << d
                    << " actual=" << actual << " fp64=" << reference << " bf16_ref=" << rounded
                    << " abs_raw=" << raw << " abs_rounded=" << error << " budget=" << budget << "\n";
          throw std::runtime_error("native16 attention FP64/BF16 rounding screen");
        }
      }
    }
  }
  unchanged();
  if (!benchmark_nq) {
    // All missing, then recover the real valid/mixed page table and prove the
    // original result returns before exercising invalid-width paths.
    const auto original_table = table;
    table.assign(pages,-1); dt.put(table); out.put(sentinel); invoke();
    (void)checked(out,true); unchanged();
    table = original_table; dt.put(table); out.put(sentinel); invoke();
    check(checked(out)==y,"recover valid table and original BF16 output"); unchanged();
    const auto original_steps = steps;
    for (int invalid_width : {0,-1,cap+1}) {
      for (int b = 0; b < nq; ++b) steps[b*kStepCount+kStepWidth] = invalid_width;
      ds.put(steps); out.put(sentinel); invoke(); (void)checked(out,true); unchanged();
    }
    steps = original_steps; ds.put(steps);
  }
  if (extended) {
    out.put(sentinel);
    auto reject = [&](QsaAttnPools pools,QsaShapes geometry,int64_t capacity,
                      int64_t queries,QsaPromptBf16Bounds extent) {
      check(!qsa_prompt_attn_bf16_batch(q.data(),pools,di.data(),ds.data(),capacity,
             geometry,out.data()+16,queries,extent,&runtime->compute()),"unsupported must return false");
      check(out.get()==sentinel,"unsupported must leave output untouched");
    };
    reject(p,shape,0,nq,bounds); reject(p,shape,-1,nq,bounds);
    reject(p,shape,int64_t(std::numeric_limits<int32_t>::max())+1,nq,bounds);
    reject(p,shape,cap,-1,bounds); reject(p,shape,cap,65536,bounds);
    auto badshape = shape; badshape.n_head = 8; reject(p,badshape,cap,nq,bounds);
    badshape = shape; badshape.head_dim = 128; reject(p,badshape,cap,nq,bounds);
    badshape = shape; badshape.n_head_kv = 1; reject(p,badshape,cap,nq,bounds);
    badshape = shape; badshape.page_size = 0; reject(p,badshape,cap,nq,bounds);
    reject(p,shape,cap,nq,{0,pages}); reject(p,shape,cap,nq,{pages,0});
    auto badp = p; badp.k_q = nullptr; reject(badp,shape,cap,nq,bounds);
    badp = p; badp.v_q = nullptr; reject(badp,shape,cap,nq,bounds);
    badp = p; badp.k_scale = nullptr; reject(badp,shape,cap,nq,bounds);
    badp = p; badp.v_scale = nullptr; reject(badp,shape,cap,nq,bounds);
    badp = p; badp.k_q4 = reinterpret_cast<const uint8_t*>(q.data()); reject(badp,shape,cap,nq,bounds);
    badp = p; badp.v_q4 = reinterpret_cast<const uint8_t*>(q.data()); reject(badp,shape,cap,nq,bounds);
    check(qsa_prompt_attn_bf16_batch(nullptr,{},nullptr,nullptr,0,{},nullptr,0,{},
                                   &runtime->compute()),"nq0 null successful no-op");
    auto invalid_span = [&](const uint16_t *query_ptr,QsaAttnPools pools,
                            const int32_t *ids_ptr,uint16_t *output_ptr) {
      bool threw = false;
      try { (void)qsa_prompt_attn_bf16_batch(query_ptr,pools,ids_ptr,ds.data(),cap,
              shape,output_ptr,nq,bounds,&runtime->compute()); }
      catch (const std::invalid_argument &) { threw = true; }
      check(threw,"invalid span must throw before submission");
      check(out.get()==sentinel,"invalid span original output untouched");
    };
    invalid_span(q.data(),p,di.data(),q.data());
    invalid_span(q.data(),p,reinterpret_cast<const int32_t*>(out.data()+16),out.data()+16);
    badp = p; badp.page_table = reinterpret_cast<const int32_t*>(out.data()+16);
    invalid_span(q.data(),badp,di.data(),out.data()+16);
    invalid_span(q.data(),p,di.data(),reinterpret_cast<uint16_t*>(dk8.data()+8));
    invalid_span(query.data(),p,di.data(),out.data()+16); // ordinary host pointer
    invalid_span(nullptr,p,di.data(),out.data()+16);
    badp = p; badp.page_table = nullptr; invalid_span(q.data(),badp,di.data(),out.data()+16);
    invalid_span(reinterpret_cast<const uint16_t*>(reinterpret_cast<const unsigned char*>(q.data())+1),
                 p,di.data(),out.data()+16);
    unchanged();
    const auto original_table = table;
    table.assign(pages,pages); dt.put(table); invoke(); (void)checked(out,true); unchanged();
    table = original_table; dt.put(table);
    const auto original_ids = ids;
    std::fill(ids.begin(),ids.end(),pages*page_size); di.put(ids); invoke();
    (void)checked(out,true); unchanged(); ids = original_ids; di.put(ids);
    // Nonfinite live scale is a declared diagnostic NaN, never an overflowing
    // probability cast. Missing rows do not read their invalid scales.
    const auto original_vs = vs;
    std::fill(vs.begin(),vs.end(),uint16_t(0x7c00)); dvs.put(vs);
    out.put(sentinel); invoke();
    auto nonfinite = checked(out,false,false);
    for (size_t i = 16; i < nonfinite.size()-16; ++i)
      check(std::isnan(expand(nonfinite[i])),"infinite V scale must diagnose NaN");
    unchanged();
    table.assign(pages,-1); dt.put(table); invoke(); (void)checked(out,true); unchanged();
    table = original_table; dt.put(table); vs = original_vs; dvs.put(vs);
    const auto original_query = query;
    query[0] = 0x7f80; q.put(query); out.put(sentinel); invoke();
    nonfinite = checked(out,false,false);
    for (int b = 0; b < nq; ++b) for (int h = 0; h < heads; ++h)
      for (int d = 0; d < dim; ++d)
        check(b==0 && h==0 ? std::isnan(expand(nonfinite[16+(b*heads+h)*dim+d]))
                           : std::isfinite(expand(nonfinite[16+(b*heads+h)*dim+d])),
              "nonfinite Q isolated to affected head");
    unchanged();
    table.assign(pages,-1); dt.put(table); invoke(); (void)checked(out,true); unchanged();
    table = original_table; dt.put(table);
    query[0] = 0x7fc1; q.put(query); invoke(); nonfinite = checked(out,false,false);
    for (int d = 0; d < dim; ++d)
      check(std::isnan(expand(nonfinite[16+d])),"NaN Q diagnostic");
    unchanged(); query = original_query; q.put(query);
    const auto original_ks = ks;
    std::fill(ks.begin(),ks.end(),uint16_t(0x7c00)); dks.put(ks); invoke();
    nonfinite = checked(out,false,false);
    for (size_t i = 16; i < nonfinite.size()-16; ++i)
      check(std::isnan(expand(nonfinite[i])),"infinite K scale must diagnose NaN");
    unchanged(); ks = original_ks; dks.put(ks); out.put(sentinel); invoke();
    check(checked(out)==y,"nonfinite-case recovery to original BF16 output"); unchanged();
  }
  std::cout << "case dtype=bf16 nq=" << nq << " shape=" << nq << ",24,256 page=" << page_size
            << " masked=" << masked << " width=" << width << " profile=" << profile
            << " oracle=" << (benchmark_nq?"FP64_queries_0_1_2_last":"FP64_full")
            << " max_abs_raw=" << worst_raw << " max_abs_bf16_ref=" << worst_rounded
            << " max_budget_ratio=" << worst_budget_ratio
            << " input_q_fnv1a64=" << hex(hash(query)) << " ids_fnv1a64=" << hex(hash(ids))
            << " steps_fnv1a64=" << hex(hash(steps)) << " table_fnv1a64=" << hex(hash(table))
            << " k_fnv1a64=" << hex(hash(k8)) << " v_fnv1a64=" << hex(hash(v8))
            << " ks_fnv1a64=" << hex(hash(ks)) << " vs_fnv1a64=" << hex(hash(vs))
            << " output_guarded_fnv1a64=" << hex(hash(y))
            << " finite=PASS guards=PASS inputs=PASS qualification=COMPONENT_ONLY_UNQUALIFIED\n";
}
} // namespace
int main(int argc,char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    // No setenv/unsetenv, before or after provider initialization. Main selects
    // traced smoke and clean benchmarks as separate owned child processes.
    const bool smoke = argc>=2 && std::string(argv[1])=="--smoke";
    const bool bench = argc==6 && std::string(argv[1])=="--bench";
    check((smoke && (argc==2 || argc==4)) || bench || (!smoke && (argc==1 || argc==3)),
          "usage: fixture [--dump PATH] | --smoke [--dump PATH] | --bench 128|256 repeats1..20 --dump PATH");
    const char *dump_path = nullptr;
    auto integer = [](const char *text) {
      char *end = nullptr; const long n = std::strtol(text,&end,10);
      check(end!=text && *end=='\0',"invalid integer"); return n;
    };
    if (bench) {
      const long n = integer(argv[2]), r = integer(argv[3]);
      check(n==128 || n==256,"benchmark nq128/256 required");
      check(r>=1 && r<=20,"benchmark repetitions1..20 required");
      benchmark_nq = int(n); repetitions = int(r);
      check(std::string(argv[4])=="--dump","benchmark requires full output dump"); dump_path = argv[5];
    } else if (argc==(smoke?4:3)) {
      check(std::string(argv[smoke?2:1])=="--dump","expected --dump"); dump_path = argv[smoke?3:2];
    }
    if (dump_path) { dump.open(dump_path,std::ios::binary|std::ios::trunc); check(bool(dump),"cannot open dump"); }
    // Algebraic write coverage for the preserved 12 active + 4 pad rows.
    std::vector<int> coverage(heads*dim);
    for (int kvh = 0; kvh < 2; ++kvh) for (int sg = 0; sg < 8; ++sg)
      for (int j = 0; j < 4; ++j) for (int r = 0; r < 8; ++r) for (int c = 0; c < 16; ++c) {
        const int h = (sg/4)*8+r, d = ((sg%4)*4+j)*16+c;
        if (h<12) ++coverage[(kvh*12+h)*dim+d];
      }
    for (int writes : coverage) check(writes==1,"head12+4 unique full write coverage");
    runtime = sycl_backend::runtime_for();
    if (bench) run(4,false,benchmark_nq,2051,"regular");
    else if (smoke) { run(4,false,1,33,"regular",true); run(4,true,3,33,"regular"); }
    else {
      for (int n : {1,3}) for (int width : {0,1,31,32,33,2051,-1,2052})
        for (int page : {1,4,16}) for (bool masked : {false,true})
          run(page,masked,n,width,"regular");
      run(4,false,3,33,"regular",true);
      for (const char *profile : {"range","zero","smallp","positive_infinity_score","negative_infinity_score"})
        for (int page : {1,4,16}) run(page,false,3,2051,profile);
    }
    runtime->wait();
    if (dump.is_open()) { dump.flush(); check(bool(dump),"dump flush"); dump.close(); check(!dump.fail(),"dump close"); }
    std::cout << "fixture=PASS model_quality=UNRUN whole_prefill=UNRUN physical262144=UNRUN faults=EXTERNAL_REQUIRED\n";
  } catch (const std::exception &error) {
    if (runtime) { try { runtime->wait(); } catch (...) { std::abort(); } }
    std::cerr << error.what() << "\n"; return 1;
  }
}
