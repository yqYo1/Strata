// Main-authored public-entry qualification; repository LICENSE.
// Reuses this fork's closed private fixture's oracle, guards and DMA owners.
#include "strata/kernels/qsa_topk_prefill_early.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

namespace sb = strata::sycl_backend;
namespace sk = strata::kernels;
constexpr size_t Guard = 16;
constexpr int32_t Marker = -123456789;
void need(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
void gate(const char *stage) {
  std::cout << "GATE stage=" << stage << std::endl;
  std::string s;
  need(bool(std::getline(std::cin, s)) && s == "continue", "controller acknowledgement");
}
struct Drain {
  sycl::queue queue;
  std::shared_ptr<sb::Runtime> runtime;
  ~Drain() { try { queue.wait_and_throw(); runtime->check(); } catch (...) { std::_Exit(86); } }
};
template <class T> struct Buffer {
  sb::Allocation allocation;
  size_t size;
  T marker;
  std::vector<T> upload, download;
  Buffer(std::shared_ptr<sb::Runtime> rt, size_t n, T m)
      : allocation(rt, (n + 2 * Guard) * sizeof(T), sb::MemoryKind::Device), size(n), marker(m) {}
  T *data() { return allocation.template as<T>() + Guard; }
  void put(sycl::queue &q, const std::vector<T> &values) {
    need(values.size() == size, "fixture upload extent"); q.wait_and_throw();
    upload.assign(size + 2 * Guard, marker);
    std::copy(values.begin(), values.end(), upload.begin() + Guard);
    q.memcpy(allocation.data(), upload.data(), allocation.size()).wait_and_throw();
  }
  const std::vector<T> &get(sycl::queue &q) {
    q.wait_and_throw(); download.resize(size + 2 * Guard);
    q.memcpy(download.data(), allocation.data(), allocation.size()).wait_and_throw();
    for (size_t i = 0; i < Guard; ++i)
      need(std::memcmp(&download[i], &marker, sizeof(T)) == 0 &&
           std::memcmp(&download[download.size() - 1 - i], &marker, sizeof(T)) == 0,
           "allocation guard changed");
    return download;
  }
  void immutable(sycl::queue &q) {
    get(q);
    need(download.size() == upload.size() &&
         std::memcmp(download.data(), upload.data(), download.size() * sizeof(T)) == 0,
         "fixture input modified");
  }
};
uint32_t random_word(uint32_t &x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }
float score(int pattern, int b, int row, uint32_t &seed) {
  const uint32_t r = random_word(seed);
  if (pattern == 0) return float(1 + r % 1000003) / 1024.f;
  if (pattern == 1) return float(r % 1024) / 16.f;
  if (pattern == 2) return std::bit_cast<float>(r);
  if (pattern == 3) return 1.f;
  if (pattern == 4) return std::bit_cast<float>(0x7fc00001u + uint32_t(b % 8192));
  if (pattern == 5) return std::bit_cast<float>(uint32_t(b & 1) << 31);
  if (pattern == 6) {
    constexpr uint32_t special[] = {0xff800000u,0x7f800000u,0x80000000u,0,
      0x7fc00001u,0xffc12345u,0xbf800000u,0x3f800000u,1,0x80000001u};
    return std::bit_cast<float>(special[(b + row) % 10]);
  }
  return b < 2 ? 3.f : (b % 13 == 0 ? -1.f : 2.f);
}
struct Spec { int cells, capacity, queries, cap, pattern; bool invalid = false; };
Spec checked(Spec s) {
  need(s.cells >= s.queries && s.queries >= 1 && s.queries <= 256 &&
       s.cells <= s.capacity && s.capacity <= 262144 && s.cap >= 1 && s.cap <= 2051 &&
       s.pattern >= 0 && s.pattern <= 7, "bounded fixture geometry");
  return s;
}
using Select = decltype(&sk::qsa_block_topk_prefill_early);
Select prefill_entry() {
  return sk::qsa_prefill_topk_early_enabled() ? sk::qsa_block_topk_prefill_early
                                            : sk::qsa_block_topk_prefill;
}
struct Fixture {
  std::shared_ptr<sb::Runtime> rt;
  sycl::queue queue;
  Spec spec;
  int blocks, active;
  std::vector<float> hs;
  std::vector<int32_t> ht, expected;
  Buffer<float> scores;
  Buffer<int32_t> steps, ids;
  Drain drain; // Last member: drain before every host/device allocation on exceptions.
  Fixture(std::shared_ptr<sb::Runtime> runtime, const sycl::queue &q, Spec s)
      : rt(std::move(runtime)), queue(q), spec(checked(s)), blocks(spec.capacity / 4 + 1),
        active(spec.cells / 4 + 1), hs(size_t(spec.queries) * blocks, INFINITY),
        ht(size_t(spec.queries) * sk::kStepCount), expected(size_t(spec.queries) * spec.cap, Marker),
        scores(rt, hs.size(), -12345.5f), steps(rt, ht.size(), Marker),
        ids(rt, expected.size(), Marker), drain{queue, rt} {
    uint32_t seed = 0x67149a3u;
    struct Cell { float value; int32_t id; };
    const auto better = [](Cell a, Cell b) {
      const bool an = std::isnan(a.value), bn = std::isnan(b.value);
      if (an != bn) return !an;
      if (!an && a.value != b.value) return a.value > b.value;
      return a.id < b.id;
    };
    for (int qi = 0; qi < spec.queries; ++qi) {
      const int n = spec.cells - spec.queries + 1 + qi, width = std::min(n, spec.cap);
      auto *st = ht.data() + qi * sk::kStepCount;
      st[sk::kStepPos] = n - 1; st[sk::kStepNKv] = n;
      st[sk::kStepNBid] = n / 4; st[sk::kStepWidth] = width;
      auto *sc = hs.data() + size_t(qi) * blocks;
      for (int b = 0; b <= n / 4; ++b) sc[b] = score(spec.pattern, b, qi, seed);
      if (spec.pattern <= 1 && n % 4) sc[n / 4] += 1e9f;
      if (n % 4 == 0) sc[n / 4] = INFINITY; // Zero-weight spare must not select cells.
      if (spec.invalid) {
        const int kind = qi % 7;
        if (kind == 0) st[sk::kStepNKv] = -1;
        if (kind == 1) st[sk::kStepNBid] = -1;
        if (kind == 2) st[sk::kStepWidth] = -1;
        if (kind == 3) st[sk::kStepWidth] = spec.cap + 1;
        if (kind == 4) st[sk::kStepWidth] = n + 1;
        if (kind == 5) st[sk::kStepWidth] = 0;
        if (kind != 6) continue;
      }
      std::vector<Cell> cells; cells.reserve(n);
      for (int id = 0; id < n; ++id) cells.push_back({sc[id / 4], id});
      if (width < n) std::nth_element(cells.begin(), cells.begin() + width, cells.end(), better);
      std::sort(cells.begin(), cells.begin() + width, better);
      auto *out = expected.data() + size_t(qi) * spec.cap;
      for (int i = 0; i < width; ++i) out[i] = cells[i].id;
      std::sort(out, out + width);
    }
    scores.put(queue, hs); steps.put(queue, ht);
  }
  void finish() { queue.wait_and_throw(); rt->check(); }
  void output() {
    const auto &x = ids.get(queue);
    need(std::equal(expected.begin(), expected.end(), x.begin() + Guard), "numeric cell-rank oracle differs");
  }
  void qualify() {
    for (int api = 0; api < 3; ++api) for (int repeat = 0; repeat < 2; ++repeat) {
      ids.put(queue, std::vector<int32_t>(expected.size(), Marker));
      const auto shape = sk::qsa_real_shapes();
      if (api == 2) sk::qsa_block_topk_ref(scores.data(), steps.data(), spec.queries, blocks,
                                         spec.cap, shape, ids.data(), &queue);
      else (api == 0 ? prefill_entry() : sk::qsa_block_topk)(scores.data(), steps.data(),
          spec.queries, blocks, spec.cap, shape, ids.data(), &queue, repeat ? -1 : active);
      finish(); output(); scores.immutable(queue); steps.immutable(queue);
    }
    std::cout << "CASE cells=" << spec.cells << " capacity=" << spec.capacity << " queries=" << spec.queries
      << " cap=" << spec.cap << " pattern=" << spec.pattern << " invalid=" << spec.invalid
      << " oracle=PASS inputs=PASS guards=PASS repeats=2 apis=3\n";
  }
};
void comparator_controls() {
  int refused = 0;
  for (const auto &x : {std::vector<int>{2,1}, std::vector<int>{1,1}, std::vector<int>{1,-1}})
    try { need(x == std::vector<int>({1,2}), "ID comparator"); } catch (const std::runtime_error &) { ++refused; }
  need(refused == 3, "comparator disabled");
  std::cout << "COMPARATOR refusals=3 identity=PASS\n";
}
void admission(std::shared_ptr<sb::Runtime> rt, const sycl::queue &q) {
  Fixture f(rt, q, {17,32,3,5,0}); int refused = 0;
  f.ids.put(f.queue, std::vector<int32_t>(f.expected.size(), Marker));
  for (Select select : {sk::qsa_block_topk_prefill_early, sk::qsa_block_topk_prefill, sk::qsa_block_topk}) {
    const auto s = sk::qsa_real_shapes();
    auto reject = [&](auto fn) {
      try { fn(); } catch (const std::invalid_argument &) { ++refused; return; }
      throw std::runtime_error("invalid admission accepted");
    };
    auto run = [&](int64_t nq, int64_t blocks, int64_t cap, int64_t active, const sk::QsaShapes &shape) {
      select(f.scores.data(), f.steps.data(), nq, blocks, cap, shape, f.ids.data(), &f.queue, active);
    };
    reject([&]{run(0,f.blocks,5,f.active,s);}); reject([&]{run(65536,f.blocks,5,f.active,s);});
    reject([&]{run(3,0,5,f.active,s);}); reject([&]{run(3,int64_t(INT_MAX)+1,5,f.active,s);});
    reject([&]{run(3,f.blocks,0,f.active,s);}); reject([&]{run(3,f.blocks,int64_t(INT_MAX)+1,f.active,s);});
    for (int64_t a : {int64_t(-2),int64_t(0),int64_t(f.blocks+1)}) reject([&]{run(3,f.blocks,5,a,s);});
    for (int k = 0; k < 5; ++k) {
      auto bad=s;
      if(k==0)bad.idx_dim=127; if(k==1)bad.idx_n_head=3; if(k==2)bad.idx_block=3;
      if(k==3)bad.idx_top_k=-1; if(k==4)bad.idx_top_k=int64_t(INT_MAX)-2;
      reject([&]{run(3,f.blocks,5,f.active,bad);});
    }
    reject([&]{select(nullptr,f.steps.data(),3,f.blocks,5,s,f.ids.data(),&f.queue,f.active);});
    reject([&]{select(f.scores.data(),nullptr,3,f.blocks,5,s,f.ids.data(),&f.queue,f.active);});
    reject([&]{select(f.scores.data(),f.steps.data(),3,f.blocks,5,s,nullptr,&f.queue,f.active);});
    reject([&]{select(f.scores.data(),f.steps.data(),3,f.blocks,5,s,f.steps.data(),&f.queue,f.active);});
    reject([&]{select(f.scores.data(),f.steps.data(),3,f.blocks,5,s,reinterpret_cast<int32_t*>(f.scores.data()),&f.queue,f.active);});
    reject([&]{select(f.scores.data(),f.steps.data(),3,f.blocks,5,s,reinterpret_cast<int32_t*>(reinterpret_cast<char*>(f.ids.data())+1),&f.queue,f.active);});
  }
  f.finish(); const auto &out = f.ids.get(f.queue);
  need(std::all_of(out.begin()+Guard,out.end()-Guard,[](int v){return v==Marker;}),"admission touched output");
  f.scores.immutable(f.queue); f.steps.immutable(f.queue);
  need(refused == 60, "admission controls count");
  std::cout << "ADMISSION refusals=60 apis=3 output=PASS\n";
}
void numerical(std::shared_ptr<sb::Runtime> rt, const sycl::queue &q, bool smoke) {
  if (smoke) { Fixture f(rt,q,{17,32,3,5,0}); f.qualify(); return; }
  admission(rt,q);
  for (int n : {1,3,4,5,31,32,33,2047,2048,2051,2052,32768,65536,117488,262141,262142,262143,262144})
    for (int p=0;p<8;++p) { Fixture f(rt,q,{n,n,std::min(3,n),2051,p}); f.qualify(); }
  Fixture f(rt,q,{65,128,14,13,6,true}); f.qualify();
}
void policy(const std::string &expected) {
  bool invalid=false, enabled=false;
  try { enabled=sk::qsa_prefill_topk_early_enabled(); } catch(const std::invalid_argument&){invalid=true;}
  need(expected=="invalid" ? invalid : !invalid && enabled==(expected=="1"), "startup policy differs");
  if(!invalid)need(sk::qsa_prefill_topk_early_enabled()==enabled,"cached policy changed");
  std::cout << "POLICY expected=" << expected << " invalid=" << invalid << " enabled=" << enabled << " PASS=1\n";
}
void properties(std::shared_ptr<sb::Runtime> rt, const sycl::queue &q, const char *path) {
  void *loader=dlopen(path,RTLD_LOCAL|RTLD_NOW|RTLD_NOLOAD);need(loader,"Level Zero loader absent");
  auto query=reinterpret_cast<decltype(&zeKernelGetProperties)>(dlsym(loader,"zeKernelGetProperties"));need(query,"kernel property API absent");
  int count=0;
  for(const auto &id:sycl::get_kernel_ids()) {
    if(std::string(id.get_name()).find("submit_topk_prefill_early")==std::string::npos)continue;
    auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(),{q.get_device()},{id});
    const auto kernel=bundle.get_kernel(id);ze_kernel_properties_t p{};p.stype=ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    need(query(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(kernel),&p)==ZE_RESULT_SUCCESS,"kernel property query failed");
    const auto maximum=kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device());
    need(maximum>=256&&(!p.requiredGroupSizeX||p.requiredGroupSizeX==256)&&
      (!p.requiredGroupSizeY||p.requiredGroupSizeY==1)&&(!p.requiredGroupSizeZ||p.requiredGroupSizeZ==1),"compiled group geometry");
    need(size_t(p.localMemSize)+4096<=rt->device().get_info<sycl::info::device::local_mem_size>(),"compiled local-memory headroom");
    std::cout<<"PROPERTY maximum_group="<<maximum<<" subgroup="<<p.requiredSubgroupSize
      <<" static_local="<<p.localMemSize<<" spill="<<p.spillMemSize<<" PASS=1\n";++count;
  }
  need(count==1,"one dedicated prefill kernel required");dlclose(loader);
}
int main(int argc,char **argv) { try {
  std::cout<<std::unitbuf; need(argc>=2,"explicit bounded mode required");const std::string mode=argv[1];
  need((argc==2&&(mode=="--smoke"||mode=="--numerics"))||(argc==3&&mode=="--properties")||
    (argc==3&&mode=="--policy"&&(std::string(argv[2])=="0"||std::string(argv[2])=="1"||std::string(argv[2])=="invalid")),"bounded mode/arguments");
  comparator_controls();gate("before-runtime");auto rt=sb::runtime_for();
  need(rt->device().is_gpu()&&rt->device().get_info<sycl::info::device::vendor_id>()==0x8086,"Intel GPU required");
  std::cout<<"DEVICE name="<<rt->device().get_info<sycl::info::device::name>()<<'\n';gate("runtime-ready");
  auto q=rt->make_queue(true);Drain drain{q,rt};
  if(mode=="--policy")policy(argv[2]);
  else if(mode=="--properties")properties(rt,q,argv[2]);
  else {
    const bool enabled=sk::qsa_prefill_topk_early_enabled();
    std::cout<<"ROUTE early="<<enabled<<" entry="<<(enabled?"qsa_block_topk_prefill_early":"qsa_block_topk_prefill")<<'\n';
    numerical(rt,q,mode=="--smoke");
  }
  q.wait_and_throw();rt->check();std::cout<<"DONE public_entry=1 model_adopted=0 decode_changed=0\n";
  gate("before-exit");return 0;
} catch(const std::exception &e) {std::cerr<<"FAIL topk="<<e.what()<<'\n';std::cerr.flush();std::_Exit(1);} }
