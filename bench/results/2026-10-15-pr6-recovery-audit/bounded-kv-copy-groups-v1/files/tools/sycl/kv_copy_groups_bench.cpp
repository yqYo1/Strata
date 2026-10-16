// Component fixture invoking the actual kv_stream_resolve implementation.
// No model, RESTORE, attention arithmetic, or actual-decode performance claim.
#include "kv_copy_groups_oracle.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/sycl/runtime.hpp"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sycl/ext/oneapi/experimental/graph.hpp>
using namespace kv_copy_fixture;
using strata::sycl_backend::Allocation;
using strata::sycl_backend::MemoryKind;
using strata::sycl_backend::Runtime;
using namespace strata::kernels;
namespace {
constexpr size_t Guard = 16;
constexpr int32_t IntGuard = -123456789;
using Clock = std::chrono::steady_clock;
namespace ex=sycl::ext::oneapi::experimental;
using ExecGraph=ex::command_graph<ex::graph_state::executable>;
uint64_t ns(Clock::time_point a, Clock::time_point b) {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count());
}
uint64_t hash_bytes(const void *p, size_t n, uint64_t h = 14695981039346656037ULL) {
  const auto *b = static_cast<const uint8_t *>(p);
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
  return h;
}
void arm(int groups) {
  require(groups == 0 || groups == 128 || groups == 256 || groups == 512 || groups == 1024,
          "invalid benchmark copy group arm");
  require(setenv("STRATA_SYCL_KV_COPY_GROUPS", std::to_string(groups).c_str(), 1) == 0,
          "set copy group arm");
}
void frozen_env() {
  for (const auto &p : {std::pair{"ONEAPI_DEVICE_SELECTOR", "level_zero:gpu"},
                        {"NEOReadDebugKeys", "1"}, {"EnableDirectSubmission", "0"},
                        {"UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", "1"},
                        {"STRATA_SYCL_VERIFY_NATIVE_CAPTURE", "0"}}) {
    const char *v = std::getenv(p.first);
    require(v && std::string(v) == p.second, std::string("preserve frozen flag: ") + p.first);
  }
}
template<class T> struct Buffer {
  std::shared_ptr<Runtime> rt;
  sycl::queue &q;
  size_t count;
  T sentinel;
  Allocation allocation;
  Buffer(std::shared_ptr<Runtime> r, sycl::queue &queue, size_t n, T guard,
         MemoryKind kind = MemoryKind::Device)
      : rt(r), q(queue), count(n), sentinel(guard),
        allocation(r, (n + 2*Guard)*sizeof(T), kind) {}
  ~Buffer() noexcept {
    // Allocation drains runtime-owned queues; this fixture also owns an external
    // queue. Do not release an address on unknown external completion/error.
    try { q.wait_and_throw(); rt->check(); }
    catch (...) { std::cerr << "FAIL external queue completion; process teardown required\n"; std::_Exit(1); }
  }
  T *data() { return allocation.template as<T>() + Guard; }
  void put(const std::vector<T> &v) {
    require(v.size() == count, "buffer upload size");
    std::vector<T> all(count+2*Guard, sentinel);
    std::copy(v.begin(), v.end(), all.begin()+Guard);
    rt->wait(q.memcpy(allocation.data(), all.data(), all.size()*sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> all(count+2*Guard);
    rt->wait(q.memcpy(all.data(), allocation.data(), all.size()*sizeof(T)));
    require(std::all_of(all.begin(), all.begin()+Guard, [&](T v) { return v == sentinel; }) &&
            std::all_of(all.end()-Guard, all.end(), [&](T v) { return v == sentinel; }), "allocation guards");
    return {all.begin()+Guard, all.end()-Guard};
  }
};
struct Fixture {
  Shape shape;
  std::shared_ptr<Runtime> rt;
  sycl::queue &q;
  Buffer<int32_t> table, block, stamp, ref, ctl, mb, ms;
  std::vector<std::unique_ptr<Buffer<uint8_t>>> src, dst;
  KvStreamMap map;
  QsaAttnPools pools{};
  KvHostPools host{};
  QsaShapes geometry;
  Fixture(Shape s, std::shared_ptr<Runtime> r, sycl::queue &queue)
      : shape(s), rt(r), q(queue), table(r,q,s.blocks,IntGuard), block(r,q,s.slots,IntGuard),
        stamp(r,q,s.slots,IntGuard), ref(r,q,s.slots,IntGuard), ctl(r,q,16,IntGuard),
        mb(r,q,s.slots,IntGuard), ms(r,q,s.slots,IntGuard) {
    // Fixed shapes only. Largest live fixture is <1.5 GiB including expected/readback copies.
    require(s.blocks <= 65536 && s.slots <= 8192 && s.heads <= 2 && s.dim <= 256 && s.page <= 4,
            "bounded component allocation dimensions");
    geometry = qsa_real_shapes(); geometry.n_head_kv = s.heads;
    geometry.head_dim = s.dim; geometry.page_size = s.page;
    const auto lens = s.lengths();
    for (size_t a = 0; a < lens.size(); ++a) {
      const size_t len = lens[a];
      src.push_back(std::make_unique<Buffer<uint8_t>>(r,q,size_t(s.blocks)*len,uint8_t(0xd3),MemoryKind::Host));
      dst.push_back(std::make_unique<Buffer<uint8_t>>(r,q,size_t(s.slots)*len,uint8_t(0xb7)));
      // CPU writes host USM only before any submission uses it.
      auto *base = src.back()->allocation.template as<uint8_t>();
      std::fill(base, base+Guard, uint8_t(0xd3));
      std::fill(base+Guard+src.back()->count, base+2*Guard+src.back()->count, uint8_t(0xd3));
      for (int b = 0; b < s.blocks; ++b)
        for (size_t i = 0; i < len; ++i) src.back()->data()[size_t(b)*len+i] = pattern(int(a), b, i);
    }
    if (s.fmt == kKvF16) {
      host.k_pool = reinterpret_cast<uint16_t *>(src[0]->data()); host.v_pool = reinterpret_cast<uint16_t *>(src[1]->data());
      pools.k_pool = reinterpret_cast<uint16_t *>(dst[0]->data()); pools.v_pool = reinterpret_cast<uint16_t *>(dst[1]->data());
    } else if (s.fmt == kKvInt8) {
      host.k_q = reinterpret_cast<int8_t *>(src[0]->data()); host.v_q = reinterpret_cast<int8_t *>(src[1]->data());
      pools.k_q = reinterpret_cast<int8_t *>(dst[0]->data()); pools.v_q = reinterpret_cast<int8_t *>(dst[1]->data());
      host.k_scale = reinterpret_cast<uint16_t *>(src[2]->data()); host.v_scale = reinterpret_cast<uint16_t *>(src[3]->data());
      pools.k_scale = reinterpret_cast<uint16_t *>(dst[2]->data()); pools.v_scale = reinterpret_cast<uint16_t *>(dst[3]->data());
    } else {
      host.k_q4 = src[0]->data(); host.v_q4 = src[1]->data();
      pools.k_q4 = dst[0]->data(); pools.v_q4 = dst[1]->data();
    }
    pools.page_table = table.data();
    map = {table.data(), block.data(), stamp.data(), ref.data(), ctl.data(), mb.data(), ms.data(), s.blocks, s.slots};
    require(kv_block_bytes(geometry,s.fmt) == s.block_bytes(), "actual production block bytes");
  }
  void put(const State &m, const std::vector<std::vector<uint8_t>> &p) {
    table.put(m.table); block.put(m.block); stamp.put(m.stamp); ref.put(m.ref);
    ctl.put(m.ctl); mb.put(m.mb); ms.put(m.ms);
    for (size_t a = 0; a < p.size(); ++a) dst[a]->put(p[a]);
  }
  State get() {
    State m(shape);
    m.table=table.get(); m.block=block.get(); m.stamp=stamp.get(); m.ref=ref.get();
    m.ctl=ctl.get(); m.mb=mb.get(); m.ms=ms.get(); return m;
  }
  void source_exact() {
    q.wait_and_throw(); rt->check(); // external queue must complete before host-USM inspection
    const auto lens = shape.lengths();
    for (size_t a = 0; a < src.size(); ++a) {
      auto *base = src[a]->allocation.template as<uint8_t>();
      for (size_t i = 0; i < Guard; ++i)
        require(base[i] == 0xd3 && base[Guard+src[a]->count+i] == 0xd3, "host source guards");
      for (int b = 0; b < shape.blocks; ++b)
        for (size_t i = 0; i < lens[a]; ++i)
          if (src[a]->data()[size_t(b)*lens[a]+i] != pattern(int(a),b,i))
            throw std::runtime_error("host source is read-only");
    }
  }
};
struct Input {
  Buffer<int32_t> ids, steps;
  int nq, cap;
  std::vector<int32_t> expected_ids, expected_steps;
  Input(Fixture &f, const Case &c)
      : ids(f.rt,f.q,std::max(size_t(1),c.rows.size()*c.cap),IntGuard),
        steps(f.rt,f.q,std::max(size_t(kStepCount),c.rows.size()*kStepCount),IntGuard),
        nq(int(c.rows.size())), cap(c.cap) {
    put(c);
  }
  void put(const Case &c) {
    require(c.cap == cap && c.rows.size() == size_t(nq), "graph input capacity/nq unchanged");
    expected_ids.assign(ids.count,-999); expected_steps.assign(steps.count,0);
    for (size_t q = 0; q < c.rows.size(); ++q) {
      std::copy(c.rows[q].begin(), c.rows[q].end(), expected_ids.begin()+q*cap);
      expected_steps[q*kStepCount+kStepWidth] = c.widths[q];
    }
    ids.put(expected_ids); steps.put(expected_steps);
  }
  void submit(Fixture &f) {
    kv_stream_resolve(f.map,f.pools,f.host,f.shape.fmt,ids.data(),steps.data(),nq,cap,f.geometry,&f.q);
  }
};
struct Timing { uint64_t submit=0, marker_submit=0, whole=0, device_envelope=0; };
Timing timed(Fixture &f, Input &in, bool events, ExecGraph *graph=nullptr) {
  f.q.wait_and_throw(); f.rt->check(); // fixture boundary, outside measurement
  sycl::event begin, end;
  const auto t0 = Clock::now();
  if (events) begin = f.q.single_task([] {});
  const auto t1 = Clock::now();
  if (graph) (void)f.q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(*graph); });
  else in.submit(f);
  const auto t2 = Clock::now();
  if (events) end = f.q.single_task([] {});
  const auto t3 = Clock::now();
  f.q.wait_and_throw(); f.rt->check();
  const auto t4 = Clock::now();
  Timing t{ns(t1,t2), ns(t0,t1)+ns(t2,t3), ns(t0,t4), 0};
  if (events) {
    const auto lo = begin.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto hi = end.get_profiling_info<sycl::info::event_profiling::command_start>();
    require(hi >= lo, "device event interval order"); t.device_envelope = hi-lo;
  }
  return t;
}
uint64_t check(Fixture &f, Input &in, const State &expected,
               const std::vector<std::vector<uint8_t>> &payload) {
  const State got = f.get(); require(got == expected, "all map/control/miss arrays exact oracle");
  invariants(f.shape,got);
  uint64_t h=14695981039346656037ULL;
  for (const auto *v : {&got.table,&got.block,&got.stamp,&got.ref,&got.ctl,&got.mb,&got.ms})
    h = hash_bytes(v->data(),v->size()*4,h);
  for (size_t a=0;a<payload.size();++a) {
    auto actual=f.dst[a]->get(); require(actual==payload[a], "all pool bytes/scales/tails exact oracle");
    h=hash_bytes(actual.data(),actual.size(),h);
  }
  require(in.ids.get() == in.expected_ids && in.steps.get() == in.expected_steps,
          "selection inputs/guards remain exact");
  return h;
}
void record(const Fixture &f, const Case &c, const State &m, int groups, int sample,
            bool events, Timing t, uint64_t fingerprint, bool graph=false) {
  const size_t workgroups = groups ? std::min(size_t(groups),size_t(f.shape.slots)) : size_t(f.shape.slots);
  uint64_t width_sum=0; for (int w:c.widths) if(w>0 && w<=c.cap) width_sum+=w;
  std::cout << "{\"phase\":\"kv_stream_resolve_component\",\"case\":\"" << c.name
    << "\",\"fmt\":" << f.shape.fmt << ",\"n_blocks\":" << f.shape.blocks
    << ",\"n_slots\":" << f.shape.slots << ",\"page_size\":" << f.shape.page
    << ",\"heads\":" << f.shape.heads << ",\"head_dim\":" << f.shape.dim
    << ",\"nq\":" << c.rows.size() << ",\"cap\":" << c.cap << ",\"widths\":[";
  for(size_t i=0;i<c.widths.size();++i) std::cout << (i?",":"") << c.widths[i];
  std::cout << "],\"width_sum\":" << width_sum << ",\"configured_group_cap\":" << groups
    << ",\"launched_copy_groups\":" << (c.rows.empty()?0:workgroups)
    << ",\"copy_workitems\":" << (c.rows.empty()?0:workgroups*128)
    << ",\"placed_pages\":" << m.ctl[2] << ",\"logical_host_read_payload_bytes\":" << uint64_t(m.ctl[2])*f.shape.block_bytes()
    << ",\"lookups_u64\":" << u64(m.ctl,6) << ",\"misses_u64\":" << u64(m.ctl,4)
    << ",\"calls_u64\":" << u64(m.ctl,8) << ",\"overflow\":" << m.ctl[3]
    << ",\"sample\":" << sample << ",\"submission_mode\":\"" << (graph?"graph_replay":"direct_resolve")
    << "\",\"timing_mode\":\"" << (events?"profiled_marker_envelope":"host_clean")
    << "\",\"host_operation_submit_ns\":" << t.submit << ",\"host_marker_submit_ns\":" << (events?std::to_string(t.marker_submit):"null")
    << ",\"host_whole_queue_ns\":" << t.whole << ",\"device_marker_envelope_ns\":" << (events?std::to_string(t.device_envelope):"null")
    << ",\"bandwidth_GBps\":null,\"oracle_exact\":true,\"fingerprint_fnv64\":" << fingerprint << "}\n";
}
void reject_and_sync_contracts(Fixture &f) {
  auto c=cold(f.shape,1,"refusals_and_nullstream"); Input in(f,c);
  const auto initial=initial_payload(f.shape,c.before);
  f.put(c.before,initial);
  auto rejects=[&](auto invoke) {
    bool rejected=false;
    try { invoke(); } catch(const std::invalid_argument &) { rejected=true; }
    require(rejected,"expected synchronous refusal before enqueue");
    f.q.wait_and_throw(); f.rt->check();
    (void)check(f,in,c.before,initial);
  };
  for(const char *bad:{"1","257","-1","junk"}) {
    require(setenv("STRATA_SYCL_KV_COPY_GROUPS",bad,1)==0,"bad setting fixture");
    rejects([&] { in.submit(f); });
  }
  arm(128);
  rejects([&] {
    auto alias=f.pools;
    if(f.shape.fmt==0) alias.k_pool=f.host.k_pool;
    else if(f.shape.fmt==1) alias.k_q=f.host.k_q;
    else alias.k_q4=f.host.k_q4;
    kv_stream_resolve(f.map,alias,f.host,f.shape.fmt,in.ids.data(),in.steps.data(),1,in.cap,f.geometry,&f.q);
  });
  rejects([&] {
    auto alias=f.map; alias.miss_slot=alias.miss_block;
    kv_stream_resolve(alias,f.pools,f.host,f.shape.fmt,in.ids.data(),in.steps.data(),1,in.cap,f.geometry,&f.q);
  });
  rejects([&] {
    kv_stream_resolve(f.map,f.pools,f.host,f.shape.fmt,in.ids.data(),in.steps.data(),1,0,f.geometry,&f.q);
  });
  const auto expected=resolve(f.shape,c);
  for(int setting=0;setting<3;++setting) {
    if(setting==0) require(unsetenv("STRATA_SYCL_KV_COPY_GROUPS")==0,"unset default arm");
    else require(setenv("STRATA_SYCL_KV_COPY_GROUPS",setting==1?"":"0",1)==0,"empty/zero default arm");
    f.put(c.before,initial); f.q.wait_and_throw(); f.rt->check();
    kv_stream_resolve(f.map,f.pools,f.host,f.shape.fmt,in.ids.data(),in.steps.data(),1,in.cap,f.geometry,nullptr);
    (void)check(f,in,expected,expected_payload(f.shape,c.before,expected));
  }
  // This debug accessor is used only after the external producing queue drains.
  f.q.wait_and_throw(); f.rt->check();
  const auto counters=kv_stream_counters(f.map);
  require(counters.calls==u64(expected.ctl,8) && counters.lookups==u64(expected.ctl,6) &&
          counters.misses==u64(expected.ctl,4) && counters.overflow==(expected.ctl[3]!=0),
          "synchronous debug counter contract");
}
void graph_contract(Fixture &f) {
  namespace ex=sycl::ext::oneapi::experimental;
  for(int groups:{0,128,256,512,1024}) {
    auto c=cold(f.shape,513,"graph_first"); c.cap=f.shape.slots;
    Input in(f,c);
    arm(groups); f.put(c.before,initial_payload(f.shape,c.before));
    ex::command_graph<ex::graph_state::modifiable> graph(f.q.get_context(),f.q.get_device());
    graph.begin_recording(f.q);
    in.submit(f);
    graph.end_recording(f.q);
    auto exec=graph.finalize();
    // No host environment string is captured. Geometry is fixed at capture;
    // ctl[2], misses, widths and ids must be re-read for every replay.
    arm(groups?0:128);
    State previous=c.before;
    for(int pass=0;pass<4;++pass) {
      auto replay=cold(f.shape,pass<2?513:pass==2?1025:f.shape.slots,"graph_replay");
      replay.cap=c.cap; replay.before=previous;
      in.put(replay);
      const auto expected=resolve(f.shape,replay);
      const auto payload=expected_payload(f.shape,replay.before,expected);
      auto done=f.q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(exec); });
      f.rt->wait(done);
      const auto fingerprint=check(f,in,expected,payload);
      std::cout << "{\"phase\":\"graph_correctness\",\"fmt\":" << f.shape.fmt
                << ",\"captured_group_cap\":" << groups << ",\"replay\":" << pass
                << ",\"placed_pages\":" << expected.ctl[2] << ",\"oracle_exact\":true,\"fingerprint_fnv64\":"
                << fingerprint << "}\n";
      previous=expected;
    }
    // q is completed before exec/graph/input owners leave this scope.
    f.q.wait_and_throw(); f.rt->check();
  }
}
void validate(std::shared_ptr<Runtime> rt, sycl::queue &q, bool exhaustive) {
  for(int fmt:{0,1,2}) {
    {
      Shape tiny{1043,17,4,1,64,fmt}; Fixture tf(tiny,rt,q);
      for (const auto &c:semantics(tiny)) {
        Input in(tf,c); const auto expected=resolve(tiny,c);
        const auto initial=initial_payload(tiny,c.before), payload=expected_payload(tiny,c.before,expected);
        for (int groups:{0,128,256,512,1024}) {
          arm(groups); tf.put(c.before,initial); const auto t=timed(tf,in,false);
          const auto h=check(tf,in,expected,payload); record(tf,c,expected,groups,-1,false,t,h);
        }
      }
      tf.source_exact();
    }
    Shape s{1043,1031,1,1,64,fmt}; Fixture f(s,rt,q);
    reject_and_sync_contracts(f); graph_contract(f);
    std::vector<Case> cases=semantics(s);
    for(int n=0;n<=513;++n)
      if(exhaustive || n==0 || n==1 || n==2 || n==127 || n==128 || n==129 ||
         n==255 || n==256 || n==257 || n==511 || n==512 || n==513) cases.push_back(cold(s,n));
    cases.push_back(cold(s,1024)); cases.push_back(cold(s,1025));
    for(const auto &c:cases) {
      Input in(f,c); const auto expected=resolve(s,c);
      const auto initial=initial_payload(s,c.before), payload=expected_payload(s,c.before,expected);
      uint64_t baseline=0;
      for(int groups:{0,128,256,512,1024}) {
        arm(groups); f.put(c.before,initial); const auto t=timed(f,in,false);
        const auto h=check(f,in,expected,payload);
        if(!groups) baseline=h; else require(h==baseline,"legacy/candidate fingerprint exact");
        record(f,c,expected,groups,-1,false,t,h);
      }
    }
    f.source_exact();
  }
  // page_size=4: arbitrary IDs, T4 shape, both partial blocks, and context-tail address.
  // This bounded component allocation does not establish a model's 256K execution.
  for(int fmt:{0,1,2}) {
    Shape s{65536,8192,4,2,256,fmt}; Fixture f(s,rt,q);
    std::vector<Case> cases;
    for(int n:{0,1,513,1025,2052}) cases.push_back(typical_t4(s,n));
    cases.push_back(cold(s,8192,"fullslots_arbitrary"));
    Case tail("two_partial_pages_context_tail",s);
    tail.add({2,3,20,21,22,23,262140,262141,262142}); cases.push_back(tail);
    Case last("last_allocated_cell_262143",s);
    last.add({262143,0,262140,262143}); cases.push_back(last);
    cases.push_back(cold(s,513,"arbitrary_pages"));
    for(const auto &c:cases) {
      Input in(f,c); const auto expected=resolve(s,c);
      const auto initial=initial_payload(s,c.before), payload=expected_payload(s,c.before,expected);
      uint64_t baseline=0;
      for(int groups:{0,128,256,512,1024}) {
        arm(groups); f.put(c.before,initial); const auto t=timed(f,in,false);
        const auto h=check(f,in,expected,payload);
        if(!groups) baseline=h; else require(h==baseline,"real-shape legacy/candidate exact");
        record(f,c,expected,groups,-1,false,t,h);
      }
    }
    f.source_exact();
  }
}
std::unique_ptr<ExecGraph> capture(Fixture &f, Input &in, int groups) {
  f.q.wait_and_throw(); f.rt->check(); arm(groups);
  ex::command_graph<ex::graph_state::modifiable> graph(f.q.get_context(),f.q.get_device());
  graph.begin_recording(f.q); in.submit(f); graph.end_recording(f.q);
  return std::make_unique<ExecGraph>(graph.finalize());
}
void benchmark(std::shared_ptr<Runtime> rt,sycl::queue &q,int candidate,int samples,bool events,bool graph) {
  require(candidate!=0 && samples>=40 && samples<=80 && samples%4==0,"40..80 samples, multiple of 4");
  arm(candidate);
  for(int fmt:{0,1,2}) {
    Shape s{65536,8192,4,2,256,fmt}; Fixture f(s,rt,q);
    std::vector<Case> cases;
    for(int n:{0,1,513,2052}) cases.push_back(typical_t4(s,n));
    cases.push_back(cold(s,candidate+1,"cap_plus_one"));
    cases.push_back(cold(s,8192,"fullslots_arbitrary"));
    for(const auto &c:cases) {
      Input in(f,c); const auto expected=resolve(s,c);
      const auto initial=initial_payload(s,c.before), payload=expected_payload(s,c.before,expected);
      require(!expected.ctl[3],"benchmark overflow invalid");
      std::unique_ptr<ExecGraph> base_exec,candidate_exec;
      if(graph) {
        f.put(c.before,initial);
        base_exec=capture(f,in,0); candidate_exec=capture(f,in,candidate);
      }
      auto executable=[&](int groups) { return groups?candidate_exec.get():base_exec.get(); };
      // Separate warmup after restoring the exact initial state for every arm.
      for(int groups:{0,candidate}) {
        arm(groups); f.put(c.before,initial); (void)timed(f,in,events,executable(groups)); (void)check(f,in,expected,payload);
      }
      for(int sample=0;sample<samples;++sample) {
        const int phase=sample%4, groups=(phase==0||phase==3)?0:candidate;
        arm(groups); f.put(c.before,initial); const auto t=timed(f,in,events,executable(groups));
        const auto h=check(f,in,expected,payload);
        record(f,c,expected,groups,sample,events,t,h,graph);
      }
    }
    f.source_exact();
  }
}
}
int main(int argc,char **argv) {
  try {
    const std::string mode=argc>1?argv[1]:"--help";
    if(mode=="--help") {
      require(argc<=2,"help takes no extra args");
      std::cout << "sycl_kv_copy_groups_bench --validate|--validate-exhaustive|--benchmark GROUPS SAMPLES|--event-benchmark GROUPS SAMPLES|--graph-benchmark GROUPS SAMPLES|--graph-event-benchmark GROUPS SAMPLES\n";
      return 0;
    }
    const bool graph=mode=="--graph-benchmark"||mode=="--graph-event-benchmark";
    const bool events=mode=="--event-benchmark"||mode=="--graph-event-benchmark";
    const bool bench=mode=="--benchmark"||mode=="--event-benchmark"||graph;
    require((bench&&argc==4)||((mode=="--validate"||mode=="--validate-exhaustive")&&argc==2),"invalid arguments");
    int candidate=0,samples=0;
    if(bench) { candidate=std::stoi(argv[2]); samples=std::stoi(argv[3]); arm(candidate);
      require(candidate!=0&&samples>=40&&samples<=80&&samples%4==0,"benchmark bounds"); }
    frozen_env();
    auto rt=strata::sycl_backend::runtime_for(); auto q=rt->make_queue(events);
    require(rt->device().is_gpu(),"actual GPU required");
    std::cerr << "device=" << rt->device().get_info<sycl::info::device::name>() << "\n";
    if(bench) benchmark(rt,q,candidate,samples,events,graph);
    else validate(rt,q,mode=="--validate-exhaustive");
    q.wait_and_throw(); rt->check();
    std::cerr << "PASS bounded component scope only\n"; return 0;
  } catch(const std::exception &e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
