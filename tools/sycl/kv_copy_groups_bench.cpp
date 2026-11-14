// Component fixture invoking the production resolve implementation by explicit geometry.
// No model, RESTORE, attention arithmetic, or actual-decode performance claim.
#include "kv_copy_groups_oracle.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_stream_copy_groups.hpp"
#include "strata/sycl/runtime.hpp"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <cstdio>
#include <unistd.h>
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
void check_groups(uint32_t groups) { validate_kv_stream_copy_group_cap(groups); }
// No stack unwinding or voluntary USM/staging release after unknown completion.
// The external Main-owned deadline terminates this held process as INCOMPLETE.
[[noreturn]] void owner_hold(const char *label) noexcept {
  std::fputs("INCOMPLETE owner_hold: ",stderr); std::fputs(label,stderr);
  std::fputs("; no release/retry, awaiting owner termination\n",stderr); std::fflush(stderr); std::fflush(stdout);
  for (;;) (void)::pause();
}
template<class F> void pending_or_hold(const char *label,F &&f) noexcept {
  try { f(); } catch (...) { owner_hold(label); }
}
void checked_queue(std::shared_ptr<Runtime> rt,sycl::queue &q,const char *label) noexcept {
  pending_or_hold(label,[&] { q.wait_and_throw(); rt->check(); });
}
void checked_all(std::shared_ptr<Runtime> rt,sycl::queue &q,const char *label) noexcept {
  pending_or_hold(label,[&] { q.wait_and_throw(); rt->wait(); rt->check(); });
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
  ~Buffer() noexcept { checked_all(rt,q,"buffer teardown"); }
  T *data() { return allocation.template as<T>() + Guard; }
  void put(const std::vector<T> &v) {
    require(v.size() == count, "buffer upload size");
    std::vector<T> all(count+2*Guard, sentinel);
    std::copy(v.begin(), v.end(), all.begin()+Guard);
    // all stays in this frame if submit or completion checking throws.
    pending_or_hold("buffer put host staging",[&] {
      auto done=q.memcpy(allocation.data(),all.data(),all.size()*sizeof(T));
      rt->wait(done);
    });
  }
  std::vector<T> get() {
    std::vector<T> all(count+2*Guard);
    // Retain readback storage too; an exception is not proof of completion.
    pending_or_hold("buffer get host staging",[&] {
      auto done=q.memcpy(all.data(),allocation.data(),all.size()*sizeof(T));
      rt->wait(done);
    });
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
      // Checked completion of external and runtime queues precedes every
      // direct host-USM write. No source owner is released on unknown status.
      checked_all(rt,q,"before host source initialization");
      auto *base = src.back()->allocation.template as<uint8_t>();
      std::fill(base, base+Guard, uint8_t(0xd3));
      std::fill(base+Guard+src.back()->count, base+2*Guard+src.back()->count, uint8_t(0xd3));
      for (int b = 0; b < s.blocks; ++b)
        for (size_t i = 0; i < len; ++i) src.back()->data()[size_t(b)*len+i] = pattern(int(a), b, i);
      for (size_t i=0;i<Guard;++i)
        require(base[i]==0xd3 && base[Guard+src.back()->count+i]==0xd3,"source initialization guards");
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
  void put_map(const State &m) {
    table.put(m.table); block.put(m.block); stamp.put(m.stamp); ref.put(m.ref);
    ctl.put(m.ctl); mb.put(m.mb); ms.put(m.ms);
  }
  void put(const State &m, const std::vector<std::vector<uint8_t>> &p) {
    put_map(m);
    for (size_t a = 0; a < p.size(); ++a) dst[a]->put(p[a]);
  }
  State get() {
    State m(shape);
    m.table=table.get(); m.block=block.get(); m.stamp=stamp.get(); m.ref=ref.get();
    m.ctl=ctl.get(); m.mb=mb.get(); m.ms=ms.get(); return m;
  }
  void source_exact() {
    checked_all(rt,q,"before host source inspection");
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
  void submit(Fixture &f,uint32_t groups,bool cooperative=false) {
    const auto resolve = cooperative ? kv_stream_resolve_with_coalesced_plan
                                     : kv_stream_resolve_with_copy_groups;
    resolve(f.map,f.pools,f.host,f.shape.fmt,
        ids.data(),steps.data(),nq,cap,f.geometry,&f.q,groups);
  }
};
struct Timing { uint64_t submit=0, marker_submit=0, whole=0, device_envelope=0; };
Timing timed(Fixture &f, Input &in, uint32_t groups, bool events, ExecGraph *graph=nullptr,
             bool cooperative=false) {
  try {
  checked_queue(f.rt,f.q,"before timed operation"); // outside measurement
  sycl::event begin, end;
  const auto t0 = Clock::now();
  if (events) begin = f.q.single_task([] {});
  const auto t1 = Clock::now();
  if (graph) (void)f.q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(*graph); });
  else in.submit(f,groups,cooperative);
  const auto t2 = Clock::now();
  if (events) end = f.q.single_task([] {});
  const auto t3 = Clock::now();
  checked_queue(f.rt,f.q,"timed operation completion");
  const auto t4 = Clock::now();
  Timing t{ns(t1,t2), ns(t0,t1)+ns(t2,t3), ns(t0,t4), 0};
  if (events) {
    const auto lo = begin.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto hi = end.get_profiling_info<sycl::info::event_profiling::command_start>();
    require(hi >= lo, "device event interval order"); t.device_envelope = hi-lo;
  }
  return t;
  } catch (...) { owner_hold("timed submit/event boundary"); }
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
void reject_and_sync_contracts(Fixture &f,uint32_t startup_groups,bool cooperative=false) {
  const auto resolve_api = cooperative ? kv_stream_resolve_with_coalesced_plan
                                       : kv_stream_resolve_with_copy_groups;
  auto c=cold(f.shape,1,"refusals_and_nullstream"); Input in(f,c);
  const auto initial=initial_payload(f.shape,c.before);
  f.put(c.before,initial);
  auto rejects=[&](auto invoke) {
    bool rejected=false;
    try { invoke(); }
    catch(const std::invalid_argument &) { rejected=true; }
    catch(...) { owner_hold("unexpected refusal-path runtime exception"); }
    checked_queue(f.rt,f.q,"after refusal probe");
    require(rejected,"expected synchronous refusal before enqueue");
    (void)check(f,in,c.before,initial);
  };
  for(uint32_t bad:{1u,257u,UINT32_MAX}) rejects([&] { in.submit(f,bad,cooperative); });
  rejects([&] {
    auto alias=f.pools;
    if(f.shape.fmt==0) alias.k_pool=f.host.k_pool;
    else if(f.shape.fmt==1) alias.k_q=f.host.k_q;
    else alias.k_q4=f.host.k_q4;
    resolve_api(f.map,alias,f.host,f.shape.fmt,in.ids.data(),in.steps.data(),1,in.cap,f.geometry,&f.q,128);
  });
  rejects([&] {
    auto alias=f.map; alias.miss_slot=alias.miss_block;
    resolve_api(alias,f.pools,f.host,f.shape.fmt,in.ids.data(),in.steps.data(),1,in.cap,f.geometry,&f.q,128);
  });
  rejects([&] {
    resolve_api(f.map,f.pools,f.host,f.shape.fmt,in.ids.data(),in.steps.data(),1,0,f.geometry,&f.q,128);
  });
  const auto expected=resolve(f.shape,c);
  const auto payload=expected_payload(f.shape,c.before,expected);
  // The wrapper's environment is immutable; compare it to the explicit policy
  // selected before Runtime construction, with the same addresses and oracle.
  for(bool production_wrapper:{false,true}) {
    f.put(c.before,initial); checked_queue(f.rt,f.q,"before API equivalence");
    pending_or_hold("API equivalence submission",[&] {
      if(production_wrapper)
        kv_stream_resolve(f.map,f.pools,f.host,f.shape.fmt,in.ids.data(),in.steps.data(),1,in.cap,f.geometry,&f.q);
      else in.submit(f,startup_groups,cooperative);
    });
    checked_queue(f.rt,f.q,"API equivalence completion");
    (void)check(f,in,expected,payload);
  }
  f.put(c.before,initial); checked_all(f.rt,f.q,"before null-stream probe");
  pending_or_hold("null-stream probe",[&] {
    resolve_api(f.map,f.pools,f.host,f.shape.fmt,
        in.ids.data(),in.steps.data(),1,in.cap,f.geometry,nullptr,128);
  });
  (void)check(f,in,expected,payload);
  // Read counters through guarded Buffer::get instead of the pre-existing
  // debug accessor's stack-backed async memcpy. That accessor is unchanged.
  checked_all(f.rt,f.q,"before guarded counter snapshot");
  const auto values=f.ctl.get();
  require(u64(values,8)==u64(expected.ctl,8) && u64(values,6)==u64(expected.ctl,6) &&
          u64(values,4)==u64(expected.ctl,4) && values[3]==expected.ctl[3],
          "guarded counter snapshot contract");
}
void graph_contract(Fixture &f,bool cooperative=false) {
  namespace ex=sycl::ext::oneapi::experimental;
  for(int groups:{0,128,256,512,1024}) {
    auto c=cold(f.shape,513,"graph_first"); c.cap=f.shape.slots;
    Input in(f,c);
    check_groups(groups); f.put(c.before,initial_payload(f.shape,c.before));
    ex::command_graph<ex::graph_state::modifiable> graph(f.q.get_context(),f.q.get_device());
    graph.begin_recording(f.q);
    in.submit(f,groups,cooperative);
    graph.end_recording(f.q);
    auto exec=graph.finalize();
    // Explicit geometry is fixed at capture; no environment mutation occurs.
    // ctl[2], misses, widths and ids must be re-read for every replay.
    State previous=c.before;
    for(int pass=0;pass<4;++pass) {
      auto replay=cold(f.shape,pass<2?513:pass==2?1025:f.shape.slots,"graph_replay");
      replay.cap=c.cap; replay.before=previous;
      in.put(replay);
      const auto expected=resolve(f.shape,replay);
      const auto payload=expected_payload(f.shape,replay.before,expected);
      pending_or_hold("graph correctness replay",[&] {
        auto done=f.q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(exec); });
        f.rt->wait(done);
      });
      const auto fingerprint=check(f,in,expected,payload);
      std::cout << "{\"phase\":\"graph_correctness\",\"fmt\":" << f.shape.fmt
                << ",\"captured_group_cap\":" << groups << ",\"replay\":" << pass
                << ",\"placed_pages\":" << expected.ctl[2] << ",\"oracle_exact\":true,\"fingerprint_fnv64\":"
                << fingerprint << "}\n";
      previous=expected;
    }
    // q is completed before exec/graph/input owners leave this scope.
    checked_queue(f.rt,f.q,"graph correctness scope completion");
  }
}
// Main-authored extension of the owned fixture/oracle. Keep the candidate's
// graph parameters fixed while refreshing input contents and persistent state.
void planner_context_graph(Fixture &f) {
  require(f.shape.blocks==65536 && f.shape.page==4 && f.shape.slots>=1025,
          "fixed physical-context planner fixture geometry");
  Case base("planner_context_tail",f.shape);
  for(int qi=0;qi<4;++qi) {
    std::vector<int32_t> row;
    for(int i=0;i<257;++i)
      row.push_back((i/4+qi*7)*f.shape.page+i%f.shape.page);
    row.back()=f.shape.blocks*f.shape.page-1;
    base.add(std::move(row));
  }
  Input in(f,base);
  f.put(base.before,initial_payload(f.shape,base.before));
  ex::command_graph<ex::graph_state::modifiable> graph(f.q.get_context(),f.q.get_device());
  graph.begin_recording(f.q); in.submit(f,0,true); graph.end_recording(f.q);
  auto executable=graph.finalize();
  State previous=base.before;
  for(int pass=0;pass<5;++pass) {
    auto c=base;c.before=previous;
    if(pass==1) c.widths={1,255,256,257};
    if(pass==2) c.widths={0,0,0,0};
    if(pass==3) c.before.ctl[0]=INT_MAX;
    if(pass==4) c.widths={c.cap+1,0,0,0};
    const auto expected=resolve(f.shape,c);
    const auto payload=expected_payload(f.shape,c.before,expected);
    f.put(c.before,initial_payload(f.shape,c.before)); in.put(c);
    pending_or_hold("planner full-context graph replay",[&] {
      auto done=f.q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(executable); });
      f.rt->wait(done);
    });
    const auto fingerprint=check(f,in,expected,payload);
    std::cout << "{\"phase\":\"planner_context_graph\",\"fmt\":" << f.shape.fmt
              << ",\"last_cell\":262143,\"replay\":" << pass
              << ",\"oracle_exact\":true,\"fingerprint_fnv64\":" << fingerprint << "}\n";
    previous=expected;
  }
  checked_all(f.rt,f.q,"planner full-context graph completion");
}
void validate(std::shared_ptr<Runtime> rt, sycl::queue &q, bool exhaustive,
              uint32_t startup_groups,bool cooperative=false) {
  for(int fmt:{0,1,2}) {
    {
      Shape tiny{1043,17,4,1,64,fmt}; Fixture tf(tiny,rt,q);
      auto cases=semantics(tiny);
      if(cooperative) for(bool resident:{false,true}) {
        Case gap(resident?"resident_invalid_gap_across256":"miss_invalid_gap_across256",tiny);
        std::vector<int32_t> row(257,tiny.page);row[255]=-1;
        gap.add(std::move(row));gap.add({tiny.page});
        if(resident) seed(gap.before,1,0);
        cases.push_back(std::move(gap));
      }
      for (const auto &c:cases) {
        Input in(tf,c); const auto expected=resolve(tiny,c);
        const auto initial=initial_payload(tiny,c.before), payload=expected_payload(tiny,c.before,expected);
        for (int groups:{0,128,256,512,1024}) {
          check_groups(groups); tf.put(c.before,initial); const auto t=timed(tf,in,groups,false,nullptr,cooperative);
          const auto h=check(tf,in,expected,payload); record(tf,c,expected,groups,-1,false,t,h);
        }
      }
      tf.source_exact();
    }
    Shape s{1043,1031,1,1,64,fmt}; Fixture f(s,rt,q);
    reject_and_sync_contracts(f,startup_groups,cooperative); graph_contract(f,cooperative);
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
        check_groups(groups); f.put(c.before,initial); const auto t=timed(f,in,groups,false,nullptr,cooperative);
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
        check_groups(groups); f.put(c.before,initial); const auto t=timed(f,in,groups,false,nullptr,cooperative);
        const auto h=check(f,in,expected,payload);
        if(!groups) baseline=h; else require(h==baseline,"real-shape legacy/candidate exact");
        record(f,c,expected,groups,-1,false,t,h);
      }
    }
    if(cooperative) planner_context_graph(f);
    f.source_exact();
  }
}
std::unique_ptr<ExecGraph> capture(Fixture &f, Input &in, int groups,bool cooperative=false) {
  checked_queue(f.rt,f.q,"before graph capture"); check_groups(groups);
  ex::command_graph<ex::graph_state::modifiable> graph(f.q.get_context(),f.q.get_device());
  graph.begin_recording(f.q); in.submit(f,groups,cooperative); graph.end_recording(f.q);
  return std::make_unique<ExecGraph>(graph.finalize());
}
void planner_smoke(std::shared_ptr<Runtime> rt,sycl::queue &q) {
  Shape shape{1043,1031,4,1,64,kKvInt8}; Fixture f(shape,rt,q);
  const auto cases=semantics(shape);
  size_t checked=0;
  for(const auto &c:cases) {
    Input in(f,c); const auto expected=resolve(shape,c);
    f.put(c.before,initial_payload(shape,c.before));
    pending_or_hold("planner bounded smoke",[&] { in.submit(f,0,true); });
    checked_all(rt,q,"planner bounded smoke completion");
    (void)check(f,in,expected,expected_payload(shape,c.before,expected)); ++checked;
  }
  reject_and_sync_contracts(f,0,true);graph_contract(f,true);f.source_exact();
  require(checked==14,"all fourteen bounded planner semantics required");
  std::cout << "planner_smoke=PASS direct_cases=" << checked << " graph_replays=20\n";
}
// Synthetic disjoint selections bound the miss range; the model's actual hit
// distribution and inter-query overlap are not inferred from these cases.
Case planner_case(const Shape &s,int nq,int missing) {
  require((nq==2||nq==4) && s.slots==5120,"current streamed-cache and target windows");
  auto c=typical_t4(s,0);c.rows.resize(size_t(nq));c.widths.resize(size_t(nq));
  c.before=State(s);c.name="planner_T"+std::to_string(nq)+"_miss_"+std::to_string(missing);
  require(c.cap==qsa_selection_width(262144,qsa_real_shapes()),"production selection capacity");
  std::vector<int> selected;std::vector<bool> named(size_t(s.blocks),false);
  for(const auto &row:c.rows) for(int cell:row) {
    const int b=cell/s.page;
    if(!named[size_t(b)]) { named[size_t(b)]=true;selected.push_back(b); }
  }
  require(selected.size()==size_t(513*nq) && missing>=0 && missing<=int(selected.size()),"selection/miss bounds");
  int count=0;
  auto place=[&](int b) { seed(c.before,b,(count++*521)%s.slots); };
  for(int i=0;i<int(selected.size())-missing;++i) place(selected[size_t(i)]);
  // All resident pages lie within the already processed first32K cells.
  for(int b=0;b<8192 && count<s.slots;++b) if(!named[size_t(b)]) place(b);
  require(count==s.slots,"full initial cache");
  c.before.ctl[0]=9;c.before.ctl[1]=3917;return c;
}
void planner_perf_validation(std::shared_ptr<Runtime> rt,sycl::queue &q) {
  const Shape s{65536,5120,4,2,256,kKvInt8};Fixture f(s,rt,q);
  for(int nq:{2,4}) for(int pattern:{0,1,2}) {
    const int missing=pattern==0?0:pattern==1?513*nq/4:513*nq;
    const auto c=planner_case(s,nq,missing);Input in(f,c);
    const auto expected=resolve(s,c),initial=c.before;
    const auto payload=expected_payload(s,initial,expected);
    require(!expected.ctl[3] && expected.ctl[2]==missing,"exact planned placement count");
    for(bool cooperative:{false,true}) {
      f.put(initial,initial_payload(s,initial));auto graph=capture(f,in,0,cooperative);
      // Payload writes are idempotent; only map/control state is reset later.
      for(int replay=0;replay<2;++replay) {
        f.put_map(initial);(void)timed(f,in,0,false,graph.get());
        (void)check(f,in,expected,payload);
      }
      std::cout << "{\"phase\":\"planner_perf_cases\",\"nq\":" << nq
        << ",\"missing\":" << missing << ",\"cooperative\":" << (cooperative?"true":"false")
        << ",\"replays\":2,\"oracle_exact\":true}\n";
    }
  }
  f.source_exact();
}
double planner_clock_pair() {
  uint64_t total=0;
  for(int i=0;i<10000;++i) { const auto a=Clock::now(),b=Clock::now();total+=ns(a,b); }
  return double(total)/10000.;
}
void planner_benchmark(std::shared_ptr<Runtime> rt,sycl::queue &q,bool cooperative) {
  const Shape s{65536,5120,4,2,256,kKvInt8};Fixture f(s,rt,q);
  const double pair_ns=planner_clock_pair();int samples=0;
  for(int nq:{2,4}) for(int pattern:{0,1,2}) {
    const int missing=pattern==0?0:pattern==1?513*nq/4:513*nq;
    const auto c=planner_case(s,nq,missing);
    Input in(f,c);const auto expected=resolve(s,c);
    const auto payload=expected_payload(s,c.before,expected);
    require(!expected.ctl[3] && expected.ctl[2]==missing,"planned benchmark state");
    f.put(c.before,initial_payload(s,c.before));auto graph=capture(f,in,0,cooperative);
    auto batch=[&](unsigned calls) {
      uint64_t total=0;
      for(unsigned i=0;i<calls;++i) {
        f.put_map(c.before); // completed reset outside every timed interval
        total+=timed(f,in,0,false,graph.get()).whole;
      }
      return double(total);
    };
    double warmup=0;do { warmup+=batch(64); } while(warmup<500000000.);
    (void)check(f,in,expected,payload);
    unsigned calls=64;double calibration=0;
    for(;;) {
      calibration=batch(calls);
      if(calibration>=750000000.) break;
      require(calls<16384,"planner calibration cap");calls*=2;
    }
    for(int rep=0;rep<3;++rep) {
      const double elapsed=batch(calls);(void)check(f,in,expected,payload);
      // Charge all five clock snapshots inside timed() at the measured cost.
      const double fraction=5.*pair_ns*calls/elapsed;
      require(elapsed>=250000000. && fraction<=.01,"eligible planner duration/timer fraction");
      std::cout << "{\"phase\":\"planner_sample\",\"nq\":" << nq << ",\"missing\":" << missing
        << ",\"unique_pages\":" << 513*nq << ",\"cooperative\":" << (cooperative?"true":"false")
        << ",\"rep\":" << rep << ",\"calls\":" << calls << ",\"elapsed_ns\":" << elapsed
        << ",\"ns_per_call\":" << elapsed/calls << ",\"clock_pair_ns\":" << pair_ns
        << ",\"timer_fraction\":" << fraction << ",\"calibration_ns\":" << calibration
        << ",\"warmup_ns\":" << warmup << ",\"requested_host_payload_bytes_per_call\":"
        << uint64_t(missing)*s.block_bytes() << ",\"oracle_exact\":true,\"graph_nodes\":2}\n";
      ++samples;
    }
    checked_all(rt,q,"planner benchmark graph retirement");
  }
  require(samples==18,"complete planner timing matrix");f.source_exact();
  std::cout << "planner_benchmark=PASS samples=18 timing=sum_of_isolated_graph_launch_and_wait reset_and_oracle=excluded physical_bus_bytes=unknown\n";
}
void benchmark(std::shared_ptr<Runtime> rt,sycl::queue &q,int candidate,int samples,bool events,bool graph) {
  require(candidate!=0 && samples>=40 && samples<=80 && samples%4==0,"40..80 samples, multiple of 4");
  check_groups(candidate);
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
        check_groups(groups); f.put(c.before,initial); (void)timed(f,in,groups,events,executable(groups)); (void)check(f,in,expected,payload);
      }
      for(int sample=0;sample<samples;++sample) {
        const int phase=sample%4, groups=(phase==0||phase==3)?0:candidate;
        check_groups(groups); f.put(c.before,initial); const auto t=timed(f,in,groups,events,executable(groups));
        const auto h=check(f,in,expected,payload);
        record(f,c,expected,groups,sample,events,t,h,graph);
      }
    }
    f.source_exact();
  }
}
}
int main(int argc,char **argv) {
  std::cout << std::setprecision(17) << std::unitbuf;
  try {
    const std::string mode=argc>1?argv[1]:"--help";
    if(mode=="--host-env") {
      require(argc==2,"host-env takes no extra args");
      const auto cap=kv_stream_copy_group_cap_from_text(std::getenv("STRATA_SYCL_KV_COPY_GROUPS"));
      std::cout << "copy_group_cap=" << cap << "; no Runtime construction\n"; return 0;
    }
    if(mode=="--help") {
      require(argc<=2,"help takes no extra args");
      std::cout << "sycl_kv_copy_groups_bench --host-env|--validate|--validate-exhaustive|--planner-smoke|--planner-validate|--planner-benchmark scalar|cooperative|--benchmark GROUPS SAMPLES|--event-benchmark GROUPS SAMPLES|--graph-benchmark GROUPS SAMPLES|--graph-event-benchmark GROUPS SAMPLES\n";
      return 0;
    }
    const bool graph=mode=="--graph-benchmark"||mode=="--graph-event-benchmark";
    const bool events=mode=="--event-benchmark"||mode=="--graph-event-benchmark";
    const bool bench=mode=="--benchmark"||mode=="--event-benchmark"||graph;
    const bool smoke=mode=="--planner-smoke";
    const bool planner_speed=mode=="--planner-benchmark";
    const bool planner=mode=="--planner-validate"||smoke||planner_speed;
    require((bench&&argc==4)||(planner_speed&&argc==3)||
        ((mode=="--validate"||mode=="--validate-exhaustive"||planner)&&!planner_speed&&argc==2),"invalid arguments");
    if(planner_speed) require(std::string(argv[2])=="scalar"||std::string(argv[2])=="cooperative","planner arm required");
    int candidate=0,samples=0;
    if(bench) { candidate=std::stoi(argv[2]); samples=std::stoi(argv[3]); check_groups(candidate);
      require(candidate!=0&&samples>=40&&samples<=80&&samples%4==0,"benchmark bounds"); }
    // Read-only snapshots happen before constructing Runtime and queues.
    const auto startup_groups=kv_stream_copy_group_cap_from_text(std::getenv("STRATA_SYCL_KV_COPY_GROUPS"));
    if(planner) require(startup_groups==0,"planner screen preserves production copy geometry0");
    frozen_env();
    auto rt=strata::sycl_backend::runtime_for(); auto q=rt->make_queue(events);
    require(rt->device().is_gpu(),"actual GPU required");
    if(planner) require(q.has_property<sycl::property::queue::in_order>() &&
        rt->device().get_info<sycl::info::device::max_work_group_size>()>=256 &&
        rt->device().get_info<sycl::info::device::name>().find("B570")!=std::string::npos,
        "planner screen requires the qualified B570/in-order/workgroup256 setup");
    std::cerr << "device=" << rt->device().get_info<sycl::info::device::name>() << "\n";
    if(planner_speed) planner_benchmark(rt,q,std::string(argv[2])=="cooperative");
    else if(smoke) planner_smoke(rt,q);
    else if(bench) benchmark(rt,q,candidate,samples,events,graph);
    else validate(rt,q,mode=="--validate-exhaustive",startup_groups,planner);
    if(mode=="--planner-validate") planner_perf_validation(rt,q);
    checked_all(rt,q,"final completion");
    if(planner) {
      std::cout << "KV_PLAN_WAIT\n" << std::flush;
      std::string reply;
      require(bool(std::getline(std::cin,reply)) && reply=="QUIT","Main-owned provider/retirement gate required");
    }
    std::cerr << "PASS bounded component scope only\n"; return 0;
  } catch(const std::exception &e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
