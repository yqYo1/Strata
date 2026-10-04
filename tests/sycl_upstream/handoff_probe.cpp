// B570 visibility/optimizer diagnostic; see docs/sycl-audit/runtime-handoff.md.
// The no-fence variant intentionally reproduces a failed wait; it is not a CTest.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <atomic>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <thread>
namespace es = sycl::ext::intel::esimd;
struct alignas(64) Flag { uint32_t value; char pad[60]; };
struct State { Flag seq, ack; alignas(64) uint32_t sent[16], reply[16], result[16]; uint32_t timedout; };
int main(int argc, char** argv) try {
  const bool esimd = argc > 1;
  constexpr bool poll_fence = STRATA_HANDOFF_POLL_FENCE;
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
  auto d=q.get_device();
  std::cout << "device=" << d.get_info<sycl::info::device::name>() << " host_atomic=" << d.has(sycl::aspect::usm_atomic_host_allocations) << " shared_atomic=" << d.has(sycl::aspect::usm_atomic_shared_allocations) << " scopes=";
  for (auto s:d.get_info<sycl::info::device::atomic_memory_scope_capabilities>()) std::cout << int(s) << ',';
  std::cout << " esimd=" << esimd << " poll_fence=" << poll_fence << std::endl;
  State* p=sycl::aligned_alloc_host<State>(64,1,q);
  if (!p) throw std::bad_alloc();
  *p={};
  bool seen=false, payload=false;
  std::atomic<bool> ready{false};
  auto worker=std::jthread([&] {
    volatile State* v=p;
    ready.store(true);
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(v->seq.value!=1 && std::chrono::steady_clock::now()<deadline) _mm_pause();
    seen=v->seq.value==1; payload=true;
    const auto release=std::chrono::steady_clock::now()+std::chrono::milliseconds(1);
    while(std::chrono::steady_clock::now()<release) _mm_pause();
    for(int i=0;i<16;++i) {payload &= v->sent[i]==100u+i; v->reply[i]=1000+i;}
    _mm_sfence(); v->ack.value=1;
  });
  while(!ready.load()) _mm_pause();
  auto t=std::chrono::steady_clock::now();
  sycl::event e;
  if (!esimd) e=q.single_task([=] {
    volatile State* v=p;
    for(int i=0;i<16;++i) v->sent[i]=100+i;
    sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::system);
    v->seq.value=1;
    sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::system);
    uint32_t n=0;
    for(;n<1000000 && v->ack.value!=1;++n) {
      if constexpr (poll_fence) sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::system);
    }
    sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::system);
    for(int i=0;i<16;++i) v->result[i]=v->reply[i];
    v->timedout=(n==1000000);
  });
  else e=q.single_task([=]() SYCL_ESIMD_KERNEL {
    es::simd<uint32_t,16> sent(100,1);
    es::block_store(p->sent,sent,es::properties{es::cache_hint_L1<es::cache_hint::uncached>,es::cache_hint_L2<es::cache_hint::uncached>});
    es::fence<es::memory_kind::global,es::fence_flush_op::none,es::fence_scope::system>();
    es::simd<uint32_t,1> offsets(0),one(1);
    es::scatter<uint32_t,1>(&p->seq.value,offsets,one,es::properties{es::cache_hint_L1<es::cache_hint::uncached>,es::cache_hint_L2<es::cache_hint::uncached>});
    es::fence<es::memory_kind::global,es::fence_flush_op::none,es::fence_scope::system>();
    uint32_t n=0;
    for(;n<1000000;++n) {
      auto ack=es::gather<uint32_t,1>(&p->ack.value,offsets,es::properties{es::cache_hint_L1<es::cache_hint::uncached>,es::cache_hint_L2<es::cache_hint::uncached>});
      if(ack[0]==1) break;
      if constexpr (poll_fence) es::fence<es::memory_kind::global,es::fence_flush_op::none,es::fence_scope::group>();
    }
    es::fence<es::memory_kind::global,es::fence_flush_op::none,es::fence_scope::system>();
    auto reply=es::block_load<uint32_t,16>(p->reply,es::properties{es::cache_hint_L1<es::cache_hint::uncached>,es::cache_hint_L2<es::cache_hint::uncached>});
    es::block_store(p->result,reply);
    p->timedout=(n==1000000);
  });
  const double submitted=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
  worker.join(); e.wait_and_throw();
  bool answer=true;for(int i=0;i<16;++i) answer &= p->result[i]==1000u+i;
  std::cout << "submitted_seconds=" << submitted << " host_seen=" << seen << " host_payload=" << payload << " gpu_payload=" << answer << " gpu_timeout=" << p->timedout << std::endl;
  bool ok=seen&&payload&&answer&&!p->timedout;
  sycl::free(p,q);
  return !ok;
} catch(std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
