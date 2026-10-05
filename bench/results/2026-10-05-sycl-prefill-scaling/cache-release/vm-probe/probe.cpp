#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>
#include <cstring>
#include <stdexcept>

int main() try {
 namespace v=sycl::ext::oneapi::experimental;
 sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
 std::printf("device=%s virtual_mem=%d\n",q.get_device().get_info<sycl::info::device::name>().c_str(),q.get_device().has(sycl::aspect::ext_oneapi_virtual_mem));std::fflush(stdout);
 if(!q.get_device().has(sycl::aspect::ext_oneapi_virtual_mem))return 2;
 const size_t gran=v::get_mem_granularity(q.get_device(),q.get_context());
 for(size_t mib:{64,320,1536}) {
  const size_t bytes=((mib*1024*1024+gran-1)/gran)*gran,n=bytes/4;
  auto address=v::reserve_virtual_mem(bytes,q.get_context());
  auto physical=std::make_unique<v::physical_mem>(q,bytes);
  auto* ptr=static_cast<uint32_t*>(physical->map(address,bytes,v::address_access_mode::read_write));
  std::vector<uint32_t> expected(n),backup(n),got(n);
  for(size_t i=0;i<n;++i)expected[i]=uint32_t(i*2654435761u)^0x713AF0C5u;
  q.memcpy(ptr,expected.data(),bytes).wait_and_throw();
  v::command_graph graph(q.get_context(),q.get_device());
  graph.add([=](sycl::handler& cgh){cgh.parallel_for<class CacheVmXor>(sycl::range<1>(n),[=](sycl::id<1> i){ptr[i]^=0xA5A5A5A5u;});});
  auto captured=graph.finalize();
  for(int round=0;round<5;++round) {
   const auto start=std::chrono::steady_clock::now();
   q.memcpy(backup.data(),ptr,bytes).wait_and_throw();
   const auto saved=std::chrono::steady_clock::now();
   v::unmap(ptr,bytes,q.get_context());physical.reset();
   const auto released=std::chrono::steady_clock::now();
   physical=std::make_unique<v::physical_mem>(q,bytes);
   auto* restored=physical->map(address,bytes,v::address_access_mode::read_write);
   if(restored!=ptr)throw std::runtime_error("cache address changed");
   const auto allocated=std::chrono::steady_clock::now();
   q.memcpy(ptr,backup.data(),bytes).wait_and_throw();
   const auto complete=std::chrono::steady_clock::now();
   if(std::memcmp(backup.data(),expected.data(),bytes))throw std::runtime_error("backup bytes differ");
   q.ext_oneapi_graph(captured).wait_and_throw();
   q.memcpy(got.data(),ptr,bytes).wait_and_throw();
   for(size_t i=0;i<n;++i)if(got[i]!=(expected[i]^0xA5A5A5A5u))throw std::runtime_error("captured kernel sees wrong restored bytes");
   q.ext_oneapi_graph(captured).wait_and_throw();
   q.memcpy(got.data(),ptr,bytes).wait_and_throw();
   if(std::memcmp(got.data(),expected.data(),bytes))throw std::runtime_error("full restored bytes differ");
   auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
   std::printf("{\"bytes\":%zu,\"granularity\":%zu,\"round\":%d,\"d2h_ms\":%.6f,\"release_ms\":%.6f,\"allocate_map_ms\":%.6f,\"h2d_ms\":%.6f,\"total_ms\":%.6f,\"same_address\":true,\"all_bytes_identical\":true,\"captured_graph_replay_correct\":true}\n",bytes,gran,round,ms(start,saved),ms(saved,released),ms(released,allocated),ms(allocated,complete),ms(start,complete));std::fflush(stdout);
  }
  q.wait_and_throw();v::unmap(ptr,bytes,q.get_context());physical.reset();v::free_virtual_mem(address,bytes,q.get_context());
 }
 return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"cache VM probe failed: %s\n",e.what());return 1;}
