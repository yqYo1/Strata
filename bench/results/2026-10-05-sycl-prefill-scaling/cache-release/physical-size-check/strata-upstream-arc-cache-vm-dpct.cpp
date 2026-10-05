#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#include <cstdio>
#include <dpct/dpct.hpp>
#include <vector>
#include <memory>
int main() {
 namespace v=sycl::ext::oneapi::experimental;
 auto& q=dpct::get_in_order_queue();
 auto gran=v::get_mem_granularity(q.get_device(),q.get_context());
 std::printf("gran=%zu\n",gran);
 for(size_t total : {size_t(5<<20),size_t(6<<20),size_t(64<<20),size_t((64<<20)+65536),size_t(325<<20)}) {
  uintptr_t address=0; size_t mapped=0;
  std::vector<std::unique_ptr<v::physical_mem>> blocks;
  try {
   address=v::reserve_virtual_mem(total,q.get_context());
   for(size_t at=0;at<total;at+=64<<20) {
    size_t bytes=std::min<size_t>(64<<20,total-at);
    std::printf("total=%zu address=%p offset=%zu physical=%zu\n",total,(void*)address,at,bytes); std::fflush(stdout);
    auto p=std::make_unique<v::physical_mem>(q,bytes);
    std::printf("physical created\n");std::fflush(stdout);
    p->map(address+at,bytes,v::address_access_mode::read_write);
    blocks.push_back(std::move(p));mapped=at+bytes;
   }
   q.memset((void*)address,0,total).wait_and_throw();
   std::printf("PASS\n");
  } catch(const std::exception& e) {std::printf("FAIL %s\n",e.what());}
  for(size_t at=0;at<mapped;at+=64<<20)v::unmap((void*)(address+at),std::min<size_t>(64<<20,total-at),q.get_context());
  blocks.clear(); if(address)v::free_virtual_mem(address,total,q.get_context());
 }
}
