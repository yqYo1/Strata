// Single arm, immutable environment, product entry points. No setenv/unsetenv.
// Main owns deadline, log/capture budgets, exclusive lock and fresh fault checks.
#include "fixture_common.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
namespace {
void require(bool ok, const char *m) { if (!ok) throw std::runtime_error(m); }
template <class T> void dump(const fs::path &path, const T *p, size_t n) {
  require(!fs::exists(path), "capture exists");
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(p), std::streamsize(n*sizeof(T)));
  require(bool(f), "capture write");
}
template <class T> struct Shared {
  sycl::queue &q; T *p;
  Shared(sycl::queue &queue, size_t n): q(queue), p(sycl::malloc_shared<T>(n,q)) {
    if (!p) throw std::bad_alloc();
  }
  // No destructor frees device storage: an exception can mean unknown queue
  // completion. Explicit retire is called only after a successful drain.
  ~Shared() = default;
  void retire() { sycl::free(p,q); p=nullptr; }
};
template <class F> void invalid(F f) {
  bool threw = false;
  try { f(); } catch (const std::invalid_argument &) { threw = true; }
  require(threw, "invalid call accepted");
}
size_t negative(sycl::queue &q) {
  using namespace strata::kernels;
  Shared<uint8_t> x(q, 4096), u(q,4096);
  Shared<uint16_t> y(q,4096);
  size_t checks=0;
  auto test = [&](auto f) { invalid(f); ++checks; };
  test([&] { iq_dequant_f16(-1,x.p,256,y.p,&q); });
  test([&] { iq_dequant_f16(18,x.p,0,y.p,&q); });
  test([&] { iq_dequant_f16(18,x.p,-256,y.p,&q); });
  test([&] { iq_dequant_f16(18,x.p,255,y.p,&q); });
  test([&] { iq_dequant_f16(42,x.p,32,y.p,&q); });
  test([&] { iq_dequant_f16(18,x.p+1,256,y.p,&q); });
  test([&] { iq_dequant_f16(18,x.p,256,
                  reinterpret_cast<uint16_t *>(reinterpret_cast<uint8_t *>(y.p)+1),&q); });
  test([&] { iq_dequant_f16(18,nullptr,256,y.p,&q); });
  test([&] { iq_dequant_f16(18,x.p,256,nullptr,&q); });
  test([&] { iq_dequant_f16(18,x.p,256,reinterpret_cast<uint16_t *>(x.p),&q); });
  test([&] { iq_dequant_f16(18,x.p,INT64_MAX-255,y.p,&q); });
  test([&] { iq_dequant_f16(18,reinterpret_cast<void *>(UINTPTR_MAX-1),256,y.p,&q); });
  test([&] { iq_dequant_gu_f16(18,x.p,u.p,0,256,y.p,&q); });
  test([&] { iq_dequant_gu_f16(18,x.p,u.p,1,255,y.p,&q); });
  test([&] { iq_dequant_gu_f16(18,x.p+2,u.p,1,256,y.p,&q); });
  test([&] { iq_dequant_gu_f16(18,x.p,u.p,1,256,
                  reinterpret_cast<uint16_t *>(reinterpret_cast<uint8_t *>(y.p)+2),&q); });
  test([&] { iq_dequant_gu_f16(18,x.p,u.p,INT64_MAX,256,y.p,&q); });
  test([&] { iq_dequant_gu_f16(18,x.p,u.p,INT64_MAX/256,256,y.p,&q); });
  test([&] { iq_dequant_gu_f16(18,x.p,u.p,1,256,reinterpret_cast<uint16_t *>(u.p),&q); });
  test([&] { iq_dequant_gu_f16(18,x.p,nullptr,1,256,y.p,&q); });
  q.wait_and_throw();
  x.retire(); u.retire(); y.retire();
  return checks;
}
void run(sycl::queue &q, const fs::path &root, unsigned number, int type,
         bool gu, size_t rows, size_t width, unsigned offset, bool edges) {
  using namespace strata::kernels;
  const auto f = strata::sycl_backend::block_format(type);
  const size_t count = rows*width, bytes = count/f.width*f.bytes;
  const size_t nout = count*(gu?2:1), allocbytes=bytes+64+offset;
  Shared<uint8_t> g(q,allocbytes), u(q,allocbytes);
  Shared<uint16_t> out(q,nout+32);
  std::fill_n(g.p,allocbytes,0xa5); std::fill_n(u.p,allocbytes,0x5a);
  std::fill_n(out.p,nout+32,0xa55a);
  uint8_t *gp=g.p+32+offset, *up=u.p+32+offset;
  for (size_t b=0; b<count/f.width; ++b) {
    const uint16_t d=edges?fixture::edge_scales[b%fixture::edge_scales.size()]:0x2801;
    fixture::make_block(type,gp+b*f.bytes,unsigned(b),d);
    fixture::make_block(type,up+b*f.bytes,unsigned(b+337),d^0x8000);
  }
  const std::vector<uint8_t> gcopy(g.p,g.p+allocbytes), ucopy(u.p,u.p+allocbytes);
  std::vector<uint16_t> expected;
  if (gu) {
    const size_t rb=width/f.width*f.bytes;
    for (size_t row=0;row<rows;++row)
      for (const uint8_t *p : {gp,up}) {
        auto values=fixture::oracle(type,p+row*rb,width);
        expected.insert(expected.end(),values.begin(),values.end());
      }
  } else expected=fixture::oracle(type,gp,count);
  q.wait_and_throw();
  // Markers belong to the fixture. They bracket queue time but their interval
  // can include host-submit gaps; it is not exclusive dequant device duration.
  auto begin=q.single_task([]{});
  const auto t0=Clock::now();
  if (gu) iq_dequant_gu_f16(type,gp,up,int64_t(rows),int64_t(width),out.p+16,&q);
  else iq_dequant_f16(type,gp,int64_t(count),out.p+16,&q);
  const auto t1=Clock::now();
  auto end=q.single_task([]{});
  q.wait_and_throw();
  const auto t2=Clock::now();
  require(std::equal(gcopy.begin(),gcopy.end(),g.p),"gate/input modified");
  require(std::equal(ucopy.begin(),ucopy.end(),u.p),"up/input modified");
  for (unsigned i=0;i<16;++i)
    require(out.p[i]==0xa55a && out.p[16+nout+i]==0xa55a,"output guards");
  size_t mismatch=0, nan_sign_mismatch=0;
  for (size_t i=0;i<nout;++i) {
    const uint16_t a=expected[i],b=out.p[16+i];
    const bool an=(a&0x7fff)>0x7c00, bn=(b&0x7fff)>0x7c00;
    if (a!=b) {
      // CPU/device may choose different NaN signs on invalid multiply. This
      // exception applies only to this CPU oracle; OFF/ON must match ALL bits.
      if (an && bn && (a&0x7fff)==0x7e00 && (b&0x7fff)==0x7e00) ++nan_sign_mismatch;
      else ++mismatch;
    }
  }
  const fs::path base=root/std::to_string(number);
  dump(base.string()+".input",g.p,allocbytes);
  if (gu) dump(base.string()+".up",u.p,allocbytes);
  dump(base.string()+".half",out.p,nout+32);
  dump(base.string()+".oracle",expected.data(),expected.size());
  std::ofstream meta(base.string()+".json");
  const auto ns=[](auto a,auto b){return std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();};
  const auto d0=begin.get_profiling_info<sycl::info::event_profiling::command_end>();
  const auto d1=end.get_profiling_info<sycl::info::event_profiling::command_start>();
  require(d1>=d0,"nonmonotonic marker timestamps");
  meta << "{\"type\":"<<type<<",\"gu\":"<<(gu?"true":"false")
       <<",\"rows\":"<<rows<<",\"width\":"<<width<<",\"offset\":"<<offset
       <<",\"elements\":"<<nout<<",\"edges\":"<<(edges?"true":"false")
       <<",\"cpu_mismatch\":"<<mismatch<<",\"cpu_nan_sign_mismatch\":"<<nan_sign_mismatch
       <<",\"host_submit_ns\":"<<ns(t0,t1)<<",\"drained_wall_ns\":"<<ns(t0,t2)
       <<",\"marker_interval_ns\":"<<(d1-d0)<<",\"guards\":true,\"inputs_unchanged\":true}\n";
  meta.close();
  require(bool(meta),"metadata write");
  require(!mismatch,"GPU/CPU half mismatch");
  std::cout<<"case="<<number<<" type="<<type<<" gu="<<gu<<" complete\n";
  g.retire(); u.retire(); out.retire();
}
}
int main(int argc,char **argv) {
  try {
    require(argc==4,"usage: gpu_fixture off|on quick|full NEW_CAPTURE_DIR");
    const std::string arm=argv[1], mode=argv[2];
    const char *env=std::getenv("STRATA_SYCL_PARALLEL_IQ_DEQUANT");
    require((arm=="on" && env && std::string(env)=="1") ||
            (arm=="off" && (!env || std::string(env)!="1")),"immutable arm/env mismatch");
    require(mode=="quick" || mode=="full","mode");
    const fs::path root=argv[3];
    require(fs::create_directory(root),"capture directory must be new (parent exists)");
    // Heap ownership avoids queue destruction during unwinding of an unknown
    // completion. A normal path drains, retires all buffers, then deletes it.
    auto *queue_owner = new sycl::queue{sycl::gpu_selector_v, [](sycl::exception_list list) {
      for (auto &e:list) std::rethrow_exception(e);
    }, {sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}}};
    auto &q=*queue_owner;
    std::ofstream device(root/"device.txt");
    device<<q.get_device().get_info<sycl::info::device::name>()<<'\n'
          <<q.get_device().get_info<sycl::info::device::driver_version>()<<'\n';
    device.close();
    const size_t negatives=negative(q);
    unsigned number=0;
    for (int type:fixture::types) {
      const auto f=strata::sycl_backend::block_format(type);
      for (unsigned offset:{0u,2u})
        run(q,root,number++,type,false,1,size_t(f.width)*1024,offset,true);
      run(q,root,number++,type,true,3,512,0,true);
    }
    // Unsupported ON types must keep the existing path. F32 is checked below.
    for (int type:{2,6,7,8,11,12,13,14,16,17,29,30}) {
      const auto f=strata::sycl_backend::block_format(type);
      run(q,root,number++,type,false,1,size_t(f.width)*1024,2,true);
      run(q,root,number++,type,true,1,256,0,true);
    }
    // F32 path all six types: expected scalar f32 bits, separately dumped for
    // cross-arm comparison. Finite scales avoid cross-device NaN sign latitude.
    for (int type:fixture::types) {
      const auto f=strata::sycl_backend::block_format(type);
      Shared<uint8_t> src(q,size_t(f.bytes)*3+2);
      Shared<float> out(q,size_t(f.width)*3+2);
      std::fill_n(src.p,size_t(f.bytes)*3+2,0xa5);
      std::fill_n(out.p,size_t(f.width)*3+2,123456.f);
      for (unsigned b=0;b<3;++b) fixture::make_block(type,src.p+2+b*f.bytes,b,0x2801);
      const std::vector<uint8_t> original(src.p,src.p+size_t(f.bytes)*3+2);
      strata::kernels::iq_dequant_f32(type,src.p+2,int64_t(f.width)*3,out.p+1,&q);
      q.wait_and_throw();
      require(std::equal(original.begin(),original.end(),src.p),"f32 input changed");
      require(out.p[0]==123456.f && out.p[1+size_t(f.width)*3]==123456.f,"f32 guards");
      for(unsigned b=0;b<3;++b) {
        float values[256]; strata::sycl_backend::decode_block(type,src.p+2+b*f.bytes,values);
        require(std::memcmp(out.p+1+b*f.width,values,size_t(f.width)*4)==0,"f32 mismatch");
      }
      dump(root/("f32-"+std::to_string(type)+".bin"),out.p,size_t(f.width)*3+2);
      dump(root/("f32-"+std::to_string(type)+".input"),src.p,size_t(f.bytes)*3+2);
      src.retire(); out.retire();
    }
    if(mode=="full") {
      for(int type:{18,22,21,23}) run(q,root,number++,type,true,640,2560,0,false);
      for(int type:{20,42}) run(q,root,number++,type,false,2560,640,2,false);
    }
    q.wait_and_throw();
    delete queue_owner;
    std::ofstream done(root/"complete.json");
    done<<"{\"complete\":true,\"arm\":\""<<arm<<"\",\"mode\":\""<<mode
        <<"\",\"cases\":"<<number<<",\"negative_checks\":"<<negatives
        <<",\"f32_cases\":6,\"expected_product_launches_per_valid_case\":1,\"fixture_markers_per_half_case\":2}\n";
    done.close(); require(bool(done),"completion receipt"); return 0;
  } catch(const std::exception &e) {
    std::cerr<<"FAIL "<<e.what()<<'\n';
    // Driver/runtime errors must not unwind USM owners whose completion is
    // unknown. Main records abnormal completion and handles its owned process.
    std::cerr.flush(); std::_Exit(1);
  }
}
