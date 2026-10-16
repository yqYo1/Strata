#include "xmx-prototype.cpp"
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace strata::kernels;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
uint32_t state = 0x937ac654;
uint32_t random32() { state ^= state<<13; state^=state>>17; state^=state<<5; return state; }
template<int Blocks> void run(int rows, bool edges) {
  const int width=Blocks*256;
  const size_t wb=size_t(rows)*Blocks*136, xb=size_t(width/32)*36;
  std::vector<uint8_t> weights(wb), activation(xb);
  for (auto &v:weights) v=uint8_t(random32());
  for (auto &v:activation) v=uint8_t(random32());
  const uint16_t scales[]={0,0x8000,1,0x3ff,0x400,0x8400,0x8001,0x7bff};
  for (int block=0;block<rows*Blocks;++block) {
    sycl::half d=float(int(random32()%1025)-512)/65536.f;
    uint16_t bits=sycl::bit_cast<uint16_t>(d);
    if (edges) bits=scales[block%8];
    std::memcpy(weights.data()+size_t(block)*136,&bits,2);
  }
  for (int block=0;block<width/32;++block) {
    sycl::half d=float(1+random32()%512)/65536.f;
    uint16_t bits=sycl::bit_cast<uint16_t>(d);
    if (edges) bits=scales[(block+3)%8];
    std::memcpy(activation.data()+size_t(block)*36,&bits,2);
  }
  cudaStream_t stream{}; check(cudaStreamCreate(&stream));
  uint8_t *w{},*x{};float *old{},*next{};
  check(cudaMalloc(&w,wb));check(cudaMalloc(&x,xb));
  check(cudaMalloc(&old,size_t(rows)*4));check(cudaMalloc(&next,size_t(rows)*4));
  check(cudaMemcpyAsync(w,weights.data(),wb,cudaMemcpyHostToDevice,stream));
  check(cudaMemcpyAsync(x,activation.data(),xb,cudaMemcpyHostToDevice,stream));
  auto launch=[&](bool optimized) {
    if (!optimized) stream->parallel_for(sycl::nd_range<1>(size_t(rows),16),
      [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL { const int row=int(item.get_global_linear_id());
        old[row]=row_dot(w+size_t(row)*Blocks*136,x,Blocks); });
    else {const size_t tiles=size_t(rows/16), padded=(tiles+3)&~size_t(3);
      stream->parallel_for(sycl::nd_range<1>(padded,4),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {const int row=int(item.get_global_linear_id())*16;
          if(row<rows) {auto result=xmx_tile<Blocks>(w+size_t(row)*Blocks*136,x);
            e::scatter<float,16>(next+row,U(0,4),result);}});
    }
  };
  launch(false);launch(true);check(cudaStreamSynchronize(stream));
  std::vector<float> a(rows),b(rows);
  check(cudaMemcpy(a.data(),old,size_t(rows)*4,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(b.data(),next,size_t(rows)*4,cudaMemcpyDeviceToHost));
  int unequal=0;for(int i=0;i<rows;++i) if(std::memcmp(&a[i],&b[i],4)!=0) {
    if(unequal<4) std::cerr<<"difference "<<width<<' '<<rows<<' '<<edges<<' '<<i<<' '<<a[i]<<' '<<b[i]<<'\n';++unequal;
  }
  if(unequal) throw std::runtime_error("FP32 bits differ");
  std::cout<<"width="<<width<<" rows="<<rows<<" edges="<<edges<<" all bits equal"<<std::endl;
  if(!edges && rows>=2560) {
    for(bool optimized:{false,true}) {
      cudaGraph_t graph{};cudaGraphExec_t exec{};check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
      launch(optimized);check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,0ull));check(cudaGraphDestroy(graph));
      cudaEvent_t begin{},end{};check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
      std::array<float,3> timings;
      for(float &ms:timings) {
        check(cudaEventRecord(begin,stream));for(int j=0;j<100;++j)check(cudaGraphLaunch(exec,stream));
        check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));check(cudaEventElapsedTime(&ms,begin,end));ms/=100;
      }
      std::sort(timings.begin(),timings.end());std::cout<<(optimized?"xmx":"row")<<","<<width<<","<<rows<<","<<timings[1]<<std::endl;
      check(cudaEventDestroy(begin));check(cudaEventDestroy(end));check(cudaGraphExecDestroy(exec));
    }
  }
  check(cudaFree(w));check(cudaFree(x));check(cudaFree(old));check(cudaFree(next));check(cudaStreamDestroy(stream));
}
int main() {try {for(bool edges:{false,true}) {run<10>(32,edges);run<24>(32,edges);run<10>(640,edges);run<10>(6144,edges);run<10>(10240,edges);run<24>(2560,edges);}return 0;}catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}}
