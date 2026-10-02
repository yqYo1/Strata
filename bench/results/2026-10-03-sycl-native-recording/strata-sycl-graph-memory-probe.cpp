#include "strata/sycl/runtime.hpp"
#include "strata/sycl/launch.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
void ck(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
int main(int argc,char**argv){try {
 const bool reuse=argc>1&&std::string(argv[1])=="reuse";
 auto rt=strata::sycl_backend::runtime_for();float*x;ck(cudaMalloc(&x,2560*4));ck(cudaMemset(x,0,2560*4));size_t free,total;ck(cudaMemGetInfo(&free,&total));std::cout<<"before "<<free/1048576<<" MiB\n";
 cudaStream_t common{};if(reuse)ck(cudaStreamCreate(&common));std::vector<cudaGraphExec_t>graphs;
 for(int i=0;i<192;i++){
  cudaStream_t st=common;if(!reuse)ck(cudaStreamCreate(&st));ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
  for(int node=0;node<15;node++)strata::sycl_backend::for_each(2560,st,[=](size_t j){x[j]=x[j]+float(node)*.001f;});
  cudaGraph_t g{};ck(cudaStreamEndCapture(st,&g));if(!reuse)ck(cudaStreamDestroy(st));cudaGraphExec_t ge{};ck(cudaGraphInstantiate(&ge,g,0ull));ck(cudaGraphDestroy(g));graphs.push_back(ge);
 }
 if(reuse)ck(cudaStreamDestroy(common));ck(cudaMemGetInfo(&free,&total));std::cout<<"after "<<free/1048576<<" MiB\n";
 for(auto ge:graphs)ck(cudaGraphExecDestroy(ge));ck(cudaFree(x));
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
