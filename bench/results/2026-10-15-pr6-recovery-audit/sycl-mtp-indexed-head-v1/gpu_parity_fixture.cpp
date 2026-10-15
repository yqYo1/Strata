// SOURCE ONLY: Main owns build, serial GPU process, budget/fault/retirement gates.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/core/indexed_head_contract.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata::kernels;
static void need(bool b,const char* text){if(!b)throw std::runtime_error(text);}
static void checked(cudaError_t e,const char* text){if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",text,cudaGetErrorString(e));strata::sycl_backend::cleanup_fail_stop(text);}}
struct Buffer {
    uint8_t* base=nullptr;size_t bytes;std::vector<uint8_t> initial;
    explicit Buffer(size_t n):bytes(n),initial(n+128,0xa5){checked(cudaMalloc(reinterpret_cast<void**>(&base),initial.size()),"allocation");}
    uint8_t* data(){return base+64;}
    void upload(){checked(cudaMemcpy(base,initial.data(),initial.size(),cudaMemcpyHostToDevice),"upload");}
    std::vector<uint8_t> read(){std::vector<uint8_t> v(initial.size());checked(cudaMemcpy(v.data(),base,v.size(),cudaMemcpyDeviceToHost),"readback");return v;}
    void guards(const std::vector<uint8_t>& v){need(std::equal(v.begin(),v.begin()+64,initial.begin()) && std::equal(v.end()-64,v.end(),initial.end()-64),"guards");}
    void immutable(){need(read()==initial,"input payload or guard changed");}
    void retire(){checked(cudaFree(base),"checked buffer free");base=nullptr;}
    // Deliberately no destructor: unknown GPU completion never unwinds device owners.
};
static uint64_t hash(const std::vector<uint8_t>& bytes){uint64_t h=14695981039346656037ull;for(auto b:bytes){h^=b;h*=1099511628211ull;}return h;}
static void rejected(const std::function<void()>& fn){bool bad=false;try{fn();}catch(const std::invalid_argument&){bad=true;}need(bad,"negative input accepted");}
static void run(int cols,const std::vector<int32_t>& ids,cudaStream_t stream,bool exact){
    constexpr int ni=2560,full=37;const int subset=int(ids.size());
    const strata::core::detail::IndexedHeadShape shape(ni,full,subset,cols);
    strata::core::detail::validate_indexed_head_ids(ids.data(),ids.size(),full);
    Buffer weights(shape.weights),mapping(shape.ids),input(size_t(cols)*ni*4),x(shape.activation),gathered(size_t(subset)*shape.row),old(shape.output),indexed(shape.output);
    for(size_t block=0;block<shape.weights/210;++block){auto* dst=weights.initial.data()+64+block*210;for(int j=0;j<192;++j)dst[j]=uint8_t((block*37+size_t(j)*19)%256);for(int j=0;j<16;++j)dst[192+j]=uint8_t(int((block+j)%7)-3);dst[208]=0;dst[209]=0x24;}
    std::memcpy(mapping.initial.data()+64,ids.data(),shape.ids);
    for(size_t i=0;i<size_t(cols)*ni;++i){const float v=float(int((i*13)%127)-63)/64;std::memcpy(input.initial.data()+64+i*4,&v,4);}
    for(Buffer* b:{&weights,&mapping,&input,&x,&gathered,&old,&indexed})b->upload();
    native_quantize_q8_1(reinterpret_cast<float*>(input.data()),x.data(),ni,cols,stream);
    gather_rows(weights.data(),int64_t(shape.row),reinterpret_cast<int32_t*>(mapping.data()),subset,gathered.data(),stream);
    native_mmvq(14,gathered.data(),x.data(),reinterpret_cast<float*>(old.data()),ni,subset,cols,stream);
    checked(cudaStreamSynchronize(stream),"legacy gather/MMVQ drain");
    const auto quantized=x.read();
    // Captured indexed map/weights must remain live until graph destruction.
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"begin indexed graph");
    native_q6_k_mmvq_indexed(weights.data(),reinterpret_cast<int32_t*>(mapping.data()),x.data(),reinterpret_cast<float*>(indexed.data()),ni,full,subset,cols,stream);
    checked(cudaStreamEndCapture(stream,&graph),"end indexed graph");
    checked(cudaGraphInstantiate(&exec,graph,0),"instantiate indexed graph");
    checked(cudaGraphLaunch(exec,stream),"launch indexed graph");checked(cudaStreamSynchronize(stream),"indexed graph drain");
    const auto reference=old.read(),actual=indexed.read();need(reference==actual,"complete indexed/gather MMVQ bits differ");old.guards(reference);indexed.guards(actual);
    for(size_t i=0;i<shape.output/4;++i){float value;std::memcpy(&value,actual.data()+64+4*i,4);need(std::isfinite(value),"nonfinite output");}
    weights.immutable();mapping.immutable();input.immutable();need(x.read()==quantized,"Q8 activation payload changed");x.guards(quantized);gathered.guards(gathered.read());
    checked(cudaGraphExecDestroy(exec),"indexed exec retirement");checked(cudaGraphDestroy(graph),"indexed graph retirement");
    auto invoke=[&](const void* w,const int32_t* map,const void* activation,float* out,int n,int all,int sub,int nc){native_q6_k_mmvq_indexed(w,map,activation,out,n,all,sub,nc,stream);};
    const auto* map=reinterpret_cast<int32_t*>(mapping.data());auto* y=reinterpret_cast<float*>(indexed.data());
    rejected([&]{invoke(weights.data(),map,x.data(),y,ni,full,0,cols);});
    rejected([&]{invoke(weights.data(),map,x.data(),y,ni+1,full,subset,cols);});
    rejected([&]{invoke(weights.data(),map,x.data(),y,ni,full,subset,9);});
    rejected([&]{invoke(weights.data(),map,x.data(),y,ni,full+1,subset,cols);}); // registered full span too short
    rejected([&]{invoke(weights.data()+2,map,x.data(),y,ni,full,subset,cols);});
    rejected([&]{invoke(weights.data(),map,x.data(),reinterpret_cast<float*>(weights.data()),ni,full,subset,cols);});
    std::vector<uint8_t> host_weights(shape.weights);rejected([&]{invoke(host_weights.data(),map,x.data(),y,ni,full,subset,cols);});
    rejected([&]{invoke(weights.data(),reinterpret_cast<const int32_t*>(mapping.data()+2),x.data(),y,ni,full,subset,cols);});
    rejected([&]{invoke(weights.data(),map,x.data(),reinterpret_cast<float*>(mapping.data()),ni,full,subset,cols);});
    rejected([&]{native_q6_k_mmvq_indexed(weights.data(),map,x.data(),y,ni,full,subset,cols,nullptr);});
    void* short_span=nullptr;checked(cudaMalloc(&short_span,4),"short-span negative allocation");
    if(subset>1)rejected([&]{invoke(weights.data(),static_cast<int32_t*>(short_span),x.data(),y,ni,full,subset,cols);});
    rejected([&]{invoke(weights.data(),map,short_span,y,ni,full,subset,cols);});
    if(shape.output>4)rejected([&]{invoke(weights.data(),map,x.data(),static_cast<float*>(short_span),ni,full,subset,cols);});
    checked(cudaFree(short_span),"short-span negative retirement");
    // Invalid device IDs are never used for weight reads: defensive zero rows.
    std::vector<int32_t> invalid(size_t(subset),-1);invalid.back()=full;
    checked(cudaMemcpy(mapping.data(),invalid.data(),shape.ids,cudaMemcpyHostToDevice),"invalid ID diagnostic upload");
    invoke(weights.data(),map,x.data(),y,ni,full,subset,cols);checked(cudaStreamSynchronize(stream),"invalid ID diagnostic drain");
    const auto zeros=indexed.read();indexed.guards(zeros);need(x.read()==quantized,"invalid-map input changed");
    auto expected_map=mapping.initial;std::memcpy(expected_map.data()+64,invalid.data(),shape.ids);
    need(mapping.read()==expected_map,"invalid map payload or guards changed");weights.immutable();input.immutable();for(size_t i=64;i<zeros.size()-64;++i)need(zeros[i]==0,"invalid map must produce positive zero");
    std::printf("{\"cols\":%d,\"subset\":%d,\"exact\":%s,\"bitwise\":true,\"output_fnv1a64\":%llu,\"invalid_map_zero\":true}\n",cols,subset,exact?"true":"false",(unsigned long long)hash(actual));std::fflush(stdout);
    checked(cudaStreamSynchronize(stream),"final fixture drain");for(Buffer* b:{&weights,&mapping,&input,&x,&gathered,&old,&indexed})b->retire();
}
int main(int argc,char** argv){try{
    need(argc==2 && (std::string(argv[1])=="0" || std::string(argv[1])=="1"),"usage: GPU fixture EXACT0|1 (separate owned processes)");
    const bool exact=std::string(argv[1])=="1";native_mmvq_set_multi_exact(exact); // explicit test parameter before runtime
    cudaStream_t stream=nullptr;checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
    std::vector<std::vector<int32_t>> maps{{0},{36},{36,0,17,17,1}};
    for(int n:{17,33,37,51}){std::vector<int32_t> ids;for(int i=0;i<n;++i)ids.push_back((36-i*7%37+37)%37);maps.push_back(ids);}
    size_t cases=0;for(int cols=1;cols<=8;++cols)for(const auto& ids:maps){run(cols,ids,stream,exact);++cases;}
    checked(cudaStreamSynchronize(stream),"all cases drain");checked(cudaStreamDestroy(stream),"stream destroy");std::printf("{\"complete\":true,\"cases\":%zu,\"performance_qualified\":false}\n",cases);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"fixture failure: %s\n",error.what());strata::sycl_backend::cleanup_fail_stop("GPU fixture incomplete; device owners retained");}}
