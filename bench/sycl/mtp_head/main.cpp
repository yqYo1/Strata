// Main-authored calibration. Inference math is the unchanged production Q6_K
// path. Diagnostic reconstruction follows tests/sycl/mmvq.cpp's Q8_1 contract.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace strata;
using namespace strata::kernels;
namespace compat=strata::sycl_backend::compat;
using Clock=std::chrono::steady_clock;
constexpr int N=2560,full_rows=248320,subset_rows=106299,guard=64;
constexpr size_t row_bytes=2100,qbytes=N/32*36;
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void check(cudaError_t s,const char* why){if(s!=cudaSuccess)throw std::runtime_error(std::string(why)+": "+cudaGetErrorString(s));}
uint64_t number(const char* p){std::string s(p);need(!s.empty()&&s.size()<19&&std::all_of(s.begin(),s.end(),[](char c){return c>='0'&&c<='9';}),"decimal argument");return std::stoull(s);}
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
double cpu_time(){rusage r{};need(getrusage(RUSAGE_SELF,&r)==0,"rusage");return r.ru_utime.tv_sec+r.ru_stime.tv_sec+1e-6*(r.ru_utime.tv_usec+r.ru_stime.tv_usec);}
void read_at(int fd,void* to,size_t n,uint64_t at){
    struct stat st{};need(fstat(fd,&st)==0&&st.st_size>=0&&at<=uint64_t(st.st_size)&&n<=uint64_t(st.st_size)-at,"file slice extent");
    auto* p=static_cast<uint8_t*>(to);
    while(n){const auto k=pread(fd,p,n,at);need(k>0,"short file read");p+=k;n-=k;at+=k;}
}
uint16_t half(float v){const _Float16 h=_Float16(v);uint16_t bits;std::memcpy(&bits,&h,2);return bits;}
float unhalf(const uint8_t* p){_Float16 h;std::memcpy(&h,p,2);return float(h);}
std::vector<uint8_t> quantize(const std::vector<float>& x){
    std::vector<uint8_t> out(qbytes);
    for(int b=0;b<N/32;++b){std::array<float,32> sum;float maximum=0;
        for(int i=0;i<32;++i){sum[i]=x[b*32+i];maximum=std::max(maximum,std::abs(sum[i]));}
        for(int d=16;d;d/=2){const auto old=sum;for(int i=0;i<32;++i)sum[i]=old[i]+old[i^d];}
        const float scale=maximum/127.f;const uint16_t db=half(scale),sb=half(sum[0]);
        std::memcpy(out.data()+b*36,&db,2);std::memcpy(out.data()+b*36+2,&sb,2);
        for(int i=0;i<32;++i)out[b*36+4+i]=uint8_t(int8_t(std::round(maximum==0?0:x[b*32+i]/scale)));
    }return out;
}
struct Owners{
    cudaStream_t stream{};cudaGraph_t graph{};cudaGraphExec_t executable{};cudaEvent_t done{};
    void *weights{},*input{},*scratch{},*output{};bool capturing=false;
    static void close_check(cudaError_t s)noexcept{if(s!=cudaSuccess){std::fprintf(stderr,"head calibration: uncertain retirement: %s\n",cudaGetErrorString(s));std::fflush(stderr);std::_Exit(EXIT_FAILURE);}}
    void close()noexcept{
        if(capturing){close_check(cudaStreamEndCapture(stream,&graph));capturing=false;}
        if(stream)close_check(cudaStreamSynchronize(stream));
        if(executable){close_check(cudaGraphExecDestroy(executable));executable=nullptr;}
        if(graph){close_check(cudaGraphDestroy(graph));graph=nullptr;}
        if(done){close_check(cudaEventDestroy(done));done=nullptr;}
        for(auto* p:{&output,&scratch,&input,&weights})if(*p){close_check(cudaFree(*p));*p=nullptr;}
        if(stream){close_check(cudaStreamDestroy(stream));stream=nullptr;}
    }
    ~Owners(){close();}
};
int main(int argc,char** argv)try{
    std::cout<<std::unitbuf<<std::setprecision(12);
    need(argc==7,"shard absolute-head-offset vocab-map rows check|measure profile0|1");
    const uint64_t offset=number(argv[2]),rows_arg=number(argv[4]),profile_arg=number(argv[6]);
    need(rows_arg<=subset_rows&&profile_arg<=1,"integer argument bounds");
    const int rows=int(rows_arg),profile=int(profile_arg);const std::string mode=argv[5];
    need((rows==17||rows==subset_rows)&&(profile==0||profile==1)&&(mode=="check"||mode=="measure")&&(mode!="measure"||rows==subset_rows),"shape/mode");
    std::vector<int32_t> ids(subset_rows);const int mapfd=open(argv[3],O_RDONLY|O_CLOEXEC);need(mapfd>=0,"open map");
    struct stat ms{};need(fstat(mapfd,&ms)==0&&ms.st_size==int64_t(ids.size()*4),"map extent");read_at(mapfd,ids.data(),ids.size()*4,0);close(mapfd);
    std::unordered_set<int32_t> unique;for(int id:ids)need(id>=0&&id<full_rows&&unique.insert(id).second,"map ID bounds/uniqueness");
    std::vector<uint8_t> w(size_t(rows)*row_bytes+2*guard*4,0x5c);
    const int fd=open(argv[1],O_RDONLY|O_CLOEXEC);need(fd>=0,"open shard");
    need(offset<=std::numeric_limits<uint64_t>::max()-uint64_t(full_rows)*row_bytes,"head offset overflow");
    for(int r=0;r<rows;++r){const int index=rows==subset_rows?r:(r==16?subset_rows-1:(r<8?r:subset_rows/2+r-8));read_at(fd,w.data()+guard*4+size_t(r)*row_bytes,row_bytes,offset+uint64_t(ids[index])*row_bytes);}close(fd);
    std::vector<float> x(N);uint32_t seed=20261022;
    for(auto& f:x){seed=1664525u*seed+1013904223u;f=float(int(seed>>8)-8388608)*0x1p-23f;}
    std::fill_n(x.begin(),32,0.f);const auto xq=quantize(x);
    std::vector<float> guarded_x(N+2*guard,12345.f),guarded_y(rows+2*guard,12345.f);
    std::copy(x.begin(),x.end(),guarded_x.begin()+guard);std::vector<uint8_t> guarded_q(qbytes+2*guard*4,0xa5);
    Owners own;check(cudaStreamCreate(&own.stream),"stream");check(cudaEventCreateWithFlags(&own.done,cudaEventDisableTiming),"done event");
    cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,0),"device properties");
    need(std::string(properties.name).find("B570")!=std::string::npos,"expected Arc B570");
    std::cout<<"{\"kind\":\"device\",\"name\":\""<<properties.name<<"\",\"global_bytes\":"<<properties.totalGlobalMem<<"}"<<std::endl;
    check(cudaMalloc(&own.weights,w.size()),"weight allocation");check(cudaMalloc(&own.input,guarded_x.size()*4),"input allocation");
    check(cudaMalloc(&own.output,guarded_y.size()*4),"output allocation");check(cudaMalloc(&own.scratch,guarded_q.size()),"scratch allocation");
    check(cudaMemcpy(own.weights,w.data(),w.size(),cudaMemcpyHostToDevice),"weight upload");
    check(cudaMemcpy(own.input,guarded_x.data(),guarded_x.size()*4,cudaMemcpyHostToDevice),"input upload");
    check(cudaMemcpy(own.output,guarded_y.data(),guarded_y.size()*4,cudaMemcpyHostToDevice),"output init");
    check(cudaMemcpy(own.scratch,guarded_q.data(),guarded_q.size(),cudaMemcpyHostToDevice),"scratch init");
    const auto* wd=static_cast<uint8_t*>(own.weights)+guard*4;const auto* xd=static_cast<float*>(own.input)+guard;
    auto* qd=static_cast<uint8_t*>(own.scratch)+guard*4;auto* yd=static_cast<float*>(own.output)+guard;
    auto launch=[&]{native_quantize_q8_1(xd,qd,N,1,own.stream);native_mmvq(14,wd,qd,yd,N,rows,1,own.stream);};
    launch();check(cudaStreamSynchronize(own.stream),"direct completion");
    std::vector<float> direct(rows+2*guard);check(cudaMemcpy(direct.data(),own.output,direct.size()*4,cudaMemcpyDeviceToHost),"direct output");
    check(cudaMemcpy(guarded_q.data(),own.scratch,guarded_q.size(),cudaMemcpyDeviceToHost),"direct quantization");
    need(std::equal(xq.begin(),xq.end(),guarded_q.begin()+guard*4),"Q8_1 reference bytes");
    double max_error=0;
    for(int r=0;r<rows;++r){double expected=0,magnitude=0;
        for(int b=0;b<N/256;++b){float decoded[256];dequantize_q6_K(w.data()+guard*4+size_t(r)*row_bytes+b*210,decoded);
            for(int i=0;i<256;++i){const int at=b*256+i;const auto* qb=xq.data()+at/32*36;
                const double term=double(decoded[i])*unhalf(qb)*int8_t(qb[4+at%32]);expected+=term;magnitude+=std::abs(term);}}
        const double err=std::abs(double(direct[guard+r])-expected)/(1+magnitude);max_error=std::max(max_error,err);
        need(std::isfinite(direct[guard+r])&&err<=3e-6,"Q6_K dequant/Q8_1 numerical reference");
    }
    std::cout<<"{\"kind\":\"reference\",\"rows\":"<<rows<<",\"type\":14,\"N\":"<<N<<",\"NT\":1,\"weight_bytes\":"<<size_t(rows)*row_bytes<<",\"map_unique\":true,\"Q8_1_bytes_equal\":true,\"max_scaled_error\":"<<max_error<<"}"<<std::endl;
    check(cudaStreamBeginCapture(own.stream,cudaStreamCaptureModeThreadLocal),"capture begin");own.capturing=true;launch();
    check(cudaStreamEndCapture(own.stream,&own.graph),"capture end");own.capturing=false;
    check(profile?compat::graph_instantiate_profiled(&own.executable,own.graph):cudaGraphInstantiate(&own.executable,own.graph,0ull),"instantiate");
    check(cudaGraphDestroy(own.graph),"source graph retirement");own.graph=nullptr;
    check(cudaGraphUpload(own.executable,own.stream),"graph upload");check(cudaStreamSynchronize(own.stream),"upload completion");
    auto replay=[&]{check(compat::graph_launch_with_completion(own.executable,own.stream,own.done),"replay");check(cudaStreamSynchronize(own.stream),"replay completion");};
    auto validate=[&]{
        std::vector<float> out(rows+2*guard),in(N+2*guard);std::vector<uint8_t> q(guarded_q.size());std::array<uint8_t,guard*8> edges;
        check(cudaMemcpy(out.data(),own.output,out.size()*4,cudaMemcpyDeviceToHost),"output read");
        check(cudaMemcpy(in.data(),own.input,in.size()*4,cudaMemcpyDeviceToHost),"input read");check(cudaMemcpy(q.data(),own.scratch,q.size(),cudaMemcpyDeviceToHost),"scratch read");
        check(cudaMemcpy(edges.data(),own.weights,guard*4,cudaMemcpyDeviceToHost),"weight prefix");
        check(cudaMemcpy(edges.data()+guard*4,wd+size_t(rows)*row_bytes,guard*4,cudaMemcpyDeviceToHost),"weight suffix");
        need(std::memcmp(out.data(),direct.data(),out.size()*4)==0&&std::memcmp(in.data(),guarded_x.data(),in.size()*4)==0&&q==guarded_q,"graph/direct bits or input/scratch mismatch");
        for(int i=0;i<guard;++i)need(out[i]==12345.f&&out[guard+rows+i]==12345.f,"output guard");
        for(int i=0;i<guard*4;++i)need(q[i]==0xa5&&q[guard*4+qbytes+i]==0xa5,"scratch guard");
        need(std::all_of(edges.begin(),edges.end(),[](uint8_t v){return v==0x5c;}),"weight guards");
        std::cout<<"{\"kind\":\"validation\",\"graph_direct_bits_equal\":true,\"all_guards\":true,\"inputs_unchanged\":true}"<<std::endl;
    };
    replay();validate();if(mode=="measure")for(int i=0;i<8;++i)replay();uint64_t previous_end=0;
    for(int i=0;i<(mode=="measure"?64:2);++i){
        const auto begin=Clock::now();replay();const double wall=elapsed(begin);
        compat::GraphEnvelopeTimestamps t{};const auto query=Clock::now();const auto status=compat::graph_completion_timestamps(own.done,&t);const double cost=elapsed(query);
        if(profile){need(status==cudaSuccess&&t.available_fields==7&&t.timer_resolution>0&&t.end>t.start&&t.end-t.start>=t.timer_resolution&&t.start>=previous_end&&cost<1,"graph clock qualification");previous_end=t.end;
            std::cout<<"{\"kind\":\"envelope\",\"rep\":"<<i<<",\"submit\":"<<t.submit<<",\"start\":"<<t.start<<",\"end\":"<<t.end<<",\"resolution\":"<<t.timer_resolution<<",\"wall_s\":"<<wall<<",\"query_s\":"<<cost<<"}"<<std::endl;
        }else need(status==cudaErrorNotSupported&&t.available_fields==0,"unexpected unprofiled graph clock");
    }
    if(mode=="measure"){
        double fastest=1e6;for(int i=0;i<8;++i){const auto t=Clock::now();replay();fastest=std::min(fastest,elapsed(t));}
        const auto t=Clock::now();for(int i=0;i<10000;++i){const auto now=Clock::now();(void)now;}const double query=elapsed(t)/10000;
        need(query/fastest<.001,"timer overhead");std::cout<<"{\"kind\":\"timer\",\"query_s\":"<<query<<",\"fastest_replay_s\":"<<fastest<<"}"<<std::endl;
        for(int rep=0;rep<5;++rep){const double before=cpu_time();const auto t0=Clock::now();int replays=0;
            do{replay();++replays;}while(elapsed(t0)<1.3&&replays<1000000);
            const double wall=elapsed(t0),cpu=cpu_time()-before;need(wall>=1&&replays<1000000,"timing sample floor/budget");
            std::cout<<"{\"kind\":\"sample\",\"rep\":"<<rep<<",\"replays\":"<<replays<<",\"wall_s\":"<<wall<<",\"cpu_s\":"<<cpu<<"}"<<std::endl;
        }
    }
    validate();own.close();std::cout<<"{\"kind\":\"closed\",\"profile\":"<<profile<<",\"passed\":true,\"checked_retirement\":true}"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<"head calibration failed: "<<e.what()<<std::endl;return 1;}
