#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_qsa_score.hpp"
#include "strata/sycl_upstream/cuda_backend.hpp"
#include <ggml.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>
namespace k=strata::kernels;
namespace backend=strata::sycl_upstream::cuda;
std::string context;
size_t numerical=0,bitwise=0,guards=0,cases=0,replays=0;
double max_scaled_error=0;
constexpr float guard=-1234567.f;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(context+": "+why);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string(cudaGetErrorName(e))+": "+backend::backend_error_detail());}
void near(float actual,double expected,const char* why,double scale=0){
    const double error=std::abs(double(actual)-expected),magnitude=std::max(std::abs(expected),scale);
    max_scaled_error=std::max(max_scaled_error,error/(magnitude+1));
    if(!std::isfinite(actual)||error>3e-5*magnitude+3e-6){std::cerr<<"got="<<actual<<" expected="<<expected<<" scale="<<magnitude<<'\n';require(false,why);}++numerical;
}
void same(const std::vector<float>& a,const std::vector<float>& b,const char* why){require(a.size()==b.size(),"bit comparison shape");for(size_t i=0;i<a.size();++i){require(std::bit_cast<uint32_t>(a[i])==std::bit_cast<uint32_t>(b[i]),why);++bitwise;}}
void tail(const std::vector<float>& x,size_t live){for(size_t i=live;i<x.size();++i){require(x[i]==guard,"output guard");++guards;}}
uint32_t seed=0xb316c489;
uint32_t random_word(){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
std::vector<float> random_values(size_t n,float scale=1){std::vector<float> v(n);for(auto& x:v)x=float(int(random_word()%1025)-512)*(scale/4096);return v;}
std::vector<float> guarded(size_t n){return std::vector<float>(n+16,guard);}
struct Gpu{
 void* p{};size_t bytes;
 explicit Gpu(size_t n):bytes(n){ck(cudaMalloc(&p,n));}
 ~Gpu(){cudaFree(p);}
 template<class T>T* as(){return static_cast<T*>(p);}
 template<class T>void put(const std::vector<T>& x,cudaStream_t s){require(x.size()*sizeof(T)<=bytes,"upload extent");ck(cudaMemcpyAsync(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice,s));ck(cudaStreamSynchronize(s));}
 template<class T>std::vector<T> get(cudaStream_t s){std::vector<T> x(bytes/sizeof(T));ck(cudaMemcpyAsync(x.data(),p,bytes,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return x;}
};
struct Graph{
 cudaGraphExec_t exec{};
 Graph(cudaStream_t s,const std::function<void()>& record,int kernels){cudaGraph_t def{};ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));record();ck(cudaStreamEndCapture(s,&def));size_t n=0;ck(cudaGraphGetNodes(def,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);ck(cudaGraphGetNodes(def,nodes.data(),&n));int actual=0;for(auto node:nodes){cudaGraphNodeType t;ck(cudaGraphNodeGetType(node,&t));actual+=t==cudaGraphNodeTypeKernel;}require(actual==kernels,"captured native kernel count");ck(cudaGraphInstantiate(&exec,def,0ull));ck(cudaGraphDestroy(def));}
 ~Graph(){cudaGraphExecDestroy(exec);}
 void run(cudaStream_t s){ck(cudaGraphLaunch(exec,s));++replays;}
};
double sigmoid(double x){return 1/(1+std::exp(-x));}
std::vector<double> rms(const std::vector<float>& x,const std::vector<float>& gamma,int width,int rows,double epsilon,bool repeated){
 std::vector<double> y(x.size());for(int r=0;r<rows;++r){double ss=0;for(int c=0;c<width;++c)ss+=double(x[r*width+c])*x[r*width+c];const double inv=1/std::sqrt(ss/width+epsilon);for(int c=0;c<width;++c)y[r*width+c]=inv*x[r*width+c]*gamma[(repeated?0:r)*width+c];}return y;
}
void norm_tests(cudaStream_t s){
 for(int width:{1,127,128,256,512,1023,1024,2560}){
  context="norm width="+std::to_string(width);constexpr int rows=3;auto x=random_values(size_t(width)*rows),g=random_values(x.size(),2);std::fill_n(x.begin(),width,0.f);Gpu dx(x.size()*4),dg(g.size()*4),dy((x.size()+16)*4);dx.put(x,s);dg.put(g,s);auto init=guarded(x.size());dy.put(init,s);
  k::native_gr_rms_norm_weighted(dx.as<float>(),dg.as<float>(),dy.as<float>(),width,rows,1e-6f,s);auto out=dy.get<float>(s);auto ref=rms(x,g,width,rows,1e-6f,false);for(size_t i=0;i<x.size();++i)near(out[i],ref[i],"weighted GR norm");tail(out,x.size());++cases;
  g.resize(width);dg.put(g,s);auto broadcast=rms(x,g,width,rows,1e-6f,true);
  for(bool alias:{false,true}){dx.put(x,s);dy.put(init,s);k::native_qsa_rms_norm_weighted(dx.as<float>(),dg.as<float>(),alias?dx.as<float>():dy.as<float>(),width,rows,1e-6f,s);auto got=alias?dx.get<float>(s):dy.get<float>(s);for(size_t i=0;i<x.size();++i)near(got[i],broadcast[i],"QSA norm broadcast/alias");if(!alias)tail(got,x.size());++cases;}
 }
}
void gr_tests(cudaStream_t s){
 for(int n:{7,2560})for(int hc:{1,4}){
  context="GR postops N="+std::to_string(n)+" hc="+std::to_string(hc);auto x=random_values(size_t(n)*hc,3),gate0=random_values(x.size(),10),lo=random_values(n,4),residual=random_values(x.size(),2),block=random_values(n),inject=random_values(hc,12);Gpu dx(x.size()*4),dg((x.size()+16)*4),dm((n+16)*4),dl((n+16)*4),dr((x.size()+16)*4),db(block.size()*4),di(inject.size()*4);dx.put(x,s);db.put(block,s);di.put(inject,s);auto init=guarded(x.size()),mixinit=guarded(n),lowinit=mixinit;std::copy(lo.begin(),lo.end(),lowinit.begin());dl.put(lowinit,s);
  k::native_gr_down_silu(dl.as<float>(),n,hc,s);auto low=dl.get<float>(s);for(int i=0;i<n;++i){double v=double(lo[i])/hc;near(low[i],v*sigmoid(v),"GR down scale/SiLU");}tail(low,n);++cases;
  for(bool fused:{false,true}){std::copy(gate0.begin(),gate0.end(),init.begin());dg.put(init,s);dm.put(mixinit,s);k::native_gr_pre_gated(dx.as<float>(),dg.as<float>(),dm.as<float>(),n,hc,fused,s);auto gp=dg.get<float>(s),m=dm.get<float>(s);for(int d=0;d<n;++d){double sum=0;for(int c=0;c<hc;++c){const size_t at=size_t(c)*n+d;const double v=double(x[at])*sigmoid(gate0[at]);sum+=v;near(gp[at],v,"GR gated stream");}near(m[d],sum/hc,"GR pre-gated mean");}tail(gp,x.size());tail(m,n);++cases;}
  auto rinit=guarded(x.size());std::copy(residual.begin(),residual.end(),rinit.begin());dr.put(rinit,s);k::native_gr_post(dr.as<float>(),db.as<float>(),di.as<float>(),dr.as<float>(),n,hc,s);auto result=dr.get<float>(s);for(int c=0;c<hc;++c)for(int d=0;d<n;++d)near(result[c*n+d],double(residual[c*n+d])+block[d]*2*sigmoid(double(inject[c])/hc),"GR residual exact alias");tail(result,x.size());++cases;
 }
}
void moe_router_tests(cudaStream_t s){
 for(int n:{7,2560})for(int experts=1;experts<=15;++experts){
  context="MoE combine N="+std::to_string(n)+" K="+std::to_string(experts);constexpr int tokens=3;auto p=random_values(size_t(tokens)*experts*n,3),w=random_values(tokens*experts,2),shared=random_values(size_t(tokens)*n);Gpu dp(p.size()*4),dw(w.size()*4),dh(shared.size()*4),dy((shared.size()+16)*4),single((n+16)*4);dp.put(p,s);dw.put(w,s);dh.put(shared,s);auto init=guarded(shared.size());
  for(bool use_shared:{false,true}){dy.put(init,s);k::native_moe_combine_multi(dp.as<float>(),dw.as<float>(),use_shared?dh.as<float>():nullptr,dy.as<float>(),n,experts,tokens,s);auto out=dy.get<float>(s);
   for(int t=0;t<tokens;++t){for(int d=0;d<n;++d){double v=use_shared?shared[t*n+d]:0,l1=std::abs(v);for(int e=0;e<experts;++e){const double term=double(p[(size_t(t)*experts+e)*n+d])*w[t*experts+e];v+=term;l1+=std::abs(term);}near(out[t*n+d],v,"MoE weighted experts and shared",l1);}auto oneinit=guarded(n);single.put(oneinit,s);k::native_moe_combine(dp.as<float>()+size_t(t)*experts*n,dw.as<float>()+t*experts,use_shared?dh.as<float>()+t*n:nullptr,single.as<float>(),n,experts,s);auto one=single.get<float>(s);for(int d=0;d<n;++d){require(std::bit_cast<uint32_t>(one[d])==std::bit_cast<uint32_t>(out[t*n+d]),"MoE multi differs from single");++bitwise;}tail(one,n);}tail(out,shared.size());++cases;
  }
 }
 for(int pattern=0;pattern<5;++pattern){
  context="router pattern="+std::to_string(pattern);constexpr int tokens=3;std::vector<float> logits(tokens*512);for(int t=0;t<tokens;++t)for(int i=0;i<512;++i)logits[t*512+i]=pattern==0?0.f:pattern==1?float(i)/16:pattern==2?float((i*173+t*11)%512)/32:pattern==3?(i%32==0?7.f:-4.f):float((i+t)%8)-80.f;
  Gpu dl(logits.size()*4),di((tokens*10+16)*4),dw((tokens*10+16)*4),oi((10+16)*4),ow((10+16)*4);dl.put(logits,s);std::vector<int32_t> idinit(tokens*10+16,-777),oneids(26,-777);auto init=guarded(tokens*10),onew=guarded(10);di.put(idinit,s);dw.put(init,s);k::native_router_top10_multi(dl.as<float>(),di.as<int32_t>(),dw.as<float>(),tokens,s);auto ids=di.get<int32_t>(s);auto weights=dw.get<float>(s);
  for(int t=0;t<tokens;++t){std::vector<int> ranked(512);std::iota(ranked.begin(),ranked.end(),0);std::stable_sort(ranked.begin(),ranked.end(),[&](int a,int b){return logits[t*512+a]>logits[t*512+b];});double denom=0;for(int j=0;j<10;++j)denom+=std::exp(double(logits[t*512+ranked[j]])-logits[t*512+ranked[0]]);for(int j=0;j<10;++j){require(ids[t*10+j]==ranked[j],"router selected ID/tie order");++bitwise;near(weights[t*10+j],std::exp(double(logits[t*512+ranked[j]])-logits[t*512+ranked[0]])/denom,"router normalized weights");}
   oi.put(oneids,s);ow.put(onew,s);k::native_router_top10(dl.as<float>()+t*512,oi.as<int32_t>(),ow.as<float>(),s);auto ix=oi.get<int32_t>(s);auto wx=ow.get<float>(s);for(int j=0;j<10;++j){require(ix[j]==ids[t*10+j]&&std::bit_cast<uint32_t>(wx[j])==std::bit_cast<uint32_t>(weights[t*10+j]),"router single/multi bits");bitwise+=2;}tail(wx,10);
  }for(size_t i=tokens*10;i<ids.size();++i){require(ids[i]==-777,"router ID guard");++guards;}tail(weights,tokens*10);++cases;
 }
}
void qsa_gate_tests(cudaStream_t s,cudaStream_t other){
 for(auto geometry:{std::pair{1,1},std::pair{3,17},std::pair{24,256}}){const auto [heads,width]=geometry;const int n=heads*width;
  for(bool alias:{false,true}){context="QSA gate heads="+std::to_string(heads)+" width="+std::to_string(width)+" alias="+std::to_string(alias);auto attn=random_values(n,3),full=random_values(n*2,4);Gpu da((n+16)*4),df((n*2+16)*4),dy((n+16)*4);auto input=guarded(n),fin=guarded(n*2),init=guarded(n);std::copy(attn.begin(),attn.end(),input.begin());std::copy(full.begin(),full.end(),fin.begin());da.put(input,s);df.put(fin,s);dy.put(init,s);Graph graph(s,[&]{k::native_qsa_gate_apply(da.as<float>(),df.as<float>(),alias?da.as<float>():dy.as<float>(),heads,width,s);},1);require((alias?da.get<float>(other):dy.get<float>(other))==(alias?input:init),"QSA gate graph executed at instantiation");
   for(int replay=0;replay<3;++replay){attn=random_values(n,3);constexpr float extremes[]={-80,-20,-1,0,.25f,5,20,80};for(int h=0;h<heads;++h)for(int d=0;d<width;++d){full[h*2*width+d]=(d%2?1:-1)*float(1000+d);full[h*2*width+width+d]=extremes[(h+d+replay)%8];}std::copy(attn.begin(),attn.end(),input.begin());std::copy(full.begin(),full.end(),fin.begin());da.put(input,s);df.put(fin,s);dy.put(init,s);graph.run(s);const auto out=alias?da.get<float>(s):dy.get<float>(s);for(int h=0;h<heads;++h)for(int d=0;d<width;++d)near(out[h*width+d],double(attn[h*width+d])*sigmoid(full[h*2*width+width+d]),"QSA second-half gate independent sigmoid");tail(out,n);same(df.get<float>(s),fin,"QSA gate changed full query");++cases;}
  }
 }
}
#include "primitive_state.hpp"
#include "primitive_attention.hpp"
#include "primitive_ple.hpp"
int main()try{
 setenv("STRATA_ROPE_TABLE","1",1);auto* cpu=ggml_init({16384,nullptr,true});require(cpu,"GGML CPU conversion tables");ggml_free(cpu);
 ck(cudaSetDevice(0));cudaStream_t s{},other{};ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));sycl::device device;ck(backend::stream_device(s,&device));std::cout<<"device="<<device.get_info<sycl::info::device::name>()<<std::endl;
 norm_tests(s);gr_tests(s);moe_router_tests(s);qsa_gate_tests(s,other);std::cout<<"PASS norm/GR/MoE/router/QSA gate cases="<<cases<<std::endl;
 const auto before_state=replays;state_tests(s,other);std::cout<<"PASS native GDN state replays="<<replays-before_state<<std::endl;
 rope_tests(s,other);indexer_tests(s,other);flash_tests(s,other);std::cout<<"PASS rotary/indexer/attention cases="<<cases<<std::endl;
 ple_tests(s,other);ck(cudaStreamDestroy(s));ck(cudaStreamDestroy(other));
 std::cout<<"PASS primitive_cases="<<cases<<" graph_replays="<<replays<<" numerical_checks="<<numerical<<" bitwise_checks="<<bitwise<<" guard_checks="<<guards<<" max_scaled_error="<<max_scaled_error<<'\n';
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
