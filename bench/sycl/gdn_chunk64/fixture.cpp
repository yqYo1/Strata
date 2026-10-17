// Main-authored Strata fixture. Repository LICENSE applies.
// First qualification: exact dyadic, orthogonal-key cases only, not model quality.
#include "candidate.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace gc=strata::gdn_chunk64;
namespace sb=strata::sycl_backend;
namespace k=strata::kernels;
std::shared_ptr<sb::Runtime> rt;
void need(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
void drain() noexcept { sb::checked_cleanup([]{if(rt)rt->wait();},[]{},"GDN chunk64 fixture drain"); }
struct Drain {~Drain() noexcept {drain();}};
template<class T> struct Buffer {
  sb::Allocation storage;size_t count;
  std::vector<T> upload,download;
  explicit Buffer(size_t n):storage(rt,n*sizeof(T),sb::MemoryKind::Device),count(n){}
  T *data(){return storage.as<T>();}
  void put(const std::vector<T>& v){need(v.size()==count,"upload extent");rt->wait();upload=v;rt->wait(rt->compute().memcpy(data(),upload.data(),count*sizeof(T)));}
  std::vector<T> get(){rt->wait();download.resize(count);rt->wait(rt->compute().memcpy(download.data(),data(),count*sizeof(T)));return download;}
};
struct Scratch {
  Buffer<gc::BF> q,k,kb,vb,inverse,w,u,history,delta,raw;
  Buffer<float> g,lower;
  explicit Scratch(gc::Sizes n):q(n.qk),k(n.qk),kb(n.values),vb(n.values),
    inverse(n.inverse),w(n.values),u(n.values),history(n.history),delta(n.values),
    raw(n.values),g(n.gates),lower(n.lower){}
  gc::Workspace view(){return {q.data(),k.data(),kb.data(),vb.data(),inverse.data(),
    w.data(),u.data(),history.data(),delta.data(),raw.data(),g.data(),lower.data()};}
};
void gate(const char *phase){std::cout<<"GDN_CHUNK64_READY "<<phase<<std::endl;need(std::cin.get()=='X',"Main gate token");}
void run(int T){
  constexpr int C=10240,HV=48,D=128,Pad=16,State=HV*D*D;
  constexpr float Sentinel=1234.f;
  std::vector<float> hf(size_t(T)*C,0.f),zf(size_t(T)*HV*D,0.f),
    g(size_t(T)*HV,0.f),beta(g.size(),.5f),gamma(D,1.f),initial(State+2*Pad,Sentinel);
  for(int r=0;r<D;++r)for(int head=0;head<HV;++head)for(int d=0;d<D;++d)
    initial[Pad+(r*HV+head)*D+d]=float((r*3+head*5+d*7)%17-8)/16.f;
  for(int t=0;t<T;++t){
    for(int head=0;head<16;++head){
      const int r=(head*5+t)%D;
      hf[size_t(t)*C+head*D+r]=1.f;
      hf[size_t(t)*C+16*D+head*D+r]=1.f;
    }
    for(int head=0;head<HV;++head)for(int d=0;d<D;++d)
      hf[size_t(t)*C+32*D+head*D+d]=float((t*3+head*7+d*5)%9-4)/8.f;
  }
  std::vector<uint16_t> h(hf.size()),z(zf.size());
  for(size_t i=0;i<h.size();++i)h[i]=k::bf16_from_f32(hf[i]);
  for(size_t i=0;i<z.size();++i)z[i]=k::bf16_from_f32(zf[i]);
  // Independent exact arithmetic for orthogonal, dyadic one-hot keys.
  // No inverse or matrix helper is shared with the candidate.
  auto expected=initial;
  std::vector<uint16_t> expected_output(size_t(T)*HV*D);
  for(int t=0;t<T;++t)for(int head=0;head<HV;++head){
    const int r=((head%16)*5+t)%D;
    float raw[D],ss=0.f;
    for(int d=0;d<D;++d){
      const size_t index=Pad+(r*HV+head)*D+d;
      expected[index]=.5f*expected[index]+.5f*hf[size_t(t)*C+32*D+head*D+d];
      raw[d]=k::f32_from_bf16(k::bf16_from_f32(expected[index]/std::sqrt(128.f)));
      ss+=raw[d]*raw[d];
    }
    for(int d=0;d<D;++d)
      expected_output[(size_t(t)*HV+head)*D+d]=k::f16_from_f32((raw[d]/std::sqrt(ss/128.f+1e-6f))*.5f);
  }
  Buffer<uint16_t> H(h.size()),Z(z.size()),out(expected_output.size()+2*Pad);
  Buffer<float> G(g.size()),B(beta.size()),N(D),state(initial.size());
  Scratch scratch(gc::sizes(T));Drain pending;
  H.put(h);Z.put(z);G.put(g);B.put(beta);N.put(gamma);state.put(initial);
  std::vector<uint16_t> blank(out.count,0x5500);out.put(blank);
  auto invoke=[&](float *s,uint16_t *o,int count,gc::Workspace ws,void *stream){
    gc::run(s,H.data(),G.data(),B.data(),Z.data(),N.data(),1e-6f,o,count,ws,stream);
  };
  invoke(state.data()+Pad,out.data()+Pad,T,scratch.view(),&rt->compute());rt->wait();
  const auto actual=state.get();const auto result=out.get();
  need(actual==expected,"exact FP32 carry mismatch");
  size_t unequal=0;float max_abs=0.f;
  for(size_t i=0;i<expected_output.size();++i){
    unequal+=result[Pad+i]!=expected_output[i];
    max_abs=std::max(max_abs,std::abs(k::f32_from_f16(result[Pad+i])-k::f32_from_f16(expected_output[i])));
    need(std::isfinite(k::f32_from_f16(result[Pad+i])),"nonfinite dyadic output");
  }
  need(unequal==0,"exact FP16 oracle mismatch");
  for(int i=0;i<Pad;++i)need(result[i]==0x5500 && result[result.size()-Pad+i]==0x5500,"output guards");
  need(H.get()==h && Z.get()==z && G.get()==g && B.get()==beta && N.get()==gamma,"input mutation");
  auto reject=[&](auto action){bool refused=false;try{action();}catch(const std::invalid_argument&){refused=true;}need(refused,"invalid call admitted");};
  reject([&]{invoke(state.data()+Pad,out.data()+Pad,0,scratch.view(),&rt->compute());});
  reject([&]{invoke(state.data()+Pad,out.data()+Pad,8193,scratch.view(),&rt->compute());});
  reject([&]{invoke(state.data()+Pad,out.data()+Pad,T,scratch.view(),nullptr);});
  reject([&]{invoke(state.data()+Pad,reinterpret_cast<uint16_t*>(H.data()),T,scratch.view(),&rt->compute());});
  auto invalid=scratch.view();invalid.w=invalid.u;
  reject([&]{invoke(state.data()+Pad,out.data()+Pad,T,invalid,&rt->compute());});
  invalid=scratch.view();invalid.g=g.data();
  reject([&]{invoke(state.data()+Pad,out.data()+Pad,T,invalid,&rt->compute());});
  {Buffer<float> short_state(1);Drain more;
    reject([&]{invoke(short_state.data(),out.data()+Pad,T,scratch.view(),&rt->compute());});}
  state.put(initial);out.put(blank);
  invoke(state.data()+Pad,out.data()+Pad,T,scratch.view(),&rt->compute());rt->wait();
  need(state.get()==actual && out.get()==result,"repeat differs");
  std::cout<<"case T="<<T<<" heads=48 key_groups=16 state_width=128 oracle=exact_dyadic_orthogonal"
    <<" carry=PASS output_FP16=PASS unequal="<<unequal<<" max_abs="<<max_abs
    <<" guards=PASS inputs=PASS rejection=PASS repeat=PASS status=PASS"<<std::endl;
}
int main(int argc,char **argv){
  try{
    need(argc==2 && std::string(argv[1])=="--smoke","only smoke is qualified");
    gate("before-runtime");rt=sb::runtime_for();
    std::cout<<"device="<<rt->device().get_info<sycl::info::device::name>()
      <<" driver="<<rt->device().get_info<sycl::info::device::driver_version>()<<std::endl;
    gate("runtime-ready");run(1);run(3);rt->wait();rt.reset();gate("after-cleanup");
    std::cout<<"fixture_status=PASS general_numerics=UNRUN long_component=UNRUN model_PP=UNRUN physical262144_model=UNRUN RESTORE=UNRUN faults=EXTERNAL_REQUIRED"<<std::endl;
    return 0;
  }catch(const std::exception &error){drain();std::cerr<<"fixture_status=FAIL "<<error.what()<<std::endl;return 1;}
}
