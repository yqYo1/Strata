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
// The readout's RMS normalization/sigmoid contract does not distinguish signed
// zero. Keep exact bits for every nonzero BF16/FP16 value; use no error tolerance.
bool same_value(uint16_t a,uint16_t b){return a==b || ((a|b)&0x7fff)==0;}
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
  std::vector<uint16_t> expected_output(size_t(T)*HV*D),expected_raw(expected_output.size());
  for(int t=0;t<T;++t)for(int head=0;head<HV;++head){
    const int r=((head%16)*5+t)%D;
    float raw[D],ss=0.f;
    for(int d=0;d<D;++d){
      const size_t index=Pad+(r*HV+head)*D+d;
      expected[index]=.5f*expected[index]+.5f*hf[size_t(t)*C+32*D+head*D+d];
      raw[d]=k::f32_from_bf16(k::bf16_from_f32(expected[index]/std::sqrt(128.f)));
      expected_raw[(size_t(t)*HV+head)*D+d]=k::bf16_from_f32(raw[d]);
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
  const auto raw=scratch.raw.get();
  size_t raw_unequal=0,raw_value_unequal=0,double_unequal=0,unequal=0,value_unequal=0;float max_abs=0.f,raw_max_abs=0.f;
  unsigned max_ulp=0;
  for(size_t i=0;i<expected_output.size();++i){
    const int t=i/(HV*D),head=(i/D)%HV,d=i%D;
    const uint16_t raw_bits=sycl::bit_cast<uint16_t>(raw[((size_t(t/64)*HV+head)*64+t%64)*D+d]);
    raw_unequal+=raw_bits!=expected_raw[i];
    raw_value_unequal+=!same_value(raw_bits,expected_raw[i]);
    raw_max_abs=std::max(raw_max_abs,std::abs(k::f32_from_bf16(raw_bits)-k::f32_from_bf16(expected_raw[i])));
    unequal+=result[Pad+i]!=expected_output[i];
    value_unequal+=!same_value(result[Pad+i],expected_output[i]);
    auto ordered=[](uint16_t x){return (x&0x8000)?unsigned(0xffffu-x):unsigned(x+0x8000u);};
    const unsigned a=ordered(result[Pad+i]),b=ordered(expected_output[i]);
    max_ulp=std::max(max_ulp,a>b?a-b:b-a);
    max_abs=std::max(max_abs,std::abs(k::f32_from_f16(result[Pad+i])-k::f32_from_f16(expected_output[i])));
    need(std::isfinite(k::f32_from_f16(result[Pad+i])),"nonfinite dyadic output");
    double ss=0.;
    for(int col=0;col<D;++col){double x=k::f32_from_bf16(expected_raw[(i/D)*D+col]);ss+=x*x;}
    const uint16_t exact=k::f16_from_f32(float(double(k::f32_from_bf16(expected_raw[i]))/std::sqrt(ss/128.+double(1e-6f))*.5));
    double_unequal+=result[Pad+i]!=exact;
    if(result[Pad+i]!=expected_output[i] && unequal<=4)
      std::cout<<"output_difference i="<<i<<" actual_bits="<<result[Pad+i]<<" scalar_FP32_bits="<<expected_output[i]<<" FP64_norm_bits="<<exact<<std::endl;
  }
  std::cout<<"dyadic_diagnostic T="<<T<<" carry_exact=PASS raw_BF16_unequal="<<raw_unequal
    <<" raw_value_unequal="<<raw_value_unequal<<" raw_max_abs="<<raw_max_abs
    <<" output_unequal="<<unequal<<" output_max_ULP="<<max_ulp<<" output_max_abs="<<max_abs
    <<" output_value_unequal="<<value_unequal
    <<" FP64_norm_unequal="<<double_unequal<<std::endl;
  need(raw_value_unequal==0,"exact BF16 nonzero value mismatch");
  need(value_unequal==0,"exact FP16 nonzero value mismatch");
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
    <<" carry=PASS raw_BF16=PASS output_FP16=PASS zero_sign_ignored=true unequal="<<unequal<<" max_abs="<<max_abs
    <<" guards=PASS inputs=PASS rejection=PASS repeat=PASS status=PASS"<<std::endl;
}
// Independent sequential mathematical reference. It does not reproduce the
// candidate's full64 inverse or BF16 intermediates. Errors are diagnostic;
// framework-contract and model-quality acceptance remain separate gates.
void run_general(int T){
  constexpr int C=10240,HV=48,D=128,Pad=16,State=HV*D*D;
  constexpr float Sentinel=1234.f,Eps=1e-6f;
  std::vector<uint16_t> h(size_t(T)*C),z(size_t(T)*HV*D);
  std::vector<float> gates(size_t(T)*HV),beta(gates.size()),gamma(D),initial(State+2*Pad,Sentinel);
  uint32_t random=0x8badf00du;
  auto sample=[&](){random^=random<<13;random^=random>>17;random^=random<<5;return float(int(random%1025)-512)/512.f;};
  for(int i=0;i<State;++i)initial[Pad+i]=sample()/32.f;
  for(int d=0;d<D;++d)gamma[d]=.875f+float(d%17)/64.f;
  for(int t=0;t<T;++t){
    for(int kind=0;kind<2;++kind)for(int head=0;head<16;++head){
      float row[D];double ss=0.;
      for(float &v:row){v=sample();ss+=double(v)*v;}
      for(int d=0;d<D;++d)h[size_t(t)*C+(kind*16+head)*D+d]=k::bf16_from_f32(float(double(row[d])/std::sqrt(ss)));
    }
    for(int head=0;head<HV;++head){
      gates[size_t(t)*HV+head]=(t==2 && head%7==0)?-16.f:-float((t+head)%5)/64.f;
      beta[size_t(t)*HV+head]=float((t*3+head)%8)/8.f;
      for(int d=0;d<D;++d){h[size_t(t)*C+32*D+head*D+d]=k::bf16_from_f32(sample()/2.f);
        z[(size_t(t)*HV+head)*D+d]=k::bf16_from_f32(sample());}
    }
  }
  std::vector<double> reference(State);
  for(int i=0;i<State;++i)reference[i]=initial[Pad+i];
  std::vector<uint16_t> reference_raw(size_t(T)*HV*D),reference_out(reference_raw.size());
  for(int t=0;t<T;++t)for(int head=0;head<HV;++head){
    const double decay=std::exp(double(gates[size_t(t)*HV+head]));
    double projection[D]={},delta[D],readout[D]={};
    for(int r=0;r<D;++r){
      const double key=k::f32_from_bf16(h[size_t(t)*C+16*D+(head%16)*D+r]);
      for(int d=0;d<D;++d)projection[d]+=reference[(size_t(r)*HV+head)*D+d]*key;
    }
    for(int d=0;d<D;++d)delta[d]=(double(k::f32_from_bf16(h[size_t(t)*C+32*D+head*D+d]))-decay*projection[d])*beta[size_t(t)*HV+head];
    for(int r=0;r<D;++r){
      const double key=k::f32_from_bf16(h[size_t(t)*C+16*D+(head%16)*D+r]);
      const double query=k::f32_from_bf16(h[size_t(t)*C+(head%16)*D+r]);
      for(int d=0;d<D;++d){double &value=reference[(size_t(r)*HV+head)*D+d];
        value=value*decay+key*delta[d];readout[d]+=value*query;}
    }
    double ss=0.;
    for(int d=0;d<D;++d){const size_t i=(size_t(t)*HV+head)*D+d;
      reference_raw[i]=k::bf16_from_f32(float(readout[d]/std::sqrt(128.)));
      readout[d]=k::f32_from_bf16(reference_raw[i]);ss+=readout[d]*readout[d];}
    for(int d=0;d<D;++d){const size_t i=(size_t(t)*HV+head)*D+d;
      reference_out[i]=k::f16_from_f32(float((readout[d]/std::sqrt(ss/128.+double(Eps)))*gamma[d]/(1.+std::exp(-double(k::f32_from_bf16(z[i]))))));}
  }
  Buffer<uint16_t> H(h.size()),Z(z.size()),out(reference_out.size()+2*Pad);
  Buffer<float> G(gates.size()),B(beta.size()),N(D),state(initial.size());
  Scratch scratch(gc::sizes(T));Drain pending;
  H.put(h);Z.put(z);G.put(gates);B.put(beta);N.put(gamma);state.put(initial);
  std::vector<uint16_t> blank(out.count,0x5500);out.put(blank);
  auto invoke=[&](int begin,int count){gc::run(state.data()+Pad,H.data()+size_t(begin)*C,
    G.data()+size_t(begin)*HV,B.data()+size_t(begin)*HV,Z.data()+size_t(begin)*HV*D,N.data(),Eps,
    out.data()+Pad+size_t(begin)*HV*D,count,scratch.view(),&rt->compute());};
  invoke(0,T);rt->wait();const auto actual=state.get();const auto result=out.get();const auto raw=scratch.raw.get();
  struct Error {double squared=0.,reference_squared=0.,maximum=0.;
    void add(double x,double y){need(std::isfinite(x)&&std::isfinite(y),"nonfinite general value");
      const double e=x-y;squared+=e*e;reference_squared+=y*y;maximum=std::max(maximum,std::abs(e));}
    double relative()const{return std::sqrt(squared/std::max(reference_squared,1e-300));}} carry_error,raw_error,output_error;
  for(int i=0;i<State;++i)carry_error.add(actual[Pad+i],reference[i]);
  for(size_t i=0;i<reference_out.size();++i){const int t=i/(HV*D),head=(i/D)%HV,d=i%D;
    raw_error.add(float(raw[((size_t(t/64)*HV+head)*64+t%64)*D+d]),k::f32_from_bf16(reference_raw[i]));
    output_error.add(k::f32_from_f16(result[Pad+i]),k::f32_from_f16(reference_out[i]));}
  for(int i=0;i<Pad;++i)need(actual[i]==Sentinel && actual[actual.size()-Pad+i]==Sentinel &&
    result[i]==0x5500 && result[result.size()-Pad+i]==0x5500,"general guards");
  need(H.get()==h && Z.get()==z && G.get()==gates && B.get()==beta && N.get()==gamma,"general input mutation");
  state.put(initial);out.put(blank);invoke(0,T);rt->wait();
  need(state.get()==actual && out.get()==result,"general repeat differs");
  if(T>64){state.put(initial);out.put(blank);invoke(0,64);invoke(64,T-64);rt->wait();
    need(state.get()==actual && out.get()==result,"aligned-call carry handoff differs");}
  std::cout<<"case T="<<T<<" oracle=sequential_FP64_math accuracy=DIAGNOSTIC_ONLY"
    <<" carry_rel_L2="<<carry_error.relative()<<" carry_max_abs="<<carry_error.maximum
    <<" raw_rel_L2="<<raw_error.relative()<<" raw_max_abs="<<raw_error.maximum
    <<" output_rel_L2="<<output_error.relative()<<" output_max_abs="<<output_error.maximum
    <<" finite=PASS guards=PASS inputs=PASS repeat=PASS aligned_handoff="<<(T>64?"PASS":"NA")
    <<" status=SAFETY_PASS"<<std::endl;
}
int main(int argc,char **argv){
  try{
    need(argc==2 && (std::string(argv[1])=="--smoke" || std::string(argv[1])=="--boundaries" || std::string(argv[1])=="--general"),"expected smoke, boundaries or general");
    gate("before-runtime");rt=sb::runtime_for();
    std::cout<<"device="<<rt->device().get_info<sycl::info::device::name>()
      <<" driver="<<rt->device().get_info<sycl::info::device::driver_version>()<<std::endl;
    gate("runtime-ready");
    if(std::string(argv[1])=="--smoke"){run(1);run(3);}
    else if(std::string(argv[1])=="--general")for(int T:{63,64,65,127,128,129})run_general(T);
    else for(int T:{63,64,65,127,128,129})run(T);
    rt->wait();rt.reset();gate("after-cleanup");
    std::cout<<"fixture_status=PASS general_numerics="<<(std::string(argv[1])=="--general"?"DIAGNOSTIC_ONLY":"UNRUN")
      <<" long_component=UNRUN model_PP=UNRUN physical262144_model=UNRUN RESTORE=UNRUN faults=EXTERNAL_REQUIRED"<<std::endl;
    return 0;
  }catch(const std::exception &error){drain();std::cerr<<"fixture_status=FAIL "<<error.what()<<std::endl;return 1;}
}
