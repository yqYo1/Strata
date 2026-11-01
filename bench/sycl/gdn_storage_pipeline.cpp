// Main-authored; repository LICENSE (MIT). Packed resident GDN3 component.
// No XeStrata source; no model dispatch, decode or physical-context claim.
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/detail/gdn_storage_casts.hpp"
#include "strata/sycl/runtime.hpp"
#include <oneapi/mkl/blas.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace s = strata::sycl_backend;
namespace p = strata::prefill;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;
constexpr size_t K=2560, C=10240, V=6144, HV=48, ST=128*V, Pad=32;
constexpr int Batch=64, Rounds=5;
constexpr float Eps=1e-6f;
void need(bool ok,const char *why) { if(!ok)throw std::runtime_error(why); }
struct CheckedDrain {
  s::Runtime &rt; bool armed=true;
  void finish() { rt.wait(); armed=false; }
  ~CheckedDrain() noexcept {
    if(armed)try { finish(); } catch(...) {
      std::cerr<<"checked retirement failed;owners remain unreleased\n"<<std::flush;
      std::_Exit(86);
    }
  }
};
template<class T> T sentinel();
template<> float sentinel<float>() { return -731.25f; }
template<> uint16_t sentinel<uint16_t>() { return 0x5a5a; }
template<class T> struct Buffer {
  std::shared_ptr<s::Runtime> rt; s::Allocation owner; size_t n;
  Buffer(const std::shared_ptr<s::Runtime> &runtime,size_t count)
      :rt(runtime),owner(rt,(count+2*Pad)*sizeof(T),s::MemoryKind::Device),n(count) {
    rt->wait(rt->compute().fill(owner.template as<T>(),sentinel<T>(),n+2*Pad));
  }
  T *data() { return owner.template as<T>()+Pad; }
  void set(const std::vector<T> &values) {
    need(values.size()==n,"input extent");
    rt->wait(rt->compute().memcpy(data(),values.data(),n*sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> all(n+2*Pad);
    rt->wait(rt->compute().memcpy(all.data(),owner.data(),all.size()*sizeof(T)));
    for(size_t i=0;i<Pad;++i)
      need(all[i]==sentinel<T>()&&all[Pad+n+i]==sentinel<T>(),"guard changed");
    all.erase(all.begin(),all.begin()+Pad);all.resize(n);return all;
  }
  void immutable(const std::vector<T> &expected) {
    const auto actual=get();
    need(actual.size()==expected.size()&&std::memcmp(actual.data(),expected.data(),n*sizeof(T))==0,
         "read-only input changed");
  }
};
template<class T> void exact(const std::vector<T> &a,const std::vector<T> &b,const char *why) {
  need(a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0,why);
}
void conversion_corpus(const std::shared_ptr<s::Runtime> &rt) {
  std::vector<uint16_t> half(65536);
  for(size_t i=0;i<half.size();++i)half[i]=uint16_t(i);
  std::vector<float> values;values.reserve(253964);
  auto both_signs=[&](float value) { values.push_back(value);values.push_back(-value); };
  const float inf=std::numeric_limits<float>::infinity();
  for(uint16_t h=0;h<0x7bff;++h) {
    const float a=f32_from_f16(h),b=f32_from_f16(uint16_t(h+1)),mid=(a+b)*.5f;
    for(float value:{a,mid,std::nextafter(mid,-inf),std::nextafter(mid,inf)})both_signs(value);
  }
  for(float value:{65504.f,65520.f,std::nextafter(65520.f,-inf),std::nextafter(65520.f,inf),
      65536.f,std::numeric_limits<float>::max(),std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),inf,std::numeric_limits<float>::quiet_NaN()})both_signs(value);
  need(values.size()==253964,"conversion corpus extent");
  Buffer<uint16_t> input_half(rt,half.size()),output_half(rt,values.size());
  Buffer<float> input_float(rt,values.size()),output_float(rt,half.size());
  CheckedDrain drain{*rt};input_half.set(half);input_float.set(values);
  const auto *h=input_half.data();auto *f=output_float.data();
  rt->compute().parallel_for(sycl::range<1>(half.size()),[=](sycl::id<1> i) {
    f[i]=p::detail::gdn_half_load(h[i]);
  });
  const auto *a=input_float.data();auto *b=output_half.data();
  rt->compute().parallel_for(sycl::range<1>(values.size()),[=](sycl::id<1> i) {
    b[i]=p::detail::gdn_half_store(a[i]);
  });
  rt->wait();const auto promoted=output_float.get();const auto rounded=output_half.get();
  for(size_t i=0;i<half.size();++i) {
    const float expected=f32_from_f16(half[i]);
    uint32_t want=0,got=0;std::memcpy(&want,&expected,4);std::memcpy(&got,&promoted[i],4);
    if(std::isnan(expected)?!std::isnan(promoted[i]):want!=got)
      throw std::runtime_error("half-load mismatch;input_bits="+std::to_string(i)+
          ";expected_f32_bits="+std::to_string(want)+";actual="+std::to_string(got));
  }
  for(size_t i=0;i<values.size();++i) {
    const uint16_t want=f16_from_f32(values[i]),got=rounded[i];
    const bool nan=(got&0x7c00)==0x7c00&&(got&0x03ff)!=0;
    if(std::isnan(values[i])?!nan:want!=got) {
      uint32_t bits=0;std::memcpy(&bits,&values[i],4);
      throw std::runtime_error("half-store mismatch;case="+std::to_string(i)+
          ";input_f32_bits="+std::to_string(bits)+";expected_half_bits="+std::to_string(want)+
          ";actual="+std::to_string(got));
    }
  }
  input_half.immutable(half);input_float.immutable(values);drain.finish();
  std::cout<<"{\"kind\":\"conversion\",\"half_inputs\":65536,\"float_inputs\":253964,"
      "\"finite_bits_exact\":true,\"nan_classes\":true,\"passed\":true}\n"<<std::flush;
}
uint32_t mix(uint32_t h) {
  h^=h>>16;h*=0x7feb352du;h^=h>>15;h*=0x846ca68bu;return h^(h>>16);
}
float dyadic(size_t i,uint32_t salt,int divisor) {
  return float(int(mix(uint32_t(i)^salt)%127)-63)/float(divisor);
}
std::vector<float> expanded(const std::vector<uint16_t> &values) {
  std::vector<float> out(values.size());
  for(size_t i=0;i<out.size();++i) {
    out[i]=f32_from_f16(values[i]);need(std::isfinite(out[i]),"nonfinite FP16 value");
  }
  return out;
}
struct Trial {
  Buffer<float> state,history,raw;
  Buffer<uint16_t> out;
  Trial(const std::shared_ptr<s::Runtime> &rt,int t)
      :state(rt,ST),history(rt,C*3),raw(rt,size_t(t)*V),out(rt,size_t(t)*V) {}
};
struct Case {
  std::shared_ptr<s::Runtime> rt; int t; std::string pattern;
  std::vector<uint16_t> x,wq,wz;
  std::vector<float> weights,gates,betas,gamma,initial,history;
  Buffer<uint16_t> X,WQ,WZ,Q16,Z16,H16;
  Buffer<float> W,G,B,N,Q32,Z32,H32;
  Trial reference,candidate;
  Case(const std::shared_ptr<s::Runtime> &runtime,int rows,const std::string &kind)
      :rt(runtime),t(rows),pattern(kind),x(size_t(t)*K),wq(C*K),wz(V*K),
       weights(C*4),gates(size_t(t)*HV),betas(gates.size()),gamma(128),initial(ST),history(C*3),
       X(rt,x.size()),WQ(rt,wq.size()),WZ(rt,wz.size()),Q16(rt,size_t(t)*C),
       Z16(rt,size_t(t)*V),H16(rt,size_t(t)*C),W(rt,weights.size()),G(rt,gates.size()),
       B(rt,betas.size()),N(rt,gamma.size()),Q32(rt,size_t(t)*C),Z32(rt,size_t(t)*V),
       H32(rt,size_t(t)*C),reference(rt,t),candidate(rt,t) {
    for(size_t i=0;i<x.size();++i) {
      const size_t column=i%K,row=i/K;
      float value=dyadic(kind=="cancel"?row*K+column/2:i,0x31415926u,512);
      if(kind=="cancel"&&column%2)value=-value;
      if(kind=="zero")value=0;
      if(kind=="onehot")value=column==mix(uint32_t(row))%K?.125f:0;
      x[i]=f16_from_f32(value);
    }
    for(auto *values:{&wq,&wz})for(size_t i=0;i<values->size();++i) {
      const size_t index=kind=="cancel"?(i/K)*K+(i%K)/2:i;
      (*values)[i]=f16_from_f32(dyadic(index,values==&wq?0x27182818u:0x16180339u,1024));
    }
    for(size_t i=0;i<weights.size();++i)weights[i]=dyadic(i,0x12345678u,256);
    for(size_t i=0;i<gates.size();++i) {
      gates[i]=-.125f-float(mix(uint32_t(i))%16)/32.f;
      betas[i]=float(1+mix(uint32_t(i)^0xabcdu)%15)/32.f;
    }
    for(size_t i=0;i<gamma.size();++i)gamma[i]=.5f+float(i%17)/32.f;
    for(size_t i=0;i<initial.size();++i)initial[i]=dyadic(i,0x13579u,4096);
    for(size_t i=0;i<history.size();++i)history[i]=dyadic(i,0x24680u,512);
  }
  void upload() {
    X.set(x);WQ.set(wq);WZ.set(wz);W.set(weights);G.set(gates);B.set(betas);N.set(gamma);
  }
  void reset(Trial &a) { a.state.set(initial);a.history.set(history); }
  void project(bool half) {
    auto &q=rt->compute();using oneapi::mkl::transpose;
    const auto *a=reinterpret_cast<const sycl::half *>(X.data());
    if(half) {
      oneapi::mkl::blas::row_major::gemm(q,transpose::nontrans,transpose::trans,t,C,K,
          sycl::half(1),a,K,reinterpret_cast<const sycl::half *>(WQ.data()),K,sycl::half(0),
          reinterpret_cast<sycl::half *>(Q16.data()),C);
      oneapi::mkl::blas::row_major::gemm(q,transpose::nontrans,transpose::trans,t,V,K,
          sycl::half(1),a,K,reinterpret_cast<const sycl::half *>(WZ.data()),K,sycl::half(0),
          reinterpret_cast<sycl::half *>(Z16.data()),V);
    } else {
      oneapi::mkl::blas::row_major::gemm(q,transpose::nontrans,transpose::trans,t,C,K,
          1.f,a,K,reinterpret_cast<const sycl::half *>(WQ.data()),K,0.f,Q32.data(),C);
      oneapi::mkl::blas::row_major::gemm(q,transpose::nontrans,transpose::trans,t,V,K,
          1.f,a,K,reinterpret_cast<const sycl::half *>(WZ.data()),K,0.f,Z32.data(),V);
    }
  }
  void consume16(int rows,int offset=0) {
    auto *stream=&rt->compute();const size_t start=size_t(offset);
    p::gdn_conv_f16(candidate.history.data(),Q16.data()+start*C,W.data(),H16.data()+start*C,rows,Eps,stream);
    p::gdn_recurrence_barriers3_f16(candidate.state.data(),H16.data()+start*C,G.data()+start*HV,
        B.data()+start*HV,Z16.data()+start*V,N.data(),Eps,candidate.raw.data()+start*V,
        candidate.out.data()+start*V,rows,stream);
  }
  void pipeline(bool half) {
    project(half);
    if(half)consume16(t);
    else {
      auto *stream=&rt->compute();
      p::gdn_conv(reference.history.data(),Q32.data(),W.data(),H32.data(),t,Eps,stream);
      p::gdn_recurrence_barriers3(reference.state.data(),H32.data(),G.data(),B.data(),Z32.data(),
          N.data(),Eps,reference.raw.data(),reference.out.data(),t,stream);
    }
  }
  void producers() {
    const auto q=Q16.get(),z=Z16.get();
    const auto a=Q32.get(),b=Z32.get();double worst=0;
    for(int which=0;which<2;++which) {
      const auto &output=which?z:q;const auto &fp32=which?b:a;const auto &w=which?wz:wq;
      const size_t n=which?V:C;
      for(size_t spot=0;spot<32;++spot) {
        const size_t row=mix(uint32_t(spot)^0x13579u)%size_t(t),column=mix(uint32_t(spot)^0x24680u)%n;
        double sum=0,abs_sum=0;
        for(size_t k=0;k<K;++k) {
          const double product=double(f32_from_f16(x[row*K+k]))*f32_from_f16(w[column*K+k]);
          sum+=product;abs_sum+=std::abs(product);
        }
        constexpr double g32=double(K-1)*0x1p-24/(1.-double(K-1)*0x1p-24);
        const double bound=2*g32*abs_sum+0x1p-11*std::abs(sum)+0x1p-25;
        const double value=f32_from_f16(output[row*n+column]);
        need(abs_sum<65504&&std::isfinite(value)&&std::isfinite(fp32[row*n+column]),"producer range");
        need(std::abs(value-sum)<=bound&&std::abs(fp32[row*n+column]-sum)<=bound,"packed producer oracle");
        worst=std::max(worst,std::abs(value-sum)/bound);
      }
    }
    (void)expanded(q);(void)expanded(z);
    std::cout<<"{\"kind\":\"packed_producers\",\"T\":"<<t<<",\"FP64_spots\":64,\"max_bound_fraction\":"<<worst<<",\"passed\":true}\n";
  }
  void diagnostic() {
    const auto a=reference.out.get(),b=candidate.out.get();
    const auto sa=reference.state.get(),sb=candidate.state.get();
    double squared=0,base=0,maximum=0,state_maximum=0;
    for(size_t i=0;i<a.size();++i) {
      const double u=f32_from_f16(a[i]),v=f32_from_f16(b[i]);
      need(std::isfinite(u)&&std::isfinite(v),"nonfinite output");
      squared+=(v-u)*(v-u);base+=u*u;maximum=std::max(maximum,std::abs(v-u));
    }
    for(size_t i=0;i<sa.size();++i) {
      need(std::isfinite(sa[i])&&std::isfinite(sb[i]),"nonfinite state");
      state_maximum=std::max(state_maximum,double(std::abs(sa[i]-sb[i])));
    }
    std::cout<<"{\"kind\":\"FP32_difference\",\"T\":"<<t<<",\"relative_output_RMS\":"
      <<std::sqrt(squared/std::max(base,1e-30))<<",\"max_output_abs\":"<<maximum
      <<",\"max_state_abs\":"<<state_maximum<<",\"diagnostic_only\":true,\"passed\":true}\n";
  }
  void canonical(int repeats) {
    need(repeats==1||(t>=3&&repeats==Batch),"canonical repeated-case domain");
    const auto cq=Q16.get(),cz=Z16.get(),ch=H16.get();
    const auto expected_state=candidate.state.get(),expected_history=candidate.history.get();
    const auto expected_output=candidate.out.get();
    Q32.set(expanded(cq));Z32.set(expanded(cz));reset(reference);
    for(int iteration=0;iteration<repeats;++iteration) {
      if(iteration<2) {
        p::gdn_conv(reference.history.data(),Q32.data(),W.data(),H32.data(),t,Eps,&rt->compute());
        auto h=H32.get();
        for(size_t i=0;i<h.size();++i) {
          const uint16_t bits=f16_from_f32(h[i]);
          if(iteration==std::min(repeats-1,1))need(bits==ch[i],"typed conv differs from rounded FP32 conv");
          h[i]=f32_from_f16(bits);need(std::isfinite(h[i]),"conv storage overflow");
        }
        H32.set(h);
      }
      p::gdn_recurrence_barriers3(reference.state.data(),H32.data(),G.data(),B.data(),Z32.data(),
          N.data(),Eps,reference.raw.data(),reference.out.data(),t,&rt->compute());
      if((iteration+1)%4==0)rt->wait();
    }
    rt->wait();
    exact(reference.state.get(),expected_state,"typed recurrence FP32 state differs from rounded-storage reference");
    exact(reference.history.get(),expected_history,"typed conv FP32 history differs");
    exact(reference.out.get(),expected_output,"typed recurrence FP16 output differs");
    exact(Q16.get(),cq,"QKV16 consumer modified input");exact(Z16.get(),cz,"Z16 consumer modified input");
    if(repeats==1&&t>=4) {
      reset(candidate);consume16(1);consume16(2,1);consume16(t-3,3);rt->wait();
      exact(candidate.state.get(),expected_state,"split recurrence state differs");
      exact(candidate.history.get(),expected_history,"split conv history differs");
      exact(candidate.out.get(),expected_output,"split output differs");exact(H16.get(),ch,"split conv rounding differs");
    }
    std::cout<<"{\"kind\":\"canonical\",\"T\":"<<t<<",\"pattern\":\""<<pattern
      <<"\",\"repeats\":"<<repeats<<",\"exact_rounded_storage_state_history_output\":true,\"split_1_2_tail\":"
      <<(repeats==1&&t>=4?"true":"false")<<",\"guards\":true,\"passed\":true}\n";
  }
  void immutable() { X.immutable(x);WQ.immutable(wq);WZ.immutable(wz);W.immutable(weights);G.immutable(gates);B.immutable(betas);N.immutable(gamma); }
  void legacy() {
    reset(reference);reset(candidate);project(false);
    p::gdn_conv(reference.history.data(),Q32.data(),W.data(),H32.data(),t,Eps,&rt->compute());
    p::gdn_recurrence(reference.state.data(),H32.data(),G.data(),B.data(),Z32.data(),N.data(),Eps,
        reference.raw.data(),reference.out.data(),t,&rt->compute());
    p::gdn_recurrence_barriers3(candidate.state.data(),H32.data(),G.data(),B.data(),Z32.data(),N.data(),Eps,
        candidate.raw.data(),candidate.out.data(),t,&rt->compute());rt->wait();
    exact(reference.state.get(),candidate.state.get(),"legacy FP32 state changed");
    exact(reference.raw.get(),candidate.raw.get(),"legacy FP32 output changed");
    exact(reference.out.get(),candidate.out.get(),"legacy FP16 output changed");
    std::cout<<"{\"kind\":\"legacy\",\"T\":"<<t<<",\"FP32_fused_vs_columns_exact\":true,\"passed\":true}\n";
  }
  void reject() {
    reset(candidate);const auto state=candidate.state.get(),hist=candidate.history.get();
    const auto output=candidate.out.get(),h=H16.get();int count=0;
    for(int which=0;which<2;++which)for(int mode=0;mode<9;++mode) {
      bool refused=false;float host=0;
      try {
        const int rows=mode==0?0:mode==1?-1:mode==2?8193:1;
        const float eps=mode==3?NAN:mode==4?-1.f:Eps;
        if(which==0)p::gdn_conv_f16(mode==8?&host:candidate.history.data(),
            mode==5?nullptr:Q16.data(),W.data(),mode==6?Q16.data():mode==7?H16.data()+1:H16.data(),rows,eps,&rt->compute());
        else p::gdn_recurrence_barriers3_f16(mode==8?&host:candidate.state.data(),
            mode==5?nullptr:H16.data(),G.data(),B.data(),Z16.data(),N.data(),eps,
            mode==6?candidate.state.data():candidate.raw.data(),mode==7?candidate.out.data()+1:candidate.out.data(),rows,&rt->compute());
      } catch(const std::invalid_argument &) { refused=true; }
        catch(const std::out_of_range &) { refused=true; }
      need(refused,"invalid typed GDN input accepted");++count;rt->wait();
    }
    exact(candidate.state.get(),state,"rejection wrote state");exact(candidate.history.get(),hist,"rejection wrote history");
    exact(candidate.out.get(),output,"rejection wrote output");exact(H16.get(),h,"rejection wrote conv output");
    std::cout<<"{\"kind\":\"rejection\",\"cases\":"<<count<<",\"synchronous_no_write\":true,\"passed\":true}\n";
  }
};
void timing(Case &c,int arm,int count,const char *phase,int round,bool empty=false) {
  auto &trial=arm?c.candidate:c.reference;c.reset(trial);c.rt->wait();
  const auto start=std::chrono::steady_clock::now();
  for(int i=0;i<count;++i) {
    if(!empty)c.pipeline(arm!=0);
    if((i+1)%4==0)c.rt->wait();
  }
  c.rt->wait();const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  std::cout<<"{\"kind\":\"timing\",\"T\":"<<c.t<<",\"arm\":\""<<(empty?"empty":arm?"FP16":"FP32")
    <<"\",\"phase\":\""<<phase<<"\",\"round\":"<<round<<",\"calls\":"<<count<<",\"host_ms\":"<<ms<<",\"passed\":true}\n"<<std::flush;
}
void run(const std::shared_ptr<s::Runtime> &rt,int t,const std::string &kind,int order=-1) {
  std::cerr<<"case-start T="<<t<<" pattern="<<kind<<std::endl;
  Case c(rt,t,kind);CheckedDrain drain{*rt};c.upload();
  if(order<0) {
    if(t==17)c.legacy();
    for(bool half:{false,true}) { c.reset(half?c.candidate:c.reference);c.pipeline(half);rt->wait(); }
    c.producers();c.diagnostic();c.canonical(1);if(t==1)c.reject();
  } else {
    for(int a=0;a<2;++a)timing(c,(a+order)%2,1,"first_shape_call",-1);
    c.producers();c.diagnostic();c.canonical(1);
    for(int a=0;a<2;++a)timing(c,(a+order)%2,4,"warmup",-1);
    for(int i=0;i<5;++i)timing(c,0,Batch,"timer_control",i,true);
    for(int round=0;round<Rounds;++round)for(int a=0;a<2;++a)timing(c,(a+round+order)%2,Batch,"steady",round);
    c.diagnostic();c.canonical(Batch);
  }
  c.immutable();drain.finish();std::cerr<<"case-complete"<<std::endl;
}
} // namespace
int main(int argc,char **argv) {
  try {
    std::cout<<std::setprecision(17);
    need(argc==2||(argc==3&&std::string(argv[1])=="--bench"),"use --smoke|--validate|--bench 0|1|2");
    const std::string mode=argv[1];
    need(mode=="--smoke"||mode=="--validate"||mode=="--bench","invalid mode");
    if(mode=="--bench")need(argc==3&&(std::string(argv[2])=="0"||std::string(argv[2])=="1"||std::string(argv[2])=="2"),"invalid process ordinal");
    auto rt=std::make_shared<s::Runtime>(0);
    need(rt->device().is_gpu()&&rt->device().has(sycl::aspect::fp16),"GPU FP16 required");
    need(rt->compute().has_property<sycl::property::queue::in_order>(),"in-order compute required");
    conversion_corpus(rt);
    if(mode=="--smoke") { run(rt,1,"general");run(rt,17,"general"); }
    else if(mode=="--validate") {
      for(int t:{1,2,3,127,129,8191,8192})run(rt,t,"general");
      for(const char *kind:{"zero","cancel","onehot"})run(rt,129,kind);
    } else for(int t:{4096,8192})run(rt,t,"general",argv[2][0]-'0');
    rt->wait();std::cout<<"fixture_status=PASS;packed resident GDN3 only\n"<<std::flush;return 0;
  } catch(const std::exception &e) { std::cerr<<e.what()<<std::endl;return 1; }
}
