// Synthetic mathematical discriminator only; no model or production dispatch.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
constexpr int HK=16, HV=48;
constexpr size_t MAX_BYTES=1ULL<<30;
void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
enum class Kind {General, BetaZero, GateZero, BetaOne, Identity, Literal};
const char* kind_name(Kind k){switch(k){case Kind::General:return "general";case Kind::BetaZero:return "beta0_decay";case Kind::GateZero:return "g0";case Kind::BetaOne:return "beta1";case Kind::Identity:return "beta0_g0";case Kind::Literal:return "literal";}return "unknown";}
struct Fixture {int K,V,T,C;bool nonzero;Kind kind;};
size_t index(int r,int h,int c,int V){return (size_t(r)*HV+h)*V+c;} // Native [key row,value head,value column].
int key_head(int h,bool wrong){return wrong?h/(HV/HK):h%HK;}
template<class R> struct Guarded {
 static constexpr size_t G=8;
 static constexpr R sentinel=R(12345.25);
 static size_t storage_size(size_t n){require(n<=MAX_BYTES/sizeof(R)-2*G,"buffer cap");return n+2*G;}
 size_t size;
 std::vector<R> storage;
 explicit Guarded(size_t n):size(n),storage(storage_size(n),sentinel){}
 R* data(){return storage.data()+G;}
 const R* data()const{return storage.data()+G;}
 void fill(R x){std::fill(data(),data()+size,x);}
 void check()const{for(size_t i=0;i<G;++i){require(storage[i]==sentinel&&storage[G+size+i]==sentinel,"sentinel overwritten");}}
};
// Inputs materialize only the current chunk. Unused final-chunk slots stay NaN.
template<class R> struct Input {
 int K,V,C,live;
 Guarded<R> q,k,v,g,beta;
 Input(const Fixture& f,int start,int m):K(f.K),V(f.V),C(f.C),live(m),q(size_t(C)*HK*K),k(size_t(C)*HK*K),v(size_t(C)*HV*V),g(size_t(C)*HV),beta(size_t(C)*HV){
  q.fill(std::numeric_limits<R>::quiet_NaN());k.fill(std::numeric_limits<R>::quiet_NaN());v.fill(std::numeric_limits<R>::quiet_NaN());g.fill(std::numeric_limits<R>::quiet_NaN());beta.fill(std::numeric_limits<R>::quiet_NaN());
  for(int i=0;i<live;++i){int t=start+i;
   for(int h=0;h<HK;++h){double qnorm=0,knorm=0;std::array<double,128> qr{},kr{};
    for(int r=0;r<K;++r){qr[r]=double((t*3+h*11+r*7)%29-14)/17.0;kr[r]=double((t*5+h*17+r*13+3)%37-18)/19.0;qnorm+=qr[r]*qr[r];knorm+=kr[r]*kr[r];}
    require(f.kind==Kind::Literal||(qnorm>0&&knorm>0),"degenerate synthetic key/query");
    for(int r=0;r<K;++r){q.data()[(size_t(i)*HK+h)*K+r]=f.kind==Kind::Literal?R(1):R(qr[r]/std::sqrt(qnorm));k.data()[(size_t(i)*HK+h)*K+r]=f.kind==Kind::Literal?R(1):R(kr[r]/std::sqrt(knorm));}
   }
   for(int h=0;h<HV;++h){double gate=-0.002*double(1+(t*7+h*3)%11);double b=0.1+0.8*double((t*11+h*5)%23)/22.0;
    if(f.kind==Kind::GateZero||f.kind==Kind::Identity)gate=0;
    if(f.kind==Kind::BetaZero||f.kind==Kind::Identity)b=0;
    if(f.kind==Kind::BetaOne)b=1;
    if(f.kind==Kind::Literal){gate=-std::log(2.0);b=0.5;}
    g.data()[size_t(i)*HV+h]=R(gate);beta.data()[size_t(i)*HV+h]=R(b);
    for(int c=0;c<V;++c){double value=double((t*13+h*7+c*17)%41-20)/23.0;if(f.kind==Kind::Literal){constexpr double values[3]={2,4,-1};value=values[t];}v.data()[(size_t(i)*HV+h)*V+c]=R(value);}
   }
  }
 }
 void check()const{q.check();k.check();v.check();g.check();beta.check();
  for(size_t i=0;i<size_t(live)*HK*K;++i)require(std::isfinite(q.data()[i])&&std::isfinite(k.data()[i]),"nonfinite Q/K");
  for(size_t i=0;i<size_t(live)*HV*V;++i)require(std::isfinite(v.data()[i]),"nonfinite V");
  for(size_t i=0;i<size_t(live)*HV;++i)require(std::isfinite(g.data()[i])&&std::isfinite(beta.data()[i])&&beta.data()[i]>=R(0)&&beta.data()[i]<=R(1),"invalid gate/beta");
  for(size_t i=size_t(live)*HK*K;i<q.size;++i)require(std::isnan(q.data()[i])&&std::isnan(k.data()[i]),"padded Q/K changed");
  for(size_t i=size_t(live)*HV*V;i<v.size;++i)require(std::isnan(v.data()[i]),"padded V changed");
  for(size_t i=size_t(live)*HV;i<g.size;++i)require(std::isnan(g.data()[i])&&std::isnan(beta.data()[i]),"padded gates changed");
 }
};
template<class R> Guarded<R> initial(const Fixture& f){Guarded<R> s(size_t(f.K)*HV*f.V);for(int r=0;r<f.K;++r)for(int h=0;h<HV;++h)for(int c=0;c<f.V;++c){double value=f.nonzero?double((r*7+h*3+c*5)%31-15)/128.0:0;if(f.kind==Kind::Literal)value=3;s.data()[index(r,h,c,f.V)]=R(value);}return s;}
// Independent token recurrence: decay old state, project old state, rank-one correction, output updated state.
// Deliberately ordinary ordered sums, not an emulation of SYCL's four partial FMA reductions.
template<class R> void serial(const Input<R>& in,Guarded<R>& state,Guarded<R>& out,bool wrong=false){
 const R scale=R(1)/std::sqrt(R(in.K));
 for(int i=0;i<in.live;++i)for(int h=0;h<HV;++h){int kh=key_head(h,wrong);const R* k=in.k.data()+(size_t(i)*HK+kh)*in.K;const R* q=in.q.data()+(size_t(i)*HK+kh)*in.K;R a=std::exp(in.g.data()[size_t(i)*HV+h]),b=in.beta.data()[size_t(i)*HV+h];
  for(int c=0;c<in.V;++c){R projection=0;for(int r=0;r<in.K;++r)projection+=state.data()[index(r,h,c,in.V)]*k[r];R delta=b*(in.v.data()[(size_t(i)*HV+h)*in.V+c]-a*projection);
   for(int r=0;r<in.K;++r){auto p=index(r,h,c,in.V);state.data()[p]=a*state.data()[p]+k[r]*delta;}
   R output=0;for(int r=0;r<in.K;++r)output+=state.data()[index(r,h,c,in.V)]*q[r];out.data()[(size_t(i)*HV+h)*in.V+c]=output*scale;
  }
 }
}
// Independent causal/WY construction. No tokenwise state update occurs here.
// L_ij=beta_i exp(G_i-G_j) k_i^T k_j (j<i), L_ii=1.
// L U=diag(beta)V; L W=diag(beta exp(G))K; D=U-W S_start.
// Outputs and final state follow the expanded rank-one sum, carrying only final state between chunks.
template<class R> void wy(const Input<R>& in,Guarded<R>& state,Guarded<R>& out){
 const int m=in.live;const R scale=R(1)/std::sqrt(R(in.K));
 Guarded<R> L(size_t(m)*m),QK(size_t(m)*m),W(size_t(m)*in.K),U(size_t(m)*in.V),D(size_t(m)*in.V),G(m);
 L.fill(R(0));QK.fill(R(0));W.fill(R(0));U.fill(R(0));D.fill(R(0));
 for(int h=0;h<HV;++h){int kh=h%HK;R sum=0;
  for(int i=0;i<m;++i){sum+=in.g.data()[size_t(i)*HV+h];G.data()[i]=sum;L.data()[size_t(i)*m+i]=1;
   const R* ki=in.k.data()+(size_t(i)*HK+kh)*in.K;
   for(int j=0;j<i;++j){const R* kj=in.k.data()+(size_t(j)*HK+kh)*in.K;R dot=0;for(int r=0;r<in.K;++r)dot+=ki[r]*kj[r];L.data()[size_t(i)*m+j]=in.beta.data()[size_t(i)*HV+h]*std::exp(G.data()[i]-G.data()[j])*dot;}
  }
  // Two independent forward substitutions, before any use of the incoming state.
  for(int i=0;i<m;++i){R b=in.beta.data()[size_t(i)*HV+h];const R* ki=in.k.data()+(size_t(i)*HK+kh)*in.K;
   for(int r=0;r<in.K;++r){R x=b*std::exp(G.data()[i])*ki[r];for(int j=0;j<i;++j)x-=L.data()[size_t(i)*m+j]*W.data()[size_t(j)*in.K+r];W.data()[size_t(i)*in.K+r]=x;}
   for(int c=0;c<in.V;++c){R x=b*in.v.data()[(size_t(i)*HV+h)*in.V+c];for(int j=0;j<i;++j)x-=L.data()[size_t(i)*m+j]*U.data()[size_t(j)*in.V+c];U.data()[size_t(i)*in.V+c]=x;}
  }
  for(int i=0;i<m;++i)for(int c=0;c<in.V;++c){R x=U.data()[size_t(i)*in.V+c];for(int r=0;r<in.K;++r)x-=W.data()[size_t(i)*in.K+r]*state.data()[index(r,h,c,in.V)];D.data()[size_t(i)*in.V+c]=x;}
  for(int i=0;i<m;++i){const R* qi=in.q.data()+(size_t(i)*HK+kh)*in.K;
   for(int j=0;j<=i;++j){const R* kj=in.k.data()+(size_t(j)*HK+kh)*in.K;R dot=0;for(int r=0;r<in.K;++r)dot+=qi[r]*kj[r];QK.data()[size_t(i)*m+j]=std::exp(G.data()[i]-G.data()[j])*dot;}
   for(int c=0;c<in.V;++c){R base=0;for(int r=0;r<in.K;++r)base+=qi[r]*state.data()[index(r,h,c,in.V)];R output=std::exp(G.data()[i])*base;
    for(int j=0;j<=i;++j)output+=QK.data()[size_t(i)*m+j]*D.data()[size_t(j)*in.V+c];
    out.data()[(size_t(i)*HV+h)*in.V+c]=output*scale;
   }
  }
  if(m)for(int r=0;r<in.K;++r)for(int c=0;c<in.V;++c){R x=std::exp(G.data()[m-1])*state.data()[index(r,h,c,in.V)];for(int j=0;j<m;++j)x+=std::exp(G.data()[m-1]-G.data()[j])*in.k.data()[(size_t(j)*HK+kh)*in.K+r]*D.data()[size_t(j)*in.V+c];state.data()[index(r,h,c,in.V)]=x;}
 }
 L.check();QK.check();W.check();U.check();D.check();G.check();
 for(const auto* buffer:{&L,&QK,&W,&U,&D,&G})for(size_t p=0;p<buffer->size;++p)require(std::isfinite(buffer->data()[p]),"nonfinite WY scratch");
}
struct Error {double abs=0,reference=0;};
template<class A,class B> Error compare(const A* a,const B* b,size_t n){Error e;for(size_t i=0;i<n;++i){require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite computed value");e.abs=std::max(e.abs,std::abs(double(a[i])-double(b[i])));e.reference=std::max(e.reference,std::abs(double(a[i])));}return e;}
void merge(Error& a,Error b){a.abs=std::max(a.abs,b.abs);a.reference=std::max(a.reference,b.reference);}
void math_gate(Error e){require(e.abs<=1e-10*(1+e.reference),"synthetic double equation bound exceeded");}
template<class R> void output_guard(const Guarded<R>& output,size_t live){output.check();for(size_t i=live;i<output.size;++i)require(output.data()[i]==Guarded<R>::sentinel,"padded output written");}
void run(const Fixture& f,size_t ordinal){
 require(f.K>=1&&f.K<=128&&f.V>=1&&f.V<=128&&f.T>=0&&f.T<=129&&f.C>=1&&f.C<=64,"fixture bounds");
 if(f.kind==Kind::Literal)require(f.K==1&&f.V==1&&f.T==3&&f.nonzero,"literal fixture geometry");
 auto sd=initial<double>(f),wd=initial<double>(f);auto sf=initial<float>(f),wf=initial<float>(f);
 Error output_d,state_d,output_f,state_f,serial_precision,wy_precision,analytic;
 size_t chunks=0,live_updates=0;
 for(int start=0;start<f.T;start+=f.C){int m=std::min(f.C,f.T-start);Input<double> d(f,start,m);Input<float> a(f,start,m);d.check();a.check();
  Guarded<double> od(size_t(f.C)*HV*f.V),ow(size_t(f.C)*HV*f.V);Guarded<float> of(size_t(f.C)*HV*f.V),ofw(size_t(f.C)*HV*f.V);
  serial(d,sd,od);wy(d,wd,ow);serial(a,sf,of);wy(a,wf,ofw);
  size_t n=size_t(m)*HV*f.V;auto de=compare(od.data(),ow.data(),n);auto se=compare(sd.data(),wd.data(),sd.size);math_gate(de);math_gate(se);merge(output_d,de);merge(state_d,se);
  merge(output_f,compare(of.data(),ofw.data(),n));merge(state_f,compare(sf.data(),wf.data(),sf.size));merge(serial_precision,compare(od.data(),of.data(),n));merge(wy_precision,compare(ow.data(),ofw.data(),n));
  if(f.kind==Kind::Literal){constexpr double expected[3]={1.75,2.4375,0.109375};for(int i=0;i<m;++i)for(int h=0;h<HV;++h){Error x{std::abs(od.data()[size_t(i)*HV+h]-expected[start+i]),std::abs(expected[start+i])};math_gate(x);merge(analytic,x);}}
  output_guard(od,n);output_guard(ow,n);output_guard(of,n);output_guard(ofw,n);sd.check();wd.check();sf.check();wf.check();d.check();a.check();++chunks;live_updates+=size_t(m)*HV;
 }
 require(live_updates==size_t(f.T)*HV,"live token count");
 require(chunks==size_t((f.T+f.C-1)/f.C),"chunk/handoff count");
 sd.check();wd.check();sf.check();wf.check();math_gate(compare(sd.data(),wd.data(),sd.size));
 compare(sf.data(),wf.data(),sf.size); // Enforce finiteness even for zero live tokens; no FP32 numerical acceptance gate.
 if(f.kind==Kind::BetaZero||f.kind==Kind::Identity){auto init=initial<double>(f);for(int h=0;h<HV;++h){double sum=0;for(int t=0;t<f.T;++t){double gate=f.kind==Kind::Identity?0:-0.002*double(1+(t*7+h*3)%11);sum+=gate;}for(int r=0;r<f.K;++r)for(int c=0;c<f.V;++c){size_t p=index(r,h,c,f.V);double expected=std::exp(sum)*init.data()[p];Error e{std::abs(sd.data()[p]-expected),std::abs(expected)};math_gate(e);merge(analytic,e);}}
  if(f.kind==Kind::Identity){require(std::equal(sd.data(),sd.data()+sd.size,init.data()),"beta0/g0 identity changed state");}
 }
 if(f.T==0){auto init=initial<double>(f);require(std::equal(sd.data(),sd.data()+sd.size,init.data())&&std::equal(wd.data(),wd.data()+wd.size,init.data()),"empty live input updated state");}
 std::cout<<"CASE,"<<ordinal<<','<<kind_name(f.kind)<<','<<f.K<<','<<f.V<<','<<f.T<<','<<f.C<<','<<f.nonzero<<','<<chunks<<','<<live_updates<<','<<output_d.abs<<','<<state_d.abs<<','<<output_f.abs<<','<<state_f.abs<<','<<serial_precision.abs<<','<<wy_precision.abs<<','<<analytic.abs<<",pass\n";
}
void wrong_mapping(){Fixture f{4,3,17,8,true,Kind::General};auto good=initial<double>(f),bad=initial<double>(f);Error e;
 for(int start=0;start<f.T;start+=f.C){int m=std::min(f.C,f.T-start);Input<double> in(f,start,m);Guarded<double> a(size_t(f.C)*HV*f.V),b(size_t(f.C)*HV*f.V);serial(in,good,a);serial(in,bad,b,true);merge(e,compare(a.data(),b.data(),size_t(m)*HV*f.V));merge(e,compare(good.data(),bad.data(),good.size));output_guard(a,size_t(m)*HV*f.V);output_guard(b,size_t(m)*HV*f.V);good.check();bad.check();in.check();}
 require(e.abs>1e-6,"wrong contiguous key-head map was not discriminated");std::cout<<"NEGATIVE,contiguous_head_div3,"<<e.abs<<",rejected\n";
}
void physical_counts(){constexpr uint64_t T=262144,heads=48,K=128,V=128,bytes=4;for(uint64_t C:{1,8,16,32,64}){
 uint64_t chunks=(T+C-1)/C;uint64_t state=heads*K*V*bytes;
 // Hypothetical FP32 sequential-head working storage, not measured allocator/GPU peak.
 uint64_t head_scratch=(2*C*C+C*K+2*C*V+C)*bytes;
 uint64_t inputs=(2*C*16*K+C*heads*V+2*C*heads)*bytes,outputs=C*heads*V*bytes;
 std::cout<<"PHYSICAL_COUNTS,"<<T<<','<<C<<','<<chunks<<','<<state<<','<<head_scratch<<','<<inputs<<','<<outputs<<','<<chunks*state<<'\n';
 }}
} // namespace
int main(int argc,char**){try{require(argc==1,"no arguments; fixed bounded synthetic suite only");std::cout<<std::setprecision(17)<<"META,scope,synthetic_equations_only_no_model_parity_adoption_or_performance\nMETA,key_mapping,value_head_mod16\nMETA,double_bound,1e-10_times_1_plus_max_reference_abs_per_handoff\nMETA,float32,characterization_only\nCASE,ordinal,kind,K,V,T,chunk,nonzero,chunks,live_head_updates,double_output_abs,double_state_abs,float_output_abs,float_state_abs,serial_double_float_abs,WY_double_float_abs,analytic_abs,status\n";
 size_t count=0;
 for(int C:{1,8,16,32,64}){
  std::vector<int> lengths{0,C-1,C,C+1,2*C+1};std::sort(lengths.begin(),lengths.end());lengths.erase(std::unique(lengths.begin(),lengths.end()),lengths.end());
  for(int T:lengths)for(bool nz:{false,true})run({4,3,T,C,nz,Kind::General},++count);
  for(Kind kind:{Kind::BetaZero,Kind::GateZero,Kind::BetaOne,Kind::Identity})for(bool nz:{false,true})run({4,3,2*C+1,C,nz,kind},++count);
  if(C!=64)for(bool nz:{false,true})run({4,3,129,C,nz,Kind::General},++count);
 }
 run({1,1,3,1,true,Kind::Literal},++count);run({1,1,3,8,true,Kind::Literal},++count);
 // Only two full-model-head/width representatives, never T262144 execution.
 run({128,128,17,16,true,Kind::General},++count);run({128,128,65,64,false,Kind::General},++count);
 wrong_mapping();physical_counts();std::cout<<"RESULT,synthetic_math_pass,cases,"<<count<<",model_parity,false,adopted,false,performance,false,full_lifecycle,false\n";
 std::cout.flush();require(bool(std::cout),"output failure");return 0;
 }catch(const std::exception& e){std::cerr<<"RESULT,fail,"<<e.what()<<'\n';return 1;}}
