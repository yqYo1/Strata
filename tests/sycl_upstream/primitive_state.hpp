#pragma once
// The mathematical delta recurrence is evaluated in FP64 on the CPU. Store
// boundaries are explicit; the oracle does not execute the generated GPU body.
void state_tests(cudaStream_t s,cudaStream_t other){
 for(auto heads:{std::pair{1,1},std::pair{2,6},std::pair{16,48}}){
  const auto [hk,hv]=heads;constexpr int S=128;const int channels=(2*hk+hv)*S;
  context="GDN pipeline hk="+std::to_string(hk)+" hv="+std::to_string(hv);
  auto history=random_values(size_t(channels)*3,2),weights=random_values(size_t(channels)*4,3),state=random_values(size_t(S)*hv*S,.5f),gamma=random_values(S);
  for(auto& v:gamma)v+=1.f;
  std::vector<float> alpha(hv),dt(hv),a(hv),beta(hv),x(channels);
  for(int h=0;h<hv;++h){dt[h]=float(h%3-1)/8;a[h]=-.05f-float(h%5)/64;}
  Gpu dh((history.size()+16)*4),dw(weights.size()*4),dx(x.size()*4),dst((state.size()+16)*4),dalpha(hv*4),ddt(hv*4),da(hv*4),dbeta((hv+16)*4),dgate((hv+16)*4),dgamma(gamma.size()*4),draw((channels+16)*4),dsilu((channels+16)*4),dy((size_t(hv)*S+16)*4),dout((size_t(hv)*S+16)*4);
  auto histinit=guarded(history.size()),stateinit=guarded(state.size());std::copy(history.begin(),history.end(),histinit.begin());std::copy(state.begin(),state.end(),stateinit.begin());dh.put(histinit,s);dst.put(stateinit,s);dw.put(weights,s);ddt.put(dt,s);da.put(a,s);dgamma.put(gamma,s);
  auto rawinit=guarded(channels),outputinit=guarded(size_t(hv)*S),headinit=guarded(hv);draw.put(rawinit,s);dsilu.put(rawinit,s);dy.put(outputinit,s);dout.put(outputinit,s);dbeta.put(headinit,s);dgate.put(headinit,s);ck(cudaStreamSynchronize(s));
  const k::GdnShapes shape{S,hk,hv};
  auto record=[&]{
   k::native_gdn_conv_silu(dh.as<float>(),dx.as<float>(),dw.as<float>(),draw.as<float>(),dsilu.as<float>(),channels,4,s);
   k::native_gdn_l2_norm(dsilu.as<float>(),hk,S,1e-6f,s);
   k::native_gdn_l2_norm(dsilu.as<float>()+hk*S,hk,S,1e-6f,s);
   k::native_gdn_beta_gate(dbeta.as<float>(),hv,s);
   k::native_gdn_gate(dalpha.as<float>(),ddt.as<float>(),da.as<float>(),dgate.as<float>(),hv,s);
   k::native_gdn_step(dst.as<float>(),dsilu.as<float>(),dsilu.as<float>()+hk*S,dsilu.as<float>()+2*hk*S,dgate.as<float>(),dbeta.as<float>(),dy.as<float>(),shape,s);
   k::native_gdn_out_norm(dy.as<float>(),dx.as<float>()+2*hk*S,dgamma.as<float>(),dout.as<float>(),hv,S,1e-6f,s);
  };
  Graph graph(s,record,7);require(dst.get<float>(other)==stateinit && dh.get<float>(other)==histinit,"GDN capture changed state/history");
  for(int step=0;step<5;++step){
   x=random_values(channels,3);for(int h=0;h<hv;++h){constexpr float extremes[]={-40,-16,-.5f,0,8,20,22,40};alpha[h]=extremes[(h+step)%8];beta[h]=float((h*7+step)%19-9);}
   dx.put(x,s);dalpha.put(alpha,s);auto binit=headinit;std::copy(beta.begin(),beta.end(),binit.begin());dbeta.put(binit,s);graph.run(s);
   auto raw=draw.get<float>(s),silu=dsilu.get<float>(s),bg=dbeta.get<float>(s),gg=dgate.get<float>(s),gpuState=dst.get<float>(s),output=dy.get<float>(s),closed=dout.get<float>(s),hist=dh.get<float>(s);
   std::vector<float> activated(channels),betaRef(hv),gateRef(hv);
   for(int c=0;c<channels;++c){double conv=0;for(int tap=0;tap<4;++tap)conv+=double(tap==3?x[c]:history[c*3+tap])*weights[c*4+tap];near(raw[c],conv,"GDN convolution CPU reference");activated[c]=float(conv*sigmoid(conv));
    history[c*3]=history[c*3+1];history[c*3+1]=history[c*3+2];history[c*3+2]=x[c];
   }
   for(int row=0;row<2*hk;++row){double ss=0;for(int i=0;i<S;++i)ss+=double(activated[row*S+i])*activated[row*S+i];for(int i=0;i<S;++i)activated[row*S+i]=float(activated[row*S+i]/std::sqrt(ss+1e-6f));}
   for(int c=0;c<channels;++c)near(silu[c],activated[c],"GDN SiLU/L2 preprocessing");
   for(int h=0;h<hv;++h){betaRef[h]=float(sigmoid(beta[h]));const float u=alpha[h]+dt[h];gateRef[h]=float((u>20?double(u):std::log1p(std::exp(double(u))))*a[h]);near(bg[h],betaRef[h],"GDN sigmoid beta");near(gg[h],gateRef[h],"GDN softplus/log1p threshold");}
   std::vector<float> expected_output(size_t(hv)*S);
   for(int h=0;h<hv;++h)for(int col=0;col<S;++col){double kv=0;const int qh=h%hk;for(int row=0;row<S;++row)kv+=double(state[(size_t(row)*hv+h)*S+col])*activated[(hk+qh)*S+row];const double decay=std::exp(double(gateRef[h])),delta=(activated[(2*hk+h)*S+col]-decay*kv)*betaRef[h];double readout=0,l1=0;
    for(int row=0;row<S;++row){const size_t at=(size_t(row)*hv+h)*S+col;const double updated=decay*state[at]+activated[(hk+qh)*S+row]*delta;near(gpuState[at],updated,"GDN persistent delta state");state[at]=float(updated);const double term=double(state[at])*activated[qh*S+row];readout+=term;l1+=std::abs(term);}
    expected_output[h*S+col]=float(readout/std::sqrt(double(S)));near(output[h*S+col],expected_output[h*S+col],"GDN readout scaling/head map",l1/std::sqrt(double(S)));
   }
   for(int h=0;h<hv;++h){double ss=0;for(int i=0;i<S;++i)ss+=double(expected_output[h*S+i])*expected_output[h*S+i];const double norm=1/std::sqrt(ss/S+1e-6f);for(int i=0;i<S;++i)near(closed[h*S+i],norm*expected_output[h*S+i]*gamma[i]*sigmoid(x[(2*hk+h)*S+i]),"GDN closing RMS/gamma/gate");}
   for(size_t i=0;i<history.size();++i){require(hist[i]==history[i],"GDN history chronology");++bitwise;}
   tail(raw,channels);tail(silu,channels);tail(bg,hv);tail(gg,hv);tail(gpuState,state.size());tail(output,size_t(hv)*S);tail(closed,size_t(hv)*S);tail(hist,history.size());++cases;
  }
  require(dw.get<float>(s)==weights && dgamma.get<float>(s)==gamma && ddt.get<float>(s)==dt && da.get<float>(s)==a,"GDN changed read-only inputs");
 }
}
