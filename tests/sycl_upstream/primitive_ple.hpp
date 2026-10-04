#pragma once
// Fixed original model geometry, including channel-major chronological history.
constexpr int pleN=2560,pleH=4,pleD=pleN*pleH,pleHistory=9;
struct PleResult {std::vector<float> key,query,gate,gated,normalized,conv,result;};
struct PleSingle {
 Gpu projected{(pleD+16)*4},hidden{(pleD+16)*4},value{(pleN+16)*4},history{(pleD*pleHistory+16)*4};
 Gpu key{(pleD+16)*4},query{(pleD+16)*4},gate{(pleH+16)*4},gated{(pleD+16)*4},normalized{(pleD+16)*4},conv{(pleD+16)*4},result{(pleD+16)*4};
 bool queryAlias,resultAlias;k::PleWeights weights;
 PleSingle(bool qa,bool ra,k::PleWeights w):queryAlias(qa),resultAlias(ra),weights(w){}
 void upload(const std::vector<float>& p,const std::vector<float>& h,const std::vector<float>& v,const std::vector<float>& hist,cudaStream_t s){
  auto extend=[](const std::vector<float>& x){auto y=guarded(x.size());std::copy(x.begin(),x.end(),y.begin());return y;};
  // Fixture uploads complete before temporary CPU vectors are released.
  projected.put(extend(p),s);hidden.put(extend(h),s);value.put(extend(v),s);history.put(extend(hist),s);
  key.put(guarded(pleD),s);query.put(guarded(pleD),s);gate.put(guarded(pleH),s);gated.put(guarded(pleD),s);normalized.put(guarded(pleD),s);conv.put(guarded(pleD),s);result.put(guarded(pleD),s);
 }
 k::NativePlePostopsBuffers buffers(){return {key.as<float>(),query.as<float>(),gate.as<float>(),gated.as<float>(),queryAlias?query.as<float>():normalized.as<float>(),conv.as<float>(),resultAlias?hidden.as<float>():result.as<float>()};}
 void launch(cudaStream_t s){k::native_ple_postops(projected.as<float>(),hidden.as<float>(),value.as<float>(),history.as<float>(),weights,buffers(),s);}
 PleResult read(cudaStream_t s){return {key.get<float>(s),query.get<float>(s),gate.get<float>(s),gated.get<float>(s),queryAlias?query.get<float>(s):normalized.get<float>(s),conv.get<float>(s),resultAlias?hidden.get<float>(s):result.get<float>(s)};}
};
void ple_reference(const PleResult& out,const std::vector<float>& p,const std::vector<float>& h,const std::vector<float>& v,const std::vector<float>& hist,const std::vector<float>& nk,const std::vector<float>& nq,const std::vector<float>& nc,const std::vector<uint16_t>& taps,bool queryAlias){
 const auto kr=rms(p,nk,pleN,pleH,1e-6f,false),qr=rms(h,nq,pleN,pleH,1e-6f,false);
 for(int d=0;d<pleD;++d){near(out.key[d],kr[d],"PLE projected-key weighted norm");if(!queryAlias)near(out.query[d],qr[d],"PLE hidden weighted norm");}
 for(int c=0;c<pleH;++c){double dot=0;for(int d=0;d<pleN;++d)dot+=kr[c*pleN+d]*qr[c*pleN+d];const double scaled=dot/std::sqrt(double(pleN));const double signedRoot=(scaled>0?1:scaled<0?-1:0)*std::sqrt(std::max(std::abs(scaled),1e-6));near(out.gate[c],sigmoid(signedRoot),"PLE signed-root sigmoid gate");}
 std::vector<float> storedGated(pleD);for(int d=0;d<pleD;++d){near(out.gated[d],double(v[d%pleN])*out.gate[d/pleN],"PLE value broadcast and materialized product");storedGated[d]=out.gated[d];}
 const auto nr=rms(storedGated,nc,pleN,pleH,1e-6f,false);for(int d=0;d<pleD;++d){near(out.normalized[d],nr[d],"PLE gated weighted norm");double sum=0,l1=0;for(int tap=0;tap<4;++tap){const double x=tap==3?out.normalized[d]:hist[d*pleHistory+tap*3];const double term=x*ggml_fp16_to_fp32(taps[d*4+tap]);sum+=term;l1+=std::abs(term);}const double activation=sum*sigmoid(sum);near(out.conv[d],activation,"PLE F16 dilated convolution and SiLU",l1);near(out.result[d],double(h[d])+out.gated[d]+out.conv[d],"PLE hidden residual",std::abs(double(h[d]))+std::abs(double(out.gated[d]))+std::abs(double(out.conv[d])));}
 tail(out.key,pleD);tail(out.query,pleD);tail(out.gate,pleH);tail(out.gated,pleD);tail(out.normalized,pleD);tail(out.conv,pleD);tail(out.result,pleD);
}
void ple_advance(std::vector<float>& hist,const std::vector<float>& row){for(int c=0;c<pleD;++c){for(int r=0;r<pleHistory-1;++r)hist[c*pleHistory+r]=hist[c*pleHistory+r+1];hist[c*pleHistory+pleHistory-1]=row[c];}}
void ple_tests(cudaStream_t s,cudaStream_t other){
 auto nk=random_values(pleD),nq=random_values(pleD),nc=random_values(pleD);for(auto* gamma:{&nk,&nq,&nc})for(auto& x:*gamma)x+=1;
 auto wf=random_values(pleD*4,3);std::vector<uint16_t> taps(wf.size());for(size_t i=0;i<taps.size();++i)taps[i]=ggml_fp32_to_fp16(wf[i]);
 Gpu dnk(nk.size()*4),dnq(nq.size()*4),dnc(nc.size()*4),dw(taps.size()*2);dnk.put(nk,s);dnq.put(nq,s);dnc.put(nc,s);dw.put(taps,s);k::PleWeights w;w.norm_key=dnk.as<float>();w.norm_query=dnq.as<float>();w.norm_conv=dnc.as<float>();w.conv1d_f16=dw.as<uint16_t>();
 for(bool queryAlias:{false,true})for(bool resultAlias:{false,true}){
  context="PLE single query-alias="+std::to_string(queryAlias)+" result-alias="+std::to_string(resultAlias);PleSingle one(queryAlias,resultAlias,w);auto p=random_values(pleD,3),h=random_values(pleD,2),v=random_values(pleN,3),hist=random_values(pleD*pleHistory,2);one.upload(p,h,v,hist,s);ck(cudaStreamSynchronize(s));Graph graph(s,[&]{one.launch(s);},6);
  require(one.key.get<float>(other)==guarded(pleD),"PLE graph executed at instantiation");
  for(int replay=0;replay<3;++replay){p=random_values(pleD,3);h=random_values(pleD,2);v=random_values(pleN,3);hist=random_values(pleD*pleHistory,2);one.upload(p,h,v,hist,s);graph.run(s);const auto out=one.read(s);ple_reference(out,p,h,v,hist,nk,nq,nc,taps,queryAlias);
   auto actualHistory=one.history.get<float>(s);for(size_t i=0;i<hist.size();++i){require(actualHistory[i]==hist[i],"single PLE advanced caller-owned history");++bitwise;}tail(actualHistory,hist.size());++cases;
  }
 }
 for(int T:{1,3,10}){
  context="PLE batch T="+std::to_string(T);const size_t size=size_t(T)*pleD;Gpu dk((size+16)*4),dh((size+16)*4),dv((size_t(T)*pleN+16)*4),dhist((pleD*pleHistory+16)*4),dquery((size+16)*4),dgated((size+16)*4),dgate((size_t(T)*pleH+16)*4);
  auto p=random_values(size,3),h=random_values(size,2),v=random_values(size_t(T)*pleN,3),hist=random_values(pleD*pleHistory,2);
  auto extend=[](const std::vector<float>& x){auto y=guarded(x.size());std::copy(x.begin(),x.end(),y.begin());return y;};
  auto upload=[&]{dk.put(extend(p),s);dh.put(extend(h),s);dv.put(extend(v),s);dhist.put(extend(hist),s);dquery.put(guarded(size),s);dgated.put(guarded(size),s);dgate.put(guarded(size_t(T)*pleH),s);};upload();ck(cudaStreamSynchronize(s));Graph graph(s,[&]{k::native_ple_postops_batch(dk.as<float>(),dh.as<float>(),dv.as<float>(),dhist.as<float>(),w,dquery.as<float>(),dgated.as<float>(),dgate.as<float>(),T,s);},7);
  require(dk.get<float>(other)==extend(p)&&dhist.get<float>(other)==extend(hist),"PLE batch graph executed at instantiation");PleSingle one(true,true,w);
  for(int replay=0;replay<3;++replay){p=random_values(size,3);h=random_values(size,2);v=random_values(size_t(T)*pleN,3);hist=random_values(pleD*pleHistory,2);upload();graph.run(s);auto bk=dk.get<float>(s),bh=dh.get<float>(s),bn=dquery.get<float>(s),bg=dgated.get<float>(s),bgt=dgate.get<float>(s),bHist=dhist.get<float>(s);auto seqHist=hist;
   for(int t=0;t<T;++t){context="PLE batch T="+std::to_string(T)+" replay="+std::to_string(replay)+" token="+std::to_string(t);std::vector<float> pt(p.begin()+t*pleD,p.begin()+(t+1)*pleD),ht(h.begin()+t*pleD,h.begin()+(t+1)*pleD),vt(v.begin()+t*pleN,v.begin()+(t+1)*pleN);one.upload(pt,ht,vt,seqHist,s);one.launch(s);auto out=one.read(s);ple_reference(out,pt,ht,vt,seqHist,nk,nq,nc,taps,true);
    auto compare=[&](const std::vector<float>& batch,const std::vector<float>& seq,int width,const char* stage){for(int d=0;d<width;++d){if(std::bit_cast<uint32_t>(batch[size_t(t)*width+d])!=std::bit_cast<uint32_t>(seq[d]))std::cerr<<stage<<" d="<<d<<std::hexfloat<<" batch="<<batch[size_t(t)*width+d]<<" seq="<<seq[d]<<std::defaultfloat<<'\n';require(std::bit_cast<uint32_t>(batch[size_t(t)*width+d])==std::bit_cast<uint32_t>(seq[d]),"PLE batch differs from sequential stored bits");++bitwise;}};compare(bk,out.key,pleD,"key");compare(bh,out.result,pleD,"result");compare(bn,out.normalized,pleD,"normalized");compare(bg,out.gated,pleD,"gated");compare(bgt,out.gate,pleH,"gate");ple_advance(seqHist,out.normalized);++cases;
   }
   for(size_t i=0;i<seqHist.size();++i){require(std::bit_cast<uint32_t>(bHist[i])==std::bit_cast<uint32_t>(seqHist[i]),"PLE batch chronological history differs from sequential");++bitwise;}tail(bk,size);tail(bh,size);tail(bn,size);tail(bg,size);tail(bgt,size_t(T)*pleH);tail(bHist,seqHist.size());++cases;
  }
 }
 same(dnk.get<float>(s),nk,"PLE key gamma modified");same(dnq.get<float>(s),nq,"PLE query gamma modified");same(dnc.get<float>(s),nc,"PLE conv gamma modified");require(dw.get<uint16_t>(s)==taps,"PLE F16 weights modified");bitwise+=taps.size();std::cout<<"PASS native PLE single aliases and batch chronology"<<std::endl;
}
