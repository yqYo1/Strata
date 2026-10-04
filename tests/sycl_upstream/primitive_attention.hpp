#pragma once
std::pair<double,double> rotation(const k::RopeScaling& scaling,int pos,int pair){
 const auto a=scaling.kernel_args(64);const float base=std::pow(float(scaling.freq_base),-2.f/64);
 // The original analytic path materializes its power and angle as F32
 // before trigonometry. The CPU oracle retains those operand boundaries,
 // then evaluates cosine/sine independently in FP64.
 const float power=std::pow(base,float(pair)),extrap=float(pos)*power;
 float theta=a.freq_scale*extrap,m=a.attn_factor;
 if(a.ext_factor!=0){const float ramp=1.f-std::clamp((float(pair)-a.corr_low)/std::max(.001f,a.corr_high-a.corr_low),0.f,1.f),mix=ramp*a.ext_factor;theta=std::fma(theta,1.f-mix,extrap*mix);m*=1.f+.1f*std::log(1.f/a.freq_scale);}
 return {std::cos(double(theta))*m,std::sin(double(theta))*m};
}
std::vector<double> rotated(const std::vector<float>& x,int width,const std::vector<int32_t>& positions,const k::RopeScaling& scaling,const std::vector<int32_t>& mtab,const std::vector<float>& cos,const std::vector<float>& sin,int maxpos){
 std::vector<double> y(x.begin(),x.end());for(size_t row=0;row<positions.size();++row)for(int pair=0;pair<32;++pair){const int p=mtab.empty()?positions[row]:mtab[positions[row]*3+pair%3];auto [c,s]=rotation(scaling,p,pair);if(!cos.empty()&&p<maxpos){c=cos[p*32+pair];s=sin[p*32+pair];}const double a=x[row*width+pair],b=x[row*width+pair+32];y[row*width+pair]=a*c-b*s;y[row*width+pair+32]=a*s+b*c;}return y;
}
void rope_tests(cudaStream_t s,cudaStream_t other){
 for(int mode=0;mode<3;++mode)for(int width:{128,256})for(bool multimodal:{false,true})for(bool table:{false,true})for(bool alias:{false,true}){
  context="RoPE mode="+std::to_string(mode)+" width="+std::to_string(width)+" m="+std::to_string(multimodal)+" table="+std::to_string(table)+" alias="+std::to_string(alias);
  constexpr int rows=5,maxpos=64,cells=96;k::RopeScaling scaling;scaling.freq_base=10000;scaling.type=mode==0?k::RopeScalingType::None:mode==1?k::RopeScalingType::Linear:k::RopeScalingType::YaRN;scaling.factor=4;scaling.orig_ctx=1024;scaling.ext_factor=mode==2?1:0;
  auto x=random_values(rows*width,2);std::vector<int32_t> pos{0,1,3,31,67},mtab(cells*3),newtab(cells*3);for(int p=0;p<cells;++p)for(int sector=0;sector<3;++sector){mtab[p*3+sector]=(p+sector*7)%61;newtab[p*3+sector]=(p+17+sector*5)%61;}
  std::vector<float> cos(maxpos*32),sin(cos.size());for(int p=0;p<maxpos;++p)for(int i=0;i<32;++i){auto [c,z]=rotation(scaling,p,i);cos[p*32+i]=float(c);sin[p*32+i]=float(z);}
  Gpu dx((x.size()+16)*4),dy(dx.bytes),dp(pos.size()*4),dm(mtab.size()*4),dn(newtab.size()*4),dc(cos.size()*4),ds(sin.size()*4);auto init=guarded(x.size());auto xin=init;std::copy(x.begin(),x.end(),xin.begin());dx.put(xin,s);dy.put(init,s);dp.put(pos,s);dm.put(mtab,s);dn.put(newtab,s);dc.put(cos,s);ds.put(sin,s);
  k::mrope_table_set(multimodal?dm.as<int32_t>():nullptr);if(table)k::rope_table_set(dc.as<float>(),ds.as<float>(),maxpos,scaling);
  auto launch=[&]{k::native_rope_apply(dx.as<float>(),alias?dx.as<float>():dy.as<float>(),rows,width,64,scaling,dp.as<int>(),s);};launch();auto out=alias?dx.get<float>(s):dy.get<float>(s);auto expected=rotated(x,width,pos,scaling,multimodal?mtab:std::vector<int32_t>{},table?cos:std::vector<float>{},sin,maxpos);for(size_t i=0;i<x.size();++i)near(out[i],expected[i],"RoPE independent NEOX/YaRN/IMRoPE reference");tail(out,x.size());++cases;
  if(mode==0&&width==256&&multimodal&&table&&!alias){
   dy.put(init,s);ck(cudaStreamSynchronize(s));Graph graph(s,launch,1);require(dy.get<float>(other)==init,"RoPE instantiation executed rotation");
   // A captured launch retains the old registry pointer. Its pointee contents
   // and device positions remain live, while later registrations change.
   k::mrope_table_set(dn.as<int32_t>());
   for(int replay=0;replay<3;++replay){x=random_values(rows*width,2);std::copy(x.begin(),x.end(),xin.begin());for(int p=0;p<cells;++p)for(int sector=0;sector<3;++sector)mtab[p*3+sector]=(p+replay*13+sector*7)%91;dx.put(xin,s);dm.put(mtab,s);std::rotate(pos.begin(),pos.begin()+1,pos.end());dp.put(pos,s);dy.put(init,s);graph.run(s);auto actual=dy.get<float>(s);auto ref=rotated(x,width,pos,scaling,mtab,cos,sin,maxpos);for(size_t i=0;i<x.size();++i)near(actual[i],ref[i],"captured old mrope pointer/live contents");tail(actual,x.size());++cases;}
  }
  k::mrope_table_set(nullptr);if(table)k::rope_table_release(dc.as<float>());
 }
}
void indexer_tests(cudaStream_t s,cudaStream_t other){
 constexpr int D=128,R=4,capacity=48,blocks=capacity/R+1,base=16;
 for(int mode=0;mode<3;++mode)for(bool table:{false,true})for(bool multimodal:{false,true}){
  const std::string label="indexer mode="+std::to_string(mode)+" table="+std::to_string(table)+" mrope="+std::to_string(multimodal);
  context=label;auto raw=random_values(capacity*D,3),gamma=random_values(D);for(auto& v:gamma)v+=1;
  auto query=random_values(4*D,2),bias=random_values(blocks);k::RopeScaling scaling;scaling.freq_base=10000;scaling.type=mode==0?k::RopeScalingType::None:mode==1?k::RopeScalingType::Linear:k::RopeScalingType::YaRN;scaling.factor=4;scaling.orig_ctx=1024;scaling.ext_factor=mode==2?1:0;const auto shape=k::qsa_real_shapes();
  constexpr int cells=96,maxpos=32;std::vector<int32_t> mtab(cells*3);for(int p=0;p<cells;++p)for(int sector=0;sector<3;++sector)mtab[p*3+sector]=(p+sector*7)%91;
  std::vector<float> cos(maxpos*32),sin(cos.size());for(int p=0;p<maxpos;++p)for(int i=0;i<32;++i){auto [c,z]=rotation(scaling,p,i);cos[p*32+i]=float(c);sin[p*32+i]=float(z);}
  Gpu dm(mtab.size()*4),dc(cos.size()*4),dsn(sin.size()*4);dm.put(mtab,s);dc.put(cos,s);dsn.put(sin,s);k::mrope_table_set(multimodal?dm.as<int32_t>():nullptr);if(table)k::rope_table_set(dc.as<float>(),dsn.as<float>(),maxpos,scaling);
  Gpu dr(raw.size()*4),dg(gamma.size()*4),dq(query.size()*4),db(bias.size()*4),dt((3*D+16)*4),dd((D+16)*4),dp((blocks*D+16)*4),dbp(17*4),dstep(4*4),dscores((capacity+16)*4);
  dr.put(raw,s);dg.put(gamma,s);dq.put(query,s);db.put(bias,s);auto ti=guarded(3*D),di=guarded(D),pi=guarded(blocks*D),scoreinit=guarded(capacity);std::fill_n(ti.begin(),3*D,0.f);std::fill_n(di.begin(),D,0.f);std::fill_n(pi.begin(),blocks*D,0.f);std::vector<int32_t> bi(17,-777);bi[0]=-1;dt.put(ti,s);dd.put(di,s);dp.put(pi,s);dbp.put(bi,s);dscores.put(scoreinit,s);ck(cudaStreamSynchronize(s));const k::QsaIndexerBuffers b{dt.as<float>(),dd.as<float>(),dp.as<float>(),dbp.as<int32_t>()};
  // Keep one stable raw-key buffer on graph replay, and update its values.
  Gpu live(D*4);Graph graph(s,[&]{k::native_qsa_indexer_append(live.as<float>(),dstep.as<int32_t>()+k::kStepPos,base,dg.as<float>(),1e-6f,b,shape,capacity,scaling,s);k::native_qsa_score(dp.as<float>(),dq.as<float>(),db.as<float>(),shape,dstep.as<int32_t>(),blocks,capacity,dscores.as<float>(),s);},2);
  require(dp.get<float>(other)==pi,"indexer capture modified pooled state");std::vector<float> cpuTail(3*D),cpuDead(D),cpuPool(blocks*D);int blockpos=-1;
  for(int pos=0;pos<capacity;++pos){
   context=label+" pos="+std::to_string(pos);std::vector<float> incoming(raw.begin()+pos*D,raw.begin()+(pos+1)*D);live.put(incoming,s);dstep.put(std::vector<int32_t>{pos,pos+1,(pos+1)/R,pos+1},s);dscores.put(scoreinit,s);graph.run(s);
   const int slot=pos%R;std::vector<float> rounded(D);for(int i=0;i<D;++i)rounded[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(incoming[i]));if(slot<3)std::copy(rounded.begin(),rounded.end(),cpuTail.begin()+slot*D);
   if(pos==0||slot==3){std::vector<float> mean(D);for(int i=0;i<D;++i){float sum=pos==0?rounded[i]:cpuTail[i];for(int tap=1;tap<4;++tap)sum+=pos==0||tap==3?rounded[i]:cpuTail[tap*D+i];mean[i]=sum*.25f;}auto normalized=rms(mean,gamma,D,1,1e-6f,true);std::vector<float> fp(D);for(int i=0;i<D;++i)fp[i]=float(normalized[i]);auto rotated_values=rotated(fp,D,std::vector<int32_t>{pos==0?0:base+R*(pos/R)},scaling,multimodal&&pos!=0?mtab:std::vector<int32_t>{},table?cos:std::vector<float>{},sin,maxpos);const int row=pos/R;for(int i=0;i<D;++i)cpuPool[row*D+i]=float(rotated_values[i]);if(pos==0)std::copy(cpuPool.begin(),cpuPool.begin()+D,cpuDead.begin());else{std::copy(cpuDead.begin(),cpuDead.end(),cpuPool.begin()+(row+1)*D);blockpos=base+R*row;}}
   auto tailv=dt.get<float>(s),dead=dd.get<float>(s),pool=dp.get<float>(s),scores=dscores.get<float>(s);auto positions=dbp.get<int32_t>(s);for(int i=0;i<3*D;++i){require(tailv[i]==cpuTail[i],"indexer half-rounded tail state");++bitwise;}for(int i=0;i<D;++i)near(dead[i],cpuDead[i],"indexer cell-zero spare");for(int i=0;i<blocks*D;++i){context=label+" pos="+std::to_string(pos)+" row="+std::to_string(i/D)+" d="+std::to_string(i%D);near(pool[i],cpuPool[i],"indexer chronological pooled state");}require(positions[0]==blockpos,"indexer completion position");++bitwise;
   // Use the actual stored pooled key as the independent score input, so
   // score error is distinguished from a key-pooling error.
   for(int row=0;row<=(pos+1)/R;++row){double sum=0;for(int h=0;h<4;++h){double dot=0;for(int i=0;i<D;++i)dot+=double(pool[row*D+i])*query[h*D+i];sum+=std::max(0.,dot);}sum+=bias[row];if(row==(pos+1)/R&&(pos+1)%R)sum=double(float(sum+1e9));for(int cell=row*R;cell<std::min(pos+1,(row+1)*R);++cell){near(scores[cell],sum,"original portable QSA score/head ReLU/tail bias");if(row==(pos+1)/R&&(pos+1)%R){require(scores[cell]==float(sum),"indexer unfinished-tail bias exact stored float");++bitwise;}}}
   for(int cell=pos+1;cell<capacity;++cell){require(scores[cell]==guard,"indexer score padded cell");++guards;}tail(tailv,3*D);tail(dead,D);tail(pool,blocks*D);tail(scores,capacity);for(size_t i=1;i<positions.size();++i){require(positions[i]==-777,"indexer position guard");++guards;}++cases;
  }
  const auto finalTail=dt.get<float>(s),finalDead=dd.get<float>(s),finalPool=dp.get<float>(s);const auto finalPos=dbp.get<int32_t>(s);
  // All batch helpers, nonaligned chunk starts/ends and carried tail state.
  dt.put(ti,s);dd.put(di,s);dp.put(pi,s);dbp.put(bi,s);int p0=0;for(int chunk:{1,2,5,3,7,10,20}){k::native_qsa_indexer_append_batch(dr.as<float>()+p0*D,chunk,p0,base,dg.as<float>(),1e-6f,b,shape,capacity,scaling,s);p0+=chunk;++cases;}require(p0==capacity,"indexer chunk fixture extent");same(dt.get<float>(s),finalTail,"indexer batch/single tail bits");same(dd.get<float>(s),finalDead,"indexer batch/single spare bits");same(dp.get<float>(s),finalPool,"indexer batch/single pooled bits");require(dbp.get<int32_t>(s)==finalPos,"indexer batch/single completion metadata");
  dstep.put(std::vector<int32_t>{capacity,0,0,0},s);dscores.put(scoreinit,s);graph.run(s);require(dscores.get<float>(s)==scoreinit,"invalid score step wrote output");same(dp.get<float>(s),finalPool,"out-of-range append modified pool");++cases;
  k::mrope_table_set(nullptr);if(table)k::rope_table_release(dc.as<float>());
 }
}
void flash_tests(cudaStream_t s,cudaStream_t other){
 constexpr int capacity=256,D=256,H=24,KV=2;const auto shape=k::qsa_real_shapes();auto q=random_values(H*D,2),kf=random_values(capacity*KV*D,3),vf=random_values(kf.size(),2);std::vector<uint16_t> kh(kf.size()),vh(vf.size()),mask(capacity);for(size_t i=0;i<kh.size();++i){kh[i]=ggml_fp32_to_fp16(kf[i]);vh[i]=ggml_fp32_to_fp16(vf[i]);}
 Gpu dq(q.size()*4),dk(kh.size()*2),dv(vh.size()*2),dm(mask.size()*2),dstep(4*4),dy((q.size()+16)*4),dstatus(17*4);auto init=guarded(q.size());std::vector<int32_t> si(17,-777);
 auto verify=[&](int width,bool masked){auto out=dy.get<float>(s);auto status=dstatus.get<int32_t>(s);require(status[0]==k::kNativeFlashAttnSuccess,"flash status valid step");for(size_t i=1;i<status.size();++i){require(status[i]==-777,"flash status guard");++guards;}
  for(int h=0;h<H;++h){const int kv=h/12;std::vector<double> scores(width),prob(width);double maximum=-INFINITY,total=0;for(int cell=0;cell<width;++cell){double dot=0;for(int d=0;d<D;++d)dot+=double(q[h*D+d])*ggml_fp16_to_fp32(kh[(cell*KV+kv)*D+d]);scores[cell]=dot/16+(masked?ggml_fp16_to_fp32(mask[cell]):0);maximum=std::max(maximum,scores[cell]);}for(int cell=0;cell<width;++cell)total+=(prob[cell]=std::exp(scores[cell]-maximum));for(int d=0;d<D;++d){double sum=0,l1=0;for(int cell=0;cell<width;++cell){const double term=prob[cell]/total*ggml_fp16_to_fp32(vh[(cell*KV+kv)*D+d]);sum+=term;l1+=std::abs(term);}near(out[h*D+d],sum,"native vector attention FP64 GQA/softmax/mask",l1);}}
  tail(out,q.size());++cases;
 };
 for(bool masked:{false,true})for(int width:{1,3,4,127,128,129,255,256}){
  context="flash width="+std::to_string(width)+" masked="+std::to_string(masked);auto keys=kh,values=vh;for(size_t i=size_t(width)*KV*D;i<keys.size();++i){keys[i]=values[i]=0x7e00;}for(int i=0;i<capacity;++i)mask[i]=i>=width?0x7e00:ggml_fp32_to_fp16(i%7==6?-INFINITY:float(i%3)/16);
  dq.put(q,s);dk.put(keys,s);dv.put(values,s);dm.put(mask,s);dstep.put(std::vector<int32_t>{width-1,width,width/4,width},s);dy.put(init,s);dstatus.put(si,s);k::native_flash_attn_short_step(dq.as<float>(),dk.as<uint16_t>(),dv.as<uint16_t>(),dstep.as<int32_t>(),capacity,capacity,shape,dy.as<float>(),dstatus.as<int32_t>(),masked?dm.as<uint16_t>():nullptr,s);verify(width,masked);
 }
 context="flash changed-step graph";dk.put(kh,s);dv.put(vh,s);dy.put(init,s);dstatus.put(si,s);ck(cudaStreamSynchronize(s));Graph graph(s,[&]{k::native_flash_attn_short_step(dq.as<float>(),dk.as<uint16_t>(),dv.as<uint16_t>(),dstep.as<int32_t>(),capacity,capacity,shape,dy.as<float>(),dstatus.as<int32_t>(),nullptr,s);},1);require(dy.get<float>(other)==init,"flash graph executed at instantiation");
 for(int width:{3,128,256}){q=random_values(H*D,2);dq.put(q,s);dstep.put(std::vector<int32_t>{width-1,width,width/4,width},s);dy.put(init,s);graph.run(s);verify(width,false);}
 dq.put(std::vector<float>(q.size(),NAN),s);dstep.put(std::vector<int32_t>{256,257,64,257},s);dy.put(init,s);graph.run(s);auto bad=dy.get<float>(s);require(dstatus.get<int32_t>(s)[0]==k::kNativeFlashAttnUnsupportedStep,"flash invalid step status");for(size_t i=0;i<q.size();++i){require(std::isnan(bad[i]),"flash invalid step poison");++guards;}tail(bad,q.size());++cases;
}
