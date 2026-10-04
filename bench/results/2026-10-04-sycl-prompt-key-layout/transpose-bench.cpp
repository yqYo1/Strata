#define main attention_test_main
#include "../../../tests/sycl/decode_attention.cpp"
#undef main
#include <chrono>
#include <iomanip>
int main(int argc, char**argv) {
 try {
  setenv("STRATA_SYCL_PROMPT_DIRECT","1",1);
  runtime=sycl_backend::runtime_for();
  const int nq=argc>1?std::atoi(argv[1]):512, cap=4096, hd=256, heads=24;
  auto s=qsa_real_shapes();s.page_size=16;
  std::vector<uint16_t> keys(cap*2*hd),values(keys.size());
  for(size_t i=0;i<keys.size();++i){keys[i]=half(float(std::sin(i*.073)));values[i]=half(float(std::cos(i*.017)));}
  std::vector<float> queries(size_t(nq)*heads*hd);
  for(size_t i=0;i<queries.size();++i)queries[i]=float(std::sin(i*.019));
  std::vector<int32_t> table(cap/16),ids(size_t(nq)*cap),steps(nq*kStepCount);
  for(int i=0;i<cap/16;++i)table[i]=cap/16-1-i;
  for(int b=0;b<nq;++b){steps[b*kStepCount+kStepWidth]=(b+1)*cap/nq;for(int c=0;c<cap;++c)ids[size_t(b)*cap+c]=c;}
  Buffer<uint16_t> k(keys.size()),v(values.size());Buffer<float> q(queries.size()),out(queries.size()+16);Buffer<int32_t> t(table.size()),i(ids.size()),st(steps.size());
  k.put(keys);v.put(values);q.put(queries);t.put(table);i.put(ids);st.put(steps);out.put(std::vector<float>(queries.size()+16,1234.f));
  QsaAttnPools p;p.page_table=t.data();p.k_pool=k.data();p.v_pool=v.data();
  auto invoke=[&]{check(qsa_prompt_attn_batch(q.data(),p,i.data(),st.data(),cap,s,out.data(),nq,&runtime->compute()),"matrix unavailable");runtime->wait();};
  std::vector<float> baseline;
  for(int direct: {0,1,2,3,4,4,3,2,1,0,0,1,2,3,4}) {
   setenv("STRATA_SYCL_PROMPT_TRANSPOSE_KEY",std::to_string(direct).c_str(),1);invoke();
   auto y=out.get();
   if(baseline.empty())baseline=y;
   check(std::memcmp(y.data(),baseline.data(),y.size()*sizeof(float))==0,"long direct/staged bitwise");
   for(size_t j=queries.size();j<y.size();++j)check(y[j]==1234.f,"long output guard");
   for(int rep=0;rep<3;++rep){auto start=std::chrono::steady_clock::now();invoke();auto end=std::chrono::steady_clock::now();std::cout<<"nq="<<nq<<" transpose_key="<<direct<<" ms="<<std::setprecision(9)<<std::chrono::duration<double,std::milli>(end-start).count()<<"\n"<<std::flush;}
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
