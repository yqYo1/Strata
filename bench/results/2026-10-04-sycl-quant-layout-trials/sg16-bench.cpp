#define main prefill_mmq_test_main
#include "../../../tests/sycl/prefill_mmq.cpp"
#undef main
#include <chrono>
#include <iomanip>
int main() {
 try {
  runtime=sycl_backend::runtime_for();
  for (int cols : {1280,2560}) for(int rows : {1,512,4007,40070}) for(bool mapped : {false,true}) {
   const int source_rows=mapped?std::min(rows,4007):rows,ld=cols+5;
   std::vector<float> x(size_t(source_rows)*ld);
   for(size_t i=0;i<x.size();++i)x[i]=float(int((i*37+i/251)%8191)-4095)/4096.f;
   for(int r=0;r<source_rows;++r)std::fill_n(x.begin()+size_t(r)*ld,32,0.f);
   std::vector<int32_t> ids(rows);
   for(int r=0;r<rows;++r)ids[r]=(r%17==0)?-1:(r*13)%source_rows;
   Buffer<float> dx(x.size());Buffer<int32_t> di(ids.size());
   Buffer<uint8_t> dst(mmq::q8_bytes(rows,cols)+32);
   dx.put(x);di.put(ids);dst.put(std::vector<uint8_t>(dst.n,0xa5));
   auto invoke=[&]{mmq::quantize(dx.data(),mapped?di.data():nullptr,dst.data(),21,cols,ld,rows,&runtime->compute());runtime->wait();};
   std::vector<uint8_t> baseline;
   for(int mode:{0,1,2,2,1,0,0,1,2}) {
    setenv("STRATA_SYCL_PROMPT_QUANT_2D",mode==2?"2":mode?"1":"0",1);invoke();
    const auto result=dst.get();
    if(baseline.empty())baseline=result;
    check(result==baseline,"quantize layout output mismatch");
    check(std::all_of(result.end()-32,result.end(),[](uint8_t b){return b==0xa5;}),"quantize layout guard");
    for(int rep=0;rep<3;++rep) {
     auto start=std::chrono::steady_clock::now();invoke();auto end=std::chrono::steady_clock::now();
     std::cout<<"rows="<<rows<<" cols="<<cols<<" mapped="<<mapped<<" grid="<<mode<<" ms="<<std::setprecision(9)<<std::chrono::duration<double,std::milli>(end-start).count()<<"\n"<<std::flush;
    }
   }
  }
  std::cout<<"ALL QUANT LAYOUT BYTES/GUARDS PASS\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
