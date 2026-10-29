// Authored for Strata;repository LICENSE. Lossless numerical evidence only.
#include "qsa-head.hpp"
#include "strata/sycl/launch.hpp"
#include <sstream>

#define TRACE_ARGS const uint16_t *query, const QsaAttnPools &pools, \
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape, \
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds, \
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream
#define TRACE_VALUES query,pools,ids,steps,cap,shape,output,queries,bounds,live_cells,scratch,stream
namespace strata::kernels { bool traced_batch(TRACE_ARGS); }
#define qsa_bf16_prefix_batch traced_batch
#define main qsa_existing_tensor_oracle
#include "../../tests/sycl/qsa_bf16_prefix.cpp"
#undef main
#undef qsa_bf16_prefix_batch

namespace {
bool wide=false;
uint64_t calls=0,records=0;
constexpr uint32_t TraceMark=0xdeadbeef;
constexpr float StateMark=-913.75f;
template<class T> void array(std::ostream &out,const std::vector<T> &values,size_t first,size_t count) {
  require(first<=values.size() && count<=values.size()-first,"trace array bounds");
  out<<'[';
  for(size_t i=0;i<count;++i) {if(i)out<<',';out<<uint64_t(values[first+i]);}
  out<<']';
}
bool collect(TRACE_ARGS) {
  require(queries>=0 && queries<=3 && cap<=2051,"bounded numerical fixture only");
  const bool ready=strata::kernels::qsa_head_candidate::batch(TRACE_VALUES);
  if(!ready || !queries)return ready;
  rt->wait();
  const uint64_t call=calls++;
  const size_t elements=sycl_backend::checked_count(queries,24*256);
  const size_t tiles=(uint64_t(cap)+31)/32,words=tiles*64+512;
  std::vector<uint16_t> candidate(elements),original;
  rt->wait(rt->compute().memcpy(candidate.data(),output,elements*2));
  std::vector<int32_t> host_steps(size_t(queries)*kStepCount);
  rt->wait(rt->compute().memcpy(host_steps.data(),steps,host_steps.size()*4));
  Buffer<uint16_t> temporary(elements+32);
  Buffer<float> state(tiles*4+32);
  Buffer<uint32_t> details(words+32);
  Drain pending; // Drain before every DMA/device owner on failure.
  const uint16_t output_mark=round_bf16(1234.f);
  const std::vector<uint16_t> initial_output(elements+32,output_mark);
  const std::vector<float> initial_state(tiles*4+32,StateMark);
  const std::vector<uint32_t> initial_details(words+32,TraceMark);
  temporary.put(initial_output);
  require(strata::kernels::qsa_head_original::batch(query,pools,ids,steps,cap,shape,
      temporary.data()+16,queries,bounds,live_cells,scratch,stream),"serial calibration unavailable");
  auto original_guarded=temporary.get();
  for(int i=0;i<16;++i)require(original_guarded[i]==output_mark &&
      original_guarded[original_guarded.size()-16+i]==output_mark,"serial calibration guards");
  original.assign(original_guarded.begin()+16,original_guarded.end()-16);
  std::vector<int> heads=wide?std::vector<int>{0,1,2,3,4,5,6,7,8,9,10,11,15}:std::vector<int>{0,15};
  std::vector<int64_t> selected_queries{0};
  if(wide && queries>1)selected_queries.push_back(queries-1);
  for(auto selected:selected_queries)for(int kv:{0,1})for(int head:heads)for(bool parallel:{false,true}) {
    state.put(initial_state);details.put(initial_details);temporary.put(initial_output);
    const QsaPrefixStateProbe probe{selected,kv,head,state.data()+16};
    const bool ok=parallel?strata::kernels::qsa_head_candidate::capture(query,pools,ids,steps,cap,shape,
        temporary.data()+16,queries,bounds,live_cells,scratch,stream,probe,nullptr,nullptr,details.data()+16):
      strata::kernels::qsa_head_original::capture(query,pools,ids,steps,cap,shape,
        temporary.data()+16,queries,bounds,live_cells,scratch,stream,probe,nullptr,nullptr,details.data()+16);
    require(ok,"lossless capture unavailable");
    const auto y=temporary.get();const auto s=state.get();const auto d=details.get();
    const auto &normal=parallel?candidate:original;
    require(std::equal(normal.begin(),normal.end(),y.begin()+16),"capture changes matching normal output");
    for(int i=0;i<16;++i) {
      require(y[i]==output_mark && y[y.size()-16+i]==output_mark,"trace output guards");
      require(s[i]==StateMark && s[s.size()-16+i]==StateMark,"trace state guards");
      require(d[i]==TraceMark && d[d.size()-16+i]==TraceMark,"trace detail guards");
    }
    const int width=host_steps[size_t(selected)*kStepCount+kStepWidth];
    const size_t used=(width>=0 && width<=cap)?size_t((width+31)/32):0;
    for(size_t i=used*4;i<tiles*4;++i)require(s[16+i]==StateMark,"state past live tile");
    for(size_t i=used*64;i<tiles*64;++i)require(d[16+i]==TraceMark,"probability past live tile");
    std::vector<uint32_t> state_bits(tiles*4);
    for(size_t i=0;i<state_bits.size();++i)state_bits[i]=std::bit_cast<uint32_t>(s[16+i]);
    std::ostringstream record;
    record<<"QSA_NORM_TRACE {\"call\":"<<call<<",\"arm\":\""<<(parallel?"candidate":"original")
      <<"\",\"cap\":"<<cap<<",\"width\":"<<width<<",\"tiles\":"<<used
      <<",\"queries\":"<<queries<<",\"query\":"<<selected<<",\"kv_head\":"<<kv<<",\"head\":"<<head
      <<",\"live_cells\":"<<live_cells<<",\"page\":"<<shape.page_size
      <<",\"normal_parity\":true,\"guards\":true,\"state\":";
    array(record,state_bits,0,used*4);record<<",\"details\":";array(record,d,16,words);
    record<<",\"output\":";
    array(record,y,head<12?16+(size_t(selected)*24+kv*12+head)*256:0,head<12?256:0);
    record<<'}';std::cout<<record.str()<<'\n';++records;
  }
  return true;
}
}
namespace strata::kernels { bool traced_batch(TRACE_ARGS) {return collect(TRACE_VALUES);} }
#undef TRACE_ARGS
#undef TRACE_VALUES

int main(int argc,char **argv) {
  try {
    require(argc==2,"--tiny or --component required");
    const std::string mode=argv[1];require(mode=="--tiny" || mode=="--component","explicit trace mode");
    wide=mode=="--component";std::cout.setf(std::ios::unitbuf);
    gate(true,"before-runtime");rt=sycl_backend::runtime_for();Drain pending;
    require(rt->device().get_info<sycl::info::device::name>()=="Intel(R) Arc(TM) B570 Graphics","qualified B570 required");
    std::cout<<"QSA_FP_CONFIG";
    for(auto value:rt->device().get_info<sycl::info::device::single_fp_config>())std::cout<<' '<<int(value);
    std::cout<<'\n';gate(true,"runtime-ready");
    if(!wide)run(17,17,1,false,false);
    else {
      run(97,33,1,false,true,49);run(97,33,3,true,false,71);
      for(int width:{1,17,31,32,63,65,2051})run(2113,width,3,true,false);
      run(262144,2051,1,false,false);run(262144,33,1,false,false,13);
    }
    rt->wait();gate(true,"after-cleanup");
    std::cout<<"QSA_NORM_DONE calls="<<calls<<" records="<<records<<" tensor_oracle=PASS model_quality=UNRUN\n";
    return 0;
  } catch(const std::exception &error) {drain();std::cerr<<"QSA_NORM_ERROR "<<error.what()<<'\n';return 1;}
}
