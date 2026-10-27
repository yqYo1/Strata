// Main-authored source-bound QSA diagnostic; repository LICENSE applies.
// Real selected IDs; synthetic finite Q/K/V. No engine route is changed.
// Own archived4f2fffe03 harness;pitch-only32 preserves the original recurrence.
#include "qsa-slm-pitch.hpp"
#include "qsa_slm_pitch_admission.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::kernels;
namespace {
constexpr int Dim=256,Heads=24,KV=2,Page=4,Guard=16,Cap=2051;
std::shared_ptr<sycl_backend::Runtime> rt;
void require(bool ok,const char* why) { if (!ok) throw std::runtime_error(why); }
struct Drain {
  ~Drain() noexcept { sycl_backend::checked_cleanup([] { if(rt) rt->wait(); },[] {},"QSA SLM pitch fixture"); }
};
template<class T> struct Buffer {
  size_t count;
  sycl_backend::Allocation allocation;
  // Host transfer storage is a member: retained through checked retirement.
  std::vector<T> host;
  static size_t bytes(size_t n) {
    require(n <= std::numeric_limits<size_t>::max()/sizeof(T)-2*Guard,"allocation size overflow");
    const size_t value=(n+2*Guard)*sizeof(T);
    require(value<=rt->device().get_info<sycl::info::device::max_mem_alloc_size>(),"device single-allocation limit");
    return value;
  }
  explicit Buffer(size_t n):count(n),allocation(rt,bytes(n),sycl_backend::MemoryKind::Device) {}
  T* data() { return allocation.as<T>()+Guard; }
  void put(const std::vector<T>& input) {
    require(input.size()==count,"upload extent");
    host.assign(count+2*Guard,T(73));std::copy(input.begin(),input.end(),host.begin()+Guard);
    Drain drain;rt->wait(rt->compute().memcpy(allocation.as<T>(),host.data(),host.size()*sizeof(T)));
  }
  std::vector<T> get() {
    host.resize(count+2*Guard);Drain drain;
    rt->wait(rt->compute().memcpy(host.data(),allocation.as<T>(),host.size()*sizeof(T)));
    for(int i=0;i<Guard;++i) require(host[i]==T(73) && host[count+Guard+i]==T(73),"device guard");
    return {host.begin()+Guard,host.end()-Guard};
  }
};
uint16_t bf16(float x) {
  uint32_t bits=std::bit_cast<uint32_t>(x);
  return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
}
uint16_t half(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
void gate(const char* stage) {
  std::cout<<"QSA_SLM_PITCH_GATE "<<stage<<'\n';require(std::cin.get()=='X',"Main gate token");
}
uint64_t service(const sycl::event& event,const char* stage,int replica) {
  const auto begin=event.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end=event.get_profiling_info<sycl::info::event_profiling::command_end>();
  if(end<=begin) std::cerr<<"event_clock_invalid stage="<<stage<<" replica="<<replica<<" start="<<begin<<" end="<<end<<'\n';
  require(end>begin,"attention event clock invalid");return end-begin;
}
struct Selected {
  std::array<int32_t,4> header;
  std::vector<int32_t> ids;
  explicit Selected(const char* file) {
    require(std::endian::native==std::endian::little,"capture integer ABI");
    std::ifstream stream(file,std::ios::binary|std::ios::ate);require(bool(stream),"open selected fixture");
    require(stream.tellg()==16+int64_t(6491)*Cap*4,"selected fixture extent");stream.seekg(0);
    stream.read(reinterpret_cast<char*>(header.data()),16);
    require(header==std::array<int32_t,4>{0,32768,6491,Cap},"source-bound capture header");
    ids.resize(size_t(6491)*Cap);stream.read(reinterpret_cast<char*>(ids.data()),ids.size()*4);
    require(bool(stream) && stream.peek()==std::char_traits<char>::eof(),"selected fixture payload");
    for(int q=0;q<6491;++q) {
      int previous=-1;
      for(int c=0;c<Cap;++c) {
        const int cell=ids[size_t(q)*Cap+c];require(previous<cell && cell<=32768+q,"causal sorted IDs");previous=cell;
      }
    }
  }
};
struct Packed {
  Buffer<uint16_t> k,v;
  Buffer<int32_t> bad;
  explicit Packed(size_t rows):k(rows*Dim),v(rows*Dim),bad(rows) {
    k.put(std::vector<uint16_t>(rows*Dim));v.put(std::vector<uint16_t>(rows*Dim));bad.put(std::vector<int32_t>(rows));
  }
  QsaBf16PackedScratch view() { return {k.data(),v.data(),bad.data()}; }
};

struct Numerical { size_t queries=0,compared=0;double max_abs=0,max_budget_ratio=0; };
float value_bf16(uint16_t x) { return std::bit_cast<float>(uint32_t(x)<<16); }
float value_half(uint16_t x) { return float(std::bit_cast<_Float16>(x)); }
Numerical reference_spots(const std::vector<uint16_t>& query,
    const std::vector<int8_t>& codes,const std::vector<int8_t>& values,
    const std::vector<uint16_t>& scales,const std::vector<uint16_t>& vscales,
    const std::vector<int32_t>& table,const std::vector<int32_t>& ids,
    const std::vector<uint16_t>& observed,int queries,int tile) {
  std::vector<int> spots{0,queries-1,queries/2,1,17,127,511,4095};
  std::erase_if(spots,[&](int q){return q<0 || q>=queries;});
  std::sort(spots.begin(),spots.end());spots.erase(std::unique(spots.begin(),spots.end()),spots.end());
  Numerical result;result.queries=spots.size();
  for(int q:spots) for(int h=0;h<Heads;++h) {
    std::vector<float> score(Cap),v(size_t(Cap)*Dim),acc(Dim,0.f);
    for(int c=0;c<Cap;++c) {
      const int cell=ids[size_t(q)*Cap+c];
      const size_t row=(size_t(table[cell/Page])*KV+h/(Heads/KV))*Page+cell%Page;
      double dot=0;
      for(int d=0;d<Dim;++d) {
        float x=value_bf16(query[(size_t(q)*Heads+h)*Dim+d])*.0625f;
        x=value_bf16(bf16(x*1.4426950408f));
        const float k=value_bf16(bf16(float(codes[row*Dim+d])*value_half(scales[row*4+d/64])));
        dot+=double(x)*k;
        v[size_t(c)*Dim+d]=value_bf16(bf16(float(values[row*Dim+d])*value_half(vscales[row*4+d/64])));
      }
      score[c]=float(dot);
    }
    float maximum=-INFINITY,normalizer=0;
    for(int first=0;first<Cap;first+=tile) {
      const int last=std::min(Cap,first+tile);float next=maximum;
      for(int c=first;c<last;++c) next=std::fmax(next,score[c]);
      const float alpha=normalizer>0?std::exp2(maximum-next):0.f;float sum=0;
      std::vector<float> probability(last-first);
      for(int c=first;c<last;++c) {
        const float p=std::exp2(score[c]-next);sum+=p;
        probability[c-first]=value_bf16(bf16(p));
      }
      for(int d=0;d<Dim;++d) {
        double term=0;for(int c=first;c<last;++c)term+=double(probability[c-first])*v[size_t(c)*Dim+d];
        acc[d]=float(double(acc[d]*alpha)+term);
      }
      normalizer=std::fma(normalizer,alpha,sum);maximum=next;
    }
    for(int d=0;d<Dim;++d) {
      const float ref=value_bf16(bf16(acc[d]/normalizer));
      const float actual=value_bf16(observed[(size_t(q)*Heads+h)*Dim+d]);
      require(std::isfinite(ref) && std::isfinite(actual),"actual-ID scalar output finite");
      const uint16_t bits=bf16(std::fabs(ref));
      const double budget=double(value_bf16(uint16_t(bits+1))-value_bf16(bits))+2e-6;
      const double error=std::fabs(double(actual)-ref);
      require(error<=budget,"actual-ID scalar recurrence bound");
      result.max_abs=std::max(result.max_abs,error);result.max_budget_ratio=std::max(result.max_budget_ratio,error/budget);++result.compared;
    }
  }
  return result;
}

void run(const Selected& source,int queries,bool timing,int process,bool boundary=false) {
  require(!boundary || (queries==3 && !timing),"bounded final-address case");
  const int start=boundary?262141:32768;
  const int cells=boundary?262144:39259,pages=(cells+Page-1)/Page,rows=pages*KV*Page;
  require(queries>0 && queries<=6491,"fixture domain");
  std::vector<int8_t> codes(size_t(rows)*Dim),values(codes.size());
  for(size_t i=0;i<codes.size();++i) {codes[i]=int8_t(int((i*7+3)%15)-7);values[i]=int8_t(int((i*11+5)%31)-15);}
  std::vector<uint16_t> scales(size_t(rows)*4),vscales(scales.size()),query(size_t(queries)*Heads*Dim);
  for(size_t i=0;i<scales.size();++i) {
    scales[i]=half(std::ldexp(1.f,-5-int(i%4)));
    vscales[i]=half(std::ldexp(i%3?1.f:-1.f,-7-int(i%4)));
  }
  for(size_t i=0;i<query.size();++i) query[i]=bf16(float(int(i*13%17)-8)*.125f);
  std::vector<int32_t> table(pages),ids(size_t(queries)*Cap),steps(size_t(queries)*kStepCount);
  for(int p=0;p<pages;++p)table[p]=p;
  for(int q=0;q<queries;++q) {
    steps[q*kStepCount+kStepPos]=start+q;steps[q*kStepCount+kStepNKv]=start+1+q;
    steps[q*kStepCount+kStepNBid]=(start+1+q)/4;steps[q*kStepCount+kStepWidth]=Cap;
    std::transform(source.ids.data()+size_t(q)*Cap,source.ids.data()+size_t(q+1)*Cap,
      ids.data()+size_t(q)*Cap,[&](int cell){return cell+start-32768;});
    require(ids[size_t(q+1)*Cap-1]==start+q,"current/final-slot ID coverage");
  }
  Buffer<int8_t> dk(codes.size()),dv(values.size());
  Buffer<uint16_t> dks(scales.size()),dvs(vscales.size()),dq(query.size()),out(query.size());
  Buffer<int32_t> dt(table.size()),di(ids.size()),ds(steps.size());
  dk.put(codes);dv.put(values);dks.put(scales);dvs.put(vscales);dq.put(query);
  dt.put(table);di.put(ids);ds.put(steps);out.put(std::vector<uint16_t>(query.size()));
  Packed packed(rows);Drain drain;auto shape=qsa_real_shapes();QsaAttnPools pools{};
  pools.page_table=dt.data();pools.k_q=dk.data();pools.v_q=dv.data();pools.k_scale=dks.data();pools.v_scale=dvs.data();
  QsaPromptBf16Bounds bounds{pages,pages};
  auto call=[&](int arm,sycl::event* submitted) {
    bool accepted=false;
    if(!arm) {
      require(!submitted,"unmodified production has no direct-event output");
      accepted=qsa_bf16_prefix_ordered_exp_batch(dq.data(),pools,di.data(),ds.data(),Cap,shape,out.data(),queries,bounds,cells,packed.view(),&rt->compute());
    } else {
      const auto fn=arm==1?pitch_control::qsa_bf16_prefix_ordered_exp_batch:pitch_candidate::qsa_bf16_prefix_ordered_exp_batch;
      accepted=fn(dq.data(),pools,di.data(),ds.data(),Cap,shape,out.data(),queries,bounds,cells,packed.view(),&rt->compute(),submitted);
    }
    require(accepted,"selected cell arm refused");
  };
  std::array<std::vector<uint16_t>,3> expected;
  std::vector<uint16_t> packed_k,packed_v;std::vector<int32_t> packed_bad;
  for(int arm=0;arm<3;++arm)for(int repeat=0;repeat<2;++repeat) {
    sycl::event event;call(arm,arm?&event:nullptr);rt->wait();
    if(arm)require(service(event,arm==1?"control":"candidate",repeat)>0,"direct event clock");
    auto actual=out.get();
    if(!repeat) {
      expected[arm]=actual;
      if(arm)require(actual==expected[0],"pitch arm differs from original32 output bits");
      const auto check=reference_spots(query,codes,values,scales,vscales,table,ids,actual,queries,32);
      std::cout<<"{\"kind\":\"numerical\",\"queries\":"<<queries<<",\"arm\":"<<arm
        <<",\"reference_cells\":"<<(32)<<",\"spot_queries\":"<<check.queries
        <<",\"compared\":"<<check.compared<<",\"max_abs\":"<<check.max_abs
        <<",\"max_budget_ratio\":"<<check.max_budget_ratio<<"}\n";
      if(arm==2) {
        double sum_difference=0,sum_reference=0,maximum=0;size_t changed=0;
        for(size_t i=0;i<actual.size();++i) {
          const double a=value_bf16(actual[i]),b=value_bf16(expected[0][i]);
          require(std::isfinite(a) && std::isfinite(b),"all actual-ID outputs finite");
          sum_difference+=(a-b)*(a-b);sum_reference+=b*b;
          maximum=std::max(maximum,std::fabs(a-b));changed+=actual[i]!=expected[0][i];
        }
        std::cout<<"{\"kind\":\"delta\",\"queries\":"<<queries<<",\"elements\":"<<actual.size()
          <<",\"changed_from32\":"<<changed<<",\"max_abs\":"<<maximum
          <<",\"relative_l2\":"<<std::sqrt(sum_difference/std::max(sum_reference,1e-30))<<"}\n";
      }
    } else require(actual==expected[arm],"same arm output changed");
    if(!arm && !repeat) {packed_k=packed.k.get();packed_v=packed.v.get();packed_bad=packed.bad.get();}
    else require(packed.k.get()==packed_k && packed.v.get()==packed_v && packed.bad.get()==packed_bad,"packed reference differs");
  }
  int final_arm=2;
  if(timing) {
    long long floor=std::numeric_limits<long long>::max();
    for(int i=0;i<128;++i) {
      const auto a=std::chrono::steady_clock::now(),b=std::chrono::steady_clock::now();
      const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
      if(ns>0)floor=std::min(floor,static_cast<long long>(ns));
    }
    require(floor<std::numeric_limits<long long>::max(),"clock floor");
    const int repeats=queries==512?64:4;
    for(int round=0;round<6;++round)for(int offset=0;offset<3;++offset) {
      const int arm=(process+round+offset)%3;
      for(int i=0;i<2;++i)call(arm,nullptr);rt->wait();
      std::vector<sycl::event> events(arm?repeats:0);
      const auto begin=std::chrono::steady_clock::now();
      for(int i=0;i<repeats;++i)call(arm,arm?&events[size_t(i)]:nullptr);
      rt->wait();
      const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
      require(ns>=100000000 && ns>=1000*floor,"timing batch below prospective floor");
      uint64_t device_ns=0,previous_end=0;int call_index=0;
      for(const auto& event:events) {
        const auto start=event.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto end=event.get_profiling_info<sycl::info::event_profiling::command_end>();
        if(start<previous_end || end<=start)std::cerr<<"event_clock_invalid queries="<<queries<<" process="<<process<<" round="<<round<<" arm="<<arm<<" call="<<call_index<<" start="<<start<<" end="<<end<<" previous_end="<<previous_end<<'\n';
        require(start>=previous_end && end>start,"invalid/overlapping direct event");
        device_ns+=end-start;previous_end=end;++call_index;
      }
      if(arm)require(device_ns>0 && device_ns<=uint64_t(ns)+1000000,"event service exceeds host envelope");
      require(out.get()==expected[arm],"timed same-arm output differs");final_arm=arm;
      std::cout<<"{\"kind\":\"sample\",\"process\":"<<process<<",\"queries\":"<<queries<<",\"round\":"<<round<<",\"arm\":"<<arm
        <<",\"repeats\":"<<repeats<<",\"batch_ns\":"<<ns<<",\"attention_device_ns\":";
      if(arm)std::cout<<device_ns;else std::cout<<"null";
      std::cout<<",\"event_calls\":"<<events.size()<<",\"timer_floor_ns\":"<<floor<<"}\n";
    }
  }
  require(dk.get()==codes && dv.get()==values && dks.get()==scales && dvs.get()==vscales && dq.get()==query && dt.get()==table && di.get()==ids && ds.get()==steps,"immutable input changed");
  require(packed.k.get()==packed_k && packed.v.get()==packed_v && packed.bad.get()==packed_bad,"packed operand bytes changed");
  require(out.get()==expected[final_arm],"final same-arm output differs");
  std::cout<<"{\"kind\":\"checked\",\"queries\":"<<queries<<",\"cells\":"<<cells<<",\"guards\":true,\"inputs_immutable\":true,\"cell32_exact\":true,\"own_reference\":true,\"candidate_old_bits_required\":true}\n";
}
} // namespace
int admission_main(int argc, char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    if (argc != 6 || std::string(argv[1]) != "--admit")
      throw std::runtime_error("--admit original32 private32 padded32 loader required");
    std::cout << "admission_stage=before-runtime\n";
    auto runtime = strata::sycl_backend::runtime_for();
    std::cout << "admission_stage=runtime-ready device="
              << runtime->device().get_info<sycl::info::device::name>() << '\n';
    const auto properties = qsa_slm_pitch_admit(runtime->compute(),
        {argv[2], argv[3], argv[4]}, argv[5]);
    for (std::size_t arm = 0; arm < properties.size(); ++arm) {
      const auto &p = properties[arm];
      std::cout << "arm=" << arm << " max_workgroup=" << p.maximum_workgroup
                << " subgroup=" << p.subgroup << " arguments=" << p.arguments
                << " reported_local_bytes=" << p.reported_local_bytes
                << " declared_dynamic_local_bytes=" << p.declared_local_bytes
                << " reserve_local_bytes=4096 conservative_local_budget="
                << p.conservative_local_budget << " per_group_local_max="
                << p.per_group_local_max << " spill_descriptor="
                << p.spill_descriptor << " status=PASS\n";
    }
    runtime->wait();
    runtime.reset();
    std::cout << "admission_status=PASS kernel_submissions=0 numerical=UNRUN "
                 "performance=UNRUN physical262144_model=UNRUN\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "admission_status=FAIL error=" << error.what() << '\n';
    return 1;
  }
}

int main(int argc,char** argv) {
  try {
    if(argc==6 && std::string(argv[1])=="--admit")return admission_main(argc,argv);
    require(argc==3 || argc==4,"--smoke|--boundary|--bench FIXTURE [PROCESS]");
    const std::string mode=argv[1];require(mode=="--smoke" || mode=="--boundary" || mode=="--bench","unknown mode");
    require((mode=="--bench")==bool(argc==4),"process only for bench");
    const int process=argc==4?std::stoi(argv[3]):0;require(process>=0 && process<3,"process ordinal");
    const Selected source(argv[2]);std::cout.setf(std::ios::unitbuf);std::cout<<std::setprecision(10);
    gate("before-runtime");rt=sycl_backend::runtime_for();
    require(rt->compute().has_property<sycl::property::queue::in_order>() && rt->compute().has_property<sycl::property::queue::enable_profiling>(),"queue properties");
    std::cout<<"device="<<rt->device().get_info<sycl::info::device::name>()<<'\n';gate("runtime-ready");
    if(mode=="--smoke" || mode=="--boundary")run(source,3,false,0,mode=="--boundary");
    else {run(source,512,true,process);run(source,6491,true,process);}
    rt->wait();rt.reset();gate("after-cleanup");std::cout<<"{\"kind\":\"PASS\"}\n";
    return 0;
  }catch(const std::exception& e) {std::cerr<<"QSA SLM pitch failure: "<<e.what()<<'\n';return 1;}
}
