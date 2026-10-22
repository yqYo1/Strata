// Host-only equivalence and cost fixture for the actual generated PTI methods.
// No Level Zero, SYCL, device enumeration or inference code is linked.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct Group { uint32_t x, y, z; };
struct Key {
  uint64_t kernel_command_id_, mem_size_;
  int tile_;
  Group group_count_;
  auto fields() const { return std::tie(kernel_command_id_,mem_size_,tile_,group_count_.x,group_count_.y,group_count_.z); }
  bool operator<(const Key& r) const { return fields()<r.fields(); }
};
struct Time {
  uint64_t append_time_,submit_time_,execute_time_,min_time_,max_time_,call_count_;
  auto fields() const { return std::tie(append_time_,submit_time_,execute_time_,min_time_,max_time_,call_count_); }
};
using Stats=std::map<Key,Time>;
Stats* global_device_time_stats_;
std::vector<std::string> base_names;
uint64_t name_lookups;
struct Options { bool verbose; };

// Deliberately controllable names exercise collisions, tile/group/memory
// distinctions and comma/quote characters. Production naming is unchanged by
// the overlay; this stub does not reproduce its locks,demangling or cost.
std::string GetZeKernelCommandName(uint64_t id,const Group& group,size_t bytes,bool detailed) {
  ++name_lookups;
  auto name=base_names.at(id);
  if(detailed) name+="["+std::to_string(group.x)+";"+std::to_string(group.y)+";"+
      std::to_string(group.z)+"]["+std::to_string(bytes)+"]";
  return name;
}

struct Reference {
  Options options_;
#include "aggregate-original.inc"
};
struct Candidate {
  Options options_;
#include "aggregate-candidate.inc"
};

void require(bool b,const char* text) { if(!b) throw std::runtime_error(text); }
void equal(const Stats& a,const Stats& b) {
  require(a.size()==b.size(),"representative count differs");
  auto x=a.begin(),y=b.begin();
  for(;x!=a.end();++x,++y) {
    require(x->first.fields()==y->first.fields(),"representative key/order differs");
    require(x->second.fields()==y->second.fields(),"six integer fields differ");
  }
}

Stats input(size_t n,bool unique,bool overflow) {
  Stats stats;std::mt19937_64 random(0x3ef2c41);
  base_names.assign(n+1,"");
  std::vector<size_t> order(n);
  for(size_t i=0;i<n;++i) order[i]=i+1;
  std::shuffle(order.begin(),order.end(),random);
  for(auto id:order) {
    const uint64_t shape=unique?id:id%7;
    base_names[id]="\"kernel<half, "+std::to_string(shape)+">\"";
    Key key{id,shape*64,int(shape%3)-1,{uint32_t(shape%5+1),2,3}};
    uint64_t calls=random()%16+1,lo=random()%100+1,hi=lo+500;
    Time time{random()%1000,random()%1000,(lo+200)*calls,lo,hi,calls};
    if(overflow) {
      time.append_time_=UINT64_MAX-id;
      time.submit_time_=UINT64_MAX-id*3;
      time.execute_time_=UINT64_MAX-id*7;
      time.call_count_=UINT64_MAX-id*11;
    }
    stats.emplace(key,time);
  }
  return stats;
}

template<class Collector> uint64_t apply(Stats& stats,bool detailed) {
  global_device_time_stats_=&stats;name_lookups=0;
  Collector{Options{detailed}}.AggregateDeviceTimeStats();
  global_device_time_stats_=nullptr;
  return name_lookups;
}

template<class Collector> uint64_t timed(const Stats& original,Stats& result) {
  result=original; // Allocation and reset stay outside the measured region.
  global_device_time_stats_=&result;name_lookups=0;
  auto begin=std::chrono::steady_clock::now();
  Collector{Options{true}}.AggregateDeviceTimeStats();
  auto end=std::chrono::steady_clock::now();
  global_device_time_stats_=nullptr;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
}

int main() try {
  unsigned cases=0;
  for(size_t n:{size_t(0),size_t(1),size_t(17),size_t(257),size_t(513)})
    for(bool unique:{false,true}) for(bool overflow:{false,true}) for(bool detailed:{false,true}) {
      auto source=input(n,unique,overflow),a=source,b=source;
      apply<Reference>(a,detailed);
      require(apply<Candidate>(b,detailed)==n,"candidate name evaluations are not linear");
      equal(a,b);
      const auto once=b;
      apply<Reference>(a,detailed);apply<Candidate>(b,detailed);
      equal(a,b);equal(once,b);++cases;
    }
  const size_t n=2048;
  auto source=input(n,true,false),a=source,b=source;
  const auto reference_calls=apply<Reference>(a,true);
  const auto candidate_calls=apply<Candidate>(b,true);
  equal(a,b);
  require(reference_calls==n*(n+1)/2 && candidate_calls==n,"unique-name work formula differs");
  uint64_t timer=UINT64_MAX;
  for(int i=0;i<128;++i) {
    auto x=std::chrono::steady_clock::now(),y=std::chrono::steady_clock::now();
    auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(y-x).count();
    if(ns>0) timer=std::min(timer,uint64_t(ns));
  }
  require(timer!=UINT64_MAX,"missing positive clock calibration");
  std::printf("aggregation_equivalence cases=%u entries=%zu reference_name_calls=%llu candidate_name_calls=%llu "
      "keys=PASS fields=PASS overflow=PASS idempotence=PASS timer_ns=%llu\n",cases,n,
      (unsigned long long)reference_calls,(unsigned long long)candidate_calls,(unsigned long long)timer);
  for(int i=0;i<7;++i) {
    uint64_t old_ns,new_ns;
    if(i%2==0) { old_ns=timed<Reference>(source,a);new_ns=timed<Candidate>(source,b); }
    else { new_ns=timed<Candidate>(source,b);old_ns=timed<Reference>(source,a); }
    equal(a,b);
    require(old_ns>=100000 && new_ns>=100000 && std::min(old_ns,new_ns)>=timer*1000,
        "timing batch below committed duration/overhead floor");
    std::printf("aggregation_sample index=%d order=%s reference_ns=%llu candidate_ns=%llu\n",i,
        i%2==0?"reference,candidate":"candidate,reference",(unsigned long long)old_ns,(unsigned long long)new_ns);
  }
  std::puts("aggregation_fixture PASS GPU=NONE inference=UNCHANGED");
  return 0;
} catch(const std::exception& e) {
  std::fprintf(stderr,"aggregation_fixture FAIL %s\n",e.what());return 1;
}
