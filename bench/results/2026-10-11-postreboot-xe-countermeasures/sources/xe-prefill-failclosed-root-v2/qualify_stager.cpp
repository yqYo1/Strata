// Compiles the complete actual production Stager/guard with GPU API stand-ins.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <iostream>
#include <cassert>
#include "strata/prefill/publication.hpp"
namespace strata::gpu {
struct Event {};
using Stream=void*;
std::atomic<bool> record_ok{true}, sync_ok{true};
std::vector<Stream> drains;
Stream refused=nullptr;
bool alloc_host(void**,size_t) { return false; }
bool event_create(Event** e) { *e=new Event;return true; }
void event_destroy(Event* e) {delete e;}
void free(void*) {assert(false);}
bool event_record(Event*,Stream) {return record_ok.load();}
bool event_sync(Event*) {return sync_ok.load();}
const char* last_error(){return "injected GPU API failure";}
bool stream_sync(Stream s){drains.push_back(s);return s!=refused;}
}
namespace strata::core {
int current_device(){return 0;}
struct OnDevice {explicit OnDevice(int){}};
struct ExpertSource { bool copy_blob(int32_t,int32_t,uint8_t*){return false;} };
}
namespace subject {
namespace publication=strata::prefill::publication;
namespace core=strata::core;
bool force_pageable(){return true;}
bool stager_sleep(){return true;}
#include "stager-exact.inc"
#include "stager-guard-exact.inc"
}
int main(){
 using subject::Stager;
 setenv("STRATA_STAGER_RING","2",1);
 unsigned cases=0;
 // Full actual worker/start/wait/finish across wraps and reused generations.
 {
  Stager s;assert(s.init(8,2));
  for(int generation=0;generation<2;++generation){
   std::vector<std::vector<uint8_t>> data(12,std::vector<uint8_t>(8));
   std::vector<Stager::Job> jobs;
   for(int j=0;j<12;++j){std::fill(data[j].begin(),data[j].end(),uint8_t(j+generation*20));jobs.push_back({data[j].data(),8});}
   s.start(std::move(jobs));
   for(int j=0;j<12;++j){const auto* p=s.wait(j);for(int k=0;k<8;++k)assert(p[k]==uint8_t(j+generation*20));s.issued_one(j,nullptr);}
   s.finish();assert(s.active.load()==0);++cases;
  }
 }
 // Actual sleeping worker cancellation and late normal publication cannot undo abort.
 {
  Stager s;assert(s.init(8,2));uint8_t data[8]={};std::vector<Stager::Job> jobs(8,{data,8});s.start(std::move(jobs));
  s.wait(0);s.wait(1);s.cancel();s.issued_one(0,nullptr);assert(s.issued.load()==(1<<30));s.finish();assert(s.active.load()==0);++cases;
 }
 // Failed event publication must not advance normal issued state.
 {
  Stager s;assert(s.init(8,0));strata::gpu::record_ok=false;bool threw=false;
  try{s.issued_one(0,nullptr);}catch(const std::runtime_error&){threw=true;}
  assert(threw&&s.issued.load()==0);strata::gpu::record_ok=true;s.cancel();++cases;
 }
 // Actual worker failure is transmitted and the active count retires normally.
 {
  Stager s;assert(s.init(8,1));uint8_t data[8]={};strata::gpu::sync_ok=false;s.start({{data,8}});bool threw=false;
  try{s.wait(0);}catch(const std::runtime_error&){threw=true;}
  assert(threw);s.cancel();s.finish();assert(s.active.load()==0);strata::gpu::sync_ok=true;++cases;
 }
 // A source-copy refusal must not hand a successful buffer to the consumer.
 {
  Stager s;assert(s.init(8,1));strata::core::ExpertSource source;s.start({{nullptr,8,&source,0,0}});bool threw=false;
  try{s.wait(0);}catch(const std::runtime_error&){threw=true;}
  assert(threw);s.cancel();s.finish();++cases;
 }
 // The complete actual guard checks both drains before reporting normal completion.
 auto cs=reinterpret_cast<void*>(1),copy=reinterpret_cast<void*>(2);
 for(auto fail: {cs,copy}){
  Stager s;assert(s.init(8,0));strata::gpu::drains.clear();strata::gpu::refused=fail;
  subject::StagerDone g{&s,cs,copy};bool threw=false;
  try{g.complete();}catch(const std::runtime_error&){threw=true;}
  assert(threw&&g.st==&s);strata::gpu::refused=nullptr;++cases;
 }
 {
  Stager s;assert(s.init(8,0));strata::gpu::drains.clear();subject::StagerDone g{&s,cs,copy};g.complete();
  assert(g.st==nullptr&&(strata::gpu::drains==std::vector<strata::gpu::Stream>{cs,copy}));++cases;
 }
 assert(cases==9);
 std::cout<<"{\"passed\":true,\"actual_stager_cases\":9,\"gpu_executed\":false}\n";
}
