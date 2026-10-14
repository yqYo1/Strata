// Standalone proposal: production SYCL Runtime + CUDA-named compatibility API.
// No model payload access, kernels, allocation fallback, or overlap experiment.
#include <cuda_runtime.h>
#include "strata/sycl/runtime.hpp"
#include "transfer_failure_guard.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using Clock = std::chrono::steady_clock;
constexpr size_t cap = 256ull << 20, guard = 64;
struct Counts { uint64_t copies=0, bytes=0, records=0, syncs=0, elapsed=0; } counts;
void ck(cudaError_t e, const char* api) {
  if(e != cudaSuccess) {
    std::fprintf(stderr, "transfer: CUDA-compat operation failed: %s: %s\n", api, cudaGetErrorString(e));
    std::fflush(stderr);
    std::_Exit(1); // No host-source vector unwinding after a failed device API.
  }
}
void copy(void* d,const void* s,size_t n,cudaMemcpyKind k,cudaStream_t q) {
  ++counts.copies; counts.bytes+=n; ck(cudaMemcpyAsync(d,s,n,k,q),"cudaMemcpyAsync");
}
void record(cudaEvent_t e,cudaStream_t q) { ++counts.records; ck(cudaEventRecord(e,q),"cudaEventRecord"); }
void sync(cudaEvent_t e) { ++counts.syncs; ck(cudaEventSynchronize(e),"cudaEventSynchronize"); }
double elapsed(cudaEvent_t a,cudaEvent_t b) { float ms=0; ++counts.elapsed; ck(cudaEventElapsedTime(&ms,a,b),"cudaEventElapsedTime"); if(!std::isfinite(ms)||ms<=0) throw std::runtime_error("nonpositive/nonfinite event interval"); return ms; }
double ms(Clock::time_point a,Clock::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); }
std::string quoted(const std::string& s) { std::ostringstream o; o << '"'; for(unsigned char c:s) { if(c=='"'||c=='\\') o << '\\' << c; else if(c<32) o << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(c) << std::dec; else o << c; } o << '"'; return o.str(); }
uint8_t pattern(size_t slot,size_t byte,unsigned seed) {
  uint64_t x=byte+0x9e3779b97f4a7c15ull*(slot+1)+0xd1b54a32d192ed03ull*seed;
  x=(x^(x>>30))*0xbf58476d1ce4e5b9ull; x=(x^(x>>27))*0x94d049bb133111ebull;
  return uint8_t(x^(x>>31));
}
// Explicit close reports every cleanup failure. Failed in-flight work is fail-stop:
// main uses _Exit rather than destroying allocations while commands may still run.
struct Resources {
  cudaStream_t q=nullptr; std::array<cudaEvent_t,5> ev{};
  uint8_t *h=nullptr,*d=nullptr; bool usm=false; size_t allocated=0;
  void open(size_t n,bool host_usm) {
    usm=host_usm; allocated=n;
    ck(cudaStreamCreate(&q),"cudaStreamCreate");
    if(!q->has_property<sycl::property::queue::in_order>() || !q->has_property<sycl::property::queue::enable_profiling>()) throw std::runtime_error("stream properties mismatch");
    auto r=strata::sycl_backend::runtime_for();
    if(q->get_context()!=r->context() || q->get_device()!=r->device()) throw std::runtime_error("stream context/device mismatch");
    for(auto& e:ev) ck(cudaEventCreate(&e),"cudaEventCreate");
    ck(cudaMalloc(reinterpret_cast<void**>(&d),n),"cudaMalloc");
    if(usm) ck(cudaMallocHost(reinterpret_cast<void**>(&h),n),"cudaMallocHost");
    else { h=static_cast<uint8_t*>(std::malloc(n)); if(!h) throw std::runtime_error("ordinary malloc failed"); }
  }
  bool close() {
    bool ok=true;
    auto check=[&](cudaError_t e,const char* api) { if(e!=cudaSuccess) { std::cerr << "{\"type\":\"cleanup_error\",\"api\":" << quoted(api) << ",\"error\":" << quoted(cudaGetErrorString(e)) << "}\n"; ok=false; } };
    check(cudaStreamSynchronize(q),"cudaStreamSynchronize(cleanup)");
    if(!ok) return false; // caller fail-stops; never free possibly active storage
    for(auto& e:ev) if(e) { check(cudaEventDestroy(e),"cudaEventDestroy"); e=nullptr; }
    if(d) { check(cudaFree(d),"cudaFree"); d=nullptr; }
    if(h) { if(usm) check(cudaFreeHost(h),"cudaFreeHost"); else std::free(h); h=nullptr; }
    if(q) { check(cudaStreamDestroy(q),"cudaStreamDestroy"); q=nullptr; }
    return ok;
  }
};
struct Result { double event_ms=0,enqueue_ms=0,enqueue_span_ms=0,completion_ms=0; std::vector<double> calls; };
void array(const std::vector<double>& v) { std::cout << '['; for(size_t i=0;i<v.size();++i) { if(i) std::cout << ','; std::cout << v[i]; } std::cout << ']'; }
void sample(const char* lane,bool usm,bool h2d,size_t size,size_t slots,size_t start,int batch,int rep,int burst,const Result& t,const Counts& before,const Counts& timed,uint64_t verify_copies,uint64_t verify_bytes,size_t allocation,const Counts& arm_begin) {
  std::cout << "{\"type\":\"sample\",\"lane\":" << quoted(lane)
    << ",\"allocation\":" << quoted(usm?"Host-USM":"ordinary-host") << ",\"direction\":" << quoted(h2d?"H2D":"D2H")
    << ",\"rep\":" << rep << ",\"burst\":" << burst << ",\"blob_bytes\":" << size << ",\"copies\":" << batch << ",\"payload_bytes\":" << size*size_t(batch)
    << ",\"corpus_slots\":" << slots << ",\"slot_start\":" << start << ",\"allocation_bytes_per_buffer\":" << allocation
    << ",\"event_ms\":" << t.event_ms << ",\"copy_api_total_ms\":" << t.enqueue_ms << ",\"copy_api_ms\":"; array(t.calls);
  std::cout << ",\"host_group_enqueue_span_ms\":" << t.enqueue_span_ms
    << ",\"host_group_completion_ms\":" << t.completion_ms
    << ",\"event_GBps\":" << double(size*size_t(batch))/(t.event_ms*1e6)
    << ",\"host_group_GBps\":" << double(size*size_t(batch)*(std::string(lane)=="bulk"?4:1))/(t.completion_ms*1e6)
    << ",\"host_group_completion_scope\":" << quoted(std::string(lane)=="bulk"?"four bursts including event records; shared by four samples":"one queued batch including start/end events")
    << ",\"timed_api_counts\":{\"copy\":" << timed.copies-before.copies << ",\"copy_bytes\":" << timed.bytes-before.bytes << ",\"event_record\":" << timed.records-before.records << ",\"event_sync\":" << timed.syncs-before.syncs << ",\"event_elapsed\":" << timed.elapsed-before.elapsed << "}"
    << ",\"post_completion_api_counts\":{\"event_elapsed\":" << (std::string(lane)=="bulk"?4:1) << "}"
    << ",\"setup_api_counts\":{\"copy\":" << before.copies-arm_begin.copies << ",\"copy_bytes\":" << before.bytes-arm_begin.bytes << "}"
    << ",\"allocation_api_counts\":{\"cudaMalloc\":1,\"cudaMallocHost\":" << (usm?1:0) << ",\"malloc\":" << (usm?0:1) << ",\"cudaFree\":1,\"cudaFreeHost\":" << (usm?1:0) << ",\"free\":" << (usm?0:1) << "}"
    << ",\"lifecycle_api_counts\":{\"event_create\":5,\"event_destroy\":5,\"stream_create\":1,\"stream_destroy\":1,\"stream_sync\":" << (h2d?6:5) << "}"
    << ",\"verification_api_counts\":{\"copy\":" << verify_copies << ",\"copy_bytes\":" << verify_bytes << "}"
    << ",\"verified_full_payload\":true,\"verified_all_corpus_slots\":true,\"verified_guards\":true,\"guard_bytes_per_side\":64,\"status\":\"pass\"}\n";
}
void arm(size_t bytes,bool bulk,bool usm,bool h2d,int batch,int rep,int expert_count,bool diagnostic=false) {
  std::cout << "{\"type\":\"arm_begin\",\"lane\":" << quoted(bulk?"bulk":"expert") << ",\"allocation\":" << quoted(usm?"Host-USM":"ordinary-host") << ",\"direction\":" << quoted(h2d?"H2D":"D2H") << ",\"blob_bytes\":" << bytes << ",\"batch\":" << batch << ",\"rep\":" << rep << "}\n"; std::cout.flush();
  const Counts arm_begin=counts;
  size_t stride=bytes+2*guard;
  size_t slots=bulk?1:std::min(size_t(diagnostic?32:expert_count),cap/stride);
  if(slots<size_t(batch)) throw std::runtime_error("batch does not fit distinct-slot lane bound");
  // Bulk payload is exactly 256 MiB; 128 guard bytes are a disclosed exception.
  size_t allocation=slots*stride;
  Resources r; r.open(allocation,usm);
  std::vector<uint8_t> original(allocation),expected_device(allocation),readback(allocation);
  TransferFailureGuard copy_lifetime; // Runs before the copy-buffer destructors during unwinding.
  for(size_t s=0;s<slots;++s) {
    std::fill(original.begin()+s*stride,original.begin()+(s+1)*stride,0xa7);
    for(size_t i=0;i<bytes;++i) original[s*stride+guard+i]=pattern(s,i,unsigned(rep+1));
  }
  expected_device=original;
  std::memcpy(r.h,original.data(),allocation); // faults every ordinary-host page
  copy(r.d,original.data(),allocation,cudaMemcpyHostToDevice,r.q);
  ck(cudaStreamSynchronize(r.q),"initialization sync");
  const size_t start=bulk?0:(size_t(rep)*size_t(batch))%slots;
  // One exact-copy warm-up, outside timing. D2H does not access the host again before completion.
  copy(h2d?r.d+guard:r.h+guard,h2d?r.h+guard:r.d+guard,bytes,h2d?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost,r.q);
  ck(cudaStreamSynchronize(r.q),"warmup sync");
  if(h2d) {
    // Timed destinations start poisoned so readback cannot pass due to initialization.
    std::fill(expected_device.begin(),expected_device.end(),0xa7);
    copy(r.d,expected_device.data(),allocation,cudaMemcpyHostToDevice,r.q);
    // This vector remains a live asynchronous copy source until the wait.
    // Build the post-transfer expectation only after the poison upload ends.
    ck(cudaStreamSynchronize(r.q),"poison upload sync");
    for(int j=0;j<batch;++j) { size_t s=(start+size_t(j))%slots; std::memcpy(expected_device.data()+s*stride+guard,original.data()+s*stride+guard,bytes); }
  } else {
    for(size_t s=0;s<slots;++s) std::memset(r.h+s*stride+guard,0x53,bytes);
  }
  ck(cudaStreamSynchronize(r.q),"premeasurement sync");
  Counts before=counts;
  std::vector<Result> results(bulk?4:1);
  for(auto& result:results) result.calls.reserve(size_t(batch));
  auto begin=Clock::now(); record(r.ev[0],r.q);
  for(size_t b=0;b<results.size();++b) {
    auto& t=results[b];
    for(int j=0;j<batch;++j) {
      size_t s=(start+size_t(j))%slots;
      auto a=Clock::now();
      copy(h2d?r.d+s*stride+guard:r.h+s*stride+guard,h2d?r.h+s*stride+guard:r.d+s*stride+guard,bytes,h2d?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost,r.q);
      double call_ms=ms(a,Clock::now()); t.calls.push_back(call_ms); t.enqueue_ms+=call_ms;
    }
    record(r.ev[b+1],r.q);
  }
  auto submitted=Clock::now(); sync(r.ev[results.size()]); auto finish=Clock::now();
  Counts timed=counts; // Queries below are outside the timed interval.
  for(size_t b=0;b<results.size();++b) { results[b].event_ms=elapsed(r.ev[b],r.ev[b+1]); results[b].completion_ms=ms(begin,finish); results[b].enqueue_span_ms=ms(begin,submitted); }
  // Full device corpus and both guards read back after completion, never timed.
  copy(readback.data(),r.d,allocation,cudaMemcpyDeviceToHost,r.q);
  ck(cudaStreamSynchronize(r.q),"verification sync");
  if(readback!=expected_device) throw std::runtime_error("device corpus/payload/guard mismatch");
  std::vector<uint8_t> expected_host=original;
  if(!h2d) {
    for(size_t s=0;s<slots;++s) std::fill(expected_host.begin()+s*stride+guard,expected_host.begin()+s*stride+guard+bytes,0x53);
    for(int j=0;j<batch;++j) { size_t s=(start+size_t(j))%slots; std::memcpy(expected_host.data()+s*stride+guard,original.data()+s*stride+guard,bytes); }
  }
  if(std::memcmp(r.h,expected_host.data(),allocation)!=0) throw std::runtime_error("host corpus/payload/guard mismatch");
  if(!r.close()) throw std::runtime_error("cleanup failed");
  copy_lifetime.closed();
  for(size_t b=0;b<results.size();++b) sample(bulk?"bulk":"expert",usm,h2d,bytes,slots,start,batch,rep,int(b),results[b],before,timed,counts.copies-timed.copies,counts.bytes-timed.bytes,allocation,arm_begin);
  if(bulk) {
    std::vector<double> rates; for(auto& t:results) rates.push_back(double(bytes)/(t.event_ms*1e6));
    auto sorted=rates; std::sort(sorted.begin(),sorted.end());
    std::cout << "{\"type\":\"bulk_summary\",\"allocation\":" << quoted(usm?"Host-USM":"ordinary-host") << ",\"direction\":" << quoted(h2d?"H2D":"D2H") << ",\"rep\":" << rep << ",\"individual_GBps\":"; array(rates);
    std::cout << ",\"best_GBps\":" << sorted.back() << ",\"median_GBps\":" << (sorted[1]+sorted[2])/2 << ",\"host_group_GBps\":" << double(bytes)*4/(results[0].completion_ms*1e6) << "}\n";
  }
  std::cout.flush();
}
int main(int argc,char** argv) {
  // Required positional configuration prevents accidentally launching a default GPU run.
  // EXE extents.tsv repetitions(3..9) deadline_seconds(30..1800) bulk|expert|all|diagnostic
  if(argc!=5) { std::cerr << "usage: sycl_transfer_ceiling extents.tsv reps deadline_seconds bulk|expert|all|diagnostic\n"; return 2; }
  std::atomic<bool> finished{false};
  std::thread watchdog;
  try {
    int reps=std::stoi(argv[2]),deadline=std::stoi(argv[3]); std::string mode=argv[4];
    if(reps<3||reps>9||deadline<30||deadline>1800||(mode!="bulk"&&mode!="expert"&&mode!="all"&&mode!="diagnostic")) throw std::runtime_error("invalid finite configuration");
    std::ifstream input(argv[1]); int experts=0,layers=0;
    if(!(input>>experts>>layers)||experts<32||experts>4096||layers<1||layers>4096) throw std::runtime_error("bad metadata extent header");
    std::vector<size_t> sizes;
    for(int l=0;l<layers;++l) { int id; uint64_t n; if(!(input>>id>>n)||id!=l||n==0||n>(cap-128)/32) throw std::runtime_error("bad metadata extent row / batch32 exceeds bound"); sizes.push_back(size_t(n)); }
    std::string extra; if(input>>extra) throw std::runtime_error("trailing extent data");
    std::sort(sizes.begin(),sizes.end()); std::vector<size_t> shapes{sizes.front(),sizes[(sizes.size()-1)/2],sizes.back()};
    std::sort(shapes.begin(),shapes.end()); shapes.erase(std::unique(shapes.begin(),shapes.end()),shapes.end());
    watchdog=std::thread([&] { auto stop=Clock::now()+std::chrono::seconds(deadline); while(!finished.load()) { if(Clock::now()>=stop) { std::fputs("{\"type\":\"failure\",\"reason\":\"deadline; in-flight buffers not freed\"}\n",stderr); std::fflush(stderr); std::_Exit(124); } std::this_thread::sleep_for(std::chrono::milliseconds(100)); } });
    auto runtime=strata::sycl_backend::runtime_for();
    std::cout << std::setprecision(12) << "{\"type\":\"run\",\"schema\":\"strata-transfer-ceiling-v1\",\"device\":" << quoted(runtime->device().get_info<sycl::info::device::name>()) << ",\"driver\":" << quoted(runtime->device().get_info<sycl::info::device::driver_version>()) << ",\"platform\":" << quoted(runtime->device().get_platform().get_info<sycl::info::platform::name>()) << ",\"compiler\":" << quoted(__VERSION__) << ",\"extent_file\":" << quoted(argv[1]) << ",\"mode\":" << quoted(mode) << ",\"runtime_ordinal\":0,\"n_expert\":" << experts << ",\"n_layers\":" << layers << ",\"repetitions\":" << reps << ",\"deadline_seconds\":" << deadline << ",\"lane_cap_bytes\":" << cap << ",\"bulk_guard_exception_bytes\":128,\"stream\":\"explicit production make_queue(true) in-order context checked\",\"event_interval\":\"compat barrier command_end difference\",\"overlap_tested\":false,\"physical_OS_pin_state_assumed\":false}\n";
    // Rotate allocation/direction order between repetitions; no quiet sleep is invented.
    for(int rep=0;rep<reps;++rep) for(int k=0;k<4;++k) { int a=(k+rep)%4; bool usm=(a&1)==0,h2d=(a&2)==0;
      if(mode!="expert"&&mode!="diagnostic") arm(cap,true,usm,h2d,1,rep,experts);
      if(mode=="diagnostic") arm(shapes.front(),false,usm,h2d,1,rep,experts,true);
      if(mode=="all"||mode=="expert") for(size_t shape:shapes) for(int batch:{1,8,32}) arm(shape,false,usm,h2d,batch,rep,experts);
    }
    runtime->wait(); runtime->check(); finished=true; watchdog.join();
    std::cout << "{\"type\":\"complete\",\"status\":\"pass\",\"cleanup\":\"checked\",\"total_copy_api_calls\":" << counts.copies << ",\"total_copy_api_bytes\":" << counts.bytes << ",\"total_event_records\":" << counts.records << ",\"total_event_sync\":" << counts.syncs << ",\"total_event_elapsed\":" << counts.elapsed << "}\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "{\"type\":\"failure\",\"status\":\"fail\",\"reason\":" << quoted(e.what()) << ",\"cleanup\":\"process fail-stop; incomplete work never frees buffers\"}\n"; std::cerr.flush(); std::cout.flush(); std::_Exit(1);
  }
}
