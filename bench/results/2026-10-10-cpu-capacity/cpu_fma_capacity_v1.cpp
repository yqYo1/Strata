#include <immintrin.h>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <thread>
#include <vector>
using Clock=std::chrono::steady_clock;
constexpr uint64_t iterations=64000000;
using Out=std::array<float,96>;
__attribute__((noinline)) Out compute(float m,float b){
 const auto mul=_mm256_set1_ps(m),add=_mm256_set1_ps(b);
 auto a0=_mm256_set1_ps(0.f/16),a1=_mm256_set1_ps(1.f/16),a2=_mm256_set1_ps(2.f/16),a3=_mm256_set1_ps(3.f/16),a4=_mm256_set1_ps(4.f/16),a5=_mm256_set1_ps(5.f/16),a6=_mm256_set1_ps(6.f/16),a7=_mm256_set1_ps(7.f/16),a8=_mm256_set1_ps(8.f/16),a9=_mm256_set1_ps(9.f/16),a10=_mm256_set1_ps(10.f/16),a11=_mm256_set1_ps(11.f/16);
 for(uint64_t i=0;i<iterations;++i){
 a0=_mm256_fmadd_ps(a0,mul,add);a1=_mm256_fmadd_ps(a1,mul,add);a2=_mm256_fmadd_ps(a2,mul,add);a3=_mm256_fmadd_ps(a3,mul,add);a4=_mm256_fmadd_ps(a4,mul,add);a5=_mm256_fmadd_ps(a5,mul,add);a6=_mm256_fmadd_ps(a6,mul,add);a7=_mm256_fmadd_ps(a7,mul,add);a8=_mm256_fmadd_ps(a8,mul,add);a9=_mm256_fmadd_ps(a9,mul,add);a10=_mm256_fmadd_ps(a10,mul,add);a11=_mm256_fmadd_ps(a11,mul,add);}
 Out o{};_mm256_storeu_ps(o.data()+0,a0);_mm256_storeu_ps(o.data()+8,a1);_mm256_storeu_ps(o.data()+16,a2);_mm256_storeu_ps(o.data()+24,a3);_mm256_storeu_ps(o.data()+32,a4);_mm256_storeu_ps(o.data()+40,a5);_mm256_storeu_ps(o.data()+48,a6);_mm256_storeu_ps(o.data()+56,a7);_mm256_storeu_ps(o.data()+64,a8);_mm256_storeu_ps(o.data()+72,a9);_mm256_storeu_ps(o.data()+80,a10);_mm256_storeu_ps(o.data()+88,a11);return o;
}
int main(int argc,char**argv)try{
 if(argc!=3)throw std::runtime_error("need exactly multiplier and addend");char*e1=nullptr,*e2=nullptr;float m=std::strtof(argv[1],&e1),b=std::strtof(argv[2],&e2);if(*e1||*e2||!std::isfinite(m)||!std::isfinite(b)||m!=0.99999904632568359375f||b!=0.00000095367431640625f)throw std::runtime_error("fixed recurrence inputs only");
 std::array<float,12> reference{};for(int a=0;a<12;++a){float x=float(a)/16;for(uint64_t i=0;i<iterations;++i)x=std::fma(x,m,b);reference[a]=x;}
 for(int count:{1,6})for(int r=-1;r<7;++r){
  std::vector<Out>out(count);std::vector<int> observed(count,-1),errors(count,0);std::vector<std::thread> threads;threads.reserve(count);std::barrier ready(count+1),gate(count+1),finish(count+1);
  for(int i=0;i<count;++i)threads.emplace_back([&,i]{cpu_set_t set;CPU_ZERO(&set);CPU_SET(i,&set);errors[i]=pthread_setaffinity_np(pthread_self(),sizeof(set),&set);cpu_set_t actual;CPU_ZERO(&actual);if(pthread_getaffinity_np(pthread_self(),sizeof(actual),&actual)||CPU_COUNT(&actual)!=1||!CPU_ISSET(i,&actual))errors[i]=1;observed[i]=sched_getcpu();ready.arrive_and_wait();gate.arrive_and_wait();out[i]=compute(m,b);finish.arrive_and_wait();});
  ready.arrive_and_wait();auto start=Clock::now();gate.arrive_and_wait();finish.arrive_and_wait();auto end=Clock::now();for(auto&t:threads)t.join();
  for(int i=0;i<count;++i){if(errors[i]||observed[i]!=i)throw std::runtime_error("affinity mismatch");for(int a=0;a<12;++a)for(int lane=0;lane<8;++lane)if(!std::isfinite(out[i][a*8+lane])||out[i][a*8+lane]!=reference[a])throw std::runtime_error("complete96 float reference mismatch");}
  const double seconds=std::chrono::duration<double>(end-start).count();if(!std::isfinite(seconds)||seconds<=0)throw std::runtime_error("badclock");
  std::cout<<std::setprecision(17)<<"{\"sample\":"<<r<<",\"threads\":"<<count<<",\"iterations_per_thread\":"<<iterations<<",\"independent_accumulators\":12,\"FP32_lanes\":8,\"operations_per_fma\":2,\"seconds\":"<<seconds<<",\"GFLOPs\":"<<double(iterations)*12*8*2*count/seconds/1e9<<",\"full_reference_pass\":true,\"affinity_pass\":true}"<<'\n';
 }
 std::cout<<"{\"success\":true,\"scope\":\"register FP32 AVX2 FMA capacity; not IQ2/IQ4 or model throughput\"}\n";std::cout.flush();if(!std::cout)throw std::runtime_error("output failure");return 0;
}catch(const std::exception&e){std::cerr<<"FMA failure: "<<e.what()<<'\n';return 1;}
