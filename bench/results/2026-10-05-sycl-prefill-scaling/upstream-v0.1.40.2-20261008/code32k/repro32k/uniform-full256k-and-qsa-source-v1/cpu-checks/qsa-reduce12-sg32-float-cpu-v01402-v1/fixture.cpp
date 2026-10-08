#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <coroutine>
#include <cstdint>
#include <cstdio>
#include <utility>
struct Task {
 struct promise_type {
  float argument=0, received=0, result=0;
  int mask=0; bool ready=false;
  Task get_return_object() { return Task{std::coroutine_handle<promise_type>::from_promise(*this)}; }
  std::suspend_always initial_suspend() { return {}; }
  std::suspend_always final_suspend() noexcept { return {}; }
  void return_value(float value) { result=value; }
  void unhandled_exception() { std::terminate(); }
 };
 std::coroutine_handle<promise_type> handle{};
 Task() = default;
 explicit Task(std::coroutine_handle<promise_type> h):handle(h) {}
 Task(Task&& other) noexcept:handle(std::exchange(other.handle,{})) {}
 Task& operator=(Task&& other) noexcept { if(handle)handle.destroy();handle=std::exchange(other.handle,{});return *this; }
 ~Task() { if(handle)handle.destroy(); }
};
struct XorExchange {
 float argument; int mask; Task::promise_type* promise=nullptr;
 bool await_ready() const { return false; }
 void await_suspend(std::coroutine_handle<Task::promise_type> h) {
  promise=&h.promise();assert(!promise->ready);promise->argument=argument;promise->mask=mask;promise->ready=true;
 }
 float await_resume() { assert(promise->ready);promise->ready=false;return promise->received; }
};
Task reduce12_sg32(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + (co_await XorExchange{hi ? part[i] : part[i + 8], 16});
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + (co_await XorExchange{hi ? r8[i] : r8[i + 4], 8});
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + (co_await XorExchange{hi ? r4[i] : r4[i + 2], 4});
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + (co_await XorExchange{hi2 ? r2[0] : r2[1], 2});
    co_return r1 + (co_await XorExchange{r1, 1});
}
uint64_t random_state=0x81aba1696d995e31ULL;
uint32_t randbits() { random_state^=random_state<<13;random_state^=random_state>>7;random_state^=random_state<<17;return uint32_t(random_state); }
uint32_t bits(float v) { return std::bit_cast<uint32_t>(v); }
int groups=0;
void check(float (&parts)[32][16]) {
 float expected[16][32]{};
 for(int head=0;head<16;++head) {
  for(int lane=0;lane<32;++lane) { expected[head][lane]=parts[lane][head];assert(std::isfinite(parts[lane][head])); }
  for(int offset=16;offset>0;offset>>=1) {
   float prior[32];for(int lane=0;lane<32;++lane)prior[lane]=expected[head][lane];
   for(int lane=0;lane<32;++lane) { expected[head][lane]=prior[lane]+prior[lane^offset];assert(std::isfinite(expected[head][lane])); }
  }
 }
 std::array<Task,32> tasks;
 for(int lane=0;lane<32;++lane)tasks[lane]=reduce12_sg32(parts[lane],lane);
 for(auto& task:tasks)task.handle.resume();
 int rounds=0;
 while(!tasks[0].handle.done()) {
  const int mask=tasks[0].handle.promise().mask;
  assert(mask>0 && mask<32);
  for(auto& task:tasks) { assert(!task.handle.done());assert(task.handle.promise().ready);assert(task.handle.promise().mask==mask); }
  for(int lane=0;lane<32;++lane)tasks[lane].handle.promise().received=tasks[lane^mask].handle.promise().argument;
  for(auto& task:tasks)task.handle.resume();
  ++rounds;assert(rounds<=16);
 }
 assert(rounds==16);
 for(int lane=0;lane<32;++lane) {
  assert(tasks[lane].handle.done());const int head=lane>>1;
  assert(bits(tasks[lane].handle.promise().result)==bits(expected[head][0]));
 }
 ++groups;
}
int main() {
 float parts[32][16]{};
 for(int pattern=0;pattern<8;++pattern) {
  for(int lane=0;lane<32;++lane)for(int head=0;head<16;++head) {
   switch(pattern) {
    case 0: parts[lane][head]=0.0f;break;
    case 1: parts[lane][head]=-0.0f;break;
    case 2: parts[lane][head]=(lane&1)?-0.0f:0.0f;break;
    case 3: parts[lane][head]=std::bit_cast<float>(uint32_t((lane&1)?0x80000001:1));break;
    case 4: parts[lane][head]=(lane&1)?-1.0f:1.0f;break;
    case 5: parts[lane][head]=(lane&1)?-0x1p80f:0x1p80f;break;
    case 6: parts[lane][head]=float((lane-16)*(head+1))*0.03125f;break;
    default:parts[lane][head]=std::bit_cast<float>((lane&1)?0x807fffffU:0x007fffffU);break;
   }
  }
  check(parts);
 }
 // Isolate every input location to verify head/lane routing, including padded heads.
 for(int source_head=0;source_head<16;++source_head)for(int source_lane=0;source_lane<32;++source_lane) {
  for(int lane=0;lane<32;++lane)for(int head=0;head<16;++head)parts[lane][head]=0.0f;
  parts[source_lane][source_head]=1.0f;check(parts);
 }
 for(int fixture=0;fixture<2048;++fixture) {
  for(int lane=0;lane<32;++lane)for(int head=0;head<16;++head) {
   uint32_t v=randbits();const uint32_t exponent=randbits()%220;
   parts[lane][head]=std::bit_cast<float>((v&0x807fffffU)|(exponent<<23));
  }
  check(parts);
 }
 std::printf("groups=%d outputs=%d exchanges_per_group=16 finite_intermediates=1\n",groups,groups*32);
}
