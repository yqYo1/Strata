"""Execute the actual reduction body with cooperative32-lane CPU exchanges.

An independent original XOR reduction supplies each head's FP32 reference.
This is a numerical/convergence unit check, not an inference benchmark.
"""
from pathlib import Path
import datetime, hashlib, json, re, subprocess, time

base = Path(__file__).parent
prepared_dir = base/'qsa-reduce12-sg32-v01402-source-v1'
prepared = json.loads((prepared_dir/'record.json').read_text())
source = Path(prepared['candidate_source'])
assert hashlib.sha256(source.read_bytes()).hexdigest() == prepared['candidate_source_sha256']
s = source.read_text()
start = s.index('__dpct_inline__ float reduce12_sg32(')
end = s.index('\n}', start)+2
actual_body = s[start:end]
host_body = actual_body.replace('__dpct_inline__ float reduce12_sg32', 'Task reduce12_sg32')
assert host_body.count('    const auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();') == 1
host_body = host_body.replace('    const auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();\n', '')
host_body, exchanges = re.subn(r'sycl::permute_group_by_xor\(sg, ([^\n]*?), (16|8|4|2|1)\)',
                              r'(co_await XorExchange{\1, \2})', host_body)
assert exchanges == 5 and host_body.count('    return ') == 1
host_body = host_body.replace('    return ', '    co_return ')
out = base/'qsa-reduce12-sg32-float-cpu-v01402-v2'
out.mkdir(mode=0o700)
prefix = r'''#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <coroutine>
#include <exception>
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
'''
suffix = r'''
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
'''
(out/'actual-helper-body.txt').write_text(actual_body+'\n')
(out/'fixture.cpp').write_text(prefix+host_body+suffix)
compiler = ['/usr/bin/g++','-std=c++20','-O2','-g','-fno-fast-math','-ffp-contract=off','-frounding-math',
            '-fsanitize=address,undefined','-fno-omit-frame-pointer',str(out/'fixture.cpp'),'-o',str(out/'fixture')]
started = time.monotonic()
record = {'active': True, 'passed': False, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'CPU mathematical/convergence check of actual helper body, cooperative coroutines replace only subgroup exchanges. Original per-head XOR FP32 reference is independent. Not a real SYCL subgroup, inference, transfer or performance test.',
          'compiler_argv': compiler, 'candidate_source_sha256': prepared['candidate_source_sha256'],
          'actual_helper_body_sha256': hashlib.sha256(actual_body.encode()).hexdigest(), 'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'gpu_used': False, 'compiled_engine': False, 'adopted': False}
(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
try:
    with (out/'compile.stdout').open('wb') as stdout, (out/'compile.stderr').open('wb') as stderr:
        subprocess.run(compiler, stdout=stdout, stderr=stderr, timeout=30, check=True)
    with (out/'run.stdout').open('wb') as stdout, (out/'run.stderr').open('wb') as stderr:
        subprocess.run([str(out/'fixture')], stdout=stdout, stderr=stderr, timeout=30, check=True)
    text = (out/'run.stdout').read_text()
    match = re.fullmatch(r'groups=(\d+) outputs=(\d+) exchanges_per_group=16 finite_intermediates=1\n',text)
    assert match and int(match[1]) == 2568 and int(match[2]) == 82176
    assert not (out/'run.stderr').stat().st_size
    record.update(passed=True, groups=int(match[1]), bitwise_equal_outputs=int(match[2]),
                  signed_zero_subnormal_cancellation_basis_random_passed=True, all32_lanes_and_masks_converged=True,
                  finite_intermediates_only=True, asan_ubsan_passed=True)
except BaseException as error:
    record['error'] = repr(error)
finally:
    record.update(active=False, elapsed_seconds=time.monotonic()-started,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
if not record['passed']:
    raise SystemExit(1)
