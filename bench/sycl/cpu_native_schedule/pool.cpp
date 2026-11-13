// Main-authored scheduling comparison using the owned pool fixture utilities.
// Repository LICENSE. CPU-only;unchanged production GU/quant/Down arithmetic.
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <fcntl.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace cpu = strata::kernels::cpu;
namespace gu = cpu::compact_gu;
using Clock = std::chrono::steady_clock;
constexpr int H = 2560, FF = 640, E = 64;
constexpr float guard = 123.f;
[[gnu::cold]] void need(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}
uint64_t number(const char* value) {
    const std::string s(value);
    need(!s.empty() && s.size() <= 20 && std::all_of(s.begin(), s.end(),
        [](char c) { return c >= '0' && c <= '9'; }), "unsigned decimal required");
    return std::stoull(s);
}
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
double cpu_seconds() {
    timespec t{}; need(!clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t), "CPU clock");
    return t.tv_sec + t.tv_nsec * 1e-9;
}
long peak_kib() {
    rusage value{}; need(!getrusage(RUSAGE_SELF, &value), "peak RSS"); return value.ru_maxrss;
}
struct File {
    int fd;
    explicit File(const char* name) : fd(open(name, O_RDONLY | O_CLOEXEC)) { need(fd >= 0, "model open"); }
    ~File() { close(fd); }
    void read(uint8_t* dst, size_t bytes, uint64_t offset) {
        struct stat s{};
        need(!fstat(fd, &s) && s.st_size >= 0 && offset <= uint64_t(s.st_size) &&
             bytes <= uint64_t(s.st_size) - offset, "model extent");
        while (bytes) {
            const auto n = pread(fd, dst, bytes, offset);
            need(n > 0, "short model read"); dst += n; bytes -= n; offset += n;
        }
    }
};
struct alignas(64) Output {
    std::array<float, H + 2> data{};
    Output() { data.fill(guard); }
    float* values() { return data.data() + 1; }
    void check() const {
        need(data.front() == guard && data.back() == guard, "output bounds");
        for (int i = 1; i <= H; ++i) need(std::isfinite(data[i]), "finite output");
    }
};
bool equal(const Output& a, const Output& b) {
    return std::memcmp(a.data.data(), b.data.data(), sizeof a.data) == 0;
}
std::set<int> tids() {
    std::set<int> result;
    for (const auto& p : std::filesystem::directory_iterator("/proc/self/task"))
        result.insert(std::stoi(p.path().filename().string()));
    return result;
}
int single_cpu(int tid) {
    cpu_set_t mask; CPU_ZERO(&mask);
    if (sched_getaffinity(tid, sizeof mask, &mask) || CPU_COUNT(&mask) != 1) return -1;
    for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &mask)) return c;
    return -1;
}
struct RestoreAffinity {
    cpu_set_t saved{};
    RestoreAffinity() {
        need(!sched_getaffinity(0, sizeof saved, &saved) && CPU_COUNT(&saved) == 12,
             "full initial twelve-CPU affinity required");
    }
    ~RestoreAffinity() { if (sched_setaffinity(0, sizeof saved, &saved)) std::terminate(); }
};
struct Pool {
    // Capture broad affinity before the pool detects topology. Pinning the
    // caller first would make new workers inherit/detect only CPU0.
    RestoreAffinity restore;
    std::set<int> before = tids();
    cpu::ExpertPool native;
    std::map<int, int> workers;
    explicit Pool(int n) : native(n, true, n == 5) {
        need(n == 1 || n == 5, "worker geometry");
        cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(0, &mask);
        need(!sched_setaffinity(0, sizeof mask, &mask) && single_cpu(getpid()) == 0, "host CPU0 pin");
        const auto deadline = Clock::now() + std::chrono::seconds(2);
        do {
            workers.clear(); std::set<int> cores;
            for (int tid : tids()) if (!before.contains(tid)) {
                const int c = single_cpu(tid); workers[tid] = c; cores.insert(c);
            }
            std::set<int> expected; for (int c = 1; c <= n; ++c) expected.insert(c);
            if (int(workers.size()) == n && cores == expected) { verify(); return; }
            std::this_thread::yield();
        } while (Clock::now() < deadline);
        need(false, "actual worker affinity admission");
    }
    void verify() const {
        need(single_cpu(getpid()) == 0 && native.workers() == int(workers.size()) &&
             native.host_works() == (workers.size() == 5), "host/pool geometry changed");
        auto expected = before;
        for (const auto& [tid, core] : workers) {
            expected.insert(tid); need(single_cpu(tid) == core, "worker affinity changed");
        }
        need(tids() == expected, "unexpected or retired native thread");
    }
};
struct Fixture {
    cpu::NativeFmt format{};
    std::vector<uint8_t> raw;
    std::array<std::vector<uint8_t>,4> acts;
    using PerInput = std::array<std::array<Output,4>,E>;
    std::vector<std::array<PerInput,4>> expected;
    double read_s=0,oracle_s=0;
    uint64_t input_hash=0;
    uint64_t hash() const {
        uint64_t h=1469598103934665603ULL;
        for (uint8_t b:raw) h=(h^b)*1099511628211ULL;
        for (const auto& a:acts) for (uint8_t b:a) h=(h^b)*1099511628211ULL;
        return h;
    }
    explicit Fixture(char** argv) {
        std::string error;const auto down=number(argv[2]);
        need((down==20||down==42)&&cpu::native_fmt(18,int(down),H,FF,format,error)&&
            format.bytes==number(argv[6]),"actual native format");
        const std::array<uint64_t,3> offsets{number(argv[3]),number(argv[4]),number(argv[5])};
        const std::array<size_t,3> sizes{format.up_off,format.up_off,format.bytes-format.down_off};
        const auto start=Clock::now();File file(argv[1]);raw.resize(size_t(E)*format.bytes);
        for (int e=0;e<E;++e) {
            auto* dst=raw.data()+size_t(e)*format.bytes;
            for (int r=0;r<3;++r) {need(offsets[r]<=UINT64_MAX-size_t(E)*sizes[r],"offset overflow");
                file.read(dst,sizes[r],offsets[r]+size_t(e)*sizes[r]);dst+=sizes[r];}
        }
        read_s=seconds(start);uint32_t seed=20261113;
        for (int t=0;t<4;++t) {
            std::array<float,H> input{};
            for (int i=0;i<H;++i) {seed=1664525u*seed+1013904223u;
                input[i]=t==1?0.f:(int(seed>>8)-8388608)*0x1p-23f;
                if(t==2) input[i]*=i%37==0?32.f:.002f;
                if(t==3&&i%256==0)input[i]=-16.f;}
            acts[t].resize(format.act_bytes);cpu::native_quant_act(format,input.data(),acts[t].data());
            need(acts[t].size()==10*sizeof(block_q8_K),"native IQ3 Q8_K layout");
            for(int b=0;b<10;++b){block_q8_K q;std::memcpy(&q,acts[t].data()+b*sizeof q,sizeof q);
                for(int8_t v:q.qs)need(v>=-127,"actual Q8_K range");}
        }
        input_hash=hash();const auto before=Clock::now();expected.resize(8);
        Pool oracle(1);std::array<cpu::ExpertJobMulti,E> jobs{};
        for(auto phase:{cpu::NativeGuPhase::Decode,cpu::NativeGuPhase::Prefill})
            for(int nt=1;nt<=4;++nt)for(int a=0;a<4;++a){
                const int mode=(phase==cpu::NativeGuPhase::Prefill?4:0)+nt-1;
                for(int e=0;e<E;++e){jobs[e].blob=raw.data()+size_t(e)*format.bytes;jobs[e].nt=nt;
                    for(int t=0;t<nt;++t){jobs[e].nact[t]=acts[(a+t)%4].data();jobs[e].out[t]=expected[mode][a][e][t].values();}}
                oracle.native.run_split_multi_native(format,jobs.data(),E,phase);
                for(int e=0;e<E;++e)for(int t=0;t<nt;++t)expected[mode][a][e][t].check();
            }
        oracle.verify();need(oracle.native.native_whole_calls==0,"independent reference route");oracle_s=seconds(before);
    }
    const Output& reference(cpu::NativeGuPhase phase,int nt,int act,int expert,int t)const{
        return expected[(phase==cpu::NativeGuPhase::Prefill?4:0)+nt-1][act][expert][t];
    }
    void guards()const{need(hash()==input_hash,"immutable raw weights and activations");}
};
int expert(int j,uint64_t round){return int((13u*j+7u*(j/E)+13u*round+3u)%E);}
int activation(int j,uint64_t round){return int((3u*j+j/5+round)%4);}
int nt_value(int j,uint64_t round,int policy){
    if(policy)return policy;const auto x=(11u*j+7u*round)%100;
    return x<75?1:x<92?2:x<97?3:4;
}
struct Case {
    Fixture& f;std::vector<cpu::ExpertJobMulti> jobs;std::vector<Output> out;
    uint64_t rows=0;
    Case(Fixture& fixture,int n):f(fixture),jobs(n),out(size_t(n)*4){}
    void bind(uint64_t round,bool stream,int policy){rows=0;
        for(int j=0;j<int(jobs.size());++j){auto& job=jobs[j];const int e=expert(j,stream?round:0);
            job.blob=f.raw.data()+size_t(e)*f.format.bytes;job.nt=nt_value(j,round,policy);rows+=job.nt;
            const int a=activation(j,round);
            for(int t=0;t<job.nt;++t){job.nact[t]=f.acts[(a+t)%4].data();job.out[t]=out[size_t(j)*4+t].values();}}
    }
    void run(Pool& pool,bool whole,cpu::NativeGuPhase phase){
        if(whole)pool.native.run_whole_multi_native(f.format,jobs.data(),int(jobs.size()),phase);
        else pool.native.run_split_multi_native(f.format,jobs.data(),int(jobs.size()),phase);
    }
    void check(uint64_t round,bool stream,cpu::NativeGuPhase phase)const{
        for(int j=0;j<int(jobs.size());++j)for(int t=0;t<jobs[j].nt;++t){const auto& v=out[size_t(j)*4+t];v.check();
            need(equal(v,f.reference(phase,jobs[j].nt,activation(j,round),expert(j,stream?round:0),t)),"independent row-split oracle bits");}
    }
    void inactive()const{
        for(int j=0;j<int(jobs.size());++j)for(int t=jobs[j].nt;t<4;++t)
            for(float v:out[size_t(j)*4+t].data)need(v==guard,"inactive output untouched");
    }
};
void numerical(Fixture& f){uint64_t cases=0,whole_calls=0,active_rows=0;
    need(cpu::ExpertPool::kMaxSplitMulti>=36,"whole expert buffer capacity");
    for(int workers:{1,5}){Pool pool(workers);
        pool.native.run_whole_multi_native(f.format,nullptr,0,cpu::NativeGuPhase::Decode);
        need(pool.native.native_whole_calls==0,"empty call stays empty");
        for(int n:{1,6,7,12,24,36,37,97,193})for(int policy:{0,1,2,3,4})
            for(auto phase:{cpu::NativeGuPhase::Decode,cpu::NativeGuPhase::Prefill}){
                Case ref(f,n),candidate(f,n);const uint64_t round=cases+11;
                ref.bind(round,true,policy);candidate.bind(round,true,policy);
                const auto count=pool.native.native_whole_calls;
                const auto a=cpu::iq3_gu_audit_snapshot();need(a.enabled&&!a.overflow,"native audit enabled");
                ref.run(pool,false,phase);const auto b=cpu::iq3_gu_audit_snapshot();candidate.run(pool,true,phase);
                const auto c=cpu::iq3_gu_audit_snapshot();uint64_t nt1_rows=0;
                for(const auto& job:ref.jobs)if(job.nt==1)nt1_rows+=FF;
                const auto paired=phase==cpu::NativeGuPhase::Decode?nt1_rows:0;
                need(!c.overflow&&b.nt1_rows-a.nt1_rows==nt1_rows&&c.nt1_rows-b.nt1_rows==nt1_rows&&
                    b.pair_rows-a.pair_rows==paired&&c.pair_rows-b.pair_rows==paired,"native phase/NT dispatch accounting");
                const bool eligible=phase==cpu::NativeGuPhase::Decode&&n>=7&&n<=36;
                need(pool.native.native_whole_calls-count==uint64_t(eligible),"actual schedule dispatch");
                ref.check(round,true,phase);candidate.check(round,true,phase);ref.inactive();candidate.inactive();
                for(size_t i=0;i<ref.out.size();++i)need(equal(ref.out[i],candidate.out[i]),"whole pool parity");
                active_rows+=ref.rows;whole_calls+=eligible;++cases;
            }
        // A changed input must break the oracle;do not validate only finiteness.
        Case bad(f,24);bad.bind(0,true,1);for(auto& j:bad.jobs)j.nact[0]=f.acts[1].data();bad.run(pool,true,cpu::NativeGuPhase::Decode);
        bool detected=false;for(int j=0;j<24;++j){bad.out[size_t(j)*4].check();if(!equal(bad.out[size_t(j)*4],f.reference(cpu::NativeGuPhase::Decode,1,activation(j,0),expert(j,0),0)))detected=true;}
        need(detected,"wrong activation control detected");pool.verify();
    }
    f.guards();std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"whole_calls\":"<<whole_calls
        <<",\"token_expert_rows\":"<<active_rows<<",\"wrong_activation_controls\":2,\"guards\":true,\"inputs_immutable\":true,\"performance_qualified\":false}\n";
}
struct Sample{double wall,cpu,gu,quant,down,whole;uint64_t calls,jobs,rows,selected;};
Sample batch(Case& work,Pool& pool,bool whole,bool stream,int policy,uint64_t rounds){
    pool.verify();const auto g=pool.native.ms_multi_gu,q=pool.native.ms_multi_q,d=pool.native.ms_multi_down,w=pool.native.ms_multi_whole;
    const auto selected=pool.native.native_whole_calls;const double c=cpu_seconds();const auto start=Clock::now();uint64_t rows=0;
    for(uint64_t i=0;i<rounds;++i){work.bind(i,stream,policy);work.run(pool,whole,cpu::NativeGuPhase::Decode);rows+=work.rows;}
    Sample s{seconds(start),cpu_seconds()-c,(pool.native.ms_multi_gu-g)*.001,(pool.native.ms_multi_q-q)*.001,
        (pool.native.ms_multi_down-d)*.001,(pool.native.ms_multi_whole-w)*.001,rounds,rounds*work.jobs.size(),rows,pool.native.native_whole_calls-selected};
    need(s.selected==(whole?rounds:0)&&s.gu>=0&&s.quant>=0&&s.down>=0&&s.whole>=0&&s.gu+s.quant+s.down+s.whole<=s.wall+1e-6,"timed route/service accounting");
    work.check(rounds-1,stream,cpu::NativeGuPhase::Decode);pool.verify();return s;
}
int main(int argc,char**argv){try{
    need(argc==8,"pool SHARD DOWN GATE UP DOWN_OFFSET BLOB_BYTES check|reference|whole");const std::string arm(argv[7]);
    need(arm=="check"||arm=="reference"||arm=="whole","arm");Fixture f(argv);std::cout<<std::setprecision(17);
    if(arm=="check"){numerical(f);return 0;}need(!cpu::iq3_gu_audit_snapshot().enabled,"clean audit setting");
    const auto now=Clock::now();for(int i=0;i<10000;++i){auto t=Clock::now().time_since_epoch().count();asm volatile(""::"g"(t):"memory");}
    const double timer=2*seconds(now)/10000;need(timer<1e-4,"timer overhead");
    std::cout<<"{\"passed\":true,\"scope\":\"CPU_schedule_only\",\"arm\":\""<<arm<<"\",\"down\":"<<f.format.d_type
        <<",\"read_s\":"<<f.read_s<<",\"oracle_setup_s\":"<<f.oracle_s<<",\"timer_pair_s\":"<<timer<<",\"raw_bytes_per_job\":"<<f.format.bytes<<",\"cells\":[";
    Pool pool(5);bool comma=false;
    for(int n:{12,24,36})for(int policy:{0,1})for(bool stream:{false,true}){
        if(comma)std::cout<<',';comma=true;Case work(f,n);
        std::cout<<"{\"workers\":5,\"host_works\":true,\"jobs_per_call\":"<<n<<",\"NT_policy\":\""<<(policy?"NT1":"observed_mix")<<"\",\"pattern\":\""<<(stream?"stream64":"reuse")<<"\",\"calibration\":[";
        uint64_t rounds=1;bool sep=false;for(;;){const auto s=batch(work,pool,arm=="whole",stream,policy,rounds);if(sep)std::cout<<',';sep=true;
            std::cout<<"{\"rounds\":"<<rounds<<",\"wall_s\":"<<s.wall<<'}';if(s.wall>=.5)break;need(rounds<65536,"calibration floor");rounds*=2;}
        std::cout<<"],\"rounds\":"<<rounds<<",\"samples\":[";
        for(int i=0;i<3;++i){const auto s=batch(work,pool,arm=="whole",stream,policy,rounds);need(s.wall>=.25&&timer/s.wall<.001,"timing floor");if(i)std::cout<<',';
            std::cout<<"{\"wall_s\":"<<s.wall<<",\"process_cpu_s\":"<<s.cpu<<",\"calls\":"<<s.calls<<",\"jobs\":"<<s.jobs<<",\"token_expert_rows\":"<<s.rows
                <<",\"whole_calls\":"<<s.selected<<",\"pool_gu_s\":"<<s.gu<<",\"pool_quant_s\":"<<s.quant<<",\"pool_down_s\":"<<s.down<<",\"pool_whole_s\":"<<s.whole<<'}';}
        std::cout<<"]}";
    }
    pool.verify();f.guards();std::cout<<"],\"peak_kib\":"<<peak_kib()<<",\"inputs_immutable\":true,\"model_qualified\":false}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
