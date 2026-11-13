// Main-authored mixed-NT cache comparison, extending the owned64-expert
// fixture from5660b2f41. Repository LICENSE;no external implementation reuse.
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
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <utility>
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
static_assert(sizeof(cpu::ExpertJobMulti{}.nact)/sizeof(void*)>=8);
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
        bool written=false;
        for(int i=1;i<=H;++i){uint32_t bits;std::memcpy(&bits,&data[i],sizeof bits);
            need((bits&0x7f800000u)!=0x7f800000u,"finite output bits");written|=data[i]!=guard;}
        need(written,"active output really written");
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
    std::shared_ptr<const gu::Cache> cache;
    double read_s=0,oracle_s=0,encode_s=0;
    uint64_t input_hash=0;
    uint64_t hash() const {
        uint64_t h=1469598103934665603ULL;
        for (uint8_t b:raw) h=(h^b)*1099511628211ULL;
        for (const auto& a:acts) for (uint8_t b:a) h=(h^b)*1099511628211ULL;
        return h;
    }
    explicit Fixture(char** argv,bool encode) {
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
        read_s=seconds(start);
        if(encode){
            const auto begin=Clock::now();
            const gu::LayerView view{0,E,format.bytes,raw.size(),raw.data(),format};
            cache=gu::prepare(1,E,128u<<20,{&view,1},error);
            need(cache&&cache->prepared_entries()==E,"complete immutable cache");
            encode_s=seconds(begin);
        }
        uint32_t seed=20261113;
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
    void guards()const{
        need(hash()==input_hash,"immutable raw weights and activations");
        if(cache)for(int e=0;e<E;++e)need(cache->lookup({0,size_t(e)},format,
            cpu::NativeGuPhase::Decode,1).guards_intact(),"immutable cache guards");
    }
};

struct Shape { std::string name; std::vector<int> nt; };
Shape counted(const char* name,std::array<int,4> counts){
    Shape s{name,{}};for(int t=0;t<4;++t)for(int i=0;i<counts[t];++i)s.nt.push_back(t+1);return s;
}
Shape uniform(const char* name,int n,int nt=1){return {name,std::vector<int>(n,nt)};}
std::vector<Shape> observed(int down){
    if(down==20)return {counted("minimum7",{7,0,0,0}),counted("mixed25",{20,4,0,1}),
        counted("maximum33",{33,0,0,0})};
    return {counted("minimum9",{9,0,0,0}),counted("mixed13",{7,2,2,2}),
        counted("maximum33",{31,2,0,0})};
}
int expert(int j,uint64_t round){return int((13u*j+13u*round+3u)%E);}
int activation(int j,uint64_t round){return int((3u*j+j/5+round)%4);}
struct Case {
    Fixture& f;Shape shape;std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<gu::Lease> views;std::vector<Output> out;
    bool has_views=false;uint64_t rows=0,bound_nt1=0;
    Case(Fixture& fixture,Shape s):f(fixture),shape(std::move(s)),jobs(shape.nt.size()),
        views(jobs.size()),out(jobs.size()*8){
        need(!jobs.empty()&&std::gcd(17,int(jobs.size()))==1,"complete NTpermutation");
        for(int n:shape.nt)need(n>=1&&n<=8,"native activation capacity");
    }
    void bind(uint64_t round,bool stream,bool compact,bool misses=false,
              bool wrong=false,bool force_all=false){
        has_views=compact;rows=bound_nt1=0;
        for(int j=0;j<int(jobs.size());++j){auto& job=jobs[j];
            const int e=expert(j,stream?round:0),a=activation(j,round);
            job.blob=f.raw.data()+size_t(e)*f.format.bytes;
            job.nt=shape.nt[(17*size_t(j)+(round&3))%jobs.size()];rows+=job.nt;
            for(int t=0;t<job.nt;++t){job.nact[t]=f.acts[(a+t)%4].data();
                job.out[t]=out[size_t(j)*8+t].values();}
            views[j]={};
            if(compact&&(force_all||job.nt==1)&&(!misses||(size_t(j)+round)%5)){
                const int key=wrong?expert((j+1)%int(jobs.size()),stream?round:0):e;
                views[j]=f.cache->lookup({0,size_t(key)},f.format,cpu::NativeGuPhase::Decode,1);
                need(bool(views[j]),"complete prepared coordinate binding");
            }
            if(job.nt==1&&views[j])++bound_nt1;
        }
    }
    uint64_t selected_rows(cpu::NativeGuPhase phase,gu::BatchPolicy policy)const{
        if(phase!=cpu::NativeGuPhase::Decode)return 0;
        if(policy==gu::BatchPolicy::SmallJobs){if(jobs.size()>6)return 0;}
        else if(policy==gu::BatchPolicy::MixedNt){
            if(jobs.size()>40||rows>40)return 0;
            for(auto& j:jobs)if(j.nt>4)return 0;
        }else return 0;
        return bound_nt1*FF;
    }
    void run(Pool& pool,cpu::NativeGuPhase phase,gu::BatchPolicy policy){
        pool.native.run_multi_native(f.format,jobs.data(),int(jobs.size()),phase,false,
            has_views?std::span<const gu::Lease>(views):std::span<const gu::Lease>{},policy);
    }
    void check(uint64_t round,bool stream,cpu::NativeGuPhase phase,bool fresh=true)const{
        for(int j=0;j<int(jobs.size());++j){
            for(int t=0;t<jobs[j].nt;++t){const auto& v=out[size_t(j)*8+t];v.check();
                if(jobs[j].nt<=4)need(equal(v,f.reference(phase,jobs[j].nt,
                    activation(j,round),expert(j,stream?round:0),t)),"independent native oracle bits");}
            if(fresh)for(int t=jobs[j].nt;t<8;++t)for(float v:out[size_t(j)*8+t].data)
                need(v==guard,"inactive output slots");
        }
    }
};
struct Counts{uint64_t native_rows,selected_rows;};
Counts parity(Fixture& f,Pool& pool,const Shape& shape,uint64_t round,
              cpu::NativeGuPhase phase,bool misses,gu::BatchPolicy policy){
    Case ref(f,shape),candidate(f,shape);ref.bind(round,true,false);
    candidate.bind(round,true,true,misses,false,true);
    uint64_t rows=0;for(auto& j:ref.jobs)if(j.nt==1)rows+=FF;
    const uint64_t selected=candidate.selected_rows(phase,policy);
    const auto a=cpu::iq3_gu_audit_snapshot();need(a.enabled&&!a.overflow,"actual native audit");
    ref.run(pool,phase,gu::BatchPolicy::SmallJobs);const auto b=cpu::iq3_gu_audit_snapshot();
    candidate.run(pool,phase,policy);const auto c=cpu::iq3_gu_audit_snapshot();
    need(!c.overflow&&b.nt1_rows-a.nt1_rows==rows&&c.nt1_rows-b.nt1_rows==rows-selected&&
        b.pair_rows-a.pair_rows==(phase==cpu::NativeGuPhase::Decode?rows:0)&&
        c.pair_rows-b.pair_rows==(phase==cpu::NativeGuPhase::Decode?rows-selected:0),
        "actual cached versus native NT1route");
    for(size_t j=0;j<ref.jobs.size();++j)for(int t=0;t<ref.jobs[j].nt;++t)
        need(equal(ref.out[j*8+t],candidate.out[j*8+t]),"mixed pool exact parity");
    ref.check(round,true,phase);candidate.check(round,true,phase);
    need(pool.native.native_whole_calls==0,"no whole-expert experiment");
    return {rows,selected};
}
void numerical(Fixture& f){
    uint64_t cases=0,nt1_rows=0,cached_rows=0,special=0;
    auto shapes=observed(f.format.d_type);
    for(int n:{1,3,6})shapes.push_back(uniform("legacy",n));
    for(int nt:{5,6,7,8}){auto s=uniform("wide_native",7);s.nt.back()=nt;shapes.push_back(s);}
    auto over=uniform("over_volume",13,4);over.nt[0]=1;shapes.push_back(over);
    shapes.push_back(uniform("over_jobs",41));shapes.push_back(uniform("boundary40",40));
    shapes.push_back(uniform("all_NT2",9,2));need(shapes.size()==14,"numeric shape coverage");
    for(int workers:{1,5}){Pool pool(workers);
        auto compare=[&](const Shape& s,gu::BatchPolicy policy){
            for(auto phase:{cpu::NativeGuPhase::Decode,cpu::NativeGuPhase::Prefill})
                for(bool misses:{false,true}){const auto c=parity(f,pool,s,cases+11,phase,misses,policy);
                    ++cases;nt1_rows+=c.native_rows;cached_rows+=c.selected_rows;}
        };
        for(const auto& s:shapes)compare(s,gu::BatchPolicy::MixedNt);
        for(int n:{1,3,6,7,33})compare(uniform("legacy_default",n),gu::BatchPolicy::SmallJobs);
        Case wrong(f,uniform("wrong_expert",7));wrong.bind(9,true,true,false,true,true);
        const auto a=cpu::iq3_gu_audit_snapshot();wrong.run(pool,cpu::NativeGuPhase::Decode,gu::BatchPolicy::MixedNt);
        need(cpu::iq3_gu_audit_snapshot().nt1_rows==a.nt1_rows,"negative control must use cached math");
        bool detected=false;for(int j=0;j<7;++j){wrong.out[size_t(j)*8].check();
            detected|=!equal(wrong.out[size_t(j)*8],f.reference(cpu::NativeGuPhase::Decode,1,
                activation(j,9),expert(j,9),0));}need(detected,"wrong-expert negative control");++special;
        Case valid(f,uniform("span",7));valid.bind(17,true,true);
        bool rejected=false;const auto before=cpu::iq3_gu_audit_snapshot();
        try{pool.native.run_multi_native(f.format,valid.jobs.data(),7,cpu::NativeGuPhase::Decode,
            false,{valid.views.data(),6},gu::BatchPolicy::MixedNt);}catch(const std::invalid_argument&){rejected=true;}
        need(rejected&&cpu::iq3_gu_audit_snapshot().nt1_rows==before.nt1_rows,"short span before publication");
        valid.run(pool,cpu::NativeGuPhase::Decode,gu::BatchPolicy::MixedNt);
        valid.check(17,true,cpu::NativeGuPhase::Decode);++special;
        rejected=false;
        try{pool.native.run_multi_native(f.format,nullptr,7,cpu::NativeGuPhase::Decode,
            false,valid.views,gu::BatchPolicy::MixedNt);}catch(const std::invalid_argument&){rejected=true;}
        need(rejected,"null mixed batch before publication");++special;
        auto temporary=std::make_shared<gu::Cache>(1,E,16u<<20);
        Case lifetime(f,uniform("retirement",7));lifetime.bind(7,true,false);
        for(int j=0;j<7;++j){const size_t e=size_t(expert(j,7));need(temporary->prepare({0,e},f.format,
            {f.raw.data()+e*f.format.bytes,f.format.down_off}),"temporary cache prepare");}
        temporary->freeze();for(int j=0;j<7;++j)lifetime.views[j]=temporary->lookup({0,size_t(expert(j,7))},
            f.format,cpu::NativeGuPhase::Decode,1);
        std::weak_ptr<const gu::Cache> weak=temporary;temporary.reset();need(weak.expired(),"cache owner really retired");
        lifetime.has_views=true;const auto prior=cpu::iq3_gu_audit_snapshot();
        lifetime.run(pool,cpu::NativeGuPhase::Decode,gu::BatchPolicy::MixedNt);
        lifetime.check(7,true,cpu::NativeGuPhase::Decode);
        need(cpu::iq3_gu_audit_snapshot().nt1_rows==prior.nt1_rows,"retired leases actually used");
        for(auto& lease:lifetime.views)need(lease.guards_intact(),"retired entry bounds");++special;
        parity(f,pool,uniform("invalid_policy",7),29,cpu::NativeGuPhase::Decode,false,
            static_cast<gu::BatchPolicy>(255));++special;
        const auto empty=cpu::iq3_gu_audit_snapshot();
        for(int n:{-1,0})pool.native.run_multi_native(f.format,nullptr,n,cpu::NativeGuPhase::Decode,
            false,{},gu::BatchPolicy::MixedNt);
        need(cpu::iq3_gu_audit_snapshot().nt1_rows==empty.nt1_rows,"empty call");++special;
        auto mismatch=f.format;mismatch.gu_type=22;
        need(!f.cache->lookup({0,0},f.format,cpu::NativeGuPhase::Decode,1).matches(mismatch)&&
            !f.cache->lookup({1,0},f.format,cpu::NativeGuPhase::Decode,1),"format/key refusal");++special;
        pool.verify();need(pool.native.native_whole_calls==0,"no unqualified whole route");
    }
    f.guards();need(cases==152&&special==14,"complete numeric controls");
    std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"nt1_rows\":"<<nt1_rows
        <<",\"compact_rows\":"<<cached_rows<<",\"special_controls\":"<<special
        <<",\"whole_calls\":0,\"inputs_immutable\":true,\"guards\":true,\"performance_qualified\":false}\n";
}
struct Sample{double wall,cpu,gu,quant,down;uint64_t calls,jobs,rows,bound;};
Sample batch(Case& work,Pool& pool,bool compact,bool stream,uint64_t rounds){
    pool.verify();const auto gu0=pool.native.ms_multi_gu,q0=pool.native.ms_multi_q,d0=pool.native.ms_multi_down;
    const auto whole0=pool.native.native_whole_calls;const double cpu0=cpu_seconds();const auto start=Clock::now();
    uint64_t rows=0,bound=0;
    for(uint64_t i=0;i<rounds;++i){work.bind(i,stream,compact);rows+=work.rows;bound+=work.bound_nt1;
        work.run(pool,cpu::NativeGuPhase::Decode,compact?gu::BatchPolicy::MixedNt:gu::BatchPolicy::SmallJobs);}
    Sample s{seconds(start),cpu_seconds()-cpu0,(pool.native.ms_multi_gu-gu0)*.001,
        (pool.native.ms_multi_q-q0)*.001,(pool.native.ms_multi_down-d0)*.001,
        rounds,rounds*work.jobs.size(),rows,bound};
    need(pool.native.native_whole_calls==whole0&&s.gu>=0&&s.quant>=0&&s.down>=0&&
        s.gu+s.quant+s.down<=s.wall+1e-6,"timed production-entry stage conservation");
    work.check(rounds-1,stream,cpu::NativeGuPhase::Decode,false);pool.verify();return s;
}
int main(int argc,char** argv){try{
    need(argc==8,"SHARD DOWN GATE UP DOWN_OFFSET BLOB_BYTES check|reference|compact");
    const std::string arm=argv[7];need(arm=="check"||arm=="reference"||arm=="compact","arm");
    Fixture f(argv,arm!="reference");std::cout<<std::setprecision(17);
    if(arm=="check"){numerical(f);return 0;}
    need(!cpu::iq3_gu_audit_snapshot().enabled,"clean timing audit disabled");
    Pool pool(5);
    const auto start=Clock::now();for(int i=0;i<10000;++i){const auto v=Clock::now().time_since_epoch().count();
        const double c=cpu_seconds();asm volatile(""::"g"(v),"g"(c):"memory");}
    const double observer=4*seconds(start)/10000;need(observer>0&&observer<1e-4,"observer overhead");
    std::cout<<"{\"scope\":\"CPU_mixed_NT_cache\",\"arm\":\""<<arm<<"\",\"down\":"<<f.format.d_type
        <<",\"read_s\":"<<f.read_s<<",\"encode_s\":"<<f.encode_s<<",\"oracle_s\":"<<f.oracle_s
        <<",\"known_view_bytes\":"<<(f.cache?f.cache->known_bytes():0)<<",\"observer_estimate_s\":"<<observer
        <<",\"cells\":[";bool comma=false;
    for(const auto& shape:observed(f.format.d_type))for(bool stream:{false,true}){
        if(comma)std::cout<<',';comma=true;Case work(f,shape);
        std::array<int,4> nt{};uint64_t rows=0;for(int n:shape.nt){++nt[n-1];rows+=n;}
        need(rows<=40&&shape.nt.size()<=40,"physical spec4/topk10corpus");
        std::cout<<"{\"shape\":\""<<shape.name<<"\",\"jobs_per_call\":"<<shape.nt.size()
            <<",\"NT_counts\":["<<nt[0]<<','<<nt[1]<<','<<nt[2]<<','<<nt[3]
            <<"],\"rows_per_call\":"<<rows<<",\"pattern\":\""<<(stream?"stream64":"reuse")
            <<"\",\"workers\":5,\"host_works\":true,\"calibration\":[";
        uint64_t rounds=64;bool sep=false;
        for(;;){const auto s=batch(work,pool,arm=="compact",stream,rounds);if(sep)std::cout<<',';sep=true;
            std::cout<<"{\"rounds\":"<<rounds<<",\"wall_s\":"<<s.wall<<'}';
            if(s.wall>=.5)break;need(rounds<65536,"calibration floor");rounds*=2;}
        std::cout<<"],\"rounds\":"<<rounds<<",\"samples\":[";
        for(int i=0;i<3;++i){const auto s=batch(work,pool,arm=="compact",stream,rounds);
            need(s.wall>=.25&&observer/s.wall<.001&&s.rows==rounds*rows&&
                s.bound==(arm=="compact"?rounds*uint64_t(nt[0]):0),"duration/work/binding eligibility");
            if(i)std::cout<<',';std::cout<<"{\"wall_s\":"<<s.wall<<",\"process_cpu_s\":"<<s.cpu
                <<",\"calls\":"<<s.calls<<",\"jobs\":"<<s.jobs<<",\"token_expert_rows\":"<<s.rows
                <<",\"bound_NT1_leases\":"<<s.bound<<",\"pool_gu_s\":"<<s.gu
                <<",\"pool_quant_s\":"<<s.quant<<",\"pool_down_s\":"<<s.down<<'}';}
        std::cout<<"]}";
    }
    f.guards();std::cout<<"],\"peak_kib\":"<<peak_kib()<<",\"passed\":true,\"inputs_immutable\":true,"
        "\"binding_limit\":\"Timed producer bindings;actual kernel route is audited in separate numeric controls. "
        "Performance cases have immutable complete leases and deterministic valid phase/NT/format guards.\","
        "\"model_qualified\":false}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
