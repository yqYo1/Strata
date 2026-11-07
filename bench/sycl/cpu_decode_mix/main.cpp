// Main-authored CPU component fixtures; raw references link frozen production code.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "compact_iq3xxs.hpp"
#include <algorithm>
#include <atomic>
#include <barrier>
#include <functional>
#include <thread>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
constexpr int batches = 100, repetitions = 5;
constexpr uint32_t unused = 0x7fc12345;
void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
double process_cpu() {
    rusage r{}; need(getrusage(RUSAGE_SELF,&r)==0,"rusage");
    return r.ru_utime.tv_sec+r.ru_stime.tv_sec+(r.ru_utime.tv_usec+r.ru_stime.tv_usec)*1e-6;
}
uint64_t number(const char* value) {
    const std::string s(value); need(!s.empty() && s.size()<=18 &&
        std::all_of(s.begin(),s.end(),[](char c){return c>='0'&&c<='9';}),"unsigned decimal required");
    return std::stoull(s);
}
struct Buffer {
    uint8_t* p=nullptr; size_t bytes;
    explicit Buffer(size_t n):bytes(n) {
        need(n && n<=4ull<<20,"buffer extent");
        need(posix_memalign(reinterpret_cast<void**>(&p),64,n+128)==0,"allocation");
        std::memset(p,0xA7,n+128);
    }
    ~Buffer(){std::free(p);}
    Buffer(const Buffer&)=delete;
    uint8_t* data() const {return p+64;}
    void check() const {
        for(int i=0;i<64;++i) need(p[i]==0xA7 && p[64+bytes+i]==0xA7,"canary");
    }
};
void read_slice(int fd,uint8_t* out,size_t n,uint64_t offset) {
    struct stat st{}; need(fstat(fd,&st)==0 && offset<=uint64_t(st.st_size) &&
        n<=uint64_t(st.st_size)-offset,"model slice bounds");
    while(n){const auto k=pread(fd,out,n,offset);need(k>0,"model slice read");out+=k;n-=k;offset+=k;}
}
std::vector<int> schedule(int group,const std::array<uint64_t,4>& observed) {
    const uint64_t sum=std::accumulate(observed.begin(),observed.end(),uint64_t{0});
    need(sum>0 && sum<=4000000000ull,"histogram total");
    const int n=batches*group; std::array<int,4> count{};std::array<uint64_t,4> remainder{};
    int assigned=0;
    for(int i=0;i<4;++i){count[i]=n*observed[i]/sum;remainder[i]=n*observed[i]%sum;assigned+=count[i];}
    for(;assigned<n;++assigned){int at=std::max_element(remainder.begin(),remainder.end())-remainder.begin();++count[at];remainder[at]=0;}
    std::vector<int> result;
    for(int i=0;i<4;++i)result.insert(result.end(),count[i],i+1);
    std::mt19937 rng(20261022);std::shuffle(result.begin(),result.end(),rng);
    return result;
}
void verify_affinity(const cpu::CpuTopology& topology,int workers) {
    std::vector<int> expected{topology.host_core};
    expected.insert(expected.end(),topology.worker_cores.begin(),topology.worker_cores.begin()+workers);
    std::vector<int> found(expected.size());int tasks=0;
    for(const auto& entry:std::filesystem::directory_iterator("/proc/self/task")) {
        need(++tasks<=128,"unexpected task extent");
        cpu_set_t mask;CPU_ZERO(&mask);
        const auto tid=number(entry.path().filename().c_str());
        need(sched_getaffinity(tid,sizeof mask,&mask)==0,"read actual thread affinity");
        for(size_t i=0;i<expected.size();++i)
            if(CPU_COUNT(&mask)==1 && CPU_ISSET(expected[i],&mask))++found[i];
    }
    need(std::all_of(found.begin(),found.end(),[](int n){return n==1;}),"actual pool affinity differs");
}

class GuTeam {
    int compute_;
    std::function<void(int,int)> work_;
    std::barrier<> phase_;
    std::vector<std::thread> threads_;
    std::atomic<bool> stop_{false},failed_{false};
    void invoke(int id) noexcept {
        try { if(!failed_.load())work_(id,compute_); }
        catch(...) { failed_.store(true); }
    }
public:
    GuTeam(int compute,const cpu::CpuTopology& topo,std::function<void(int,int)> fn):
        compute_(compute),work_(std::move(fn)),phase_(compute==6?6:2) {
        need(compute==1 || compute==6,"GU compute threads");
        const int wanted=compute==6?5:1;threads_.reserve(wanted);
        try {
            for(int i=0;i<wanted;++i)threads_.emplace_back([&,i] {
                cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(topo.worker_cores[i],&mask);
                if(sched_setaffinity(0,sizeof mask,&mask)!=0)failed_.store(true);
                for(;;) {
                    phase_.arrive_and_wait();if(stop_.load())break;
                    invoke(compute_==6?i+1:0);phase_.arrive_and_wait();
                }
            });
        } catch(...) {
            stop_.store(true);
            for(size_t i=threads_.size();i<size_t(wanted);++i)phase_.arrive_and_drop();
            phase_.arrive_and_wait();for(auto& t:threads_)t.join();throw;
        }
    }
    ~GuTeam() { stop_.store(true);phase_.arrive_and_wait();for(auto& t:threads_)t.join(); }
    GuTeam(const GuTeam&)=delete;
    void run() {
        phase_.arrive_and_wait();if(compute_==6)invoke(0);phase_.arrive_and_wait();
        need(!failed_.load(),"GU team work/affinity failure");
    }
};
struct Work {
    struct alignas(64) GuCounts { std::array<uint64_t,4> values{}; };
    std::array<GuCounts,6> gu_counts{};
    std::vector<std::vector<compact_iq3xxs::Block>> gu_views;
    int gu_offset=0;bool gu_compact=false;
    cpu::NativeFmt fmt;
    int group,count;
    std::vector<std::unique_ptr<Buffer>> weights,outputs;
    std::array<std::unique_ptr<Buffer>,4> activations;
    std::array<std::array<float,cpu::H>,4> input;
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<int> nt;
    Work(char** a,int g,const std::vector<int>& sequence):group(g),nt(sequence) {
        std::string err;const auto gu=number(a[2]),dn=number(a[3]);need(gu<=42 && dn<=42,"format ID");
        const bool valid=cpu::native_fmt(gu,dn,cpu::H,cpu::FF,fmt,err);need(valid,err.c_str());
        need(fmt.bytes==number(a[7]) && fmt.gu_type!=42,"native geometry");
        count=group*int(((128ull<<20)+group*fmt.bytes-1)/(group*fmt.bytes));
        need(count>=group && count<=256 && count*fmt.bytes>=128ull<<20,"stream extent");
        const int fd=open(a[1],O_RDONLY|O_CLOEXEC);need(fd>=0,"open model");
        for(int e=0;e<count;++e){
            auto w=std::make_unique<Buffer>(fmt.bytes);
            read_slice(fd,w->data(),fmt.up_off,number(a[4])+e*fmt.up_off);
            read_slice(fd,w->data()+fmt.up_off,fmt.up_off,number(a[5])+e*fmt.up_off);
            const auto dn=fmt.bytes-fmt.down_off;
            read_slice(fd,w->data()+fmt.down_off,dn,number(a[6])+e*dn);
            weights.push_back(std::move(w));outputs.push_back(std::make_unique<Buffer>(4*cpu::H*sizeof(float)));
        }
        close(fd);uint32_t seed=20261022;
        for(int t=0;t<4;++t){
            for(auto& x:input[t]){seed=1664525u*seed+1013904223u;x=(int(seed>>8)-8388608)*0x1p-23f;}
            activations[t]=std::make_unique<Buffer>(fmt.act_bytes);
            cpu::native_quant_act(fmt,input[t].data(),activations[t]->data());
        }
        jobs.resize(count);
        for(int e=0;e<count;++e){
            jobs[e].blob=weights[e]->data();
            for(int t=0;t<4;++t){jobs[e].nact[t]=activations[t]->data();jobs[e].out[t]=reinterpret_cast<float*>(outputs[e]->data())+t*cpu::H;}
        }
    }
    void run(cpu::ExpertPool& pool,bool stream) {
        for(int b=0;b<batches;++b){
            const int offset=stream?(b%(count/group))*group:0;
            for(int e=0;e<group;++e)jobs[offset+e].nt=nt[b*group+e];
            pool.run_split_multi_native(fmt,jobs.data()+offset,group,cpu::NativeGuPhase::Decode);
        }
    }

    void gu_setup() {
        need(fmt.gu_type==18 && fmt.gu_row==980,"GU view format");
        const auto begin=Clock::now();const double c0=process_cpu();
        gu_views.resize(count);
        constexpr size_t n=compact_iq3xxs::projection_blocks;
        for(int e=0;e<count;++e) {
            auto& v=gu_views[e];v.resize(2*n+2);std::memset(v.data(),0xA7,v.size()*sizeof(v[0]));
            for(size_t b=0;b<n;++b) {
                need(compact_iq3xxs::encode_block(std::span<const uint8_t>(weights[e]->data()+98*b,98),v[1+b]),"GU gate setup");
                need(compact_iq3xxs::encode_block(std::span<const uint8_t>(weights[e]->data()+fmt.up_off+98*b,98),v[1+n+b]),"GU up setup");
            }
        }
        std::cout<<"{\"kind\":\"gu_setup\",\"experts\":"<<count
            <<",\"logical_view_bytes\":"<<uint64_t(count)*1715200
            <<",\"view_storage_bytes\":"<<uint64_t(count)*(2*n+2)*sizeof(compact_iq3xxs::Block)
            <<",\"raw_blob_bytes\":"<<uint64_t(count)*fmt.bytes
            <<",\"allocation_encode_wall_s\":"<<seconds(begin)<<",\"allocation_encode_cpu_s\":"<<process_cpu()-c0<<"}"<<std::endl;
    }
    std::array<uint64_t,4> gu_totals() const {
        std::array<uint64_t,4> result{};
        for(const auto& c:gu_counts)for(int k=0;k<4;++k)result[k]+=c.values[k];
        return result;
    }
    void gu_part(int id,int participants) {
        // Partition on16-row/cache-line boundaries,including cross-expert cuts.
        const int units=group*(cpu::FF/16);
        int at=(units*id/participants)*16;const int end=(units*(id+1)/participants)*16;
        for(;at<end;) {
            const int j=gu_offset+at/cpu::FF,r0=at%cpu::FF,r1=std::min(cpu::FF,r0+end-at);
            const int nt=jobs[j].nt;const int nr=r1-r0;
            auto& c=gu_counts[id].values;c[0]+=nr;if(nt==1)c[1]+=nr;
            if(gu_compact && nt==1) {
                constexpr size_t n=compact_iq3xxs::projection_blocks;
                const auto all=std::span<const compact_iq3xxs::Block>(gu_views[j]);
                need(compact_iq3xxs::gu_rows(all.subspan(1,n),all.subspan(1+n,n),
                    std::span<const uint8_t>(activations[0]->data(),fmt.act_bytes),
                    std::span<float>(jobs[j].out[0],cpu::FF),r0,r1),"GU compact dispatch");
                c[2]+=nr;
            } else {
                if(nt==1)need(cpu::iq3xxs_pair_gu_rows(fmt,weights[j]->data(),activations[0]->data(),
                    jobs[j].out[0],r0,r1),"GU raw pair dispatch");
                else {
                    const void* act[4];float* out[4];
                    for(int t=0;t<nt;++t){act[t]=activations[t]->data();out[t]=jobs[j].out[t];}
                    cpu::native_gu_rows(fmt,weights[j]->data(),act,nt,out,r0,r1,cpu::NativeGuPhase::Decode);
                }
                c[3]+=nr;
            }
            at+=nr;
        }
    }
    void gu_pass(GuTeam& team,bool stream) {
        for(int b=0;b<batches;++b) {
            gu_offset=stream?(b%(count/group))*group:0;
            for(int j=0;j<group;++j)jobs[gu_offset+j].nt=nt[b*group+j];
            team.run();
        }
    }

    void gu_check_current() {
        Buffer reference(4*cpu::H*sizeof(float));auto* want=reinterpret_cast<float*>(reference.data());
        for(int j=0;j<count;++j) {
            const int n=jobs[j].nt;need(n>=1 && n<=4,"GU final NT");
            if(n==1)need(cpu::iq3xxs_pair_gu_rows(fmt,weights[j]->data(),activations[0]->data(),want,0,cpu::FF),"GU final reference");
            else {
                const void* act[4];float* out[4];
                for(int t=0;t<n;++t){act[t]=activations[t]->data();out[t]=want+t*cpu::H;}
                cpu::native_gu_rows(fmt,weights[j]->data(),act,n,out,0,cpu::FF,cpu::NativeGuPhase::Decode);
            }
            for(int t=0;t<n;++t) {
                need(std::memcmp(want+t*cpu::H,jobs[j].out[t],cpu::FF*sizeof(float))==0,"GU final timed bits");
                for(int r=0;r<cpu::FF;++r)need(std::isfinite(jobs[j].out[t][r]),"GU final finite");
            }
            reference.check();weights[j]->check();outputs[j]->check();
        }
        std::cout<<"{\"kind\":\"gu_result\",\"experts\":"<<count
            <<",\"active_values_equal\":true,\"finite_guards\":true}"<<std::endl;
    }
    void gu_validate(GuTeam& team,int participants) {
        Buffer reference(4*cpu::H*sizeof(float));
        for(int test=0;test<=4;++test)for(int offset=0;offset<count;offset+=group)for(bool compact:{false,true}) {
            gu_offset=offset;gu_compact=compact;
            for(int j=0;j<group;++j) {
                jobs[offset+j].nt=test?test:(j<4?j+1:nt[(offset+j)%nt.size()]);
                std::memset(outputs[offset+j]->data(),0xA7,4*cpu::H*sizeof(float));
            }
            team.run();
            for(int j=offset;j<offset+group;++j) {
                const int n=jobs[j].nt;std::memset(reference.data(),0xA7,reference.bytes);
                auto* want=reinterpret_cast<float*>(reference.data());
                if(n==1)need(cpu::iq3xxs_pair_gu_rows(fmt,weights[j]->data(),activations[0]->data(),want,0,cpu::FF),"GU reference pair");
                else {
                    const void* act[4];float* out[4];
                    for(int t=0;t<n;++t){act[t]=activations[t]->data();out[t]=want+t*cpu::H;}
                    cpu::native_gu_rows(fmt,weights[j]->data(),act,n,out,0,cpu::FF,cpu::NativeGuPhase::Decode);
                }
                need(std::memcmp(reference.data(),outputs[j]->data(),reference.bytes)==0,"GU team bits/inactive rows");
                for(int t=0;t<n;++t)for(int r=0;r<cpu::FF;++r)
                    need(std::isfinite(jobs[j].out[t][r]),"GU team finite");
                outputs[j]->check();weights[j]->check();reference.check();
            }
        }
        compact_iq3xxs::Block guard;std::memset(&guard,0xA7,sizeof guard);
        for(const auto& v:gu_views)need(std::memcmp(&v.front(),&guard,sizeof guard)==0 &&
            std::memcmp(&v.back(),&guard,sizeof guard)==0,"GU view canaries");
        for(const auto& a:activations)a->check();
        std::cout<<"{\"kind\":\"gu_validation\",\"compute_threads\":"<<participants
            <<",\"group\":"<<group<<",\"experts\":"<<count
            <<",\"NT_mask\":30,\"mixed_NT\":true,\"bits\":true,\"inactive\":true,\"canaries\":true}"<<std::endl;
    }
    void validate_compact() {
        namespace view=compact_iq3xxs;
        need(fmt.gu_type==18 && fmt.gu_row==980 && fmt.n_embd==2560 && fmt.n_ff==640,
            "compact IQ3_XXS geometry");
        constexpr size_t n=view::projection_blocks;
        std::vector<view::Block> storage(2*n+2);
        std::memset(storage.data(),0xA7,storage.size()*sizeof(view::Block));
        const auto guard=storage.front();
        auto gate=std::span(storage).subspan(1,n),up=std::span(storage).subspan(1+n,n);
        Buffer expected(cpu::FF*sizeof(float)),actual(cpu::FF*sizeof(float));
        auto* want=reinterpret_cast<float*>(expected.data());
        auto* got=reinterpret_cast<float*>(actual.data());
        for(int e=0;e<count;++e) {
            const auto* raw=weights[e]->data();
            for(size_t b=0;b<n;++b) {
                need(view::encode_block(std::span<const uint8_t>(raw+98*b,98),gate[b]),"encode gate");
                need(view::encode_block(std::span<const uint8_t>(raw+fmt.up_off+98*b,98),up[b]),"encode up");
            }
            for(int t=0;t<4;++t) {
                const auto* act=activations[t]->data();
                need(cpu::iq3xxs_pair_gu_rows(fmt,raw,act,want,0,cpu::FF),"native paired GU reference");
                need(view::gu_rows(gate,up,std::span<const uint8_t>(act,fmt.act_bytes),
                    std::span<float>(got,cpu::FF),0,cpu::FF),"compact paired GU");
                need(std::memcmp(want,got,cpu::FF*sizeof(float))==0,"compact/native GU bits");
                for(int r=0;r<cpu::FF;++r)need(std::isfinite(got[r]),"nonfinite compact GU");
                expected.check();actual.check();
            }
            need(std::memcmp(&storage.front(),&guard,sizeof guard)==0 &&
                std::memcmp(&storage.back(),&guard,sizeof guard)==0,"compact storage guards");
            weights[e]->check();
        }
        for(auto& a:activations)a->check();
        std::cout<<"{\"kind\":\"compact_validation\",\"format\":18,\"nt\":1,\"activation_cases\":4,\"experts\":"<<count
            <<",\"native_pair_bits_equal\":true,\"finite\":true,\"canaries\":true,\"compact_GU_bytes_per_expert\":"<<2*n*sizeof(view::Block)
            <<",\"raw_plus_compact_bytes_per_expert\":"<<fmt.bytes+2*n*sizeof(view::Block)<<"}"<<std::endl;
    }
    void serial(const cpu::ExpertJobMulti& j,float* out) {
        std::array<float,4*cpu::FF> ff{};
        std::array<std::unique_ptr<Buffer>,4> hq;
        std::array<cpu::ActQ,4> aq{};
        const void* ap[4];const void* hp[4];const cpu::ActQ* qp[4];float* fp[4];float* op[4];
        for(int t=0;t<j.nt;++t){ap[t]=j.nact[t];fp[t]=ff.data()+t*cpu::FF;op[t]=out+t*cpu::H;
            hq[t]=std::make_unique<Buffer>(fmt.h_bytes);hp[t]=hq[t]->data();qp[t]=&aq[t];}
        cpu::native_gu_rows(fmt,j.blob,ap,j.nt,fp,0,cpu::FF,cpu::NativeGuPhase::Decode);
        for(int t=0;t<j.nt;++t){
            if(fmt.d_type==42)cpu::act_quant_any(fp[t],cpu::FF,aq[t]);
            else cpu::native_quant_h(fmt,fp[t],hq[t]->data());
        }
        if(fmt.d_type==42)cpu::q2_rows_any(j.blob+fmt.down_off,fmt.d_row,cpu::FF/64,qp,j.nt,op,0,cpu::H);
        else cpu::native_down_rows(fmt,j.blob,hp,j.nt,op,0,cpu::H);
        for(int t=0;t<j.nt;++t)hq[t]->check();
    }
    void validate(cpu::ExpertPool& pool) {
        // Exercise every loaded expert with every NT, outside timed work. Unlike
        // identical blobs, distinct model experts detect job/index permutations.
        std::array<float,4*cpu::H> want{};
        auto reset=[&](int e){
            auto* words=reinterpret_cast<uint32_t*>(outputs[e]->data());
            std::fill(words,words+4*cpu::H,unused);
        };
        auto compare=[&](int e){
            const int n=jobs[e].nt;
            serial(jobs[e],want.data());
            need(std::memcmp(want.data(),outputs[e]->data(),n*cpu::H*sizeof(float))==0,"pool/serial bits");
            const auto* words=reinterpret_cast<const uint32_t*>(outputs[e]->data());
            for(int i=n*cpu::H;i<4*cpu::H;++i)need(words[i]==unused,"inactive NT output written");
            for(int i=0;i<n*cpu::H;++i)need(std::isfinite(want[i]),"nonfinite output");
        };
        for(int n=1;n<=4;++n){
            for(int e=0;e<count;++e){jobs[e].nt=n;reset(e);}
            pool.run_split_multi_native(fmt,jobs.data(),count,cpu::NativeGuPhase::Decode);
            for(int e=0;e<count;++e)compare(e);
        }
        // Validate the same group size as timing with heterogeneous NTs. Force
        // NT1..4 into each group so sparse marginal bins cannot omit coverage.
        for(int offset=0;offset<count;offset+=group){
            for(int e=0;e<group;++e){
                jobs[offset+e].nt=e<4?e+1:nt[(offset+e)%nt.size()];
                reset(offset+e);
            }
            pool.run_split_multi_native(fmt,jobs.data()+offset,group,cpu::NativeGuPhase::Decode);
            for(int e=0;e<group;++e)compare(offset+e);
        }
        for(auto& w:weights)w->check();for(auto& o:outputs)o->check();for(auto& a:activations)a->check();
        std::cout<<"{\"kind\":\"validation\",\"distinct_experts\":"<<count<<",\"NT_mask\":30,\"pool_serial_bits_equal\":true,\"mixed_NT_bits_equal\":true,\"inactive_outputs_untouched\":true,\"canaries\":true}"<<std::endl;
    }
};

void compact_controls() {
    namespace view=compact_iq3xxs;
    std::array<uint8_t,98> raw{};view::Block packed;
    constexpr uint16_t half=0x3c00;std::memcpy(raw.data(),&half,2);
    uint64_t cases=0;
    for(unsigned grid=0;grid<256;++grid)for(unsigned sign=0;sign<128;++sign)for(unsigned scale=0;scale<16;++scale) {
        for(unsigned p=0;p<8;++p) {
            for(unsigned j=0;j<8;++j)raw[2+8*p+j]=uint8_t(grid+8*p+j);
            uint32_t gas=((scale+p)&15)<<28;
            for(unsigned j=0;j<4;++j)gas|=((sign+17*p+29*j)&127)<<(7*j);
            std::memcpy(raw.data()+66+4*p,&gas,4);
        }
        need(view::encode_block(raw,packed) && packed.d==half,"compact exhaustive encoding");
        for(unsigned p=0;p<8;++p)for(unsigned j=0;j<32;++j) {
            uint32_t gas;std::memcpy(&gas,raw.data()+66+4*p,4);
            const unsigned magnitude=(iq3xxs_grid[raw[2+8*p+j/4]]>>(8*(j&3)))&255;
            const bool negative=(ksigns_iq2xs[(gas>>(7*(j/8)))&127]>>(j&7))&1;
            const unsigned code=(packed.codes[16*p+(j&15)]>>(4*(j/16)))&15;
            const unsigned decoded=(code&7)==7?62:4+8*(code&7);
            need(decoded==magnitude && bool(code&8)==negative &&
                ((packed.scales[p/2]>>(4*(p&1)))&15)==(gas>>28),"compact codec value");
        }
        const unsigned p=(grid+sign+scale)&7;
        alignas(32) int8_t act[32];alignas(32) int32_t got[8],want[8]{};
        uint32_t gas;std::memcpy(&gas,raw.data()+66+4*p,4);
        for(unsigned j=0;j<32;++j) {
            act[j]=int8_t(int((j*17+grid+sign)&255)-128);
            const unsigned magnitude=(iq3xxs_grid[raw[2+8*p+j/4]]>>(8*(j&3)))&255;
            const bool negative=(ksigns_iq2xs[(gas>>(7*(j/8)))&127]>>(j&7))&1;
            const int signed_act=negative ? (act[j]==-128?-128:-int(act[j])) : int(act[j]);
            want[j/4]+=int(magnitude)*signed_act*int(2*(gas>>28)+1);
        }
        _mm256_store_si256(reinterpret_cast<__m256i*>(got),
            view::chunk(packed,p,_mm256_load_si256(reinterpret_cast<const __m256i*>(act))));
        need(std::memcmp(got,want,sizeof got)==0,"compact integer lanes");++cases;
    }
    _mm256_zeroupper();
    const auto saved=packed;
    need(!view::encode_block(std::span<const uint8_t>(raw).first(97),packed) &&
         std::memcmp(&saved,&packed,sizeof packed)==0,"compact short source refusal");
    need(!view::encode_block(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&packed),98),packed) &&
         std::memcmp(&saved,&packed,sizeof packed)==0,"compact encode alias refusal");

    cpu::NativeFmt fmt;std::string err;
    need(cpu::native_fmt(18,20,cpu::H,cpu::FF,fmt,err),"synthetic native geometry");
    Buffer weights(fmt.bytes),activation(fmt.act_bytes),expected(cpu::FF*sizeof(float)),actual(cpu::FF*sizeof(float));
    std::vector<view::Block> gate(view::projection_blocks),up(view::projection_blocks);
    auto* want=reinterpret_cast<float*>(expected.data());auto* got=reinterpret_cast<float*>(actual.data());
    const std::array<uint16_t,8> halves{0x0000,0x8000,0x0001,0x03ff,0x0400,0x2400,0x3c00,0x7bff};
    const std::array<float,4> act_scales{0.f,0.0001f,1.f,-.5f};
    for(unsigned test=0;test<16;++test) {
        for(unsigned r=0;r<cpu::FF;++r)for(unsigned b=0;b<10;++b)for(unsigned proj=0;proj<2;++proj) {
            auto* raw=weights.data()+proj*fmt.up_off+r*980+b*98;
            const auto d=halves[(r+b+test+2*proj)&7];std::memcpy(raw,&d,2);
            for(unsigned j=0;j<64;++j)raw[2+j]=uint8_t(r+11*b+7*j+13*test+17*proj);
            for(unsigned p=0;p<8;++p) {
                uint32_t gas=((r+b+p+test+proj)&15)<<28;
                for(unsigned j=0;j<4;++j)gas|=((r+19*b+11*p+7*j+test+proj)&127)<<(7*j);
                std::memcpy(raw+66+4*p,&gas,4);
            }
            need(view::encode_block(std::span<const uint8_t>(raw,98),(proj?up:gate)[r*10+b]),"synthetic view");
        }
        std::memset(activation.data(),0,fmt.act_bytes);
        for(unsigned b=0;b<10;++b) {
            auto* act=activation.data()+b*sizeof(block_q8_K);
            const float d=act_scales[(b+test)&3];std::memcpy(act,&d,sizeof d);
            for(unsigned j=0;j<256;++j) {
                const int8_t value=int8_t(int((17*j+13*b+19*test)&255)-128);
                std::memcpy(act+offsetof(block_q8_K,qs)+j,&value,1);
            }
        }
        need(cpu::iq3xxs_pair_gu_rows(fmt,weights.data(),activation.data(),want,0,cpu::FF),"synthetic native pair");
        need(view::gu_rows(gate,up,std::span<const uint8_t>(activation.data(),fmt.act_bytes),
            std::span<float>(got,cpu::FF),0,cpu::FF),"synthetic compact pair");
        need(std::memcmp(want,got,cpu::FF*sizeof(float))==0,"synthetic FP32/FMA bits");
        for(unsigned r=0;r<cpu::FF;++r)need(std::isfinite(got[r]),"synthetic finite GU");
        for(auto range:std::array<std::array<int,2>,4>{{{0,1},{319,321},{639,640},{640,640}}}) {
            std::memset(actual.data(),0xA7,cpu::FF*sizeof(float));
            std::array<uint8_t,cpu::FF*sizeof(float)> before;std::memcpy(before.data(),actual.data(),before.size());
            need(view::gu_rows(gate,up,std::span<const uint8_t>(activation.data(),fmt.act_bytes),
                std::span<float>(got,cpu::FF),range[0],range[1]),"partial compact range");
            for(int r=0;r<cpu::FF;++r)
                need(std::memcmp(actual.data()+r*4,(r>=range[0]&&r<range[1])?
                    reinterpret_cast<const uint8_t*>(want+r):before.data()+r*4,4)==0,"inactive compact row");
        }
        weights.check();activation.check();expected.check();actual.check();
    }
    std::array<uint8_t,cpu::FF*sizeof(float)> before;std::memcpy(before.data(),actual.data(),before.size());
    const auto g=std::span<const view::Block>(gate),u=std::span<const view::Block>(up);
    const auto a=std::span<const uint8_t>(activation.data(),fmt.act_bytes);const auto o=std::span<float>(got,cpu::FF);
    unsigned refusals=0;
    auto refuse=[&](bool accepted) {
        need(!accepted && std::memcmp(before.data(),actual.data(),before.size())==0,"compact shape/range refusal");++refusals;
    };
    refuse(view::gu_rows(g.first(g.size()-1),u,a,o,0,640));
    refuse(view::gu_rows(g,u.first(u.size()-1),a,o,0,640));
    refuse(view::gu_rows(g,u,a.first(a.size()-1),o,0,640));
    refuse(view::gu_rows(g,u,a,o.first(639),0,640));
    refuse(view::gu_rows(g,u,a,o,-1,1));refuse(view::gu_rows(g,u,a,o,2,1));
    refuse(view::gu_rows(g,u,a,o,0,641));refuse(view::gu_rows(g,g,a,o,0,640));
    refuse(view::gu_rows(g,u,std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(g.data()),fmt.act_bytes),o,0,640));
    // A byte view of a real float array is legal; no invented overlapping object lifetimes.
    std::vector<float> alias(1024,1.f);const auto alias_before=alias;
    need(!view::gu_rows(g,u,std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(alias.data()),fmt.act_bytes),
        std::span<float>(alias).first(640),0,640) && alias==alias_before,"compact activation/output alias");
    ++refusals;actual.check();
    std::cout<<"{\"kind\":\"compact_controls\",\"codec_cases\":"<<cases
        <<",\"integer_chunks\":"<<cases<<",\"encoded_values\":"<<cases*256
        <<",\"FMA_cases\":16,\"partial_ranges\":64,\"GU_refusals\":"<<refusals
        <<",\"encode_refusals\":2,\"finite_canaries\":true}"<<std::endl;
}
int main(int argc,char** argv) try {
    std::cout<<std::setprecision(12);
    if(argc==2 && std::string(argv[1])=="--compact-controls") {
        compact_controls();std::cout<<"{\"kind\":\"closed\",\"passed\":true}"<<std::endl;return 0;
    }
    need(argc==16,"shard gu dn gate up down bytes workers group check|measure|compact-check n1 n2 n3 n4 version");
    need(std::string(argv[15])=="20261022","protocol version");
    const auto requested_workers=number(argv[8]),requested_group=number(argv[9]);
    const std::string mode=argv[10];
    const bool gu_mode=mode=="gu-check" || mode=="gu-raw" || mode=="gu-compact";
    need(requested_workers>=1 && requested_workers<=5 && requested_group>=6 && requested_group<=32 &&
        (mode=="check"||mode=="measure"||mode=="compact-check"||gu_mode),"mode/shape");
    need((mode!="compact-check" && !gu_mode) || number(argv[2])==18,"compact IQ3_XXS only");
    // Intermediate worker counts expose the scaling curve. Five workers also
    // enable the production host compute thread; smaller counts leave it waiting.
    const int workers=static_cast<int>(requested_workers),group=static_cast<int>(requested_group);
    std::array<uint64_t,4> observed{};for(int i=0;i<4;++i){observed[i]=number(argv[11+i]);need(observed[i]<=1000000000ull,"histogram count");}
    const auto nt=schedule(group,observed);std::array<int,4> actual{};for(int n:nt)++actual[n-1];
    const auto topology=cpu::detect_cpu_topology(true);
    need(topology.host_core>=0 && topology.worker_cores.size()>=5,"six physical cores required");

    if(gu_mode) {
        need(workers==1 || workers==5,"GU planned worker count");
        cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(topology.host_core,&mask);
        need(sched_setaffinity(0,sizeof mask,&mask)==0,"GU host affinity");
        Work work(argv,group,nt);work.gu_setup();
        const int participants=workers==5?6:1;
        GuTeam team(participants,topology,[&](int id,int n){work.gu_part(id,n);});
        work.gu_validate(team,participants);verify_affinity(topology,workers);
        std::cout<<"{\"kind\":\"gu_work\",\"arm\":\""<<mode<<"\",\"compute_threads\":"<<participants
            <<",\"group\":"<<group<<",\"experts\":"<<work.count<<",\"batches\":100,\"jobs_NT\":["
            <<actual[0]<<","<<actual[1]<<","<<actual[2]<<","<<actual[3]
            <<"],\"host_cpu\":"<<topology.host_core<<",\"affinity_verified\":true}"<<std::endl;
        if(mode!="gu-check")for(bool stream:{false,true}) {
            work.gu_compact=mode=="gu-compact";
            auto pass=[&]{work.gu_pass(team,stream);};pass();
            double fastest=1e9;for(int i=0;i<3;++i){const auto begin=Clock::now();pass();fastest=std::min(fastest,seconds(begin));}
            const auto timer=Clock::now();for(int i=0;i<10000;++i){const auto query=Clock::now();(void)query;}
            const double cost=seconds(timer)/10000;need(fastest>0 && cost/fastest<.001,"GU timer overhead");
            std::cout<<"{\"kind\":\"gu_timer\",\"stream\":"<<(stream?"true":"false")
                <<",\"query_s\":"<<cost<<",\"fastest_pass_s\":"<<fastest<<"}"<<std::endl;
            for(int rep=0;rep<5;++rep) {
                const auto before=work.gu_totals();const auto begin=Clock::now();const double c0=process_cpu();
                int passes=0;do{pass();++passes;}while(seconds(begin)<1.3 && passes<1000000);
                const double wall=seconds(begin),cpu_time=process_cpu()-c0;
                need(wall>=1 && passes<1000000,"GU timing floor/budget");
                auto total=work.gu_totals();for(int k=0;k<4;++k)total[k]-=before[k];
                need(total[0]==uint64_t(passes)*100*group*cpu::FF &&
                    total[1]==uint64_t(passes)*actual[0]*cpu::FF &&
                    total[2]==(work.gu_compact?total[1]:0) && total[3]==total[0]-total[2],"GU timed dispatch rows");
                std::cout<<"{\"kind\":\"gu_sample\",\"stream\":"<<(stream?"true":"false")<<",\"rep\":"<<rep
                    <<",\"passes\":"<<passes<<",\"wall_s\":"<<wall<<",\"cpu_s\":"<<cpu_time
                    <<",\"rows\":"<<total[0]<<",\"NT1_rows\":"<<total[1]<<",\"compact_rows\":"<<total[2]
                    <<",\"raw_rows\":"<<total[3]<<"}"<<std::endl;
            }
            work.gu_check_current();work.gu_validate(team,participants);
        }
        if(mode=="gu-check")work.gu_check_current();
        verify_affinity(topology,workers);
        std::cout<<"{\"kind\":\"closed\",\"passed\":true}"<<std::endl;return 0;
    }
    cpu::ExpertPool pool(workers,true,workers==5);
    cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(topology.host_core,&mask);need(sched_setaffinity(0,sizeof mask,&mask)==0,"host affinity");
    Work work(argv,group,nt);
    if(mode=="compact-check") {
        work.validate_compact();verify_affinity(topology,workers);
        std::cout<<"{\"kind\":\"closed\",\"passed\":true}"<<std::endl;return 0;
    }
    std::cout<<"{\"kind\":\"work\",\"workers\":"<<workers<<",\"host_works\":"<<(workers==5?"true":"false")
        <<",\"group\":"<<group<<",\"batches\":"<<batches<<",\"unique_stream_experts\":"<<work.count<<",\"blob_bytes\":"<<work.fmt.bytes
        <<",\"jobs_NT\":["<<actual[0]<<","<<actual[1]<<","<<actual[2]<<","<<actual[3]<<"],\"phase\":\"decode\"}"<<std::endl;
    work.validate(pool);verify_affinity(topology,workers);
    std::cout<<"{\"kind\":\"placement\",\"workers\":"<<pool.workers()<<",\"host_cpu\":"<<topology.host_core<<",\"worker_cpus\":[";
    for(int i=0;i<workers;++i)std::cout<<(i?",":"")<<topology.worker_cores[i];
    std::cout<<"],\"actual_affinity_verified\":true}"<<std::endl;
    if(mode=="measure")for(bool stream:{false,true}){
        auto pass=[&]{work.run(pool,stream);};pass();
        double fastest=1e9;for(int i=0;i<3;++i){const auto t=Clock::now();pass();fastest=std::min(fastest,seconds(t));}
        const auto timer=Clock::now();for(int i=0;i<10000;++i){const auto t=Clock::now();(void)t;}
        const double cost=seconds(timer)/10000;need(cost/fastest<.001,"timer overhead");
        std::cout<<"{\"kind\":\"timer\",\"stream\":"<<(stream?"true":"false")<<",\"query_s\":"<<cost<<",\"fastest_pass_s\":"<<fastest<<"}"<<std::endl;
        for(int rep=0;rep<repetitions;++rep){
            const double gu=pool.ms_multi_gu,q=pool.ms_multi_q,dn=pool.ms_multi_down,cpu0=process_cpu();
            const auto begin=Clock::now();int passes=0;do{pass();++passes;}while(seconds(begin)<1.3 && passes<1000000);
            const double wall=seconds(begin),scheduled=process_cpu()-cpu0;need(wall>=1 && passes<1000000,"sample floor/budget");
            std::cout<<"{\"kind\":\"sample\",\"stream\":"<<(stream?"true":"false")<<",\"rep\":"<<rep<<",\"passes\":"<<passes
                <<",\"wall_s\":"<<wall<<",\"cpu_s\":"<<scheduled<<",\"gu_s\":"<<(pool.ms_multi_gu-gu)*.001
                <<",\"quant_s\":"<<(pool.ms_multi_q-q)*.001<<",\"down_s\":"<<(pool.ms_multi_down-dn)*.001<<"}"<<std::endl;
        }
        work.validate(pool);
    }
    verify_affinity(topology,workers);
    std::cout<<"{\"kind\":\"closed\",\"passed\":true}"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<"CPU decode-mix failed: "<<e.what()<<std::endl;return 1;}
