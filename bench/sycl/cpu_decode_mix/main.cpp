// Main-authored CPU component fixtures; raw references link frozen production code.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "activation_lut.hpp"
#include <algorithm>
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
#include <sched.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
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

namespace lut=activation_lut;
template<class T> struct alignas(64) Guarded {
    std::array<uint8_t,64> before;T value;std::array<uint8_t,64> after;
    Guarded(){before.fill(0xA7);after.fill(0xA7);}
    void check() const {for(unsigned i=0;i<64;++i)need(before[i]==0xA7 && after[i]==0xA7,"scratch guard");}
};
void affinity(int core) {
    int tasks=0;
    for(const auto& e:std::filesystem::directory_iterator("/proc/self/task")) {
        cpu_set_t mask;CPU_ZERO(&mask);
        need(sched_getaffinity(number(e.path().filename().c_str()),sizeof mask,&mask)==0 &&
             CPU_COUNT(&mask)==1 && CPU_ISSET(core,&mask),"actual serial affinity");++tasks;
    }
    need(tasks==1,"unexpected helper threads");
}

void controls() {
    auto slab=std::make_unique<Guarded<lut::Slab>>();
    std::array<uint8_t,sizeof(block_q8_K)> activation{};
    std::array<uint8_t,98> raw{};uint64_t entries=0,chunks=0;
    for(int edge=0;edge<8;++edge) {
        for(unsigned i=0;i<256;++i) {
            const int v=edge==0?0:edge==1?127:edge==2?-128:edge==3?(i&1?127:-128):
                edge==4?-1:edge==5?1:edge==6?(i%4==0?-128:i%4==1?0:i%4==2?127:-1):int((37*i+11)&255)-128;
            const int8_t q=int8_t(v);std::memcpy(activation.data()+offsetof(block_q8_K,qs)+i,&q,1);
        }
        need(lut::build(activation,slab->value),"table build");
        for(unsigned group=0;group<64;++group)for(unsigned code=0;code<256;++code)for(unsigned sign=0;sign<16;++sign) {
            int32_t sum=0;
            for(unsigned lane=0;lane<4;++lane) {
                int8_t a;std::memcpy(&a,activation.data()+offsetof(block_q8_K,qs)+4*group+lane,1);
                // Independent scalar PSIGN oracle; -128 keeps its bit pattern.
                int q=a;if((sign>>lane)&1)q=a==-128?-128:-q;
                sum+=int((iq3xxs_grid[code]>>(8*lane))&255)*q;
            }
            need(slab->value.values[group][16*code+sign]==sum,"table entry differs");++entries;
        }
        for(unsigned code=0;code<256;++code) {
            std::fill(raw.begin()+2,raw.begin()+66,uint8_t(code));
            for(unsigned sign=0;sign<128;++sign)for(unsigned scale=0;scale<16;++scale) {
                const uint32_t gas=(scale<<28)|sign|(sign<<7)|(sign<<14)|(sign<<21);
                for(unsigned p=0;p<8;++p)std::memcpy(raw.data()+66+4*p,&gas,4);
                const unsigned p=(code+sign+scale)&7;
                alignas(32) std::array<int32_t,8> got{};
                _mm256_store_si256(reinterpret_cast<__m256i*>(got.data()),lut::chunk(slab->value,raw.data(),p));
                for(unsigned lane=0;lane<8;++lane) {
                    int32_t sum=0;const unsigned signs=ksigns_iq2xs[sign]>>(4*(lane&1));
                    for(unsigned q=0;q<4;++q) {
                        int8_t a;std::memcpy(&a,activation.data()+offsetof(block_q8_K,qs)+32*p+4*lane+q,1);
                        int v=a;if((signs>>q)&1)v=a==-128?-128:-v;
                        sum+=int((iq3xxs_grid[code]>>(8*q))&255)*v;
                    }
                    need(got[lane]==sum*int(2*scale+1),"scaled chunk differs");
                }
                ++chunks;
            }
        }
        slab->check();
    }
    Buffer raw_weight(2*lut::projection_bytes),act(10*sizeof(block_q8_K)),out(lut::rows*sizeof(float));
    std::vector<lut::Accumulator> acc(lut::rows);std::memset(acc.data(),0xA7,acc.size()*sizeof acc[0]);
    lut::Projection valid{std::span<const uint8_t>(raw_weight.data(),raw_weight.bytes),lut::projection_bytes,
        std::span<float>(reinterpret_cast<float*>(out.data()),lut::rows)};
    std::array<lut::Projection,1> e{valid};lut::Counts count{7,9,11};
    auto saved=std::make_unique<lut::Slab>();std::memcpy(saved.get(),&slab->value,sizeof *saved);
    const auto unchanged=[&]{
        need(std::memcmp(saved.get(),&slab->value,sizeof *saved)==0,"refusal modified table");
        for(size_t i=0;i<acc.size()*sizeof acc[0];++i)need(reinterpret_cast<const uint8_t*>(acc.data())[i]==0xA7,"refusal modified accumulator");
        for(size_t i=0;i<out.bytes;++i)need(out.data()[i]==0xA7,"refusal modified output");
        need(count.slabs==7&&count.block_rows==9&&count.output_rows==11,"refusal modified counts");
        raw_weight.check();act.check();out.check();slab->check();
    };
    const auto a=std::span<const uint8_t>(act.data(),act.bytes);unsigned refused=0;
    const auto check=[&](bool result){need(!result,"invalid shape accepted");unchanged();++refused;};
    check(lut::build(a.first(sizeof(block_q8_K)-1),slab->value));
    check(lut::build(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&slab->value),sizeof(block_q8_K)),slab->value));
    check(lut::gu({},a,slab->value,acc,count));
    std::array<lut::Projection,11> too_many;too_many.fill(valid);
    check(lut::gu(too_many,a,slab->value,acc,count));
    check(lut::gu(e,a.first(a.size()-1),slab->value,acc,count));
    check(lut::gu(e,a,slab->value,std::span(acc).first(acc.size()-1),count));
    e[0].up_offset=lut::projection_bytes-1;check(lut::gu(e,a,slab->value,acc,count));e[0]=valid;
    e[0].up_offset=raw_weight.bytes+1;check(lut::gu(e,a,slab->value,acc,count));e[0]=valid;
    e[0].raw=e[0].raw.first(e[0].raw.size()-1);check(lut::gu(e,a,slab->value,acc,count));e[0]=valid;
    e[0].out=e[0].out.first(lut::rows-1);check(lut::gu(e,a,slab->value,acc,count));e[0]=valid;
    check(lut::gu(e,std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(acc.data()),act.bytes),slab->value,acc,count));
    check(lut::gu(e,std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&slab->value),act.bytes),slab->value,acc,count));
    std::vector<float> activation_alias(cpu::H),weight_alias(raw_weight.bytes/sizeof(float));
    auto original_activation=activation_alias,original_weight=weight_alias;
    e[0].out=std::span<float>(activation_alias).first(lut::rows);
    check(lut::gu(e,std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(activation_alias.data()),act.bytes),slab->value,acc,count));e[0]=valid;
    e[0].out=std::span<float>(weight_alias).first(lut::rows);
    e[0].raw=std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(weight_alias.data()),raw_weight.bytes);
    check(lut::gu(e,a,slab->value,acc,count));e[0]=valid;
    need(activation_alias==original_activation && weight_alias==original_weight,"alias refusal wrote data");
    _mm256_zeroupper();
    std::cout<<"{\"kind\":\"lut_controls\",\"table_entries\":"<<entries<<",\"scaled_chunks\":"<<chunks
        <<",\"activation_edges\":8,\"refusals\":"<<refused<<",\"guards\":true}"<<std::endl;
}

struct Work {
    cpu::NativeFmt fmt;int group,count,offset=0;uint64_t calls=0;
    std::vector<std::unique_ptr<Buffer>> weights,outputs;
    std::unique_ptr<Guarded<lut::Slab>> slab=std::make_unique<Guarded<lut::Slab>>();
    std::vector<lut::Accumulator> scratch;
    std::vector<lut::Projection> experts;
    std::array<std::unique_ptr<Buffer>,4> acts;
    lut::Counts counts;
    Work(char** argv,int g):group(g),scratch(size_t(g)*lut::rows+2),experts(g) {
        std::string error;need(number(argv[2])<=42 && number(argv[3])<=42,"format ID");
        const bool valid=cpu::native_fmt(number(argv[2]),number(argv[3]),cpu::H,cpu::FF,fmt,error);need(valid,error.c_str());
        need(fmt.gu_type==18 && fmt.gu_row==lut::row_bytes && fmt.up_off==lut::projection_bytes &&
            fmt.bytes==number(argv[7]) && fmt.act_bytes==10*sizeof(block_q8_K),"native geometry");
        count=10*int(((128ull<<20)+10*fmt.bytes-1)/(10*fmt.bytes));need(count<=256 && count%group==0,"stream extent");
        struct File {int fd;~File(){if(fd>=0)close(fd);}} file{open(argv[1],O_RDONLY|O_CLOEXEC)};need(file.fd>=0,"open model");
        for(int i=0;i<count;++i) {
            auto w=std::make_unique<Buffer>(fmt.bytes);
            read_slice(file.fd,w->data(),fmt.up_off,number(argv[4])+size_t(i)*fmt.up_off);
            read_slice(file.fd,w->data()+fmt.up_off,fmt.up_off,number(argv[5])+size_t(i)*fmt.up_off);
            read_slice(file.fd,w->data()+fmt.down_off,fmt.bytes-fmt.down_off,number(argv[6])+size_t(i)*(fmt.bytes-fmt.down_off));
            weights.push_back(std::move(w));outputs.push_back(std::make_unique<Buffer>(lut::rows*sizeof(float)));
        }
        std::memset(scratch.data(),0xA7,scratch.size()*sizeof scratch[0]);
        uint32_t seed=20261108;std::array<float,cpu::H> input;
        for(auto& a:acts) {
            for(float& x:input){seed=1664525u*seed+1013904223u;x=(int(seed>>8)-8388608)*0x1p-23f;}
            a=std::make_unique<Buffer>(fmt.act_bytes);cpu::native_quant_act(fmt,input.data(),a->data());
        }
    }
    void run(bool candidate,int activation) {
        if(candidate) {
            for(int e=0;e<group;++e)experts[e]={std::span<const uint8_t>(weights[offset+e]->data(),fmt.bytes),
                fmt.up_off,std::span<float>(reinterpret_cast<float*>(outputs[offset+e]->data()),lut::rows)};
            need(lut::gu(experts,std::span<const uint8_t>(acts[activation]->data(),fmt.act_bytes),slab->value,
                std::span(scratch).subspan(1,size_t(group)*lut::rows),counts),"LUT GU dispatch");
        } else for(int e=0;e<group;++e) {
            const void* act[1]={acts[activation]->data()};float* out[1]={reinterpret_cast<float*>(outputs[offset+e]->data())};
            cpu::native_gu_rows(fmt,weights[offset+e]->data(),act,1,out,0,cpu::FF,cpu::NativeGuPhase::Decode);
        }
        ++calls;
    }
    void guards() const {
        for(const auto& x:weights)x->check();for(const auto& x:outputs)x->check();for(const auto& x:acts)x->check();slab->check();
        const auto* first=reinterpret_cast<const uint8_t*>(scratch.data());
        const auto* last=reinterpret_cast<const uint8_t*>(&scratch.back());
        for(size_t i=0;i<sizeof(lut::Accumulator);++i)need(first[i]==0xA7 && last[i]==0xA7,"accumulator guards");
    }
    void validate() {
        std::vector<float> expected(size_t(group)*lut::rows);uint64_t comparisons=0;
        for(int a=0;a<4;++a)for(offset=0;offset<count;offset+=group) {
            run(false,a);
            for(int e=0;e<group;++e)std::memcpy(expected.data()+e*lut::rows,outputs[offset+e]->data(),lut::rows*sizeof(float));
            run(true,a);
            for(int e=0;e<group;++e) {
                need(std::memcmp(expected.data()+e*lut::rows,outputs[offset+e]->data(),lut::rows*sizeof(float))==0,"real GU native bits");
                for(unsigned r=0;r<lut::rows;++r)need(std::isfinite(reinterpret_cast<const float*>(outputs[offset+e]->data())[r]),"real GU finite");
                ++comparisons;
            }
        }
        guards();std::cout<<"{\"kind\":\"lut_validation\",\"experts\":"<<count<<",\"activation_cases\":4,\"comparisons\":"<<comparisons
            <<",\"native_bits_equal\":true,\"finite_guards\":true}"<<std::endl;
    }
};
int main(int argc,char** argv) try {
    std::cout<<std::setprecision(12);
    __builtin_cpu_init();need(__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma")&&__builtin_cpu_supports("f16c"),"AVX2/FMA/F16C required");
    if(argc==2 && std::string(argv[1])=="--lut-controls")controls();
    else {
        need(argc==11,"shard gu down gate up down_offset blob_bytes group check|raw|lut version");
        need(std::string(argv[10])=="20261108","protocol version");const auto g=number(argv[8]);
        const std::string mode=argv[9];need((g==1||g==10)&&(mode=="check"||mode=="raw"||mode=="lut"),"mode/group");
        const auto topo=cpu::detect_cpu_topology(true);need(topo.host_core>=0,"physical host CPU");
        cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(topo.host_core,&mask);need(sched_setaffinity(0,sizeof mask,&mask)==0,"pin serial host");
        Work work(argv,int(g));work.validate();affinity(topo.host_core);
        std::cout<<"{\"kind\":\"lut_work\",\"arm\":\""<<mode<<"\",\"group\":"<<g<<",\"NT\":1,\"experts\":"<<work.count
            <<",\"host_cpu\":"<<topo.host_core<<",\"affinity_verified\":true,\"slab_bytes\":"<<sizeof(lut::Slab)
            <<",\"accumulator_bytes\":"<<g*lut::rows*sizeof(lut::Accumulator)<<",\"raw_blob_bytes\":"<<work.count*work.fmt.bytes<<"}"<<std::endl;
        if(mode!="check")for(bool stream:{false,true}) {
            unsigned index=0;const bool candidate=mode=="lut";
            const auto pass=[&]{work.offset=stream?int(index++%(work.count/work.group))*work.group:0;work.run(candidate,0);};
            pass();double fastest=1e9;
            for(int i=0;i<3;++i){const auto t=Clock::now();pass();fastest=std::min(fastest,seconds(t));}
            const auto t=Clock::now();for(int i=0;i<10000;++i){const auto query=Clock::now();(void)query;}
            const double cost=seconds(t)/10000;need(fastest>0&&cost/fastest<.001,"timer overhead");
            std::cout<<"{\"kind\":\"lut_timer\",\"stream\":"<<(stream?"true":"false")<<",\"query_s\":"<<cost<<",\"fastest_call_s\":"<<fastest<<"}"<<std::endl;
            for(int rep=0;rep<5;++rep) {
                const auto before=work.counts;const auto audit0=cpu::iq3_gu_audit_snapshot();const auto calls0=work.calls;
                const auto begin=Clock::now();const double c0=process_cpu();uint64_t passes=0;
                do{pass();++passes;}while(seconds(begin)<1.3 && passes<1000000);
                const double wall=seconds(begin),cpu_time=process_cpu()-c0;
                const auto audit=cpu::iq3_gu_audit_snapshot();need(wall>=1&&passes<1000000&&work.calls-calls0==passes,"timing floor/calls");
                const auto rows=passes*g*lut::rows;
                need(audit.enabled&&!audit.overflow&&audit.threads==1&&
                    audit.pair_rows-audit0.pair_rows==(candidate?0:rows)&&audit.nt1_rows-audit0.nt1_rows==(candidate?0:rows)&&
                    audit.calls-audit0.calls==(candidate?0:passes*g),"actual native pair dispatch");
                need(work.counts.slabs-before.slabs==(candidate?passes*10:0)&&
                    work.counts.block_rows-before.block_rows==(candidate?rows*10:0)&&
                    work.counts.output_rows-before.output_rows==(candidate?rows:0),"actual LUT builds/rows");
                std::cout<<"{\"kind\":\"lut_sample\",\"stream\":"<<(stream?"true":"false")<<",\"rep\":"<<rep<<",\"calls\":"<<passes
                    <<",\"wall_s\":"<<wall<<",\"cpu_s\":"<<cpu_time<<",\"output_rows\":"<<rows<<",\"slabs\":"<<work.counts.slabs-before.slabs
                    <<",\"block_rows\":"<<work.counts.block_rows-before.block_rows<<",\"native_pair_rows\":"<<audit.pair_rows-audit0.pair_rows<<"}"<<std::endl;
            }
            // Validate the final timed output before any other execution resets it.
            const int last=work.offset;std::vector<float> got(size_t(g)*lut::rows);
            for(int e=0;e<int(g);++e)std::memcpy(got.data()+e*lut::rows,work.outputs[last+e]->data(),lut::rows*sizeof(float));
            work.run(false,0);
            for(int e=0;e<int(g);++e)need(std::memcmp(got.data()+e*lut::rows,work.outputs[last+e]->data(),lut::rows*sizeof(float))==0,"final timed bits");
            work.guards();std::cout<<"{\"kind\":\"lut_result\",\"stream\":"<<(stream?"true":"false")<<",\"native_bits_equal\":true,\"finite_guards\":true}"<<std::endl;
        }
        affinity(topo.host_core);
    }
    std::cout<<"{\"kind\":\"closed\",\"passed\":true}"<<std::endl;return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
