// Main-authored CPU calibration. Production math is linked, never copied.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sched.h>
#include <stdexcept>
#include <string_view>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
constexpr int group = 6, reps = 5;
constexpr size_t guard = 64;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
double cpu_seconds(int who = RUSAGE_SELF) {
    rusage r{}; require(getrusage(who, &r)==0, "getrusage");
    return r.ru_utime.tv_sec+r.ru_utime.tv_usec*1e-6+r.ru_stime.tv_sec+r.ru_stime.tv_usec*1e-6;
}
struct Buffer {
    uint8_t* base = nullptr; size_t bytes;
    explicit Buffer(size_t n) : bytes(n) {
        require(n <= std::numeric_limits<size_t>::max() - 3 * guard, "allocation extent");
        const size_t aligned = ((n + 3 * guard - 1) / guard) * guard;
        base = static_cast<uint8_t*>(std::aligned_alloc(guard, aligned));
        require(base != nullptr, "allocation");
        std::memset(base, 0xA7, n+2*guard);
    }
    ~Buffer() { std::free(base); }
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    uint8_t* data() const { return base+guard; }
    void check() const {
        for (size_t i=0;i<guard;++i) require(base[i]==0xA7 && base[guard+bytes+i]==0xA7, "buffer canary");
    }
};
void pin(int core) {
    cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(core,&mask);
    require(sched_setaffinity(0,sizeof mask,&mask)==0, "CPU affinity");
}
void pread_exact(int fd, uint8_t* dest, size_t n, uint64_t off) {
    struct stat st{}; require(fstat(fd,&st)==0, "model stat");
    require(off<=static_cast<uint64_t>(st.st_size) && n<=static_cast<uint64_t>(st.st_size)-off, "model slice bounds");
    while (n) { const auto k=pread(fd,dest,n,static_cast<off_t>(off));
        require(k>0,"model slice read"); dest+=k; off+=k; n-=k; }
}
struct Inputs {
    cpu::NativeFmt fmt;
    std::vector<std::unique_ptr<Buffer>> blobs, outputs;
    std::array<std::unique_ptr<Buffer>,4> acts;
    std::array<std::array<float,cpu::H>,4> x;
    std::vector<cpu::ExpertJobMulti> jobs;
    int count;
    Inputs(const char* path, int gu, int dn, uint64_t go, uint64_t uo, uint64_t dno, size_t blob_bytes) {
        std::string err; require(cpu::native_fmt(gu,dn,cpu::H,cpu::FF,fmt,err),err.c_str());
        require(fmt.bytes==blob_bytes && fmt.gu_type!=42, "unsupported fixture geometry");
        count=group*static_cast<int>((((96ull<<20)+fmt.bytes-1)/fmt.bytes+group-1)/group);
        require(count<=96 && count*fmt.bytes>=96ull<<20, "stream extent");
        const int fd=open(path,O_RDONLY|O_CLOEXEC); require(fd>=0,"open model shard");
        for (int e=0;e<count;++e) {
            auto b=std::make_unique<Buffer>(fmt.bytes);
            pread_exact(fd,b->data(),fmt.up_off,go+e*fmt.up_off);
            pread_exact(fd,b->data()+fmt.up_off,fmt.up_off,uo+e*fmt.up_off);
            const auto down_bytes=fmt.bytes-fmt.down_off;
            pread_exact(fd,b->data()+fmt.down_off,down_bytes,dno+e*down_bytes);
            blobs.push_back(std::move(b)); outputs.push_back(std::make_unique<Buffer>(4*cpu::H*sizeof(float)));
        }
        close(fd);
        // A reproducible finite activation; token t retains its input at every NT.
        uint32_t state=20261020;
        for (int t=0;t<4;++t) {
            for (auto& value:x[t]) { state=1664525u*state+1013904223u; value=(int(state>>8)-8388608)*0x1p-23f; }
            acts[t]=std::make_unique<Buffer>(fmt.act_bytes);
        }
        jobs.resize(count);
        for (int e=0;e<count;++e) {
            jobs[e].blob=blobs[e]->data();
            for (int t=0;t<4;++t) {
                jobs[e].nact[t]=acts[t]->data();
                jobs[e].out[t]=reinterpret_cast<float*>(outputs[e]->data())+t*cpu::H;
            }
        }
    }
    void quant(int nt) { for (int t=0;t<nt;++t) cpu::native_quant_act(fmt,x[t].data(),acts[t]->data()); }
    void check(int n, int nt) {
        for (auto& b:blobs) b->check();
        for (auto& b:acts) b->check();
        for (int e=0;e<n;++e) {
            outputs[e]->check();
            for (int t=0;t<nt;++t) for (int r=0;r<cpu::H;++r)
                require(std::isfinite(jobs[e].out[t][r]),"nonfinite result");
        }
    }
    // Outside timing: check GU and Down separately against ggml's same-input
    // vec_dot, then the complete production pool against the serial same route.
    std::array<double,4> validate(cpu::ExpertPool& pool, int nt) {
        quant(nt); for (auto& j:jobs) j.nt=nt;
        pool.run_split_multi_native(fmt,jobs.data(),group);
        double gu_error=0,down_error=0,pool_error=0,q2_ggml_gap=0;
        for (int e=0;e<group;++e) {
            Buffer ff(4*cpu::FF*sizeof(float)), out(4*cpu::H*sizeof(float));
            std::array<std::unique_ptr<Buffer>,4> hq;
            std::array<cpu::ActQ,4> a2{};
            const void* ap[4]; const void* hp[4]; const cpu::ActQ* a2p[4]; float* fp[4]; float* op[4];
            for (int t=0;t<nt;++t) {
                ap[t]=acts[t]->data(); fp[t]=reinterpret_cast<float*>(ff.data())+t*cpu::FF;
                op[t]=reinterpret_cast<float*>(out.data())+t*cpu::H;
                hq[t]=std::make_unique<Buffer>(fmt.h_bytes); hp[t]=hq[t]->data(); a2p[t]=&a2[t];
            }
            cpu::native_gu_rows(fmt,blobs[e]->data(),ap,nt,fp,0,cpu::FF);
            double gn=0,gd=0;
            const auto dotg=ggml_get_type_traits_cpu(static_cast<ggml_type>(fmt.gu_type))->vec_dot;
            for (int t=0;t<nt;++t) for (int r=0;r<cpu::FF;++r) {
                float g=0,u=0;
                dotg(cpu::H,&g,0,blobs[e]->data()+r*fmt.gu_row,0,ap[t],0,1);
                dotg(cpu::H,&u,0,blobs[e]->data()+fmt.up_off+r*fmt.gu_row,0,ap[t],0,1);
                float want=(g/(1.f+std::exp(-g)))*u;
                require(std::isfinite(want),"nonfinite reference");
                gn+=std::abs(double(fp[t][r])-want); gd+=std::abs(double(want));
            }
            gu_error=std::max(gu_error,gn/(gd+1e-30));
            for (int t=0;t<nt;++t) {
                cpu::native_quant_h(fmt,fp[t],hq[t]->data());
                if (fmt.d_type==42) cpu::act_quant_any(fp[t],cpu::FF,a2[t]);
            }
            if (fmt.d_type==42) cpu::q2_rows_any(blobs[e]->data()+fmt.down_off,fmt.d_row,cpu::FF/64,a2p,nt,op,0,cpu::H);
            else cpu::native_down_rows(fmt,blobs[e]->data(),hp,nt,op,0,cpu::H);
            const auto dotd=ggml_get_type_traits_cpu(static_cast<ggml_type>(fmt.d_type))->vec_dot;
            double dn=0,dd=0,pn=0,pd=0,ggn=0,ggd=0;
            for (int t=0;t<nt;++t) for (int r=0;r<cpu::H;++r) {
                float want=0; dotd(cpu::FF,&want,0,blobs[e]->data()+fmt.down_off+r*fmt.d_row,0,hp[t],0,1);
                require(std::isfinite(want) && std::isfinite(op[t][r]),"nonfinite Down reference");
                double same_contract=want;
                if(fmt.d_type==42) {
                    // Q2_0's production ActQ retains FLOAT scales. GGML Q8_0
                    // stores HALF scales, so it is a different quantization
                    // contract. Independent scalar dot, same ActQ codes/scales;
                    // no vector unpack/reduction or production dot is reused.
                    same_contract=0;
                    const uint8_t* row=blobs[e]->data()+fmt.down_off+r*fmt.d_row;
                    for(int b=0;b<cpu::FF/64;++b) {
                        ggml_fp16_t wh;std::memcpy(&wh,row+b*18,2);
                        const double scale=ggml_fp16_to_fp32(wh);
                        for(int half=0;half<2;++half) {
                            int sum=0;
                            for(int j=0;j<32;++j) {
                                const int i=half*32+j;
                                const int code=((row[b*18+2+i/4]>>(2*(i%4)))&3)-1;
                                sum+=code*int(a2[t].q[b*64+i]);
                            }
                            same_contract+=scale*double(a2[t].scale[2*b+half])*sum;
                        }
                    }
                    ggn+=std::abs(double(op[t][r])-want);ggd+=std::abs(double(want));
                }
                dn+=std::abs(double(op[t][r])-same_contract); dd+=std::abs(same_contract);
                pn+=std::abs(double(jobs[e].out[t][r])-op[t][r]); pd+=std::abs(double(op[t][r]));
            }
            down_error=std::max(down_error,dn/(dd+1e-30)); pool_error=std::max(pool_error,pn/(pd+1e-30));
            q2_ggml_gap=std::max(q2_ggml_gap,ggn/(ggd+1e-30));
            ff.check(); out.check(); for(int t=0;t<nt;++t) hq[t]->check();
        }
        check(group,nt);
        if(!(gu_error<=1e-5 && down_error<=1e-5 && pool_error==0))
            std::cerr<<"GU="<<gu_error<<" Down="<<down_error<<" pool="<<pool_error<<" Q2/GGML-scale-gap="<<q2_ggml_gap<<std::endl;
        require(gu_error<=1e-5 && down_error<=1e-5 && pool_error==0, "numerical calibration gate");
        return {gu_error,down_error,pool_error,q2_ggml_gap};
    }
};
template<class F> void measure(F pass, uint64_t bytes, uint64_t visits, const std::string& fields,
                              cpu::ExpertPool* pool = nullptr, bool audit_host = false) {
    pass();
    double fastest=1e9;
    for(int i=0;i<5;++i) { const auto t=Clock::now(); pass(); fastest=std::min(fastest,seconds(t)); }
    const auto timer_start=Clock::now();
    for(int i=0;i<10000;++i) { const auto t=Clock::now(); (void)t; }
    const double timer_s=seconds(timer_start)/10000;
    require(fastest>0 && timer_s/fastest<.001,"per-pass timer overhead exceeds0.1%");
    std::cout<<"{\"kind\":\"timer_check\","<<fields<<",\"query_s\":"<<timer_s<<",\"fastest_pass_s\":"<<fastest<<"}"<<std::endl;
    // Integer complete passes, fixed minimum time. An unusually slow pilot
    // cannot select an undersized batch. Preserve the original >=1s gate.
    for(int r=0;r<reps;++r) {
        const double gu=pool?pool->ms_multi_gu:0, q=pool?pool->ms_multi_q:0, down=pool?pool->ms_multi_down:0;
        const auto pool_bytes = pool ? pool->multi_bytes : 0;
        const auto h0 = audit_host ? cpu_seconds(RUSAGE_THREAD) : 0;
        const auto c0=cpu_seconds(); const auto t0=Clock::now();
        uint64_t iterations=0; double elapsed;
        do { pass(); ++iterations; elapsed=seconds(t0); } while(elapsed<1.3 && iterations<1000000);
        const double wall=seconds(t0), process_cpu=cpu_seconds()-c0;
        const double host_cpu = audit_host ? cpu_seconds(RUSAGE_THREAD) - h0 : 0;
        require(wall>=1.0 && iterations<1000000,"timing floor/iteration budget");
        if (audit_host && pool)
            require(static_cast<uint64_t>(pool->multi_bytes - pool_bytes) == bytes * iterations,
                    "expert replay byte accounting");
        std::cout<<"{\"kind\":\"sample\","<<fields<<",\"attempt\":0,\"rep\":"<<r
            <<",\"iterations\":"<<iterations<<",\"visits\":"<<visits*iterations
            <<",\"requested_bytes\":"<<bytes*iterations<<",\"wall_s\":"<<wall<<",\"cpu_s\":"<<process_cpu;
        if(pool) std::cout<<",\"gu_s\":"<<(pool->ms_multi_gu-gu)*.001<<",\"q_s\":"<<(pool->ms_multi_q-q)*.001<<",\"down_s\":"<<(pool->ms_multi_down-down)*.001;
        if (audit_host) std::cout << ",\"host_cpu_s\":" << host_cpu;
        std::cout<<"}"<<std::endl;
    }
}
// Every loaded word contributes to a checksum. No stores to the measured
// buffer; workers persist across timed passes. This is a sequential-read
// reference, not physical DRAM traffic or a low-bit arithmetic upper bound.
__attribute__((noinline)) uint64_t scan(const uint64_t* p, size_t n) {
    __m256i v=_mm256_setzero_si256();
    for(size_t i=0;i<n;i+=4) v=_mm256_add_epi64(v,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i)));
    alignas(32) uint64_t a[4]; _mm256_store_si256(reinterpret_cast<__m256i*>(a),v);
    return a[0]+a[1]+a[2]+a[3];
}
void verify_placement(int host, std::vector<int> workers) {
    std::vector<int> observed;
    bool found_host = false;
    for (const auto& task : std::filesystem::directory_iterator("/proc/self/task")) {
        const int tid = std::stoi(task.path().filename().string());
        cpu_set_t mask;
        CPU_ZERO(&mask);
        require(sched_getaffinity(tid, sizeof mask, &mask) == 0 && CPU_COUNT(&mask) == 1,
                "replay thread has no single-CPU affinity");
        if (tid == getpid()) {
            require(CPU_ISSET(host, &mask), "replay caller affinity differs");
            found_host = true;
        } else {
            for (int id = 0; id < CPU_SETSIZE; ++id)
                if (CPU_ISSET(id, &mask)) observed.push_back(id);
        }
    }
    std::sort(workers.begin(), workers.end());
    std::sort(observed.begin(), observed.end());
    require(found_host && workers == observed, "replay worker affinity differs");
}

void report_placement(int host, const std::vector<int>& workers, bool caller_computes) {
    std::cout << "{\"kind\":\"replay_placement\",\"host_cpu\":" << host << ",\"workers\":[";
    for (size_t i = 0; i < workers.size(); ++i)
        std::cout << (i ? "," : "") << workers[i];
    std::cout << "],\"caller_computes\":" << (caller_computes ? "true" : "false") << "}" << std::endl;
}

void ram_scan(const std::vector<int>& cores, size_t bytes = 256ull << 20, int caller_core = -1,
              bool timing = true) {
    require(bytes >= 32 && bytes % 32 == 0, "scan layout");
    const bool replay = caller_core >= 0;
    const int host = replay ? caller_core : cores.front();
    const bool caller_reads = host == cores.front();
    const size_t first_worker = caller_reads ? 1 : 0;
    const size_t words = bytes / sizeof(uint64_t);
    std::vector<uint64_t> data(words); uint64_t expected=0;
    for(size_t i=0;i<words;++i) { data[i]=i*6364136223846793005ull+1442695040888963407ull; expected+=data[i]; }
    std::barrier phase(static_cast<ptrdiff_t>(cores.size() + !caller_reads));
    std::atomic<bool> stop{false}; std::vector<uint64_t> sums(cores.size()); std::vector<std::thread> workers;
    auto work=[&](size_t id) {
        pin(cores[id]);
        const size_t b=(words/4*id/cores.size())*4,e=(words/4*(id+1)/cores.size())*4;
        for(;;) { phase.arrive_and_wait(); if(stop.load()) break; sums[id]=scan(data.data()+b,e-b); phase.arrive_and_wait(); }
    };
    for(size_t i=first_worker;i<cores.size();++i) workers.emplace_back(work,i);
    pin(host);
    auto pass=[&] {
        phase.arrive_and_wait(); const size_t end=(words/4/cores.size())*4;
        if (caller_reads) sums[0] = scan(data.data(), end);
        phase.arrive_and_wait();
        require(std::accumulate(sums.begin(),sums.end(),uint64_t(0))==expected,"RAM checksum");
    };
    try {
        const std::vector<int> worker_cores(cores.begin() + first_worker, cores.end());
        if (replay) {
            pass();
            verify_placement(host, worker_cores);
            report_placement(host, worker_cores, caller_reads);
        }
        if (timing)
            measure(pass, bytes, 1, "\"mode\":\"scan\",\"threads\":" + std::to_string(cores.size()),
                    nullptr, replay);
        if (replay) verify_placement(host, worker_cores);
    }
    catch(...) { stop=true; phase.arrive_and_wait(); for(auto& t:workers)t.join(); throw; }
    stop=true; phase.arrive_and_wait(); for(auto& t:workers)t.join();
}

constexpr int replay_rows = 4, replay_experts = 512;

struct ReplayLayer {
    cpu::NativeFmt format;
    std::array<uint64_t, 3> offsets{};
    std::array<std::unique_ptr<Buffer>, replay_experts> weights;
    int fd = -1;

    ~ReplayLayer() { if (fd >= 0) close(fd); }

    void open_source(std::istream& in) {
        int gu = 0, down = 0;
        uint64_t bytes = 0;
        std::string path, error;
        in >> gu >> down >> bytes >> offsets[0] >> offsets[1] >> offsets[2] >> std::quoted(path);
        require(in.good() && (gu == 18 || gu == 21) && down == 20 && path.starts_with('/'),
                "replay layer schema");
        require(cpu::native_fmt(gu, down, cpu::H, cpu::FF, format, error), error.c_str());
        require(format.bytes == bytes &&
                std::ranges::all_of(offsets, [&](uint64_t offset) {
                    return offset <= std::numeric_limits<uint64_t>::max() - replay_experts * bytes;
                }), "replay weight layout");
        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        require(fd >= 0, "open replay model shard");
    }

    const uint8_t* weight(int expert) {
        auto& blob = weights[expert];
        if (!blob) {
            auto slice = std::make_unique<Buffer>(format.bytes);
            const auto gu = format.up_off;
            const auto down = format.bytes - format.down_off;
            pread_exact(fd, slice->data(), gu, offsets[0] + expert * gu);
            pread_exact(fd, slice->data() + gu, gu, offsets[1] + expert * gu);
            pread_exact(fd, slice->data() + format.down_off, down, offsets[2] + expert * down);
            blob = std::move(slice);
        }
        return blob->data();
    }
};

struct ReplayBatch {
    std::array<std::unique_ptr<Buffer>, replay_rows> acts;
    std::vector<std::unique_ptr<Buffer>> outputs;
    std::vector<cpu::ExpertJobMulti> jobs;
#ifdef STRATA_CPU_EXPERT_SCHEDULE_TEST
    std::vector<int> first_token;
#endif
    const cpu::NativeFmt* format = nullptr;
    int width = 0;

    void read(std::istream& in, std::array<ReplayLayer, 2>& layers) {
        int window = 0, layer_index = 0, count = 0;
        in >> window >> width >> layer_index >> count;
        require(in.good() && window >= 0 && window < 262144 && width >= 1 && width <= replay_rows &&
                layer_index >= 0 && layer_index < static_cast<int>(layers.size()) &&
                count >= 1 && count <= 10 * width, "replay batch geometry");
        auto& layer = layers[layer_index];
        format = &layer.format;
        uint32_t seed = 20261125u ^ (static_cast<uint32_t>(window) * 0x9e3779b9u + layer_index);
        for (int t = 0; t < width; ++t) {
            std::array<float, cpu::H> x;
            for (auto& value : x) {
                seed = 1664525u * seed + 1013904223u;
                value = (int(seed >> 8) - 8388608) * 0x1p-23f;
            }
            acts[t] = std::make_unique<Buffer>(format->act_bytes);
            cpu::native_quant_act(*format, x.data(), acts[t]->data());
        }
        std::array<bool, replay_experts> seen{};
        for (int i = 0; i < count; ++i) {
            int expert = 0;
            cpu::ExpertJobMulti job;
            in >> expert >> job.nt;
            require(in.good() && expert >= 0 && expert < replay_experts && !seen[expert] &&
                    job.nt > 0 && job.nt <= width, "replay job geometry");
            seen[expert] = true;
            job.blob = layer.weight(expert);
            auto out = std::make_unique<Buffer>(job.nt * cpu::H * sizeof(float));
            int previous_row = -1;
            for (int t = 0; t < job.nt; ++t) {
                int row = 0;
                in >> row;
                require(!in.fail() && row > previous_row && row < width, "replay job row order");
#ifdef STRATA_CPU_EXPERT_SCHEDULE_TEST
                if (t == 0) first_token.push_back(row);
#endif
                previous_row = row;
                job.nact[t] = acts[row]->data();
                job.out[t] = reinterpret_cast<float*>(out->data()) + t * cpu::H;
            }
            jobs.push_back(job);
            outputs.push_back(std::move(out));
        }
    }

    // The pool borrows these buffers until its synchronous retirement.
    void run(cpu::ExpertPool& pool, bool by_expert = false) {
#ifdef STRATA_CPU_EXPERT_SCHEDULE_TEST
        if (by_expert) {
            require(pool.run_decode_native_by_expert(*format, jobs.data(), static_cast<int>(jobs.size())),
                    "by-expert schedule refused the replay batch");
            return;
        }
#endif
        (void) by_expert;
        pool.run_split_multi_native(*format, jobs.data(), static_cast<int>(jobs.size()),
                                    cpu::NativeGuPhase::Decode);
    }

    void validate(cpu::ExpertPool& pool, bool by_expert = false) {
        for (const auto& job : jobs)
            for (int t = 0; t < job.nt; ++t)
                std::fill_n(job.out[t], cpu::H, std::numeric_limits<float>::quiet_NaN());
        run(pool, by_expert);
        for (const auto& job : jobs) {
            std::array<std::array<float, cpu::FF>, replay_rows> hidden;
            std::array<std::array<float, cpu::H>, replay_rows> expected;
            std::array<std::unique_ptr<Buffer>, replay_rows> quant;
            std::array<float*, replay_rows> hidden_rows{}, expected_rows{};
            std::array<const void*, replay_rows> quant_rows{};
            for (int t = 0; t < job.nt; ++t) {
                hidden_rows[t] = hidden[t].data();
                expected_rows[t] = expected[t].data();
                quant[t] = std::make_unique<Buffer>(format->h_bytes);
                quant_rows[t] = quant[t]->data();
            }
            cpu::native_gu_rows(*format, job.blob, job.nact, job.nt, hidden_rows.data(), 0, cpu::FF);
            for (int t = 0; t < job.nt; ++t)
                cpu::native_quant_h(*format, hidden_rows[t], quant[t]->data());
            cpu::native_down_rows(*format, job.blob, quant_rows.data(), job.nt, expected_rows.data(), 0, cpu::H);
            for (int t = 0; t < job.nt; ++t) {
                quant[t]->check();
                for (int row = 0; row < cpu::H; ++row)
                    require(std::isfinite(expected[t][row]) && job.out[t][row] == expected[t][row],
                            "replay pool differs from serial native output");
                require(std::memcmp(job.out[t], expected[t].data(), cpu::H * sizeof(float)) == 0,
                        "replay pool changes the serialized output bits");
            }
        }
        for (const auto& in : acts) if (in) in->check();
        for (const auto& out : outputs) out->check();
    }

#ifdef STRATA_CPU_EXPERT_SCHEDULE_TEST
    void validate_publication(cpu::ExpertPool& pool) {
        const int total = static_cast<int>(jobs.size());
        const int prefix = static_cast<int>(std::count_if(first_token.begin(), first_token.end(),
                                                         [](int token) { return token < 2; }));
        require(width == 4 && first_token.size() == jobs.size() && std::is_sorted(first_token.begin(), first_token.end())
                && 0 < prefix && prefix < total, "exact T4 publication prefix required");
        validate(pool, true);

        std::vector<std::vector<float>> expected;
        for (const auto& job : jobs) {
            auto& values = expected.emplace_back();
            for (int t = 0; t < job.nt; ++t) values.insert(values.end(), job.out[t], job.out[t] + cpu::H);
        }
        struct Readback {
            const std::vector<cpu::ExpertJobMulti>& jobs;
            const std::vector<std::vector<float>>& expected;
            int ready_jobs;
            int calls = 0;
            bool exact = true;

            bool matches(int count) const noexcept {
                for (int j = 0; j < count; ++j)
                    for (int t = 0; t < jobs[j].nt; ++t)
                        if (std::memcmp(jobs[j].out[t], expected[j].data() + t * cpu::H,
                                        cpu::H * sizeof(float)) != 0) return false;
                return true;
            }
        };
        const auto ready = +[](void* opaque) noexcept {
            auto& result = *static_cast<Readback*>(opaque);
            ++result.calls;
            result.exact &= result.matches(result.ready_jobs);
        };
        int cases = 0;
        for (int cutoff : {prefix, 0, total, prefix}) {
            for (bool after_all : {false, true}) {
                for (const auto& job : jobs)
                    for (int t = 0; t < job.nt; ++t)
                        std::fill_n(job.out[t], cpu::H, std::numeric_limits<float>::quiet_NaN());
                Readback check{jobs, expected, after_all ? total : cutoff};
                const cpu::ExpertPool::DecodePublication publication{cutoff, ready, &check, after_all};
                require(pool.run_decode_native_by_expert(*format, jobs.data(), total, &publication),
                        "publication schedule refused a captured layer");
                require(check.calls == 1 && check.exact && check.matches(total),
                        "publication callback or final result differs from serial native output");
                for (const auto& out : outputs) out->check();
                ++cases;
            }
        }
        for (const auto& in : acts) if (in) in->check();
        std::cout << "{\"kind\":\"publication_input\",\"jobs\":" << total << ",\"prefix_jobs\":" << prefix
                  << ",\"gu\":" << format->gu_type << ",\"logical_bytes\":" << total * format->bytes
                  << ",\"cases\":" << cases << ",\"bitwise\":true}" << std::endl;
    }
#endif
};

constexpr int locality_jobs = 8, locality_banks = 4, locality_observed_passes = 32;

void replay_locality(ReplayBatch& batch, cpu::ExpertPool& pool, int batch_index, bool rotating, bool timing) {
    require(batch.jobs.size() >= locality_jobs, "locality descriptor must supply eight jobs");
    batch.jobs.resize(locality_jobs);
    batch.outputs.resize(locality_jobs);
    batch.validate(pool);
    const uint64_t blob_bytes = batch.format->bytes;
    struct Bank {
        std::array<std::unique_ptr<Buffer>, locality_jobs> blobs;
        std::array<cpu::ExpertJobMulti, locality_jobs> jobs;
    };
    std::array<Bank, locality_banks> banks;
    std::array<std::vector<float>, locality_jobs> expected;
    int entries = 0;
    for (int e = 0; e < locality_jobs; ++e) {
        const auto& job = batch.jobs[e];
        expected[e].assign(job.out[0], job.out[0] + job.nt * cpu::H);
        entries += job.nt;
        for (auto& bank : banks) {
            bank.blobs[e] = std::make_unique<Buffer>(blob_bytes);
            std::memcpy(bank.blobs[e]->data(), job.blob, blob_bytes);
            bank.jobs[e] = job;
            bank.jobs[e].blob = rotating ? bank.blobs[e]->data() : banks[0].blobs[e]->data();
        }
    }
    auto run = [&](Bank& bank) {
        pool.run_split_multi_native(*batch.format, bank.jobs.data(), locality_jobs, cpu::NativeGuPhase::Decode);
    };
    auto pass = [&] { for (auto& bank : banks) run(bank); };
    auto validate = [&] {
        for (auto& bank : banks) {
            for (const auto& job : bank.jobs)
                std::fill_n(job.out[0], job.nt * cpu::H, std::numeric_limits<float>::quiet_NaN());
            run(bank);
            for (int e = 0; e < locality_jobs; ++e) {
                bank.blobs[e]->check();
                batch.outputs[e]->check();
                require(std::memcmp(bank.blobs[e]->data(), batch.jobs[e].blob, blob_bytes) == 0 &&
                        std::equal(expected[e].begin(), expected[e].end(), bank.jobs[e].out[0]),
                        "locality changed packed weights or serial output");
            }
        }
        for (const auto& act : batch.acts) if (act) act->check();
    };
    validate();
    const uint64_t work = locality_jobs * locality_banks;
    const char* arm = rotating ? "rotate" : "hot";
    std::cout << "{\"kind\":\"locality_input\",\"batch\":" << batch_index
              << ",\"layout\":\"" << arm << "\",\"gu\":" << batch.format->gu_type
              << ",\"down\":" << batch.format->d_type << ",\"jobs\":" << locality_jobs
              << ",\"entries\":" << entries << ",\"logical_bytes\":" << work * blob_bytes
              << ",\"allocated_bytes\":" << work * blob_bytes
              << ",\"address_bytes\":" << (rotating ? work : locality_jobs) * blob_bytes
              << ",\"serial_output_equal\":true}" << std::endl;
    const auto before = pool.service_snapshot();
    require(before.enabled != timing, "use worker census only for the untimed locality check");
    if (timing) {
        measure(pass, work * blob_bytes, work,
                "\"mode\":\"locality\",\"threads\":5,\"layout\":\"" + std::string(arm) + "\"", &pool, true);
    } else {
        for (int i = 0; i < locality_observed_passes; ++i) pass();
        const auto after = pool.service_snapshot();
        require(before.valid && after.valid, "invalid locality worker census");
        std::cout << "{\"kind\":\"locality_workers\",\"passes\":" << locality_observed_passes << ",\"slots\":[";
        bool comma = false;
        cpu::service::Counts full;
        for (size_t i = 0; i < after.threads.size(); ++i) {
            cpu::service::Counts count;
            require(cpu::service::subtract(after.threads[i].total, before.threads[i].total, count) &&
                    cpu::service::add(full, count), "inconsistent locality worker counts");
            if (count.calls == 0) continue;
            require(after.threads[i].tid > 0 && (i < 5 || i == cpu::service::kWorkers), "unexpected locality worker");
            std::cout << (comma ? "," : "") << "{\"slot\":" << i << ",\"calls\":" << count.calls
                      << ",\"rows\":" << count.rows << ",\"token_rows\":" << count.token_rows
                      << ",\"wall_ns\":" << count.wall_ns << ",\"cpu_ns\":" << count.cpu_ns << "}";
            comma = true;
        }
        require(full.rows == locality_observed_passes * work * (2 * cpu::FF + cpu::H) &&
                full.token_rows == locality_observed_passes * locality_banks * entries * (2 * cpu::FF + cpu::H),
                "locality worker census lost rows");
        std::cout << "]}" << std::endl;
    }
    validate();
}

void route_replay(const char* descriptor, const cpu::CpuTopology& topo, int threads, bool timing,
                  int locality_batch = -1, bool rotating = false, std::string_view schedule = {}) {
    const bool host_works = threads == 6;
    const int workers = threads - static_cast<int>(host_works);
    cpu::ExpertPool pool(workers, true, host_works);
    require(pool.workers() == workers && pool.host_works() == host_works, "replay pool topology");
    pin(topo.host_core);
    std::ifstream in(descriptor);
    std::string version;
    in >> version;
    require(in.good() && version == "strata-expert-replay-v1", "replay descriptor version");
    std::array<ReplayLayer, 2> layers;
    for (auto& layer : layers) layer.open_source(in);
    std::array<ReplayBatch, 8> batches;
    for (auto& batch : batches) batch.read(in, layers);
    in >> std::ws;
    require(in.eof(), "trailing replay instructions");

    const std::vector<int> worker_cores(topo.worker_cores.begin(), topo.worker_cores.begin() + workers);
#ifdef STRATA_CPU_EXPERT_SCHEDULE_TEST
    if (schedule == "publication") {
        verify_placement(topo.host_core, worker_cores);
        report_placement(topo.host_core, worker_cores, host_works);
        for (auto& batch : batches) if (batch.width == 4) batch.validate_publication(pool);
        for (const auto& layer : layers)
            for (const auto& blob : layer.weights) if (blob) blob->check();
        verify_placement(topo.host_core, worker_cores);
        return;
    }
#endif
    if (locality_batch >= 0) {
        verify_placement(topo.host_core, worker_cores);
        report_placement(topo.host_core, worker_cores, host_works);
        replay_locality(batches[locality_batch], pool, locality_batch, rotating, timing);
        for (const auto& layer : layers)
            for (const auto& blob : layer.weights) if (blob) blob->check();
        verify_placement(topo.host_core, worker_cores);
        return;
    }

    uint64_t jobs = 0, entries = 0, visits = 0, unique_blobs = 0, unique_bytes = 0;
    for (const auto& batch : batches) {
        jobs += batch.jobs.size();
        visits += batch.jobs.size() * batch.format->bytes;
        for (const auto& job : batch.jobs) entries += job.nt;
    }
    for (const auto& layer : layers)
        for (const auto& blob : layer.weights)
            if (blob) { ++unique_blobs; unique_bytes += blob->bytes; }
    const bool by_expert = schedule == "by-expert";
    const std::string schedule_field = schedule.empty() ? "" : ",\"schedule\":\"" + std::string(schedule) + "\"";
    auto validate = [&] {
        for (auto& batch : batches) batch.validate(pool, by_expert);
        for (const auto& layer : layers)
            for (const auto& blob : layer.weights) if (blob) blob->check();
        verify_placement(topo.host_core, worker_cores);
    };
    validate();
    report_placement(topo.host_core, worker_cores, host_works);
    std::cout << "{\"kind\":\"replay_input\",\"batches\":" << batches.size() << ",\"jobs\":" << jobs
              << ",\"entries\":" << entries << ",\"logical_bytes\":" << visits
              << ",\"unique_blobs\":" << unique_blobs << ",\"unique_bytes\":" << unique_bytes
              << ",\"threads\":" << threads << ",\"host_works\":" << (host_works ? "true" : "false")
              << ",\"serial_output_equal\":true,\"verified_affinity\":true" << schedule_field << '}' << std::endl;
    if (timing) {
        const auto fields = "\"mode\":\"replay\",\"threads\":" + std::to_string(threads) + schedule_field;
        const auto full_fields = schedule.empty() ? fields : fields + ",\"gu\":0,\"width\":0";
        measure([&] { for (auto& batch : batches) batch.run(pool, by_expert); },
                visits, jobs, full_fields, &pool, true);
        if (!schedule.empty()) {
            for (int gu : {18, 21}) for (int width : {2, 4}) {
                std::vector<ReplayBatch*> selected;
                uint64_t shape_bytes = 0, shape_jobs = 0;
                for (auto& batch : batches) {
                    if (batch.format->gu_type != gu || batch.width != width) continue;
                    selected.push_back(&batch);
                    shape_jobs += batch.jobs.size();
                    shape_bytes += batch.jobs.size() * batch.format->bytes;
                }
                require(selected.size() == 2, "expected two captured batches per scheduling shape");
                const auto shape_fields = fields + ",\"gu\":" + std::to_string(gu)
                    + ",\"width\":" + std::to_string(width);
                measure([&] { for (auto* batch : selected) batch->run(pool, by_expert); },
                        shape_bytes, shape_jobs, shape_fields, &pool, true);
            }
        }
        validate();
    }
}

int main(int argc,char** argv) try {
    std::cout<<std::setprecision(12);
    const auto topo=cpu::detect_cpu_topology(true);
    require(topo.host_core>=0 && topo.worker_cores.size()>=5,"six physical CPUs required");
#ifdef STRATA_CPU_EXPERT_SCHEDULE_TEST
    if (argc == 3 && std::string_view(argv[1]) == "replay-publication-check") {
        route_replay(argv[2], topo, 5, false, -1, false, "publication");
    } else if (argc == 5 && std::string_view(argv[1]) == "replay-schedule") {
        const std::string_view schedule(argv[3]), mode(argv[4]);
        require((schedule == "split" || schedule == "by-expert") &&
                (mode == "measure" || mode == "check"), "expert schedule replay arm");
        route_replay(argv[2], topo, 6, mode == "measure", -1, false, schedule);
    } else
#endif
    if (argc == 5 && std::string_view(argv[1]) == "replay") {
        const int threads = std::stoi(argv[3]);
        const std::string_view mode(argv[4]);
        require(threads >= 1 && threads <= 6 && (mode == "measure" || mode == "check"), "replay arm");
        route_replay(argv[2], topo, threads, mode == "measure");
    } else if (argc == 6 && std::string_view(argv[1]) == "replay-locality") {
        const int batch = std::stoi(argv[3]);
        const std::string_view layout(argv[4]), mode(argv[5]);
        require((batch == 4 || batch == 5) && (layout == "hot" || layout == "rotate") &&
                (mode == "measure" || mode == "check"), "locality replay arm");
        route_replay(argv[2], topo, 5, mode == "measure", batch, layout == "rotate");
    } else if (argc == 5 && std::string_view(argv[1]) == "replay-scan") {
        const int threads = std::stoi(argv[2]);
        const auto bytes = std::stoull(argv[3]);
        const std::string_view mode(argv[4]);
        require(threads >= 1 && threads <= 6 && bytes <= (512ull << 20) &&
                (mode == "measure" || mode == "check"), "replay scan arm");
        const int workers = threads == 6 ? 5 : threads;
        std::vector<int> cores(topo.worker_cores.begin(), topo.worker_cores.begin() + workers);
        if (threads == 6) cores.insert(cores.begin(), topo.host_core);
        pin(topo.host_core);
        ram_scan(cores, static_cast<size_t>(bytes), topo.host_core, mode == "measure");
    } else if(argc==3 && std::string(argv[1])=="scan") {
        std::vector<int> ids; std::string s=argv[2]; size_t at=0;
        while(at<s.size()) { size_t end=s.find(',',at); ids.push_back(std::stoi(s.substr(at,end-at))); if(end==std::string::npos)break; at=end+1; }
        require((ids.size()>=1 && ids.size()<=6) || ids.size()==12,"scan thread count");
        cpu_set_t allowed;CPU_ZERO(&allowed);
        require(sched_getaffinity(0,sizeof allowed,&allowed)==0,"scan allowed CPUs");
        require(std::all_of(ids.begin(),ids.end(),[&](int c){return c>=0 && c<CPU_SETSIZE && CPU_ISSET(c,&allowed);}),
                "scan CPU outside allowed affinity");
        auto unique=ids;std::sort(unique.begin(),unique.end());
        require(std::adjacent_find(unique.begin(),unique.end())==unique.end(),"duplicate scan CPU");
        ram_scan(ids);
    } else {
        require(argc==11,"expected shard gu down gate_offset up_offset down_offset bytes workers check|measure");
        Inputs input(argv[1],std::stoi(argv[2]),std::stoi(argv[3]),std::stoull(argv[4]),std::stoull(argv[5]),std::stoull(argv[6]),std::stoull(argv[7]));
        const int workers=std::stoi(argv[8]); const bool host=workers==5;
        require(workers==1 || workers==5,"worker arm");
        const bool timing=std::string(argv[9])=="measure"; require(timing || std::string(argv[9])=="check","mode");
        require(std::string(argv[10])=="20261020","schedule version");
        cpu::ExpertPool pool(workers,true,host); pin(topo.host_core);
        std::cout<<"{\"kind\":\"placement\",\"workers\":"<<pool.workers()<<",\"host_works\":"<<(host?"true":"false")<<",\"host_cpu\":"<<topo.host_core<<",\"worker_cpus\":[";
        for(int i=0;i<workers;++i) std::cout<<(i?",":"")<<topo.worker_cores[i]; std::cout<<"]}"<<std::endl;
        for(int nt=1;nt<=4;++nt) {
            const auto v=input.validate(pool,nt);
            std::cout<<"{\"kind\":\"validation\",\"nt\":"<<nt<<",\"gu_rel_l1\":"<<v[0]<<",\"down_rel_l1\":"<<v[1]<<",\"pool_rel_l1\":"<<v[2]<<",\"q2_ggml_scale_gap\":"<<v[3]<<",\"finite_canaries\":true}"<<std::endl;
        }
        std::vector<std::pair<int,bool>> cells;
        for(int nt=1;nt<=4;++nt) for(bool stream:{false,true}) cells.emplace_back(nt,stream);
        std::mt19937 rng(20261020); std::shuffle(cells.begin(),cells.end(),rng);
        if(timing) for(auto [nt,stream]:cells) {
            const int n=stream?input.count:group;
            std::vector<int> order(n/group); std::iota(order.begin(),order.end(),0); std::shuffle(order.begin(),order.end(),rng);
            for(auto& j:input.jobs) j.nt=nt;
            auto pass=[&] {
                for(int i:order) { input.quant(nt); pool.run_split_multi_native(input.fmt,input.jobs.data()+i*group,group); }
            };
            measure(pass,input.fmt.bytes*n,n,"\"mode\":\""+std::string(stream?"stream":"hot")+"\",\"nt\":"+std::to_string(nt)+",\"working_set_bytes\":"+std::to_string(input.fmt.bytes*n),&pool);
            input.check(n,nt); input.validate(pool,nt);
        }
    }
    std::cout<<"{\"kind\":\"closed\",\"passed\":true}"<<std::endl; return 0;
} catch(const std::exception& e) { std::cerr<<"CPU calibration failed: "<<e.what()<<std::endl; return 1; }
