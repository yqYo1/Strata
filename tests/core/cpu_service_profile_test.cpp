// Collector correctness only: no kernel speed, model or GPU qualification.
#include "strata/kernels/cpu/service_profile.hpp"
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace strata::kernels::cpu;
using namespace strata::kernels::cpu::service;

static void require(bool value,const char* message) {
    if (!value) {std::fprintf(stderr,"cpu service test: %s\n",message);std::exit(1);}
}
static NativeFmt format(int gu=22,int down=20) {
    NativeFmt f;f.gu_type=gu;f.d_type=down;f.n_embd=2560;f.n_ff=640;return f;
}
static void work() {volatile unsigned sum=0;for (unsigned i=0;i<5000;++i) sum=sum+i;}
static Counts total(const Snapshot& s) {
    Counts n;for (const auto& c:s.cells) require(add(n,c),"sum overflow");return n;
}
static std::string output(const Snapshot& before,const Snapshot& after,bool expected) {
    FILE* f=std::tmpfile();require(f!=nullptr,"tmpfile");
    require(write_delta(f,9,before,after)==expected,"delta status");
    std::rewind(f);std::string text;char buf[1024];
    while (const size_t n=std::fread(buf,1,sizeof(buf),f)) text.append(buf,n);
    std::fclose(f);return text;
}

int main() {
    const auto fmt=format();
    {
        Profile p(false);{Profile::Scope s(p,900,fmt,Op::Gu,999,-1,NativeGuPhase::Decode);}
        const auto a=p.snapshot(true);require(!a.enabled && !a.valid && !total(a).calls,"disabled");
        require(output(a,a,true).empty(),"disabled emits nothing");
    }
    {
        Profile p(true);{Profile::BatchScope batch(p,NativeGuPhase::Prefill);
            Profile::Scope s(p,900,fmt,Op::Gu,999,-1,NativeGuPhase::Prefill);}
        const auto a=p.snapshot(true);require(a.valid && !total(a).calls,"prefill excluded before validation");
        require(!p.snapshot(false).valid && p.snapshot(false).error_mask==16,"nonquiescent snapshot refused");
    }
    {
        Profile p(true);const auto before=p.snapshot(true);
        std::vector<std::thread> workers;
        for (int id=0;id<4;++id) workers.emplace_back([&,id] {
            for (int nt=1;nt<=4;++nt) {
                {Profile::Scope s(p,id,fmt,Op::Gu,nt,160,NativeGuPhase::Decode);work();}
                {Profile::Scope s(p,id,fmt,Op::Down,nt,640,NativeGuPhase::Decode);work();}
            }
        });
        for (auto& t:workers) t.join();
        {Profile::Scope s(p,-1,fmt,Op::HiddenQuant,4,640,NativeGuPhase::Decode);work();}
        const auto after=p.snapshot(true);const auto n=total(after);
        require(after.valid && n.calls==33 && n.rows==13440 && n.token_rows==34560,"worker/width/row conservation");
        require(n.cpu_ns>0 && n.wall_ns>=n.cpu_ns,"nested clocks");
        require(after.threads[64].total.calls==1,"calling thread slot");
        const auto text=output(before,after,true);
        require(text.find("valid=1")!=std::string::npos && text.find("op=HQ")!=std::string::npos,"valid output");
        auto bad=after;--bad.threads[0].total.calls;
        const auto invalid=output(before,bad,false);
        require(invalid.find("valid=0")!=std::string::npos && invalid.find("CPU_SERVICE_BIN")==std::string::npos,
            "reject mismatched totals before any valid bins");
        bad=before;bad.cells[0].calls=1;
        require(output(bad,after,false).find("valid=0")!=std::string::npos,"nonmonotonic cells");
        bad=after;bad.threads[0].tid+=1;
        require(!write_delta(stderr,9,after,bad),"thread identity mismatch");
    }
    {
        Profile p(true);{Profile::Scope s(p,0,fmt,Op::Gu,1,640,NativeGuPhase::Decode);
            timespec delay{0,20000000};while (nanosleep(&delay,&delay)) {}}
        const auto a=p.snapshot(true);const auto n=total(a);
        require(a.valid && n.wall_ns>n.cpu_ns,"sleep remains wall, not CPU service");
    }
    {
        Profile p(true);std::atomic<bool> started{false};
        std::thread dispatch([&] {
            Profile::BatchScope batch(p,NativeGuPhase::Decode);
            {Profile::Scope s(p,-1,fmt,Op::HiddenQuant,1,640,NativeGuPhase::Decode);
                started.store(true,std::memory_order_release);
                timespec delay{0,20000000};while (nanosleep(&delay,&delay)) {}}
        });
        while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
        const auto a=p.snapshot(true);dispatch.join();
        require(a.valid && total(a).calls==1 && a.threads[64].total.calls==1,
            "separate generation reader waits for dedicated dispatch host publication");
        {Profile::BatchScope batch(p,NativeGuPhase::Decode);
            Profile::Scope s(p,-1,fmt,Op::HiddenQuant,1,640,NativeGuPhase::Decode);}
        require(!p.snapshot(true).valid && (p.snapshot(true).error_mask&64),"slot owner change rejected");
    }
    for (int which=0;which<6;++which) {
        Profile p(true);auto f=fmt;if (which==0) f.gu_type=42;if (which==1) f.n_ff=641;
        {Profile::Scope s(p,which==5?64:0,f,Op::Gu,which==2?5:1,which==3?641:which==4?0:640,NativeGuPhase::Decode);}
        require(!p.snapshot(true).valid,"unsupported tuple/shape/width/rows/worker rejected");
    }
    {
        Counts a{1,2,3,4,std::numeric_limits<uint64_t>::max()},b{1,1,1,1,1};
        require(!add(a,b) && a.calls==1 && a.rows==2,"overflow leaves every field untouched");
        Counts out{9,9,9,9,9};require(!subtract(b,a,out) && out.calls==9,"underflow leaves output untouched");
    }
    std::puts("cpu_service_profile counters,phase isolation,thread ownership,clocks,refusals=PASS");
}
