// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/prefill/issuer_timing.hpp"
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
using namespace strata::prefill;
static void check(bool v) { if (!v) { std::fputs("issuer_timing_test: failed\n", stderr); std::exit(1); } }
struct FakeClock {
    using time_point=std::chrono::time_point<FakeClock,std::chrono::nanoseconds>;
    static constexpr bool is_steady=true;
    static inline int calls=0;
    static inline int64_t begin=0,end=10;
    static time_point now() { return time_point(std::chrono::nanoseconds((calls++%2) ? end : begin)); }
};
using Timing=BasicIssuerTiming<FakeClock>;
template<class T> std::string closed(T& t,uint64_t expected,bool threaded=false) {
    FILE* f=std::tmpfile();check(f!=nullptr);
    t.emit(0,0,1,0,0,8192,true,threaded,16,expected,f);
    check(std::fflush(f)==0 && std::fseek(f,0,SEEK_SET)==0);
    std::string s;char buf[2048];while(std::fgets(buf,sizeof buf,f))s+=buf;
    check(std::feof(f) && std::fclose(f)==0);check(s.size()<=4096);return s;
}
static bool field(const std::string& s,const char* value) { return s.find(std::string(" ")+value+" ")!=std::string::npos; }
int main() {
    for (const char* value:{static_cast<const char*>(nullptr),"","0","2","01","true"}) check(!Timing::enabled_value(value));
    check(Timing::enabled_value("1"));
    Timing off(false,std::numeric_limits<size_t>::max());int called=0;
    check(off.measure(999,Timing::TransientStaged,true,[&]{++called;return 42;})==42);
    off.entry(999,Timing::TransientStaged,1);check(called==1 && FakeClock::calls==0 && closed(off,0).empty());
    Timing direct(true,1);check(direct.measure(0,Timing::PinnedDirect,false,[]{return true;}));
    direct.entry(0,Timing::PinnedDirect,17);auto d=closed(direct,1);
    check(field(d,"entries=1") && field(d,"expert_bytes=17") && field(d,"wait_count=0"));
    check(field(d,"submit_sum_ns=10") && field(d,"counter_overflow=0") && field(d,"coverage_complete=1"));
    Timing staged(true,1);check(staged.measure(0,Timing::TransientStaged,true,[]{return 2;})==2);
    check(staged.measure(0,Timing::TransientStaged,false,[]{return true;}));staged.entry(0,Timing::TransientStaged,128);
    auto s=closed(staged,1);check(field(s,"wait_count=1") && field(s,"wait_sum_ns=10") && field(s,"host_observation_eligible=1"));
    Timing absent(true,1);auto a=closed(absent,0);check(field(a,"coverage_complete=1") && !field(a,"host_observation_eligible=1"));
    Timing missing(true,1);check(!field(closed(missing,1),"coverage_complete=1"));
    Timing regressing(true,1);FakeClock::begin=20;FakeClock::end=10;
    check(regressing.measure(0,Timing::PinnedDirect,false,[]{return true;}));regressing.entry(0,Timing::PinnedDirect,4);
    check(field(closed(regressing,1),"clock_valid=0"));FakeClock::begin=0;FakeClock::end=10;
    Timing byte_overflow(true,1);
    for (const auto n:{std::numeric_limits<size_t>::max(),size_t(1)}) {
        check(byte_overflow.measure(0,Timing::PinnedDirect,false,[]{return true;}));byte_overflow.entry(0,Timing::PinnedDirect,n);
    }
    auto ov=closed(byte_overflow,2);check(field(ov,"counter_overflow=1") && field(ov,"accounting_valid=0") && field(ov,"coverage_complete=0"));
    Timing duration_overflow(true,1);FakeClock::end=std::numeric_limits<int64_t>::max();
    for (int i=0;i<3;++i) {check(duration_overflow.measure(0,Timing::PinnedDirect,false,[]{return true;}));duration_overflow.entry(0,Timing::PinnedDirect,1);}
    check(field(closed(duration_overflow,3),"counter_overflow=1"));FakeClock::end=10;
    Timing failed(true,1);int attempts=0;bool caught=false;
    try { failed.measure(0,Timing::PinnedDirect,false,[&]()->bool{++attempts;throw std::runtime_error("expected");}); }
    catch(const std::runtime_error&) {caught=true;}
    check(caught && attempts==1 && !field(closed(failed,1),"coverage_complete=1"));
    IssuerTiming joined(true,1);
    std::thread producer([&]{for(int i=0;i<3;++i) {check(joined.measure(0,IssuerTiming::PinnedDirect,false,[]{return true;}));joined.entry(0,IssuerTiming::PinnedDirect,7);}});
    producer.join();auto j=closed(joined,3,true);check(field(j,"threaded_issue=1") && field(j,"entries=3") && field(j,"expert_bytes=21") && field(j,"coverage_complete=1"));
    std::puts("issuer_timing_test: disabled/no-clock, direct/staged, post-join, missing/empty, exception, clock regression, byte/duration overflow PASS");
}
