"""Check actual MTP prefill source lifetime and async error handling on CPU.

The queue defers copies, while capture and submission failures are injected.
No kernels, GPU, driver calls or model arithmetic are exercised.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

from test_decode_graph_lease_host import complete_function


CONTROL = '990a631d0f2b7a14670d0808fd9572f0eefa7712'
PROGRAM = r'''
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
}
std::string mode;
struct Queue;
struct Event { Queue* queue; void wait_and_throw(); };
struct Queue {
    std::vector<std::function<void()>> jobs;
    int waits=0,launches=0;
    bool async_error=false;
    Event memcpy(void* destination,const void* source,size_t bytes) {
        if(mode=="upload-rejected" && jobs.empty() && launches==0)
            throw std::runtime_error("injected synchronous upload failure");
        jobs.push_back([=] { std::memcpy(destination,source,bytes); });
        return {this};
    }
    void wait() {
        ++waits;
        for(auto& job:jobs) job();
        jobs.clear();
    }
    void wait_and_throw() {
        wait();
        if(async_error) { async_error=false; throw std::runtime_error("injected async failure"); }
    }
    void ext_oneapi_graph(int) {
        if(++launches==2 && mode=="launch-failure")
            throw std::runtime_error("injected submission failure");
        if(mode=="async-final-error") async_error=true;
    }
};
void Event::wait_and_throw() { queue->wait_and_throw(); }
namespace dpct {
Queue queue;
Queue& get_in_order_queue() { return queue; }
const char* get_error_string_dummy(int) { return "injected error"; }
}
namespace sycl {
struct exception:std::runtime_error { using std::runtime_error::runtime_error; };
template<class T> T* malloc_device(size_t n,Queue&) { return new T[n]; }
template<class T> void free(T* p,Queue&) { delete[] p; }
}
namespace strata {
template<class T> T* checked_usm(T* p) { return p; }
}
#define DPCT_CHECK_ERROR(expr) [&](){ try { expr; return 0; } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }()
struct OnDevice { explicit OnDevice(int) {} };
class MtpDrafter {
public:
    struct Geometry { int64_t hc=1,n_embd=2,n_head=1; } geometry;
    Geometry* g_=&geometry;
    int device_=0,max_t_=4;
    int64_t window_=0,prompt_len_=0,pf_cap_=0;
    double ms_prefill=0;
    int32_t* pf_dev_=nullptr;
    Queue* cs_=&dpct::queue;
    int graphs[9]{};
    int* prefill_exec_[9]{}, *prefill_dev_exec_[9]{};
    int32_t tok[4]{},step[16]{},pos[4]{};
    float Rin[8]{};
    int32_t h_tok_[4]{},h_step_[16]{},h_pos_[4]{};
    int32_t* tok_=tok,*step_=step,*pos_=pos;
    float* Rin_=Rin;
    bool capture_prefill_dev(int t,std::string& e) {
        if(mode=="capture-failure") { e="injected capture failure"; return false; }
        prefill_dev_exec_[t]=&graphs[t]; return true;
    }
    bool capture_prefill(int t,std::string&) { prefill_exec_[t]=&graphs[t]; return true; }
#if CONTROL
    bool prefill(const float*,const int32_t*,int64_t,int64_t,std::string&);
#else
    bool prefill(const float*,const int32_t*,int64_t,int64_t,std::string&,bool=true);
#endif
    ~MtpDrafter() { assert(cs_->jobs.empty()); delete[] pf_dev_; }
};
#include "prefill.inc"
int main(int argc,char** argv) {
    assert(argc==2); mode=argv[1];
    MtpDrafter drafter; std::string error;
    float residual[16]{}; int32_t tokens[8]={1,2,3,4,5,6,7,8};
    bool ok=drafter.prefill(residual,tokens,8,0,error);
    if(mode=="healthy") assert(ok && error.empty());
    else if(mode=="async-final-error") {
#if CONTROL
        assert(ok && dpct::queue.async_error); // wait() missed the queued error.
#else
        assert(!ok && !error.empty() && !dpct::queue.async_error);
#endif
    } else assert(!ok && !error.empty());
    // The caller drains after return. The control's local rec no longer exists.
    dpct::queue.async_error=false; dpct::queue.wait_and_throw();
    std::printf("PASS %s: result=%d waits=%d launches=%d\n",mode.c_str(),ok,dpct::queue.waits,dpct::queue.launches);
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]; path='sycl/src/core/mtp.cpp'
    current=(root/path).read_text()
    old=subprocess.check_output(['git','show',CONTROL+':'+path],cwd=root,text=True)
    record=dict(scope=__doc__,control_commit=CONTROL,
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source_sha256=hashlib.sha256(current.encode()).hexdigest(),runs=[],passed=False)
    env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1:abort_on_error=0',
             UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',STRATA_MTP_PREFILL_SYNC='0')
    for variant,text,modes in [('control',old,['capture-failure','async-final-error']),
                              ('candidate',current,['healthy','capture-failure','launch-failure','upload-rejected','async-final-error'])]:
        out=args.output/variant; out.mkdir()
        (out/'prefill.inc').write_text(complete_function(text,'bool MtpDrafter::prefill(')+'\n')
        (out/'probe.cpp').write_text(PROGRAM)
        command=['g++','-std=c++20','-O1','-g','-fno-omit-frame-pointer','-fno-pie','-no-pie',
                 '-fsanitize=address,undefined','-DCONTROL='+str(int(variant=='control')),
                 str(out/'probe.cpp'),'-o',str(out/'probe')]
        build=subprocess.run(command,capture_output=True,text=True,timeout=30)
        (out/'build.stderr').write_text(build.stderr); assert build.returncode==0,build.stderr
        for mode in modes:
            run=subprocess.run([str(out/'probe'),mode],capture_output=True,text=True,env=env,timeout=5)
            (out/(mode+'.stdout')).write_text(run.stdout); (out/(mode+'.stderr')).write_text(run.stderr)
            expected_uaf=variant=='control' and mode=='capture-failure'
            passed=(run.returncode!=0 and 'AddressSanitizer: heap-use-after-free' in run.stderr) if expected_uaf else (run.returncode==0 and 'PASS' in run.stdout)
            record['runs'].append(dict(variant=variant,mode=mode,exit_code=run.returncode,expected_uaf=expected_uaf,passed=passed))
            (args.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
            assert passed,(variant,mode,run.stdout,run.stderr)
        if variant=='candidate':
            for mode in ['healthy','async-final-error']:
                sync_env=dict(env,STRATA_MTP_PREFILL_SYNC='1')
                run=subprocess.run([str(out/'probe'),mode],capture_output=True,text=True,env=sync_env,timeout=5)
                (out/('sync-'+mode+'.stdout')).write_text(run.stdout)
                (out/('sync-'+mode+'.stderr')).write_text(run.stderr)
                passed=run.returncode==0 and 'PASS' in run.stdout
                record['runs'].append(dict(variant=variant,mode=mode,per_group_sync=True,exit_code=run.returncode,passed=passed))
                (args.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
                assert passed,(variant,mode,run.stdout,run.stderr)
    record.update(passed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'passed':True,'runs':record['runs']},indent=2))


if __name__=='__main__': main()
