"""Exercise production shutdown code and removed waits without a GPU."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def function(source, signature):
    start = source.index(signature)
    end = source.index('{', start) + 1
    depth = 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def footer(source):
    start = source.rindex('    strata::core::session_graphs_free(gr);')
    start = source.rfind('\n\n', 0, start) + 2
    return source[start:source.index('\n}', start)]


SHUTDOWN = r'''
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include "strata/sycl_error.hpp"
struct Device {
    std::vector<std::function<void()>> jobs;
    bool fail = false;
    int waits = 0;
    void queues_wait_and_throw() {
        ++waits;
        if (fail) throw std::runtime_error("injected queue error");
        for (auto& job : jobs) job();
        jobs.clear();
    }
};
namespace dpct {
Device devices[2];
unsigned int device_count() { return 2; }
Device& get_device(unsigned int n) { return devices[n]; }
Device& get_in_order_queue() { return devices[0]; }
}
namespace strata::core {
void session_graphs_free(int) {}
void doorbell_free(int) {}
}
namespace sycl {
void free(float* p, Device&) { std::puts("FREE"); delete[] p; }
}
#include "drain.inc"
float *d_next, *d_logits, *d_emb, *d_parts, *sbuf, *arena;
int cli_footer() {
    int gr=0, db=0;
#include "footer.inc"
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    assert(argc==2);
    std::string mode=argv[1];
    d_next=new float[1]{1}; d_logits=new float[1]{2};
    d_emb=new float[1]{3}; d_parts=new float[1]{4};
    sbuf=new float[1]{5}; arena=new float[1]{6};
    unsigned int device=mode=="primary" ? 0 : 1;
    dpct::devices[device].jobs.push_back([p=arena] { assert(p[0]==6); });
    if (mode=="failed-drain") dpct::devices[1].fail=true;
    assert(cli_footer()==0);
    // Old production code leaves this read queued until after arena reclamation.
    for (auto& d : dpct::devices) for (auto& job : d.jobs) job();
    assert(dpct::devices[0].waits==1 && dpct::devices[1].waits==1);
    std::puts("PASS completion before reclamation");
}
'''

SPIN = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#define __dpct_inline__ inline
namespace strata {
constexpr uint32_t kSpinMax=20000;
unsigned int reads=0;
uint32_t sys_load(const volatile uint32_t* p) { ++reads; return *p; }
}
namespace sycl {
enum class memory_order { acq_rel };
enum class memory_scope { system };
void atomic_fence(memory_order, memory_scope) {}
}
void strata_spin_pause() {}
#include "waits.inc"
int main(int argc, char** argv) {
    assert(argc==2);
    std::string mode=argv[1];
    uint32_t flag=0, wanted=1, skip=0;
#if CONTROL
    if (mode=="ge") wait_flag_ge_kernel(&flag,wanted);
    else if (mode=="ge-or") wait_flag_ge_or_kernel(&flag,wanted,&skip);
    else doorbell_wait_kernel(&flag,&wanted);
    assert(flag<wanted && strata::reads>=strata::kSpinMax);
    std::puts("PASS old wait returned with flag unsatisfied");
#else
    bool rejected=false;
    try {
        if (mode=="ge") wait_flag_ge(&flag,wanted,nullptr);
        else if (mode=="ge-or") wait_flag_ge_or(&flag,wanted,&skip,nullptr);
        else doorbell_wait(&flag,&wanted,nullptr);
    } catch (const std::logic_error&) { rejected=true; }
    assert(rejected && strata::reads==0);
    std::puts("PASS legacy wait rejected before queue access");
#endif
}
'''

POLICY = r'''
#include "strata/sycl_execution_policy.hpp"
#include <cassert>
#include <cstdio>
int main() {
    std::string error;
    unsetenv("STRATA_VERIFY_NO_HOST"); unsetenv("STRATA_SYCL_HOST_BOUNDARY");
    assert(strata::require_sycl_host_boundaries(error));
    for (const char* value : {"1", "2"}) {
        setenv("STRATA_SYCL_HOST_BOUNDARY",value,1);
        assert(strata::require_sycl_host_boundaries(error));
    }
    for (const char* value : {"0", "", "bad"}) {
        setenv("STRATA_SYCL_HOST_BOUNDARY",value,1);
        assert(!strata::require_sycl_host_boundaries(error) && !error.empty());
    }
    unsetenv("STRATA_SYCL_HOST_BOUNDARY");
    for (const char* value : {"0", "1", ""}) {
        setenv("STRATA_VERIFY_NO_HOST",value,1);
        assert(!strata::require_sycl_host_boundaries(error));
    }
    std::puts("PASS default/explicit boundaries and all legacy selectors");
}
'''

DESCRIPTOR = r'''
#include <cassert>
#include <cstdio>
#include <stdexcept>
int default_calls=0;
bool fail_default=false;
namespace sycl { struct queue { int value; }; }
namespace dpct::cs {
using queue_ptr=sycl::queue*;
sycl::queue& get_default_queue() {
    ++default_calls;
    if (fail_default) throw std::runtime_error("injected device selection failure");
    static sycl::queue q{41}; return q;
}
}
enum class math_mode { mm_default };
#include "descriptor.inc"
int main() {
#if CONTROL
    assert(default_calls==1);
    std::puts("PASS old descriptor initialized the device before main");
#else
    assert(default_calls==0);
    assert(descriptor::get_saved_queue().value==41 && default_calls==1);
    sycl::queue explicit_queue{42};
    descriptor::set_saved_queue(&explicit_queue);
    assert(descriptor::get_saved_queue().value==42 && default_calls==1);
    descriptor::set_saved_queue(nullptr);
    fail_default=true;
    bool reported=false;
    try { descriptor::get_saved_queue(); } catch (const std::runtime_error&) { reported=true; }
    assert(reported);
    std::puts("PASS lazy default, explicit queue, and propagated initialization error");
#endif
}
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    control = '60fc342df60ec970b60c54a3d07a3f7a53273bdd'
    paths = ['sycl/src/program/generate.cpp', 'sycl/src/kernels/cuda/verify_kernels.dp.cpp',
             'sycl/src/kernels/cuda/elementwise.dp.cpp']
    sources = {
        'control': [subprocess.check_output(['git', 'show', control + ':' + p], cwd=root, text=True) for p in paths],
        'candidate': [(root / p).read_text() for p in paths],
    }
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1:abort_on_error=1',
               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    record = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  control_commit=control, scope='Actual CLI footer/drain and wait fragments; CPU queue stand-ins, no GPU',
                  sources={}, runs=[], passed=False)

    def run_probe(variant, group, fragments, program, modes):
        out = args.output / variant / group
        out.mkdir(parents=True)
        for name, text in fragments.items():
            (out / name).write_text(text+'\n')
        (out / 'probe.cpp').write_text(program)
        command = ['g++', '-std=c++20', '-O1', '-g', '-fno-omit-frame-pointer', '-fno-pie', '-no-pie',
                   '-fsanitize=address,undefined', '-DCONTROL='+str(int(variant=='control')),
                   '-I'+str(root/'sycl/include'), '-I'+str(out), str(out/'probe.cpp'), '-o', str(out/'probe')]
        build = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (out/'build.stderr').write_text(build.stderr)
        assert build.returncode==0, build.stderr
        for mode in modes:
            run = subprocess.run([str(out/'probe')]+([mode] if mode else []), capture_output=True, text=True, env=env, timeout=5)
            label=mode or 'policy'
            (out/(label+'.stdout')).write_text(run.stdout)
            (out/(label+'.stderr')).write_text(run.stderr)
            if group=='shutdown' and variant=='control':
                passed=run.returncode!=0 and 'AddressSanitizer: heap-use-after-free' in run.stderr
            elif mode=='failed-drain':
                passed=run.returncode==1 and 'FREE' not in run.stdout and 'shutdown failed' in run.stderr
            else:
                passed=run.returncode==0 and 'PASS' in run.stdout
            record['runs'].append(dict(variant=variant, group=group, mode=label, exit_code=run.returncode, expected_result=passed))
            (args.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
            assert passed, (variant, group, mode, run.stdout, run.stderr)

    for variant, (main, verify, elementwise) in sources.items():
        record['sources'][variant]=dict(zip(paths,[hashlib.sha256(s.encode()).hexdigest() for s in (main,verify,elementwise)]))
        drain=function(main,'static void drain_sycl_devices()') if variant=='candidate' else ''
        run_probe(variant,'shutdown',{'drain.inc':drain,'footer.inc':footer(main)},SHUTDOWN,
                  ['primary','secondary']+(['failed-drain'] if variant=='candidate' else []))
        if variant=='control':
            waits='\n'.join([function(verify,'__dpct_inline__ void wait_flag_ge_kernel('),
                             function(verify,'__dpct_inline__ void wait_flag_ge_or_kernel('),
                             function(elementwise,'__dpct_inline__ void doorbell_wait_kernel(')])
        else:
            waits='\n'.join([function(verify,'void wait_flag_ge('), function(verify,'void wait_flag_ge_or('),
                             function(elementwise,'void doorbell_wait(')])
        run_probe(variant,'waits',{'waits.inc':waits},SPIN,['ge','ge-or','doorbell'])
        path='sycl/include/dpct/blas_utils.hpp'
        descriptor=(subprocess.check_output(['git','show',control+':'+path],cwd=root,text=True)
                    if variant=='control' else (root/path).read_text())
        record['sources'][variant][path]=hashlib.sha256(descriptor.encode()).hexdigest()
        run_probe(variant,'descriptor',{'descriptor.inc':function(descriptor,'class descriptor {')+';'},DESCRIPTOR,[''])
    run_probe('candidate','policy',{},POLICY,[''])
    record.update(passed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
