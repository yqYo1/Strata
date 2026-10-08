
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
