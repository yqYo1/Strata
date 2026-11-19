// Main-authored CPU-only loader control. Repository LICENSE applies.
#include <cstdio>
#include <stdexcept>
#include <string_view>

#ifdef STRATA_WAIT_CANARY_PROVIDER
extern "C" void wait_provider(int* result, const int* mode)
    asm("_ZN4sycl3_V15queue20wait_and_throw_proxyERKNS0_6detail13code_locationE");
void wait_provider(int* result, const int* mode) {
    if (*mode < 0) throw std::runtime_error("canary dispatch failure");
    *result = 37 + *mode;
}
#else
extern "C" void strata_wait_canary_a(int*, const int*);
extern "C" void strata_wait_canary_b(int*, const int*);

// Match the direct PLT-entry/return structure using an independently known
// 16-byte stack delta. CFI also lets this exercise normal C++ unwinding.
asm(R"(
.text
.type strata_wait_canary_wrapper,@function
strata_wait_canary_wrapper:
  .cfi_startproc
  subq $8,%rsp
  .cfi_adjust_cfa_offset 8
  call _ZN4sycl3_V15queue20wait_and_throw_proxyERKNS0_6detail13code_locationE@PLT
.globl strata_wait_canary_return
strata_wait_canary_return:
  addq $8,%rsp
  .cfi_adjust_cfa_offset -8
  ret
  .cfi_endproc
.size strata_wait_canary_wrapper,.-strata_wait_canary_wrapper
.macro outside name
.globl strata_wait_canary_\name
.type strata_wait_canary_\name,@function
strata_wait_canary_\name:
  .cfi_startproc
  subq $8,%rsp
  .cfi_adjust_cfa_offset 8
  call strata_wait_canary_wrapper
.globl strata_wait_canary_parent_\name
strata_wait_canary_parent_\name:
  addq $8,%rsp
  .cfi_adjust_cfa_offset -8
  ret
  .cfi_endproc
.size strata_wait_canary_\name,.-strata_wait_canary_\name
.endm
outside a
outside b
)");

int main() {
    int answer = 0;
    int mode = 2;
    strata_wait_canary_a(&answer, &mode);
    if (answer != 39) return 1;
    mode = -1;
    try {
        strata_wait_canary_a(&answer, &mode);
        return 2;
    } catch (const std::runtime_error& error) {
        if (answer != 39 || std::string_view(error.what()) != "canary dispatch failure") return 3;
    }
    mode = 5;
    strata_wait_canary_b(&answer, &mode);
    if (answer != 42) return 4;
    std::puts("compat_wait_canary PASS value=42 exception=passed");
    return 0;
}
#endif
