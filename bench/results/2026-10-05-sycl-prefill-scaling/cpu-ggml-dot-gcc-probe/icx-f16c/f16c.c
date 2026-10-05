#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"
#include "ggml-cpu/quants.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include <immintrin.h>
#undef GGML_CPU_FP16_TO_FP32
#define GGML_CPU_FP16_TO_FP32(x) _cvtsh_ss(x)
#include "/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml/src/ggml-cpu/arch/x86/quants.c"
