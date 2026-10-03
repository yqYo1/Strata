set -e
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -uo pipefail
icpx -O3 -DNDEBUG -std=c++20 -march=znver3 -mtune=znver3 -mavx2 -mfma -mf16c -Iinclude -Ithird_party/ggml -I../ggml-sycl-pinned/ggml/include -c /tmp/strata-sycl-goal-iq-looped-isolated.cpp -o /tmp/strata-sycl-goal-iq-looped-isolated.o
icpx -O3 -DNDEBUG -std=c++20 -march=znver3 -mtune=znver3 -mavx2 -mfma -mf16c -Iinclude -Ithird_party/ggml -I../ggml-sycl-pinned/ggml/include /tmp/strata-sycl-goal-iq-gather-bench.cpp /tmp/strata-sycl-goal-iq-looped-isolated.o build-sycl-aot/CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/iq_avx2.cpp.o -o /tmp/strata-sycl-goal-iq-looped-isolated-bench
