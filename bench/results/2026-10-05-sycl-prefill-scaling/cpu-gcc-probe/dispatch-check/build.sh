#!/bin/bash
set -eo pipefail
source /opt/intel/oneapi/setvars.sh > /home/yayoi/.local/state/strata-sycl/cpu-gcc-dispatch-check/env.log 2>&1
root=/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05
probe=/home/yayoi/.local/state/strata-sycl/cpu-gcc-dispatch-check
compiler=/opt/intel/oneapi/compiler/2026.1/bin/icpx
"$compiler" -O3 -DNDEBUG -std=c++20 -fp-model=precise -mavx2 -mfma -mf16c -fsycl -I"$root/include" -I"$root/third_party/ggml" -I/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml/include "$probe/probe.cpp" "$probe/original.o" "$root/build-sycl-upstream-jit/libstrata_kernels_cpu.a" "$root/build-sycl-upstream-jit/ggml/src/libggml-cpu.a" "$root/build-sycl-upstream-jit/ggml/src/libggml-base.a" -lpthread -ldl -o "$probe/probe"
model=/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
"$probe/probe" "$model" 21 > "$probe/type21.jsonl" 2> "$probe/type21.stderr"
for round in 1 2 3; do
 "$probe/probe" "$model" 22 > "$probe/type22-round$round.jsonl" 2> "$probe/type22-round$round.stderr"
done
