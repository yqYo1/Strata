#!/bin/bash
set -eo pipefail
root=/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05
probe=/home/yayoi/.local/state/strata-sycl/cpu-ggml-dot-gcc-probe
ggml=/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml
mkdir -p "$probe"
source /opt/intel/oneapi/setvars.sh > "$probe/env.log" 2>&1
/usr/bin/python3 "$root/bench/results/2026-10-05-sycl-prefill-scaling/cpu-ggml-dot-gcc-probe/prepare.py"
/opt/intel/oneapi/compiler/2026.1/bin/icpx -O3 -DNDEBUG -std=c++20 -fp-model=precise -mavx2 -mfma -mf16c \
 -I"$root/include" -I"$ggml/include" -I"$ggml/src" \
 "$probe/probe.cpp" "$probe/reference_quants.o" "$probe/gcc_quants.o" \
 "$root/build-sycl-upstream-jit/ggml/src/libggml-cpu.a" \
 "$root/build-sycl-upstream-jit/ggml/src/libggml-base.a" -lpthread -ldl -o "$probe/probe"
model=/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
for round in 1 2 3; do
 "$probe/probe" "$model" > "$probe/round$round.jsonl" 2> "$probe/round$round.stderr"
done
