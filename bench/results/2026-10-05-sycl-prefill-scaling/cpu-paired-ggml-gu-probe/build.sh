#!/bin/bash
set -eo pipefail
root=/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05
probe=/home/yayoi/.local/state/strata-sycl/cpu-paired-ggml-gu-probe
ggml=/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml
mkdir -p "$probe"
source /opt/intel/oneapi/setvars.sh > "$probe/env.log" 2>&1
/usr/bin/python3 "$root/bench/results/2026-10-05-sycl-prefill-scaling/cpu-paired-ggml-gu-probe/generate.py"
for arm in icx gcc; do
 if [ "$arm" = icx ]; then compiler=/opt/intel/oneapi/compiler/2026.1/bin/icx; math=(-fp-model=precise); else compiler=gcc; math=(-ffp-contract=fast); fi
 "$compiler" -O3 -DNDEBUG -std=gnu11 -march=native -fno-associative-math "${math[@]}" \
  -DGGML_SCHED_MAX_COPIES=4 -DGGML_USE_CPU_REPACK -D_GNU_SOURCE -D_XOPEN_SOURCE=600 \
  -I"$ggml" -I"$ggml/src" -I"$ggml/src/ggml-cpu" -I"$ggml/include" \
  -c "$probe/paired.c" -o "$probe/paired-$arm-original.o"
 nm -g --defined-only "$probe/paired-$arm-original.o" | awk -v prefix="$arm" '{print $3, prefix "_" $3}' > "$probe/symbols-$arm.txt"
 objcopy --redefine-syms="$probe/symbols-$arm.txt" "$probe/paired-$arm-original.o" "$probe/paired-$arm.o"
done
compiler=/opt/intel/oneapi/compiler/2026.1/bin/icpx
for file in rows probe; do
 "$compiler" -O3 -DNDEBUG -std=c++20 -fp-model=precise -mavx2 -mfma -mf16c -fsycl \
  -I"$root/include" -I"$ggml/include" -c "$probe/$file.cpp" -o "$probe/$file.o"
done
"$compiler" -fsycl "$probe/probe.o" "$probe/rows.o" "$probe/paired-icx.o" "$probe/paired-gcc.o" \
 "$root/build-sycl-upstream-jit/libstrata_kernels_cpu.a" \
 "$root/build-sycl-upstream-jit/ggml/src/libggml-cpu.a" \
 "$root/build-sycl-upstream-jit/ggml/src/libggml-base.a" -lpthread -ldl -o "$probe/probe"
/usr/bin/python3 "$root/bench/results/2026-10-05-sycl-prefill-scaling/cpu-paired-ggml-gu-probe/run.py"
