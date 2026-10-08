#!/bin/bash
set -eo pipefail
root=/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05
probe=/home/yayoi/.local/state/strata-sycl/cpu-gcc-pool-probe
source /opt/intel/oneapi/setvars.sh > "$probe/env.log" 2>&1
compiler=/opt/intel/oneapi/compiler/2026.1/bin/icpx
"$compiler" -O3 -DNDEBUG -std=c++20 -fp-model=precise -mavx2 -mfma -mf16c -fsycl \
 -I"$root/include" -I/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml/include \
 -c "$probe/probe.cpp" -o "$probe/probe.o"
for arm in default gcc; do
 "$compiler" -fsycl "$probe/probe.o" "$probe/production-$arm.a" \
  "$root/build-sycl-upstream-jit/ggml/src/libggml-cpu.a" \
  "$root/build-sycl-upstream-jit/ggml/src/libggml-base.a" \
  -lpthread -ldl -o "$probe/pool-$arm"
done
python3 "$probe/run.py" "$probe"
