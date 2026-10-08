#!/bin/bash
set -eo pipefail
root=/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05
probe=/home/yayoi/.local/state/strata-sycl/cpu-ggml-dot-gcc-probe/pool-iq3xxs
ggml=/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml
mkdir -p "$probe"
source /opt/intel/oneapi/setvars.sh > "$probe/env.log" 2>&1
/usr/bin/python3 "$root/bench/results/2026-10-05-sycl-prefill-scaling/cpu-ggml-dot-gcc-probe/prepare_pool.py"
compiler=/opt/intel/oneapi/compiler/2026.1/bin/icpx
"$compiler" -O3 -DNDEBUG -std=c++20 -fp-model=precise -mavx2 -mfma -mf16c -fsycl \
 -I"$root/include" -I"$ggml/include" -c "$probe/probe.cpp" -o "$probe/probe.o"
for arm in baseline iq3xxs; do
 option=()
 if [ "$arm" = iq3xxs ]; then option=(-DSELECT_IQ3XXS=1); fi
 "$compiler" -O3 -DNDEBUG -std=c++20 -fp-model=precise -mavx2 -mfma -mf16c \
  -I"$ggml/include" "${option[@]}" -c "$probe/wrap.cpp" -o "$probe/wrap-$arm.o"
 "$compiler" -fsycl "$probe/probe.o" "$probe/wrap-$arm.o" "$probe/alternate_quants.o" \
  "$probe/production-$arm.a" "$root/build-sycl-upstream-jit/ggml/src/libggml-cpu.a" \
  "$root/build-sycl-upstream-jit/ggml/src/libggml-base.a" \
  -Wl,--wrap=ggml_get_type_traits_cpu -lpthread -ldl -o "$probe/pool-$arm"
done
/usr/bin/python3 "$probe/run.py" "$probe"
