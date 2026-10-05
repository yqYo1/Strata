#!/bin/bash
set -eo pipefail
root=/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05
probe=/home/yayoi/.local/state/strata-sycl/cpu-down-scales-probe
pool=/home/yayoi/.local/state/strata-sycl/cpu-down-scales-fixed-pool-check
source /opt/intel/oneapi/setvars.sh > "$probe/production-check-env.log" 2>&1
compiler=/opt/intel/oneapi/compiler/2026.1/bin/icpx
# Call the actual production scale helper, including its wider-reduction fallback.
"$compiler" -O3 -DNDEBUG -std=c++20 -fp-model=precise -mavx2 -mfma -mf16c -fsycl \
 -I"$root/include" -I"$root/third_party/ggml" -I/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml/include \
 "$probe/fixed-production-probe.cpp" "$probe/production-reference.o" "$pool/production-scales.a" \
 "$root/build-sycl-upstream-jit/ggml/src/libggml-cpu.a" \
 "$root/build-sycl-upstream-jit/ggml/src/libggml-base.a" -lpthread -ldl -o "$probe/fixed-production-probe"
model=/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
"$probe/fixed-production-probe" "$model" > "$probe/fixed-production.jsonl" 2> "$probe/fixed-production.stderr"
"$compiler" -O3 -DNDEBUG -std=c++20 -fp-model=precise -mavx2 -mfma -mf16c -fsycl \
 -I"$root/include" -I/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml/include \
 -c "$pool/probe.cpp" -o "$pool/probe.o"
for arm in baseline scales; do
 "$compiler" -fsycl "$pool/probe.o" "$pool/production-$arm.a" \
  "$root/build-sycl-upstream-jit/ggml/src/libggml-cpu.a" \
  "$root/build-sycl-upstream-jit/ggml/src/libggml-base.a" -lpthread -ldl -o "$pool/pool-$arm"
done
python3 "$pool/run.py" "$pool"
