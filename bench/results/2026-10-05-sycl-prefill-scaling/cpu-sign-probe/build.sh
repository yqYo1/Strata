#!/bin/bash
set -euo pipefail
root=/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05
probe=/home/yayoi/.local/state/strata-sycl/cpu-sign-probe
compiler=/opt/intel/oneapi/compiler/2026.1/bin/icpx
includes=(-I"$root/include" -I"$root/third_party/ggml" -I/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml/include)
flags=(-O3 -DNDEBUG -std=c++20 -fp-model=precise -mavx2 -mfma -mf16c)
"$compiler" "${flags[@]}" "${includes[@]}" -c "$probe/original.cpp" -o "$probe/original.o"
"$compiler" "${flags[@]}" "${includes[@]}" -c "$probe/wide.cpp" -o "$probe/wide.o"
"$compiler" "${flags[@]}" "${includes[@]}" "$probe/probe.cpp" "$probe/original.o" "$probe/wide.o" "$root/build-sycl-upstream-jit/ggml/src/libggml-cpu.a" "$root/build-sycl-upstream-jit/ggml/src/libggml-base.a" -lpthread -ldl -o "$probe/probe"
objdump -d -C "$probe/probe" > "$probe/disassembly.txt"
"$probe/probe" /home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf > "$probe/run.jsonl" 2> "$probe/run.stderr"
