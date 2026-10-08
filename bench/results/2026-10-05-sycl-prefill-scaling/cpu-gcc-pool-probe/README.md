# CPU expert pool: selective GCC IQ2_S gate/up

Ryzen 5 5600X / 128 GiB RAM, precise oneAPI 2026.1, GCC 13.3, real mixed
GSQ-RCO IQ3_S Qwen3.8 Flash Next weights, 2026-10-05. This CPU-only harness
links the actual production CPU archives with the option off/on. It runs the
engine's `ExpertPool::run_split_multi_native`: gate/up, intermediate
quantization and down, with five pinned workers plus the pinned host thread.
No SYCL queue is created. No TG or GPU-prefill speed is measured here.

Both archives are linked to the same compiled harness object. Host CPU is 0;
worker CPUs are 1–5. Geometry is 2560 / 640. The default rounding/group-size
policy is retained. Only two/four-token IQ2_S gate/up calls select GCC.

## Correctness

The two executions compare 34,412,784 finite FP32 values bit-for-bit,
including untouched inactive tokens and per-output guards. There are 396
cases: all 48 native layers, experts 0/7/31, 1–8 tokens; then mixed groups and
changing batches of 1/2/12/13/96/97 experts through one live pool. The 97-expert
cases cross the pool's 96-entry internal buffering boundary. Both IQ4_NL and
Q2_0 down paths are exercised.

Each complete dump is 137,651,136 bytes, SHA-256
`092acfb9cb9d0b96931490e96d709afd53b56f95fff07e9e33da252555489482`.
Every timed dataset also has an actual byte comparison after the first full
pool call. All 96 dataset comparisons pass. Dumps, binaries and archives are
preserved outside `/tmp` in `~/.local/state/strata-sycl/cpu-gcc-pool-probe`;
their hashes are in `manifest.json`. Large binaries/dumps are not in Git.

## Full pool timing

Each row has 27 alternating-order pairs over three fresh process pairs. One
timing contains 96 expert evaluations: eight repetitions of 12 experts, or
two of 48. The 12-expert dataset fits within the 32 MiB L3; 48 expert weights
are 69–107 MiB. Before each timing the other pool is allowed to reach its
default 20 ms sleep boundary, and the measured pool executes a warm call.
All timings and outliers are retained.

Median paired speed ratio, original time / candidate time, including all
three expert phases and pool dispatch:

| Gate/up + down | Experts | 1 token | 2 tokens | 4 tokens | Mixed 1/2/3/4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ2_S + Q2_0, layer 1 | 12 | 0.966 | 1.114 | 1.152 | 1.074 |
| IQ2_S + Q2_0, layer 1 | 48 | 1.036 | 1.139 | 1.177 | 1.044 |
| IQ2_S + IQ4_NL, layer 2 | 12 | 0.981 | 1.076 | 1.068 | 1.100 |
| IQ2_S + IQ4_NL, layer 2 | 48 | 0.943 | 1.133 | 1.168 | 1.024 |
| IQ3_S + IQ4_NL, layer 17 | 12 | 0.993 | 1.032 | 1.025 | 1.003 |
| IQ3_S + IQ4_NL, layer 17 | 48 | 0.979 | 0.998 | 0.982 | 1.022 |
| IQ3_S + Q2_0, layer 21 | 12 | 1.006 | 1.031 | 0.960 | 0.990 |
| IQ3_S + Q2_0, layer 21 | 48 | 0.990 | 0.997 | 0.993 | 1.023 |

The model-load control PID 1074147 remained stuck in the GPU driver and used
about one CPU core throughout this measurement. Its affinity was not changed.
This background load limits the timing evidence: unchanged one-token IQ2_S
cases range from 0.943 to 1.036, and unchanged IQ3_S cases from 0.960 to 1.032.
Some individual process medians are much noisier, including 0.759 in an
unchanged case. Two-token layer 2 / 12 experts has one process median of 0.960
despite a pooled median of 1.076. These are retained in `summary.json` and
`pairs.jsonl`, along with process placement/counters. Every sampled benchmark
process has VmSwap 0. This is not a recovered-GPU whole-engine benchmark.

The optional compiler selection shows a pool-level benefit in these IQ2_S
cases, while intermediate quantization and down remain unchanged. Whole-engine
heads/IDs/logprobs, MTP/checkpoints, TG and full 256K tests remain pending GPU
recovery; the option stays off by default and the PR remains a draft.

## Reproduction

`probe.cpp`, `run.py` and `build.sh` contain the exact harness, orchestration
and compiler/link flags. The runner removes inherited `STRATA_*` variables.
The off archive is preserved from the actual CMake build used by the prior
[production-object comparison](../cpu-gcc-probe/dispatch-check/README.md).
The on archive is copied from `build-sycl-upstream-jit/libstrata_kernels_cpu.a`
with `STRATA_IQ2S_GCC=ON`, then frozen before timing. Copy the harness/runner
and both archives to the persistent probe directory as
`production-default.a` / `production-gcc.a`, then execute `bash build.sh`.

`run.log` includes build output and all 96 completed case summaries. Raw
per-process protocol output and validation logs accompany `pairs.jsonl`,
`manifest.json` and `summary.json`. Do not treat the compile-time `-fsycl`
linkage as evidence of any GPU execution: this harness only calls CPU APIs.
