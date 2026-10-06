# CPU native expert pool: row task factor 9

Ryzen 5 5600X, 128 GiB RAM, oneAPI 2026.1.1, real mixed GSQ-RCO IQ3_S
Qwen3.8 Flash Next weights, 2026-10-06. Five pinned worker threads and the
pinned host execute the production `ExpertPool::run_split_multi_native`.
The reference uses three row tasks per participating thread; the candidate
uses nine. Quantization, dot products, reduction order and synchronization
inside the original CPU pool are unchanged. Selective GCC IQ2_S gate/up is
enabled in both arms.

## Results and limits

The quiet repeat has 80 cases per configuration, nine alternating-order
pairs per case and three fresh process pairs: 2,160 paired samples for the
candidate and 2,160 for the unchanged-reference control. Each case takes its
median paired original-time/candidate-time ratio; these 80 medians have a
geometric mean of **1.06234** for factor 9 and **1.00912** for the control.
The ratio of those two summaries is 1.05274. This is descriptive, without a
causal bias correction or confidence interval. All outliers remain in the
raw records; the control's case ratios range from 0.89981 to 1.10216.

Cases cover layers 1/2/17/21, 1/2/12/48/96 experts, and one/two/four or mixed
one-to-four token groups. The measurement includes gate/up, intermediate
quantization, down and dispatch. No agent-owned compiler or competing CPU
benchmark ran during this repeat. External host activity was not controlled.
All 8,640 recorded process snapshots have no DRM descriptors and zero swap;
this is sampled evidence, not continuous monitoring.

The earlier `production-measurement-*` run overlapped SYCL compilation and
is retained as exploratory data. The prototype also tried parallel
intermediate quantization; that variant is **not** in the production
candidate. `candidate.diff` describes that prototype; the shipped candidate
is the single multiplier replacement in `production-constant9.diff`.

This does not measure engine prefill or decode speed. The compile-time
`STRATA_NATIVE_POOL_TASK_FACTOR` defaults to 0, compiling the original pool.
Factor 9 remains experimental pending healthy-GPU whole-engine comparisons.

## Exact outputs and sanitizer

Both actual production archives pass 396 cases: all 48 layers with experts
0/7/31 and one-to-eight tokens, then repeated mixed groups with 1/2/12/13/96/97
experts through a live pool. The 97-expert case crosses the 96-entry buffering
boundary. All 34,412,784 floats, including guards and inactive output slots,
are finite and the complete 137,651,136-byte files compare equal.
Their SHA-256 is `092acfb9cb9d0b96931490e96d709afd53b56f95fff07e9e33da252555489482`.
Each timed dataset also has a full byte comparison: 480 checks in the quiet
repeat and 480 in the earlier production run.

ASan/UBSan instrumentation of the production constant-9 pool passes the
same 396 cases and produces the same complete bytes. External quantization
libraries are uninstrumented. This CPU executable links no SYCL/Level Zero/UR
runtime; `asan-record.json`, `asan-validation.jsonl` and `asan-ldd.txt` record
the scope. A CPU sanitizer pass does not establish GPU correctness or exclude
all data races.

The latest engine build includes the SYCL safety changes. Its factor-9
candidate SHA-256 is `752abec884dc9caee99ba80435aa5018689ec021b045d2050b9748dccc4d7d80`.
The reference was restored to factor 0 after building. The CPU archives and
harness executables have exactly the same hashes as the measured and
sanitized artifacts; see `current-production-build.json`. This engine has
not been run on the GPU. Large binaries and output dumps stay in the host's
persistent state directory, outside Git.

## Reproduction

The harness is unchanged from [the production CPU probe](../cpu-gcc-pool-probe/probe.cpp).
Build the SYCL CPU archive with factor 0 and factor 9, preserving each archive
before changing the CMake cache. Link the same compiled harness object to each
archive and the same ggml CPU/base archives. `current-build-production.sh`
records the exact investigation-host commands and all source/artifact hashes.
It restores factor 0 and verifies the reference executable afterward.

For a new measurement on this six-core CPU, use `run.py` with paths to the
reference executable, candidate executable, first GGUF shard, expert-layout
pack and a new output directory. It runs full validation, actual byte
comparisons, the factor-9 and unchanged-reference control measurements, and
records process placement and raw timings. `run-observed.py` is the exact
runner used for the saved quiet observations; its original machine paths
and the earlier validation references are preserved rather than rewritten.
Avoid compiling or running another CPU benchmark during timing.
