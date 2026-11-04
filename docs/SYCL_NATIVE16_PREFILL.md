# SYCL prefill and decode decisions

This page records current selections, the evidence needed for the next decision,
and open qualification gates. Closed experiments and detailed samples remain
in Git history and the linked cumulative result; this is not a run log. The
closed chronology is recoverable from the [original maintained decision at
commit `3abcceba08b25a632f4e26bde7551039120ac42a`](https://github.com/yqYo1/Strata/blob/3abcceba08b25a632f4e26bde7551039120ac42a/docs/SYCL_NATIVE16_PREFILL.md).

## Current selections

- Keep the **479a / engine 552b** generation as the validated fallback and
  development base for prefill. Its clean 117489-token throughput-route result
  is three independent processes, warmed subsequent-three medians
  **1044.845, 1047.067, 1047.336 tok/s** (median 1047.067); first-use results
  are separately **965.892, 968.399, 969.346 tok/s**, with client TTFT
  **121.812, 121.489, 121.375 s**. This is a same-engine clean revalidation,
  not a matched gain over a retired engine, general model adoption, sustained
  decode, task quality, resource saturation or full-context qualification.
  The warmed aggregate is **112.2077 s prefill and 112.385 s client TTFT**.
  Source/result: `cache_map_prefill_speed117489_v1.decision` in
  [`fused-moe-service-current.json`](../bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json);
  cumulative row in [`performance-records.csv`](../bench/performance-records.csv).
- Keep **05a9 (`05a9b45c0a3a266779d4757a855b45a3bb42a62e`) / engine a99aa1** as
  the named decode candidate. Its six-process, three-pair comparison is closed:
  code median TG **13.2594 -> 13.5052** tok/s and weak-MTP prose
  **8.6795 -> 8.8299** tok/s (reference -> candidate). All three paired TG ratios
  exceed one in each workload; their geometric means are **1.017949 / 1.016802**.
  All 12 long requests pass, with normal exits, quiet validation, no fresh fault
  and final hashes for all 15 model assets. This qualifies reproducible decode
  gain, not the 70 tok/s target, inference adoption, dedicated prefill/TTFT
  preservation, agent/task quality, prompt-verify or physical-context behavior.
  Code prompt/TTFT directions remain mixed. Follow
  `verify_window_generation_speed_v1.decision`, evidence commit
  `dd685c75fd6e77be17f826672bcd3486a70cbdc0`, and the CSV for exact samples.
  Keep the fallback while this same candidate supplies the imminent dedicated
  PP/TTFT and bounded sampled-cache entry/feasibility checks; no extra engine
  generation is required for those comparisons.
- `native_prefill_preservation_v1` failed during the first 65537-input request
  of the matched comparison
  of **552b versus a99aa1**, not just Q6 on/off within a99aa1. Source
  `tools/sycl/compare_native_prefill.py` fixes 12 processes: three opposite-order
  pairs at each of 65537 and 117489 exact input tokens, four fresh requests per
  process, first use separate from the subsequent-three median. Both arms use
  the throughput route and 20ms pool spin; the historical 1047 PP result used
  the pool default and is context, not the matched reference. Head retirement
  stays manually 0 at 64K and 1 at 117K in both arms. PP and client TTFT are
  separate decisions; the `ttft` strategy itself is not tested by this plan.
  Native inputs/fixtures and all 15 model assets were checked at preparation.
  The inherited 32K MTP checker incorrectly required every callback to be
  batched: at 64K, the declared 32768-cell MTP window leaves a 63-cell boundary
  that intentionally takes the token path. The request finished but the end
  gate stopped the benchmark; supervisor cleanup closed all owned identities,
  with no fresh driver fault. Keep V1 failed and its bounded parser fixture.
  A new prospective V2 must recognize only that source-derived boundary and
  reject other unexpected fallbacks before collecting another comparison.
- `native_prefill_preservation_v2` is closed with the same two engines,
  cases and runtime safety. It explicitly fixes the existing 32768 MTP window
  in both argument lists and checks the source-derived callback boundary.
  Five explicit shapes, 25 corrupt variants and the actual failed long-request
  window pass offline controls. All 15 model assets were hashed again; fresh
  frontend probes pass for both benchmark and server. All 12 processes and
  48 requests pass with normal exits, quiet validation, closed owners and
  no fresh fault; all 15 model-asset hashes pass again at final closure.
  All three completed 64K pairs support warmed prompt-time and
  client-TTFT nonregression separately: candidate/reference prompt-time ratios
  are 0.998945/0.995090/0.998678 and TTFT ratios are
  0.998943/0.993840/0.999426. Reference/candidate process-median PP is
  991.250/992.747 tok/s and client TTFT is 66.258/66.161 s. Each arm has three
  independent processes and nine subsequent requests; first use remains separate.
  All three 117K pairs are also complete. Their candidate/reference warmed
  prompt-time ratios are 0.995565/1.002483/1.001587, and TTFT ratios are
  0.995503/1.002465/1.001551. One pair favors the candidate and two favor the
  reference, so neither nonregression nor a consistent slowdown is established.
  Reference/candidate aggregate warmed PP is 1048.981/1048.455 tok/s and client
  TTFT is 112.180/112.225 s, each from three process medians and nine warmed
  requests. First-use PP remains separate: reference
  966.354/959.964/969.472 and candidate 968.437/967.808/968.894 tok/s.
  Successful execution does not satisfy the 117K performance-preservation gate;
  no inference adoption or sustained decode qualification follows from these
  short-output checks.
  V1 remains failed.
- Retain two explicit prefill strategies, `--prefill-strategy ttft` and
  `--prefill-strategy throughput`, within the same binary; selection is manual.
  Keep decode independently selected.
  Maintain the short-input path and avoid environment mutation after runtime
  initialization. No automatic length threshold is selected.
- The candidate's BF16 residual/interlayer storage does not make every hidden
  or intermediate tensor 16-bit. At 117488 rows, the HC bank's BF16 payload is
  `117488 * (4 * 2560) * 2 = 2406154240` bytes. Conditional PLE processing and
  final float callbacks still materialize chunk-sized FP32 HC residuals
  (`05a9b45c:src/prefill/prefill.cpp` and `src/prefill/layer_major.inc`).
  This HC payload differs from the MoE accumulator. Its FP32 split begins at
  65,536 of 117,488 rows; at width 2,560, its full payload extent is
  1,203,077,120 bytes, including a 531,988,480-byte FP32 tail
  (source `05a9b45c`, [accumulator plan](../include/strata/prefill/moe_accum_plan.hpp)). Keep local FP32
  accumulation/reductions, routing decisions/weights, and recurrent/online
  state distinct from storage dtype. A prospectively validated SGLang/vLLM
  numerical contract is acceptable; FP32/ggml remains diagnostic and does not
  impose its extra precision. Flags alone do not prove native-16 storage.
- Q6 specialization is a decode-only candidate detail. In
  `q6_byte_load_component_v2`, 30/35 geometry/column combinations pass the
  performance screen: each candidate/reference median time ratio is below one
  in all three independent processes. These are the **30 eligible shapes**.
  Both arms include the byte-read safety repair; this measures column
  specialization, not the repair's effect. Separate `q6_columns_component_v4`
  reports 28/35 performance-eligible shapes. The
  explicit `STRATA_SYCL_DECODE_Q6_COLUMNS` selector (default `0`), phase/token
  cache ownership, and prompt-verification reference path are in
  `verify_window_policy_host_v3`. These
  component results do not qualify model TG or adoption; follow also
  `verify_window_generation_speed_v1`.

## Prefill envelope and scaling

The 117489 result above is the current repeated warmed PP reference. A separate
same-engine curve has one process per length and describes actual routes, not
an adoption law. On the same **layer-major, bank-384, non-split** route,
19665..78445 actual input tokens are descriptively affine. With `n` actual read
tokens, `T_prompt[s] ≈ 13.0863 + 0.00082309*n` and
`TTFT[s] ≈ 13.1867 + 0.00082431*n`.
The prompt fit's maximum residual is 1.58%; leave-one-length-out mean/max
relative errors are 1.30% / 2.86% (TTFT 1.38% / 3.03%).
This is an empirical same-route description only: the 1286..8193
chunked route has a different fit, 117489 uses split, and route/chunk boundaries
change the curve. Do not interpret the intercept as a physical fixed cost,
extrapolate outside the measured interval, infer a limiting resource, or assume
one fixed-plus-linear law. See `cache_map_prefill_curve_v7.decision` and
`.interpretation` in the linked service result.

The slower same-engine 117K clean/observed comparison pair, **918.3721 / 943.8786
tok/s**, remains below the 1047.0672 tok/s reference; environmental variability
is unresolved. Retrieve its exact samples/configuration from the [original
maintained record at commit
`3abcceba08b25a632f4e26bde7551039120ac42a`](https://github.com/yqYo1/Strata/blob/3abcceba08b25a632f4e26bde7551039120ac42a/docs/SYCL_NATIVE16_PREFILL.md);
the cumulative CSV does not contain both values. The earlier adaptive-cache
model comparison recorded **15.529 tok/s** within a failed cohort: its code
sample qualified locally; the prose benchmark failed its answer gate and no
repeat decision exists. Preserve it as a failed observation via the CSV row
(`cache_map_generation_speed_v1.processes[1].samples[0]`), never as adopted
speed.
The separate **32K-output feasibility probe also failed**: its code sample was
15.5924 tok/s, but prose consumed all 32768 generated tokens in thought with
zero answer characters. It is not an unexecuted next experiment or a repeated
speed gain. Preserve `cache_map_generation_room_v1.failure_diagnosis` and the
original evidence at commit `a24593e1f84402244d8660914bb5e80587bcdc7d`;
do not repeat it or merely escalate its output allowance. Before another cache
comparison, choose and commit a distinct answer-eligible workload/sampling plan
and verify it on the safety-corrected native source. Keep thinking enabled;
do not waive the answer gate or assume sampling caused the previous failure.

## Decode and resource evidence

Keep prefill and decode kernels, placement, scheduling and selectors independent.
The completed 05a9 comparison changes only its decode selector. Required
adoption evidence still includes both observed MTP-strong
and MTP-weak long generation, dedicated PP/TTFT preservation, long-input and
agent-session benchmarks, prompt-verify behavior, and the prefill-to-decode
handoff. A component pass or one request does not qualify a route.

CPU census reports logical packed-blob/job work and nested pool wall; pool wall
includes scheduling and waits. GPU Down service and GPU-reach waits are service
attribution, not useful CPU work or critical-path wall time. Neither useful CPU
arithmetic nor physical RAM/PCIe transactions, VRAM traffic/bandwidth, hardware
ceilings or saturation are established. Keep requested transfer bytes distinct
from measured physical traffic. Current source result pointers are
`cache_map_decode_census_v1.decision`, `cache_map_worker_scaling_v1.decision`,
and `cache_map_decode_gpu_service_v2`; diagnostic observations are not
clean speed comparisons.

The earlier CPU-only [mixed-format calibration](../bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json)
at source `6e40afc1b` already records five samples per cell and achieved
256 MiB read references of 30.642 GB/s with one thread and 29.403 GB/s with
six. Its 28-job stream cells scale 2.69..4.10 times from one to six compute
threads. For six of seven formats the six-job stream/reuse latency ratio is
1.014..1.098; IQ4_XS is 1.563. These are prior-source calibration sensitivities,
not physical DRAM traffic, a bandwidth ceiling, current model CPU service or
proof of saturation. Reuse this evidence when selecting the next resource
question rather than repeating the same calibration without a changed question.
Read-only `lscpu -J` on 2026-11-04 identifies a Ryzen 5 5600X with six physical
cores, 12 SMT threads and 32 MiB L3; the earlier 256 MiB read buffer is eight
times that L3 capacity. Configured DIMM speed and physical RAM bandwidth remain
unknown; the current user cannot read the DMI table, and no ceiling is inferred.

The existing decode-only `service_profile.hpp` observer separates GU, hidden
quantization and Down by format, token count and worker, with nested wall and
thread-CPU clocks. `cache_map_cpu_service_sources_v1` is source-checked;
`cache_map_cpu_service_build_v1` is only prepared. Neither supplies measured
service or qualified observer overhead. After the current cohort, reconcile
the retained generations and update that pending observer recipe to the selected
native source, preserving the Q6 safety correction. Reuse the existing observer
and host/pool controls, then qualify observer cost before model attribution.
The retained build resolves per-file inputs through its recipe: its two pool
files still match `6e40afc1b`, without observer hooks, rather than the plain
Git `05a9` pool files. Reapply the scoped hooks to those measured bytes while
preserving the eight candidate-specific files; a checkout of the recipe's
commit alone is not the complete measured source snapshot.
Useful CPU service still does not prove RAM bandwidth or arithmetic saturation.
The reviewed retarget contract preserves the candidate's Q6/phase behavior and
requires checked observer-output completion: a truncated record or failed
write/flush cannot qualify attribution. The source review supplies no measured
CPU service or observer-overhead result.
The source-only output-v2 fix adds a versioned completion trailer, checks every
write and the final flush, and propagates failure at the generator call site.
The reader and retargeted engine are still pending; the active PP engines are
unchanged. The host-only check passes at source `49c20a41c` using
`/usr/bin/g++ -std=c++17 -O2 -pthread -DSTRATA_CPU_SERVICE_PROFILE=1 -Iinclude tests/core/cpu_service_profile_test.cpp -o /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/cpu-service-output-v2/profile_test`,
then running that fixture without `--clock-screen`. GCC is Ubuntu 13.3.0-6ubuntu2~24.04.1,
executable SHA256 `1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769`;
development packages are libstdc++ 13.3.0-6ubuntu2~24.04.1,
libc 2.39-0ubuntu8.9 and linux-libc 6.8.0-147.147. Pin the three project inputs
(header, test and native_expert.hpp) to the committed source before execution.
Budgets: 60 s compile, 30 s fixture, 1 GiB address space and 64 KiB text.
The fixture passes buffered flush and unbuffered write failure through
`/dev/full`, preexisting/null stream refusals and the exact completion counts;
no model, engine-call-site execution or observer-cost claim follows.

The prospective next cache feasibility test uses the [official thinking-mode
sampling parameters](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): temperature
1, top-p .95 and top-k 20 (also in the [pinned generation configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/generation_config.json)).
Keep the existing greedy benchmark available and choose the new profile
explicitly. Retain the 16K total output allowance, uncapped thought/xhigh and a
prospectively fixed positive seed; do not combine a new output cap with sampling
or compare it with old greedy TG as a cache gain. The compiled-source sampler
and uncoupled-MTP audit is closed: greedy proposals with independently sampled
verifier draws are source-consistent. Its RNG transform is custom, not standard
Philox4x32-10, and non-finite logits silently select fallback tokens. Pin those
contracts and resolve invalid-logit handling before sampled qualification;
parity fixtures sharing the custom RNG are not independent RNG evidence.
The reviewed compiled path submits the same generic `launch<false>` sampler
for both runtime modes; FP64 exp/CDF operations remain in that body even with
greedy parameters, including graph capture. The installed B570 SYCL
`aspect::fp64` and actual FP64 dispatch remain unverified. Intel's
[B580 architecture entry](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html)
does not establish B570's installed capability. After this
frozen cohort, commit a bounded exact-device query and arithmetic/exp plan
before sampler or sampled-model execution. A later precision change must
name the selected production path and invalid-logit policy. SGLang's
[sampler entry](https://github.com/sgl-project/sglang/blob/27aedf7f93cc7ac3bff43e7150e42a3c0baa6957/python/sglang/srt/layers/sampler.py)
selects distinct default, deterministic, seeded and XPU behavior. Pin its
actual kernel, input dtype and non-finite settings for comparison.
The Main-authored [exact-device diagnostic](../tools/sycl/device_fp64_probe.cpp)
is source-prepared only, without a build or device execution. It separates a
no-kernel capability query from 64-element FP64 arithmetic and exponential
checks. It requires the exact Level Zero B570 at `0000:05:00.0`, fixed runtime
safety and an in-order queue; an absent `aspect::fp64` refuses arithmetic before
submission. Arithmetic must preserve a `2^-40` increment that FP32 loses;
exponentials over -63..0 need relative error at most `1e-12`, checked finite
outputs, unchanged source and allocation guards. These are bounded capability
and numerical checks, not sampler equivalence, native arithmetic throughput,
absence of software emulation or model qualification. The diagnostic deliberately
skips provider teardown after completed copies and freed allocations.
After the frozen cohort closes, pin its source/compiler/provider identities
and a new owned-controller plan before building or running. Use query,
arithmetic, then exp in separate processes, once each; stop on the first
unsupported capability, error, timeout, validation message or fresh fault.
Keep the existing compiler/runtime without emulation flags. Prospective caps:
180 seconds/4 GiB RSS for compilation, 60 seconds/1 GiB per device child,
1 MiB text per command and 16 MiB total in a fresh private output root.
Retain the small diagnostic executable only until its recorded capability
consumer ends; preserve source and compact outcomes, then retire normal traces.
Benchmark source `1f39425ddea4dd98e172d0033f78fb00a8586675` adds the explicit
profile. Its 73 host tests finish with 72 passes and one optional real-tokenizer
fixture skipped. They check wire/identity and legacy behavior, not GPU execution.
Bounded sampler/device and actual draft-route validation must precede a sampled
model trial; no such execution or sampling-causality claim is established yet.

## Qualification and operating constraints

Before adoption, retain the model prompt-verify, task/agent quality and numerical
checks, failure ownership and graph/allocation retirement, sustained decode,
and **physical 262144-token full fill**, final tails/boundaries,
prefill-to-decode handoff and RESTORE gates. Component address tests or a
smaller context do not satisfy these gates. An API/CLI admission refusal that
reserves output room does not exercise the final physical 262144-token slots;
record the engine entrypoint and actual processed extent for full fill.
Ordinary provider/runtime teardown is not qualified by the retained pre-exec
validation reference through process exit.

Preserve the qualified runtime safety settings and immutable pre-exec
SYCL/Level Zero environment in
[`measurement_environment.json`](../tools/sycl/measurement_environment.json).
No provider-teardown claim follows from this setting. Keep exactly one validated
fallback plus a named imminent candidate while that comparison is active; do
not infer global cleanup from the current-build inventory. Keep unresolved
legacy source/recipe recovery holds separate from these two comparison
executables, with the missing inputs, consumer, next bounded recovery action
and review point recorded in the retained-build result. The fallback currently
retains nine ELF artifacts and the candidate three. At frozen comparison
closure, select the validated generation and retire an extra generation once
its named comparison ends and original inputs are recovered. The legacy
source/recipe holds remain independent.

Niko1221/Strata remains the only upstream. Last actual check:
**2026-11-02 21:17:40 UTC**, `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`
(`v0.2.9`); check again on the approximately 48-hour cadence using the
project-only sync skill. Required tracking/recovery/alias/API/CLI imports and
hardware-tuned unmodified-upstream measurements remain incomplete. PR 180 is
blocked by a squash-only ruleset; preserve measured source commit identities
and do not claim default-branch integration.
Remote publication recovered: the actual ref check at 2026-11-04 10:12 UTC
confirms `feature/sycl` at `394e6689e`, including the first 64K pair, CSV and
raw-output retirement. The default remains `05069aeed`; PR integration is
still blocked independently of the resolved push error.

## Evidence pointers

The cumulative user-requested index is
[`bench/performance-records.csv`](../bench/performance-records.csv). Detailed
current evidence and frozen plans are in
[`fused-moe-service-current.json`](../bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json)
under `cache_map_prefill_speed117489_v1`, `cache_map_prefill_curve_v7`,
`cache_map_decode_census_v1`, `cache_map_worker_scaling_v1`,
`verify_window_generation_speed_v1`, and `retained_builds_review_20261103`.
Earlier closed phases are retrievable from the baseline Git revision above;
consult them only for a concrete comparison or provenance recovery. Follow
[`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md) for retained builds and cleanup.
