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
  the named candidate for its frozen decode comparison. It is not a replacement
  for the fallback. Its active generation
  record's initial pair was code TG **13.259 -> 13.456** tok/s and weak-MTP prose
  **8.694 -> 8.832** tok/s (reference -> candidate). These are provisional
  observations, not the frozen three-pair decision; the live cohort remains
  undecided until its current samples satisfy that plan. They do not establish
  the 70 tok/s target, repeated gain, prefill preservation, agent/task quality
  or physical-context behavior.
  Do not freeze a running-process count in this document; follow
  `verify_window_generation_speed_v1` in the current service result and preserve
  its frozen plan and actual samples.
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
The 05a9 comparison changes only its decode selector; review all three matched
pairs before selecting. Required evidence includes both observed MTP-strong
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

The existing decode-only `service_profile.hpp` observer separates GU, hidden
quantization and Down by format, token count and worker, with nested wall and
thread-CPU clocks. `cache_map_cpu_service_sources_v1` is source-checked;
`cache_map_cpu_service_build_v1` is only prepared. Neither supplies measured
service or qualified observer overhead. After the current cohort, reconcile
the retained generations and update that pending observer recipe to the selected
native source, preserving the Q6 safety correction. Reuse the existing observer
and host/pool controls, then qualify observer cost before model attribution.
Useful CPU service still does not prove RAM bandwidth or arithmetic saturation.

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
