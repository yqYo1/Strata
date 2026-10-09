# Standalone streaming native CPU calibration

Root built the source-only Sol handoff in 24.255s with the same standalone
CPU-only closure and per-TU flags as the qualified smoke tool. This receipt
covers compile/link/import checks only: no 384-ID cohort, statistical gate,
performance result or model inference was run by this build.

R81 identifies an omitted metadata setting, STRATA_IQ_PREFETCH. Root launchers
already use a recorded minimal environment, so this is not evidence of a
nondefault existing run. Record the setting in tool output before comparing
future NT2 timing profiles. Keep all earlier source/build/research statuses.

Root added the environment metadata field, rebuilt incrementally in 4.937s and
re-audited the full target compile/link closure. Both build receipts passed,
with normal exits and no owned survivors. The original v1 source remains in
commit 64f189c6 and the original build receipt remains unchanged. No kernel,
task policy or production engine source was changed; neither build executes
the 384-ID streaming cohort or qualifies timing, statistics or full context.

The two receipts, target commands, actual compile flags, direct imports and
root supervisors are retained as compact reproducibility evidence. Full
tracked-file listing is redundant with each receipt's hashes and is not copied.
No build binary or model payload is committed.

## Actual 384-ID check and corrected task calibration

Root selected the exact GU type22 (IQ2_S) / Down type20 (IQ4_NL) / NT1
cohort before timing: 192 train and192 disjoint holdout expert IDs, balanced
across15 layers, from the closed r6 route trace. This is a synthetic-input,
host-resident, equal-layer kernel characterization. A/B routed sources are
related requests; these are not independent inference workloads. The frozen
manifest and cohort are retained, with one deduplicated list of all1152
selected real weight-role extents and hashes. No payload weights are archived.

Original correctness r2 passed all384 native/independent-GGML, quantization,
repeat and partition checks. Its four following timing processes r1/r2 all
ended normally and passed output checks, but root found all six threads had
inherited CPU0. The fixture pinned the caller before constructing ExpertPool,
which rescans caller affinity. `native-service-calibration-affinity-invalidation-v1.json`
invalidates every timing interpretation from those four processes while
preserving their original output-correct statuses, individual samples and hashes.
No production engine placement bug was established: the actual SYCL generator
constructs the pool before SessionLoopScratch initializes host affinity, in both
the current source and the retained869 baseline. This source audit is not a new
runtime placement observation of the full engine.

Root moved caller pinning after pool construction and added a mandatory audit
of the actual six distinct singleton thread masks against the planned CPUs.
Corrected buildv4 and correctness r3 passed, then four new normal-exit processes
(tasks6 r3 / tasks0 r3 / tasks0 r4 / tasks6 r4) verified CPUs0..5. Requested0
means18 effective tasks. Each process records25 measured rounds plus3 warmups
for five arms on each of the two cohorts. Every individual round is retained;
repeated EXTENT/ID/REFERENCE lines were removed only from the compact timing
CSV, with original file hashes and the canonical identities preserved.

The corrected whole192-job cohort pool medians are17.208–17.496ms for6 tasks
and14.821–15.898ms for18 tasks. These homogeneous cross-layer six-job calls
differ from production per-layer mixed-NT callbacks. Not all direct fit gates
passed; each pool cell also failed its required interval/holdout criterion.
The 6-task GU-phase medians were1.217–1.258 times their paired18-task medians,
but these microbench results do not qualify an engine task-policy change or a
route-service weighting model. The deterministic10000-resample per-process,
per-cohort bootstrap intervals/CV and all failed gates are in summary-v3.
No new >=32K inference, decode throughput or full262144 lifecycle claim is made.

Original launcher and offline admission failures are retained: the initial
GGML root path, the cleanup-dict interpretation before payload launch, and the
summary-v2 TASK_PLAN check (0 is printed as normalized18,18). Their corrected
successors do not rewrite the failures or weaken output/placement/math gates.
The original summary-v1 is retained with its later affinity invalidation.

Root also inspected only `ggml_vec_dot_iq2_s_q8_K` in the exact buildv4 binary,
without executing it. The linked function contains AVX2 packed multiply-adds
and YMM fused accumulation; the full5600-byte single-function listing and exact
binary/tool/controller hashes are retained. This closes R87's emitted-ISA gap,
not every operation-level paired-kernel prerequisite or a speed hypothesis.
