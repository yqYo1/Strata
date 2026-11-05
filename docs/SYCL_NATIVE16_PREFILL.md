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
  The dedicated PP/TTFT comparison below is now closed; 117K preservation is
  inconclusive. Keep the fallback. The next CPU-service observer uses this same
  recoverable 05a9 source. The closed a99 binary is retired after source/recipe
  recovery and consumer updates; the 479a fallback remains available.
  Sampled-cache execution still needs its sampler/device admission and a newly
  qualified engine; this does not adopt the candidate's prefill path.
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

The compact, originally failed `cache_map_decode_gpu_service_v2` capture is
useful only for selecting a new question. Its 2.154 s interior window observes
342.910 ms summed Q6K service (868 launches) and 76.589 ms mapped-row copy
service (316 launches); these overlapping services are not critical-path costs.
That earlier 479a, short-output configuration does not establish today's
long-decode ranking, and its original artifact-cap failure remains unchanged.
The current mapped-row copy checks the same hit-row list for every float in a
row (`src/kernels/sycl/elementwise.cpp`), whereas the hit decision is row-uniform.
A bounded row-mask/broadcast/vector-copy component can test this concrete
redundancy after the CPU component; compare actual host-USM and device input
placement and preserve hit zeros, source lifetime and graph ordering. The
unselected `src/kernels/sycl/mapped_row_copy.hpp` prototype defines a hit-list
scan and converged broadcast per row group. An optional column-tiled variant
repeats that scan per tile:at the actual40x2560 shape,one group per row provides
40 groups,while256x4 and64x4 column tiles provide120/400 groups. This tests
whether more independent work offsets the extra membership work;these are
group counts,not occupancy measurements. Scalar/four-element source groupings
allow four-byte-aligned rows without promising emitted vector instructions.
The independent `tests/sycl/mapped_row_copy.cpp` fixture compares the existing
production entry with four unselected variants,using device metadata and both
host-USM/device sources. Its prepared checks include changed/duplicate/unsorted
hit lists,positive-zero hit output,NaN/signed-zero bit copies,odd tails,offset
alignment and source/output/metadata guards. The timing path separates untimed
checks and warmups from dependent-batch wall/host CPU clocks,with one/64 source
banks and two hit ratios. Submission and checked-wait regions each retain wall
and thread CPU time;their sums conserve each batch's total. A wait's thread CPU
time is not host tensor arithmetic,and submission can overlap device execution.
The fixture does not measure physical traffic or kernel service.
Neither the modified prototype nor fixture has been compiled or run. The
component still needs a committed pinned build recipe,bounded traced admission
and repeated actual-placement timing before any decode-only route proposal.
The excluded `sycl_mapped_row_copy_component` CMake target links only the
runtime,production elementwise translation unit and fixture. It does not build
an inference engine or require native-expert/GGML and test-suite targets.
Prepare a separate pinned Release configuration with`STRATA_ENABLE_SYCL=ON`,
`STRATA_NATIVE_EXPERTS=OFF`,`STRATA_BUILD_TESTS=OFF` and`STRATA_SYCL_TESTS=OFF`;
then build only this target with one compiler job. No configure/build has run.
The existing production copy and measured engine remain unchanged.
Every lane reaches the two-dimensional group broadcast with the same valid
source ID before the hit-dependent branch,following the
[Khronos group-function contract](https://github.khronos.org/SYCL_Reference/iface/group-functions.html).

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
The observer build's actual GGML x86 quantizer compile command already uses
`icx -O3 -march=native -fno-associative-math`. Static disassembly of
`ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/quants.c.o`, SHA256
`b8942d84db7db78a72afeb43896206c0196a8e49e205601f860897106bd8a72e`,
finds AVX2 `vpsignb`, `vpmaddubsw`, `vpmaddwd` and vector FMA in
`ggml_vec_dot_iq3_xxs_q8_K`. Missing vector code is therefore not the premise
for the next pair experiment. This is static code evidence, not measured
dispatch frequency, arithmetic throughput or saturation.

The current native-expert inventory has 17 IQ3_XXS, 20 IQ2_S, ten IQ3_S and one
IQ4_XS GU layers, each with 512 experts. Its `native_experts.txt` SHA256 is
`d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d`.
For a possible future exact 4-bit IQ3_XXS CPU view, 128 code bytes, two scale
bytes and four group-scale bytes make 134 bytes per 256-value block, versus
98 in the original packed form. At 2560x640 per GU matrix, all 8704 affected
experts would need `17*512*2*640*10*134 = 14929100800` additional bytes
(13.9038 GiB). The original GU payload is 10918297600 bytes; both would total
25847398400 bytes. The current raw arena remains necessary for GPU cache fill
and fallback, so the proposed CPU view adds its full size, rather than only the
4.01 GB representation-size difference. Down, activations, other formats and
allocator overhead are outside these GU payload totals. No such view, owner,
conversion or allocation admission is implemented or qualified. Compare its
conversion/reuse cost only after a named component plan; the immediate pair
candidate below avoids this expansion.

The decode-only `service_profile.hpp` observer separates GU, hidden
quantization and Down by format, token count and worker, with nested wall and
thread-CPU clocks. Its replacement build, real-pool off/on parity, small device
checks and short model/census entry are now closed. Representative long-decode
collection is closed: all six processes and twelve long requests pass, and all
three paired incremental observer-cost bounds qualify as detailed below.
The retired a99 generation's measured snapshot used two pool files from
`6e40afc1b`, rather than the plain Git `05a9` pool files. The observer hooks
are applied to those recovered bytes while preserving the eight
candidate-specific files; a checkout of the recipe's commit alone would not
recover the complete measured source snapshot.
The observer constructor is now retargeted to that complete 560-file snapshot,
with the checked output-v2 delta. Its source check passes at
`1e791e80bde6366c37469a07476f4518594bc1a9`: all 563 projected files are checked,
and reversing the seven observer/build/test files recovers all 560 measured
inputs byte for byte. The eight Q6/phase files remain preserved. Follow
`cache_map_cpu_service_sources_v1.retarget_v2` for the source-only proof.
The replacement builder targets that snapshot and preserves the 479a fallback.
Its source preparation passes at `8e2499937969dbfa1dcd950b95025d9712037a83`;
follow `verify_window_cpu_service_build_v2`. It keeps the existing build settings and enables only
the observer. The four targets are the engine, real CPU-pool fixture, submission
fixture and phase-cache fixture. The separately qualified pure writer fixture
stays excluded because `STRATA_BUILD_TESTS` remains OFF; the target-list
correction is checked separately from the unchanged source constructor.
The closed a99 comparison outputs and superseded unbuilt observer archive are
now physically retired: 1293 files and 141950711 bytes, with original source,
recipe and compile evidence checked before deletion. The original immutable
build receipt remains a source-only input to the replacement; its deleted ELF
is not reused as qualification. Follow
`retained_builds_review_20261103.current_replacement_retirement` for the manifest
and remaining consumers. All nine fallback ELFs and the research-pinned
runtime/configuration inputs remain intact. The observer build is now closed
with actual compiler/link input coverage and four hashed outputs, from source
`8e2499937969dbfa1dcd950b95025d9712037a83`. The real-pool off/on checks both
pass 28 cases with identical output checksums and normal closed processes.
The same-build small device gate passes. No measured CPU service or speed gain
follows from those host/device checks alone. The
`tools/sycl/cpu_service_device.py` reuses the maintained small submission/phase
runner in a private context and requires that exact real-pool parity. Its
prospective preparation does not execute a GPU kernel.
V2 preparation failed before GPU execution: the supervising process allowed a
1 MiB file while preparation writes the maintained result JSON (about 5 MiB).
The exact committed prefix was verified, the complete committed state restored,
and the closed exit-1 failure retained. V3 preparation passes at controller
`1ef6e18ec5e266ed6ced67b704f6cf08aa038314`, with a 16 MiB file cap and
explicit result-size admission; raw output retains its 4 MiB private budget.
The normal exit-0 process is closed with quiet stderr. The exact same-build
submission/phase-cache plan is prepared under
`verify_window_cpu_service_device_v3`. Both device cases now pass: submission
has ten checks, and phase-cache covers sixteen windows with twelve captures.
Both processes exit normally with closed owners, checked validation, unchanged
boot/providers and no fresh driver fault. These synthetic payloads do not
qualify model arithmetic, measured CPU service, observer overhead or ordinary
provider teardown. The short model/census entry below uses that same build.
The Main-authored `tools/sycl/cpu_service_model_entry.py` now defines that entry
using the maintained model supervisor in a private context. It requires the
closed device gate, unchanged twelve-case reader qualification and three
complete model/census/service joins, including the warmup. Its preparation now
passes at controller `40c4efeaec6427a81db5181840b27fabddf38086`, with all fifteen
model assets hashed and the current source/provider/configuration pinned.
The owned preparation exits normally in 111.821 s, with quiet stderr and no
GPU/model execution. Both current-source frontend probes now pass with normal
exit and closed owners. The actual model entry now passes: one short warmup
and both fresh 32769-input/24-output thinking-on requests bind complete v2
service records to their independent censuses and native/API clocks. All three
requests conserve rows, jobs, entries, format/NT bins and six thread slots;
the benchmark/native exit normally, validation is quiet, all fifteen final
model hashes match, owners close and there is no fresh fault.
For the two 32K requests, summed leaf CPU time is 3.310/3.091 s, while nested
pool wall is 0.676/0.650 s. GU consumes 69.33/70.06% of that CPU time, Down
30.49/29.75%, and hidden quantization 0.185/0.190%; NT1 is 72.88/72.36% of
CPU jobs. These concurrent leaf sums include memory stalls and observer cost;
they are not request critical-path fractions or hardware saturation evidence.
The different 74-token warmup remains separate. Follow
`verify_window_cpu_service_model_entry_v1.cpu_service`; no representative
long-generation mix, observer overhead or clean-speed gain is qualified.
The Main-authored `tools/sycl/cpu_service_overhead.py` closes the repeated
comparison: three alternating off/on pairs in six independent processes, each
with both fixed thinking-on long-generation workloads. Common census, decode
clocks and trace settings stay identical. A five-percent per-pair latency bound
is interpreted only when aggregate logical work matches and both actual MTP
classes are observed; changed work is inconclusive and keeps every sample.
Preparation passes at controller `dab5a31d91a6c724e944ab921915526ec144556e`:
the six generated plans have one invariant configuration/runtime signature,
all fifteen model assets match, and the bounded decision controls pass.
Both current-source frontend probes pass. All six processes pass six native/API
census joins apiece and both thinking-on workloads, with normal exits, closed
owners, quiet validation and no fresh fault. All fifteen final model-asset
hashes match. Logical work and actual MTP classes match in all three pairs.
On/off per-output latency ratios are 1.002853/1.017201/1.026476 for code and
1.008799/1.016185/1.017022 for prose, each below the predeclared 1.05 bound;
geometric means are 1.015464/1.013995. This qualifies incremental CPU_SERVICE
cost with common observers retained, not clean speed or model adoption.
In the three on processes, GU is 68.94..68.99% of code leaf CPU and
69.72..69.74% for prose. IQ3_XXS NT1 GU, the selected component, is
20.27..20.29% / 16.15..16.16% of total leaf CPU; it is not the entire GU stage.
The first on process has summed leaf CPU 520.298/1686.105 s for 3170/7275 outputs.
If those instrumented CPU costs per output stayed fixed, six concurrent leaf
threads would have optimistic aggregate budgets of 36.6/25.9 output/s before
any other work. This conditional accounting is not a physical CPU ceiling or
clean-speed prediction: observer cost, CPU frequency, memory stalls, cache
residency and the MTP/NT mix still matter. Reaching70 requires lower CPU-service
cost or less CPU work, alongside GPU and transfer improvements.
The 69–70% GU figure is a share of summed leaf CPU time, not of total decode
latency. In the three on processes, nested pool wall is 44.53..45.01% of code
decode and 41.23..41.70% of prose decode. Removing all recorded pool wall in
the first two on processes while holding everything else fixed would yield only
about 24.0–24.1 code or 14.8 prose output/s
(`generated_tokens*1000/(decode_ms-pool_ms)`). This is conditional accounting,
not a physical ceiling: pool includes waits and can overlap device progress,
and changing it may affect other scopes. CPU tuning alone under that fixed-rest
assumption does not reach 70; retain the whole-decode question alongside the
CPU component. Before the final pair, a prospective retention addendum keeps
the already-emitted outer/nested timing footer for processes 4/5's long requests.
No instrumentation, workload, ordering, acceptance or stopping rule changes.
Those rounded host clocks can distinguish the next draft/verify/host question;
they do not measure GPU arithmetic service, physical traffic or saturation.
The final off process's retained rounded clocks show code/prose
total 257.72/247.17 ms per window: CPU pool 113.21/100.88, GPU-reach wait
84.50/84.77, and draft 33.63/33.53. In the final on process these are
264.54/251.37 total,119.07/104.81 pool,84.51/84.64 wait and33.63/33.56 draft.
These are nested host scopes, not additive device service. Trace the called
wait wrapper before attributing its cost to GPU arithmetic.
The next diagnostic source adds `STRATA_SYCL_DECODE_SERVICE_PROFILE=1`, fixed
before native exec. It partitions the main thread's decode-through-final-commit
interval into capture/cache lookup, staging, complete layer-wait wrappers,
pool calls, expert-copy retirement, graph submission and final stream drains.
Nested native blocking calls are recorded inside each region; do not add them
to the containing region. Draft/commit and other uncovered work stay explicitly
unclassified. One integer-nanosecond record per request replaces per-call logs.
The host-only `decode_service_profile_test.cpp` checks exact conservation,
nested waits, invalid nesting/clocks, exceptions and disabled observation.
The host-only fixture now compiles with GCC13.3/O2 and passes all seven cases,
with normal exits, quiet stderr and closed owners;168 actual dependency files
are fingerprinted. See `decode_service_host_contract_v1`. This does not compile
or qualify the diagnostic engine. Its next consumer is a separately qualified
diagnostic build after the current IQ3 cohort. The measured engine/source remain
unchanged;no model attribution,physical-resource ceiling or speedup follows yet.
No engine rebuild or GPU fault follows from the V2 preparation failure.
Useful CPU service still does not prove RAM bandwidth or arithmetic saturation.
The six closed CPU-service processes also agree exactly on logical work volume
for each workload. Dividing `logical_job_blob_bytes` by generated tokens gives
the following target-verifier demand, including rejected verification rows but
excluding separate MTP draft work. GB below means decimal10^9 bytes.

| Workload | Resident expert entry share | CPU logical blob GB/output | At70 output/s,logical GB/s |
| --- | ---: | ---: | ---: |
| Code,33212 input /3170 output | 10.8501% | 0.786020 | 55.0214 |
| Prose,33282 input /7275 output | 6.7215% | 1.049027 | 73.4319 |

The numerator is one full native expert blob per unique CPU job,not one per
token entry:the existing jobs already group repeated experts across NT1..4.
Code/prose have1221484/3743079 CPU jobs;78.1594%/63.6290% are NT1 expert jobs,
which does not mean that the verifier itself mostly runs one-token windows.
Resident-entry share measures expert placement,not GPU compute share;dense,
attention and recurrent work are outside that count. Source is
`verify_window_cpu_service_overhead_v1.processes[*].cpu_census.long_requests`
at commit `c74d1539cceec80c13e9356ff73d54dfa4545587`.
These logical bytes are not measured DRAM/PCIe traffic or a physical lower
bound:cache reuse,partial reads and actual memory access still matter. The
55/73 GB/s demand exceeds the prior achieved30.642 GB/s read reference,which
is itself not a hardware ceiling. Before treating a wider CPU weight view as
the next model route,compare its extra bytes with compatible read capacity
and evaluate fewer CPU jobs through expert placement/reuse. Neither cache
placement nor arithmetic is established as the limiting physical resource yet.
Before testing `STRATA_IQ_SINGLE_EXACT=1`, the worktree now confines that
single-token GU selector to `NativeGuPhase::Decode`, preserving the prefill
dispatch. This source-only containment is outside the frozen observer build;
its compiled qualification and matched model comparison remain pending.
The unselected `iq3xxs_pair_gu_rows` component is the next IQ3_XXS NT1 GU
candidate. It shares activation loads between gate/up, keeps packed weights
unchanged and preserves each output's block FMA and final float reduction.
Its integer lane bound is 7872512; byte loads avoid unaligned scale/sign objects.
The maintained single-exact fixture now defines GGML parity, padded/unaligned
input, output-canary, invalid-layout, overlap and overflow checks for this
component. Its V2 same-build correctness fixture and direct single-core
component comparison now pass as detailed below. The helper
is not selected by production; qualify the actual decode entry/pool dispatch before a model
trial. Retire it if the component does not justify that trial.
The independent source audit uses the frozen scalar oracle at
`/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/cpu-service-verify-window-v2/source/src/kernels/cpu/iq_single_avx2.cpp`,
SHA256 `8579db5f02086cc660d376a77e00a9285e038f6757a53e8b2a8ef6d79eb2074f`.
Its `iq3xxs_dot` body is byte-identical to the candidate's scalar reference.
The refusal fixture now snapshots the actual weight and activation allocations,
so aliased output cases check those targets as well as the ordinary output.
It also defines a separate misaligned-output refusal. These source changes
still need the committed compiler/ISA and correctness gate before timing.
The prospective CPU-only `cpu_iq3_pair` benchmark uses 252 actual IQ3_XXS/IQ4_NL
experts from one pinned model layer. It compares the existing decode NT1
`native_gu_rows` entry with the direct pair entry, including that helper's
layout checks, at 640 and 128 rows. Each shape has six-expert reuse and 252-expert
stream access, with prequantized activation or one production activation
quantization per six experts. Even the 128-row stream's requested GU footprint
is about 60.3 MiB, above this CPU's 32 MiB L3; this does not prove DRAM traffic or
saturation. Both entries keep the packed weights unchanged. All 252 experts and
four deterministic activations need finite, bit-equal reference outputs before
and after timing; every sample also checks its timed outputs against an untimed
reference. Five samples per cell target 0.35 s with a 0.25 s eligibility floor,
four warm passes and timer cost below 0.1% of the fastest warm pass. Model
loading, hashing, allocation and reference checking stay outside timed work;
the pack-inclusive cells include the specified activation setup. The eventual
frozen plan must pin the full build/fixture/source inputs and alternate three
independent reference/candidate pairs. No minimum gain is imposed; compare all
three paired median directions separately for each cell. This is a direct
single-core component experiment, not a production-pool route or model gain.
The required observer cohort is now closed. V1 compiles, but its linker-input
admission fails before any correctness or timing execution, as detailed below.
The CPU-only CMake project also exposes `cpu_iq_single_exact_test`, reusing the
maintained correctness fixture and the same freshly compiled `expert_cpu`
library as the benchmark. This permits the compiler/ISA and correctness gate
without linking or rebuilding a GPU engine. V2 pins the complete linker inputs
before rebuilding and now passes the new fixture.
The benchmark emits deterministic weight/activation input fingerprints before
timing; require them to match across all six processes in addition to the
pinned model-range hashes. FNV fingerprints are consistency checks, not
cryptographic source provenance. The inspected CPU reference dependency is
official `ggml-org/llama.cpp` commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`; its CPU subtree is
`c38dca42a07f0aa703bdbe3e55639ad56c91dc1e`, and the selected working CPU,
headers and CMake paths have no diff from that revision. The checkout's old
directory name is not implementation provenance.
The prospective CPU recipe uses a fresh external
`cpu-iq3-pair-component-v2/build`, icpx/icx 2026.2, CMake Release/Ninja with
compile commands, and this CPU-only project with its pinned GGML directory.
Build only `cpu_iq_single_exact_test` and `cpu_iq3_pair`, with four compile jobs;
keep the known-good engine and active observer generation unchanged.
Before execution, commit the exact argv/environment and complete transitive
project, compiler and dependency identities in the compact result. Check the
actual compile/link inputs and CPU-only configuration before the fixture.
`bench/sycl/cpu_roofline/iq3_pair.py` now defines that serialized lifecycle:
prepare/commit, build/commit, correctness/commit, then six separately committed
processes. Its prospective input map covers committed project/GGML bytes,
official SDK and host linker inputs, generated version header and tools. It
requires actual compile/link coverage and CPU-only linkage, the same archive,
384 original plus 80 paired fixture cases, and complete real-model validation
before/after each process. Preparation now passes at source
`229fe7376dfeeb4c478b322934ce6fda882506cf`: 188 committed project files,
1376 official GGML files and 12172 external inputs are frozen, including the
exact generated header, tools, environment and 252-expert model-range hash.
V1 configure, compile and dependency collection all exit0 with closed owners;
compilation takes20.035s. Actual linker-input admission then fails on
`/usr/lib/x86_64-linux-gnu/libm.so.6`. Static inspection of the complete trace
finds four additional missing runtime objects: libc, libgcc_s, libstdc++ and
the loader. The pinned `.so` linker scripts do not cover these selected ELF
targets. V1 remains failed, and neither unadmitted executable may feed a test.
V2 uses a fresh root and queries/hashes all five targets and their package
versions before compilation, preserving the shared helper's existing contract
for the retained engine. The same80 pair cases now explicitly cover all128
sign selectors,16 scale nibbles,256 grid indices and signed FP16
zero/subnormal/normal/max-finite scales, with coverage checks. The arithmetic
candidate and production dispatch are unchanged. V2 preparation passes at
`39963e5ab`, with188 project,1376 official GGML and12177 external inputs,
including all five runtime targets. V2 compilation now passes, with complete
actual compiler/link input coverage, two hashed CPU ELFs, the same freshly
compiled expert archive, and only pre-pinned CPU runtime libraries. All five
owned build/inspection commands exit0 with closed owners. The same-build
correctness fixture exits0 and passes384 single plus80 paired cases, including
the explicit sign/grid/scale and FP16-edge coverage, bit parity and refusal
targets. Its process closes with quiet stderr. All six component processes
and240 individual samples now pass, with identical actual weight/activation
fingerprints and2016 real-model comparisons before and after each process.
All eight cells favor the pair in all three independent pairs. For640 rows,
candidate/reference geometric time ratios are0.9474/0.9461 without activation
packing (reuse/stream), and0.9460/0.9408 with packing. For128 rows they are
0.9490/0.9466 and0.9479/0.8947. The last cell includes one slower reference
process (63.38us/expert versus roughly53..54us); all samples remain recorded.
This is typically about5% lower direct-component time, not model gain, RAM
saturation or production adoption. This motivates the explicitly decode-only
production-entry comparison below; no model speed is qualified.
After original V1 source/recipe recovery from pushed Git history, its failed
unqualified generation and V2's closed normal build/correctness output are
retired:89 files,20090636 bytes. The failed outcome remains intact. After the
direct comparison closes, V2's two CPU executables, archive and remaining
outputs are also retired:83 files,19278175 bytes. Its original source/recipe
and all240 samples remain at `ee5a8e41307a97bdb7155004e4bc9c2a33af5e33`;
the validated GPU fallback is unchanged.
The source now offers default-off `STRATA_IQ3_PAIR=1` only at the decode NT1
production entry, below the unchanged multi-token dispatch. Prefill keeps its
existing selection. `STRATA_IQ3_PAIR_AUDIT=1` uses separate per-thread IQ3
eligible/selected row counters; the legacy IQ2 audit retains its own fields.
No process environment changes between phases. The `cpu_iq3_pair_dispatch_v3`
build now passes from source `1ec110452cdd70c425bbc31edd48bbd0421563ca`,
with three hashed CPU ELFs, checked actual compiler/link inputs and the same
fresh expert archive. No GPU runtime is linked. All four correctness children
now close normally with quiet stderr. Each selector passes384 single and80
paired bitwise GGML cases and80 audited production-entry calls; decode selects
0/10240 paired rows for selector0/1 while prefill selects none. Both real-pool
fixtures pass28 cases covering repeated prefill/decode, seven GU/Down format
pairs, NT1..4 and the scratch-batch boundary; their full output checksums both
equal `e5d7a50a8e7d5ac5`. All six timing processes and240 samples now pass,
including2016 real-model output cases before and after each process and
actual dispatch/eligible/selected-row counts for every timed batch. All eight
cells improve in all three independent pairs. Candidate/reference geometric
time ratios are0.9390/0.9425 (640 rows,prequantized,reuse/stream),0.9385/0.9627
(640 rows,packing included),0.9417/0.9257 (128 rows,prequantized),and
0.9504/0.9619 (128 rows,packing included):3.7..7.4% lower time. Both arms use
the actual production entry with the same audit counters enabled. This is
single-core entry timing; the pool fixtures qualify phase/output behavior,
not pool speed. `decode_iq3_pair_model_build_v1` now compiles from controller
source `9922f8a5a`, with563 recovered source files,1449 actual compiler inputs,
29 frozen linker inputs and five hashed outputs. Its first host gate fails
at the observer artifact cap:the launcher counts the entire compiled tree
against a1MiB host-output budget. The child exits0 and is reaped with an empty
session,but its stdout was not recovered;no numerical pass follows. Preserve
that failure and use a fresh host-v2 controller/root with the same binaries.
The two actual CPU units have O3/znver3;the fixture body is generic O3.
An auxiliary assertion mistakenly requiring znver3 on the fixture remains
recorded as failed;it was not part of the frozen native build acceptance.
The fresh `decode_iq3_host_v2` gate now passes all four same-build children
with the original exact384/80 and28-pool-case criteria;both pool checksums
match `e5d7a50a8e7d5ac5`. All native outputs are unchanged. V1 stays failed;
no GPU or model speed qualification follows. Only the six
qualified CPU/fixture files and checked per-request IQ3 route-output patch
`7969ccb9f` differ from the prior fixed-Q6 observer source; GPU math, prefill,
placement and graph/runtime sources remain byte-identical. The same-build
`decode_iq3_phase_device_v1` gate now passes submission10 and phase-cache16
windows/12 captures, with strict traced validation, normal exits, closed owners
and no fresh driver fault. Arithmetic/pool and small device prerequisites pass.
The IQ3-off model entry now also passes its two frontend probes, warmup and
two fresh32769-input/24-output thinking-on requests. Actual eligible NT1 rows
are1400960/1378560 and selected paired rows are0/0, with normal native exit,
quiet validation, all15 final asset hashes, closed owners and no fresh fault.
IQ3-on now passes the same probes and both short requests, with identical
calls/eligible rows and selected paired rows1400960/1378560. Both selectors
close normally with all15 final asset hashes, quiet validation and no new fault.
The six-process long-generation comparison has closed its first matched pair:
reference/candidate code TG10.2352/10.4004 and prose6.6905/6.8850,with exact
candidate/reference ratios1.016142/1.029070. Both arms generate3170/7275 tokens
with MTP acceptance0.82439/0.38924. All four requests satisfy their extent,
answer and actual-route gates;both native exits0,quiet validation,closed
ownership and no fresh fault. Candidate selected NT1 rows221271040/566117760
equal the eligible rows,while reference selected rows are0.
Four independent processes remain;one pair does not qualify repeated gain,
prefill/TTFT preservation or inference adoption.
Both arms retain the common IQ3 audit,whose incremental cost is not separately
measured;these are not a matched comparison against the older clean TG values.
The diagnostic preset's
`STRATA_DECODE_TIMING=0` is presence-enabled by the measured generator and emits
timing in both short arms; the new speed controller omits that variable.
These short diagnostics do not qualify sustained TG. The preload retains a
runtime reference until exit, so ordinary
runtime teardown and model speed remain unqualified. The closed observer and
CPU-only comparison outputs are now retired after original Git source/recipe
recovery and consumer checks:859 files,160869493 bytes. All240 samples and
the original CPU recipe remain at `cf0a6b175261f013099965dcb9ca19d1e840ad2d`;
the observer source is reconstructed by the pinned constructor. Protect479a.
No model gain or adoption follows from the component result or prepared build. The closed V2 generation was retired before
building the three V3 CPU-only targets; keep the validated GPU fallback unchanged.
The observer cohort is closed and its successful full output is
retired (408 files,84331126 bytes), with both final requests' clocks preserved.
The benchmark's 252 full blobs alone occupy 548352000 bytes, exceeding a
512 MiB process cap before library/scratch overhead. Its prospective measurement
cap is therefore 1 GiB; the small correctness fixture keeps its 512 MiB cap.
Prospective build limits are 300 s wall,1200 s CPU,8 GiB RSS,1 MiB text and384 MiB
private artifacts; the correctness child gets 60 s wall/CPU,512 MiB RSS and64 KiB
text. Normal exit0 plus supported ISA, all numerical/refusal cases and closed
owned processes are required before timing. A skip77 is not a correctness pass.
The reviewed retarget contract preserves the candidate's Q6/phase behavior and
requires checked observer-output completion: a truncated record or failed
write/flush cannot qualify attribution. The source review supplies no measured
CPU service or observer-overhead result.
The source-only output-v2 fix adds a versioned completion trailer, checks every
write and the final flush, and propagates failure at the generator call site.
The retargeted engine passes this short model entry, with representative
long-generation attribution and overhead still unqualified. The validated
fallback and recorded PP reference remain unchanged.
The reader now requires version 2 and an exact completion trailer, including
row counts and ordered thread slots; its host plan and result are in
`cache_map_cpu_service_sources_v1.reader_v2_host`. All 12 tests pass at source
`73642e6324b3403e196339dafb5aa5096538fa42`, including all 15 truncated prefixes
with an otherwise complete independent census. The owned host process exits
normally with no cleanup or survivors. This cannot qualify native write/flush
handling or observer overhead. The writer's host-only check passes at source `49c20a41c` using
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
retains nine ELF artifacts. The comparison is closed and its three candidate
ELFs are retired after committed observer-source preparation and original-input
recovery. Seventeen named project ELFs remain before the observer build,
including the existing configured serving and validation/profiling consumers.
The new observer adds four named outputs for its immediate qualification and
comparison, bringing the current retained count to 21. This is not a claim of
global cleanup. The legacy
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
