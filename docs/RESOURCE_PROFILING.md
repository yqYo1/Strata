# Hardware resources in prefill tuning

Measure which resource limits the dependency path before selecting an optimization.
Stage times alone do not identify PCIe, RAM, VRAM, matrix arithmetic, local memory,
synchronization or CPU submission as the limiting resource.

For each fixed long-input case, record source/binary/runtime/driver, actual tokens,
chunk/context, precision, KV format, placement and effective pre-exec environment.
Use the official benchmark workload and keep instrumented profiles separate from
repeated clean speed samples. Tiny runs are numerical/collector safety gates.

Check startup values observed inside the executable as well as the initial
environment supplied before exec. `/proc/PID/environ` records the initial exec
region and does not generally reflect later environment changes; matching it
alone does not prove the current process environment stayed unchanged. See the
[Linux environment documentation](https://man7.org/linux/man-pages/man5/proc_pid_environ.5.html).
If startup adds a runtime control, identify the observed value and supply it
before exec for the next qualified run; do not attribute its origin without evidence.

| Resource | Application evidence | Capacity comparison and limits |
| --- | --- | --- |
| PCIe | Copy direction, requested bytes, intervals, dependencies and copy/compute overlap | Measure both directions with matching sizes/allocation types. Requested bytes are not wire traffic; topology is not attained bandwidth. A fast busy copy can occupy a small fraction of the request. |
| RAM | Staging/packing bytes when source-proved, CPU time, measured memory traffic and data access pattern | Match single/multiple threads and streaming versus gather/copy. Separate logical read/write accounting from physical traffic. |
| VRAM/cache | Per-kernel physical read/write/cache counters when available, plus copy measurements | A logical device-copy roof is operation-specific. Do not infer kernel DRAM saturation from source load counts or copy bandwidth. |
| GPU arithmetic | Actual kernel work, dimensions, datatype/accumulation, supported arithmetic activity counters | Compare matching kernels/shapes. Large GEMM throughput and vendor peak are not utilization measurements for a tiled attention kernel. |
| SLM/registers/synchronization | Actual supported stall/occupancy/conflict counters or controlled component comparisons | Compiled SLM/register metadata and barrier counts do not measure occupancy or barrier wait. |
| CPU/control | Issuing-thread CPU time inside waits versus outside waits, submission/packing/grouping, other threads, observed command gaps | Completion polling can saturate one core. Gaps are not proof of hardware idle or CPU causality; check library fallback explicitly. |

Calculate service sums and interval unions separately; report overlapping work
and uncovered intervals. Name exact observed windows. An attention envelope is
not the whole prefill, and startup/decode/teardown must not silently enter its
utilization denominator. Do not bridge unrelated host/device clocks by assumption.

Call a compute-command interval union **command coverage**, never arithmetic
utilization. A command can spend its execution time on gathers, memory waits,
barriers, reductions or poorly filled matrix tiles. For example, the recorded
77.31% covers compute commands in a 94.945-second attention envelope; it does
not mean 77.31% of the GPU's FLOP capacity, or of the whole prefill. Multiplying
a resident model's tok/s by that fraction does not predict this engine's speed.

Keep measured resident-model results as hardware references. Record actual input
length, weight quantization, KV formats, flash attention, batch/microbatch,
warmup and timed boundaries. Active parameter counts alone do not account for
recurrent/attention work, routing, hyper-connections or matrix shapes. Such
differences are hypotheses to quantify, not evidence that a slower implementation
has reached its hardware limit. Preserve supplied results as user observations
until reproduced; do not invent their individual samples from a mean and spread.

Measure resource capacities for the operations actually used. Label older-stack
or different-shape figures as references. Use controlled changes to distinguish
competing explanations: lower bytes, better overlap, fewer submissions or better
arithmetic should predict a measured dependency-path change. Record uncertainty;
"GPU compute commands dominate" does not identify the resource inside kernels.

Recheck capacity when an unchanged binary becomes slower. Keep allocation type,
sizes, repetitions, CPU affinity and effective environment explicit, and compare
with the telemetry observer disabled. A difference across measurement dates is
not a source regression by itself. Read PCIe link metadata during the relevant
work, but reconcile it with attained copy bytes/time; a contradictory sysfs field
is not a usable performance ceiling. Keep the contradictory observations.
Capture the endpoint and every parent bridge, with PCI addresses, rather than
reading only the endpoint's link fields. Keep idle topology snapshots separate
from link samples taken during copies. An encoding-adjusted estimate from an
upstream link is still a theoretical link estimate, not a measured payload roof.

Do not declare counters unavailable solely from a policy file. Test bounded
capability enumeration without changing permissions, then separately qualify
activation/collection and interpretation. Conversely, enumeration does not prove
collection works. Preserve API failures and stop before unqualified model runs.

Keep numerical/lifetime/fault gates, physical262144 and RESTORE requirements.
Serialize builds/tests/GPU/cleanup, fix provider controls before native exec,
and follow [artifact retention](ARTIFACT_RETENTION.md). Commit compact evidence
and all samples used by decisions; retain large captures only for a named current
question with a finite budget and review point.

## Collection through the production graph adapter

The selected decode configuration sets `STRATA_SYCL_VERIFY_NATIVE_CAPTURE=0`.
Its ordinary compatibility graph uses `cudaStreamBeginCapture`, unprofiled
`cudaGraphInstantiate` and `graph_launch_with_completion` from the SYCL backend.
An SDK-only `command_graph` collection pass does not cover these wrappers.
[profile_compat_window.cpp](../tests/sycl/profile_compat_window.cpp) exercises
that adapter, its kernel submission hooks and checked completion/retirement,
with two named integer kernels and guarded allocations. It requires an external
owner to acknowledge each gate; its source alone establishes no collector result.
The manual `sycl_profile_compat_window` CMake target links the production
`strata_sycl_runtime`; it is excluded from the default build and from CTest.
Its controlled build recipe, compiler-input census and execution owner must be
committed before running it. Do not reuse an older SDK fixture's qualification.
[build_compat_profile.py](../tools/sycl/build_compat_profile.py) prepares and
builds only this target from the selected engine's configure recipe. It requires
the frozen six-process comparison to close and the original measured child
revision to be reachable from the pushed primary. It verifies the21runtime
sources against that build, then checks the four actual translation units and
compiler/link inputs. It performs no device work. The fixture-only build at
source `f4f83be4f` produces ELF `5046a535…`; its four translation units cover
711 actual compiler inputs and 24 external link inputs, with zero uncovered
inputs. The earlier preflight refusal remains failed at `7b8f50629`. A new
reviewed source audit proves that three Python-only changes leave the collector's
generated native files, 460 compiler dependencies, recipe and ELFs unchanged.
[profile_compat.py](../tools/sycl/profile_compat.py) controls the three cases in
separate processes,with a result commit between them. Its prepared host controls
in`tools/sycl/test_profile_compat.py`reject missing post-window copies,wrong
kernel inventories,unobserved replays and ambiguous/boundary-crossing instances.
The device controller requires all five host controls to pass without skips,
with their original committed source/plan recoverable and unchanged. The
device controller requires `compat_profile_window_host_v2`. Its
620-input plan at`4e5d7074d`passes all five controls with zero skips,normal0
and closed ownership at source`65b6b6349`. V1remains available in Git history;
the active proof is V2. This qualifies the host reader, not a new collector
device result. The selected-native reader recovers its original841
project inputs from the pushed primary and checks the original1461-compiler/
29-link census,SDK and same-build ELFs. It no longer needs the integrated child
checkout or the ended f633 fallback binaries. The fixture build instead retains
the selected90f binary and both independent phase guards. Admission now passes
before and after physical retirement of the old checkout/build and private
cohort root,including the production-adapter build-basis check.

Qualify direct execution, fully active graph construction/collection, and then
paused construction as separate cases. Keep the original paused-construction
SDK failure intact. The clean replay window contains six named kernel instances
and three readbacks. A subsequent replay adds two named instances and one
readback; the initial upload precedes the window. Verify native flow identities,
clock bounds, exact instance counts, numerical/guard results, normal exit, owner
closure and fresh faults. Keep collection active through native exit/flush.
The direct case passes at `e2ccc46a6`, and the active-before-construction case
passes at `44b5fac45`: all eight named instances and five copies are retained,
with six kernels/three readbacks in the inner window and two kernels/one
readback after it. The paused-construction case remains failed at `4d1528859`:
native guarded execution and normal exit0 pass, but the timeline contains only
four readbacks and no kernels. All three owners close with quiet validation
and no fresh GPU fault. Do not use this failed mode for model attribution.
[profile_compat_exec.py](../tools/sycl/profile_compat_exec.py) prepares a separate
active-from-exec control using the existing aggregate launcher with explicit
host/call logging. At controller source `6aa32694f`, all three CPU controls pass
with zero skips. Its same-ELF device process also passes: eight kernels/five
copies, unique native joins, contained clocks, normal0, quiet validation,
closed ownership and no fresh fault. `compat_profile_exec_v1` records this
separate pass; the three-case V1 and paused construction remain failed.
The model screen in `dflash2_device_service_v2` is now closed: three fresh
90f processes, each with one short thinking warmup and two exact32769-input,
24-output requests. All native/benchmark exits are0, owners close, validation
is quiet, no fresh fault occurs and all15final model hashes match. The626-input
controller is `66313f4f2`; its20CPU controls pass without skips.
Against the median of the two bracketing references, profile/reference prompt
ratios are1.066501/1.099303, clientTTFT1.070085/1.101877 and decode
1.516427/1.515178. Both decode ratios fail the declared<=1.5observer screen;
the cohort remains `passed=false`. Do not promote these observed services into
representative unprofiled attribution, clean speed or sustained TG.

The two guarded profile windows observe Q6-column service292.242/294.660ms
and mapped-row-copy service186.532/215.580ms. These sums select questions,
not hardware saturation or critical-path costs. The unprofiled caller ledgers
instead retain roughly0.96–1.02s in the CPU-pool wrapper and0.95–1.08s nested
native waits per24-output request; pool time includes scheduling/waits and the
wait thread's CPU time is not expert arithmetic. Mapped host-memory reads are
not counted by explicit copy-command payloads. Keep PCIe/RAM traffic and
complete device coverage unknown. The row-uniform-copy component is a bounded
next hypothesis. Its separately owned direct-entry screen now passes five CPU
controls,716compiler/24link input checks,80+1120guarded numerical cases and
600timing samples in three fresh processes. No host-USM cell improves in all
three;the only repeated win uses device input. Keep production unchanged and
model admission false. The next graph-replay comparison needs its own plan;
the direct-entry result does not settle that different dependency path.
An unprofiled completion must expose no graph timestamps. These checks do not
measure kernel arithmetic utilization, physical traffic, a lossless pause with
pending work, model observer overhead or inference performance. Requalify the
current collector/provider inputs before execution, preserve the owned launcher's
`ZES_ENABLE_SYSMAN=0` policy, and serialize this work after the frozen model cohort.

The current a347 engine, built from `221e95e9ad890383deb418d0a865ee32c99168b6`,
already has `STRATA_CPU_SERVICE_PROFILE:BOOL=ON`. Its compiled `pool.cpp` and
`pool.hpp` match the producers in `verify_window_cpu_service_overhead_v1`.
The existing `STRATA_CPU_SERVICE=1` startup setting can therefore expose the
CPU pool's GU/Down and other leaf wall/thread-CPU clocks without a new native
build. This is source/build evidence, not a current collection or overhead pass.
The older overhead result bounds only its original incremental observer cost;
it does not transfer that bound to a347 or establish RAM/arithmetic saturation.
The dense-DPAS official comparison is now closed. The source-prepared
`tools/sycl/profile_q6_cpu_leaves.py` defines six fresh observer-off/on,
on/off, off/on processes on its dense1/head2/backing0 decode candidate. It uses
the official four warmups and unchanged full thinking-enabled long-code task,
with at least32K fresh input,1K generated tokens and an answer; it does not
replace answer eligibility with a truncated thinking prefix. Only the startup
leaf flag differs. Current reader/controller controls and frontend probes must
pass before model execution. Matching work/output hashes and all three
per-token latency ratios within1.05 qualify the incremental observer cost only.
The new admission verifies the original native inputs, five ELFs and complete
compiler/link census, separately from this noncompiled observer. It consumes
the closed comparison evidence, without rerunning its historical V1 raw reader
or changing the original failure or adoption status. No new current-engine
profile or observer-cost pass follows from source preparation. Keep
leaf CPU time separate from completion polling and nested pool/wait intervals.
Test physical counter collection separately before making a RAM-bandwidth claim.

## Numerical reference for replacement kernels

Use the reached SGLang/vLLM path for the same model as the preferred algorithm
and precision reference. Record input/output storage, matrix accumulation,
reductions, conversion/rounding points, normalization constants, routing and
persistent-state updates at an immutable source revision. Related-model paths
need an explicit semantic mapping. A framework's enabled flag or a BF16 tensor
declaration alone does not prove the complete numerical contract.

The user accepts the validated SGLang/vLLM contract, including its lower
precision relative to llama.cpp. Do not require extra precision or identical
outputs merely to match the old ggml/FP32 route. Keep that route and FP64 small
oracles for finding implementation errors; explain their expected numerical
differences instead of treating every difference as model degradation. Preserve
input/lifetime/fault checks, state continuity and physical262144 boundaries.

Apply changed acceptance criteria to new named comparisons. Preserve original
failures and their criteria. Source agreement is useful evidence, but does not
by itself prove the new kernel implemented the contract correctly. A precision
change and a corrupted state or incorrect operation still require separate
diagnosis. Speed adoption requires repeated clean long-input measurements.
