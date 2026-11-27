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

### DFlash2 decode time budget

The closed V2 request-clock bracket at `e26315def2` has two unobserved
controls and one observed process using the same 33,217-token code prompt.
The [request-clock controller](../bench/sycl/dflash2_v2_decode_clocks.py)
checks the native footer against the client journal.
The caller ledger conserves the observed request wall time. Its 4,111 outputs
use 1,345 verifier windows, or 3.057 outputs per window. The caller spends
152.19 of its 327.28 CPU seconds inside native completion waits (46.50%).

| Disjoint caller wall interval | ms per window |
| --- | ---: |
| GPU layer completion | 106.115 |
| Expert pool wrapper | 110.293 |
| All other caller intervals | 35.370 |
| Total | 251.778 |

At this yield, 70 tok/s gives the whole window 43.664 ms. A sensitivity
calculation replaces the sum of the first two intervals with their maximum,
keeping their durations fixed. Perfect overlap at the recorded yield gives
20.98 tok/s with the other intervals unchanged and 27.71 tok/s with those
intervals entirely removed. The calculation assumes overlap even for intervals
whose dependencies may forbid it. Use it to prioritize cutting service time
and increasing useful verifier yield together.

In the retained Arc verifier at `968e71988`, a QSA layer scores blocks and
then launches one 256-thread selection workgroup per query. With a step at
96,204 cells, the selection path visits 24,052 scores in each of 32 threshold
rounds and two emission passes, or 3,271,072 logical score bytes per query.
At a 262,144-cell step the corresponding count is 8,913,032 bytes. These
counts describe loop visits; the repeated scans can reuse cached scores.
The SYCL build turns off the CUDA `STRATA_VERIFY_PROFILE` timer, so these
source counts supply neither physical traffic nor GPU stage time.

The radix histogram mapping available to the prefill selector scans the
same scores in eight threshold rounds, with more group reductions. It is a
candidate for a separate decode entry. A component comparison must keep the
current score inputs, cell weighting, tie order, partial final blocks and
exact selected IDs at verifier widths one through four. Compare the total
score-to-selection time and the isolated selector time. The existing shared
dispatcher also serves prefill and the single-token layer path, so a verifier
trial needs a phase-specific call site.

The separate [two-bit selector fixture](../bench/decode_qsa_topk_packed.py)
keeps the reference scorer and starts from the retained binary-threshold
selection entry. It tests sixteen two-bit scans in place of thirty-two
one-bit scans. Each scan counts the three nonzero digits of the current
prefix in 19-bit fields within one unsigned 64-bit sum. Each query covers
at most 262,144 cells, so no field can carry into its neighbor.
The zero digit is the remainder. Rank advances inside the selected prefix,
while the retained emission pass keeps the existing weighted tail and
ascending tie order.

For width four, subtracting the original scorer envelope from the original
combined graph in the profiled run below gives about 0.386 ms at 96K and
1.561 ms at capacity. This motivates a selector-only clock. Separate graph
envelopes can also differ in overhead. Compare selector-only and combined
completion on the same device allocation and stock score input, in three
independent processes. Check raw repeated keys, signed zero, infinities,
NaNs and negative scores alongside live scorer values, guards, all tail
remainders and replay across changing extents. Faster completion needs
measurement because the packed count adds integer operations.

The [decode scorer grid comparison](../bench/decode_qsa_score_grid.py)
isolates the preceding score dispatch. At the 262,144-cell allocation,
the existing scorer launches 65,538 workgroups per query. A 96,204-cell step
uses 24,052 of those groups. The candidate uses 4,096 groups per query and
reads the live step bounds inside the captured command. It retains the exact
FP32 operations and work-item reduction order from the committed stock
source. The independent fixture links the verified serving archives and
compiles two isolated units. Serving continues to call its original scorer.

The fixture replays the same graphs through short, 96K and physical-capacity
steps, including all tail remainders, tied scores, invalid steps and a
large-to-small transition. It checks complete score bits, ordered selected
IDs, guards and actual completion. After the traced smoke and boundary
checks, use three independent timing processes for widths one through four.
Measure scorer replay and scorer-plus-existing-selection replay separately,
with 512 dispatch/completion iterations per batch. Record graph setup and its
amortization. Advance a shape only if both clocks improve in all three
processes. Model speed and physical GPU traffic require separate evidence.
Run the component at a model-comparison idle boundary.

After the three clean timing processes close, use one separate profiled
process on the same fixture. Measure five alternating graph completions per
arm and shape. Record the outer graph's start-to-end device interval, timer
resolution and host launch-to-completion time. Check timestamp order on the
serial stream and repeat the score and ID checks. Graph profiling can change
graph realization. Use these samples for attribution and retain the clean
decision separately if device timestamps are unavailable.

Both correctness processes and all three timing processes close normally
without fresh Xe faults. Across widths one to four, the same graph passes
short, 96K and 262144-cell steps, including all four tail remainders, tied
scores, invalid steps and large-to-small transitions. All 288 timing
batches pass the duration and timer gates. Time ratios below are the
geometric means of paired process medians. Every shape passes both gates
in all three processes at `0bdc2446e`.

| Verifier width | 96K scorer | 96K scorer + selector | 256K scorer | 256K scorer + selector |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8443 | 0.9635 | 0.9600 | 0.9954 |
| 2 | 0.7700 | 0.9306 | 0.9449 | 0.9906 |
| 3 | 0.7169 | 0.8979 | 0.9011 | 0.9793 |
| 4 | 0.6925 | 0.8741 | 0.9059 | 0.9766 |

At width four, the combined median moves from 0.65484 to 0.57247 ms
at 96K and from 2.05994 to 2.01132 ms at capacity. One measured width-one
setup charges extra first capture to the candidate in reverse order.
That process would need 15,793 combined replays at 96K or 35,374 at
capacity to recover the extra setup. Preserve separate first-request
and later-request clocks in the model comparison. The measured 96K width-four
difference saves about one millisecond per window if applied to all twelve
QSA layers at that shape. Actual widths, surrounding dependencies and setup
determine the model-level gain. Advance the decoder's verifier call site
as a separate candidate with its own numerical and matched model checks.

The separate profiled process passes 160 ordered device-interval samples
with a 52 ns timer resolution and no fresh Xe fault. At width four the
combined graph's median device interval is 0.591 → 0.509 ms at 96K and
1.994 → 1.942 ms at capacity. The same process's wall medians are
0.656 → 0.574 and 2.066 → 2.011 ms. These are outer graph timestamps
with profiling enabled. They locate the reduction inside the graph in
that test. Arithmetic occupancy and physical memory traffic remain open.

The [coherent build](../bench/dflash2_decode_qsa_grid_native.py) applies
`d1bdd6872` to the exact retained primary source and compiles every target
from fresh sources. The changes touch verifier class definitions, so the
build includes the same-build host policy target and checks every affected
compiler input. The [admission runner](../bench/dflash2_decode_qsa_grid_admission.py)
checks real finalized graph scorer calls and short thinking-prompt parity
with the grid off and on. The grid stays off by default.

Short admission closes at `22ae164ed` with the same 82-token input, 32
generated IDs and 23 of 23 accepted drafts in both native processes. The
reference uses its existing scorer in every capture. The candidate uses the
existing scorer in both prompt captures and the bounded grid in all twelve
QSA layers of each decode capture at widths one, two and four. Both native
owners exit normally, with quiet parameter validation and no fresh Xe fault.

The [model timing plan](../bench/dflash2_decode_qsa_grid_speed.py) keeps
the stock global expert profile fixed and switches only
`STRATA_SYCL_DECODE_QSA_SCORE_GRID` between six fresh processes. It uses the
same 96,204- and 96,274-token thinking requests to resolve sustained decode,
prefill and first output timing. Finish the short native control before
timing. A successful timing screen permits the separate full-context,
RESTORE and quality checks.

The reverse-order second pair closes with both owners reaped and no fresh
Xe fault. Stock and grid emit 3,195 code tokens and 8,618 prose tokens.
Stock to grid changes code TG from 16.333 to 16.298 tok/s (−0.21%) and
prose TG from 11.636 to 11.676 tok/s (+0.35%). Code PP changes from
941.59 to 934.02 tok/s and TTFT from 103.530 to 104.354 seconds. Prose PP
changes from 1,040.19 to 1,041.07 tok/s and TTFT from 92.740 to 92.657
seconds. Finish the third matched pair before deciding.

The sixth process finishes both workloads and the native server exits 0
without fresh Xe faults. Its code and prose clocks are 938.642 and
1,040.725 tok/s PP, 16.263 and 11.625 tok/s TG, and 103.842 and 92.695 s
TTFT. The original comparison ends failed at its final build-controller
check. All 836 frozen measurement inputs match. The retained build's
historical controller selection includes all of `bench/sycl`, and three
CPU replay files changed during execution at `4a251fbf6`. Preserve the
sixth receipt and its failure and do not use the cohort for adoption.
The next cooperative-KV comparison needs a retained-build reader that
recovers the original controls from Git and verifies the current run's
inputs separately. It still needs the same candidate executable.

The closed samples in `eb99d69fb` limit the case for repeating the grid
immediately. In the first pair, the grid gains 1.99% code TG while changing
the generated count from 3,195 to 3,269 and the draft totals. The second
pair preserves code work and loses 0.21%. The two fully checked prose pairs
preserve work and gain 0.13% and 0.35%. The positive differences from the
third pair remain observations from the failed final build check. Keep the
reference scorer fixed for the cooperative-KV comparison. Revisit the grid
on independent model evidence before combining it with another change.

Retired the raw comparison and admission output scopes below after verifying
the closed owners and the compact committed decisions. Verified the root
`record.json` files. The cooperative KV comparison used the primary
fallback and the QSA build.
The inventories hash sorted state-relative paths, byte counts and SHA-256
values. No current reader uses those output directories.

| Scope under the tuning state root | Evidence | Files | Bytes | Inventory SHA-256 |
| --- | --- | ---: | ---: | --- |
| `dflash2-decode-ranked-generation-blend-v1/{host,0,1,2,3,4,5}` | `5e75c7dbe` | 436 | 90736645 | `03df792418095e2ec885876c36282b4f34befb1b05898a682109bc2958defb36` |
| `dflash2-decode-qsa-grid-admission-v1/{host,reference,grid}` | `22ae164ed` | 95 | 23090175 | `af84718cb8856aefc04661263fcbe113d9bc491ea1a277c426c0119ea8ce3b30` |
| `dflash2-decode-qsa-grid-speed-v1/{host,0,1,2,3,4,5}` | `eb99d69fb` | 436 | 90839294 | `ee1f2723cef99b759ffff0b3c5ced4a4aeef30ebd007183bc0c5bbb2d0f8e350` |

The closed cooperative KV comparison at `7017faf91` ends the QSA
candidate executable's last named use. The width-three model build
recovers its source and recipe from the compact record at `4ea51e5bc`
and applies the four committed width-three files. The pre-retirement
audit matches 1,327 original project files, 520 Git-pinned build controls,
1,476 compiler inputs, 30 link inputs, 140 objects and all 12 executable
identities, including the measured `strata` executable. The QSA admission,
QSA timing, KV admission and KV timing owners have all closed. Preserve their
original outcomes, including the failed QSA timing postcheck.

The retirement selection is every descendant of
`dflash2-decode-qsa-grid-native-v1` under the tuning state root except its
original `record.json`. It contains 1,525 regular files and 223,866,799
bytes, with SHA-256
`7e74fb0de99e62cc51ec4cabb9bf19a382530e94c1b322e75c4ce920271813c9`
over the compact JSON array of sorted state-relative paths, byte counts
and file hashes. The inventory matched immediately before deletion under
the measurement lock. The original receipt remains byte-identical. All 23
executable identities across the primary prefill and independent decode
references match before and after retirement.

Retired the component's raw runs and fixture from `decode-qsa-score-grid-v1`
under the tuning state directory and verified the path is absent. The
recovery check reproduces the recipe and generated
sources from `44bb81e4b`, all 693 compiler inputs and 27 link inputs,
and the exact fixture SHA-256
`dd309b0d35a51a389ac401417e3e7a3c63c9f15eaf19a0690348c0c2db9d6013`.
The closed evidence is reachable at `a3e2a9a3d` and the CSV at `973f5697b`.
The tree contains 70 files and 2,524,102 bytes. Its SHA-256 over sorted
relative paths, byte counts and file hashes is
`8fa279557ece815396e13e9346524edbc67d268aa14a964bdfa32b617931a679`.

### Cooperative KV resolution on DFlash2

The earlier MTP comparison `decode_kv_speed_v1` found paired decode gains
of 6.77% for code and 7.00% for prose. Its separate warmed 64K prefill check
failed preservation. The retained primary DFlash2 source `968e71988` already
contains the selectable decode resolver. The 262,144-cell K8V8 configuration
with 20,480 resident cells gives 65,536 logical and 5,120 resident four-cell
pages. Those dimensions match the existing gate. The gate also requires the
two-head, 256-element, 2,051-selection shape and width two or four. It
excludes prompt captures. The retained environment already fixes
`STRATA_SYCL_KV_COPY_GROUPS=0`, so the experiment can switch only
`STRATA_SYCL_DECODE_KV_COALESCED` before launch on the same executable.

Keep the QSA scorer and expert placement fixed. The [short admission runner](../bench/dflash2_decode_kv_admission.py)
compares a bounded native request in each arm with validation enabled. It requires matching
output IDs and drafts, scalar KV in every prompt capture and verified
cooperative calls at every intended layer of a decode width-two or width-four
capture. Verify scalar fallback at width one. Record resident page shape and
actual dispatch. The [timing runner](../bench/dflash2_decode_kv_speed.py)
then compares the same 96K code and prose bodies in three counterbalanced
process pairs against scalar KV. Report emitted tokens,
draft acceptance, PP, client TTFT and sustained TG. Advance only on repeated
decode gains and preserved prefill and TTFT within each workload. Test
physical context and RESTORE on the selected route before any default change.
The MTP result gives a reason to test this component. DFlash2 benefit and
interaction with the QSA grid need their own measurements.

The reader reconciles all 18 finalized captures from the first clean QSA
reference at `40c76d2f1`. A synthetic cooperative version activates seven
eligible captures. Four malformed variants are rejected, covering prompt
dispatch, width-one fallback, double dispatch and a missing capture. The
prepared short pair pins the exact executable and 82-input/32-output
request. Both native cases now pass with the same 32 output IDs and 23 of
23 accepted drafts. Both native owners exit normally with quiet validation
and no fresh Xe fault. The control uses the scalar resolver in all twelve
layers of the five captures. The candidate preserves both prompt captures
and width-one decode, and uses the cooperative resolver in all twelve
layers of the width-two and width-four decode captures. This qualifies the
short dispatch and parity checks. The 96K timing, capacity and RESTORE
checks remain pending.

The retained-build reader at `460621507` recovers the original QSA build's
controller files from the recorded Git revision and checks them against the
committed build receipt. The subsequent model comparison pins its own live
inputs separately. The cooperative-KV plan at `8e12b034d` prepares all six
processes with the same retained executable and passes both frontend host
probes. The original QSA comparison remains failed.

The first scalar process closes all route, executable and asset checks with
exit 0 and no fresh Xe faults. All 18 verifier captures use the scalar KV
resolver on the 12 QSA layers. The 96,204-token code request emits 3,195
tokens at 934.726 tok/s PP and 15.963 tok/s TG, with 104.284 s client TTFT.
The 96,274-token prose request emits 8,618 at 1,038.092 tok/s PP and 11.478
tok/s TG, with 92.928 s TTFT. Draft acceptance is 2,173/2,248 and
3,442/4,864 respectively. During this process the separately capped
report worker and native reviews use 62.186 CPU seconds in total, including
one timed-out review.

The first cooperative process closes with exit 0, quiet validation, no fresh
Xe fault, and matching final executable and model assets. Seven decode
captures switch all twelve layers to cooperative resolution. The two
prompt captures and the nine width-one or width-three decode captures use
scalar resolution. Both requests match the first scalar process's input
hashes, generated counts and draft totals.

| 96K workload | Prefill tok/s, scalar → cooperative | Client TTFT s, scalar → cooperative | Decode tok/s, scalar → cooperative |
| --- | ---: | ---: | ---: |
| Code | 934.726 → 937.323 | 104.284 → 103.978 | 15.963 → 17.134 |
| Prose | 1,038.092 → 1,039.622 | 92.928 → 92.790 | 11.478 → 11.867 |

The first pair improves decode by 7.34% and 3.38% respectively. The code
clock comes from the first long request and the prose clock from the next.
Complete the other two independent pairs before deciding whether the gain
and prefill preservation reproduce.

Both processes report the same KV resolver counters at the long request
footers. Code ends at 20,162,064 page lookups and 1,356.5 MiB of rounded
logical miss payload. Prose ends at 61,872,480 lookups and 1,878.2 MiB.
The footer reads the cache counters accumulated since map initialization
or reset and computes payload at 4,224 bytes per miss. Use the matched
totals as a work check at each footer. They include the requests' preceding
work since the reset. They give no physical PCIe byte count, achieved
transfer rate or separate decode interval. The TG gain motivates measuring
resolver work and its place in the dependency chain at fixed recorded demand.
[Counter readout](../src/program/generate.cpp#L7614) ·
[Map reset](../src/kernels/sycl/kv_stream.cpp#L147).

The second cooperative process closes normally with quiet validation,
unchanged assets and no fresh Xe fault. Code measures 938.946 tok/s PP,
103.797 s TTFT and 17.091 tok/s TG. Prose measures 1,040.538 tok/s PP,
92.706 s TTFT and 11.947 tok/s TG. Both requests match the preceding
cooperative process's input hashes, generated counts, draft totals, KV
footer totals and all 18 capture dispatch records. Compare this process
with the following scalar process in the reverse-order pair.

The reverse-order scalar process closes with quiet validation, matching assets
and no fresh Xe fault. Its code request measures 939.920 PP, 103.683 s TTFT
and 16.272 TG. The cooperative code request is 5.03% faster in TG at the
same output and reasoning hashes, 3,195 outputs and 2,173/2,248 drafts. Its
prefill is 0.10% slower and TTFT is 0.114 s longer. The scalar prose request
measures 1,035.824 PP, 93.127 s TTFT and 11.601 TG. Cooperative prose TG is
2.98% higher in this pair, with 8,618 outputs and 3,442/4,864 drafts versus
9,371 and 3,771/5,378 for scalar. The prose rate includes changed generation
work. Keep the original speed screen and report the work difference.

The third scalar process closes normally with quiet validation, matching
assets and no fresh Xe fault. Code measures 941.014 PP, 103.580 s TTFT and
16.264 TG. Prose measures 1,029.666 PP, 93.688 s TTFT and 11.601 TG. Both
requests reproduce the first scalar process's input, output and reasoning
hashes, emitted token counts and draft totals. The final cooperative process
matches both requests' exact output and reasoning hashes and draft totals.
Code measures 940.448 PP, 103.638 s TTFT and 17.128 TG. Prose measures
1,039.353 PP, 92.813 s TTFT and 11.948 TG.

All six processes close normally with quiet validation and no fresh Xe
fault. Decode improves in every pair, with geometric-mean gains of 5.889%
for code and 3.120% for prose. Code has the same work in all three pairs.
The second prose pair includes the different generated work recorded above.
The paired median candidate/reference prefill-time and TTFT ratios are
1.000602 and 1.000560 for code, and 0.995469 and 0.995479 for prose. The
predeclared prefill and TTFT limits therefore fail on code. Preserve the
scalar model selection and the original outcome. These paired differences
establish the screen result. The exact prompt dispatch is unchanged, and
the differences alone do not establish a causal prefill slowdown.

The archived prose trace at `b28a9372b` records a 33,287-token request
with 11,783 outputs on the unsplit DFlash2 cap-four path. It uses the same
262,144-cell logical and 20,480-cell resident K8V8 geometry. Replaying its
request footers and verifier positions accounts for 6,485 windows and 13,387
target rows, including all 6,902 offered drafts.

| Actual verifier width | Windows | Target rows |
| ---: | ---: | ---: |
| 1 | 2,533 | 2,533 |
| 2 | 1,987 | 3,974 |
| 3 | 980 | 2,940 |
| 4 | 985 | 3,940 |

The admitted width-two and width-four shapes cover 45.83% of these windows.
Width three adds 15.11% of windows and 21.96% of target rows. The resolver in
`decode_kv_stream.cpp` iterates across the provided query count, while the
verifier admission is restricted to the two measured widths. The same page
bound below gives at most 1,542 simultaneously selected pages for three
queries, within the stock 5,120 slots. These counts cover the archived 33K
workload and do not predict 96K TG.

After the current comparison, build the existing
[Arc fixture](../tools/sycl/kv_copy_groups_bench.cpp) in a private component
directory. The retained primary build has the three standalone compile
commands for the fixture and both resolver translation units, plus its
original `libstrata_sycl_runtime.a`. Both resolver sources, their public
headers and the oracle header match `968e71988`, the compared `d1bdd6872`
image and the current worktree. Reuse the retained runtime archive after
checking the actual header and link-input census. Build only those three
translation units, with the new committed fixture as the sole source overlay.
The [width-three controller](../bench/decode_kv_width3.py) freezes these
commands, the runtime and the comparison order in its `prepare` stage.
It uses the shared measurement lock and the existing input and ownership
checks. Commit the recorded stage before advancing through `build`,
`smoke`, `validate` and the six numbered timing processes.

Run `--planner-width3-smoke` first in a separate traced process. Require
two direct cases and two graph replays before proceeding to
`--planner-width3-validate` with parameter validation. Require five graph
replays per resolver in that second process, including zero and partial widths,
the last physical cell, epoch rollover and a rejected width. Require two
exact replays per resolver and miss count in the 5,120-slot shape at 0, 384
and 1,539 misses. Both resolvers must match the CPU reference for page maps,
counters, K/V and scale payloads, and guards. Stop on a failure, including
an input mismatch, provider warning, budget stop, unclosed owner or fresh
Xe fault.

The three-unit component build verifies 672 compiler inputs and 25 link
inputs at `8c6935fa4`. Its traced smoke passes two direct calls and two graph
replays. The full Arc check matches five fingerprints across both resolvers,
including the final physical cell, and checks two exact replays per resolver
at each of the three miss counts. Both device processes close normally and
report quiet validation and no fresh Xe fault.

Run `--planner-width3-benchmark scalar|cooperative` in three fresh process
pairs, alternating scalar-first, cooperative-first, scalar-first. Keep all
nine samples per process. Follow the existing warm-up, calibration and timer
checks. Require every paired median completion-time ratio below one for all
three miss counts before a model trial. The fixture covers the KV input to
attention. Validate downstream outputs and actual width-three dispatch in a
separate same-binary model check, with other widths and prefill fixed.
Keep the temporary fixture build for the bounded kernel trace below.
Review the build and its logs when the trace decision closes.

All three independent pairs complete their 54 timing samples with quiet
validation and no fresh Xe fault. The table gives the median of the three
process medians per arm and the range of paired reductions.

| Missing pages | Scalar completion (ms) | Cooperative completion (ms) | Paired reduction |
| ---: | ---: | ---: | ---: |
| 0 | 1.412 | 0.531 | 62.39–62.43% |
| 384 | 3.615 | 2.821 | 21.95–21.99% |
| 1,539 | 5.709 | 5.013 | 12.16–12.24% |

All nine paired ratios pass the component timing gate. Run the bounded
trace to attribute the change between the resolver and copy kernels.
Model throughput, actual width-three dispatch, prefill preservation and
physical-context/RESTORE checks remain open.

The model-trial source reads `STRATA_SYCL_DECODE_KV_COALESCED=3` at verifier
initialization. It admits cooperative resolution only for the three-query
decode operation on the DFlash2 target, at the existing streamed K8V8
geometry with the existing copy-group cap of zero. A fresh process with
`=0` runs the scalar reference. The earlier `=1` experiment still selects
only widths two and four. The host verifier-policy fixture covers all three
settings, malformed options, phase, owner and geometry. Build the model-trial
source only after the component speed gate. Rebuild all translation units that
consume the changed verifier or window policy headers. In both model arms
verify the finalized prompt and decode captures, and require actual width-three
work before comparing the results.

The [model-trial controller](../bench/dflash2_decode_kv_width3_native.py)
reconstructs the measured QSA model source from its committed projection and
applies only the four selector and policy-fixture files. Start its `prepare`,
`configure`, `compile` and `policy` stages after the committed width-three
component decision, with the applicable engine retention slot available.
Require the same-build host policy fixture before model execution. The exact
original QSA build receipt in Git supplies the baseline identity and recipe.
One new executable serves both scalar and width-three arms. Retain the
validated primary fallback through the comparison.

The previous 82-input admission processed widths one, two and four. For
width-three admission, recover the exact 33,287-input varied-prose prompt from
`b28a9372b` and bound the output for the initial probe. The archived process
verifies widths 1, 3, 3, 3, 4, 4, 1 and 2 in its first eight decode windows.
Keep spec4, draft confidence, starting placement and the unsplit verifier
fixed in both arms. Require width three and at least one scalar fallback width
in both actual traces. Keep the bounded output probe separate from the paired
96K throughput and physical-context/RESTORE checks.

The coherent build at `9f09133ae` checks 140 fresh objects, all 12 executable
identities and zero uncovered inputs. The same-build verifier policy fixture
passes at `ee148b6a7`. The [bounded model probe](../bench/dflash2_decode_kv_width3_admission.py)
recreates the archived thinking request using the original 200- and 800-word
calibration counts. Its HTTP body hash matches the archive. Both arms consume
the same 33,287 rendered tokens through native GEN with a 48-token output cap.

The original scalar attempt at `f2b5bd202` executes fresh layer-major
prefill, emits 48 tokens, accepts 21/30 drafts and captures decode widths
1, 3, 4 and 2. It exits normally with no fresh Xe fault. The checker
incorrectly expected a verifier prompt capture on the dedicated long
prefill path. Keep the failed outcome. The new plan binds the layer-major
route to one native footer with 33,287 fresh input tokens and 48 outputs.
It requires actual width-three and scalar fallback decode captures in both
fresh processes with the same native commands and prompt.

Both corrected model processes pass with quiet validation, normal exit and
no fresh Xe fault. Each reads 33,287 fresh tokens, emits the same 48 token
IDs and accepts 21 of 30 drafts. The scalar arm reports 12 scalar KV calls
at each captured width (1, 3, 4 and 2). The cooperative arm reports 12
cooperative calls at width three and 12 scalar calls at widths one, four
and two. Both enter the checked layer-major prefill route. This
establishes bounded same-model dispatch and output parity.

The [matched 96K controller](../bench/dflash2_decode_kv_width3_speed.py)
requires closed same-binary admission. Bind both workloads to the previous
96K request hashes. Validate scalar prompt captures in the short warm-ups
and layer-major prefill in both long workloads. Require width-three and
scalar fallback decode captures in each long request on the same binary.

The six independent processes match request, output and reasoning hashes,
draft counts and expert work in both workloads. The code request consumes
96,204 fresh inputs and emits 3,195 outputs. Prose consumes 96,274 and emits
8,618. Each candidate capture uses cooperative resolution in all 12 KV
layers at width three, with scalar captures at widths one, two and four.
All six exit normally with quiet validation and zero fresh Xe faults.

| Workload and pair | Prefill tok/s, scalar → cooperative | TTFT s, scalar → cooperative | Decode tok/s, scalar → cooperative |
| --- | ---: | ---: | ---: |
| Code 1 | 940.447 → 939.464 | 103.637 → 103.768 | 16.404 → 16.395 |
| Code 2, reverse order | 925.623 → 941.370 | 105.327 → 103.553 | 15.505 → 16.490 |
| Code 3 | 932.837 → 884.840 | 104.523 → 110.134 | 15.533 → 15.560 |
| Prose 1 | 1,040.051 → 1,038.072 | 92.752 → 92.941 | 11.692 → 11.758 |
| Prose 2, reverse order | 1,032.650 → 1,040.008 | 93.444 → 92.757 | 11.117 → 11.340 |
| Prose 3 | 1,033.254 → 1,022.204 | 93.372 → 94.377 | 11.105 → 11.205 |

The decode screen passes with paired geometric gains of 2.117% for code
and 1.157% for prose. Median paired prompt time increases by 0.105% and
0.191%, and median TTFT increases by 0.126% and 0.204%. Both preservation
gates fail. Keep the scalar route and end model qualification for this
candidate. The scalar code and prose rates fall about 5% between the first
and last scalar processes. The last pair's four warm-up prefill times agree
within 0.40%, while the measured code prompt time increases 5.42%. The
finalized prefill dispatch and work agree across arms. These clocks
leave the cause of the spread unresolved. The compact result retains
all six samples under `dflash2_decode_kv_width3_speed_v1`.

Retired the candidate build and the six closed timing processes below.
The two original `record.json` receipts remain byte-identical. The comparison is reachable at `0944ce4dd`
on `feature/sycl`. The recovery audit matches 1,327 original project files,
522 committed build control inputs, the original compiler and link inputs,
140 fresh objects and all 12 executable identities. It also matches the
six terminal process receipts to the committed outcomes. Inventory hashes
cover sorted state-relative paths, byte counts and SHA-256 values.

| Scope under tuning state | Files | Bytes | Inventory SHA-256 |
| --- | ---: | ---: | --- |
| `dflash2-decode-kv-width3-native-v1` | 1,528 | 223,893,554 | `0eefd91b0d0c473331639f0f5284b0169d0b668835628d98343b256fe018c708` |
| `dflash2-decode-kv-width3-speed-v1` | 439 | 90,973,435 | `9a415fa3562d2393eca72f94f1b8ba2da4298f04bf179e6ee6c69ae838d99746` |

All 23 executable identities match before and after retirement across the primary
prefill and independent expert-service builds. Removed the opt-in width-three
selector from the active engine and restored the four affected files byte for
byte to their pre-trial state. Recover the trial source from `9df3b8c66`.
Review the separate width-three component
fixture and both short admission directories at the next cleanup boundary.
Their evidence is closed. Check source recovery for the component fixture
and retain the original failed admission outcome before selecting files.

The earlier varied-prose model capture at `b28a9372b` has 6,485 decode
windows at 33,287 input tokens. The width counts are 2,533, 1,987, 980 and
985 for widths one to four. Their weighted sum of 13,387 rows agrees with
6,902 draft offers plus 6,485 windows. Width three accounts for 15.1% of
windows and can reach 11,760 layer calls across the 12 QSA layers. The
source log SHA-256 is
`6d4a4924485de26670e199318b8653cf3435da7237b3ee239e22d275430a0eb5`.

Applying the fixture's 0.696–0.881 ms savings to every such call in
series would save 8.19–10.36 seconds, or 0.71–0.90% of that capture's
1,151.62-second decode. This sensitivity calculation assumes the fixture
intervals transfer and every call is on the critical path. Miss density and
overlap can change the effect. The corresponding archived 96K window and
row totals permit at most 821 width-three windows in code and 2,432 in
prose, using widths one to four and the recorded width-one initial window.
Use the ongoing paired measurements for the 96K speed decision. The
limited scope of this change keeps CPU expert work and other verifier
widths in the optimization queue.

The earlier component result at `34cbaf29c` gives a control for the
mechanism. With zero missing pages, median launch-through-completion time
falls from 0.965 to 0.375 ms at width two and from 1.876 to 0.699 ms at width
four. Both arms still launch the copy kernel with zero payload. A separate
older trial that reduced copy workgroups to 128 or 512 had all-hit K8V8
time ratios of 0.9977 and 0.9971. It kept the scalar resolver and used
different fixture inputs. These results support inspecting the page-table
kernel before another change to copy geometry.

Use the fixture's bounded `--planner-width3-trace ARM MISSES` after clean
timing in a separate collector process. At each selected miss count it
replays the same two-node graph five times and checks the full oracle after
each launch. Begin with zero and 384 misses in both arms. Require five
distinct resolver instances and five matching copy instances on the same
native GPU, with successful setup, replay and exit and quiet parameter
validation. Separate the two kernels from map resets and readbacks. Keep
kernel service, launch gaps and wall completion separate. Match each kernel
to the fixture's `CLOCK_MONOTONIC_RAW` launch-through-completion bounds and
the collector's native process flow. The fixture records these bounds in
trace mode, between the completed map reset and oracle readback. Reuse the
pinned indexed UnitTrace collector only after its provider and instance checks
pass for this fixture. The controller runs `trace-scalar-0`,
`trace-cooperative-0`, `trace-cooperative-384` and `trace-scalar-384` in that
order and records each case separately from the unprofiled timing decision.

All four collector processes complete normally with quiet validation, no
fresh Xe fault and five native resolver/copy pairs per process. These are
medians of the instrumented device intervals and enclosing replay wall time.

| Missing pages | Resolver, scalar → cooperative (ms) | Copy, scalar → cooperative (ms) | Wall, scalar → cooperative (ms) |
| ---: | ---: | ---: | ---: |
| 0 | 3.134 → 1.117 | 0.0124 → 0.0125 | 3.271 → 1.259 |
| 384 | 7.203 → 4.794 | 0.5010 → 0.5019 | 7.843 → 5.418 |

The median gap between the kernels stays within 2.6–2.9 μs. The
resolver accounts for the observed savings at both miss counts. Use the
separate unprofiled samples for the speed decision. Physical PCIe traffic
and model TG remain unmeasured in this fixture. Retire complete traces
after reviewing the build and log dependencies.

#### Width-four KV cache working set

The retained K8V8 verifier at `968e71988` scores four-cell blocks and
allocates space for 2,051 selected cells per query. The SYCL selector emits
all four cells of each selected full block, except for at most one
threshold-tie cutoff block. At most one more block is partial at the end of
the context. Therefore a query touches at most
`floor(2051 / 4) + 2 = 514` pages. An identity selection touches at most
513 pages. The union over at most four queries fits in 2,056 pages. The
scalar resolver pins all selected hits before choosing victims and stamps
each newly placed miss in the same epoch. This bound covers the pages it
must keep simultaneously for one verified selection with a valid map.
The same conservative bound is 4,112 pages for width eight. Keep the
existing 5,120-page default for other configurations and shared callers.
[Selector](../src/kernels/sycl/qsa_select.cpp) ·
[Resolver](../src/kernels/sycl/kv_stream.cpp) ·
[Current sizing](../src/core/layer.cpp).

Each K8V8 page carries 4,224 bytes. Across 12 QSA layers, the allocation
reductions below exclude changes to the stage, indexer, page table and host
mirror. Slot metadata saves another 20 bytes per removed slot per layer.

| Resident pages per layer | Resident cells | KV payload saved versus 5,120 pages |
| ---: | ---: | ---: |
| 4,096 | 16,384 | 49.5 MiB |
| 3,072 | 12,288 | 99.0 MiB |
| 2,056 | 8,224 | 148.1 MiB |

The 3,072-page point leaves 1,016 slots above the width-four union bound.
Treat this as an experimental cache size. Window-to-window hit rate,
prefill placement, append, reject/rewind, RESTORE and capacity remain
unverified.
At 5,120 pages the scalar reference in `a5645a43d` logged 20,162,064
block lookups, a 98.33% hit rate and 1,356.5 MiB of requested KV payload
for the 96,204-token code request. The 96,274-token prose request logged
61,872,480 lookups, a 99.25% hit rate and 1,878.2 MiB. Both are fresh
requests. `session_zero` resets the per-layer map counters before each one.
The cooperative decode gate currently matches exactly 5,120 pages.
For a smaller-cache screen, keep the scalar resolver and expert placement
fixed in both arms. Check the K/V host mirror, full context, reused and
rejected suffixes and small-to-large requests before clean timing. Use the
existing KV stream miss, lookup and overflow counters with their map reset
points, plus separate clean TG and TTFT samples. Convert misses to requested
page payload using the verified K8V8 shape. The B570 bus traffic and overlap
remain separate measurements. Test allocating the released VRAM to expert
slots as its own comparison after the smaller-cache cost is known.

The initial implementation selects the smaller pool at session initialization
with `STRATA_SYCL_DFLASH2_KV_12288=1` and `--kv-resident 12288`.
Both arms request 12,288 cells. With the switch off, ordinary clamping gives
20,480 cells. With it on, require native single-device DFlash2, width four,
262,144 context, the two-head 256-element K8V8 shape and scalar resolve.
Check the actual resident count from the native session. The one-query
fallback can ask for at most 2,051 cells even with arbitrary page IDs.
Long batch prefill uses its own identity staging layout and the host mirror
still has all 262,144 cells. Their runtime checks remain separate.

The existing [SYCL KV fixture](../tests/sycl/kv.cpp) has a bounded
`--dflash2-kv-resident` case with 65,536 logical pages and 3,072 resident
slots. The four synthetic queries select 2,056 different pages, including
full pages, tied cutoffs and partial tails. Replay the same selections and
then switch the full pages while retaining the tails. Check actual selection
IDs, the four K/V and scale planes, inverse map, guard cells and hit/miss
counters before testing the real model.
The [isolated build](../bench/dflash2_decode_small_kv_build.py) applies the
four-file change through `4fd969361` on `968e71988`. It retains the original
compiler, dependencies and recipes and adds the same-build KV fixture.
The projection matches the committed bytes and excludes the separate
ranked-refill implementation. The first [syntax check](../bench/dflash2_candidate_sycl_syntax.py),
at controller `a17905e15` on source `08f852e85`, passed both generator
units (809 and 810 verified inputs) and the layer unit (726 inputs).
The fixture failed because `qsa_block_topk` had no visible declaration.
The correction in `4fd969361` includes the declaring header. The retry at
`56e6d5f53` passes all four units with no diagnostic: the two generator
units verify 809 and 810 inputs, the layer unit 726 and the fixture 695.
The full build and device check remain pending.

### DFlash2 expert service and placement

The independent worker-leaf bracket at `6bbd3e3e1` uses native source
`96832af6f` with the same request, output, reasoning and draft counts. Its two
observed runs use 156.12 and 156.24 CPU ms per emitted token across six expert
threads. Gate/Up accounts for 105.29 and 105.34 CPU ms. Six fully available
cores provide 85.71 CPU ms per token at 70 tok/s. Holding work and the other
CPU leaf times fixed would require reducing total leaf time by about 45%,
or making Gate/Up alone 3.02 to 3.03 times faster. This budget gives zero
extra CPU time to scheduling or work outside the measured leaves. Keep it
separate from the V2 caller-wall budget. Both apply to their measured routes.

The earlier default-route query observation at `55265f9e3` covers
208.368 s of decode. The resident pool-return query consumes 354.338 ms of
wrapper wall time, or 0.170% of decode. Its partner at launch adds 150.074 ms.
This puts the two resident queries at 0.242% on that measured route.

The width-four controller at `14f102cdb` uses the retained 16.694 tok/s
incumbent with two clean processes bracketing separate caller-ledger and
query observations. It checks identical serving inputs and expert work,
binds each observer to the actual request, and records its individual
overhead gate. Existing source and the offline control checks pass. Use
the completed attribution bracket's largest intervals and observed
format bins to choose placement and kernel candidates. Physical RAM
traffic, GPU saturation and the attainable overlap remain unmeasured.

The first clean process at `6dbbf2b02` emits 4,111 tokens at 16.696 tok/s
with 1,449,976 CPU expert jobs, or 352.71 jobs per output. The two worker
observations at `6bbd3e3e1` use the same executable, input, output and reasoning
hashes, draft counts, attention and KV path, and CPU job and entry totals as
the four width-four processes at `e0cb8fc25`. Their format and width counts
also agree with each other. Keep the timings from these two cohorts separate.

Using those format counts, sum `jobs × blob_bytes` for each gate/up and Down
format pair and width in `routes.leaves.format_nt_jobs`. The size for each
pair is constant across the 48 layers of `tests/data/native_experts/iq3_s.txt`
and matches the packed model manifest byte for byte. The sum is **2,956,400,972,800
logical bytes**, or **719.14 MB per output**. At 70 tok/s the same mix would
require 50.34 GB/s if each job read its full packed blob once. The separate
six-thread 256 MiB checksum-read calibration reached 24.03 GB/s of requested
payload. Expert and cache reuse, partial reads and the packed kernel's access
pattern determine actual DRAM traffic.

In both worker observations, 1,154,781 jobs (79.64%) handle a single token.
The three largest Gate/Up bins are single-token IQ3_XXS, IQ2_S and IQ3_S,
each with IQ4_NL Down. Together their Gate/Up work is 40.31–40.32% of total
leaf CPU time. Even eliminating all three would fall short of the roughly
45% leaf CPU reduction needed at fixed work and six available cores.

For decode kernels selected by Gate/Up format regardless of Down type,
sum the single-token bins across both Down types. The two pinned leaf logs
in `dflash2_decode_adapt_cost_replay.py` match their original hashes,
request work and total leaf CPU clocks. They give the following eligible
work per emitted token on that capture.

| Single-token Gate/Up format | Jobs per request | CPU ms per emitted token | Share of all CPU leaf time |
| --- | ---: | ---: | ---: |
| IQ3_XXS | 415,143 | 30.17–30.26 | 19.31–19.38% |
| IQ3_S | 228,839 | 21.32–21.36 | 13.65–13.67% |
| IQ2_S | 489,581 | 25.36–25.40 | 16.24–16.26% |

A twofold speedup of just the IQ3_S bin would leave 145.46–145.56 CPU ms
per emitted token, which gives a 41.22–41.25 tok/s CPU-leaf budget at six
fully available cores. The 70 tok/s budget would require removing about
92% of the three selected formats' combined single-token Gate/Up time
if other leaf work stayed fixed. Select the isolated IQ3_S replay for
the next kernel check. Keep expert placement, scheduling and GPU service
reduction in scope for the throughput target. These budgets use summed
CPU leaf time under fixed work and ideal core utilization.

The fixed-quota ranking replay at `410332a8f` uses the separate native
varied-prose capture at `b28a9372b`. Its 33,287-input, 11,783-output request
uses 6,485 verifier windows. Replay of the incumbent conserves 4,668,262 CPU
jobs, 5,982,938 routed CPU rows, all NT bins and 9,529,065,216,000 logical
blob bytes. Select experts per layer using only the first 3,242 windows.
Score against the original profile on the remaining 3,243 windows.
All profiles use the same 1,256 slots and 2,575,206,400 bytes of aligned
allocation.

| Profile | CPU jobs on held-out windows | CPU rows | Logical CPU blob bytes |
| --- | ---: | ---: | ---: |
| Incumbent | 2,581,636 | 3,359,952 | 5,266,123,776,000 |
| Rank by first-half distinct jobs per window | 1,928,873 | 2,361,989 | 3,923,950,464,000 |
| Rank by first-half routed rows | 1,924,474 | 2,351,266 | 3,915,350,502,400 |

The routed-row ranking replaces 1,124 slots and saves 25.46% of held-out CPU
jobs and 25.65% of logical CPU weight visits. It assigns 1,008,686 more rows
to the resident GPU path. The holdout is the second half of the same request,
so it measures within-request rank persistence after a long observation.
The earlier code-only static rank saves just 0.046% of jobs on the separate
prose request, and its traced cache-loan check at `489ac7c6c` raises CPU jobs
by 7.57–8.19% on the six synthetic requests. Keep the stock static rank.

The projection-granular replay at `c40be6f89`, run after the 126,977-token
process closed, conserves the original full-request stock counters. It uses
the same prose windows, first-half distinct-job rank and per-layer byte caps.
The table scores the held-out half. Intermediate readback counts the extra
requested 640-element FP32 transfer for each split row.

| Placement | CPU Gate/Up jobs | CPU Down jobs | Logical CPU weight bytes (TB) | Extra intermediate readback (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Stock full experts | 2,581,636 | 2,581,636 | 5.266 | 0 |
| Ranked full experts | 1,928,873 | 1,928,873 | 3.924 | 0 |
| Three quarters of slots reserved for full experts | 1,868,217 | 2,046,398 | 3.951 | 0.552 |
| Half of slots reserved for full experts | 1,818,197 | 2,169,439 | 3.995 | 1.102 |
| One quarter of slots reserved for full experts | 1,764,723 | 2,392,635 | 4.120 | 2.040 |
| Gate/Up only | 1,716,822 | 2,736,969 | 4.352 | 3.523 |

Maximizing training logical weight bytes saved selects full experts in
all 48 layers. Gate/Up-only residence removes a further 11.0% of CPU
Gate/Up jobs relative to ranked full residence, but adds 41.9% more CPU
Down jobs and 10.9% more logical CPU weight visits. Give the bounded
full-expert counting and placement comparison priority. Revisit the split
only with an execution design and measurements that can price the Down
handoff, synchronization and GPU work. Physical DRAM and PCIe traffic and
model latency are unmeasured in this replay.

The earlier `f6e3af5fa` replay offers the smaller runtime comparison. After
256 windows it projects a 3.27% suffix CPU-job reduction with 96 in-layer
swaps and 197,913,600 requested bytes in each direction. Use the same
decode counts in both arms after separately checking counting overhead.
Charge resident GPU work, publication and exact restoration of the original
profile before the next prefill. Compare a separately reserved sustained
workload and verify prefill preservation. The long-observation reader is
recoverable with `git show 410332a8f:bench/dflash2_decode_residency_replay.py`.
The counters describe assignment and logical work. Actual DRAM traffic and
end-to-end latency remain open.

The [layer-quota replay](../bench/dflash2_decode_layer_quota_replay.py)
uses the same verified prose trace and holds out its second half. It keeps
the first 132 stock experts and the exact width of every one of the 1,256
slots. Rank the remaining experts by unique CPU jobs from the first half.
Compare the stock map, a new within-layer ranking, and a ranking across
layers with the same aligned slot size. The third arm can change layer quotas
with 2,575,206,400 total arena bytes and the same 2,321,664,000-byte loan
tail. Each full global profile places a maximum-size excluded expert at the
first excluded position to preserve the loader's actual stop rule.
Advance to native admission only if the held-out CPU jobs and logical bytes
both fall against both controls. On the native build, compare the original
per-layer stock map with the same rank/slot map loaded globally first. Verify
actual mapping, byte accounting, occupied loan/refill, short handoff and
RESTORE before a profile comparison. Real traffic and latency require separate
measurements. The replay captures only compact workload results and ranking
hashes. Defer generating profile files until a named native trial needs them.
The first attempt at `46b767500` stopped before reading the trace because
the old physical checkout path for the stock profile had been retired.
The reader now recovers that tracked input from the original capture's
source revision and requires its hash to match both the recorded profile and
the working repository's profile. The original capture and hashes stay fixed.

The [decode-only capacity replay](../bench/dflash2_decode_extra_capacity_replay.py)
prices hypothetical extra residency after releasing long-prefill scratch.
Preserve all original 1,256 slots in order, then append prefixes of one
ranking at 256, 512 and 768 MiB. Rank nonresident experts by CPU jobs per
aligned slot byte on the captured training half. Measure job, row and
logical weight-byte savings on the unchanged held-out half. These caps
are hypotheses. Before a native trial, measure post-handoff headroom and
charge added upload, GPU work and retirement before the next prefill.

The replay at `a5328e29d` reproduces the original 4,668,262 CPU jobs. On
the holdout, the extra caps admit 137, 272 and 402 experts and remove 6.30%,
12.05% and 16.89% of CPU jobs, respectively. At 768 MiB the logical weight
visits drop by 0.890 TB (16.90%). These are same-work replay counts. Use
`dflash2_decode_extra_capacity_replay_v1` for the exact tested byte budgets,
rank hashes and sample counts. Check actual post-prefill headroom and
maximum allocation before choosing a native capacity test.

All three closed placement processes report 1,781 MiB free at startup and
845,776,220 accounted drafter bytes on each long request after activation.
The 96K prefill path reports a per-allocation maximum of 10,132,807,680
bytes. These observations are taken at different boundaries. The long
prefill destructors retire request scratch before cache refill, while the
draft arena is loaded after refill and the prompt tap owner is then released.
The existing loan audit observes free memory before the draft activation.
The three closed 96K processes leave at least 525,243,808 bytes above the
prefill plan's reserve; the 126,977-input primary run leaves 65,571,840.
Charging the replay's first 268,211,200 bytes at startup would consume
more than the latter margin, with other allocations fixed. A startup
trial needs an actual route check even if its 96K admission succeeds.

The stock native serve route now accepts the opt-in
`STRATA_SYCL_DECODE_HANDOFF_MEMORY=1` observer. It records byte-precise free
and total memory and the current expert arena immediately before prefill,
after refill, after DFlash2 activation and tap release, and after the last
decode commit completes if a window ran. Use a separate diagnostic process.
Bind the four ordered records to the same accepted prompt, actual split-prefill
allocation limit and unchanged stock placement. A missing query stops the
diagnostic. Source `b72ed4fb0` passes the retained serving target's C++20
`-fsyntax-only` check with oneAPI 2026.2 `icpx`, with no compiler output and
closed process ownership. The pinned compile database SHA-256 is
`660138a4f65f93f17485225cad35b3155ba262644b14dca894013d1d8e1e5e4b`.
The [handoff diagnostic controller](../bench/dflash2_decode_handoff_memory.py)
rebuilds the original source and targets with only the serving generator
observer changed. It checks the restored stock profile, two fresh
94,209-input requests and two fresh 126,977-input requests, with each
length in its own process. Keep the qualified prefill and decode references
through this temporary third generation and retire the observer build after
the diagnostic closes. The isolated configuration passes with 1,327 exact
source files and the original eleven-target recipe. Only the serving
generator is overlaid. CMake's new compile database is pinned in
`dflash2_decode_handoff_native_v1`. The build closes with 139 fresh objects,
1,475 verified compiler inputs and all 11 executable identities. The
`strata` binary SHA-256 is
`1c7531daa1667461952eb5a808e04090eb07bf5cf51e7c2f1cc13493393e8ecc`.
The model plan binds that binary and 820 committed inputs to both pairs of
original request hashes, with 94,209 followed by 126,977 input tokens.
Both current-source frontend probes pass. The 94,209- and 126,977-token
processes close normally without new Xe faults. All four fresh prompts keep
the expected mode16 route, expert arena, draft width and 24-output scope.
The final model asset hashes match. Free memory in bytes is:

| Input | Request | Before prefill | After refill | After draft activation | After decode |
| ---: | --- | ---: | ---: | ---: | ---: |
| 94,209 | First long request | 1,154,351,104 | 761,860,096 | 217,968,640 | 217,772,032 |
| 94,209 | Subsequent long request | 955,392,000 | 761,659,392 | 217,763,840 | 217,763,840 |
| 126,977 | First long request | 1,154,351,104 | 694,493,184 | 180,158,464 | 179,961,856 |
| 126,977 | Subsequent long request | 955,396,096 | 694,292,480 | 179,953,664 | 179,888,128 |

The reported single-allocation limit is 10,132,807,680 bytes. The lowest
decode free-memory sample is 179,888,128 bytes (171.6 MiB), leaving an
88,323,072-byte shortfall against the replay's first 268,211,200-byte slice.
Keep the stock expert arena and test the head allocation reduction first.
Keep diagnostic durations out of the speed comparison.

Recovered all 1,327 build inputs and the relocated recipe at `212cc99cb`,
including 1,475 compiler inputs, 139 objects and all 11 executable hashes.
Both model owners were closed when the two scopes below were retired.
Each root now contains only `record.json`. Both retained phase executables
and all four receipts match their pre-retirement hashes. Inventories hash
sorted tuples of state-relative path, bytes and SHA-256.

| Scope | Files | Bytes | Inventory SHA-256 |
| --- | ---: | ---: | --- |
| `dflash2-decode-handoff-native-v1` except `record.json` | 1523 | 223655597 | `c5cc173c8f7874a846cb11fdc4db103c46ad396784c24d8d345e32d2b108208f` |
| `dflash2-decode-handoff-memory-v1` `host`, `0` and `1` | 146 | 25058990 | `dbbb63d8a2fc9144f43841368fdf86acedf2a0a1cd6ecf0d63c96c334eb27eb1` |

The existing successful layer-major runs already verify that this runtime
exposed and returned the Intel free-memory query during prefill. The
reported bytes, per-allocation limit and physical residency are distinct.
Follow a positive post-activation margin with a separate bounded allocation
and completed device-page touch/readback on the same runtime before making
a coexistence claim.
The reviewed cache-lifetime audit confirms that existing captured hit
readers use one cache base and a residency table. A contiguous startup
tail keeps that representation at the cost of persistent capacity. A
request-local arena requires a graph-visible lookup for both allocations
and checked reader retirement before unpublishing the extra slots. Recheck
the 256 MiB budget after a measured reduction in existing decode allocations.

The 94,209-token observations make the DFlash2 vocabulary projection a
capacity candidate. `Dflash2Drafter::bind` gathers the 106,299 configured
vocabulary rows out of the resident 248,320-row canonical Q6_K head. Each
row is 2,100 bytes. The duplicate accounts for 223,227,900 bytes, or
212.9 MiB, and the same allocation appears in both requests' launch logs.
The unchanged vocabulary map has SHA-256
`b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`.
It contains 15,377 contiguous runs and one backward jump. The gathered
copy and indexed view visit the same logical weight bytes. The indexed
addresses span more pages, so measure their GPU service cost.

Test a DFlash2-only Q6_K entry that reads the canonical head through the
existing device vocabulary map. Preserve its output row order, exact
MMVQ arithmetic, Q8_1 input, top-k mapping, queue ownership and borrowed
head lifetime. Leave the gathered implementation available for the
comparison. Verify all selected rows and widths, boundary and nonmonotonic
IDs, buffer guards and actual dispatch before timing. The archived MTP
indexed packed-DPAS result uses a different kernel and reference; recover
its time and guards from `a0ede5973` and its recorded comparison.
Compare the DFlash2 head change alone first. Measure end-to-end memory at
the same handoff marks and sustained decode separately. Evaluate any
additional expert residency as its own change once real allocation coverage
and capacity allow it. Preserve long-prefill and RESTORE admission at both
steps and in any combined candidate.

Source `9866bec75` adds `STRATA_SYCL_DFLASH2_INDEXED_HEAD=1` for canonical
Q6_K with a draft vocabulary. The startup log reports the indexed path
and owned head bytes. The `--indexed-q6-head` fixture compares gathered
and indexed output bits at 256 and 2,560 input elements over seven draft
widths, including tail, invalid row and canary cases. Head release already
refuses any still-live DFlash2 owner, and the long request destroys that
owner before release and binds the next owner after restoration. The
[isolated build](https://github.com/yqYo1/Strata/blob/5a187fc59/bench/dflash2_decode_indexed_head_build.py) overlays the
five changed sources and the previously measured memory observer, then
rebuilds all primary targets with the same recipe. Configuration passes
with 1,327 exact source inputs, 522 controller inputs and the original 11
targets. The build verifies all 1,475 compiler inputs, 139 fresh objects
and 11 executable identities. The `strata` SHA-256 is
`0722edfc2fb1db9a0bc57c77ade047065cf16474de38c8fdec149c8d2f93bf23`.
The [traced GPU fixture](https://github.com/yqYo1/Strata/blob/5a187fc59/bench/dflash2_decode_indexed_head_fixture.py)
binds the new test to this build and the retained loader and runtime.
All 14 cases pass bitwise parity, invalid-row and output-canary checks.
The traced native process exits normally, ownership closes and validation
reports no fresh Xe fault. The model admission reads the committed fixture
result. After inventory verification, retired
`dflash2-decode-indexed-head-fixture-v1/indexed` and preserved the root
receipt and executable hashes. The six files occupied 4,348,547 bytes with
SHA-256
`13a6b856107cba6163b4e5b16c5358ad88925fc7d3febe9a55634d1fd3b90e0d`
over the sorted state-relative path, byte-count and SHA-256 tuples.

The [indexed head memory comparison](https://github.com/yqYo1/Strata/blob/5a187fc59/bench/dflash2_decode_indexed_head_memory.py)
uses the same executable, stock expert arena and two exact 126,977-token
bodies in each arm. Each request checks its actual head path, drafter-owned
bytes and all four memory boundaries. The gathered arm runs first and the
indexed arm uses a fresh process. The allocation-probe gate requires matching
recorded work, positive free-memory deltas at both decode boundaries and
enough reported room for the existing 268,211,200-byte candidate. A separate
device allocation and touch/readback will test physical capacity. Use a
separate sustained comparison to judge throughput. Both current-source
frontend probes pass with the pinned 823 controller inputs and exact
request bodies. The gathered arm closes normally with no fresh Xe fault.
Each request observes 223,227,900 owned head bytes. The first and subsequent
post-decode queries report 179,892,224 and 179,818,496 free bytes,
respectively. The indexed arm closes normally with no fresh Xe fault and
records zero owned head bytes in both requests. Actual mode16 prefill,
expert work, KV residency and draft counters match the gathered control.
The indexed post-decode queries report 179,892,224 and 179,822,592 bytes,
for gains of zero and 4,096 bytes. The after-activation gains are also
zero and 4,096 bytes. The second indexed request reports 261,099,520 fewer
free bytes before prefill, then reaches the same after-refill value.

The lowest indexed decode result leaves an 88,388,608-byte shortfall
against the proposed expert budget. The allocation-probe gate fails.
Keep the gathered selection. The follow-up diagnostic used the same executable
and the existing `client_resources.snapshot` reader at the
owned prefill-only request gates. Compare deduplicated Xe client requested
and resident VRAM with the SYCL free-memory marks. This tests whether
driver allocation or residency changed despite the free-memory result.
[DRM client memory counters](https://docs.kernel.org/gpu/drm-usage-stats.html#memory).

The [Xe memory comparison](https://github.com/yqYo1/Strata/blob/5a187fc59/bench/dflash2_decode_indexed_head_drm.py)
reverses the original process order and reuses the two exact 126,977-token
requests in each arm. The host reads Xe counters at each existing begin and
end gate while the client is waiting, for four snapshots per process. Retain
per-client requested, resident and shared bytes with the owned native
identity. Require unchanged, private client membership within each process
and matched work across arms before aggregating VRAM deltas. Associate the
end readings with the native post-decode marks. Admission of additional
allocations and clean speed still have separate evidence requirements.

Both arms close normally with no fresh Xe fault. All four requests match the
frozen input, prefill route, expert work, KV placement and draft counters.
Each native process has one private DRM client at all four boundaries. Its
reported requested and resident VRAM totals agree at every sampled boundary.

| Request boundary | Indexed minus gathered, Xe VRAM | Indexed minus gathered, SYCL free VRAM |
| --- | ---: | ---: |
| First begin, before prefill | −223,285,248 B | 0 B |
| First end, after decode | +4,096 B | −4,096 B |
| Second begin, before prefill | +4,096 B | −261,103,616 B |
| Second end, after decode | 0 B | 0 B |

The initial 212.94 MiB per-client reduction is consistent with removing the
gathered head. It disappears by the first post-decode boundary, including
when running the indexed process first. Aggregate counters cannot assign the
intervening growth to particular allocations. Keep gathered selection and
close this capacity candidate. The 256 MiB expert allocation remains below
its admission gate. Use a separate decode experiment for the CPU result copy
and any further residency changes.

The Xe comparison reads the closed head-memory result at `6cae88023`,
its exact executable and frozen request hashes. That result preserves the
phase bytes and all request gates. Retired its `host`, `0` and `1` directories
under `dflash2-decode-indexed-head-memory-v1` after verifying the inventory.
The root receipt and executable hashes are unchanged. The inventory contains
146 files, 25,295,661 bytes,
and hashes to `913d712ab541ffd026bfc802d358ab2a9d91536331ed26e4c96dfc1045e3e570`
over the sorted state-relative path, byte-count and SHA-256 tuples.

The indexed-head experiment is closed at `5a187fc59`. Its source-only change
at `9866bec75` and the four one-off controllers are retired from the active
tree. The original commits preserve the source and evidence. Verified the
recovered Git projection, actual compiler and linker inputs, and all 11
executable hashes before retiring the comparison generation. Retained the
root receipts with verified hashes and the independently validated prefill
and decode engines. The same sorted inventory format identifies these scopes.

| Root and retired paths | Files | Bytes | Inventory SHA-256 |
| --- | ---: | ---: | --- |
| `dflash2-decode-indexed-head-drm-v1/{host,0,1}` | 146 | 25301020 | `bdf3484ddc94ca87a5afdc94123eee6e783cba6770cc0fe5a68513bbcd85ea2a` |
| `dflash2-decode-indexed-head-native-v1/{build,cache,owned,ownership.json,source,tmp}` | 1,523 | 224263473 | `f1252cd7fa60ceabbc4d208f8ddb7b3cbe805fa33f5c341bec0eb9e73e780cab` |

The replay at `e3d371974` verifies 4,668,262 original stock CPU jobs.
Training uses the first 3,242 verifier windows and evaluation uses the
remaining 3,243. All arms preserve the same protected members, 1,256 slot
widths and loan extent. The simulated global loader requests 968 maximum-size
slots and stops after the exact 1,256-slot budget.

| Held-out map | CPU unique jobs | CPU routed entries | Logical packed bytes visited |
| --- | ---: | ---: | ---: |
| Stock | 2,581,636 | 3,359,952 | 5,266,123,776,000 |
| Within-layer rerank | 2,013,072 | 2,475,778 | 4,086,496,000,000 |
| Across equal-size layers | 1,967,341 | 2,409,104 | 3,999,896,140,800 |

The cross-layer map lowers CPU jobs by 23.79% against stock and 2.27%
against the within-layer control. Its logical bytes fall by 24.04% and
2.12% respectively, passing the prospective replay gate. Its layer quotas
range from 16 to 45. The packed rank-order identities are `07362e5b275d`
for stock, `f731c6a0fd04` for within-layer and `2863acb7b313` for cross-layer
(SHA-256 prefixes). The measured workload supplied both halves. Check the
exact stock global layout and prefill/RESTORE control first, then use separate
workloads to measure whether the change reduces decode time.

### Decode Gate/Up task claims

The reviewed Zen 3 report recommends trying coarser work claims with
unchanged packed arithmetic. The [CPU replay](https://github.com/yqYo1/Strata/blob/bed3f9f91/bench/sycl/cpu_roofline/decode_gu_groups.py)
keeps fifteen logical row partitions and five pinned worker cores with the
caller reserved. A claim handles one, two or four existing partitions in
order. The existing epoch, retirement and per-phase barriers remain in use.
There are 15, 8 or 4 Gate/Up claims per submission. Hidden quantization and
Down retain their original scheduling. The selector is compiled only in the
explicit CPU fixture target build.

Replay IQ3_XXS, IQ3_S and IQ2_S Gate/Up with IQ4_NL Down. Use eight single-row
expert jobs per batch and eight submissions per pass from the held-out
DFlash2 route. The hot case repeats its first batch and the rotating case
covers more than 64 MiB of actual packed weights. Check exact output bits,
buffer guards, actual claims and affinity before and after timing. Run three
cyclic independent process rounds with five batches of at least one second
per arm. Reconcile completed claims with the iteration count in every timed
batch. Advance a format and group only if all three rotating comparisons
improve both Gate/Up time and total pass time. The group of four can leave
one worker idle at the start, so its reduced claim count must also pass the
end-to-end comparison.

Preparation at `bed3f9f91` recovers the original route and freezes the six
descriptors and exact shard slices. All cases have 64 single-row jobs in
eight submissions. The hot cases use 15,769,600–18,636,800 packed bytes and
the rotating cases use 124,185,600–142,105,600 bytes. The verified Zen 3
fixture passes all 18 correctness cases and finishes 54 independent timing
processes. Each has five eligible timing batches. The rotating results below
are geometric means of the three paired total-time ratios.

| Gate/Up type with IQ4_NL Down | Two partitions per claim | Four partitions per claim |
| --- | ---: | ---: |
| IQ3_XXS | 1.120 | 1.322 |
| IQ3_S | 1.135 | 1.207 |
| IQ2_S | 1.098 | 1.290 |

Both group sizes take longer in every hot and rotating pair, on both the
Gate/Up and full-pass clocks. The group-claim changes fail the component gate.
Keep one partition per claim. The control validates work, placement and claim
counts. The results leave worker imbalance, weight traffic and arithmetic
cost unresolved. Use the existing worker-scaling methods with fixed native
expert work and a matched streaming control before attributing the gap to
memory bandwidth or scheduling.

The compact result at `9f19d2c6d` retains all 270 timing batches and the
original build identity. Verified the original Git source, actual compiler
and linker inputs and executable before retiring the build and run outputs
under `cpu-decode-gu-task-groups-v1`. The 226 retired files occupy
16,258,030 bytes. The sorted inventory SHA-256 is
`501b57ad0cd4392634ed9e4edd6349cba9ec289b51fa024fdc26a49368500213`.
The root receipt is unchanged. The grouped-claim fixture and test-only
scheduler are retired from the active tree. The measured source remains
recoverable at `bed3f9f91`.

The [fixed CPU replay](../bench/sycl/cpu_roofline/fixed_replay.py) now has a
locality comparison for the Gate/Up interval. Use the first eight ordered
CPU jobs from width-two window 4,801 in each of layers 0 and 17 of the same
held-out descriptor used at `cd60b5041`. Keep both packed-weight formats,
seeded activations, row order, five pinned workers and the reserved caller.
Both arms allocate four identical weight copies. Each pass submits the same
eight jobs four times, either reusing one copy or cycling through all four.
The active address footprints are 17,408,000 versus 69,632,000 bytes for
IQ3_XXS and 18,636,800 versus 74,547,200 bytes for IQ3_S.

The first preparation stopped before creating its run directory because the
capture referenced a retired worktree. The reader now checks the tracked
profile against the capture digest and the identical bytes at its original
source revision. It preserves the original resident membership and descriptor.

Check complete serial output, every copied weight byte, buffer guards and
affinity. Gather per-worker calls and rows in separate observed processes.
Use three counterbalanced clean process pairs per format with five timing
batches each. Compare whole-pass and Gate/Up intervals and keep the observed
worker distribution as context for interpreting the pair.

The CPU build verifies 633 actual compiler inputs. All four observed cases
pass exact serial-output and worker-accounting checks. The 12 clean CPU
processes complete 60 eligible timing batches. Table times are medians of
process medians. The percentage changes are geometric means of the three
paired rotating/hot time ratios per format.

| Gate/Up type | Hot pass | Rotating pass | Full-pass time change | Gate/Up time change |
| --- | ---: | ---: | ---: | ---: |
| IQ3_XXS | 2.737 ms | 2.812 ms | +2.56% | +1.28% |
| IQ3_S | 3.297 ms | 3.386 ms | +2.51% | +1.71% |

Rotating takes longer in every pair on both clocks. The effect is modest for
this repeated eight-job shape, with actual worker assignments recorded in the
separate observed runs. Keep reducing routed CPU work and measuring verifier
service before choosing a larger locality rewrite. Physical DRAM traffic, the
last worker's completion latency and model speed need separate evidence.

The [fixed replay](../bench/sycl/cpu_roofline/fixed_replay.py) also compares
`STRATA_IQ_SINGLE_EXACT=0` and `iq3s` through `--iq3s-single`. The latter
selects the already parity-tested native single-token IQ3_S Gate/Up entry
only in decode. Keep the original eight batches, five worker cores, packed
weights, IQ3_XXS pairing and multi-token dispatch. Of the captured jobs, 81
are single-token IQ3_S. Verify exactly 103,680 candidate GU rows in a
separate audited control, from the pool and its serial check, and compare
the guarded output digests across every process. Use three independent
counterbalanced, unobserved pairs with five eligible samples per process.
Require lower GU and total pool medians in all three pairs. This compares
the exact single-stream entry with GGML through the production row-split
pool, separate from the prior paired IQ3_S experiment.

The [closed IQ3_S result](../bench/results/2026-10-15-pr6-recovery-audit/cpu-fixed-expert-iq3s-single-current.json)
uses source `38006368a` and a build with 633 verified compiler inputs.
Each replay covers 201 expert jobs, 226 routed entries and 374,425,600
bytes of unique packed weights. The check processes report zero candidate
rows in the reference and exactly 103,680 in the candidate. All eight
processes return the same guarded output digest and close normally.
The six counter-free timing processes supply 30 eligible batches.

| Pair | Gate/Up ms, reference → IQ3_S | Whole pool ms, reference → IQ3_S |
| --- | ---: | ---: |
| 1 | 15.812 → 15.487 | 21.217 → 20.961 |
| 2, reverse order | 15.885 → 15.518 | 21.354 → 20.968 |
| 3 | 15.846 → 15.160 | 21.208 → 20.455 |

Paired geometric mean time falls 2.907% for Gate/Up and 2.192% for the
whole pool. Both clocks improve in every pair. This passes the CPU
component gate. An isolated model comparison must bind the production
dispatch to the same format and keep prefill and expert placement fixed.
The replay does not measure PCIe, GPU overlap or model throughput.

The isolated model build projects
the two CPU selector files from `c0db4198f` onto the measured `968e71988`
engine. Its inputs match the replay's AVX2 arithmetic source and the
primary CPU selector preimages. Retain the primary recipe and link the
existing GGML parity fixture against the same CPU library. Keep the
ranked-refill and smaller-KV options outside this source projection.
After parity, compare `STRATA_IQ_SINGLE_EXACT=0` and `iq3s` in fresh
processes with the same expert map. Verify the actual native decode work
before clean model timing. The separate route-census phase reference
remains needed through its pending comparison. Retire this temporary
candidate at a failed model gate or closing timing decision.

The isolated build finishes with 140 fresh objects from 135 translation
units and 12 executable identities. Run its existing `iq_single_exact_test`
once without arguments in the pinned build environment, with a 30-second
wall limit. Require normal closed ownership, no diagnostic stderr and the
exact 384 single and 80 paired GGML parity and row-canary result. This host
fixture uses the same CPU library as the serving executable.
It exits zero and returns that exact result with empty stderr and closed
ownership. The compact proof pins it to the serving executable's build
receipt under `dflash2_decode_iq3s_parity_v1`.

The two bounded model runs read
33,287 tokens through the dedicated layer-major prefill and emit the same
48 IDs with 21 of 30 DFlash2 drafts accepted. The CPU census matches all
48 expert residency masks and 17,075 jobs, including 2,970 single-token
IQ3_S jobs. Both processes exit normally with quiet validation and no fresh
Xe faults. The census establishes eligible work. The production executable
omits the test-only per-row selector counter.

The 96K timing comparison changes
only this selector on the same executable. It uses three counterbalanced
independent pairs with the fixed global expert map and the two thinking
workloads. The six processes complete normally with all final assets checked
and no fresh Xe fault. The table uses the paired geometric-mean change in
decode throughput and the median paired changes in prompt time and TTFT.

| Workload | Matched generated work | Decode tok/s | Prompt time | Client TTFT |
| --- | ---: | ---: | ---: | ---: |
| Code, 96,204 input tokens | 3 of 3 pairs | −0.754% | +0.442% | +0.424% |
| Prose, 96,274 input tokens | 2 of 3 pairs | −0.420% | +1.072% | +1.073% |

Code has one positive decode pair and prose has two. The last prose pair
emits 8,618 GGML and 9,053 IQ3_S tokens, with different drafts and expert
work. Treat that pair as workload throughput. The three GGML repeats span
15.571–15.891 tok/s for code and 11.108–11.371 tok/s for prose. The comparison
fails sustained gain, prompt-time preservation and TTFT preservation.
Keep GGML on the selected model route. The compact result retains the six
processes and the decision under `dflash2_decode_iq3s_speed_v1`. The CPU
fixture's gain applies to its captured work. [Original build and comparison
controllers](https://github.com/yqYo1/Strata/tree/369ea6cb1/bench).

At `369ea6cb1`, the recovery audit matches 1,327 Git source files and 521
controller inputs, the recorded compiler/link inputs, 140 objects and 12
executable identities. The six terminal timing receipts match the committed
samples. Retired the temporary build and both model-run output trees. All
three original `record.json` receipts and 23 retained phase-reference
executables match after removal. Removed this experiment's format selector,
CPU replay mode and model controllers. The original comparison is recoverable
from the published commit. The inventory hashes cover sorted state-relative
file paths, byte counts and SHA-256 digests.

| Root, excluding `record.json` | Files | Bytes | Inventory SHA-256 |
| --- | ---: | ---: | --- |
| `dflash2-decode-iq3s-build-v1` | 1,529 | 227,143,188 | `9a9f2b06753cd6d1862cefe40e79ffe0b2db345f13c16f726cef58b3f1831fe1` |
| `dflash2-decode-iq3s-model-v1` | 118 | 25,213,590 | `f1bf5c524a997a3a49b26e23984f866014030556865aa0f55b5ec89b06c743fd` |
| `dflash2-decode-iq3s-speed-v1` | 436 | 90,760,069 | `b36b85ab4d2f1e05a609c6753a2ce81fc20f41af8044059224a2dc1eabd69623` |

The timed CPU fixture has no live executable consumer. Its source and results
are reachable from `feature/sycl` at `14180e833`. The recovery check matches
210 project inputs, 1,376 GGML inputs, all 633 actual compiler inputs, the
external linker and runtime inputs and the final executable. The eight
processes have closed. Retired the descendants of
`cpu-fixed-expert-iq3s-single-v1` under the tuning state root and verified
that only its original `record.json` remains. The 93 files total
16,164,268 bytes. SHA-256 over
sorted run-relative paths, byte counts and file hashes is
`9e8274aeca70722e211780434302e3f55624a933d4ffe62989fb9bce633a6cd3`.

The [closed CPU result](../bench/results/2026-10-15-pr6-recovery-audit/cpu-fixed-expert-locality-current.json)
at `0b7646e0c` retains all samples and the exact build recipe. Retired
`cpu-fixed-expert-locality-v1` and verified its absence. Recovered the project and
GGML source from Git and verified the actual compiler and linker inputs,
runtime identities and executable. The closed inventory has 110 files and
16,503,343 bytes with SHA-256
`9277f3d5034636228953e60f273a02269c494243834bfbcf3376cbee7d89f2b2`
over the sorted compact JSON list of relative paths, sizes and file hashes.

The same [fixed replay](../bench/sycl/cpu_roofline/fixed_replay.py) has a
`--schedule` comparison for the dependency between Gate/Up, hidden
quantization and Down. The reference retires all Gate/Up row tasks,
quantizes the intermediates on the caller, and starts a second pool epoch
for Down. The candidate keeps these three operations in one claim per
expert and retires all jobs before returning. Its scratch and output
ownership remain per job. It uses the regular pool's claim and retirement
protocol. Only the CPU replay build can select this candidate.

Replay the original eight held-out batches with 13–36 distinct jobs per
batch, 201 jobs and 226 routed entries in total. The weights cover IQ3_XXS
and IQ3_S with IQ4_NL Down and the inputs exercise one to three rows per
expert. Use five pinned workers and the caller, with the current DFlash2
IQ3 selection. Check serial-output bits and guards in each arm. Three
counterbalanced pairs each time the full eight-batch pass first. Then time
the two captured batches for each combination of Gate/Up format and width.
Use five eligible timing samples per group in each process, with separate
warm-up and the existing timer-overhead check. Reconcile jobs and logical
packed bytes with the original descriptor for each group.

| Gate/Up and verifier width | CPU jobs per captured batch | CPU token rows per captured batch |
| --- | --- | --- |
| IQ3_XXS, width two | 20 / 13 | 20 / 14 |
| IQ3_XXS, width four | 36 / 35 | 39 / 39 |
| IQ3_S, width two | 17 / 14 | 20 / 16 |
| IQ3_S, width four | 35 / 31 | 40 / 38 |

Require the by-expert median to be faster in all three pairs for the
proposed scope. A broad follow-up requires the full replay and all four
groups to pass. A group that passes independently can enter a separate
scoped check. Earlier Down20 synthetic data included regressions and
some mixes exceeded the width-four routing limit. These two batches per
group are legal captured work, with seeded activations and a narrower
repeated working set when isolated. Treat scope results as component
evidence for those batches. A gain includes changes in balance, locality
and epoch costs. Per-stage wall times from the split scheduler do not
transfer to the candidate's overlapping worker stages. Native serving and
hardware throughput claims require separate evidence.

The first timing attempt at `8191a29dd` passes both host correctness
cases. Its first timed process hits the 180 CPU-second limit after
31.011 wall seconds with 22 of 25 samples and no success marker.
The owner exits with SIGXCPU and is reaped. The attempt has no eligible
pair. The retry uses the same executable hash, workloads, order, sample
floor, timer limit and decision. It gives each process 300 CPU seconds
within the same 60 wall seconds.

The comparison at `b4500d7f5` closes with all six processes and 150
timed samples. Exact outputs, job counts and worker placement pass in
every process. Use the original phase-split CPU scheduler for width two.
Restrict the early-publication follow-up to width four, where both
format cells pass all three pairs.

| Replay scope | By-expert over split time, three pairs | Geometric mean |
| --- | ---: | ---: |
| All eight batches | 0.9305 / 0.8480 / 0.9666 | 0.9136 |
| IQ3_XXS, width two | 0.9778 / 0.7926 / 1.0618 | 0.9371 |
| IQ3_XXS, width four | 0.8368 / 0.7186 / 0.9239 | 0.8221 |
| IQ3_S, width two | 1.0878 / 1.0710 / 1.0736 | 1.0774 |
| IQ3_S, width four | 0.9742 / 0.9780 / 0.9684 | 0.9735 |

The production reference and candidate both use five workers plus the
caller in this fixture. Measure the same width-four work with the
caller reserved before attributing an overlap gain to the early callback.
[CPU scheduling results](../bench/results/2026-10-15-pr6-recovery-audit/cpu-fixed-expert-schedule-current.json).

The width-four `--handoff` replay measures four arms across the same four
captured layers. All use five pinned workers. The caller either computes
with the stock scheduler, stays reserved with the stock scheduler, stays
reserved with the by-expert scheduler, or observes first-pair readiness
in the by-expert scheduler. Measure five samples per format and arm in
three independently started, counterbalanced blocks. Use the identical
137 expert jobs and bitwise output checks. The early arm divides the pool
call at the actual publication callback, retaining time through retirement
and the outer wall clock. The measured tail bounds the available CPU-side
overlap. A format can advance when hiding that tail would leave positive
headroom against stock with caller participation in each block. GPU graph
costs and actual overlap need their own measurement.
[CPU replay controller](../bench/sycl/cpu_roofline/fixed_replay.py).

The replay at `7d38b0680` closes all 12 timed processes with 120
eligible samples and exact work, guards, affinity and numerical results.
Each format row sums two captured layers. The values are medians over
the three independent blocks, in milliseconds.

| Format | Stock with caller | Stock reserved caller | By-expert reserved caller | Early reserved caller | Early readiness | Time after readiness | Ideal headroom versus stock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IQ3_XXS | 5.402 | 6.073 | 5.731 | 5.747 | 3.126 | 2.638 | 2.272 |
| IQ3_S | 6.026 | 6.833 | 6.857 | 6.818 | 4.059 | 2.767 | 1.962 |

The early arm costs 0.346–0.390 ms more than stock for IQ3_XXS and
0.774–0.812 ms more for IQ3_S across the matched blocks. The integrated
trial must recover those intervals and charge the extra GPU work and
submission costs. Both formats have positive CPU-only headroom in all
three blocks. The replay keeps the early strategy eligible for a
combined CPU/Arc test.
[Handoff result](../bench/results/2026-10-15-pr6-recovery-audit/cpu-fixed-expert-handoff-current.json).

Keep `cpu-fixed-expert-handoff-v1` as the current fixed-work CPU
reference through the combined replay. The earlier scheduling result
supplies its width-four eligibility from Git.

The three earlier roots below have no active consumer beyond their receipts.
The recovery audit verifies each earlier build's 210 Git project files,
1,376 Git ggml files, 633 actual compiler dependencies, link inputs and
original recipe. The retired paths are the contents below except
`record.json`, with inventory hashes over sorted relative paths, byte
counts and file SHA-256 values.

| CPU fixture root | Original result | Files | Bytes | Inventory SHA-256 |
| --- | --- | ---: | ---: | --- |
| `cpu-fixed-expert-publication-v1` | `62d23a297` | 79 | 16,136,713 | `63b4355c9268d912b3c3448a5786e4129ed4174f3d45be6a70d5d764f8082c61` |
| `cpu-fixed-expert-schedule-v1` | `8191a29dd` | 83 | 16,143,593 | `6240dc46daa4c91ee0471b02c219501acf93436287399d5f61e60c8875cd8021` |
| `cpu-fixed-expert-schedule-v2` | `b4500d7f5` | 93 | 16,194,661 | `db394a41c71bbd10808009fd95b87c591444e57c9340c0a0669781790c260840` |

The [stock admission controller](../bench/dflash2_decode_stock_global_restore.py)
uses the retained primary executable and recovers the original restore plans
from `3d75f7e91`. It generates only the stock global profile. Both loaders
run the 32,769- and 253,919-token park/restore protocols with the same short
handoff and 8,209-token resumed suffix. Require identical active state,
output, full loan/refill and all 48 actual membership masks. Advance reranked
profiles only after this control passes. The stock file and four native plans
are pinned at `033583182`, and both owned host protocol cases pass. The
per-layer stock arm parks 32,769 tokens, switches to a 33-token prompt,
then restores and reads the 8,209-token suffix to reach 40,978. Its native
process exits normally with its holder closed and no fresh Xe fault. The
identical global arm passes with the same request states and outputs, snapshot
payload, placement and loan/refill. Its owner closes normally with no fresh
Xe fault. The near-capacity per-layer stock arm then restores 253,919 tokens
from RAM, reads 8,209 new tokens and emits one token at 262,128. The readback
verifies 3,607,695,360 bytes of target KV and 118,069,304 bytes of running
state. All three requests match the expected 48-layer placement, both loan
and refill cycles pass, and the owner closes normally with no fresh Xe fault.
The near-capacity global arm matches the per-layer arm exactly across the
three request states and outputs, RAM restore payload, loan/refill and
48-layer placement. Its owner closes normally with no fresh Xe fault and all
15 final asset hashes match. Both extents pass, closing the stock control.
Keep the generated profile for the ranked comparison. These cases qualify
identical placement admission and restore behavior at the tested extents.

After stock admission, the same controller's `--ranked prepare` action
materializes the exact within-layer and cross-layer rerankings from the
original captured route. It checks the expected ranking digests and
held-out counts, keeps every slot size and the first 132 members, and
freezes the two generated profiles before launching. The global loader
uses each profile's first 1,256 pairs as the expected membership. Both
profiles get the same short-handoff and near-262144 restore checks, each
against its own saved state. Native placement, clean PP/TG, and full-context
qualification follow in separate checks.

The host replay at `62125e84b` reproduces both held-out job and byte totals
and both complete ranking hashes. Writing and reading the resulting profile
files preserves every pair, the protected slots and the 1,256-slot loader
extent. The within-layer ranking passes its first native case at `f4c4f9daa`.
It restores 32,769 parked tokens and reads 8,209 more. The readback, actual
membership, two loan/refill cycles and native teardown all pass with no fresh
Xe fault. Its final output differs from the stock control, and its three
single-token requests use 471, 476 and 442 CPU jobs versus 469, 469 and 414
on stock. The cross-layer ranking passes the same case with its own verified
snapshot and membership and no fresh Xe fault. Its three outputs match the
stock outputs and it uses 465, 474 and 447 CPU jobs. The near-capacity
within-layer run also passes. It restores 253,919 parked tokens, reads 8,209
and emits one token at 262,128. Its three output hashes match the stock
control at this extent, the saved-state readback and both loan/refill cycles
pass, and no fresh Xe fault occurs. The cross-layer ranking also passes the
same near-capacity readback, extent, membership and loan/refill gates, with
normal exit and no fresh Xe fault. Its first two outputs match stock. The
resumed request emits a different token and stops at 262,128. Both placement
controls close with final model asset hashes verified. This admits the two
profiles to the matched speed comparisons. Check numerical and task behavior
before adopting either profile.

After closing both controls, verified the compact results against the local
receipts and retired their owned raw outputs. Both original receipts and all
three profile files retain their original hashes. The reviewed numerical-scope
record remains in the research index. The following inventories hash sorted
state-relative paths, file byte counts and file hashes.

| Retired scope | Files | Bytes | Inventory SHA-256 |
| --- | ---: | ---: | --- |
| Stock restore `host`, `short-stock`, `short-global`, `near-stock`, `near-global` | 233 | 50392534 | `54dbaddbe4191093dd5836c1d289861f17d921a64b96940f444c49ca00ffb817` |
| Ranked restore `host`, `short-within`, `short-cross`, `near-within`, `near-cross` | 233 | 50392654 | `0b54c8a36a1033920794db457ffb8b9c2f9eb5b80930e76530db1d776768d62d` |
| `dflash2-expert-placement-numerics-20261126` report, control, review and source-view roots | 40 | 1533998 | `6917b6bbeda44ea22a058872cc1a99570b50de9f34242cd2c9f5dd324860a6bc` |

The [ranked generation comparison](../bench/dflash2_decode_ranked_generation.py)
keeps the global loader fixed. First compare stock with the within-layer
ranking, then compare within-layer with cross-layer in a separate run plan.
Each plan uses three counterbalanced process pairs and both fixed 96K
thinking workloads. The controller verifies the generated files against the
short and near-capacity RESTORE receipts before launch. It checks actual
prefill dispatch and records decode rate, draft acceptance and CPU expert
jobs per output separately. The long-form code and prose screen can advance
a candidate to task, agent, observed-weak and physical-context qualification.
The host check at `c22a389b5` replays the prior native routes through the
original parser and the new startup branch. It checks that the comparison's
arguments differ only by profile path and that a prefill slowdown blocks
advancement even when the decode comparison passes. Both RESTORE controls
are closed. The pinned stock/within-layer comparison passes its frontend
checks at `5b4bf351d` and closes its first stock process with no fresh Xe fault.
The 96,204-input code case measures 941.5 PP and 15.92 TG tok/s with 3,195
outputs. The 96,274-input prose case measures 1,040.1 PP and 11.34 TG tok/s
with 7,687 outputs. Both exercised the same prefill routing and requested
expert-upload bytes as the earlier per-layer reference. Code output and draft
work match that reference. Prose uses the identical input and emits 931 fewer
tokens, so keep the output, stop and draft counters when comparing its decode
rates. The first within-layer process also closes with no fresh Xe fault.
Both arms match the two actual prefill routes and requested upload bytes.
Code measures 933.0 PP and 15.89 TG tok/s with 3,370 outputs and 365.4 CPU
jobs per output, versus 354.8 on stock. Prose measures 1,041.1 PP and 13.15 TG
tok/s with 6,820 outputs and 263.3 CPU jobs per output, versus 403.4 on stock.
Prose TG is 15.94% higher in this pair. Both requests change generated work.
Keep that distinction and complete the remaining counterbalanced pairs
before selecting a placement.

The third process completes the within-layer arm of the reversed pair with
normal exit and no fresh Xe fault. Code repeats the preceding within-layer
output and reasoning hashes, 3,370 outputs and 2,273 of 2,387 accepted drafts.
It gives 941.0 PP and 15.96 TG tok/s. Prose gives 1,040.9 PP and 13.10 TG
tok/s with 6,939 outputs and 2,679 of 3,901 accepted drafts. Its reasoning hash
matches the preceding within-layer arm and its answer hash differs. Both
requests keep the matched prefill dispatch and requested upload bytes. The
fourth process closes the reversed stock arm with normal exit and no fresh
Xe fault. It repeats the stock code output and reasoning hashes at 939.2 PP
and 16.05 TG tok/s. Prose gives 1,040.2 PP and 11.61 TG tok/s with 8,618
outputs and 3,442 of 4,864 accepted drafts. The prefill route and upload bytes
still match both arms. The within-layer TG ratios are 0.9979 and 0.9943 for
code and 1.1594 and 1.1285 for prose so far. The frozen screen needs two TG
wins on each workload, which the within-layer profile can no longer meet on
code. Complete the third pair to measure the remaining spread and finish the
recorded comparison without changing its criteria.
The fifth process closes the final stock arm without a fresh Xe fault.
It repeats the code and prose output and reasoning hashes of the fourth
process. Code reaches 939.9 PP and 16.23 TG tok/s on 3,195 outputs; prose
reaches 1,038.9 PP and 11.61 TG tok/s on 8,618 outputs. The final within-layer
process closes normally with no fresh Xe fault, matched prefill dispatch
and matching final model asset hashes. Its code request gives 940.3 PP and
15.93 TG tok/s with 3,370 outputs. Prose gives 1,038.8 PP and 13.08 TG tok/s
with 6,939 outputs. Both repeat the preceding within-layer draft counts.
All three process pairs preserve median prefill time. The within-layer
TG ratios are 0.9979, 0.9943 and 0.9816 for code and 1.1594, 1.1285 and
1.1268 for prose. CPU jobs per output rise from 354.8 to 365.4 for code and
fall from 398–403 to 263 for prose. Generated work changes between maps.
The sustained gain screen fails on code. Keep the stock map. Cross-layer
placement has independent RESTORE evidence and no clean generation result.

The [blended rank replay](../bench/dflash2_decode_blended_rank_replay.py)
tests whether both workloads can reduce CPU expert work at the same VRAM
budget. Use the verified code usage reconstructed at `b28a9372b` and the
first half of its separate prose capture. Preserve the original code run's
parser failure and use only the closed offline reconstruction. Normalize
distinct job counts by each workload's own stock CPU misses. Compare code
weights of 0, 25, 50, 75 and 100 percent within the same per-layer quotas,
protected prefix and slot widths. Select among the three blends by the
smallest worst training ratio across jobs and logical bytes in both workloads.
Break ties toward 50 percent, then the lower code weight. The second half of
the prose capture cannot influence selection. Advance to a new independent
capture only if both metrics fall against stock on both training workloads
and the prose suffix. Code training has no holdout and the replay measures
neither actual traffic nor throughput. Keep stock for serving.

The replay on `7a5b12b5a` conserves the native stock counts and reproduces
the prose-only global map's 2,013,072 suffix CPU jobs. The selection uses
75% code weight and 25% prose weight with the original 1,256 slots and
2,575,206,400 bytes. Relative to stock, CPU jobs and logical weight bytes
fall by 17.57% and 17.74% on code training, 8.48% and 8.52% on prose
training, and 9.24% and 9.37% on the reserved prose suffix. The two
single-workload maps each raise CPU jobs on the other training request.
This passes the replay gate. Check a separately captured current-route
code workload and a distinct prose request before native placement or
throughput testing. Keep the all-ratio counts and selected map digest in
`dflash2_decode_blended_rank_replay_v1`.

The [independent placement replay](../bench/dflash2_blend_independent_capture.py)
reconstructs only the frozen 75%-code blend at `03d0b3cac`. Require its
ranking digest, protected prefix and packed slot sizes before using the two
separate counted 96K code and prose request bodies from `9261c4b44`. Each
body runs on the retained primary engine with the ordinary stock placement
and four official warm-ups in a fresh process. The task types match training
and the request bodies and long context differ. Keep the actual warm-up and
last-request journals, route checks and CPU counters binding each trace.
Score both layouts against the exact same routes. Advance to a new placement
trial only if CPU jobs and logical packed bytes improve on both captures.
This diagnostic does not change the live map or establish throughput or
physical traffic. The bounded raw traces serve only this comparison.
The preparation at `2e4136098` stops before any host probe or model process.
The ordinary suite requires matching final bodies between arms. The new
plan keeps that rule for normal comparisons and checks matched warm-ups
plus distinct, predeclared long bodies in this capture.
The separate preparation at `dde46c573` also stops before launch. Its
appended trace option moved after resolution by the serving frontend.
The next plan uses the resolver's user option slot and checks that resolving
the saved arguments again preserves the exact argv.

The two earlier preparation roots are retired. Both held only the
verified setup configs and unlaunched receipts. Inventory hashes use sorted
state-relative paths, byte counts and file hashes.

| Retired root | Files | Bytes | Inventory SHA-256 |
| --- | ---: | ---: | --- |
| `dflash2-blend-independent-capture-v1` | 5 | 118,297 | `2f1249199a1acb377ad4e3fee02ce05c49ddcec8b266280b4975bc129daea0b9` |
| `dflash2-blend-independent-capture-v2` | 5 | 118,379 | `7f5f9cb24c8c4726e1e0aba3c93001b244c883e68409f7fb495ee7853d868157` |

The independent 96,204-input code capture completes with 3,195 outputs,
1,024 verifier windows and 3,272 verified rows. The replay reconciles
1,133,707 native CPU jobs and 1,411,108 native CPU entries. The frozen blend
would use 936,662 CPU jobs and 1,907,371,315,200 logical packed bytes versus
2,312,898,073,600 on stock. The reductions are 17.38% and 17.53% on these
captured routes.

The independent 96,274-input prose capture completes with 8,618 outputs,
5,177 verifier windows and 10,041 verified rows. The replay reconciles
3,430,897 native CPU jobs and 4,484,620 native CPU entries. The blended map
would use 3,113,832 CPU jobs and 6,353,114,547,200 logical packed bytes,
compared with 7,005,568,563,200 logical bytes on stock. CPU jobs fall 9.24%
and packed bytes fall 9.31%. Both native processes close normally with
quiet validation, closed ownership and no fresh Xe fault. The final model
asset hashes match. The frozen map passes the two-workload gate for live
placement validation. Serving keeps the measured stock placement while the
candidate's native RESTORE and clean throughput remain open.

The [native placement controller](../bench/dflash2_decode_stock_global_restore.py)
uses the exact capture decision and reconstructs the selected profile before
preparing it. Preserve the stock control's 1,256 physical slots, slot widths
and first 132 resident entries. Use the same retained executable and check
actual loader membership and loan/refill payloads, live RESTORE and the
first verifier row on the 32,769- and 253,919-token parked prefixes. The
[generation comparison](../bench/dflash2_decode_ranked_generation.py) admits
the blend only after both checks close. It keeps the global loader fixed and
compares PP, TTFT and sustained TG on the same two 96K bodies in three process
pairs, accounting for changes in output length and draft acceptance.

The host protocol checks pass. On Arc, the short case restores all 32,769
parked tokens from RAM, reads 8,209 new tokens and reaches position 40,978.
Both long requests return all 1,124 loaned slots with matching payloads, and
the recorded decode membership matches the selected profile. Validation is
quiet and the native process exits normally without a fresh Xe fault.

The near-capacity case restores 253,919 tokens, reads another 8,209 and
emits one token at a 262,128-token prompt extent. The RAM snapshot checks
3,607,695,360 KV bytes across 12 layers and 118,069,304 bytes of running
state. Its two loan/refill operations match all 1,124 payloads and all three
decode requests match the expected resident map. The process exits normally
with quiet validation, closed ownership, unchanged model assets and no
fresh Xe fault. Both native extents now qualify the frozen map for the
clean generation comparison.

The first speed pair closes under the same retained executable and global
loader. Both 96K request hashes match, the dedicated prefill dispatch and
expert upload counts match, and both processes close without a fresh Xe
fault. The raw output lengths change with the placement.

| Workload | Output, stock → blend | Decode tok/s, stock → blend | CPU jobs/output, stock → blend | Client TTFT s, stock → blend |
| --- | ---: | ---: | ---: | ---: |
| 96,204-token code | 3,195 → 3,591 | 16.212 → 17.262 | 354.84 → 295.48 | 103.836 → 103.633 |
| 96,274-token prose | 8,618 → 8,212 | 11.601 → 11.835 | 398.11 → 361.11 | 92.749 → 92.871 |

Both cases show fewer CPU jobs per emitted token. The exact draft totals
and first/subsequent prompt timings remain in the speed result. Two read-only
research scopes overlapped part of the candidate process, each capped at
30% of one logical CPU. They accounted for 26.2 and 27.3 CPU seconds before
closure.

The reverse-order second pair closes at `44bb81e4b` with the same two
requests and native executable, normal exit and no fresh Xe faults. Code
decode is 16.298 → 17.498 tok/s (+7.36%), and prose is 11.637 → 11.933
tok/s (+2.55%), reporting stock before blend. Prefill is 938.743 → 942.207
tok/s for code and 1037.003 → 1038.282 for prose. Client TTFT is 103.816 →
103.437 and 93.027 → 92.910 seconds. Stock code emits 2,997 tokens in this
process, versus 3,195 in the first pair, while both candidate processes
emit 3,591.

The third pair and frozen decision are recorded at `5e75c7dbe`. All six
processes exit normally without fresh Xe faults. The two exact request bodies
and the prefill dispatch match in each pair. Decode improves on every pair.
Across pairs, the geometric mean gain is 7.13% for code and 2.64% for prose.
Stock and blend decode medians are 16.298 and 17.498 tok/s for code and 11.616
and 11.933 tok/s for prose. The highest individual code sample is 17.583 tok/s
with 3,591 emitted tokens and 2,389 of 2,484 drafts accepted. Median CPU
expert jobs per emitted token fall from 354.84 to 295.48 and from 398.11 to
361.11, respectively. The paired median prompt-time ratios are 0.996324 for
code and 1.000694 for prose, so the prefill preservation gate fails. Keep
stock placement. To revisit the decode gain, isolate cache selection from
prefill admission and handoff, then measure under a new matched plan.

The two admitted profile files at `5e75c7dbe` have the same first 132
experts and the same aligned size in each slot. Comparing their verified
packed manifest gives 979 changed slots, all in the 1,124-slot loan tail.
The stock and blended tail each refill 2,321,664,000 payload bytes.
Restoring stock first and then installing the blend would issue another
2,010,419,200 H2D bytes. The measured bulk-copy reference puts that extra
copy at about 0.31–0.32 s before copy scheduling or queue delays.

In the serving generator, the same refill routine also runs before a
prefill window and when a larger scratch loan replaces a smaller one.
Only the final successful prefill handoff can select the alternate tail.
Earlier window refills and cancellation still need a valid map for the
ordinary request flow. Check the actual loan covers every changed slot and
that both source and destination sizes match before issuing any alternate
copy. The cache maintains both an internal CPU residency table and the
published host/device table. The current adaptive replacement requires the
cache to be ready. A refill-specific publication must complete under the
checked lease, before admitting decode readers, and stop on partial failure.
Charge final refill and draft activation to prompt time as the existing
server does. Bind intermediate window work and repeated request paths to
their actual residency before making a phase-separation claim.

The default-off `STRATA_SYCL_DFLASH2_DECODE_REFILL_PROFILE` path takes a
STRP file and starts with the normal global profile. Startup validates every
admitted slot, checks equal packed payload size at that slot, and retains
source pointers from the immutable RAM arena. At a successful final handoff,
the existing loan must cover the entire changed range. A refill under the
lease validates both residency maps, submits and drains the replacement
payloads in the existing loan's destination order, publishes the device
table and commits the CPU table before
reopening decode admission. Early window refills, cancellation and a
partial loan preserve the current valid placement. The standalone physical
context diagnostic exercises the full fill separately from the serving
request's output reservation. After a successful switch,
subsequent short requests can use the now-installed decode layout.

The [isolated build](../bench/dflash2_decode_ranked_refill_build.py) applies
the six-file change through `ffc455c22` to the retained primary DFlash2 recipe.
The generator projection excludes the separate memory observer. Under the
existing cache-loan audit switch, the generator checks the CPU residency map
and reads the GPU map at each prefill segment and after final refill. It emits
the request, phase, span, masked-slot boundary and map digest. Keep this
readback off for clean timing. Rebuild all affected objects and both same-build
fixtures. The fixture runner at `a57499a15`
checks the host lease fixture and runs the bounded SYCL ranked-refill cases
with parameter validation. The device fixture covers
wrong extent, changed prefix, missing or aliased slots, reader exclusion,
payload and both tables, and fail-stop after an injected copy or publication
failure. The standalone host precheck uses the three fixture/lease source
files from `4757120ec` with `/usr/bin/x86_64-linux-gnu-g++-13 -std=c++20
-O2 -Wall -Wextra -Werror -pthread -Iinclude tests/core/prefill_scratch_test.cpp`
in a private source tree. The compiler SHA-256 is
`1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769`.
All 40 cases pass, including rejected and repeated check admission on the
same generation. The standalone result checks 267 system headers and uses
no SYCL runtime. The temporary executable is retired. For the separate
native syntax precheck, use all six compile database entries for the
changed generator, cache implementation and host/device fixtures. Materialize
the frozen projection in a private tree. Keep the retained compiler,
options, working directory and environment. Redirect source paths to the
projection and replace `-o … -c` with `-fsyntax-only -MD -MF …`. Validate
actual compiler dependency hashes against the committed project and retained
external inputs. This check creates no native executable.
The first syntax invocation stopped at the compiler's unused
`-fsycl-id-queries-range=size_t` message. Keep the original flag. Accept only
that exact driver warning, a zero exit and a verified dependency list on
the retry. The retry on `da6ff1b5d` passes all six entries with one instance of that
warning. The two generator variants check 809 and 810 inputs, the two
ExpertCache variants check 660 each, and the host and device fixtures
check 272 and 735 inputs. The updated projection matches the five unchanged
files. The [repeat check](../bench/dflash2_candidate_sycl_syntax.py) at
`a17905e15` passes both generator units for `ffc455c22`, with 809 and 810
verified inputs and no compiler diagnostic. The full link and device runs
remain pending.
The live RESTORE runner at `a57499a15`
uses the previously checked request bodies on the new build. Compare stock
and final ranked refill at each parked extent, 32,769 and 253,919 tokens.
In each process check a fresh long prefill, an unrelated 33-token request
and the restored prefix with an 8,209-token suffix. Bind the masked prefix,
the full map used for the short prefill, and the final payload and map in
order. The near-capacity request reaches 262,128 tokens. The standalone
context runner first checks
bounds and a 16,385-token handoff, then requires consuming the final slot of
the 262,144-token context with zero output.

The final live case sends native `STOP` after the masked expert map and
the first QSA KV-stage marker on a 32,769-token request. A cancelled read
must emit zero tokens, republish the same map and payload, and admit a fresh
33-token request through the decode cache. Repeat after a successful prefill
has installed the blended placement. Only the successful long request may
publish the alternate tail. Keep the native case and the physical full fill
as separate qualification results.

The offline check at `0715d2d3c` uses the retained primary build metadata
to reproduce the four live request plans and the two standalone requests.
The phase reader accepts four synthetic traces and rejects sixteen with
changed map digests, failed payload checks, premature or extra publications,
or a missing final handoff. The masked, stock and blended digests are
`9377e60e8ad8ea63`, `9e0b070ebe90ab9b` and `68eea56314d4990b`.
The host controls at
`2c8df018b` pass five tests. An in-process native-protocol peer confirms
both STOP/recovery sequences, mask rejection and refusal to count a late
STOP as prefill cancellation. The timing reader also refuses missing,
early, late and wrong-size refill events.
The isolated build is configured at `d15fe1cb6`. Its recorded inputs
cover 1,327 project files, the six ranked-refill overlays on `968e71988`,
the two original profile files and four generated build recipe files.
The full build at `edabb557c` closes with 141 fresh objects from 136
translation units and 13 executable identities. The same-build host
fixture passes all 40 lease cases at `7a8e4195d`.
The first device fixture exits zero and verifies the replacement payloads
and both maps. The runner stops because full UR tracing exposes the
qualified preload's deliberate null-platform query before main. It reports
`UR_RESULT_ERROR_INVALID_NULL_HANDLE` once at that query. The validation
logger is empty and no fresh Xe fault occurs. Preserve this failed fixture
cohort and use a fresh plan for the corrected trace check.
The revised runner at `78fdc2e3d` reuses the shared strict trace reader.
It accepts the original complete trace and rejects six mutations, including
a duplicate probe and an identical null-handle result after startup.
The fresh four-process cohort at `c6839420b` uses the same native executable.
It passes all 40 host cases, replacement payload and CPU/GPU map checks,
and both expected failure exits with decode admission blocked. All three
device validation logs are empty and the processes close with no fresh
Xe fault. The live RESTORE plan pins this successful fixture receipt.
The stock and ranked 32,769-token RESTORE cases both pass on this build.
Each parks the long state, serves an unrelated 33-token input, and reads
8,209 fresh tokens after restoring, reaching 40,978 prompt tokens. Their
batched segments use the same masked map and report the same work. The
ranked arm publishes 2,321,664,000 bytes at both long handoffs. Its 33-token
request uses the installed decode map without another refill. Each case
verifies its own saved KV and running state, whose digests differ across
the two placements. The first verifier window processes the last prompt
token after handoff under the selected decode layout. All three one-token
outputs match across arms. Both owners close with quiet validation and
no fresh Xe fault.

Both near-capacity cases restore 253,919 tokens from RAM and read 8,209
fresh tokens to reach 262,128. Each verifies 3,607,695,360 bytes of target
KV across 12 layers and 118,069,304 bytes of running state at the parked
extent. The two long segments match in prefill attention and KV-stage work.
The ranked arm verifies both 2,321,664,000-byte payloads and the published
ranked map. The intervening short request uses the installed map without
refill. All three one-token outputs agree. Both owners close normally with
empty validation logs and no fresh Xe faults.

The cancellation case sends STOP after the first QSA KV-stage marker
with stock and then ranked placement installed. Both requests return zero
output and restore the original map and payload. Each is followed by a fresh
33-token request that completes on the preserved map. The intervening
successful 32,769-token read is the only request to publish the ranked
tail. The owner closes normally with quiet validation, no fresh Xe fault
and all 15 model asset hashes checked. This closes live RESTORE and
cancellation qualification.

The same-build standalone diagnostic passes the 16,385-token check with
16,384 batched rows on the GPU and four expected output IDs. Its full
case consumes all 262,144 context tokens. It batches 262,143 rows, with
122,880 in VRAM and 139,263 in host RAM. It verifies the final physical
slot, emits zero output tokens, exits normally and records no fresh Xe
fault. The diagnostic uses the stock profile without a serving cache loan.
The live tests above qualify the separate ranked handoff.

The timing plan at `796e418c1` consumes the committed validation results.
The two closed validation captures were retired after matching the terminal
receipts to `3d260ae32` and `d74bc9428`. Both original `record.json` files
remain byte-identical. Inventory hashes cover sorted state-relative paths,
byte counts and file hashes.

| Scope under tuning state | Files | Bytes | Inventory SHA-256 |
| --- | ---: | ---: | --- |
| `dflash2-decode-ranked-refill-restore-v1` | 290 | 62,771,581 | `52a4ba3e7761a461ecf07072d57627db1d2bb52ecfc85d219b929e96a84583bd` |
| `dflash2-decode-ranked-refill-context-v1` | 119 | 26,523,244 | `6c3f079a7f625edc8e7b8e801160aa1ac46901532f51e3b0fc0edcb18b160ecb` |

Retired the closed device fixture traces after the first live pair and
verified both original `record.json` files are unchanged. The failed V1
evidence at `1afe5f82e` remains
distinct from the closed V2 proof at `c6839420b`. Inventory hashes cover
sorted state-relative paths, byte counts and file SHA-256 values.

| Scope under tuning state | Files | Bytes | Inventory SHA-256 |
| --- | ---: | ---: | --- |
| `dflash2-decode-ranked-refill-checks-v1` | 7 | 249,123 | `e42e3a390d09f5e92a8db995b84a9cdf7b41ed878c0d442bc1e351cb4e2e5719` |
| `dflash2-decode-ranked-refill-checks-v2` | 15 | 496,278 | `971f8af45108f2f488e978c4843fd7d1a9a85897f27b9681a948ec2ebef2b455` |

The primary prefill executable supplies the ranked-refill baseline. The
pending route-census comparison also checks that executable and the
independent `96832af6f` decode build. Keep the two checked phase references
and the ranked-refill candidate through this serialized comparison. Review
the added generation at the first failed qualification gate or the closing
timing decision, before starting another engine build.

After the native gates, the timing runner at `a57499a15`
uses one binary and the same starting profile for three matched independent
process pairs. Enable only final ranked refill in the candidate. Confirm
both actual long handoffs and the 2,321,664,000-byte payload publication,
and check that the short warmups leave the startup map in place. Keep the
first-use and subsequent PP, client TTFT and TG measurements separate.
Require both workloads to keep their actual prefill work and pass the phase
preservation gates. This prototype is original fork code. The speed
controller requires the closed live RESTORE and physical-context receipts
from this executable's build before launch.
The frontend checks pass at `af9c68a6a`. Run the fixed stock/refill,
refill/stock and stock/refill process pairs on that plan.

All six independent processes exit normally with quiet validation and no
fresh Xe faults. Paired requests match in fresh input and actual prefill
work. In each candidate process, the two long handoffs each publish
2,321,664,000 bytes. The short warmups use the stock map.

| Workload and pair | Generated tokens, stock → refill | Prefill tok/s, stock → refill | TTFT s, stock → refill | Decode tok/s, stock → refill |
| --- | ---: | ---: | ---: | ---: |
| Code 1 | 3,195 → 3,591 | 932.545 → 932.471 | 104.545 → 104.549 | 15.604 → 16.833 |
| Code 2, reverse order | 3,195 → 3,591 | 933.658 → 930.479 | 104.424 → 104.769 | 15.535 → 16.789 |
| Code 3 | 3,195 → 3,591 | 930.097 → 932.845 | 104.828 → 104.516 | 15.621 → 16.616 |
| Prose 1 | 9,670 → 8,212 | 1,034.828 → 1,028.080 | 93.223 → 93.838 | 11.218 → 11.244 |
| Prose 2, reverse order | 8,743 → 8,212 | 1,024.013 → 1,036.069 | 94.210 → 93.116 | 11.184 → 11.500 |
| Prose 3 | 8,618 → 8,212 | 1,033.543 → 1,033.014 | 93.342 → 93.389 | 11.157 → 11.427 |

Decode improves in all three pairs. The paired geometric gains are 7.437%
for code and 1.819% for prose. Median CPU expert misses per output fall
from 354.84 to 295.48 for code and 396.57 to 361.11 for prose. Each layout
repeats its code output and draft counts in all three runs. The candidate's
prose outputs and draft counts also repeat, while the stock runs record
three distinct output hashes and the lengths above.

The median paired prompt-time increases are 0.00785% for code and
0.05121% for prose. The median paired TTFT increases are 0.00383% and
0.05035%, respectively. The preregistered preservation gates both fail.
Keep the stock path. The measured stock TTFT spans 0.404 s for code and
0.987 s for prose. This comparison leaves the cause of the TTFT spread
unresolved. The compact result retains all samples and the verdict under
`dflash2_decode_ranked_refill_speed_v1`.

The closed comparison at `a57499a15` is reachable on `feature/sycl`.
The recovery audit matched 1,327 original source files, 520 committed
build-control inputs, the compiler and link inputs, 141 fresh objects and
13 executable identities. All six terminal process receipts match their
committed outcomes. Retired the descendants below and verified both
original `record.json` receipts and all 23 retained phase-reference
executable identities. The inventory hashes include sorted state-relative
paths, byte counts and file hashes. Removed the failed opt-in, its native
fixture additions and the closed model runners. Recover them from
`a57499a15`. The shared build-input reader remains in use by the smaller-KV
trial.

| Scope under tuning state | Files | Bytes | Inventory SHA-256 |
| --- | ---: | ---: | --- |
| `dflash2-decode-ranked-refill-build-v1` | 1,527 | 248,016,088 | `65c426b434623d3344bc2f531d7d8dca3b7c49ee94941030cae58219db5c336a` |
| `dflash2-decode-ranked-refill-speed-v1` | 436 | 90,756,554 | `693fdc4db0b6f43491f8a74c46207ea169db2c5aed808025d9c47bc078fdbb4a` |

In the first pair, each completed verification window commits one token
plus its accepted drafts. The last window can emit up to three fewer
tokens at spec4. The verifier window count is between emitted minus
accepted and that count plus three. Dividing the measured decode time and
output count by those bounds, the refill arm's average window takes
11.15–11.63% less time for code and 3.98–4.08% less for prose. Output per
window also falls 4.16–4.67% and 3.75–3.86%, respectively. The lower output
per window offsets most of the prose timing gain in this pair.

The first two closed blend processes also bound the benefit of draft yield
at the current decode time. Let `e` be emitted tokens and `a` and `d`
be accepted and offered drafts. In these completed spec4 requests, the
number of verifier windows is at most `e - a + 3`, allowing up to three
accepted positions after the last emitted token. Giving every existing
offer acceptance would therefore yield at most `e - a + 3 + d` tokens for
the recorded decode interval.

| Workload | All existing offers accepted at fixed time | Four emitted tokens per window at fixed time | Observed window time |
| --- | ---: | ---: | ---: |
| Code | 17.73–17.98 tok/s | 23.17–23.49 tok/s | 170.31–173.07 ms |
| Prose | 13.78–13.89 tok/s | 29.16–29.40 tok/s | 136.06–137.26 ms |

The second column of rates additionally assumes every existing window can
emit four tokens at unchanged cost. This gives a 57.14 ms whole-window
budget for 70 tok/s even with perfect yield at width four. Improving draft
yield can complement reducing verifier service time. A wider verifier or
changed acceptance affects its output trajectory, expert work, GPU work and
duration, so evaluate those changes in a matched model trial.

Charging a full packed expert upload for each remaining CPU job would
require at least 31.24 GB/s for code and 38.18 GB/s for prose at 70 tok/s.
This lower bound uses the measured candidate's CPU jobs per output and the
smallest packed expert in the model, 1,510,400 bytes. The separate large
host-USM copy reference is 6.29–6.49 GB/s. At that copy rate, the entire
budget could carry about 59.5–61.3 minimum-size jobs per output. The
measured candidate has 295.48 and 361.11 remaining CPU jobs per output.
Prioritize removing misses through reuse or reducing work per miss. Revisit
streaming as a bounded subset with measured amortized bytes, GPU execution
and transfer overlap. These are estimates for one full upload per miss.

For selective transfer, reuse the two qualified CPU leaf logs at
`6bbd3e3e1`. Group all 1,449,976 jobs by their actual token count,
multiplying each format pair by its verified packed blob size. The table
shows CPU work per job and the time to copy its mean blob at the measured
bulk reference of 6.291 GB/s.

| Tokens per CPU job | Jobs | CPU leaf work per job (μs) | Full-blob copy at reference rate (μs) |
| ---: | ---: | ---: | ---: |
| 1 | 1,154,781 | 422.57–422.64 | 324.44 |
| 2 | 237,631 | 502.37–503.64 | 322.43 |
| 3 | 47,629 | 581.47–583.25 | 324.04 |
| 4 | 9,935 | 679.12–682.82 | 324.90 |

In the four-token bin the ideal per-job share with six busy cores is
113.19–113.80 μs. Matching it would require hiding about 65% of the copy
even before GPU execution. This comparison uses summed CPU work and bulk
copy time. Measure the real CPU tail and exposed copy interval on the same
layer slice before moving a missed expert. Keep resident reuse and early
publication ahead of an unconditional upload rule.

Apex places high-token-load misses using the completion time of the slower
branch. Apto claims work as each processor advances and tries backup
execution at the final expert. For the captured T4 layer slices, first
measure exact static assignment with a bounded copy budget and one result
per expert. Measure reuse and publication separately so a gain can be
attributed to the right mechanism. A dynamic claim or backup race needs
an observed residual tail after that comparison. [Apex](https://arxiv.org/html/2610.23006v1)
· [Apto](https://arxiv.org/html/2611.01859v1).

The [residency bound replay](../bench/dflash2_decode_residency_oracle.py)
uses the retained prose trace from `b28a9372b`. First reproduce its native
job, routed-row, format-width and logical byte counts. Compare the actual
stock placement and a first-half-trained rank with two idealized alternatives
on the held-out half. One profile sees the entire held-out route in advance.
The other can replace any unprotected slot every 256 windows at zero cost.
Both maintain the original 1,256 slots, the first 132 experts and each
remaining slot size. Also compute a looser static bound at the original
byte budget and at 256, 512 and 768 MiB above it by allowing fractional
experts and selecting by benefit per byte. Prove that loosening the geometry
cannot worsen the static bound and that extra budget cannot increase misses.
Use any gap above the trained rank to decide whether improved ranking or
shorter adaptation is worth testing. The replay keeps target routes fixed and
does not claim transfer feasibility, numerical quality or decode throughput.

The CPU replay at source `44bb81e4b` closes normally and reproduces all
4,668,262 CPU jobs in the original capture. The trace SHA-256 is
`08dce3920dea2de8752b16cbbbaddc07e9486515dfde39264c92fa116b27926f`.
Of 6,485 verifier windows, 3,242 train the simple rank and the remaining
3,243 form the evaluation. That rank is specific to this capture.

| Held-out placement | CPU jobs | Logical CPU weight visits (bytes) |
| --- | ---: | ---: |
| Observed stock | 2,581,636 | 5,266,123,776,000 |
| First-half-trained rank | 1,967,341 | 3,999,896,140,800 |
| One ideal profile within the current geometry | 1,775,920 | 3,601,085,465,600 |
| Ideal free replacement in each of 13 periods | 1,509,439 | 3,051,459,174,400 |

The ideal static profile removes at most 9.73% more jobs than the trained
rank. Free replacement every 256 windows removes 23.28% more. Under the
looser static budget, the remaining job lower bounds are 1,757,758,
1,700,255, 1,646,221 and 1,595,032 at base, +256, +512 and +768 MiB.
The corresponding logical byte bounds are 3,589,259,827,200,
3,471,425,587,200, 3,361,804,380,160 and 3,258,911,867,904 bytes.
Pursue lower CPU service per miss and lower GPU verifier cost alongside
placement. The ideal adaptive result assumes unrestricted swaps, zero transfer
cost and future routes already known. Use it to judge how much space remains
above realistic replacement policies, with their actual transfer and restore
cost charged in a separate comparison.

Retired `dflash2-blend-independent-capture-v3` under the tuning state
directory after verifying its absence. The exact scoring source and compact closed
results remain in Git at `53f824443` and `a7c068890`. Both captured traces match
their committed lengths and hashes, and both process owners are closed. The
native RESTORE and generation plans consume the committed result, the original
training evidence and the generated candidate profile. Their inputs do not
include this capture directory. The verified inventory has 151 files and
88,069,211 bytes with SHA-256
`3cbef7b879a2d12d1b72e5793192e0cf4e3e83f35ff3a5ae5732b3fca3143719`
over the compact JSON list of sorted paths relative to that directory, file
sizes and file hashes.

The [adaptive cost replay](../bench/dflash2_decode_adapt_cost_replay.py) extends
that same 256-window cut. Keep the protected first 132 slots and the same
96-swap and 197,913,600-byte caps. Compare early distinct jobs, routed rows
and format/width-weighted leaf cost. Calibrate each job's Gate/Up, quantization
and Down weights from the two closed code-request CPU leaf observations at
`6bbd3e3e1`. The replay reuses the prose suffix and verifies the original
routed-row result. It keeps every method's cumulative work and copy-only
crossover in output-token coordinates. For that sensitivity estimate divide
modeled thread CPU time by six and charge both upload and restoration at
the lower large-copy reference. The estimate excludes GPU cost, publication
and memory contention and does not establish real break-even latency.

At `4dc04715f`, the replay reproduces the previous routed-row result and all
4,668,262 original CPU jobs. The 256-window cut follows 408 emitted tokens.

| Rank input | Swaps | Requested H2D bytes per upload or restore | Suffix CPU jobs removed | Modeled leaf CPU seconds removed | Copy-only crossover after swap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Distinct jobs | 96 | 197,811,200 | 131,382 | 61.537 | 33 outputs |
| Routed rows | 96 | 197,913,600 | 147,795 | 69.456 | 29 outputs |
| Format/width leaf time | 92 | 197,248,000 | 131,773 | 64.247 | 33 outputs |

Routed rows also lead at the 99-, 394- and 1,589-output checks, removing
2,449, 9,915 and 36,846 CPU jobs. On the complete held-out suffix it removes
3.274% of jobs and 3.456% of modeled leaf work, with 303,143,424,000 logical
CPU weight bytes avoided. The 29-output estimate charges roughly 63 ms
for two requested full payloads at the reference rate. It is a copy-only
sensitivity calculation. Retain the simpler routed-row score for the next
bounded trial. A new model rate requires an independently reserved workload
and direct timing of expert publication, GPU service, upload and restoration.

The request-local counter fixture at `62e7cc2fc` summarizes every routed
row in those first 256 verification windows, including computed draft rows.
They contain 463 verified input rows, 4,630 expert entries per layer and
222,240 entries across 48 layers. A second raw-route unpack matches the
98,304-byte layer-major little-endian uint32 count array's SHA-256
`746a3da65686fbfde34d06df14d1e377fd7211bc8f0921a52d32cd0b58215d05`.
Compare this exact map in the captured request and check fresh per-request
counts across prompt, cancellation and RESTORE boundaries before enabling
any placement change.

The retained decode source `96832af6f` allocates the existing routed-row
counter only when both adaptive arguments are positive. Its serve loop
submits adaptive work only when the process-wide window count reaches a
multiple of `--adapt-every`. For the counting control, hold this interval
at 1,000,000 and compare `--adapt-swaps 0` with `--adapt-swaps 1` on the
file-backed expert route. Bound the sum of requested outputs across the
process below the interval, and require zero adaptive submissions or
commits and identical native work. Keep the same startup trace flag in
both arms to measure the existing counting path across the whole process.
A later 256-window implementation must explicitly start after
prefill and stop after the selected decode window.

The first count-only controller at `395aee434` stopped before model launch.
It compared the checked generated program against the unprojected
`96832af6f` Git tree. The retained build recipe applies the DFlash2 tap
overlay to that translation unit. The corrected controller at `7aaa9f423`
requires the reconstructed source, build receipt and actual compiler input
to agree. The unchanged executable remains the candidate for the
configuration-only comparison. Preparation passes at `ce94cc031` with the
no-refill plan and effective arguments frozen in the verification worktree.
Benchmark and server host admission pass at `ebcdbcc0b`. The first
counting-off process at `c4ddf9dde` closes normally with no fresh Xe fault.
The 33,217-input code request generates 4,111 tokens in 247.0915 seconds
of native decode time, or 16.638 tok/s, with 2,767 of 2,863 drafts accepted.
Prefill takes 48.3305 seconds and client TTFT is 49.663 seconds. The on arm
at `02b17d4ce` closes normally without a fresh Xe fault and matches the
workload, output, draft counts and recorded routes. It measures 16.764 tok/s
over 245.2332 seconds of native decode with zero adaptive refills. This
pair has 0.752% lower decode time with the counter on. The reverse pair's on process
at `b43bfaf88` repeats the same work and output at 16.767 tok/s over
245.1802 seconds. The final off process at `c6c2aa0ab` measures 16.527 tok/s
over 248.7465 seconds. All four match the original work and output, complete
owner and asset verification, and report zero refills and no fresh Xe faults.
The on/off latency ratios are 0.99248 and 0.98566, both inside the planned
factor of 1.05. This qualifies the existing counter's cost envelope. The
next comparison isolates a request-local count prefix and checks that it
does not accumulate prefill or prior-request routes. The complete original
configuration and process records are retained in `c6c2aa0ab`.

Read-only enumeration on the Ryzen 5 5600X (family 19h, model 21h), kernel
`7.0.0-38-generic`, exposes `cpu`, `ibs_op` and `ibs_fetch` in
`/sys/bus/event_source/devices`. The data-fabric and memory-controller PMUs
`amd_df` and `amd_umc` are absent from that list. `perf_event_paranoid` is
4. On 2026-11-26 the unprivileged user-mode cycles and instructions probe
also reports no supported events. Leave IBS activation, IPC and physical
DRAM bytes unknown. Use the earlier matched worker replay for its measured
scan and expert scaling only.

The first caller-observed process at `7340714a9` matches the clean work and
emits 4,111 tokens at 16.649 tok/s. The 1,345 windows average 83.67 ms in the
CPU expert-pool wrapper, 82.26 ms in layer and final GPU completion intervals,
and 17.66 ms elsewhere. Native waits account for 46.36% of caller thread CPU
time. Replacing the sum of the first two intervals with their maximum at
fixed work gives a sensitivity estimate of 30.17 tok/s. Dependency limits
and additional overlap work affect what can actually be achieved.

For a fixed-work test of earlier CPU result publication, the original
`cd60b5041` replay has four T4 layer slices with committed route-derived
expert job lists. Splitting the four tokens into two consecutive pairs
gives the following job dependencies.

| Window / layer | First pair only | Both pairs | Second pair only | Resident routed rows |
| --- | ---: | ---: | ---: | ---: |
| 2769 / 0 | 16 | 3 | 17 | 1 |
| 2769 / 17 | 18 | 2 | 15 | 0 |
| 5799 / 0 | 15 | 3 | 17 | 1 |
| 5799 / 17 | 15 | 2 | 14 | 2 |

Preserve the 137 original CPU jobs and publish each token pair only after
its complete set of contributions is ready. Independently running the two
pairs would duplicate ten jobs and charge 22,374,400 extra bytes against
308,249,600 bytes of logical packed work unless the shared jobs are kept
grouped. Use these slices with the existing mapped-combine parity fixture
to validate the barrier control. Their data establish routing dependencies.
Actual GPU completion and the benefit of early publication need measurement.

The pinned replay at `a213dd223` checks all 47,280 width-four layer calls
from the 33,287-input prose capture at `b28a9372b`. It reconciles all
4,668,262 CPU jobs and 5,982,938 CPU routed rows across verifier widths.
The first-pair dependencies form the first-seen CPU job prefix in every
width-four layer call.

| Width-four dependency | CPU jobs | Routed CPU rows | Logical packed bytes |
| --- | ---: | ---: | ---: |
| First pair only | 480,669 | 538,531 | 978,405,990,400 |
| Both pairs | 250,854 | 684,476 | 515,667,072,000 |
| Second pair only | 477,450 | 537,283 | 970,470,630,400 |

The first publication depends on 60.5% of jobs in aggregate. Second-only
work exists in 47,241 calls. The first pair needs at most half the CPU jobs
in 5,030 calls, and every call has at least one first-pair CPU dependency.
Use these counts to keep shared work single-copy and to select typical
and short-prefix cases for the integrated handoff comparison. Measure the
time from first-pair readiness through GPU completion alongside the whole
layer. Logical packed bytes and job counts describe the captured dependency
graph. The achievable overlap and its cost remain open.
[Publication census](../bench/dflash2_decode_combine_hits.py).

The [combine fixture](../tests/sycl/decode_moe_combine.cpp) exposes a
correctness-only `validate --check-publication` mode. A stock four-token
graph and two stock two-token graphs use the same weighted-reduction order,
with rebased hit metadata for each half. The two halves use distinct Host-USM
allocations. In the staggered check, complete the first graph while the second
half still contains poison and verify that the second output stays untouched.
In the queued check, submit the first graph, write the separate second
allocation, and submit the second graph on the same in-order stream. Its
completion retires both publications before reuse. The Arc check at
`d7c710d12` passes the stock, staggered and queued cases for all nine
resident-row counts and both input seeds. Outputs match the FP32 oracle
bitwise, including the output guards and untouched half in the staggered
case. The expected Level Zero providers load, parameter validation is
quiet and the process closes without a fresh Xe fault.
The [component controller](../bench/dflash2_decode_result_copy.py) uses
`--publication prepare`, `--publication build` and `--publication validate`,
committing each transition. This qualifies the buffer and queue contract with
generated inputs. A real CPU-worker prototype also needs independent result
readiness, fixed routed work, and timed stock, two-half-wait and early-enqueue
comparisons before claiming an overlap benefit.
[SYCL Host USM](https://github.khronos.org/SYCL_Reference/iface/usm_basic_concept.html).

The [fixed expert replay](../bench/sycl/cpu_roofline/fixed_replay.py) exposes
`--publication prepare`, `--publication build` and `--publication check`.
It reuses the original eight captured batches and qualifies the four T4 layers
above. Five pinned workers retain the full one-claim-per-expert arithmetic.
The reserved caller observes the original first-pair prefix via one completion
counter on its own cache line. Each worker increments it after writing all of
that expert's rows, with acquire-release ordering. Check each captured prefix,
an empty prefix, a full prefix and a repeated captured prefix in both
early and after-all modes. The callback compares only completed CPU outputs
with the serial-checked reference. The pool retires all claims before buffer
reuse, including when the early group has zero dependencies. This is a
CPU-only readiness check for the isolated build. The Arc fixture checks
host-to-device visibility and ordered completion with seeded host inputs.
[C++ atomic ordering](https://eel.is/c++draft/atomics.order).

The CPU check at `62d23a297` passes all 32 bitwise cases across the four
captured slices with 137 expert jobs and 308,249,600 logical packed bytes.
The pinned five workers execute the CPU jobs while the reserved caller
observes readiness. The process exits normally. Coupling the actual CPU
workers and next-layer GPU graph remains pending.
[CPU check record](../bench/results/2026-10-15-pr6-recovery-audit/cpu-fixed-expert-publication-current.json).

The selected verifier at `968e71988` can split a decode window into two
groups for `T >= 2`. In its SYCL path the host synchronizes the current
segment, calls the expert pool, retires copies and then submits the next
segment. Plan publication can launch that group's shared and resident work
while the CPU callback runs. `--spec-split` changes the 48 host completion
steps to 96 and leaves next-segment submission after each group callback.
On the fixed width-four capture, visiting each shared CPU expert in both
callbacks would raise the job count from 1,208,973 to 1,459,827 and logical
packed work by 515,667,072,000 bytes, or 20.92%. The actual CPU traffic and
executed routes can vary when switching the verifier.

Use one four-token dispatch and publish the two halves independently
while keeping the shared experts in the same jobs. Timing needs to include
the first half's combine and next-layer pre-mixer. The standalone combine
covers only about 0.15–0.18 ms for the measured width-four shapes, while
the observed 82.26 ms of whole-window GPU completion averages 1.71 ms per
layer over 48 layers. Keep stock segment and two-half-after-all controls
to charge extra dispatch, event and synchronization costs. The measured
production control uses five expert workers with caller participation.
The eight-batch stock replay at `cd60b5041` measures that same worker
topology both ways. Caller participation gives 17.598, 17.592 and 17.685
ms per pass. Reserving the caller gives 19.750, 19.405 and 20.051 ms,
a paired penalty of 10.31–13.38%. Both arms retain 201 CPU jobs,
226 routed rows and 452,275,200 logical packed bytes per pass. Keep this
comparison in the integrated trial to price the opportunity cost of
submitting the first graph early. Keep the existing shared/resident
overlap enabled in all arms.
[Verifier](../src/core/verify.cpp).

The production pre-mixer ends by publishing its routing in mapped host
buffers. The current CPU dispatcher still reads those IDs after pool
retirement to clear the job map and record the route census. The server
trace reads them again after dispatch returns. In the integrated trial,
complete the first half's next-layer pre-mixer before waiting for the
second half and defer its mapped publication until those consumers
retire, or use a separately owned destination. Verify the next-layer
IDs and the preceding layer's job map and trace as distinct payloads.
The current GDN replay loops from token zero in both half graphs and
suppresses earlier outputs in the second half. A width-four split
therefore executes two extra GDN recurrence steps per applicable layer.
The selected stock build also uses the width-four Q6 DPAS dense kernels.
The two-token pre-mixers use their own supported projection shapes. The
stock versus after-CPU comparison captures that loss of the four-column
kernel and weight reuse, and the after-CPU versus early comparison uses
the same two half graphs to isolate submission timing.
The existing unsplit segment observer excludes split windows. Count the
actual graph and recurrent work and time the two-half control through
its own completion events.
[CPU dispatcher](../src/core/expert_source.cpp) and
[server trace](../src/program/generate.cpp).

For the first coupled replay, keep one width-four CPU dispatch and one
resident plan per layer. Queue the first combine and first-half
next-layer math from the readiness callback, then queue the second
combine and next-layer math at pool retirement. Publish both halves'
next-layer IDs after the old layer's CPU dispatch and routing trace
return. Use disjoint host result allocations and preserve ordered
completion through the second graph. Compare with the same pair of
graphs submitted after retirement and with the stock four-token graph.
Charge capture/submit counts and the reduction in shared weight reuse.
Carry the decode-only selection with the captured graph family. The
legacy `groups_[T]` slot is shared by prompt and decode graph captures
at a given width, so a decode-only change in its value would also
change the prompt-side replay's expected geometry.
[Graph selection](../include/strata/core/verify_window_policy.hpp).

The model candidate exposes the three schedules via
`STRATA_SYCL_DECODE_T4_SCHEDULE=stock|after-cpu|early`. The stock setting
is the default. The opt-in arms require a full DFlash2 verifier, a single
local resident plan and no PCIe expert share. They retain one plan and
CPU dispatch per layer. Graphs read separate pinned host halves, with a
CPU copy from the original dispatcher output into each half. Include both
copies in the integrated comparison. The early callback runs synchronously
on the caller while the workers complete the second half. The second
graph publishes next-layer routing only after the old dispatch and trace
return. Prompt verification and other decode widths retain their original
graph families. `STRATA_SYCL_DECODE_T4_AUDIT=1` records the actual
by-expert readiness and early submission counts for each width-four
decode window. Request admission rejects a positive PCIe expert share
in these opt-in modes before changing state or starting GPU work.

The same-build shared-plan fixture compares the stock four-token
copy/add/reduce graph with two half kernels using the same resident IDs.
At `674402992` it passes the poisoned unpublished half and queued
handoff checks, with bitwise outputs and untouched guards at nine hit
counts and two input seeds. The same-build host graph-policy test also
passes. The expected Level Zero providers load, parameter validation is
quiet and both processes close without a fresh Xe fault. The traced
82-input/32-output model comparison checks the three schedules before
integrated timing. It pins zero PCIe share and sends the original
request body without overrides. Run sustained generation and full-context
checks after parity.
[T4 build controller](../bench/dflash2_decode_t4_build.py).

The model observation at `e36f293d5` closes all three independent
processes without a fresh Xe fault and with all model hashes verified.
All arms emit the same 32 token IDs, offer and accept the same 23
DFlash2 drafts and reproduce the five prompt tap digests. The stock arm
also matches the previously selected stock model's short-request state.
Both split arms enter the same seven width-four windows. Each has
23 compatible CPU pool layers out of 48. The deferred arm records
zero early submissions; the early arm records 161 in total.

The early and deferred arms match the full checked target-state digest
set. Against stock, the split geometry changes the GDN, QSA tail, pooled
index and KV digests. The matched control at `697559569` disables the
width-four dense DPAS route in both stock and deferred arms on that same
executable. All 32 output IDs, draft counts, tap digests, widths and
checked target-state digests match with DPAS off. Toggling DPAS changes
GDN, PLE, QSA tail, pooled index and KV digests within both arms.
Both processes exit normally with no fresh Xe fault. This resolves the
checked short-sample difference under matched arithmetic. Production
DPAS-on inference and sustained throughput still need separate checks.

The selective capture chooses the split from the CPU pool's supported-format
predicate. The model manifest has 14
IQ3_XXS/IQ4_NL and nine IQ3_S/IQ4_NL layers eligible for the current pool
contract. The other 25 steps retain the original four-column post and
next-layer pre-mixer. V3 at `2dac4cbf2` builds 141 fresh objects and 13
executables with zero uncovered inputs. Its host policy and Arc same-plan
numerical, buffer-guard and completion checks close at `460628c2a`.

The 82-input, 32-output model comparison closes at `4a36122ad`. With
dense DPAS disabled in all three arms, output IDs, active state digests,
prompt tap payloads, draft counts and width trajectories agree. Each of
seven width-four windows uses exactly 23 split steps; the early arm
submits 161 halves before full CPU retirement and the deferred arm submits
none early. All three processes exit normally with no fresh Xe fault.
The clocks in this observed short test are diagnostic only.

The same-executable production-DPAS comparison closes at `4c324029d`.
Stock reproduces its original short control and captures 81 non-head C4
dense DPAS calls across four shapes. Both split schedules retain 46 calls
across those shapes and match each other in output, active state, prompt
taps and width trajectory. Stock and split agree in output, prompt taps
and widths. Their GDN, QSA tail, pooled index and KV digests differ under
the different arithmetic paths. The two split schedules exercise all seven
width-four windows with 23 split steps. Early submits 161 times before
CPU retirement, while deferred submits none early. All three processes
exit normally with quiet validation and no fresh Xe fault.

The [long-code T4 controller](../bench/dflash2_decode_t4_speed.py)
compares stock, deferred and early in three counterbalanced blocks. It
pins the verified 96,204-input thinking-on body and uses the current
`bench.sh` server route with production DPAS. Each process has one sustained
generation after warm-up. The per-window T4 audit checks actual split
steps and charges its cost. Report logical prefill work and first-long
TTFT independently, and match early/deferred decode work before
attributing a time difference to that ordering. A positive speed screen
selects a separate clean comparison; clean performance, weak-draft and
agent checks, physical-context and RESTORE qualification remain open.

All nine processes of `dflash2_decode_t4_speed_v1` close normally with
quiet validation, no fresh Xe faults and all 15 model assets verified.
Each block matches checked prefill work across its three schedules.
Each sample reads 96,204 fresh tokens. Client TTFT spans 103.542–104.397 s.

| Block | Stock TG (tok/s) | Deferred TG (tok/s) | Early TG (tok/s) |
| --- | ---: | ---: | ---: |
| 1 | 16.139 | 14.902 | 15.788 |
| 2 | 16.194 | 14.917 | 15.794 |
| 3 | 16.313 | 14.844 | 15.798 |

The split schedules match output and reasoning hashes, 3,440 generated
tokens, 2,268 of 2,376 drafts accepted and recorded expert work in all
three blocks. Each exercises 23 split steps in each of 680 T4 windows.
Early publishes 15,640 halves and saves 12.795–13.997 seconds against
deferred. The paired geometric rate gain is 6.081%. Stock emits 3,195
tokens and accepts 2,173 of 2,248 drafts in all three repeats. Under the
final-window emission bound, early averages 185.32–185.92 ms per
verifier window and stock 191.08–193.70 ms. Early emits fewer tokens per
window and its geometric rate is 2.603% below stock. The predeclared speed
screen fails. Keep stock selected. Continue with the isolated arithmetic
check before considering another model trial.

The existing Q6 DPAS tile shares integer dot and per-column FP32
accumulation code across repeat counts. Check two canonical C2 launches
against the corresponding columns of one C4 launch at the four actual
dense shapes. Require bitwise output and guards with the original Q8
column pitch. Measure the two C2 launches against the current two-column
kernel pair and against C4 to price repeated weight traversal. If this
passes, scope C2 to the eligible split decode steps and recheck long
draft yield. This is a new candidate; restored yield and model speed
remain unmeasured.

The isolated component source is `88c34b4a5` with build and Arc checks at
`f910436e5` and the timing controller at `b79dd7ee3` on the local
`verify/t4-c2-dpas` worktree. It adds canonical
C2 to the existing Q6 fixture and uses two independent, one-node graph
submissions for split arms. The isolation control at `c96980b21` runs
each half with the other half's Q8 blocks zeroed. It requires C4 bits
for the active columns, an untouched canary in the inactive output and
unchanged inputs. The ten numerical cases include 20 isolated launches
and an unpadded output shape. The standalone component build and Arc
verification follow closure of the frozen 96K cohort. The controller recovers
the T4 build recipe and prior checked Arc safety environment. The six
timing processes alternate between C4 and the existing two-column pair,
with five eligible batches per arm and shape. Use the paired C4 times to
price the split and the existing pair to screen shapes for a follow-up.
The host reader accepts all six synthetic sample orders and rejects the
wrong graph count, short batch and C4 arithmetic mismatch. Compile and
runtime validation remain pending.

The controller at `00b3bde32` weights the four shape timing deltas using
the original same-build capture at `4c324029d`. In fixture order,
6144×2560, 2560×6144, 2560×10240 and 2560×12288 lose 19, 6, 7 and 3 C4
calls. At one canonical payload per call these 35 calls total 550,502,400
bytes. Keep the shape gates unchanged. The weighted deltas describe
isolated matrix replays using the short capture counts. Actual VRAM
traffic, long-request reuse and overlapped time must be measured separately.
[Original model result](../bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json).

The first build at `8bd19a9b9` compiles and links all three named targets.
Its build audit fails because that plan omitted the existing fixture
translation units needed by the shared Gemm and MoE class-definition
census. It has no runtime evidence. The revised plan includes the
complete primary fixture target set and the T4 policy and combine tests.
The revised build at `032690088` completes 141 fresh objects and 13
executables with zero uncovered inputs. The three-way model comparison
uses this single executable with each schedule selected before startup.

The original V1 result remains recoverable at `8bd19a9b9`. Its retirement
audit matches all 1,334 materialized project files to Git and verifies
127 objects, 1,470 compiled inputs, the generated header and 39 link
inputs against the original recipe. The closed V1 root has no next
consumer. Its build, source and logs account for 1,510 files and
147,318,485 bytes, excluding `record.json`, with inventory SHA-256
`b31f90d3e58815ac2646479212d28f788ab448d700c883ed147bdbec9eb6763f`.
The V2 comparison and matched DPAS diagnostic are now closed. Its rebuild
audit matches 1,334 original project files, 141 objects, 13 executables,
1,481 compiled inputs and 30 external link inputs against the source and
recipe at `032690088`. The V3 candidate uses the same component proof and
build target set. V2 has no further execution consumer. Its 1,534
retired files totaled 236,458,958 bytes with inventory SHA-256
`2ccf8cb8df3968f50857c40eac247978b1434b6799d1c0df95e1b673d6849e0f`,
excluding the retained `record.json`. The exact V2 build, source, cache,
temporary and ownership files were retired under the measurement lock after
verifying reachability on the pushed primary, closure and matching inventory.
The original receipt and measured DFlash2 fallback were reverified afterward.

For a copy-once control, the full CPU result slab at `T=4`, `K=10` and
`N=2560` is 409,600 bytes per layer, or 19,660,800 across 48 layers. The
6.292–6.485 GB/s large host-USM copy reference would move that payload in
3.03–3.13 ms if it sustained the same rate. At 3.057 outputs per window,
70 tok/s would request about 450 MB/s for those slabs. These calculations
cover useful result bytes. The mapped load path, small-copy throughput,
readiness waits and actual bus transactions need separate measurements.

The existing combine fixture adds a `--compare-copy` mode. Both graphs run
the same fused reduction. The control reads the mapped host rows. The
candidate first queues one 409,600-byte copy to device storage on the
ordered stream. Time the copy and synchronized graph together and keep the
existing scalar oracle, poisoned source and output-guard checks. This tests
the copy-once boundary before changing publication order. Source `c4bf3b970`
passes the SYCL fixture's C++20 `-fsyntax-only` check with oneAPI 2026.2
`icpx` and its verified CMake compiler arguments. The compiler command
database SHA-256 is `30db253a89e4ddb66374934fcec2f79d3df27ce63fb5150c83998184b9546e67`.
The component controller compiles one fixture source file and verifies the
three unchanged archives from the retained serving build against the actual
link inputs. Run traced validation first, then three independent timing
processes with alternating pairs at 4, 10 and 20 resident hits. Advance a
shape only when both launch-to-completion and publication-to-completion
medians improve in every process. The controller keeps the serving executable
unchanged. The fixture links the three verified archives and passes the traced
scalar-oracle, input-poison, invalid-list and guard checks on Arc. All three
timing processes close normally, with quiet validation and no fresh Xe fault.
The table reports candidate time divided by mapped-control time, using each
independent process's median from four alternating pairs of 512 replays.

| Resident hits per 40 rows | Launch through completion | CPU publication through completion | Component gate |
| --- | --- | --- | --- |
| 4 | 0.868 / 0.860 / 0.864 | 0.874 / 0.866 / 0.870 | Pass |
| 10 | 0.962 / 0.959 / 0.972 | 0.964 / 0.959 / 0.974 | Pass |
| 20 | 1.165 / 1.160 / 1.182 | 1.159 / 1.153 / 1.175 | Fail |

Copying 409,600 bytes on the ordered stream helps at the two tested lower hit
counts. The fixture baseline uses the opt-in fused mapped path. Count the
actual resident rows in eligible unsplit width-four verifier layers before
choosing shapes for comparison with the stock combine path. Check production
handoff dependencies and workspace cost in that comparison. Keep the mapped
arm for the tested 20-hit shape. The fixture times exclude variable token
widths and CPU expert work. Model throughput remains unmeasured.
[Original fixture](https://github.com/yqYo1/Strata/blob/4a3526351/tests/sycl/decode_moe_combine.cpp) ·
[Original controller](https://github.com/yqYo1/Strata/blob/4a3526351/bench/dflash2_decode_result_copy.py).

At `4a3526351`, the committed samples preserve all 72 timed batches. Verified
the source recipe, actual compiler and linker inputs, and executable before
retiring `dflash2-decode-result-copy-v1/{0,1,2,3,fixture-build}`. The root
receipt is unchanged. The 38 retired files occupy 12,751,098 bytes and hash
to `c5c71ca26f6af02ca27009b708c793fcdcccfb40f18ca22876c7805c639e2512`
with the sorted state-relative inventory format above.

The [resident-row census](../bench/dflash2_decode_combine_hits.py) replays the
closed 33,287-input, 11,783-output prose capture at `b28a9372b`. Verify the
captured executable, unsplit width-four setting, zero PCIe expert fraction,
zero adaptive swaps and original route/profile hashes before counting.
Reconcile CPU jobs and routed CPU rows across all widths with the captured
native totals. Count individual resident rows, including repeated expert IDs
at different token positions, once per layer of each observed width-four
window. Record only measured-shape coverage, broad bands and common shapes.
Use those counts to select a comparison against the stock combine. Rows
between the tested hit counts need their own timing evidence. The replay
covers one captured placement and workload and collects no new GPU timing.

The census at source `8bb39c78c` reconciles all 4,668,262 CPU jobs and
5,982,938 CPU routed rows. Of 6,485 windows, 985 use width four, giving
47,280 layer calls. The route payload remains pinned by SHA-256
`08dce3920dea2de8752b16cbbbaddc07e9486515dfde39264c92fa116b27926f`.

| GPU resident rows | Width-four layer calls | Share |
| --- | ---: | ---: |
| 0–3 | 33,266 | 70.360% |
| 4–10 | 12,597 | 26.643% |
| 11–19 | 1,372 | 2.902% |
| 20–40 | 45 | 0.095% |

The previous copy comparison advances the exact 4-hit and 10-hit shapes.
They cover 4,051 and 528 layer calls respectively, or 9.685% together. The
0, 1, 2, 3, 4, 5 and 10-hit shapes cover 40,594 calls (85.859%). Use these
seven shapes for matched component comparisons of the stock sequence, fused
mapped reduction and the same fused reduction after an ordered copy. Keep the
20-hit result as a measured regression boundary for copying. The census does
not establish coverage for other workloads or verifier widths.

The [stock combine comparison](../bench/dflash2_decode_result_copy.py) keeps
the prior production archives and rebuilds only the changed fixture source.
Validate all three arms with the same FP32 oracle, two CPU publications and
poisoned unused rows. Include 39 and 40 hits in correctness and invalid hit
lists for both fused arms. Under the pinned Arc settings, collect three
independent timing processes at the seven census shapes. Each process covers
all six arm orders with 512 completions per arm and order. Charge the 409,600
byte copy only to the copy-and-fuse arm. Compare the three pairs separately on
launch and CPU-publication completion time. A candidate advances at a hit
count only when it beats stock on both clocks in all three processes. The
copy-versus-mapped contrast measures the incremental copy effect. The fixture
and raw logs have one consumer through comparison closure.

All three arms pass the traced oracle and buffer checks. The three timing
processes each supply 126 valid batches and close with quiet validation and
no fresh Xe fault. Median time from CPU publication to completion is shown
per replay, aggregated across the three process medians. The last column is
the range of the three process-level ratios.

| Hits | Stock (µs) | Fused mapped (µs) | Copy and fuse (µs) | Copy and fuse / stock |
| --- | ---: | ---: | ---: | ---: |
| 0 | 178.919 | 174.259 | 143.045 | 0.799–0.805 |
| 1 | 176.211 | 171.785 | 143.112 | 0.811–0.818 |
| 2 | 173.812 | 168.862 | 143.240 | 0.821–0.824 |
| 3 | 171.345 | 166.486 | 142.347 | 0.831–0.840 |
| 4 | 168.383 | 163.930 | 142.596 | 0.841–0.861 |
| 5 | 166.168 | 160.967 | 141.876 | 0.852–0.870 |
| 10 | 152.442 | 147.781 | 141.474 | 0.926–0.935 |

Both candidates beat stock at all seven exact shapes on both clocks in each
process. Copy and fuse also beats fused mapped in all those pairs. Admit a
scoped decode candidate for the tested hit counts. The production path needs
an existing host publication boundary that can select the post operation
without extra synchronization. Check reuse of the ordinary assembled slab,
copy and completion ordering, and actual dispatch in a native run before
comparing model throughput. Component results cover only these width-four
shapes. The prior copy regression at 20 hits remains a separate result.

Weighting the fixture median differences by the seven exact census counts
gives 1.271 seconds, or 0.108 ms per emitted token for that 11,783-token
request. This is an arithmetic projection without production selection or
launch overhead. Keep CPU expert service and the GPU verifier dependency
chain as the larger optimization targets. The existing verifier loop joins
the CPU callback and outstanding expert copies before its next layer graph
launch. Use that boundary to test the copy without adding a wait. Preserve
the stock route for all other widths, placements and unmeasured hit counts.

The comparison closes at `d3f81b99a`. Verified the Git source and recipe,
691 compiler inputs, 27 linker inputs and fixture hash before retiring
`dflash2-decode-combine-stock-v1/{0,1,2,3,fixture-build}`. The 50 retired
files occupy 13,902,658 bytes. Their sorted inventory hashes to
`837f2d7f7dbeb531b37ba959b07bddc9f8e493468357a3ac88595d7a6fa6a502`.
The root receipt is unchanged.

The event-query process at `65f0f927c` repeats the same work at 16.443 tok/s.
The two resident wrappers each run 64,560 times without query or clock
errors. Their wall times are 224.99 and 314.64 ms, totaling 0.216% of the
250.02-second decode. The pool-return wrapper alone accounts for 0.126%.
The second clean control at `e0cb8fc25` closes the bracket at 16.666 tok/s.
All four processes match the original work, including 4,111 emitted tokens
and 2,767 accepted drafts out of 2,863 offered, and close without fresh Xe
faults. The two clean controls have a median of 16.681 tok/s. Caller-ledger
latency increases by 0.10–0.28% and event-query latency by 1.36–1.54% against
those controls. Both observations pass their 10% overhead gate. Prioritize
the CPU expert work and the GPU dependency chain over the resident query
wrappers. The measured intervals still leave physical memory traffic and
GPU resource saturation unresolved.

### Decode worker instrumentation on the a347 build

The current a347 engine, built from `221e95e9ad890383deb418d0a865ee32c99168b6`,
already has `STRATA_CPU_SERVICE_PROFILE:BOOL=ON`. Its compiled `pool.cpp` and
`pool.hpp` match the producers in `verify_window_cpu_service_overhead_v1`.
The existing `STRATA_CPU_SERVICE=1` startup setting can therefore expose the
CPU pool's GU/Down and other leaf wall/thread-CPU clocks without a new native
build. The fresh current-engine collection below now qualifies its observer cost.
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
or changing the original failure or adoption status. V3 is closed with all six
processes normally0, quiet validation, closed owners, no fresh fault and all15
final model-asset hashes. Each long request has33208fresh/4923generated tokens,
4419offered/3451accepted drafts and identical aggregate work/output hashes.
The three on/off per-token decode-latency ratios are
1.009036828/1.007815412/1.007968429 (geometric mean1.008273410), passing the
predeclared1.05cost limit in every pair. This qualifies the incremental leaf
observer cost, not clean speed or inference adoption.

The three observed long requests each execute1254500CPU expert jobs and
1450668CPU entries on six threads. GU occupies67.78..67.88% of concurrent
leaf CPU time, Down31.93..32.03%, and hidden quantization about0.19%.
NT1 accounts for86.7825% of CPU job count. The repeated top GU CPU bins are
GU18/Down20/NT1, GU21/Down20/NT1 and GU22/Down20/NT1; their ordering below
the leading bin varies. Concurrent leaf CPU totals span515.93..517.17s;
they include memory stalls and are not additive wall critical path or a
measurement of physical RAM/PCIe traffic. These results prioritize the GU
working set and a smaller-resident-KV/larger-expert-cache comparison, while
actual hardware saturation remains unknown. Follow `q6_dense_cpu_leaves_v3`
in the compact result for every sample and the selected operation/bin totals.
V1's prelaunch
artifact-reader lookup and V2's end-gate/client-journal ordering failures remain
failed in their original records. The end gate precedes client result append:
bind four warmups before begin, and all five client/native leaf records after
normal exit. Source4353b487f passes15CPU controls, including this ordering, and
the actual failed-V2 long window passes the unchanged BF16/head/IQ3 route checks
and new dense-C4 reader offline. The prospective fresh cohort is V3; no failed
sample is relabeled as eligible speed. Keep
leaf CPU time separate from completion polling and nested pool/wait intervals.
Test physical counter collection separately before making a RAM-bandwidth claim.

### KV and expert-cache capacity candidate

The current1256-slot per-layer cache requests2,575,206,400device-arena bytes
(2.39835GiB), including256-byte slot alignment. This is source-derived capacity,
consistent with the startup's rounded2.40GiB, not physical VRAM traffic or the
allocator's internal overhead. The frozen `native_experts.txt` has SHA256
`d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d`.
All48layer payload sizes are multiples of256; their sum is98,227,200bytes,
and the final layer uses2,662,400bytes per slot. For N slots, the existing
`per_layer_slot_sizes` gives every layer floor(N/48) slots and gives the entire
remainder to the last layer. The arena therefore requests
`floor(N/48)*98227200 + (N%48)*2662400` bytes.

| Slots | Requested device-arena bytes | GiB |
| --- | ---: | ---: |
|1256|2,575,206,400|2.39835|
|1488|3,045,043,200|2.83592|
|1536|3,143,270,400|2.92740|
|1632|3,339,724,800|3.11036|

For this model's12QSA layers, INT8 K/V payload is12,672bytes per logical cell.
131072/65536/49152resident cells reserve1,660,944,384/830,472,192/622,854,144
payload bytes. A smaller resident window also needs full-context host backing,
residency maps and a full-logical-range prefill staging pool; indexer/RoPE and
other session buffers still depend on logical context. Payload savings alone
do not establish net headroom or allocation admission. Keep262144logical
capacity and its full-fill/tail/RESTORE/handoff requirements in the proposed
residency experiment; do not reduce logical capacity to manufacture headroom.

The compiled native221e QSA16 preflight rejects mode1 main KV before request
submission. The new source adds an opt-in `STRATA_SYCL_PREFILL_QSA_STREAM=1`
route alongside the existing
native16/QSA selectors. Default mode0 retains its original producer. The new
producer quantizes BF16 K/V once on the GPU, publishing the same INT8 codes
and FP16 scales to registered Host storage, full-logical-range Device staging
and valid resident slots. Missing/invalid resident pages do not suppress the
two complete mirrors. Each writable view, input and page table is checked for
extent, domain and aliases before submission. Stage attention uses
`{n_pages,n_pages}`; resident attention keeps `{n_pages,n_slots}`. All ten
typed attention selections use the actual owner extent. Every QSA layer is
preflighted before the request submits work. Decode kernels and dispatch stay
unchanged; the shared Allocation lifetime changes need regression qualification.

The Host registry is distinct from the existing Device-only registry, retaining
the owning Runtime across Allocation moves. Runtime device selection requires
`usm_host_allocations`; `malloc_host` and the consuming queue share its context
and device. This matches the [SYCL USM accessibility contract](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_unified_shared_memory_usm).
Inputs/maps/owners must live through completion; CPU access waits for checked
completion. Production mode1 backing in `layer.cpp` comes from the registered
Host allocator through `cudaHostAlloc`/`cudaHostGetDevicePointer`, rather than
ordinary pageable memory. No foreign pointer is admitted merely by its USM kind.

The tests in `tests/sycl/qsa_native16.cpp` add Host-domain/move/retirement
refusals, three-store dyadic numerical and nonfinite/underflow/overflow checks,
missing/invalid resident maps, page tails, padding, aliases and short owners.
The default small screen preserves the resident tests. Separate
`sycl_native16_qsa_test --streamed-full-extent` fills all262144logical cells in
32chunks, overwrites the final cell and compares both full mirrors and the
first/last resident pages against an independent component oracle. The test
reads Host data directly only after checked completion. Its CMake target links
the actual producer TU and Runtime with the production strict arithmetic flags,
without building the entire engine. This is not an attention/model/RESTORE or
performance qualification, nor an identical-generated-text requirement.

At native source `7067084897574ad727bae6d93ca3a4726ea6e00c`,the supervised
five-TU component build passes24host compiler-census controls and records
724compiler inputs/24external links/zero uncovered inputs. Its executable
SHA256 is `dcc3097facc58b9603f5662709e97176746a13ac0e0a8ab0f1378a257f974cf2`.
The small traced/parameter-validated fixture and the separate262144-cell
producer fixture both pass on this Arc B570:normal0,closed owners,quiet
parameter warnings,unchanged boot and no fresh driver faults. Both use the
qualified runtime safety environment and pre-main loader probe;ordinary
provider teardown remains unqualified. Follow `qsa16_stream_components_v1`
in the compact committed results. No model or speed is measured by these tests.
Preserve the552b prefill fallback and a347 current decode reference. Retire
the temporary component build before a fresh engine generation after exact
source/recipe recovery;keep only the compact evidence required by that build.
Typed staged attention,whole-model physical262144/RESTORE/handoff and matched
PP/TG/TTFT are separate remaining gates before a cache-residency choice. No
resident size/cache count is selected from payload arithmetic alone. The
current full-engine source basis remains native221e95e9ad890383deb418d0a865ee32c99168b6;
The unchanged `expert_slot_sizes.hpp`, `expert_cache.cpp`, `layer.cpp` and
`session.cpp` supply the capacity arithmetic above. The new QSA16 prefill route
does not alone qualify attention/model execution;the bounded model result below
now supplies that separate route evidence.
Source and component evidence are merged into the pushed permanent
`feature/sycl` at `340ea79f9fe0914383643efaf0148da4fd5e4b19`. Every original
measured child commit remains reachable there;the completed
`experiment/qsa16-streamed-prefill` remote/local branch and worktree are removed.
The prospective `check_qsa16_stream.py engine-*` build reuses the a347
compiler/SDK recipe,requires exact original producer/Runtime/fixture bytes
and a complete full-engine census,and inherits no model qualification.
That full candidate now builds at source
`b9aa9c4dea9ca512ed20731fc81d805a1a695c2c`,engine SHA256
`1e598e8e6f04c4bf2e65fb295fb7c9fea9d66406084d612c7d7cda8744ddf391`.
Its closed `qsa16_stream_model_native_v2` build records128TUs,
1462compiler inputs,one generated input,29external links and zero uncovered
inputs,plus four same-build ownership/submission/retirement/policy fixtures.
V1 closed after successful configuration,before native compilation,to replace
full public parser arrays with a count/hash summary;its temporary configuration
is retired. V2 retains the exact original producer/Runtime/test bytes.
`qsa16_stream_model_device_v1` now closes all five prerequisite stages:
50CPU controls,11native refusals,the phase/shape policy,unchanged selected
head-module structure,10submission cases,16phase-cache windows,and fresh real
head0/head2 processes with backing0. Both head modes pass four numerical cases,
two retire/overwrite/restore cycles and12graph replays;head2 also passes eight
partial-column cases. All native processes exit0with closed owners,quiet
validation,unchanged boot and no fresh fault. At the closed four-stage boundary,
controller source`54e1f9ffaff3fbf54d1fa240db3e00aae5381981`adds exact seed-recipe
equality and12CPU controls before head2. The actual recipe matches;the original
observer's missing equality predicate and original results remain recorded.
No engine/fixture is rebuilt or relabelled. These prerequisites alone do not
qualify an actual typed-attention model route,whole-model physical262144/
RESTORE,task behavior or matched PP/TG/TTFT.
The first actual`qsa16_stream_model_entry_v1`is **failed**,with normal native
exit0,closed owners and no fresh fault. All32769/82/32769thinking requests
produce32tokens,but the planner selects4096chunks despite the8192argument;
the frozen actual-route predicate rejects that result. READY reports262144
logical cells,20480resident cells,1256expert slots and1206MiB free. Each long
entry borrows704slots,keeps all32768BF16residual rows on GPU(671088640bytes)
and allocates1363148800bytes for the layer cache. The full logical INT8 stage
adds276824064bytes to the shared scratch loan. Existing`STRATA_KV_STAGE_OWN=1`
allocates that stage separately. Do not
relax V1's predicate or label its4096execution a successful8192admission.
Final full model hashes were not reached after that rejection. Physical full
context,RESTORE,numerical/task and clean speed remain separate gates.

The separate`qsa16_stream_model_entry_v2`at controller source
`48e4a5e02f15b48b5f75a91e341bca4dc3dc53f9`now **passes** with that own-stage
setting and the same native source/five ELFs. The full20-field INFO reader
passes two valid and13corrupt host cases,including duplicates and extra fields.
Actual READY retains262144logical/20480resident cells,1256expert slots and
reports942MiB free. Both32769-token thinking prompts run8192chunks,borrowing
1124slots while retaining132,with all32768BF16residual rows on GPU and the
same1363148800-byte layer cache. The intervening82-token prompt still uses
decode placement. All three produce32tokens;two head retire/restore cycles
and33live decode-C4replays pass. Native exit is0,all owners close,validation
is quiet,boot unchanged,no fresh fault,and all15final model hashes match.
This qualifies the bounded actual mode1route/handoff. Numerical/task quality
and physical262144fill/tails/RESTORE remain open.

The separate clean`qsa16_stream_prefill_baseline_v2`now closes at controller
source`5e9c1f85715b862ac76f1ab8ca216d847ba6f67f`,using the same b9aa/1e598
native generation. Four thinking-on`bench.sh`exact32769-input requests each
produce24tokens with zero reuse. First measured PP is443.791tok/s and client
TTFT74.122s;subsequent prompt times are63872.0/63655.9/63720.7ms,with median
**514.2599tok/s**,63.7207s prompt time and63.863s client TTFT. All four actual
8192chunk/BF16QSA/head-handoff/residual-conservation checks pass;all32768HC4
residual rows stay on GPU. Native/benchmark exit0,owners close,validation is
quiet,no fresh fault occurs,and all15final model hashes match. Evidence is
`55f4a341ffcb4a8849c1056f3bd4bb4e5dd4985a`;the CSV keeps first and subsequent
scope separate. This is one process at one length,not an input-length curve,
independent-process repeatability,a matched gain,the>=64K throughput target,
sustained decode,model quality,physical full context or adoption.

The clean admission baseline deliberately disables nine settings from the
faster552b117Kreference. Their composition now passes the separately committed
`qsa16_stream_fast_entry_v1` diagnostic:release-MTP-experts,MoE window10,
score GEMM,mixed scratch,packed384-slot expert bank,cache scratch lease,
MTP FP16batch and split accumulator(counting bank size and packed mode
separately). Controller source is`3d2a9d4224dd82ba10e129b7ffb81a00670315f9`;
native b9aa/1e598 and independent head2/dense1 decode remain unchanged.
Thinking32769/82/32769inputs each produce32tokens. Both long requests use
8192chunks and typed BF16QSA;96packed layer entries have384..512slots,
both675MiB MTP expert restores complete,and four uncoupled round recaptures
and27decode-C4replays succeed. Eight8192-row mode2draft callbacks select
the pinned FP16batch branch. Checked cache bindings/final drains,normal0,
closed owners,quiet validation,no fresh faults and all15final model hashes
pass. These token-tiled diagnostics include observers and are not clean speed,
sustained decode,task quality,physical262144/RESTORE or adoption evidence.
The first clean fast-setting point now closes under
`qsa16_stream_fast_prefill_v1`,controller`ec9bbe3c4dca857ea7afb673c9ae962229e96da3`,
with the same native build. Four thinking-on`bench.sh`32769-input requests
produce24tokens each with zero reuse. First prompt time is51.2348s and
client TTFT51.374s(639.5848PP). Subsequent prompt times are41.5926/41.2262/
41.5664s,with median**788.3531PP**and client TTFT41.738s. All four actual
window10/staged384/ordinary-accumulator routes keep all32768BF16 residual
rows on GPU and restore the675MiB MTP expert bank. Native/benchmark exit0,
closed owners,quiet validation,no fresh fault and all15final hashes pass.
The earlier514.2599PPbaseline and this point are one process each,not a
prospective repeated pair. Keep first/subsequent scopes separate;24outputs
do not qualify sustained decode. Record evidence is`3863ba3ef`and the CSV.
In this native server path,`generate.cpp`starts`r0`before request-state and
prefill preparation and closes`prompt_ms`after the prompt wrapper restores
MTP experts,joint dense/head owners and cache slots. Lazy MTP round-graph
capture occurs later in decode/client TTFT. This source check defines the
clock scope;it does not measure any restoration's individual cost. The last
prompt token enters the first verify window,while reported PP uses the
native read-token count. Client TTFT includes the first nonempty thinking
or content output,after the recorded count-only preflight.
Observed MoE bank copy requests are35.343..35.493GB/request,plus62,914,560
bytes reported by the dequant source counter;these are requested operations,
not measured wire bytes or copy time. Job/copy counts vary across nonce
requests while placement stays all-GPU/staged384;whole route-record
inequality is not a placement change. Next repeat promising points and
characterize length/route before adoption or a>=64Ktarget claim. Preserve
the validated552b fallback. The reviewed runtime split gate is
actual prefill rows>90112and<=117488,with additional admission checks;its
accepted replan requests128slots. Do not assume the32K384-slot geometry
at117489input tokens. Actual hardware saturation remains unknown.
The closed `qsa16_stream_fast_curve_v1` campaign measured32/48/64/80/96K
actual inputs in that order,with four thinking-on24-output requests per fresh
process. All20requests pass with zero reuse,actual8192chunks,all residual rows
on GPU,675MiB MTP expert restoration,quiet validation,normal0,closed owners,
unchanged boot/driver and no fresh fault. All15model assets pass full hashing
after the fifth process. Original controller source set is
`131ce053b106df631037355fe62b0cd9eb1da79c`,native remains b9aa/1e598,and
closed evidence is`616562a28f009a2e7be2404cb3119861aea6acc2`.

| Actual input tokens | First PP (tok/s) | Subsequent-three median PP (tok/s) | Median prompt (s) | First client TTFT (s) | Subsequent median client TTFT (s) |
| --- | --- | --- | --- | --- | --- |
|32769 |636.9282 |794.9319 |41.2224 |51.583 |41.357 |
|49153 |742.0116 |876.5926 |56.0728 |66.392 |56.218 |
|57345 (added) |784.8137 |911.2347 |62.9311 |73.226 |63.084 |
|65537 |805.4761 |916.5706 |71.5024 |81.517 |71.662 |
|81921 |838.3281 |934.2008 |87.6910 |97.881 |87.866 |
|98305 |866.0715 |946.8245 |103.8260 |113.717 |104.007 |

First means the first measured long request after standard bench warmups and
count-only preflight,not cold server startup. All32..80Kpoints share the
ordinary384-slot route;96K uses the separately checked128-slot split route.
The latter splits the accumulator across GPU allocations,not CPU accumulation;
its256-byte tail guard and final drain pass. Useful samples/source limits and
both first/subsequent scopes are indexed in the performance CSV. This is one
process per length,not a repeated paired gain,sustained decode,task quality,
physical262144/RESTORE,ordinary-provider teardown or inference adoption.

The separately added `qsa16_stream_fast_fit_point_v1` closes all four57345-input
requests with the same native binary,configuration,nine FAST values,boot and
driver. All57344BF16 residual rows stay on GPU with384staged slots;quiet
validation,normal0,closed owners,no fresh fault and15final full model hashes
pass. Only controller source`6e51f04b114a9939bbfe632d8e3e937cb502a9e8`
differs;the other688measurement input hashes match. Its committed evidence is
`f48954c35eff79dff47a045a4bbd3dcdd98fbaf8`,with both scopes in the CSV.

Five distinct ordinary-route lengths now support whole-length held-out
comparison of the existing three hypotheses. Keep96Ksplit separate;it has
one point and no fit. The retained result separates first/warmed prompt and
client-TTFT series and all individual subsequent samples.

| Prompt-time hypothesis | First mean / maximum held-out error | Warmed mean / maximum held-out error |
| --- | --- | --- |
|Affine |1.438% /3.165% |1.618% /3.908% |
|Quadratic |0.972% /2.584% |0.956% /1.780% |
|n-log-n |1.272% /3.754% |1.133% /3.013% |

With`x=actual_read_tokens/65536`,the warmed quadratic description is
`T_seconds=15.1106322+48.4445914*x+7.70819048*x*x`;
the first-use description is`25.8472962+47.2465768*x+8.18681905*x*x`.
Warmed client-TTFT quadratic held-out error is0.952%mean/1.766%maximum.
Quadratic has the smallest observed prediction errors here,not a proved
universal law. Its worst warmed prompt error still exceeds the within-process
sample ranges;one process per length does not measure independent-process
variance. The56Kpoint also reveals a local slope change,so this is only an
interpolation aid within measured32..80Kand this route. Do not extrapolate or
attribute any fitted intercept to physical fixed work. Keep all three models
and their residuals/held-out errors;no hardware-bottleneck conclusion follows.
The original collector V1 remains failed before native launch because its
parent already carried the target preload. V2,implemented in the existing
benchmark observer at`9cc9711af4b6d1b36cf32cbf0a98b14ba588bec8`with plan
`ed03edb82bebdb4dd0b36d297a94e9dd19085651`,now passes under
`qsa16_stream_collector_active_v2`. All five original b9aa native ELFs,recipe,
runtime and existing owned Sysman0 UnitTrace launcher/collector remain exact.
Only the exact qualified inherited private preload/expected-loader fields are
detached before the native wrapper attaches them;runtime safety is preserved.
Six valid CPU inventories pass and six corrupt inventories are rejected,
including a same-total wrong-class case. The no-argument compatibility fixture
then records1direct,37producer and24consumer kernels,with exact type identity,
62unique native host-flow joins,249GPU events and770host events. The274453-byte
timeline is flushed with finite contained clocks. Original numerical/copy/
event/graph/lifetime checks,pre-main validation and normal0pass;owners are
closed,the named session is removed,and boot/driver remain unchanged with no
fresh fault. This qualifies current-build small-fixture collection only.

The representative `qsa16_stream_fast_profile_v2` model bracket now closes at
evidence`c2984a6d1`,with controller source`1a907b77a427ce92ad9185d988b22c4c7c7638ef`
and plan`917d3ca3682f8cb6cd1472567ea53b32523cba62`. Native b9aa/1e598 and
all five original ELFs remain unchanged. Three fresh clean/profile/clean
processes each ran two identical32769-input,thinking-on24-output requests.
Both clean second requests are785.6389/788.3740PP and41.924/41.702s client
TTFT;their first requests are633.9598/634.6437PP and51.833/51.770s TTFT.
The profile's prompt/TTFT ratios against both brackets are1.071..1.092,
passing the prospective1.2observer screen. All six requests,unique native
host-flow joins,CPU/status conservation,normal0,quiet validation,closed owners,
unchanged boot/driver,no fresh fault and15final model hashes pass. This
qualifies bounded attribution,not a speedup or precise process variance.

The second profiled request has45.2316s prompt time and45.5320s client TTFT.
Its event window ends at the first nonempty output and includes handoff/first
token work. These intervals and CPU regions overlap and must not be added.

| Measured scope | Second profiled request |
| --- | ---: |
| Union of observed GPU kernel intervals,excluding copies |29.7750s |
| Memory-copy M2D service sum;API-requested payload |10.1518s;46.505GB |
| QSA prefix kernel service sum |6.7955s |
| Streamed Q8 KV append kernel service sum |2.2177s |
| GDN recurrence kernel service sum |2.1996s |
| Main-thread CPU inside observed waits |21.9770s |
| Main-thread CPU outside observed waits |20.3037s |
| Outside-wait CPU:upload / GEMM submission / row grouping |4.5490 /3.6367 /0.0744s |
| CPU outside classified waits/host regions |3.0221s |

For this window,all observed GPU service sums total40.6446594s;their union is
`window_seconds-observed_gpu_gap_total_s=40.6340712s`. The0.0105882s excess
is an upper bound on cross-family overlap:the observed copy and compute
intervals have negligible overlap here. The4.8979411s outside that union
means gaps in observation,not established hardware idle or CPU causality.
`M2D` is the collector's memory classification;its payload and service quotient
are not physical PCIe traffic/bandwidth. GPU host-memory stores are also absent
from copy-API payload totals. Arithmetic utilization,VRAM/RAM/PCIe saturation
and the limiting hardware resource remain unknown. A busy main CPU thread
includes substantial completion waiting;row grouping alone is small.

Next assess expert-copy preparation/submission and the streamed append's
direct Host-USM mirror stores independently. The existing optional Host-USM
expert staging and GPU-local quantization followed by bounded bulk mirror
copies are candidates,not adopted improvements. Preserve exact KV codes/scales,
partial-page/tail writes,authoritative host state,checked lifetime and decode
handoff. Each candidate needs its own committed plan and clean phase comparison.
The bulk-mirror candidate is implemented in`src/prefill/sycl/qsa_native16.cpp`,
default off under`STRATA_SYCL_PREFILL_QSA_BULK_HOST_MIRROR=1`. The quantizer
keeps resident/Device-stage writes and publishes the Host mirror through
same-queue copies. Whole pages use one contiguous span;each partial page uses
an exact per-head span. Existing owner/domain/alias checks precede every
submission,and completion observation includes the copies. The five-TU
component at`a4162bf1a396c8633bcc0f34f743b1c772b2fdb5`passes all four fresh
small/full off/on processes,including all page-offset cases,physical262144
producer fill and final-cell overwrite. Full-on observes33calls,136copies and
276825120requested bytes;normal0,closed owners,quiet validation and no fresh
fault pass. This qualifies the producer,not full-model context or RESTORE.
The full candidate is compiled from`611a0fd6fecc4593e17160c294103cd4464a292b`
(engine`56cad9eb45f8f5fc…`):128TUs,1462compiler inputs,one generated file,
29external links and zero uncovered inputs. Its five native host/submission/
phase/head checks close at`65cff97e1`,with original decode-head IR preserved.
Its fresh thinking-enabled32769/82/32769input model handoff closes at
`f79a5cd2145efb91ff85445710bbb31f62f00e61`:32outputs each,normal0,closed
owners,quiet validation,no fresh fault and15final model hashes. Actual long
prefills publish96bulk calls,384copy commands and830472192requested bytes;
packed-bank/cache binding,MTP expert restoration and independent decode
recapture pass. This diagnostic charges validation/logging. A separate clean
`bench.sh`cohort at`f0b08ded893c9ed253322b330fd825c96234357c`now closes:
eight fresh processes,ABBA at32769and65537inputs,three requests each.
All24original requests have exactly24outputs and zero reuse;normal0,closed
owners,quiet validation,unchanged boot/driver,no fresh fault and15final
asset hashes pass. Subsequent-two process-median aggregates(off/on)are
790.5512/827.4974PP(+4.6735%)at32Kand914.7170/970.5080(+6.0993%)at64K;
clientTTFT is41.5905/39.74375and71.811/67.687seconds. Both independent
pairs favor bulk at each length. First-use,individual and process spread
remain separate in the compact result and eight CSV points. This is clean
software-dispatch evidence,not observed bulk-command counts or physical
traffic. Exact24is checked from every retained original sample because the
shared runner accepts1..24;fix that gate before another cohort.
This improvement does not locate physical saturation. The older2.217678s
append service is not a post-bulk arithmetic cost;the current measured cost
is recorded below before selecting another quantizer. Source-view mode1is already observed in all24
requests,so crosswalk its earlier4.5512%gain before repeating it. The two-point
adjacent slopes0.9215233/0.8523079ms pertoken(off/on)are local only. The flag
stays default off;1000PP,sustained decode,task/agent,full262144model/RESTORE,
ordinary-provider teardown and adoption remain open.
The prospective reader at source07eed6e6a now passes the current native,
recipe,all five same-build ELFs,complete compiler inputs and original closed
device/model/runtime evidence without reading historical comparator ELFs.
Its18host refusal cases preserve native-source/current-fixture/count gates;
exact24is selected by the new policy while legacy range defaults remain.
Original measurement source93d908and all24samples/outcomes remain unchanged.
The actual b9aa1132project inputs and two dependency revisions,128TUs/1462
compiler inputs,recipe and generated header are recovered in the compact
generation review,including52GGML headers absent from the old prefix list.
The superseded b9aa root is now retired:188files/151969592bytes,with the
current five-ELF/device/model/runtime and byte-identical collector-fixture
admission passing before and after deletion. Three named generations remain:
current56cad9,552bfor faster117Kprefill revalidation,and a347for independent
sustained decode. The completed same-binary bulk bracket is recorded below.
This is not an expanded build quota or a new native/GPU validation.
The post-bulk32Kclean/profile/clean bracket closes at3d5857f99:all6
thinking-on exact24-output requests pass,normal0,closed owners,quiet
validation,no fresh fault and15final model hashes. Observer prompt/TTFT
ratios1.086..1.170pass the prospective1.2screen. Clean second requests are
825.1807/826.3940PP and39.856/39.797s clientTTFT;these are controls,not a new
paired speedup or sustainedTG. The second observed request has27.6665s
kernel union,10.9801s copy union and only0.006408s observed overlap. M2D
service is10.2080s for46.505GB API-request bytes;prefix attention is6.7960s.
The maintained interval reducer identifies48append kernels totaling0.08380s
and192D2Hcopy commands totaling0.07404s/415236096requested bytes. Thus the
old2.2177s append cost no longer selects a quantizer optimization. Main CPU
has19.9011s in known waits and20.5176s outside,including4.6777s upload-submit,
3.6695s GEMM-submit,0.07558s grouping and0.000046s source copying. These
CPU categories overlap GPU work;source-view copying is already eliminated.
The cached-Q/vector-KV subgroup component now passes11numerical/boundary
cases and108paired timings across3fresh processes at5dcdec7e0. Whole-entry
ordered/candidate medians are8.7367185/8.683248ms(cap33gather),
224.2698465/223.068008ms(cap2051gather),and139.9996785/138.6366405ms
(cap2051clustered);geomean time ratio0.99292646. All process medians improve.
This re-evaluates the existing bba952e33 algorithm,whose original0.993646
ratio failed its<=0.95gate;that result remains failed. Main verified identical
template/admission source bodies. The new precommitted reproducibility gate
has no minimum gain and passes;model PP,TG,quality and adoption remain unrun.
Keep this choice for a combined prefill trial;prioritize actual upload
registration/submission or selector-locality evidence instead of another
isolated sum experiment. Keep physical-context and independent decode gates. Hardware saturation and the limiting physical resource remain
unknown;observed GPU copy services are not wire traffic.
The original model profile V1 remains failed:SIGXFSZ at its exact256MiB file
limit. Its original ownership result stays false;the four captured executors
are gone and its exact named collector session is separately removed. V2 used
a prospective1GiB/file,1.5GiB/job,2GiBwhole budget and a581101981-byte flushed
trace. After committing the compact evidence and resource decision,336closed
raw/log/JIT files totaling914207701bytes are physically retired with a hashed
deletion manifest. The33thin records/runtime/plans/ownership receipts remain
for the next candidate admission;all original outcomes and native builds are
unchanged. Task/agent quality,sustained TG,physical262144/RESTORE,
ordinary provider teardown and overall model adoption remain open.

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
