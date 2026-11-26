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

The selected verifier at `968e71988` can split a decode window into two
groups for `T >= 2`. In its SYCL path the host synchronizes the current
segment, calls the expert pool, retires copies and then submits the next
segment. Plan publication can launch that group's shared and resident work
while the CPU callback runs. Splitting still changes the 48 host completion
steps to 96 and leaves next-segment submission after the callback. Use the
planned fixed-work publication test to isolate readiness and the downstream
interval before selecting a new split-window schedule. [Verifier](../src/core/verify.cpp)

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
