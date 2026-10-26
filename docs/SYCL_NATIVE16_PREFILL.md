# Native 16-bit long prefill

## Selected implementation and independent phases

The selected engine is the original source
`6e40afc1beddd059ee518617f6c1efce1da14aca`, SHA-256
`22182f1a5e62e5f4259ce49a75ae44afbd988166ec79324816e499ff37ef60e4`.
Its engine and twelve same-build fixtures match their original hashes. All9853
original project/dependency inputs and the build recipe recover from Git.
Restoring that exact package required no recompilation and supplied no new GPU
validation. See [the selected build proof](../bench/results/2026-10-15-pr6-recovery-audit/native-gdn-engine-build.json).

Prefill and decode have independently selected kernels, scheduling and placement.
Keep the existing decode implementation, configuration and1256 expert slots
while changing prefill. Any decode impact from a prefill change is a separation
defect, including state handoff, configuration, placement or throughput changes.
A prefill gain cannot compensate for it. Shared processing requires matched
measurements demonstrating its benefit; legacy sharing is an audit obligation.

The SYCL build has a separate `strata_prefill` archive, but
`src/program/generate.cpp` controls both phases and their handoff. That archive
alone does not prove isolation. Audit the shared controller, native dense owners,
cache publication and restored runtime state, then repeat matched decode checks.
Keep short-input prefill through the decode weight/cache layout available.
Do not change process environment variables between phases.

The active source now restores nine prefill/planning/compatibility paths exactly
to6e40 and the15 coupled layout/admission/pricing/loan hunks in `generate.cpp`.
Its18 independent diagnostic/observer hunks are preserved byte-for-byte.
The unused streamed-QSA writer, host-span export and fixture are removed.
Decode kernels, configuration and1256-slot placement remain selected as before.
The measured restoration used native source`1baa0547f` and engine SHA prefix`2a128c6a`.
The original6e40 executable remains the selected fallback; restored-source
prefill speed screen now fails; task and full-context qualification is pending.

The reviewed `selected_native_tree_reconcile_20261025` decision in the
[research index](../bench/results/2026-10-15-pr6-recovery-audit/research-index.json)
archives its corrected20-path action map. Conversation RESTORE verification and
the `STRATA_STATE_HASH` KV-range helper remain for the next state checks.
CMake now defines manual `strata_full_context_diagnostic` and
`full_context_diagnostic_host_test` targets. Only the former defines
`STRATA_SYCL_FULL_CONTEXT_DIAGNOSTIC`; normal `strata` keeps its original admission.
The17 linked targets and all1416 actual compiler inputs were checked. Normal
`strata` compiles out CPU-service observation and the full-context macro.
The [closed component gate](../bench/results/2026-10-15-pr6-recovery-audit/prefill-restore-device-current.json)
reuses eight qualified host controls and three submission/graph cases, then
passes five fresh device processes with the documented Alias1/MMQ0 scratch
layout fixed before execution. These cover15 scratch cases,20 cache-loan cases
over three cycles, private-bank guards/map/drain and113 RESTORE-smoke checks.
All fresh processes pass pre-main/parameter validation and close normally,
with the pinned updated NEO observed and no fresh fault. Source, runtime and
all17 artifacts are rechecked. Earlier census, supervision-budget and fixture
startup refusals remain failed in Git; no native recompilation or relabelling
was used to correct the later controller-only cases.

The [bounded real-model state gate](../bench/results/2026-10-15-pr6-recovery-audit/prefill-restore-model-current.json)
now passes four requests in one fresh process:33-token short input,16385-token
long input,another33-token input,and16401 tokens with16385 reused. The long
BF16/mode10 pass restores all48 dense layers; every request restores the1256-slot
cache payload and map. Live RESTORE verifies all12 main state layers,the draft
and running state before continuation. All15 final model hashes match,the
updated NEO is observed at all five process boundaries,parameter validation is
silent,and the native child exits normally with no fresh fault. This synthetic
diagnostic deliberately enables conversation parking; it is not a speed or
task-quality sample. The earlier cache-disabled plan was blocked before GPU
execution and remains transport-only evidence in Git.

These gates qualify bounded fixtures and this state handoff only. The prefill
screen below rejects this package;decode speed,quality and agent-session qualification
remain open. Physical262144
allocation/final-slot execution still needs its own bounded plan and evidence;
the earlier retired streamed/layout results do not qualify this source.

The [current throughput comparison](../bench/results/2026-10-15-pr6-recovery-audit/restored-phase-bench-current.json)
used the updated NEO and the6e40/restored1baa packages,with six fresh
processes in0/1/1/0/0/1 order. Official`bench.sh` measures four117489-token inputs
per process,separating first use from the subsequent-three median. Its CPU clock
and provider controls pass; all six processes have closed. This is a prefill-only
comparison. Keep the separate repeated decode and agent-session gates open.
On this B570,all six processes pass source/provider/journal/route/owner/fault
checks with identical request hashes and no fresh fault:

| Process/package | First use,tok/s | Subsequent-three median | Subsequent range |
| --- | ---: | ---: | ---: |
| 0/original6e40 | 900.6211 | 974.7035 | 974.4836..974.7956 |
| 1/restored1baa | 898.6723 | 975.4350 | 973.5533..975.6237 |
| 2/restored1baa | 902.5098 | 973.4517 | 972.8947..973.7801 |
| 3/original6e40 | 901.5887 | 972.2731 | 971.8380..973.5170 |
| 4/original6e40 | 903.1217 | 975.2123 | 971.6010..975.6091 |
| 5/restored1baa | 901.5811 | 971.3720 | 970.5519..973.1590 |

The paired ratios are1.00075053,1.00121216 and0.99606206;their geomean
is0.99933887. The last pair fails the prospective no-decline screen;the faster
screen also fails. Do not adopt1baa or advance its gated model trials. Keep the
original6e40 selected. All15 final model hashes match,parameter validation
is silent,and no fresh fault appears. Both packages exceed600tok/s repeatedly
but remain below the historical old-runtime1003tok/s and the current1000tok/s
target. This does not attribute that difference or qualify decode preservation.

The rejected generation and its closed comparison/model controls are now retired.
Recovery checked647 project sources,9311 dependency inputs,the original build
commands/environment/generated header,all1416 compiled inputs and17 artifacts.
The selected6e40 and its12 fixtures remain unchanged. The three closed Python
readers/launcher recover from`2e99d2c49`;the measured controller is`5c35a18ce`.
Compact samples and decisions remain in Git;successful logs,JIT data,temporary
plans and consumed research transactions are removed with a path/hash/reason manifest.

## Long-prefill processing order

The selected benchmark configuration uses native BF16, the vLLM Xe2 backend,
8192-token chunks, GPU residuals, mode10 MoE windows,384 expert-bank slots and
immutable Arena source views. The complete environment and configuration are
in [the current prefill comparison](../bench/results/2026-10-15-pr6-recovery-audit/phase-prefill-speed-current.json).
Use that pinned configuration rather than assembling flags from old reports.
The implementation is in `src/prefill/layer_major.inc`, `moe_window.inc` and
`src/prefill/sycl/`; the program controller owns its entry/exit transitions.

| Order | Work and ownership |
| --- | --- |
| Admission | Check actual tensor extents, VRAM and single-allocation limits, residual placement and scratch owners before submission. |
| Loan | Retire conflicting graphs, loan the checked expert-cache suffix, and retain its payload/map restoration contract. Dense decode owners move from Full to the private Bank layout. |
| Residual | Keep the long request's HC residual on the GPU in BF16; a budget estimate alone does not establish residency. |
| Layer | Process layers in order. Within a layer, issue8192-token chunks through HC normalization/mixing, attention or GDN, and shared/routed MoE work. |
| Experts | Borrow immutable private-Arena source views and upload the required layer weights into the bounded bank. GPU routing feeds a host grouping/map step; GU/SwiGLU and Down/scatter run on the GPU. |
| State | Publish KV and recurrent state with the required dependencies. Typed BF16 interlayer callbacks avoid a full FP32 host round trip. |
| Handoff | Drain temporary owners; restore48 dense layers, the expert-cache payload and its map. Expand the final residual for existing MTP/decode interfaces. |
| Decode | Resume the independently selected verifier/draft path and1256-slot placement. Confirm actual restoration and repeated throughput. |

Private-Arena views avoid the old512-expert CPU staging copy; shared-file and
unsupported providers retain synchronous copying. Source owners stay open through
checked queue retirement. This removes a CPU copy, not the requested expert H2D
payload. The selected MoE path still reads router IDs back, groups metadata on
the CPU and uploads maps. This is a remaining host boundary to measure, not a
GPU-only routing claim. A prior warm profile charged0.25367CPU seconds to host
grouping; that is neither its full synchronization cost nor a clean speed.
At117K input, the observed requested payload is about40GB/request;
it is not measured wire traffic. GPU-local residual copies remove the old
interlayer host round trips. Neither fact alone assigns the measured speed gain.

## Storage, arithmetic and provenance

`STRATA_SYCL_PREFILL_NATIVE16=bf16` scopes native16 to long layer-major prefill.
Short prefill and decode retain their selected FP32 route. Saved residual dtype
must also be BF16. FP16 and BF16 both use two bytes; select by range, numerical
contract and measured performance.

| Data or operation | Contract |
| --- | --- |
| HC residual, projection outputs and interlayer storage | BF16 producer/consumer tensors, without restoring the whole tensor to FP32 before every operation. |
| Expert hidden activations and dequantized expert weights | FP16 on the selected expert path. |
| XMX GEMM | FP32 accumulation/epilogue, followed by one16-bit output store. |
| Normalization, sigmoid, expert summation and routing | Local FP32 arithmetic; HC coefficients and routing decisions/weights remain FP32. |
| GDN | Persistent recurrent state remains FP32; internal boundaries retain their selected kernels. The chunk64 candidate is independently selectable, not implied by native16. |
| Typed QSA | BF16 projection/norm/RoPE/append/attention/gate boundaries with FP32 local/online state. The selected prefix uses ordered exponential arithmetic. |
| PLE and final interface | Existing FP32 PLE interface and one final residual expansion for MTP/decode remain explicit. |
| Scratch | Native views alias checked unused arena ranges; narrower tensors alone do not prove a smaller total allocation. |

The selected typed QSA route uses resident, unrotated INT8 K/V and session RoPE
tables. Streaming/host mirrors and rotated/Q4/hybrid KV need separately qualified
paths. Unsupported tuples/geometry fail before GEMM submission; there is no CPU
arithmetic fallback. Keep the FP32 reference for diagnostics. A validated
SGLang/vLLM arithmetic contract is acceptable; extra llama.cpp/FP32 precision is
not the adoption standard. Identical generated text is not required.

See [the vLLM Xe2 native16 implementation](SYCL_VLLM_XE2_NATIVE16.md) for its
checked BSD-3 source, AOT object and shape/lifetime contract. The original custom
joint-matrix backend was rejected for speed. XeStrata application code is
excluded from implementation adoption; the independent installed compiler is a
separate toolchain asset. Early sparse NLL/top1 checks establish neither broad
quality nor a permitted numerical degradation. Broader task qualification remains
open; do not revive an old rejected backend from its isolated correctness pass.

## Measured performance and rejected comparisons

Hardware is Arc B57010GiB and Ryzen5600X6c/12t. The historical old-NEO comparison
used official `bench.sh`, thinking enabled and six fresh processes in
original/candidate/candidate/original/original/candidate order. Each process read
four fresh117489-token requests. First use and subsequent requests are distinct.

| Process pair | Original subsequent-three median, tok/s | Candidate median, tok/s | Candidate/original |
| --- | ---: | ---: | ---: |
| 0 | 1003.0316 | 1002.7148 | 0.99968422 |
| 1 | 1002.6507 | 1002.7970 | 1.00014595 |
| 2 | 1002.3795 | 1002.6703 | 1.00029016 |

The candidate is source`1cba8a285`, engine SHA prefix`590f8237`. Its paired
geomean1.00004008 misses the prospective improvement gate; the first pair also
fails no-decline. It is **not adopted**, and its extra build is retired.
First-use926.174..929.804tok/s stays separate. Original6e40 repeatedly exceeds
1000tok/s on the subsequent-request workload. The24-token outputs qualify
neither sustained decode nor task quality. All processes closed normally,
all15 model hashes matched and no fresh fault was recorded. Full frozen evidence
recovers at`1e676d554`; the linked compact result retains individual samples.

The older362 package is also rejected. Its repeated long-generation comparison
missed the no-decline rule for both code and prose; a prior prefill improvement
cannot excuse that outcome. See [its decode decision](../bench/results/2026-10-15-pr6-recovery-audit/canonical-restore-decode-current.json).
Do not conflate362,590f and the selected6e40 package. Static cache reordering and
window-ranked membership were rejected separately; original placement remains.

### Historical best and the current decline

The20261026 audit distinguishes two differences. The fastest reported owned128
package used source396d and recorded1005.5..1005.8tok/s subsequent-process
medians;see[the direct owned128 result](../bench/results/2026-10-15-pr6-recovery-audit/prefill-long-input-current.json).
The six-process owned128/borrowed1256 comparison recorded a0.4899%
prefill decline for borrowed1256,with every pair slower. Its old2% screen did
not classify a regression;that does not establish equal speed or satisfy the
later requirement to restore the fastest prefill independently of decode.
See[the original comparison](../bench/results/2026-10-15-pr6-recovery-audit/cache-long-comparison-current.json).
Those arms also used different source packages,396d andbb71. This is a package
comparison,not an isolated causal measurement of ownership or cache size.
Do not recover that speed by shrinking the current decode cache.

The fastest reported three-process median is1005.8tok/s. Recomputing all
three processes from engine milliseconds gives medians1005.7715,1005.7655
and1005.4556tok/s;the later selected-package median974.7035 is about3.09%
lower. All four wire-body hashes,input/read/reuse/output counts and thinking
settings match the later requests in each of those three processes. Do not add the separate0.4899% package and
2.7873% historical differences as causal components:their revisions and
measurement windows differ. Source replay finds all14 files under
`src/prefill/sycl` and43 under `src/kernels/sycl` byte-identical between396d
and6e40. The differences include scratch/cache ownership,packed-bank dispatch,
host compatibility wrappers and diagnostic/phase orchestration. Unchanged
kernel files do not prove identical generated instructions,headers,dispatch or
runtime behavior. This narrows the source audit without clearing the surrounding
prefill path of possible regressions.

Separately,the original6e40 executable itself now measures slower. Recomputing
actual read tokens divided by engine milliseconds from the old cohort at
1e676d554 and the updated-NEO cohort above gives:

| Same original6e40 package | Historical cohort | Later cohort |
| --- | ---: | ---: |
| Three subsequent-process medians,tok/s | 1003.0316,1002.6507,1002.3795 | 974.7035,972.2731,975.2123 |
| Median of those process medians | 1002.6507 | 974.7035 |
| All nine subsequent samples,range | 1002.1991..1003.7986 | 971.6010..975.6091 |
| Median first-use speed,tok/s | 929.4942 | 901.5887 |

The subsequent difference is−2.7873%,or3359.8ms added to117178.4ms for the
same117489-token request. Median first use also falls3.0022%,so the loss is not
confined to the first-request initialization cost. These are historical cohorts,not a
contemporary driver A/B. The same binary SHA22182f1a,boot,kernel,four wire-body
hashes,117489 fresh input tokens,zero reuse,24 output tokens,thinking,1256-slot
profile bytes and original prefill configuration match. Both separate first use
from the subsequent-three median. `bench.sh`,`serve/server.py` and
`tools/sycl/engine_once.py` match byte-for-byte between controllers4d805d772 and
5c35a18ce. The prefill construction,eligibility,digest,request gate and
`run_prefill_only` functions also match. The changed client definitions add
long-generation options and do not change the prefill timing formula. Different
whole-client fingerprints alone do not establish a changed prefill measurement.
The follow-up source audit also finds byte-identical `measurement_owner.py`,
`measurement_guards.py` and `clock_overhead.cpp`,with identical process budgets.
An AST comparison finds changes only in `Suite` and `main` in the benchmark
client;the prefill-specific methods and module constants are unchanged.
The shared `Suite.request` change is inside its long-generation-only branch,
which these prefill requests do not enter. Both original
arms offer21 drafts,accept16 and generate24 tokens in every request. These short
outputs remain insufficient for sustained decode qualification.

In original6e40 `src/program/generate.cpp`,the request timer starts at line5959
and stops at line6503,after native dense restoration and the checked expert-cache
refill at lines6498..6502. Thus this API prefill metric includes the handoff back
to decode;it is not isolated GPU arithmetic throughput. The long layer-major
body handles117488 positions and leaves the last input position for the first
decode window,while the benchmark's unchanged read count is117489. That one-token
convention differs by0.000851% and cannot explain the observed decline. The86-key
environment template also matches the historical original arm exactly before
per-process output-path substitution;the same native-only validation preload
and charged warning-level parameter validation were already used historically.

NEO changed26.31.39395.16→26.35.39758.17;GMM,IGDML and zlib also changed.
Replaying the historical runtime template together with its recorded asset
overrides confirms four changed original ELF hashes and27 unchanged ones.
In particular,the original cohort already used the later udev hash;comparing
only the older base template would incorrectly count udev as another change.
The validation preload was already in use. The new libmetee dependency is an
additional inventory entry,not evidence of a measured bottleneck. The exact
old/new hashes are in the maintained runtime comparison. No system packages
were replaced by this read-only audit.
The same application executable rules out application recompilation as the
cause of this particular difference. It does not identify NEO rather than the
other runtime changes,clock/power conditions or concurrent workstation activity.
Do not dismiss the nonoverlapping repeated samples as noise or call the old
1003 result the latest current speed.

This establishes an observed end-to-end prefill regression with the same
measurement contract;it does not establish an application-source regression.
The later1baa restoration candidate separately failed its prospective
no-decline screen and was not selected. Its result does not replace the
same-binary historical comparison or qualify the active worktree as restored.

A20261026 read-only host check found all12 CPUs available,nice0,and no CPU
quota or throttled periods in the current Orca cgroup or any ancestor. The
current groups' cumulative counters rule out recorded CPU-quota throttling in
those groups,not scheduler contention or lower effective CPU frequency during
the earlier measurements. Current pressure/idle frequency is not a historical
control. Keep other changed runtime libraries and their interactions,active
CPU/GPU/VRAM clocks,paging and workstation contention unresolved.

The exact old NEO,GMM,IGDML and zlib vendor ELFs recovered in a private
directory;all four matched the historical SHA-256 values. No system package was
replaced. A subsequent full-userspace crossover qualifies both arms with eight
host controls and eight bounded traced canary/submission/graph cases,then runs
the unchanged original6e40 through six official117K processes:

| Process/userspace | First use,tok/s | Subsequent-three median,tok/s |
| --- | ---: | ---: |
| 0/current | 847.4394 | 906.3034 |
| 1/historical | 836.4635 | 919.1718 |
| 2/historical | 848.7697 | 908.1430 |
| 3/current | 853.6422 | 911.7371 |
| 4/current | 855.9296 | 922.9555 |
| 5/historical | 858.3383 | 913.9805 |

Historical/current paired ratios are1.01419882,0.99605790 and0.99027589;
their geometric mean is1.00012580. The signs differ and neither prospective1%
faster screen passes. All four changed ELFs are observed in each model process
with the expected hashes. All24 original engine clocks and request journals
replay,all15 final model hashes match,all native processes close normally and
no fresh GPU fault is recorded. Full frozen/closed evidence restores from
ffa4d459f/57a3dd86c,and the controller/controls from0697c74e8. This comparison
does not adopt a runtime or qualify sustained decode.

The historical userspace set does **not** restore the historical1002.6507tok/s.
The current arm's three-process median is911.7371tok/s,9.0673% below that
same-binary historical cohort and9.3489% below the fastest1005.7655 package.
These new values must not be replaced by the older974.7035 result when reporting
current speed. The benchmark contract remains matched;the decline is observed,
but its resource-level cause remains unattributed. This is evidence against
userspace rollback as a sufficient remedy,not proof that the libraries have
identical performance or that historical clocks/contention were reproduced.
Decode implementation,configuration and1256-slot placement remain unchanged.
After replay,the485 private vendor/raw/JIT files retire,142631294bytes in total;
the installed apt cache is excluded. The one-use controller and controls also
retire to0697c74e8. The selected original engine and12fixtures remain intact.
Read-only idle settings after closure match the earlier CPU EPP/GPU power
policy audit below. Measure loaded clocks,throttle flags,paging and competing
load separately before assigning a hardware bottleneck. The diagnostic fused-command work subsequently passes its bounded model smoke
and observer-cost gates below;it does not qualify production adoption.

The subsequent loaded-conditions diagnostic uses the same original6e40 and
placement,with1Hz read-only telemetry and stage logging. Twenty synthetic
observer controls pass before two117489-token thinking-on requests. Both
requests,collector and native process close normally;all15 final model hashes
match and no fresh GPU fault appears. These are diagnostic samples,
847.7127/916.5116tok/s,not replacements for the clean911.7371 reference.

On the second request,127 contained samples report GT0 median2750MHz,
nonzero minimum2733MHz,and no zero readings. Throttle reasons are126 `none`
and one `pl2`;package/VRAM maxima are67/76C and mean card power132.83W.
This does not show sustained low GT clocks;1Hz sampling does not exclude brief
limits or establish historical conditions. CPU policy0 reported frequency has median
4616382kHz,not an APERF/MPERF effective-clock measurement. The native process
uses0.990 scheduled core-equivalents,mostly its CPU0 main thread,and incurs no
major faults. Global CPU busy is34.25%,with other CPU/IO work present;no causal
owner attribution follows from those global counters. Available RAM stays
above24GiB and swap counters do not advance in either contained window.

The associated DRM client's CCS busy fraction is98.80% on the second request. It includes
execution stalls and copies on the qualified shared queue;it is not arithmetic
efficiency or a PCIe/GDDR bandwidth measurement. Core clocks,CPU usage and this
busy fraction do not identify the limiting hardware unit. Use the cost-qualified whole-model attribution below and matched hardware-ceiling
work before selecting an optimization. The compact evidence is `execution_conditions` in
[the resource result](../bench/results/2026-10-15-pr6-recovery-audit/prefill-resources-current.json).

The retained fused-profile generation now includes a separately built diagnostic
whole engine,source022652cc5/frozen recipe9852c86a8,with1403 actual compiled inputs
and no uncovered dependency. The original1385 component inputs and3fixtures
remain identical;the original6e40 engine and12fixtures remain selected and intact.
A first smoke plan incorrectly requested16K,which the official CLI rejected
before native launch. Its failed observation remains failed;all owned processes
are confirmed terminated. The corrected32K plan freezes at890e13b4e.

That check completes two fresh39260-token thinking-on requests with0reuse and
24output each. Native/benchmark processes close normally,all15 final model
hashes match,parameter validation is silent,and no freshGPUfault appears.
Its first/subsequent prefill clocks are65299.4/55022.7ms;these charged-observer
samples do not replace the clean117K speed reference. Each request yields four
selected MoE windows,732/737 valid GPU-service records and peak188 of224events.
Host-submit/GPU clock correspondence fails,with22/27 inversions;submit-to-start
queue delays remain unqualified. GPU service does not imply complete device
coverage or hardware saturation. The build is diagnostic,not an isolated
6e40 counterfactual,restoration adoption or sustained decode qualification.
The subsequent frozen observer-cost plan10c7c7498 runs six serial117K processes
in off/on/on/off/off/on order,two fresh requests each. All12requests and six
processes close normally,all15 final model hashes match,and no freshGPUfault
appears. Subsequent-request on/off time ratios are1.00501452,1.00539514 and
1.01639250;their geometric mean1.00892029 passes the prospective maximum1.02
and every-pair maximum1.05. Eight host controls pass. This bounds observed total
observer cost to0.892% for this binary/workload;it does not replace the clean
original6e40 reference or validate cross-domain submit-to-start queue delays.

All six on-arm requests reconcile96MoE windows and four service-profile windows.
Subsequent requests use124.30–126.14s main-thread CPU,including78.82–79.03s in
explicit GPU waits. Source-level partitioning finds another13.21–13.30s inside
the ordinary-host4-byte D2H `q.memcpy` enqueue;the explicit status drain takes
about0.02s. That API may wait internally. Do not classify its time as host
arithmetic or claim a gain by moving it into an explicit wait. CPU expert
grouping is about0.27s. Sampled expert-copy payload rates usually reach about
6.2GB/s,with occasional slow events;16samples per request do not measure total
physical bus traffic or establish a bandwidth ceiling. Requested expert uploads
are about38.67GB per request. Queue intervals remain nonexclusive.

The next discriminating calibration compares ordinary mapped host memory,
host-USM and a bounded CPU staging ring,and isolates4-byte D2H after queued GPU
work. Separate API-call CPU/wall time,explicit waits and GPU service. The
historical9.1% cause remains unattributed;neither this diagnostic nor a small
fixture qualifies a prefill restoration or sustained decode. Compact samples,
source boundaries and limitations are in `attribution_after_observer_cost` and
`observer_cost` in the resource result.

The bounded transfer calibration freezes its build/device plans at98482401a/
bfb62c456. Native source isd7737c53e,controllerd851587bd;679 actual compiled
inputs are covered. One traced smoke and three parameter-validated processes
close normally,with bytewise inputs/outputs/guards,provider and fault checks
passing. Each arm has a separate first-use batch and five subsequent batches
per process. These are synthetic read-only mapped/host-USM sources,64expert-sized
segments and one in-order production Runtime queue,not a model comparison.

| Payload bytes | Ordinary host,three process medians GB/s | Host USM | Host staging ring |
| ---: | --- | --- | --- |
| 1971200 | 3.4433 / 3.4874 / 3.3220 | 6.1515 / 6.2100 / 6.2175 | 6.3056 / 6.2931 / 6.2931 |
| 2662400 | 3.7559 / 1.9710 / 3.9206 | 6.2109 / 6.3008 / 6.3040 | 6.3344 / 6.3553 / 6.3383 |

These whole-wall requested-payload rates include staging,enqueue and completion.
All144transfer batches pass the10ms wall/device-span floors;slow samples remain
in the compact result. Device-copy service usually approaches6.1–6.4GB/s,but
ordinary-host handling loses additional wall time and sometimes has slow device
service too. This identifies a local transfer-path loss,not hardware-wide bus
saturation or the historical9.1% cause. The ring does not demonstrate a gain
over directHostUSM;its purpose is bounded staging of a much larger ordinary-host
model. D2D whole-wall process medians span117.9–160.3GB/s requested payload;
that is not measured GDDR traffic or an arithmetic ceiling.

All36queued4-byte status batches are ineligible:producer service is about0.241ms,
below the prospectively required1ms. Idle status eligibility alone has no wall
floor/timer-overhead gate for fine latency claims. Preserve this limitation;
there is no qualified status speedup or model advancement. A separately frozen
longer-producer/timer-qualified plan is needed for quantitative status attribution.
The next proposed optimization is a small,event-owned host-USM staging ring in
long-prefill expert upload,with actual production dependency-path/numerical gates
before model evaluation. Decode implementation,configuration and1256placement
remain fixed. The clean117K speed reference remains911.7371tok/s.

After extraction and independent review,456closed raw/build/review/draft files
retire,119638008bytes,with a path/hash/reason manifest. The completed whole
profiling engine and command-profile executable retire too. The selected6e40
engine and12fixtures remain intact. Only two GU/Down component executables,
original component record,1385-input manifest and generated header remain for
the next bounded shape/dtype/dispatch audit and service-ceiling comparison.
Confirm their exact operand precision and production applicability before
freezing that plan;component throughput alone cannot establish a whole-model
arithmetic ceiling. Review this five-file subset at that comparison or abandonment.
Closed one-use build/cost/calibration controllers retire to their original source
commits;the new transfer fixture source remains for the immediate staging/status
work. No routine raw API trace or completed observer cohort remains.

The current defaultoff candidate stages long-prefill private-bank uploads
through four bounded host-USM slots. Source8f9abeb99 introduced it;de083f00d
corrects guard/destructor/error retirement to wait the actually bound queue.
Copy-event completion protects each host slot before CPU overwrite;the same
in-order queue orders GPU consumers and scratch-slot reuse. Decode and short
prefill keep their implementation,configuration and1256-slot placement.
No new whole engine or model has run with this candidate.

The initial compute-queue cohort reports31–33% bank-path reductions,but does
not cover the accepted dedicated queue. Its detailed samples and the failed
V2dedicated-queue cohort restore from617e1845a;their outcomes and limitations
remain in the compact current result. V2fails both timing eligibility and a
per-process performance gate. Cold timing-marker setup is subsequently
separated and recorded,with five subsequent empty controls and nine subsequent
arm pairs per payload/process. This changes component instrumentation and
repetition,not the matched117K model benchmark contract.

The former claim that the installed LLVM SDK was independent is unsupported:
its source has17dirty paths with XeStrata modification notices. Its exact
build-time dirty snapshot is not recovered. No SDK patch is imported;new builds
use verified Intel oneAPI2026.2.0-863 packages and UMF1.2.0-116. Original speed
and numerical/fault results keep their status;this finding does not establish
the historical decline's cause. The original engine/fixtures and successful
runtime record remain references while the new runtime is qualified separately.

The official fixture builds from617e1845a with724actual compiled inputs and
10945pinned vendor/host files. V3/V4host attempts fail for a loader-path alias
and an undeclared libc librt dependency,respectively;their original failures
remain recorded. V4actually exits0 but its observer is incomplete;actual
ownership is reconciled independently. V5predeclares the complete DT_NEEDED
closure and reuses the unchanged native build after source/recipe recovery.
It passes the host canary,two traced custom-queue smokes,large512-expert
validation and three independent benchmark processes. All120batches pass
contents,guards,maps,counters,10ms wall/queue floors,timer-p99<=10us and
empty-control<=5% gates. Parameter validation is silent;no fresh fault occurs.
Each batch has641uploads,383hits and2048full-data readers through the actual
private-bank entrypoint. The maintained target is `sycl_expert_bank_test`.

| Payload bytes | Three staged/ordinary subsequent-median time ratios | Geometric mean | Performance gate |
| ---: | --- | ---: | --- |
| 1971200 | 0.734182 / 0.712245 / 0.720932 | 0.722397 | PASS |
| 2662400 | 0.943385 / 1.049833 / 0.525888 | 0.804577 | FAIL:one process slower |

The required bank screen fails despite both descriptive geometric means being
lower. This candidate cannot advance to dequant/GEMM or model trials. Keep all
120samples,including the slower process;do not dismiss it as noise. Next use
a fresh bounded attribution with observeroff/on controls to separate CPU
memcpy preparation,submission,time inside waits and sampled GPUcopy/read
service. Nonexclusive queue intervals and total thread CPU do not identify
the limiting hardware unit. The official runtime qualification covers only
this copy/private-bank component;legacy6e40 ABI/GEMM,model/decode and physical
262144context remain unqualified for it. The clean117K model reference stays
911.7371tok/s;the historical9.0673% decline remains unattributed.

The native SDK snapshot has a named next consumer:official-runtime original
fixture ABI/GEMM qualification and the next attributed fixture build. Retire
closed component binaries/objects and routine raw output after source recovery;
keep the original13assets and the separately named GU/Down ceiling subset.
Actual-format pipeline work,after a new bank performance pass,uses installed
layer2 IQ2_S(type22) gate/up plus IQ4_NL(type20) down,1971200bytes,and layer47
IQ4_XS(type23) gate/up plus IQ4_NL down,2662400bytes. Preserve FP16 matrix
operands,BF16 GU/Down outputs and BF16 residual storage as distinct contracts.
Only separate long/agent/thinking,decode-nondecline/handoff and physical-context
gates can qualify a model route.

The earlier[NEO-only comparison](../bench/results/2026-10-15-pr6-recovery-audit/prefill-runtime-comparison-current.json)
uses exact cached packages in private directories. Both arms explicitly select
the driver before exec and verify its actual mapping. Other dependencies,the
selected engine and1256-slot decode configuration stay fixed. Six host negative
controls and three traced/validated submission/graph cases per driver pass
before the six official117K processes,new/old/old/new/new/old:

| Process/NEO | First use,tok/s | Subsequent-three median | Subsequent range |
| --- | ---: | ---: | ---: |
| 0/new | 902.6631 | 974.1458 | 973.9560..974.1757 |
| 1/old | 898.3727 | 972.1316 | 967.2392..972.8995 |
| 2/old | 903.1516 | 972.7553 | 971.8453..975.5265 |
| 3/new | 908.0447 | 973.0429 | 971.5680..974.4084 |
| 4/new | 901.8863 | 967.3117 | 965.9478..976.0403 |
| 5/old | 900.6598 | 971.3511 | 969.2276..972.4051 |

Old/new paired ratios0.99793227,0.99970442 and1.00417596 have geometric mean
1.00060077. Neither prospective1% faster screen passes;the signs are mixed.
All24 source clocks replay,the15 final model hashes match,all six native
processes exit normally,parameter validation stays silent,and no fresh fault
appears. Original9853 source inputs,the selected13 artifacts and33 runtime
ELFs recheck after closure. Full evidence is archived at8573fa7fb;controller and
controls recover fromd53e15bda.

Old NEO alone therefore did not restore historical1003tok/s under these current
dependencies. This is not proof that NEO has no effect or that runtime interactions
are excluded. The original matched117K difference remains unattributed. No
driver,inference,decode configuration or placement change is adopted. Next
isolate current prefill ownership/loan/handoff costs and the limiting hardware;
an owned128 diagnostic is not permission to shrink production decode.
Use the same selected6e40 binary and current qualified runtime for that diagnostic:
changing both the source package and placement again would leave the cause
confounded. Audit owned/borrowed admission and actual allocation geometry before
launch,and keep clean speed separate from instrumented host/device/resource
attribution. The comparison's `placement_diagnostic` subrecord now contains that
closed observed same-binary work. Borrowed1256 keeps lease1;owned128 uses independent
allocations and lease0,as required by native admission. It is a diagnostic,
not a production decode placement. Both use trace and passive CPU/CCS/frequency
sampling,so they do not replace clean speed or repeated adoption comparisons.

Both processes pass all four117489-token requests,normal native/collector
closure,original-source/provider checks and silent parameter validation,with
no fresh fault. All15 final model digests match. First use is901.6385tok/s for
borrowed1256 and903.3106 for owned128;the subsequent-three medians are972.9914
and975.0545 respectively,owned/borrowed1.00212042. The completed job/dequant
counts match exactly between arms. Borrowed refill takes390.4..515.2ms,under0.5%
of these request totals. Reverting ownership alone therefore does not restore
historical1003tok/s in this observed comparison;do not call the0.212% descriptive
difference a repeated clean gain or adopt the128-slot decode layout.
The client-window CPU/CCS and reported frequency/paging samples are retained
individually. Busy cycles include stalls,CPU includes polling,and historical
contemporaneous frequency/pressure controls are absent;neither these samples
nor the unchanged job counts establish hardware saturation or exclude host
effects on the old/current difference.

For the subsequent117K requests,client-window CCS busy fractions are about
99.1..99.3% in both arms and scheduled CPU is about0.99 core;reported GT0 median
is2750MHz. The shorter first-use39K gate below instead reports83.2% CCS and
0.95 core. Do not mix those extents or interpret either busy fraction as XMX/FLOP
utilization. The current observation still lacks stage-aligned GPU service,
physical PCIe/VRAM traffic and disjoint host-work/wait attribution. Those are the
next resource-attribution gaps;the proposed scratch candidate is not an
established explanation or fix for the historical decline.

After independent replay of all eight request clocks and resource summaries,
both closed observation roots are retired:222 files,47728246B,with the external
path/hash/reason deletion manifest recorded in the compact result. The single-use
controller and controls recover from`f573fc00d`,the frozen recipe from`f678f3ffd`,
and the full closed compact result from`136fe2b9d`. The selected engine and its
twelve fixtures remain unchanged. This cleanup does not resolve the separate
oneMKL coverage consumer or the14 older provenance holds.

The39K owned gate exits normally and restores all48 dense layers. Its original
collection remains failed because the parser required two wholly contained
intervals in the short decode tail. Closed-data analysis at`8c921642b` scopes the
question to pre-first-output,with the original9853 source inputs,13 artifacts,
33 runtime files and15 model identities rechecked. The parser still validates
the complete capture and returns no decode attribution in this scope;15 host
reader tests include rejecting bad identities,coverage,counters and a CPU
rollback after first output. No GPU rerun or original-status replacement was
used. This gate does not qualify117K capacity or physical262144 execution.

A possible subsequent prefill-only candidate is temporal scratch reuse in
`MoeAccumPlan`;it is **not implemented or selected**. Current mode10 splits117488
positions at81920 and records about37300 expert jobs across96 windows. In
`moe_window.inc`,router logits precede routed jobs;gathered input dies before
Down output;shared output follows the checked expert/status drain. These four
workspace planes could share one physical arena if every same-queue dependency,
kernel span and cancellation/failure lifetime remains valid. Keep FP32
accumulation,existing arithmetic and decode unchanged. Source-only byte counting
for the existing1180698624-byte region gives:

| Window rows | Existing region+mixed-tail margin | Proposed four-plane alias margin |
| --- | ---: | ---: |
| 81920 | 37680640B | 138344704B |
| 90112 | −89623040B | 11041024B |
| 98304 | −216926720B | −116262656B |

The alias saves100664064B including three obsolete reservations. Thus an11-chunk
90112-row window is a bounded candidate without another VRAM allocation;
98304 is not admitted by this calculation. This is not a runtime allocation or
speed result. Before implementation,check every planner/route bound,the fixed
jobs-per-expert histogram capacity,per-phase canaries and all scalar/fused
consumers;the existing11-entry histogram would be too small for11 jobs.
After placement diagnosis,consider an opt-in candidate only with those bounds,
device numerical/guard/drain tests,actual117K admission and clean whole-model
evidence. Reduced job/dequant counts are a mechanism to measure,not proof of a
CPU/GPU/PCIe bottleneck or a predicted throughput gain.

Earlier progress reports also contain33K/39K and profiler-charged results.
Those differ from117K subsequent-request speed. The matched comparison above
shows that measurement-condition differences do not explain the entire decline.

The GMM22.10.1→22.10.2 exact source interval is one commit,
[`733c91a19baa65c40f4b1cf95a3991cc8c84d866`](https://github.com/intel/gmmlib/commit/733c91a19baa65c40f4b1cf95a3991cc8c84d866),
parent757922744cc32f247ab4ac31416c32edb57624c7. It extends a PAT-selection
condition to OCL system-memory buffers. That condition also needs
`FtrAppTransientCaching` and transient eligibility. Counterevidence:upstream NEO
4e308a10b232dbb22a2a5c2cdb32eb2578577866 and
c9fe9ecbe5c137657c34f2ae0bea35123afe35ac zero-initialize the GMM SKU table and
neither source tree sets/transfers this flag;GMM's only assignment is in its
unit-test setup. The source therefore does not establish that the changed
condition executes on this B570. It is not an attributed performance cause or
proof of the installed package's runtime flag value.

The exact old GMM was recovered from its primary
[Launchpad build](https://api.launchpad.net/1.0/~kobuk-team/+archive/ubuntu/intel-graphics-testing/+build/33524966).
Private186038-byte package:
`prefill-runtime-candidates/libigdgmm12_22.10.1-1~24.04~ppa1_amd64.deb` under
the shared external state root;package SHA256
2c68fcb150cba1b679063266cdf0054f26f18af90de71d6d3b614eaf95c98f80,
payload SHA256a7904f18d8d59c7128dec4d6f22ae2b2e8a51f91f6946ac4e68b1ec6e0a7a6f3
matches the historical runtime manifest. It was never loaded. The GMM-only
crossover is not advanced because the changed condition lacks source-level
enabling evidence in these NEO versions;that does not prove the installed
packages have no performance effect. At cohort closure the unused package and
both private NEO extracts,normal traces/JIT and closed transaction inventory
were retired:470 files,202522512bytes. The compact comparison records the
external path/hash/reason manifest. The single-use controller and its controls
also retire from the active tree;their original source remains reachable at
d53e15bda. The selected engine and12 fixtures remain intact. This does not
reconcile the14 older provenance holds or retire the separate oneMKL probe
and capture still needed for graph/native-work coverage.

Between native processes at20261025T16:06UTC,CPU policy0 used
`amd-pstate-epp`,governor`powersave`,EPP`power`,maximum4654288kHz.
The GPU reported base power profile,GT0 maximum2750MHz and170W power cap.
These settings were only read,not changed. A targeted original-session search
also recovered the same governor,EPP,min/max and driver in a20261009T23:24UTC
host audit,and `powersave` again in a20261021 frequency snapshot. Those are not
contemporaneous controls for both117K cohorts,but they counter the hypothesis
that this policy was newly introduced. They do not explain the decline.
The later PCIe hierarchy read
matches the20261018 capability record:16GT/s x4 upstream and2.5GT/s x1 at the
endpoint/downstream functions. These fields do not prove physical wiring,
actual transfer throughput or saturation. Active frequency,paging,contention
and current copy-throughput effects remain unmeasured.

## Current MTP attribution

The [private resident-MTP build](../bench/results/2026-10-15-pr6-recovery-audit/mtp-resident-control-current.json)
regenerates6e40 with only two declared diagnostic translation units changed.
Source/admission checks,1410 compiled-input identities,32 native pre-device
refusals, two traced device fixtures and two fresh short-model processes passed.
The [short-model qualification](../bench/results/2026-10-15-pr6-recovery-audit/mtp-resident-serve-current.json)
checks actual prompt paths, accepted-prefix positions, target windows, draft
rounds, first-init validation and normal closure. It is not speed or adoption.

The [official net-cost pilot](../bench/results/2026-10-15-pr6-recovery-audit/mtp-net-bench-current.json)
uses two fresh processes, cap0 then1, on that same binary and unchanged original
prefill/configuration/1256 slots. Cap0 means the normal MTP4 route; cap1 omits
generation catch-up/proposals and uses T1 target windows. MTP weights, KV and
prompt precompute remain resident. This is not complete MTP-off or completely
allocation-matched. Both thinking-on v4 workloads have at least32K fresh input,
1K generated tokens including thought,512 answer characters and10s clocks.
The explicit speed-only client mode permits valid zero counters without calling
them agreement/MTP-classification evidence.

Both processes close normally with all15 final model hashes unchanged, no fresh
faults and exact request/route/clock/resource replay. The observed single samples
are:

| Workload | MTP4 generated / exact tok/s | No decode draft generated / exact tok/s |
| --- | ---: | ---: |
| Code,33214 fresh input | 3611 /14.884 | 3126 /11.306 |
| Prose,33288 fresh input | 8638 /9.510 | 10983 /11.190 |

No-draft/MTP4 ratios are0.759652 for code and1.176653 for prose. These are
observer-charged descriptive readings with different generated lengths, not
repeated clean gains or policy adoption. Cap1 has zero offered/accepted drafts
and zero actual draft rounds; its agreement/classification eligibility remains
false while its explicit speed eligibility passes. Existing selected6e40,
prefill/decode configuration and placement remain unchanged.
Stage/parameter/cache observers and the bounded1s CPU/CCS reader remain charged.
Each process has its same-library65536-pair C++ clock screen; endpoint/rounding
must meet0.01% at the10s duration floor. Observed timer overhead is not a universal
bound and is not subtracted. Replay work/clocks/resources before raw retirement.

The [fixed MTP2 route check](../bench/results/2026-10-15-pr6-recovery-audit/mtp2-route-current.json)
uses the exact selected6e40 binary, retaining allocated`--spec 4` and setting
`--mtp-max-t 2`. Allocating only`--spec 2` would miss the existing MTP expert
release contract. Ten host corruption controls and a fully traced ten-case
same-build GPU fixture passed before one fresh model process. Its four synthetic
inputs33/257/16385/33 produced16/2/2/16 tokens with valid length/EOS finishes.
The16384-cell prefill restored the checked cache payload/map,48 dense layers and
MTP experts. The final request replayed the recaptured T1/T2 draft graphs with
zero chain calls and valid bounded host accounting. The process exited normally,
all15 model hashes matched and no new driver fault appeared. This qualifies
short-route wiring and owner restoration only. Clean repeated speed, task quality
and full-context gates remain open; no production prefill/decode setting changed.
The original [clean width comparison](../bench/results/2026-10-15-pr6-recovery-audit/mtp-width-clean-current.json)
was frozen for nine fresh processes in balanced blocks2/4/1,4/1/2,1/2/4.
Each uses four official warmups and both thinking-on v4 long-generation workloads.
Widths2/4 use selected6e40; width1 uses the existing resident private control.
Native trace/stage/work observers are disabled; qualified runtime safety and
preexec parameter validation remain. Native arguments and actual aggregate
offered/accepted/emitted counts bind the source-fixed widths; this is not a
per-window trace or hardware attribution. Four host corruption/selection controls
and two clock controls pass, and the same-library clock probe passes its screen.
The first width2 process closed with a failed workload gate: code generated3415
tokens at exact13.285064tok/s, but prose spent the16384-token output cap in
thinking and produced no answer body. The benchmark exited1; the native engine
exited0, its owners closed and no fresh driver fault appeared. The other eight
processes remain unstarted under the prospective first-failure rule. Configured
context131072 was not filled; this was output-budget exhaustion. All15 model
hashes passed after closure. Full original plan and failure recover at`6c5857113`;
the next plan retains the useful failed samples without reclassifying them.
The maintained client supports an explicit32768-output budget, retaining
thinking and the answer-body gate, with actual input/output context checks.
The next fresh cohort uses balanced4/2/1,2/1/4,1/4/2 blocks, starting with the
unchanged MTP4 reference. It stops on another capped thinking-only output;
it does not automatically increase the budget. Exact inputs and generated plan
are committed; all64 host controls and the same-provider clock screen pass.
The first fresh MTP4 process has closed normally with both workload gates passed:
code read33209/generated3091, prompt51452.3ms and decode208381.7ms;
prose read33283/generated23179, prompt43076.7ms and decode1964048.6ms.
Both include thinking and an answer body. Draft acceptance is0.848624/0.572026;
the prose sample does not meet the predeclared weak-reference ceiling0.40.
Keep that classification failure separate from successful execution and useful
speed samples. The next width2 process also closes normally, with code3415 and
prose19750 generated tokens. Its first-pair decode ratios against width4 are
0.888372/1.079073; prefill ratios are1.000354/1.001118.
Code misses no-decline; the single faster prose sample is not a repeated gain.
The resident-no-decode-draft control also closes normally: code3064 and
prose26209 generated tokens, at exact11.334027/11.075995tok/s. Its first-pair
decode ratios are0.764091/0.938513, and prefill ratios0.999815/1.001409.
It passes the explicit speed-only contract with no proposals; draft-agreement
eligibility remains false and acceptance is undefined. The second width2 process
also closes normally, generating3415/19750 code/prose tokens with decode
times258410.8/1555581.7ms. Its block's width4 reference
has not run, so this does not supply the next paired comparison.
All four closed native/benchmark exits are0, ownership is closed and no
fresh driver fault is recorded. No route or placement is adopted.
The cohort is now closed incomplete: process4 was refused before native/owned
measurement launch after the system's2026-10-24 23:02JST NEO/OpenCL/Level Zero
upgrade from26.31.39395.16 to26.35.39758.17. Processes4..8 remain unstarted.
The first refusal is the historical SDK's changed`metee.h`; two OpenCL headers
also changed. All1410 actual private compiled inputs still match, and all15
full model hashes match after closure. This does not qualify execution on the
new runtime. Keep the old four samples; do not refresh their pins or combine
new-runtime samples with them. The bounded new-runtime device and model gates
below pass independently. Further width comparison is deferred until actual
CPU-service attribution and a valid weak-MTP workload justify a fresh plan.
Closedv1 logs/JIT, its superseded clock probe and eight unstarted plans retire
after exact188-input and clock-recipe recovery:108 files,14,594,530bytes.
Its intended advancement rules remain three nondeclining paired decode ratios
and geomean above1.02, with independent prefill no-decline in all three pairs.
No clean repeated gain, model adoption or70tok/s qualification follows.
The closed launchers and protocol corruption controls recover at`b61f376c2`.
Their successful raw logs, temporary source snapshot and consumed scope review
are retired after route/observer replay. The private13 assets and old width
raw output are now retired:888 files,153903357bytes. Exact recovery verifies
10 builder inputs,551 transformed project sources,9311 dependency inputs,the
generated header,all1410 actual compiled inputs and13 asset hashes. The old
width's188 benchmark inputs and clock recipe also recover. Its four samples
and admission failure remain unchanged. Keep original6e40 plus12 fixtures;
legacy14 provenance holds remain separate. A later width experiment can recover
the private recipe from`26d484985`; it does not retain an extra engine meanwhile.

The [new-runtime device gate](../bench/results/2026-10-15-pr6-recovery-audit/runtime-requalification-current.json)
passes six host mapping controls, then four ordered native checks: host-only UR
canary, USM/copy/event submission, ordinary graph profile0 and profile1. All five
owned commands close at exit0; parameter-validation logs are empty and no fresh
driver fault is recorded. Executable mappings bind the three device checks to
NEO`libze_intel_gpu.so.1.17.39758`. The candidate manifest explicitly accounts
for the changed NEO,IGDML,GMM and zlib; old runtime/hash records stay unchanged.
The earlier observer incorrectly treated read-only`ld.so.cache` as a library;
that failed run and its actual process-closure reconciliation recover at
`58e353eaa`, and the new host controls cover it. This qualifies these small
fixtures only. Model correctness,clean speed and ordinary runtime destruction
remain unqualified on the new runtime; keep the retained UR reference until exit.
The31 successful/resolved diagnostic files (1,118,277bytes) retire after exact
8/7-input source and recipe recovery. The compact gate retains the33-file
provider manifest and per-case provider/count checks for model qualification;
repeated observed path lists recover from`c2d8a8c9f` if a concrete question needs them.

The [bounded model gate](../bench/results/2026-10-15-pr6-recovery-audit/runtime-model-current.json)
passes on the new runtime with the original engine and placement through
official`bench.sh`:74-token warmup,then39265 fresh input tokens;each generates24
tokens with thinking enabled. Both native and benchmark close at exit0,
parameter validation is silent and no fresh driver fault appears. Live checks
bind both ends of the long request to the pinned environment and new NEO.
Actual long prefill uses39264 BF16 GPU-residual rows and mode10 MoE windows.
The observed59477.6ms prompt and1960.4ms decode are traced first-use diagnostic
samples, not clean speed or sustained decode evidence. Numerical/task quality,
decode nonregression,physical context,RESTORE and ordinary runtime destruction
remain open; keep old/new runtime samples apart.
All71 successful raw/cache/generated files retire after recovering190 benchmark
inputs,9853 original native inputs and plan`e10d6becc` from reachable Git.
The one-shot controller recovers from source`21f84a11e`; the selected engine and
its same-build fixtures remain intact.

## Attribution limits and outstanding gates

The limiting hardware resource remains incompletely attributed. CPU-pool and
GPU-reach wall scopes dominate the current decode route. CPU-pool wall
includes scheduling/waits; scheduled CPU includes idle polling. GPU-reach wait is
host wall time, while CCS busy cycles include stalls and possibly shared clients.
Neither a saturated host thread nor a GPU busy percentage proves arithmetic
saturation. Stage clocks are nested. Expert-copy counters cover only their named
route, not all physical PCIe traffic. Keep physical PCIe/VRAM traffic, CPU issued
instructions and compatible-shape hardware ceilings unknown where unmeasured.

The now-retired CPU attribution engine linked from the exact original-plus-observer
source. All1402 actual compiler inputs match their prospective pins, with no
current-worktree source leakage or changed CPU archive/fixture/configuration.
Its same-build submission and graph profile0/profile1 checks pass in three
separate owned processes under tracing and parameter validation. All close at
exit0; validation is silent, the updated NEO is observed and no fresh fault
appears. These fixtures do not invoke the model or qualify observer overhead,
decode nonregression,clean speed or ordinary runtime destruction.
Its
[host collector gate](../bench/results/2026-10-15-pr6-recovery-audit/cpu-service-current.json)
passes owned GCC13.3 compilation and all standalone counter/clock controls, with
three actual project includes matched and both processes closed at exit0.
Its8 raw/fixture files (48,600bytes) and one-shot runner retire after original
source/recipe recovery; the runner recovers at`b05a22c52`. This qualifies neither
the production pool nor a model measurement. The separate
`cpu_service_pool_test` covers the real five-worker pool, mixedNT1..4,
seven format pairs, the96-expert batch boundary, output canaries and leaf-output
parity. Both separate preexec observer-off/on processes now pass all28 cases
with matching output checksum`bb3654e880dfda0d`. The reconstructed source is
original6e40 plus four explicitly scoped observer/test patches; it contains none
of the current tree's unselected layout/full-context changes. All612 actual
compiler inputs match their prospective pins, and configure,build,census and
both test processes close at exit0. This is same-leaf parity and counter/owner
validation, not an independent mathematical oracle or a speed measurement.
That extra build, generated admission inputs and one-shot launchers are now
retired after exact source/recipe recovery. The validated original remains intact.
A bounded official-bench model check now passes:74-token warmup,then39261 fresh
input tokens,each with24 generated thinking tokens. Actual long prefill uses39260
BF16 GPU-residual rows and mode10;decode resumes with1256 expert slots. Both
requests conserve leaf calls,rows,token rows,clocks and six stable thread owners
against their independent same-request census. Native and benchmark exit0,
validation is silent and no fresh fault appears. The long request records3487.672415ms
of summed thread CPU leaf time:GU2414.041345,HQ7.352099,Down1066.278971ms.
These concurrent sums are not decode latency. Its59801.8ms prompt and1832.0ms
decode are traced first-use diagnostics,not clean speed or sustained decode.
All nine original/off/on processes now close normally with33214 fresh input and
3611 generated thinking/answer tokens each. MTP2578/3102 and format/NT work agree;
all15 final model hashes match and no fresh fault appears. The frozen overall
observer qualification **fails**: off/original prefill ratios are1.3014424,
0.9978732 and0.8577035, with the third below0.95. Their geomean1.0366031 cannot
override that individual gate. The first original72606.3ms and third off65073.3ms
remain recorded; their causes are unestablished. Off/original decode geomean
0.9992706 and on/off0.9850033 meet their separate screens, but do not change the
failed whole-comparison outcome. No phase-only adoption or hardware attribution
is qualified retroactively.

The three on arms conserve28 format/NT groups and six stable thread owners.
Their summed thread CPU times are622.722,626.381 and621.633seconds. The final
arm retains six individual CPU totals102.095..105.284seconds for the named
scheduling question. Cumulative balance does not establish per-batch critical
path balance. Concurrent leaf sums include stalls and charged observer cost;
they are not additive decode latency or evidence of arithmetic/RAM saturation.
The next action is independent prefill source restoration with original6e40 and
its decode implementation/configuration/1256 slots retained. Successful
build/device/model raw output is retired after source/recipe recovery. The closed
comparison's remaining833files/126201281bytes, including its extra build,
generated inputs and consumed source-review output, are now deleted. Full
receipts and format/NT tables recover from the pinned Git archive; current results
keep individual samples, conserved work, useful aggregates and the failed decision. Optional
`STRATA_CPU_SERVICE_PROFILE` compilation plus preexec`STRATA_CPU_SERVICE=1`
collects decode-only pool GU/SwiGLU, hidden quantization and Down leaf scopes for
the seven observed format pairs at H2560/FF640/NT1..4. It records calling-thread
CPU and monotonic wall intervals, calls and output-row/token-row counts by format,
width and pool thread. GU rows count the paired gate/up output rows, not two
independent projections. Pool dispatch, claims and barriers, input activation
quantization, routing and GPU work are outside these leaf scopes. Row counts
must reconcile against the independent work census from the same new request
and source before timings are usable; the older63af census is not that check.
The `cpu_service_evidence` reader revalidates that census from the
same native text and binds GU/HQ/Down row and token-row counts to its format/NT
jobs. It also checks request framing and thread/bin clock sums. Its six owned
host controls pass, including rejection of partial/misbound rows and changing
thread owners. This supplies no new production timing.
GU and Down intervals include their local pointer preparation and native format
dispatch, in addition to the selected kernel call.
Summed concurrent worker intervals are not an additive critical path, CPU time
includes memory stalls, and physical RAM traffic remains unknown.

Normal builds compile out this observer. The diagnostic mutex spans only a
native decode pool batch, outside leaf clocks; existing parked-worker acquires
publish worker writes, and mutex release/acquire also publishes the dedicated
dispatch host's hidden-quantization slot to the separate generation reader.
Snapshots run before the decode clock and after final commit. Unsupported
geometry/width/worker, clock errors, counter overflow, slot-owner changes and
inconsistent snapshots invalidate attribution. Observer and batch-lock overhead
remain charged to model execution; bin lookup and counter updates are outside
the reported leaf intervals. Leaf intervals include clock-read cost. The
host `cpu_service_profile_test --clock-screen` completes the same four-read
sequence without a kernel, across five threads and the caller, with65536 samples
each and uncontrolled affinity. All393216 observations pass the nested-clock
checks. Per-lane wall medians are1.132..1.583us and p99 values1.553..1.923us;
the largest wall observation is430.920us. CPU medians are0.551..0.781us.
Report this reference distribution alongside short HQ intervals, without
subtraction or treating it as a universal bound on the full Scope/model overhead.
The new source`f935f0979` host gate also repeats the collector controls and
closes all four owned commands at exit0. All11 source inputs and the original
recipe recover from Git; the original host result/source remain unchanged.
Its12 temporary fixture/log files (70,843bytes) and one-shot runner are retired;
the runner and recipe recover from the pinned source/plan in the host result.
This changes neither the selected original6e40 executable nor
the closed width samples; it is not an inference optimization or a speedup.

The standalone collector and real-pool checks above do not establish model
attribution. The former private comparison generation has retired. This
instrumented source uses original6e40 plus
only the committed observer delta; the current worktree's unselected
layout/full-context changes are excluded. Its overall overhead gate failed;
the extra executable is retired and normal compilation leaves this observer out.
A new scoped observer plan and qualified model census/clock/fault/source checks
are needed before selecting a kernel or placement optimization from leaf timing.

[Client-resource evidence](../bench/results/2026-10-15-pr6-recovery-audit/client-resources-current.json)
and [actual-mix CPU calibration](../bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json)
provide scoped work/activity bounds. CPU perf counters were denied; the metric
query's observation ioctl returned EACCES. Permissions were not changed. The
whole-model Unitrace aggregate includes loading/warmups/prefill/decode and is not
a phase critical path. Its Q6K/QSA source-attribution consumer has already closed:
69 raw files/17,918,087bytes are retired, with compact shape/source results retained.
Do not repeat that analysis or look for its deleted raw table. See
[profiler capability and coverage](../bench/results/2026-10-15-pr6-recovery-audit/cpu-profiler-current.json).

The next prefill observation requires three independent ledgers: route-owner
planned operations, actual API results and collector device executions.
[`work_ledger.py`](../tools/sycl/work_ledger.py) implements the host reconciliation
gate, including graph-child expansion, lifetime generations, rejected submissions,
host-task completion, shapes/requested bytes and drained phase bounds. Its
18 adversarial host controls pass using synthetic records only. Actual UR/PTI
adapters, model-route coverage and observer cost still require qualification
before a model observation. On current NEO, the retained6e40 submission fixture
matches606 kernels,53 copies/55,844 requested bytes,one fill/1,028 bytes and20
barriers across the source/API/device ledgers. All680 GPU executions now also
join distinct,successful UR enqueue instances by their actual caller thread and
raw-clock interval,through exact PTI host/device flow endpoints. The current
typed observer also records606 valid launch geometries and the same requested
copy/fill byte totals. Its3,453 complete call pairs reconcile4 queues,5 allocations
and680 external event reference generations through release. Seven numerical
event-handle values are reused673 times;one allocation value is reused. These
are distinct generations,not680 unique pointer values. Twenty-five adversarial
host controls and a26-call native CPU fixture exercise loss,identity,geometry,
failed-call transitions,references and synthetic graph replay. Four native CPU
controls qualify the standard environment-mutation guard. Native owners and
collector sessions close normally,with no fresh faults. One barrier lacks an
optional host-completion sink. This does not qualify internal driver references,
USM payload-reader completion,application host-memory lifetime or model coverage.

The first guarded GPU startup stopped during`urAdapterGet`,before any enqueue;
that failed result remains archived. The exact SDK source defaults to rewriting
`ZES_ENABLE_SYSMAN` even when it was supplied as0. A separately committed
diagnostic plan sets`UR_L0_ENABLE_SYSMAN_ENV_DEFAULT=0` before exec and passes
without triggering the guard. Dispatcher/subscriber/validation/collector load order
is verified. This does not qualify disabled`zesInit`,ordinary runtime destruction
or a production environment change. Graph callbacks now record parent,command,
syncpoint and replay-call metadata,but actual graph/device identity remains open.
Pinned PTI source gives each replay child a new execution ID while siblings share
the parent append's host timestamp. Name/shape signatures are candidate matches,
not exact node identity. Next,qualify a common parent/group key and child mapping,
host tasks,drained boundaries and observer cost;scope these gates to the actual
called prefill route. Production prefill/decode source and the selected binary
remain unchanged;hardware saturation remains unknown.

Bounded dispatch discovery now passes on selected6e40 with the official117489-token
request. All15 successful`DRAFT_PREFILL` markers use the direct batched path:
14x8192+2800=117488 callback input cells,mode2 drafter placement,zero callback graph fallbacks.
These are input extents;the inner`r0`/needed-window crop and scratch-sized batch
loop still determine actual computed rows and library invocation counts.
The GPU BF16 residual and MoE10 route run,and all48 dense layers restore.
Seven adversarial CPU controls pass,both request gates verify actual providers
and immutable settings,all15 model hashes match,and owners close without a fresh
fault. Both conversation caches were disabled;no snapshot-readback or draft-timer
observer ran. This is callback-route evidence,not complete device coverage,
observer cost,speed,decode preservation or library-internal graph absence.
See`route_discovery` in the same prefill resource result. Check called oneMKL
native-command/host-task submission before extending the device ledger. The
completed fixture uses one known-answer production`Gemm::bf16` call with
all-thread UR,Level Zero append and PTI observation through a checked drain.
Its built probe uses64x320x10240 BF16 inputs and FP32 output stride323,with
full20480-value exact/input/stride/end-guard checks. All1383 actual compiler
inputs are covered. The old broad SDK inventory refused3 updated system
headers before configure;the separate prospective build pins those updates,
and none occurs in this probe's actual compiled closure. The original refusal
stays failed. The fresh native process subsequently passes the full oracle and
guards,checked drain,actual provider/namespace checks and silent parameter
validation,and closes normally without a fresh fault. Original capture source
is`ccd2a46ca`,with its outcome at`cf00172d7`;the selected engine is unchanged.

Closed-data analysis source`8df13f8f2` and plan`167f50fd2` independently join all
ten observed GPU executions to ten distinct successful UR enqueue instances
through actual PTI append flows,caller thread and clock intervals:two kernel
appends and eight copies. The two`urQueueIsGraphCaptureEnabledExp` calls are
capability queries;no graph lifecycle/submission is observed in this capture.
Thus the original`graph_API_seen=true` does not establish graph execution.
The queries total23.213041ms of charged host wall time in this first-use,
instrumented call;this is not repeated model overhead or a clean speed sample.
Twenty-seven subscriber/link and nine envelope/classification host controls pass.
The buffer read and write still lack typed queue/extent/lifetime metadata;
independent library work planning and model-route/observer-cost coverage also
remain open. These exact joins do not qualify complete model attribution or
identify the cause of the historical prefill decline. Preserve the original
native-passed/coverage-pending status and retrieve further raw detail only for
those named gaps.

The source`05f0f7af4` buffer observer now passes35 parser/link controls and
seven native CPU processes,including21 synthetic buffer/query calls with five
rejected invalid-pointee cases. The original26-call fixture and four mutation
guards remain intact. Its own native library/driver are distinct from the
unchanged wrapper and selected engine. Capture source`f5e4bef35`,frozen plan
`0da73a505`,reuses the original wrapper only after recovering9865 original
project/dependency inputs,the original recipe and all1383 compiled inputs;
changed uncompiled observer/tests are outside that native artifact.
The new real GPU call passes all20480 values,input/output guards,actual
provider checks and silent parameter validation,with normal owner/session
closure and no fresh fault. All ten GPU executions join distinct successful
UR calls. Typed fields now cover both4-byte buffers,8 requested buffer-copy
bytes and15894528 explicit USM-copy bytes. External references close for two
queues,three USM allocations,two buffers and ten event generations;seven event
values are reused. Both actual graph-query results are false. These observations
do not establish driver-internal lifetimes,host-payload ownership,an independent
library work plan,complete model coverage,observer cost or clean speed.

For the next long-prefill resource observation,a separate source obstacle is
now explicit:`gemm16.cpp` rejects command-service profiling whenever selected
GU or Down fusion is enabled. An unfused profile therefore changes the measured
processing. Extend sparse event attribution to those actual fused kernels and
qualify the event/count/drain/numerical/observer-cost contract before using it
on the selected MoE route. This is Main's next attribution task,not a claimed
speedup or an explanation for the historical decline. The library-ledger gap
above remains a separate scope;do not substitute a general graph survey for
measuring the actual selected fusion route.
Both fused calls already publish their event through`sycl_backend::finish`;
do not register it twice. The missing contract is correct fused-operation
classification and sampling. The current last row bin spans the2334-row fusion
threshold;simply adding fused names can leave an eligible fused/unfused stratum
unsampled. Split that boundary,check the fixed event/host-record capacities,and
test exact population/sample conservation,retirement on errors and real fused
GEMM results before removing the admission guard. The prospective version3
extension now passes its bounded component-device gate:9 row bins,224
events per window and scoped relabeling around the existing finish calls.
Profiling with fusion still requires a separate explicit qualification opt-in.
Eight CPU parser rejection controls pass,including signed submit-clock
diagnostics and group/service conservation. Ten fresh traced,parameter-validated
GPU processes close with no fresh fault. Synthetic populations retain
195 of639 eligible events with the fused large bin and203 of655 with separate operations,including
four ranked expert-transfer samples from72 actual compatibility copies. The
bad-tag and event-capacity profiles fail as intended while their outputs and
guards pass;injected failure drains before owner release and queries no
unfinished events. GU production-hook checks at2333,2334 and8192 rows have
zero half-ULP difference off/on;Down covers those rows and irregular2334 rows
within a16384-token plan with bitwise-equal full outputs. All input and guard
checks pass. Replay reproduces all ten outcomes,including the intentional
exit1. The128KiB bound covers static host records,not total process RSS or
runtime-owned event storage. The original selected6e40 and12fixtures remain unchanged.
This is attribution infrastructure,not a throughput gain,model qualification or
an explanation for the historical decline. The three fixture binaries build with zero uncovered compiler inputs and the
pinned generated header. A separately pinned engine build,small model gate and
observer-cost qualification remain necessary before a long-input resource profile.
The source has two BF16 GEMM sites per executed inner batch,four projection
sites,and an early no-needed-rows return;do not infer dynamic counts from the
15 callback markers. The existing runtime manifest already pins the L0 v2
adapter;observe its actual mapping/capability rather than infer installed
behavior from the inspected SDK source. Any next research scope must address
one of these actual coverage gaps rather than repeat a generic graph survey.

Basic [profile-off/on graph support](../bench/results/2026-10-15-pr6-recovery-audit/mtp-graph-observer-current.json)
already passed, with exact6e40 reuse checked separately. The closed
[64-window MTP observation](../bench/results/2026-10-15-pr6-recovery-audit/mtp-model-observer-current.json)
measured a31.2083ms/window outer graph envelope; the
[Q6_K head calibration](../bench/results/2026-10-15-pr6-recovery-audit/mtp-head-current.json)
also completed and reached only its intermediate follow-up screen. These are
charged graph envelopes, not individual arithmetic service or a target-verifier
breakdown. Do not repeat basic adapter qualification from an older report's
unknown-support note. After the active clean comparison, check existing
collector qualifications and source mapping for a bounded target/per-operator
collection using the indexed expanded-graph method.

Additional source binding uses unchanged6e40 `verify.cpp`, `native_head.cpp`,
`mtp.cpp` and `native_mmvq.cpp`. The target verifier head has input2560,
output248320 and columns equal to its window T; the standalone NativeHead route
uses one column. The MTP subset has106299 output rows and its completed head
calibration used one column. The prior1035-window code request processed4137
target rows (firstT1,then1034*T4). The aggregate's1047 full-vocabulary calls still
mix request phases and column counts; their mean11.581ms is not a decode-only T4 service.

The Q6K launcher loads each packed weight fragment outside its column loop.
Its inlined apply reconstructs6-bit values, but the compiler may already hoist
that work: C++ source alone does not establish redundant machine instructions.
The existing alternative ROWS2/NW4 route differs from selected ROWS1/NW4;
register pressure, occupancy and actual service remain to be measured. The
`multi_exact` selector is global across callers. A future target-head-only
candidate must select by explicit caller/shape, preserving MTP and both prefill
paths; changing that global is not phase isolation. This is a source-design
question, not a selected kernel optimization or hardware-bottleneck conclusion.
The selected SYCL `sequence.cpp` rejects device mapped-flag polling outright;
its event-completed handoff does not make GPU-busy cycles an arithmetic ceiling.

The separate3,611-output code diagnostic also supplies a nominal CPU weight
budget. That census used native source`63af639ac`, with `STRATA_IQ2_PAIR` unset;
selected6e40 adds the decode-only IQ2_S NT1 pair path and enables it. Historical
job counts and projected pool times do not measure the current dispatch.
Match the historical seven format/NT histograms to the checked blob geometry in
the [archived calibration plan](https://github.com/yqYo1/Strata/blob/1c450997ad10b17d387d86e3e0904ae447924160/bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-plan.json):
`sum(sum(jobs_NT) * blob_bytes)` is2,796,077,900,800bytes across1,370,846
unique CPU jobs, or774,322,321 nominal bytes/output. This already groups
NT1..4 reuse within each job. It is one packed-blob visit per job, excluding
additional activation/output traffic; cache-line reuse, repeated loads and
actual DRAM transactions remain unmeasured. VRAM serves10.7728% of that
trace's target expert entries; this is neither a GPU time fraction nor a
fraction of parameter bytes.

At70tok/s the same nominal work rate would be54.203GB/s. The simple read
reference achieved30.642GB/s, not a physical memory ceiling. If every nominal
byte reached RAM at that rate, weights alone would take25.270ms/output
(39.573tok/s), before other work. Reaching70 at that reference would require
physical bytes at most56.533% of the nominal count, or lower per-output weight
demand; neither is established. Use this conditional budget to test cache,
grouping and CPU-kernel alternatives together. Do not declare RAM saturation,
an impossible70tok/s target or a new clean throughput result from these numbers.

Before any adoption, complete the relevant independent gates:

- Repeated clean thinking-on strong/weak long generation, official long and
  agent-session benchmarks, task quality and prefill/decode state preservation.
- Actual physical262144 fill, final slots/tails, prefill-to-decode/session and
  RESTORE paths. No output space at a completely full context is expected;
  crashes or corruption are not. The retired590f candidate's262120-input plus16
  outputs and scoped RESTORE results do not qualify exact fill or selected6e40.
- Failure-owner, graph/allocation retirement and ordinary runtime teardown.
  Current preexec parameter validation holds a UR reference until process exit;
  its normal exits do not qualify ordinary runtime/provider destruction.
- Source-scoped provenance/lifecycle exceptions. Selected6e40 recovery is separate
  from legacy final-input failures. Use the actual missing inputs, named consumer
  and next review in the linked build results; do not preserve old binaries merely
  because an unrelated ownership or default-merge hold remains open.

Preserve copy-offload avoidance, disabled direct submission, in-order queues
and other qualified settings in [the measurement environment](../tools/sycl/measurement_environment.json).
Set them before native exec and leave them immutable. Keep the embedding service
stopped. No reset/unbind/recovery action is part of a benchmark.

## Upstream and history

Niko1221/Strata is the only upstream. The last actual check was
2026-10-25T21:35:21.222628Z at`4b9e05a2a` (`v0.2.6`),446 commits/406 paths
beyond the prior7ddc17a38 observation. Required non-inference ports and
hardware-tuned unmodified-upstream measurements remain pending;observation is
not integration. Check again around2026-10-27T21:35Z using the project-only sync
skill. These dates reconcile the existing committed upstream-status.json;this
document edit is not another remote check.

Work is on`feature/sycl`. PR180 is mergeable but blocked by squash-only
ruleset24638744; original measured commit identities require a merge commit.
Do not squash, bypass the rule or claim default integration. Measured source
remains reachable on pushed`feature/sycl`.

Closed QSA/GDN/MoE/layout/CPU/cache/profiler narratives, obsolete recipes and
retirement transcripts are archived in the [prior document](https://github.com/yqYo1/Strata/blob/ffff15bde/docs/SYCL_NATIVE16_PREFILL.md)
and their compact result files. Restore exact Git revisions for a concrete
question; this note records the current selection and contracts, not a running
history of every trial. Follow [artifact retention](ARTIFACT_RETENTION.md) at each
boundary and leave collection-report bodies to the indexed research workflow.
