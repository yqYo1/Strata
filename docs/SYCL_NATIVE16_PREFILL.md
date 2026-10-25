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

Hardware is Arc B57010GiB and Ryzen5600X6c/12t. The latest clean prefill comparison
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
reviewed next fixture uses one known-answer production`Gemm::bf16` call with
all-thread UR,Level Zero append and PTI observation through a checked drain.
The source has two BF16 GEMM sites per executed inner batch,four projection
sites,and an early no-needed-rows return;do not infer dynamic counts from the
15 callback markers. The existing runtime manifest already pins the L0 v2
adapter;observe its actual mapping/capability rather than infer installed
behavior from the inspected SDK source. No new research scope is eligible
until that fixture identifies a concrete gap.

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
2026-10-23T21:29:32.552053Z at`7ddc17a38` (`v0.2.3.1`),131 commits/294 paths
beyond011c9e60. Required non-inference ports and hardware-tuned unmodified-upstream
measurements remain pending; observation is not integration. Check again around
2026-10-25T21:29Z using the project-only sync skill.

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
