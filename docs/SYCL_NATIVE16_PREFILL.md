# Native 16-bit long prefill

## Selected implementation and independent phases

The selected engine is the original source
`6e40afc1beddd059ee518617f6c1efce1da14aca`, SHA-256
`22182f1a5e62e5f4259ce49a75ae44afbd988166ec79324816e499ff37ef60e4`.
Its engine and twelve same-build fixtures match their original hashes. All9853
original project/dependency inputs and the build recipe recover from Git.
Restoring that exact package required no recompilation and supplied no new GPU
validation. See [the selected build proof](../bench/results/2026-10-15-pr6-recovery-audit/native-gdn-engine-build.json).

The split9a79generation is retired after recovery of551native/configuration,
9311GGML/TLA and53controller inputs,1400actual compiler inputs and all three
original command/environment owners. Its historical results and failures remain;
735files/130689718bytes,including consumed source audits,are removed. Selected6e40
and its twelve fixtures remain unchanged. Fourteen older provenance holds remain
independent. Legacy compiler provenance is unresolved;no new legacy build runs.

The prospective official2026.2replacement uses
[a pinned constructor](../tools/sycl/official_model_sources.py),not the whole
working tree: selected6e40,qualified split/Down overlays,the independent Top-k
route,official bundler discovery and a prefill-only eight-job checkpoint.
Its554source files have aggregate5f468f91a381399d120eafa9b65baa274ea6b2137176e490f8676e7c1d1f89d0;
source/recipee4617c85c is frozen before compilation. Each checkpoint follows eight
fully submitted jobs on their actual in-order stream. A split window has at most
655jobs,81interval waits and seven tail jobs covered by the existing status drain.
A failed wait does not prove retirement;the outer checked lease retains owners
or fail-stops. This bounds submissions,not elapsed time or native command count,
and does not prove avoidance of the older2048call stall. The official build now
passes configure,compile and dependency extraction with1447compiler inputs,
29observed external linker inputs and seven executables;all owners close.
Engine SHA-256 is152f43036c93b5b6a96e25874ea3055c80d280e8f9cfeb062fa36f57ac5f4354.
The fresh same-build V3component cohort now passes13owned processes with quiet
validation,no new driver fault and normal closure:host canary,compiled Top-k
properties,traced off/on smoke,submission,32dyadic matrix cases,26guards,
64public fallback GUcalls,public GUthreshold routes,original split-Down routes,
and145Top-k numeric cases in each off/on process. GUhas308balanced stage pairs,
135kernel launches and0ULP;each Top-k smoke has six launches. Actual official
providers and immutable environments are checked. Native source remains
e4617c85c;cohort source83590826a. The initial wrong-UR-loader property lookup
failed before a kernel launch;V2was closed before execution to correct nested
stage analysis. Their original failed/unexecuted outcomes remain in the compact
result. The first official-engine model entry now passes bench.sh with32769read
tokens,zero reuse,32768GPU-resident BF16 residual rows and24generated tokens.
All48MoE windows complete18293jobs with2264paired eight-job checkpoints;
the ordinary accumulator is active at this length. Parameter validation is quiet,
the native process exits normally and no fresh driver fault appears. This logged
diagnostic reports583.5tok/s and56.289s client TTFT,not clean speed or sustained
decode evidence. Clean64K/117K comparisons,matched decode preservation,task and
physical262144/RESTORE gates remain pending. See
`fused-moe-service-current.json::official_model_entry_v1`,native sourcee4617c85c,
controller source7493d0487. The earlier22GEMM ABI checks remain separate.

The clean official comparison,controller578f8435f,freezes12fresh processes and
four requests each across65537/117489read tokens. Its first64K off/on pair closes
normally without fresh faults:subsequent-three medians880.8/877.9tok/s and
on/off time ratio1.00330227. This pair does not support a speedup;the remaining
cohort stops at process2's correct route refusal. Its four117489-read requests
finish with native exit0,quiet validation and no fresh fault,but actual placement
uses512slots and73728..81920GPU rows,leaving728432640..896204800residual bytes
in RAM. Split accumulation is inactive. PP698.0/727.7/733.9/733.1and client
TTFT168.568/161.623/160.244/160.424s are diagnostic samples,not the planned
matched Top-k comparison. The all-GPU split budget requires4057356672bytes
including512MiB headroom;first available3822460928leaves234895744bytes missing.
Preserve the failed V1gate and nine unexecuted processes. Resolve capacity
before a new committed comparison;do not relax the route parser or continue
index3. No policy is adopted. See
`fused-moe-service-current.json::official_prefill_speed_v1`.

The joint dense/head release proposal addresses that capacity deficit by
temporarily freeing the521472000-byte target head,plus its optional MTP subset.
It is default-off and not adopted. Its replacement recipe032f4f8a5
pins554native sources,aggregatec3b781a2378a0d208c6ac7852bff505ce36f8c4955ae2956877c0db44dfd7dbc.
Seven executables now compile with1447matched actual compiler inputs and29
observed external link inputs;engine SHA-256 is05c489eab7254417eb0e76a6f5c4c44a14fc837d83dd4056457d0138f522b29e.
The fresh same-build13-process device cohort also passes,with quiet parameter
validation,no new driver fault and normal ownership closure. A fresh same-process
model check now completes two32769-read thinking-on requests,each with32768
GPU-resident BF16 rows and24generated tokens. Each request detaches the521472000B
target head and223227900B draft subset,restores all48dense layers and resumes
short decode. The second request also exercises retirement of the first request's
lazy graphs. Both request windows conserve jobs/checkpoints,parameter validation
is quiet,the native process exits0,model hashes match and no fresh driver fault
appears. Logged PP578.9/687.2and client TTFT56.763/47.807s are first/subsequent
diagnostics,not clean speed or sustained decode evidence. Bytewise head payload
and captured-pointer verification,matched64K/117K speed,decode preservation,
task and physical262144/RESTORE remain open;the candidate is not adopted. These
checks do not inherit the seed's model results. See
`fused-moe-service-current.json::official_prefill_joint_build_v1`,
`official_prefill_joint_device_v1` and `official_joint_model_entry_v1`.
The separate clean head-capacity comparison,controller82fdd969c,freezes12fresh
processes and48thinking-on requests:three reversed-order pairs at each64K/117K
length,Top-k fixed0,only head retirement0/1 changed before native exec. First use
is separate from each process's subsequent-three median. Each length needs all
three paired time ratios below1 and their geometric mean at most0.99;the1000
screen separately requires all three candidate process medians at that length.
The117K control may retain its legitimate512-slot/RAM-tail fallback;the candidate
must admit all117488GPU rows and checked split storage. This is a new capacity
question,not continuation of failed Top-k V1. See `official_joint_speed_v1` in
the compact result. Nine processes close normally without fresh faults. Each
table cell uses its process's subsequent-three median;first-use and all
individual samples remain in the compact result.

| Read tokens | Pair | Head-on tok/s | Head-off tok/s | On/off prompt time | On/off client TTFT seconds |
| --- | --- | --- | --- | --- | --- |
| 117489 | 1 | 967.4 | 728.2 | 0.75280023 | 121.630 / 161.518 |
| 117489 | 2,reverse order | 968.2 | 733.4 | 0.75754313 | 121.523 / 160.366 |
| 117489 | 3,control pending | 908.2 | pending | pending | 129.664 / pending |
| 65537 | 1 | 869.9 | 878.8 | 1.01017066 | 75.480 / 74.722 |
| 65537 | 2,reverse order | 795.0 | 880.4 | 1.10739703 | 82.677 / 74.591 |

All three117K head-on processes admit117488GPU rows,128staged slots and checked split
storage in all requests. The two completed control processes use81920GPU rows first,73728
later,512ordinary slots and728432640/896204800RAM residual bytes. The paired
throughput improvements are32.84%and32.01%. At64K both arms have65536GPU rows,
384staged slots,no RAM residual and matching job counts and requested expert-bank/
routing-index transfer bytes. Request hashes match at both lengths.
The engine prompt clock encloses joint head retirement/restore and the final
expert-cache refill;it is not pure GPU math time. Client TTFT separately ends
at first nonempty reasoning/content output. The64K first pair prevents this
candidate passing that length's predeclared all-three-pairs screen. Both lengths'
first on-process medians are below1000,so neither target screen can pass. The
second64K head-on process has median82438.4ms,10.74%longer than its control;
its slower samples remain eligible original outcomes,without a noise exemption.
The third117K candidate median is129367.1ms;its jobs and requested transfer
counts match the earlier candidate requests. Its lower throughput is retained
without an attribution to noise or clock/thermal behavior. Three further processes still run under
the frozen no-speed-based-stop plan;the117K mechanism decision remains open.
The candidate is unadopted. This does not
identify physical PCIe/VRAM/arithmetic saturation or attribute the64K difference
solely to head-copy cost.
The retained `requested_dequant_H2D_bytes` field is misnamed:the native
`source_H2D_bytes` counter adds `rows*K*4` once per GPU-resident MoE window for
one routing-index map. Actual Mode10 code reads router IDs back with that same
byte count,then uploads two maps totalling `rows*K*8`;the counter omits the other
map and readback. It does not count dequantized weights or RAM-tail fallback
traffic. See `moe_window.inc` router readback,map upload and `owner.source_bytes`,
and `layer_major.inc`'s final counter. This source-semantic correction changes
no original samples or comparison criteria;rename the parser field after the
frozen cohort closes. These remain requested bytes,not measured bus traffic.
The next resource-controller source is isolated on `work/prefill-resource-profile`,
commitf03176660;it is not integrated or model-executed. Static review of this
exact native generation found that `src/prefill/sycl/gemm16.cpp` rejects command
profiling together with GU/Down fusion. The prospective pair therefore retains
both production fusion settings and disables that unsupported observer in both
arms. It measures CPU partition,coarse GPU phase intervals and transition/refill
clocks,without a per-kernel service or hardware-saturation claim. Its CPU-only
`selfcheck` passes two ordinary/split controls,nine rejection cases,the127/128and
2333/2334shape-group boundaries and both fused observer environments. The
historical81920default stays fixed;117488requires an explicit prospective bound.
After the frozen cohort closes,merge the source and prepare/qualify the
clean-versus-observed pair. Retire the temporary branch after its original
commits are also verified reachable from the fork default;PR180 remains blocked.
A closed-boundary inventory after process8 finds readable Xe sysfs frequency,
temperature and energy counters,while xpu-smi reports some corresponding
domains unavailable. These are capability/idle observations,not load-time
evidence or a cause of the speed spread. The prospective source-only
`bmg_sysfs_execution_context_20261031` scope compares counter semantics and a
bounded sampling protocol;it does not repeat the PCIe-hierarchy or metric-pool
surveys,change system settings or qualify hardware saturation.
The unselected e461generation is retired after exact source/recipe/dependency
recovery;its84-provider inventory migrates to the new candidate's pinned runtime
basis. Selected6e40 remains the validated fallback. Independent source review found
that fallible head preparation inside cache-loan activation poisons the parent
on refusal/throw,even before readers retire. The correction prepares before
activation,uses a no-fail retirement callback and explicitly cancels prepared
heads if activation preflight refuses. Original proposal0391301a6 remains
unqualified. The original seven-case host plan closed without execution;
a fresh fifteen-case ASan/UBSan host gate passes315checks on the actual cache
state,the old poison mechanism and corrected cancellation/restore ordering.
Its214actual compiler inputs and28link inputs match the prospective inventory;
both owned processes exit normally. Corrected native source is29f8657ee.
The next constructor overlays only the head/verifier changes and exact prefill
phase section onto the original official candidate;all other code stays pinned.
Native graphs,head tensors,handoff,decode preservation and262144/RESTORE remain separate
gates. See `fused-moe-service-current.json::joint_prefill_heads_v1`.

The experimental `STRATA_SYCL_PREFILL_TOPK_EARLY=1` route selects a separate
exact-completion Top-k kernel only in layer-major prefill. It defaults off,
requires `STRATA_SYCL_SELECT_RADIX4=1` and excludes the cyclic selection arm.
Its body specializes the private component qualified in `1a752d9a3` without
counter instrumentation. The fresh official-SDK public-entry V2 passes all16
processes:9startup policies,3traced smoke cases and145numeric cases in each
off/on process,including the262144selector endpoint and unchanged decode/ref
APIs. No new fault or validation warning appears. This qualifies direct selector
APIs;matched whole-engine speed and decode preservation remain pending after the
official candidate's first32K production-entry diagnostic above.
See `fused-moe-service-current.json::qsa_topk_prefill_entry_v2`. Selected6e40
does not contain the route;the official candidate contains it with full model
qualification still pending.
The original selector source and small-input/decode calls remain unchanged.

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

The source restoration returned nine prefill/planning/compatibility paths
to6e40 and the15 coupled layout/admission/pricing/loan hunks in `generate.cpp`.
Its18 independent diagnostic/observer hunks are preserved byte-for-byte.
The unused streamed-QSA writer, host-span export and fixture are removed.
Decode kernels, configuration and1256-slot placement remain selected as before.
The measured restoration used native source`1baa0547f` and engine SHA prefix`2a128c6a`.
The original6e40 executable remains the selected fallback; restored-source
prefill speed screen now fails; task and full-context qualification is pending.
The current tree also contains unselected upload-staging code from the closed
component trials below. Retire its unused exports/targets at the first unfrozen
boundary,preserving independent ownership fixes. This code is not in the selected
original6e40 executable and does not explain a same-executable speed decline.

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
The selected-entrypoint audit now identifies a separate serve diagnostic:
consume the final input through the first T=1 verifier at position262143,then
suppress output/drafting. Exact-full ordinary cache matching cannot restore
that prefix;test the real snapshot restore routine directly after reset,with
actual snapshot RAM admission and explicit `ple_prev` verification. The prior
unselected `model_capacity_v5` did execute262144 inputs with zero outputs;it
does not qualify selected6e40/QSA or full-session RESTORE. No new build or
device execution follows from this source review.

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
| Materialized routed GU/Down outputs | BF16; fused jobs can avoid these intermediate outputs. The census field `gu_down_storage` describes these outputs, not the dequantized weights. |
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

## Closed prefill-only head-geometry comparison

The selected QSA kernel computes16matrix rows for12real heads. The isolated
M4/WG96 candidate computes exactly12,using six complete16-lane subgroups.
It preserves ordered FP32 online reductions,rounding,K/V packing and full-group
barriers. Public matrix loads/stores through six disjoint subgroup tiles avoid
the deployed backend's unavailable M4coordinate helper. Extra local-memory
accesses and subgroup barriers may offset the removed padded work;their net
effect requires measurement. The compiler's zero candidate spill descriptor
versus nonzero original descriptors is evidence about that build,not a measured
occupancy,bandwidth or throughput gain.

The exact native build is source593d756a6,with the portable constructor at
ba91f4669. Compiled properties,the traced tiny case and11independent oracle
cases at each Page4/Page16pass. Complete BF16 outputs and real-head FP32 online
states match the original;guards,input immutability,pre-main validation and
fault/ownership checks pass. Physical262144component addresses are covered;
this does not qualify a full-model fill,handoff or RESTORE. Production callers,
the selected6e40 engine and decode remain unchanged.

The complete pack+attention comparison is **rejected on speed**. All72samples
pass numerical,clock,provider,ownership and fault gates across three fresh
processes. With8192queries,2051selected cells,Page4and varied nonzero synthetic
operands,the candidate/original wall-time geometric means are1.089772
(32Kcontiguous),1.102496(32Kstrided),1.096115(117488contiguous)and1.007882
(117488strided). Every process is slower in every case;attention-event and
pack+attention-event service also fail. Parameter validation stays enabled on
both arms;these are matched component screens,not clean whole-model speed.
No model trial or adoption follows. Numerical validity and a zero spill
descriptor do not override the failed performance gate. Keep original6e40
and decode unchanged. See`qsa_head_geometry.component_v2` in
[the current resource result](../bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json).
Keep the existing multi-length time approximation until new model observations
justify a change.

The next bounded candidate changed a distinct operation:skip the matrix
accumulator rescale only when all eight head coefficients owned by that
subgroup equal FP32`1.f`exactly. It retained the original M8/WG128geometry,
ordered softmax,state,storage and barriers. No approximate comparison or
first-tile shortcut was used. Properties,traced tiny and Page4/Page16oracle
cases pass. A separate traced M8matrix operator checks128finite FP32states,
including80subnormals and`0x8200`(`2^-134+2^-140`):matrix load/store preserves
every input bit,and multiplying by1matches bypass exactly. The initial edge
fixture's raw-pointer API build refusal remains failed;the corrected explicit
global`multi_ptr`build and fresh gate provide the positive evidence.

The same three-process,72sample timing screen nevertheless **rejects identity
rescale skipping**. Candidate/original wall-time geometric means are1.330049,
1.635727,1.341319and1.465628in the same four cases above;every process and
all wall/attention/combined-service criteria fail. These are valid matched
component measurements,not a model regression or an adopted change.
See`qsa_identity_rescale.performance_v2`in the linked resource result.
Keep6e40selected and do not start a gated model trial. Main moves to bounded
compiler/source optimizations rather than extending failure forensics.

The CPU-only compiler discriminator at source35ebc9b89now closes all three
fresh arms with662actual dependencies each and no GPU execution. ExactWG128
adds metadata only:both normalized normal/state attention bodies match the
control,so no native trial follows. The separate ID-int flag removes five
static64bit integer operations in each body (27/28to22/23) and six lines
(730/756to724/750);floating-point operation counts remain equal. These counts
are not a speedup or hardware-saturation result. Native comparison is eligible
only with overflow-safe range checks and a size_t-compiled fallback. Production
262144context/8192query maxima give pack33554432and attention2097152global
items. The broad public pack API can exceed INT_MAX;never narrow addresses or
silently reject those callers. Keep decode and selected6e40unchanged.

The guarded ID-int component (source16ad0e76e) now passes compiled properties,
traced tiny and11Page4/11Page16oracle cases,including physical262144component
addresses. The six host range-boundary checks pass. FP32online state and BF16
outputs remain bit-identical;guards,input immutability,providers,ownership and
fault gates pass. The overflow-safe dispatcher returns broad ranges to the
separate size_t image before any int enqueue. Huge public allocations and full
model context/state are not exercised by these component tests.

Its three-process72sample full-path timing comparison nevertheless rejects
advancement:wall-time geometric means are0.999454,0.999809,0.999667and0.999726
for32Kcontiguous/strided and117488contiguous/strided. Every metric misses the
predeclared1percent gain;some per-process cases also increase slightly. Static
IR simplification is not a useful throughput gain here. No model trial or
adoption follows,and Main closes this candidate without more failure forensics.

The next candidate is only the32-term normalizer fold:use a parallel FP32
subgroup sum while keeping BF16probabilities,FP32state,PVorder and all other
math unchanged. Orca Luna source research and native Luna-max curation accept
this component experiment against pinned SGLang/vLLM reduction contracts;this
does not require ascending llama.cpp additions or identical free-running text.

The independent prospective host oracle at359a74282passes seven checks. It
uses exact dyadic operands and outward FP32bounds for each tile sum and fused
online update,including subnormals and values lost by ordinary binary64 sums.
The conservative candidate depth is16only after proving16materialized lane
pairs;depth5requires actual balanced-tree evidence. These are operand-specific
mathematical component bounds,not an empirical tensor/task degradation margin.
The source73414ebc2pins the original6e40kernel and changes only that fold,
with matched optional capture stores in both arms. Its CPU build and four
compiled normal/state property queries pass;no new normalizer kernel has been
enqueued. The Page4/Page16lossless fixture at2fb5d95d2also compiles and links
with667actual compiler inputs and zero uncovered link inputs. It captures
commonp,alpha,BF16PVbits,FP32accumulator and pre-cast outputs,and checks seven
invalid trace spans before any submission. The earlier unused fixture plan
was closed after a source change and never built;its status remains separate.

The checker977c54fc9and host oracle now pass15host tests,including a valid
reassociation crossing a BF16midpoint,invalid operand/accumulator changes,
online FMA errors,division/cast errors,zero signs and diagnostic branches.
Those tests exercise the checker,not GPU arithmetic or task accuracy.
Two Orca Luna source investigations and native Luna-max reviews leave the
exact binary's active FP32mode,division bound and BF16conversion rounding
unresolved. Compiler/library source contracts,capability flags and finite
probes are distinct evidence. The candidate is **deferred**,with no numerical
kernel or performance trial;its CPU build is not a failed speed result.
The division review's proposed zero-ULP bound for correctly rounded division
also needs correction:when comparing with an exact real quotient,the normal
RNE bound is one-half ULP,not zero.

Main moves to removing the four unused padded-head softmax rows while retaining
the original12real-head arithmetic. Initialize the padded probability/alpha
storage explicitly,keep all16lanes of each active subgroup participating,and
check real-head FP32state and complete output bits. Different subgroup loop
counts must not skip a workgroup barrier. The isolated source267a0bb23now
implements this candidate. Orca Luna research and native Luna-max review accept
its source contract for bounded component gates;Main independently regenerated
and verified the exact two-edit diff and arm hashes. The native build at
90cfc02decloses696actual compiler inputs and both links with zero uncovered
inputs. All four normal/state properties,the traced tiny gate,and11cases at
each Page4/Page16pass. All12real-head FP32states and complete BF16outputs
match;guards,immutable inputs,providers,ownership and fault checks pass.
Physical262144component addresses are covered,not a full-model context fill.
The three-process72sample full-path timing screen **rejects this candidate**.
Wall-time geometric means are1.062551,1.055563,1.056270and1.044660for
32Kcontiguous/strided and117488contiguous/strided. Every process is slower
in every case;attention and combined-service criteria also fail. Numerical,
clock,provider,owner and fault checks pass. A lower compiled spill descriptor
does not override actual timing. No model trial or adoption follows.
Main moves to a separately planned exact-WG128native comparison:the earlier
CPU-IR metadata-only result did not establish downstream IGC code or speed.
The isolated required-WG128source at29d9d5519now builds with696actual
compiler inputs and zero uncovered link inputs. Its four compiled properties
pass:both candidate kernels require128x1x1,while controls have no required
size. Pairednormal/state argument counts agree before any enqueue;the tiny
trace also matches indexed local-argument bindings. Eleven host admission
checks,the traced tiny gate and all11cases at each Page4/Page16pass. The
three-process72sample timing screen rejects this candidate:wall-time geometric
means are1.000150,1.000039,1.000112and0.999295for32Kcontiguous/strided and
117488contiguous/strided. All three timing metrics miss the prospective gate;
no model trial or adoption follows. Bit/state,clock,provider,owner and fault
checks pass. Do not extend Main's investigation of this rejected annotation.
Selected6e40and decode remain unchanged.
FullQSA/task/performance and physical-context/handoff gates remain unqualified.
Selected6e40and its twelve same-build fixtures remain.

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

A further comparison with the original fastest receipts finds an observer-policy
difference:396d did not enable UR parameter validation or the native validation
preload;the later6e40 historical/current measurements do. The benchmark's
prefill construction,eligibility and execution methods have identical ASTs across
the fastest,historical6e40 and fresh-reference clients. All retain the engine's
read-token and prompt-millisecond metric;`bench.sh` is identical. Server tokenizer
prefix caching and the client's outside-request gates/endpoint metadata also
changed. Thus the fastest package is not an identical measurement envelope,and
no isolated cost of the earlier validation transition is measured. This cannot
by itself explain the later1003→975 decline:both sides of that same-binary
comparison already use the validation/preload policy. Keep policy changes
separate from a changed timing formula or an application-source regression.

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
That current arm's three-process median is911.7371tok/s,9.0673% below that
same-binary historical cohort and9.3489% below the fastest1005.7655 package.
Keep that cohort separate from the older974.7035 and the fresh reference below.
The benchmark contract remains matched;the decline is observed,
but its resource-level cause remains unattributed. This is evidence against
userspace rollback as a sufficient remedy,not proof that the libraries have
identical performance or that historical clocks/contention were reproduced.
Decode implementation,configuration and1256-slot placement remain unchanged.

An independent installed llama.cpp control on20261026 completes the user's
Qwen3.5-9B Q4_K model with131072prompt tokens,FAon,K/Vq8_0,batch2048,ubatch512
and six CPU threads. One full-length warmup precedes three timed repetitions:
1040.4683,1042.3779 and1043.1894tok/s;mean1042.0119,standard deviation1.3970.
The2048-token traced prerequisite and the long process close normally,with
pre-main validation,45observed code maps and no fresh fault. Both GPU processes
use official oneAPI2026.1,oneDNN and the actual B570 LevelZero/OpenCL providers.
This is three within-process samples,not three independent process repetitions.

The user's earlier966.57±10.59tok/s report has no executable/runtime/environment
hashes. Current safety settings,diagnostic retained-loader validation and an
explicit Intel GPU ICD directory are charged;the installed binary's complete
build-time input receipt is unavailable. Thus the new control is not an exact
historical comparison or a hardware ceiling. It does show that this dense-model
path currently exceeds the earlier reported speed;uniform GPU slowdown is an
insufficient explanation to assume. Different shapes,algorithms,host work and
resource interactions remain open. It neither explains nor restores Strata's
911.7371versus1002.6507result.

A fresh unchanged6e40 process at09:33..09:46UTC on20261026 completes the same
four117489-token requests,with thinking and zero reuse. First use is909.4597tok/s;
the subsequent samples are975.0027,975.3014 and973.0695tok/s,median975.0027.
All15 final model hashes match,the native process exits normally,parameter
validation is silent and no fresh fault is recorded. Source/controller b8c31f1ae
and frozen plan7bf4a66f4 recover this diagnostic in the linked runtime result.
No inference source,decode placement or runtime changed for this check.

This117489-token point does not establish input-length scaling or a fixed
prefill cost. Client/controller`5a0328c1e` adds a six-length characterization through
`bench.sh --prefill-scaling`:thinking stays on,actual fresh counts and total
prefill time/pp/client TTFT are retained,first maximum-length use is separate,
and three same-process sweeps rotate their order. Keep each length's first visit
visible;maximum-length process priming does not prove every shape is warmed.
Affine,quadratic and`n log n`
hypotheses expose residuals and leave-one-length-out errors without selecting
a law automatically. Short/decode-placement and long/layer-major routes must
be assessed separately before interpreting a fixed term. Twenty-one client,
fifteen evidence and four controller host tests pass,including first-reasoning
TTFT and immediate rejection of invalid prefill clocks before the next request.
The owned sweep closes normally on20261027,with19 gated scaling requests plus
one short warmup,all15 model hashes unchanged,quiet validation and no fresh
fault. The linked runtime result's`length_characterization` retains every
individual sample and the frozen source/configuration. These are three visits
per length in one process,not independent-process repetitions. Generation24 is
only for prefill/TTFT and does not characterize tg versus context/output length.

| Actual fresh input tokens | Median prefill seconds | PP tok/s from that time | Median TTFT seconds |
| ---: | ---: | ---: | ---: |
| 1286 | 9.9825 | 128.8 | 10.095 |
| 4956 | 17.4755 | 283.6 | 17.570 |
| 19665 | 31.3533 | 627.2 | 31.518 |
| 39254 | 49.3806 | 794.9 | 49.505 |
| 78445 | 81.8309 | 958.6 | 81.969 |
| 117483 | 120.6398 | 973.8 | 120.901 |

Across the12subsequent long-route samples,client TTFT minus native prefill time
is0.1087..0.2612s,median0.1343s. The first maximum request adds0.1638s;the initial
short warmup instead adds3.7453s. These derived differences include first-token,
phase and frontend work;they are not isolated network latency or a universal
fixed offset. They support tracking TTFT alongside prefill with first-use apart.

The first maximum-length request is separate:129.6552s and129.819s TTFT.
The preceding62-token short warmup takes25.9737s with29.719s TTFT;neither
belongs in the subsequent-request curve. Source/model hashing primes the file
cache,so these are not cold-disk measurements. First visits remain visible.

For the four long-route median requests,a coarse approximation is
`T_seconds(n) = 13.21465 + 0.000903860*n`,over19665..117483actual read tokens
only. Maximum relative residual is2.795%;leaving one entire length out and
refitting the other three gives maximum prediction error4.309%. The measured
long-route time ranges are0.407..1.427% of their medians. Thus the formula is
useful for rough estimates,but does not explain the smaller variations or
provide a1% predictor. Its intercept is not a measured physical fixed cost.
The global mixed-route affine,quadratic and`n log n` models have maximum
held-out errors42.96%,52.30% and51.47%;none is adopted as a global formula.
Only two short-route lengths are available,insufficient to validate their own
time model. Higher R-squared alone would hide that problem.

Adjacent long-median slopes are0.920277,0.828004and0.994131milliseconds per
additional token. Their variation is another reason to keep the affine model
coarse rather than interpreting its coefficient as one physical kernel cost.
For65536actual read tokens,the formula predicts72.4500seconds,or904.6tok/s;
1000tok/s at that exact extent would require6.9140seconds less than this
prediction. This is an unmeasured estimate,not a64Kbenchmark result or a new
fit. The measured117489-token local process medians below instead need about
3seconds removed. The official client now accepts any minimum32768..131072
(source27de99c74),so intermediate held-out lengths need no prompt calibration
or new benchmark path. All36host client/evidence checks pass,including
context refusal and actual-count/thinking/zero-reuse checks at minimum54613.

The new held-out-length cohort(source4501a1cd3,plan4d3d1daae)uses that minimum
with unchanged selected6e40,and actually reads65345fresh tokens per request.
Official`bench.sh`keeps thinking enabled and emits24tokens;it does not qualify
sustained decode. The first use takes87.047s;subsequent requests take78.682,
76.909and76.995s,or830.5,849.6and848.7tok/s. Their median is848.7tok/s.
The frozen old formula predicts72.277s,6.127%below the median,exceeding its
old4.309%leave-one-length-out maximum. All four streams,journals,source/model/
provider/route/owner/fault gates pass;no new GPU fault occurs. One fresh process
is not independent-process replication. Do not treat this previously unmeasured
length as a same-workload regression or refit away the prediction error.
The formula remains a coarse estimate with this additional out-of-sample
failure;window/chunk and GEMM-shape dependence need characterization before
using a single line as a tuning predictor. Evaluate the next admitted route at
an unfit length before generalizing its gain across contexts.

A posthoc check of the five long-length medians gives
`T_seconds(n)=14.08822+59.30688*(n/65536)`,but its maximum leave-one-length-out
error is6.127%. Quadratic and`n log n`alternatives worsen that to16.158%and
9.963%. No new predictor is selected. Adjacent slopes with the new point are
0.920277,1.058373,0.369183and0.994131ms per added token. Token count and
8192-row chunk count correlate0.99845in these points;their costs cannot be
reliably separated by adding another fitted coefficient. Only one old length
has two MoE windows,so leaving it out also makes a window-step coefficient
unidentifiable. The new point is from another process;cohort effects and
data-dependent shape mix remain unresolved. These observations guide the next
measurements rather than identifying a hardware bottleneck by themselves.

A source-grounded nonlinear hypothesis was also checked against these same four
long medians. With fresh engine rows`n-1`,Page4and chunk8192,the prospective
packing feature above takes530880,1454112,5365008and11731728logical rows.
Fitting an intercept,token term and that feature slightly lowers the maximum
training residual to2.645percent,but leave-one-length-out error grows to14.400
percent. An intercept+token+quadratic fit gives2.593percent and13.564percent.
Both predict held-out lengths worse than the simpler existing affine fit
(2.795percent training,4.309percent held out),so neither replaces it. These
are algorithmic features applied to medians,not measured transfer bytes or
isolated costs. Four long lengths do not justify another coefficient. The next
native/model candidate still needs an unfit length and placement/route evidence;
keep first-use separate and revisit the curve after any adopted route change.

The later clean comparison supplies a local independent-process check of the
selected path at117489actual read tokens:subsequent-three medians are120.5382,
120.4567and120.7068seconds,with corresponding client TTFT medians120.696,
120.613and120.867seconds. The existing formula predicts119.4083seconds,
0.870..1.076percent below those observations. This is only a six-token extension
near the old largest length,not another independent held-out length or a refit.
At this input size,1000tok/s needs117.489seconds:the current process medians
need2.968..3.218seconds removed. Keep the original multi-length curve and its
broader held-out error;do not fit rejected pair samples into the selected path.

A future admitted geometry variant also retains source-grounded scaling features. Let`n`
be actual logged engine rows,`p`the starting position,`c_k`rows consumed before
chunk`k`,`T_k`its actual length,and`P`the page size. The page-rounded packing
feature is`12*sum_k(P*ceil((p+c_k+T_k)/P))`. For per-query attention,use the actual selected width`w`
and tile extent`32*ceil(w/32)`when a separate bounded diagnostic supplies them;
cap2051alone does not determine those values. Compare16versus12matrix rows
separately from unchanged K/V staging. These are algorithmic counts,not measured
traffic or coefficients. Prefix reservation and actual expert/residual placement
are separate features;the rejected pair reservation is excluded. Clean timing must not acquire per-layer host
readbacks merely to collect those counts.

Candidate models may add one such feature to a route-specific time approximation,
then test residuals and an unfit length. Keep the existing five-length minimum
for the general model comparison. Prospective long points include engine rows
32768,49152(held out),65536,81920and117483. Separate boundary diagnostics cover
81919/81920/81921,and90112/90113only when the reduced-bank route is requested
and actually admitted. Those are planned engine extents,not promised client
text-token counts.32768is the long-evidence floor,not a source dispatch boundary.
Record client prompt/read/reused counts,engine`n`,`p`,liveKV,route and placement
separately;do not assume`n=prompt_tokens-1`. The current clean sweep did observe
that relation,but other entrypoints and reuse can differ. New coefficients and
model gains require new matched measurements after safety gates.

An independent check against the closed official cohort's original
processes0,3and4 keeps that formula unchanged. At32089tokens it predicts42.2186s,
versus43.3232/43.1914/43.8492s measured (2.25..3.72% underprediction).
The regular-code prompt at33212tokens takes45.4577/45.6972/46.0033s,
versus43.2336s predicted (4.89..6.02% underprediction). The varied-prose
prompt at33286tokens takes45.6359/45.6852/45.3827s,versus43.3005s predicted
(4.59..5.22% underprediction). The strict-long within-workload time ranges
are1.194%and0.663% of their medians. Each sample has zero reuse and retains
client TTFT in the linked QSA cohort result. The32089case remains supplemental,
below32768. These different workload bodies and request histories are not
extra points in the original controlled sweep;all six strict-long samples
exceed its internal4.309% held-out error. Token count alone is therefore
insufficient for that precision across workloads.
Keep workload/route/first-use labels and test unfit lengths instead of changing
the coefficient to absorb this discrepancy. This is characterization,not a
new implementation regression investigation or a performance-adoption result.

Native logs bind48MoE accumulator windows to19664,39253 and78444 physical
prefill rows,and96windows to117482rows,with an81920-row cap. The adjacent
long-route slopes are0.9203,0.8280 and0.9941ms/token;this is not evidence of
one exact linear law. A source-only check of selected`6e40afc1b` adds another
boundary condition:`ExpertBankPlan::make` excludes the staged expert bank when
`n>81920` and`1<=n%81920<=8192`,because that short final FFN uses the full512-slot
path. Those`n` values are physical prefill rows,not the complete prompt count.
The measured lengths do not exercise this band. A boundary sweep must record
actual bank/residual placement as well as window count;assuming one unchanged
long-input route across it would mix configurations. This is a source predicate,
not a measured admission or timing result. If a tuning decision needs better
prediction,measure around window/chunk boundaries and confirm with independent
processes before attributing the bend or extrapolating. Requested expert H2D volumes remain
about33.8..38.3GB across these lengths;they are software-requested bytes,not
physical PCIe traffic or a saturation measurement. No inference implementation
changed for this sweep;it neither restores the historical best nor qualifies
physical262144 execution. The later packed-u32 QSA component comparison passes
numerical checks but rejects the candidate on speed;see the QSA decision below.

This fresh within-process median is6.9390% above the earlier current-userspace
cohort's911.7371,but remains2.7575% below the same binary's historical1002.6507.
These unpaired measurement windows show that the9.1% loss is not persistent at
one fixed magnitude. They do not identify its cause or establish a speedup from
an implementation change. This is one process,not three independent repetitions;
the dense control was several hours earlier,not simultaneous. Full model hashing
before this run also differs from the older crossover's cache history. Historical
effective clocks,workstation contention,paging and process/cache history remain
uncontrolled. The fastest1005.7655 package is still a separate configuration.
Neither restoration nor sustained decode is qualified. Subsequent official-SDK
public-GU component work below preserves the distinction between GPU service
and queue intervals containing host submission gaps;it has not explained this decline.
This latest result is3.0586% below the earlier fastest396d median1005.7655.
A bounded reread of
the actual Codex final answers confirms that the912 and975 reports refer to
different closed windows;neither is a newly defined throughput metric. Their
individual samples remain separate in the linked result. Do not continue to
describe a fixed9.1% current decline after the fresh975 recheck.

Reference setup failures retain their original outcomes. Three attempts refuse
previously unlisted code maps;Main also selected the CPU OpenCL registration instead of
the GPU one before correcting the private ICD directory. These are not GPU hangs
or Strata inference failures. The successfully qualified control pins the actual
GPU registration and providers. The fresh original6e40/117K reference above
closes that reference check. The subsequent production-GU calibration and
failed warmed attribution are recorded below;GPU service remains distinct from
host submission and waits. GU's existing direct component benchmark does not measure that public
path;Down's public benchmark falls back below2334rows. Static dtype review finds
that GU's FP16 comparator receives FP16 hidden outputs;BF16 intermediates and
Down outputs are checked bitwise. The proposed dtype bug is not substantiated.

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
847.7127/916.5116tok/s,not replacements for the selected-engine speed reference
above.

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

A separate arithmetic count comes from the conserved historicalbb71 borrowed-
cache census:56394240routed rows in37353jobs,GU(N,K)=(1280,2560),
Down(N,K)=(2560,640). With the conventional two FLOPs per multiply-add,
`2*56394240*(1280*2560+2560*640)` is554.377936896TFLOPs,or4.71855GFLOPs
per117489read tokens. This counts only routed logical GEMMs. It excludes dense/
shared projections,attention,indexing,recurrent work,dequantization,data movement
and padding,so it is neither the full model's operation count nor measured6e40
arithmetic service or hardware utilization. Use these actual shapes/work counts,
with compatible qualified service rates,before deriving a hardware ceiling from
an active-parameter label or a busy percentage.

The corrected source ledger in the research index also counts489.9315TFLOPs
for the36 layers' GDN projections and148.7581TFLOPs for both HC projection
halves in48layers,at117488rows. These are additional logical matrix operations,
not measured service or issued instructions. The projection call shapes remain
present in the selected6e40 source;normalization,recurrent FP32 work,attention,
shared FFN and data movement still need separate accounting. An MoE-only
operation count therefore cannot stand for the full prompt or share one
unqualified arithmetic ceiling across these different operators.

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
remain fixed. Use the fresh117K selected-engine reference above for current
speed;these component results do not replace it.

After extraction and independent review,456closed raw/build/review/draft files
retire,119638008bytes,with a path/hash/reason manifest. The completed whole
profiling engine and command-profile executable retire too. The selected6e40
engine and12fixtures remain intact. The old two GU/Down executables,1385-input
manifest and generated header are subsequently retired after recovery of9866
original inputs and their recipe. The new official four-fixture generation now
serves public-dispatch calibration;only the original component recipe record
remains consumed by its builder. Component throughput alone cannot establish
a whole-model arithmetic ceiling.
Closed one-use build/cost/calibration controllers retire to their original source
commits. The old isolated transfer fixture also retires to d7737c53e once the
whole-bank comparison closes;its4Bstatus samples stay ineligible. No maintained
script/target consumes that source. Current bank-test source remains maintained;
closed one-use staging controller restores from e64229a83. No routine raw API
trace or completed observer cohort remains.

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
262144context remain unqualified for it. That cohort's clean117K reference was
911.7371tok/s;the fresh unchanged-engine reference above is975.0027tok/s.
The historical decline remains unattributed.

The official native16 calibration build now completes four executables from
source2369886e9,with1427actual compiled inputs and zero uncovered inputs.
It retains the original kernel arithmetic and size_t ID/range ABI option.
The build permits deprecation warnings while keeping other warnings fatal;
the compiler resolves its own offload-bundler path. The earlier Werror and
tool-layout refusals retain their failed status,with partial builds retired
after9972original source inputs and their recipes were recovered.
The subsequent official-runtime compatibility cohort passes ten ordered fresh
processes:one host canary,nine GPU submission/native16/GU/Down cases. These cover
32normal dyadic input/output/tile cases,26pre-enqueue guards and the actual public
GU/Down dispatch boundaries. Source,1427compiled inputs,imported link files and
the observed providers match;parameter validation is silent,owners close and
no fresh fault appears. Its first host-only attempt failed on a short-process
`/proc/environ` exit race before any GPU case;the corrected plan requires positive
pidfd exit evidence and preserves that original observation failure.
This permits bounded normal-input public-GU calibration only. Large-range ID
edges,same-process mixed-image selection,full numerical/model,oneMKL ABI,decode
and physical-context qualifications remain separate. No new whole-model speed
or cause for the historical decline is established.
After recording those outcomes,233closed files/117078923bytes retire:successful
API/JIT/process output,disposable build objects,the superseded component pair
and consumed research/review files. The selected engine and12fixtures remain;
the official four executables,1427-input manifest,build recipe/Ninja/header and
compatibility runtime catalogue have the named upcoming public-GU consumer.
Post-reduction admission passes again:original13assets,official4fixtures,
9972recovered inputs,1427compiled inputs and84runtime entries remain intact.

The public-GU calibration now closes eight official-runtime processes without
a fresh fault:two traced profile processes and six timing processes in0/1/1/0/0/1
order. All108fixed-duration batches pass numerical,actual-dispatch,guard,timer
and endpoint-control checks. Each process median below uses five later batches;
the initial timing batch follows setup/warmup and is not cold-process latency.

| Rows | Fusion off,three process medians,us/call | Fusion enabled,us/call | Actual fusion |
| ---: | --- | --- | --- |
| 160 | 69.458 / 67.884 / 69.091 | 68.929 / 68.983 / 68.447 | Neither arm |
| 2333 | 255.322 / 299.556 / 255.106 | 255.389 / 254.839 / 255.299 | Neither arm |
| 2334 | 256.265 / 255.533 / 255.414 | 248.551 / 249.104 / 249.084 | Enabled arm only |

The2334-row paired time ratios are0.969898/0.974841/0.975215,geomean0.973315.
The2333-row negative control has one slow off process despite identical fallback
selection;its0.947858geomean is not a fusion benefit. Keep that variability.
Isolated one-call GPU-service profiles have different cache,spacing and observer
history;some exceed the warmed whole-call times. They cannot be subtracted to
invent CPU time or a critical path. Source/controller7c52325ed and the compact
`public_gu_dispatch_calibration` result preserve all samples and limits.
This establishes current component behavior,not historical-regression causality,
hardware saturation,a new model speed or inference adoption. The source followup
confirms permanent expert residency persists across windows within a layer;do
not assume every window reloads every weight. Dequant is window-local. Moving
experts outside the window loop requires current-layer post-attention/PLE input
availability,extra mixed/FP32-partial storage or recomputation,and a262144-safe
fallback/budget. The existing future-admission report already covers that input
availability constraint;no cross-window candidate or saving is selected.
The selected engine and decode1256placement remain unchanged. Existing
ownership/submission references are sufficient;no duplicate Orca survey is
eligible now. After replay,65raw/review files and the
one-use controller retire;2,230,941raw bytes are removed. Restore the controller
from7c52325ed for a concrete later comparison;do not retry its closed root.

The warmed public-GU attribution remains unqualified. The short-batch plan
`public_gu_warmed_attribution` stops at an endpoint-control violation before
new off/on timing. The longer `public_gu_warmed_attribution_v2`,sourcefdec18b9d,
passes five traced checks and four timing processes,but observer-off off2stalls
after one eligible M160batch. At300s the owner sends SIGTERM(native exit−15);
xe records a ccs engine reset at termination,with no earlier xe fault in the
observed window. All ten actual sessions are empty. Preserve its failed and
incomplete-observation status;the precise blocked operation remains unknown.
Neither cohort qualifies observer cost,resource saturation or historical cause.
Original plans,individual samples and failure evidence remain in the linked
resource result;closed controllers restore fromfdec18b9d.

The corrected source review follows public`f16_out16` with FP16input/BF16output
through vllm/TLA,then native SwiGLU. The own-backend joint-matrix kernel is not
this route. The transitive TLA helper matches original Git inputs and both
1427-entry compile closures. Source-level bounds/uniform split-barrier checks
find no proven defect;they do not establish generated-code progress.

A new bounded stage smoke now passes on the official SDK. Native source3af1c47d9,
frozen planb776d7179,executes one0.62382962s process:four groups of16public
M160fallback calls,each followed by a checked marker drain/query. All308nested
stage entries match returns;full UR trace has135kernel launches(including
setup/reference/markers),38copies and4profiling queries. FP16hidden output
matches the reference exactly,guards/inputs remain intact,parameter validation
is silent,the native child exits0normally and no fresh fault appears. At most
32ordinary kernels plus one checkpoint marker are outstanding per group.
A pass is only this smaller smoke,not reproduction/avoidance proof for2048calls,
performance,model,decode or physical262144qualification. Logged stage spans
include trace/progress output;they are not isolated API/GPU service. A3s stage
threshold is checked after return;only the30s owner deadline can terminate a
stuck process. See`public_gu_stage_smoke_v1` for counts/maxima and limits.

After original-source/recipe/1427-input/four-ELF recovery,the closed stage
outputs and generation retire:179files/116557305bytes. Official v4remains the
component fallback;original13model assets and one2797-byte unresolved-stall
diagnostic stay intact. Successful full traces/private checkpoints are gone.
The current source removes the closed high-depth timing/attribution modes and
stage-only logger/mode;restore tested source3af1c47d9 for a concrete later need.
The clean official legacy-GEMM build,sourcefa765fcb8,has1424covered compiler
inputs. Its one7.174702625s native process exits0and passes22CPU-parity/stride
checks:18FP16/BF16-to-FP32 GEMMs plus4Q8strided matrices. Transfer-ring,pipeline
and separate-queue cases remain unrun. The original controller fails because
its API-entry regex wrongly requires an opening parenthesis;UR entry lines end
at the function name. Preserve that failed outcome atd708f9ab3. Offline replay,
source9241f6d61/plan982280512,rechecks original source/SDK/runtime,logs,argv/env,
normal ownership and balanced successful entries/returns:67kernel launches,
64USM copies,one buffer read and one buffer write. Parameter validation is
silent and no fresh fault appears. Replay launches no process and does not
rewrite the failed run into a success. See`legacy_gemm_abi_v1` for the distinct
native-matrix and observer-parser outcomes. Neither establishes model,decode,
physical262144,hardware saturation or a throughput improvement.
The closed legacy build/run roots now retire after exact9974-input/1424-closure
and ELF recovery:172files/101717262bytes. The closed one-use controller and
build mode restore from9241f6d61/fa765fcb8. Compact native/replay/failure evidence
remains;official v4and original13model assets stay selected and intact.

To isolate parameter-validation cost,the independently authored control retains
one process-lifetime UR loader reference in both arms;ordinary teardown remains
unqualified. Official2026.2headers/systemGCC13 build sourcebb46c4de5 produces
ELF SHA prefixc8d080a3. Five host cases and six original6e40 submission/graph
cases pass before the clean model cohort,including native environment/provider,
numerical/guard/alias and normal-closure checks,without parameter warnings or
fresh faults. Both timed arms use empty UR_ENABLE_LAYERS;configuration alone
selects validation. This addresses the fastest396d observer-envelope difference;
both historical same6e40cohorts already used validation. Official-SDK whole-model
qualification stays open. The resource result retains the helper/gate evidence.

The matched validation-cost comparison closes all six independent original6e40
processes,on/off/off/on/on/off,with four117489-token requests each. Both arms
preload the same held-reference control and use empty environment layers;only
config-enabled parameter validation differs. First use stays separate from each
subsequent-three median. The three adjacent pairs give:

| Pair | Validation on,tok/s | Validation off,tok/s | Off/on |
| --- | ---: | ---: | ---: |
| 0/1 | 974.2977 | 973.7050 | 0.99939169 |
| 3/2 | 974.3518 | 974.9170 | 1.00058003 |
| 4/5 | 971.6122 | 973.7276 | 1.00217721 |

The ratio geometric mean is1.00071566(+0.0716%). It fails the frozen mechanism
screen(all three ratios>1 and geomean>1.01);removing validation does not restore
historical1003or fastest1006tok/s in this cohort. All24 request hashes/counts,
15model assets per process,actual provider/route checks and normal ownership
closure pass with no fresh fault. Independent replay matches24 native rounded
millisecond logs to the benchmark clocks within0.5ms and recomputes every rate,
median and pair ratio. Four native-wrapper/parser host tests also passed before
launch. This is an observer-cost result,not inference/default adoption,full
context or sustained decode qualification. Both historical same-binary cohorts
already used validation,so it cannot alone explain their1003→975decline.
See `historical_regression_audit.validation_cost_control_v1` in the
[runtime comparison](../bench/results/2026-10-15-pr6-recovery-audit/prefill-runtime-comparison-current.json).
After exact original source/recipe and actual ELF/header/link recovery from the
pushed feature branch,the five closed helper/model/CPU-pilot roots retire:
503files,78,058,081bytes. Twelve consumed research/review/launch transactions
also retire(53,035bytes);the new conditional scheduling collection remains.
Six one-use source/controller files retire to reachable Git history. The native
wrapper rejects the retired comparison's extra probe keys while preserving the
normal library/loader validation path. Source9f097bc87 passes four host tests,
including existing provenance/lifetime metadata. The actual selected build and
current separately requalified runtime also pass source/provider and native-only
environment admission without launching a native process. The legacy preload gate separately refuses changed libz;that refusal is preserved,
not repaired by refreshing hashes. No new GPU qualification follows.
The selected original engine and official component fallback remain outside
these deleted roots;older provenance holds are not cleared by this cleanup.


The source303c1b06f CPU-only pilot establishes unprivileged CPU0RDPRU access,
not the model's effective clock,MHz or SMT semantics. The new source4671d1cb0
observer pins its fresh process to CPU0,uses1Hz timerfd/EOF control,serializes
MPERF/APERF sandwiches and records raw-monotonic boundaries. Its source58cbd6d48
six-process CPU-only on/off/off/on/on/off control closes normally,with ten owned
compile/link/access/control commands,actual executable/libc/loader identities,
matching fixed-work checksums and no fresh fault. The clock includes observer
fork/exec,initialization,wakeup,output,EOF shutdown and wait. The three off-wall/
on-wall ratios are1.00758786,0.99733396,0.98675721;geomean0.99719007 passes the
frozen each>=0.98/mean>=0.99screen. Each on process records eight periodic
samples;whole childCPU is781000..912000ns over8.80..8.93s(0.00875..0.01028%).
All six native outputs and ratios replay. These CPU-fixture results do not
establish observer perturbation for actual GPU/syscall/polling work or explain
the historical decline. The separately pinned model attempt below remains failed.

The first controller fails on bare observer_ready/observer_done framing after
five normally closed native commands. Preserve that failed/incomplete result
at91b8dc832;corrected parserf1790554b replays its single eight-tick on output
ataf3e60390 without a new native launch or cohort qualification. After exact
source/header/link/ELF recovery,the old17files/127602bytes retire. The fresh
cohort uses a new plan and build. Orca source audit and native Luna/max review
accept only the declared host-fixture scope. A future model observer still needs
per-launch source/executable/provider checks,a sanitized fresh --observe process
with piped stdin,bounded EOF/ownership closure and actual model-wrapper overhead
accounting. Monotonic clocks and the inspected owner deadline exclude suspend;
a later model plan must reject suspend with BOOTTIME checks and retain counter
scope/skew limits. Rare host-control-only fd edges do not block fresh --observe
with piped stdin. The original engine and officialv4fallback remain unchanged.

The source175cd3189 request-scoped model diagnostic passes its standalone
stdin-pipe/ready/EOF/provider/ownership host check,then stops after the first
117489-token request because controller CPU/gate elapsed exceeds the frozen5%
limit. The exact rejected fraction was not persisted before the check;do not
invent it or raise the old limit. The native first request reports129257ms and
24outputs;there are no subsequent samples. Owned benchmark processes are
terminated and reaped,the session is empty,and the independent observer records
normal EOF. The normal-ownership gate remains failed;separate closure evidence
confirms no owned survivor and no fresh GPUfault. This is not a GPU hang.
Offline replay recovers134 ordered CPU0 counter intervals:ratio median1.253738,
range1.075178..1.256571,max snapshot942ns and gaps0.998988..1.000990s.
They span the first request and cleanup,not exact native-prefill boundaries or
model-thread MHz. They do not qualify model observer cost or explain history.

A separate CPU-only supervisor control uses the same unchanged Owner and native
observer,with no model or GPU. Plain/profile/plain processes all close normally.
The two unprofiled controls consume1.21017s/10.08314s and1.25918s/10.12421s of
controller CPU:12.0019% and12.4373% of one core,both above that5%model limit.
The CPU-time profiler itself raises controller cost to4.66036s/10.49667s;
its nested scan time is4.01402s of4.66016s in Owner.run(86.13%),including22467
identity calls. Artifact sizing accounts for0.40672s. These cumulative times
overlap and must not be added. Repeated whole-/proc scanning is the primary
host cost in this profiled supervision workload. This confirms a measurement
confounder to control,not a3%inference slowdown or historical-regression cause:
the supervisor source is unchanged across historical measurements. A separately
pinned whole-model comparison must establish whether changing supervision
scheduling/cost changes prefill. Do not repeat CPU-clock model runs without a
new question requiring them. No inference implementation or decode placement
changes follow from these diagnostics.

The source1a052a949 supervisor candidate preserves250ms whole-/proc discovery,
fresh pidfd/start-time/session checks,escaped-member tracking and bounded owned
cleanup. It filters foreign sessions through a bounded raw-descriptor stat read
before constructing a complete identity;registration returns only a fresh
accepted identity for accounting. All14 host controls pass,including short reads,
read errors,PID races,escaped children and owned-only cleanup. Six fresh CPU-only
wait processes give baseline fractions0.10447846,0.10473940,0.10481507 and
candidate fractions0.04662729,0.04683367,0.04631702 of one core. All three paired
candidate/baseline ratios pass the frozen<=0.8screen;geomean0.44510192 passes
<=0.75,and every candidate meets the separate<=0.05absolute cap. The preceding
two candidates remain failed at that absolute gate. This qualifies a separate
model comparison,not inference adoption,a historical cause or a55%model gain.
The default supervisor and inference implementation remain unchanged.

The current matched model control uses sourcea6a42e20e and the original6e40
engine,1256 decode slots,qualified validation/safety settings and8192 chunks.
Its input gates bind the15 model assets,committed expert profile bytes/file
identity,effective bench.sh/server Python runtime,complete expected environment
with shell-only exceptions,and normalized executable/script plus exact remaining
argument vectors. Nine fresh host controls pass. Request CPU timing starts at
begin-gate receipt before admission and ends after end-gate validation/artifact
checks;the frozen plan names its own receipt/progress/continue and final-audit
exclusions. Prior unlaunched profile/runtime/environment/argv plans remain
unchanged in [pushed history](https://github.com/yqYo1/Strata/blob/3690a67d2/bench/results/2026-10-15-pr6-recovery-audit/prefill-runtime-comparison-current.json).

V5failed before its first long request because actual server Python mappings
were absent from its definition-only runtime inventory. The original failed
normal-ownership gate stays failed;separate cleanup confirms direct-child
reaping,empty SID and all four recorded identities absent. No long speed sample
or fresh GPUfault follows. Its gate omitted unexpected paths,so do not claim
that a later mock observation proves that old process's exact missing libraries.

The replacement host probe imports the lazy tokenizer and exercises loopback
model discovery/chat through MockEngine,ByteTokenizer and production service/
telemetry initialization. It constructs no StrataEngine or SYCL/Level Zero runtime;
normal telemetry can query NVML. Two discovery and two fresh qualified host
processes pass and close normally. All3593 original identities are unchanged;
the inventory adds libnvidia-ml,libcuda,libdl and librt,for3597files/270975442bytes.
Live model admission still rejects any unobserved mapping and now records its
exact path. The environment controls change/omit expected values;they do not
explicitly add a new child key,although the live comparator rejects unapproved
extras. This private Python-startup envelope differs from historical runs.

Independent Orca and native Luna/max review accept V6's exact five-source/plan
contract. Its six-process order is baseline/candidate/candidate/baseline/baseline/
candidate,each with four117489fresh/zero-reuse/thinking/24-output requests.
All six processes pass all four requests each,source/
provider/route/journal/clock/CPU/closure gates,final15 model hashes and
no-fresh-fault checks. Individual rates and CPU operands remain in the compact
result;first use is separate:

| Process | First use,tok/s | Subsequent-three median | Controller CPU fraction,all requests |
| --- | ---: | ---: | ---: |
| 0/baseline | 904.1691 | 975.6821 | 11.4852..11.6506% |
| 1/candidate | 905.1339 | 976.8543 | 6.1229..6.2783% |
| 2/candidate | 907.1333 | 975.4528 | 6.0249..6.1065% |
| 3/baseline | 904.8292 | 975.2982 | 11.4547..11.5684% |
| 4/baseline | 901.2637 | 973.7228 | 11.4648..11.8417% |
| 5/candidate | 906.0511 | 976.5295 | 6.1060..6.1972% |

The three candidate/baseline ratios are1.00120144,1.00015858 and1.00288249;
their geometric mean is1.00141354(+0.1414%). Each pair is positive,but the
prospective greater-than1% mechanism screen fails. The supervisor CPU reduction
does not restore1000tok/s or explain the historical decline;no inference or
default-supervisor adoption follows. This closes the matched supervisor-cost
question for this workload. Return to the prefill workspace/resource question,
keeping sustained decode,agent quality and physical262144 correctness open.

The subsequent65536-minimum official workload completes one process with four
78451fresh/zero-reuse/thinking/24-output requests and one78450-row MoE window
per layer. Its original cohort fails because the new controller incorrectly
requires disabled Down fusion,although the frozen selected environment requests
`STRATA_SYCL_PREFILL_DOWN_SCATTER=large`. Keep that failure:the remaining two
processes are unlaunched under stop-first. Closed-data journal/native-clock and
route replay,owned exit0,no-fresh-fault checks and15 full model hashes preserve
a diagnostic subsequent-three median960.2831tok/s,without performance adoption
or three-process qualification. The first subsequent request has18480dequants,
21513jobs and37,746,227,200requested expert H2D bytes,versus the earlier117K
passive sample's33225dequants,37353jobs and38,673,152,000bytes. These are different
input sizes,not a matched mechanism gain or physical traffic/critical-path proof.
Close this unchanged-engine input-size screen;next resolve the service/resource
cost of the repeated work before selecting another inference experiment. Its two
one-use source files retire to949643937,and the full closed plans/results to
cbedff2f5. After replay,87raw/input/review files totaling12,729,396bytes retire;
the selected native/runtime/profile/Python admission still passes without them.

Exact original source,configuration and pure environment-recipe recovery closes
the two superseded unlaunched model roots and old Python-probe root. Their
original outcomes and plans remain in pushed Git;normal raw files and duplicate
current rows retire. The current Python manifest,ownerV3comparison evidence,
selected engine and separate legacy provenance holds remain.

After the first V6process passes independent raw/compact replay,retirement removes
150 files/26,438,140bytes:its normal raw output,the superseded V5/PythonV3roots
and consumed V5review transactions. Process0keeps its two generated config/plan
inputs;all individual clocks,CPU operands and closure/fault outcomes remain in
Git. Four old source-audit rows and superseded plans/probe rows retire to
`b97e30d9e`;the current V6review,PythonV4andownerV3qualification stay live.

After retirement of two hardware components and one closed softmax component,
the existing82-root inventory has144 live artifacts/104,598,672bytes across46roots. This is not a
current consumer/source qualification or a census of newer builds. The six-process
comparison is closed;Main must reconcile those remaining paths against current
consumers and exact source/recipe evidence before another engine build. The prior
14 named provenance holds are not cleared by the150-file retirement above.
For two of those historical component roots,Main now recovers the original
`hardware-ceiling-cpu-v1`/`hardware-ceiling-gpu-v1` sources and build records from
`7ea2650ba3ae1b15807a32a67b62d56131046569`,under
`bench/results/2026-10-10-parallel-round53/hardware`. Both source hashes and the
current final ELF hashes match their build/measurement receipts. This does not
retroactively qualify their SDK/header closure. Main checked current consumers,
exact source/recipe recovery and process closure,and retired their two executables
and two source copies:4files/478867bytes. The compact
`legacy_hardware_component_retirement_20261027` record pins the original inputs,
measurement reference and deletion manifest. The softmax component and its parent
library recover483and476 exact project inputs from5f4642931and a8aa3345d;its
closed serial/subgroup controls are preserved,and its1,099,648-byte executable
retires under`legacy_component_retirement_20261027`. The remaining46roots still
need their bounded recovery/consumer decisions. Neither retirement qualifies a
new build or recreates the historical runtime.

After replay and exact source/recipe/header/link/object/ELF recovery,these three
closed observer/model/control roots retire106files/12,760,397bytes. Four
one-use sources/controllers retire to pushed history at925ddfe8e;the selected
engine,its13artifacts and officialv4fallback remain unchanged. Fifteen model
asset identities still match after failure;only the initial full hashes were
rechecked for this failed model attempt. Keep the original failed outcome and
the individual CPU-cost values above. Future supervisor work needs a new plan;
the retired observer is not an accepted model-clock implementation.

A passive check of this kernel7.0.0-38 finds tracefs,its xe event directory and
DRM debugfs inaccessible(EACCES). The installed xe UAPI documents memory-range
queries as preferred-location/PAT/atomic-policy queries,not physical-placement
history. Its region used counters require CAP_PERFMON or CAP_SYS_ADMIN;no query
was executed. Both B570PMT telemetry endpoints expose readable GUID/size/offset metadata,
but their0440 raw telemetry files are unreadable;no registers were read. These
observations do not establish migration or rule out every userspace observation path. Physical backing/traffic and current owner overlap
remain unresolved;fdinfo aggregate changes alone cannot establish eviction.

The fresh sourcee33d16fc3 attribution fixture closes ten official-SDK processes
with240eligible/correct timing samples and120complete CPU/GPU profiles.
Both large-payload observer screens fail:one ordinary on/off ratio is1.131068
and one staged ratio is1.111921. Do not qualify its resource attribution or
advance a model. The frozen overhead clock also excludes owner setup,event
queries and formatting;the two upload-submit categories have different wrapper
boundaries. An independent Luna review confirms these interpretation limits.
Large batches often alternate between fast and slow in both this cohort and
V5. The between-batch full-bank D2H guard check is one unproved carryover
candidate. Next compare it with equivalent GPU padding validation using a
prospectively complete observer clock and balanced order strata. Selected copy
service and synthetic-reader time do not establish physical bus traffic or the
cause of the historical model decline.

The V2guard comparison stops at a safe between-process boundary after eight
normal processes,132correct timing samples and44profiles. No fresh GPUfault
occurs. Main found missing checked retirement around post-timing validation:
a copy/enqueue/wait exception could release the local host destination or the
GPU status owner too early. This diagnostic safety hold is not an observed
fault or a completed speed comparison. Source adds a checked lease after
every DMA owner. The fresh V3cohort closes14normal processes,396eligible/correct
samples and132complete profiles. Eight padding-boundary cases and two
host-exception-after-enqueue retirement controls pass;no fresh GPUfault occurs.
The selected model/decode binary is unchanged.

For2662400-byte payloads,the predeclared GPU-padding/full-D2H-check time-ratio
geometric means are0.824929ordinary and0.818257staged;all three process ratios
are below1 in both arms. This confirms that between-batch validation can change
the next timed batch. The change is a method package:host allocation,zeroing,
full-bank D2H and CPU checks versus a GPU padding check and4-byte status copy.
It does not isolate PCIe cost or measure model throughput. Both small-payload
observer screens still fail their frozen overhead limits;the overall resource
attribution is unqualified. No staging/model advancement follows,and the
historical9.0673% model decline remains unattributed. Keep V1/V2/V5outcomes.
Stop this component iteration;next use one newly planned bounded hardware or
whole-model counterfactual rather than repeat until the observer gate passes.

The native SDK snapshot has a named next consumer:official-runtime original
fixture ABI/GEMM qualification and the next attributed fixture build. Source
recovery is complete;58closed component/review files totaling2669603bytes are
retired with a path/hash/reason manifest. The original13assets and separately
named GU/Down ceiling subset remain intact.
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

A broader capacity question changes only the accumulator's hypothetical storage
width in the same source formula. These rows ignore the current81920-row
admission cap solely to calculate capacity,not to claim that larger windows
execute. Local GEMM accumulation and all other plane sizes stay unchanged:

| Window rows | FP32 accumulator+mixed-tail region | Hypothetical16-bit accumulator+mixed-tail |
| --- | ---: | ---: |
| 117488 | 1695745280B | 1094206720B |
| 131072 | 1906840064B | 1235751424B |

The existing region is1180698624B. Thus16-bit storage would fit117488 rows in
this calculation,but131072 still exceeds it by55052800B. It introduces extra
partial-sum rounding and is not the current implementation or a qualified
SGLang/vLLM contract. Before a scoped experiment,check actual reference reduction
storage/order and task evidence,then bounds/ownership and262144-context fallback;
do not treat this capacity saving as accepted numerical behavior or a speedup.

Another capacity-only option preserves FP32 sums in two accumulator banks.
For117488rows,put65536rows in the existing region and51952rows in an additional
allocation. Applying the same alignment/guard formula gives1163756800B for the
existing planes plus mixed tail,leaving16941824B in the1180698624B region;the
additional guarded FP32 bank needs531988736B. No temporal plane alias is needed
for that row count. The current process3/request0 admission log reports
4355637248B free and4206942336B fixed including512MiB headroom. Reducing the
prefill-only expert bank from384 to128 maximum-stride slots would release
681574400B,leaving298280576B after the extra accumulator and the same headroom.
These are logged budgets and prospective arithmetic,not successful allocation
or physical-residency evidence. The current planner rejects this row count.
The opt-in candidate now has a separate split-capacity planner,ordinary/fused
scatter and model caller with owned tail allocation and final guard/drain checks.
The ordinary81920-row path is preserved. Two CPU-only ASan/UBSan controls pass
complete bank coverage,guard reservations,one-byte-short refusal and split/legacy
routes with up to15jobs/expert. Four subsequent bounded GPU stages pass on
this B570/runtime:traced3/17-row smoke with16refusals/3bad-map masks and unwind;
six65537/81921/117488regular/irregular cases with complete post-BF16 FP32
CPU-oracle/capture/guard checks;and8production-route cases per immutable
off/on process. No new driver faults;all owners close. Mixed-weight mappings
cover every job,but mixed gather output checks cover selected jobs only.
These fixture allocations do not qualify model-pressure allocation,model
quality,clean speed,decode or physical262144execution.
Production selection is limited to90113..117488engine rows and default-off;
other lengths retain the existing plan,including262144. Build exact selected6e40
with only the reviewed split changes and the independent prefix maximum-allocation
guard;the source/actual1400compiler-input closure and all3executables pass.
Current-source frontend probes and the exact117489-read model diagnostic
now pass:117488engine rows fully on GPU,128slots,48MoE windows,checked tail
guard/drain and24thinking-enabled outputs. The observed diagnostic132774.1ms
(884.9tok/s),TTFT132.992s has STRATA_TRACE enabled and is not clean speed or
sustained decode evidence. The frozen clean A/B compares identical exact-count
bodies,first-use and3subsequent requests/process,advancing after the first pair
only with at least1percent warm-median improvement. Then repeat opposite orders.
All six processes now close,with24exact117489-read requests,117488engine rows,
zero reuse and identical body hashes between arms. Every native exit is0,
validation stays silent,all15final model hashes match and no fresh fault occurs.
The first-pair screen passes and all three subsequent-three-median comparisons
favor the candidate:

| Pair/order | Original median,s | Candidate median,s | Time reduction |
| --- | ---: | ---: | ---: |
| original,candidate | 127.7833 | 123.4984 | 3.353% |
| candidate,original | 127.6995 | 123.4156 | 3.355% |
| original,candidate | 127.5184 | 124.5006 | 2.367% |

The paired geometric-mean time ratio is0.96974056,a3.026percent reduction,
passing the prospective1percent performance gate. Original process medians
are919.4,920.0and921.3tok/s;candidate medians are951.3,952.0and943.7tok/s.
All nine subsequent samples range917.0..923.3and941.4..957.0tok/s respectively.
Subsequent TTFT medians are127.972,127.875and127.700s for original,and123.679,
123.605and124.722s for candidate. Keep first-use samples separate(817.3..867.3
and853.8..880.2tok/s). This is a clean prefill mechanism result;1000tok/s,
decode,task quality and physical262144qualification are not established.
The split route remains default-off and unadopted. This exact-count factory
produces different request bodies from the previous minimum-count factory's
approximately975tok/s cohort;that is not a matched regression comparison.
Individual samples and the frozen stopping rule are in
[`moe_split_accumulator.clean_prefill_comparison`](../bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json).
Measure job/dequant/upload changes and critical-path service;
larger windows and a smaller expert bank do not establish a throughput gain
or the limiting hardware unit. Decode/handoff,model quality and physical-context
qualification remain separate pending gates.

The independent Down row-address component now passes its frozen comparison.
Source`fcca1e850`caches one64-bit destination and FP32 weight per workgroup row,
instead of repeating the split-bank address calculation for every output cell.
It retains BF16 rounding,FP32 expert-order FMA,all bounds and the unconditional
local-memory barrier;declared row metadata grows from1024to1536bytes.
Build recipe`72717a037`produces only the Down fixture,not another model engine.
All1384actual compiler inputs are covered and the three build commands close0.
The controller`0c6d76613`and plan`bb8d20987`run two correctness processes and
three fresh timing processes,with44complete post-BF16 FP32 host-oracle checks,
guards and immutable inputs. All five close normally with no fresh GPU fault.
The new modes do not add invalid-map device cases or qualify a model route.

Each timing process tests M2334/5963/8192,N2560,K640,regular and distributed
maps,a117488-row split accumulator and65536primary rows. Three rotating arms
compare the original split epilogue,the row-address epilogue and GEMM-only.
All162measured and54separate warmup batches satisfy the prospective5ms floor
(minimum7.578333ms);13824actual kernel events pass order/interval checks.
Candidate/original ratios are geometric means of three paired process medians,
each from three measured rounds:

| M | Mapping | GPU-service ratio | Clean-wall ratio |
| ---: | --- | ---: | ---: |
| 2334 | Regular | 0.927659 | 0.928231 |
| 2334 | Distributed | 0.947969 | 0.951152 |
| 5963 | Regular | 0.942690 | 0.943322 |
| 5963 | Distributed | 0.946060 | 0.945998 |
| 8192 | Regular | 0.939769 | 0.940123 |
| 8192 | Distributed | 0.940709 | 0.939970 |

Overall GPU service decreases5.921percent and wall time5.856percent,passing the
prospective component gate. Timed batches contain no PCIe copy;host submission
and completion CPU time are recorded separately. Regular-map GPU-service
medians for original/candidate/GEMM-only are0.339346/0.314883/0.118001ms at2334,
0.730778/0.689357/0.274141ms at5963,and0.957033/0.899906/0.364504ms at8192.
GEMM-only writes dense BF16 output while fused arms update scattered FP32;
their difference is not an additive epilogue cost or a hardware ceiling.
Resets and host oracles are outside timing;repeated resident inputs are not the
whole-model cache/shape mix. Physical memory traffic,occupancy and integer/tensor
saturation remain unmeasured. Keep individual samples in
[`moe_down_row_address`](../bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json).
The epilogue has no production hook and remains unadopted. The following
matched geometry comparison precedes a separately scoped long-prefill model
candidate;these component results add no measured PP/TG gain.

That geometry comparison now closes successfully at native source`bf330a8f9`,
controller`6e43896a6`and frozen plan`df7900ed6`. The independent M64/N128
policy has64threads and768bytes of requested row metadata,versus128threads
and1536bytes for row128. Both request SG16/256GRF and retain128logical
accumulator cells per work-item;this does not establish emitted register use
or occupancy. Existing row128 bodies and shared GU/decode selectors are unchanged.
Four rotating arms compare both scattered policies and their dense controls.
All six owned processes close0 with no fresh fault:14host ownership cases,
53full post-BF16 FP32 host oracles and30full dense-BF16 checks pass,including
invalid-map masks1/2/3 and63/64/65row and split-tail boundaries.
Three independent timing processes contain288measured and72warmup batches,
128events each;all46080events pass order/interval checks and the smallest batch
is15.064687ms. The paired process-median row64/row128 ratios are:

| M | Mapping | GPU-service ratio | Wall ratio |
| ---: | --- | ---: | ---: |
| 2334 | Regular | 1.017055 | 1.017070 |
| 2334 | Distributed | 0.974093 | 0.973497 |
| 5963 | Regular | 1.002609 | 1.002235 |
| 5963 | Distributed | 0.959994 | 0.960515 |
| 8192 | Regular | 1.009199 | 1.009216 |
| 8192 | Distributed | 0.963742 | 0.962626 |

Overall GPU service decreases1.248percent and wall time1.274percent,passing
the prospective gate while preserving the three regular-map regressions.
Dense64/128 service ratios are1.39..1.51;the scattered result is not a general
GEMM improvement. Different output traffic still prevents subtracting these
times into an additive epilogue cost or declaring a physical resource saturated.
The prefill-only model selector is built from source `9a79cd352`.
It chooses original, row128 or row64 once before runtime initialization;
original remains the default, and decode processing is unchanged in source.
The selected 6e40 fallback is retained. All eight fresh same-build device
checks pass, including 63/64/65-row boundaries, split primary/tail/final rows,
invalid-map masks, and all 32 production dispatch/fallback cases. Actual UR
create-to-launch identities confirm the chosen six split kernels in each
production arm and zero split launches with fusion disabled. Numerical outputs,
guards and immutable inputs pass; all owners close with no fresh GPU fault.
The current-runtime frontend and all three model-policy diagnostics now pass:
each has 117489 fresh read tokens, 117488 GPU-resident BF16 residual rows,
48 windows, actual fused Down jobs, checked tail/drain and 24 thinking-enabled
output tokens. All owners close normally with quiet validation and no fresh GPU
fault. These observed diagnostics do not establish a clean speed improvement.
The superseded standalone Down64 component and closed diagnostic/device output
are retired after original source/recipe recovery; fresh row64 model admission
passes without them. The selected fallback and current same-build fixtures remain.
The clean comparison closes all nine independent processes in three Latin-order
blocks, four fresh 117489-token requests each. First long use is separate from
the median of the next three requests. All 36 requests and final full model-asset
hashes pass; owners close normally with quiet validation and no fresh GPU fault.

| Policy | Independent warmed-process median PP (tok/s) | Geometric-mean time / original |
| --- | --- | ---: |
| Same-build original split | 1002.2 / 1001.7 / 1001.9 | 1 |
| Row128 | 1006.8 / 1003.2 / 1006.0 | 0.99661606 |
| Row64 | 1007.3 / 1009.1 / 1003.5 | 0.99535906 |

Neither candidate passes the original 0.995 time-ratio gate, so neither advances.
All original-split warmed medians exceed 1000 tok/s, but this is not a matched
comparison against selected 6e40 or an adoption. Individual first-use samples,
all warmed samples and client TTFT remain in the committed result. Successful
raw output from all nine processes is retired after verification; current native
source/build, fixtures, basis and frontend records remain for the imminent
original-split versus selected-fallback comparison. Decode/handoff, agent session,
long thinking generation, task quality and physical 262144 qualification remain
pending. See
[`moe_down_row64`](../bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json).

The next comparison uses the retained selected6e40 and the same-build original
split policy, without either rejected row-cache epilogue. The independently
authored [`controller`](../tools/sycl/compare_selected_split.py) prepares a fresh
frontend scope rather than refreshing the old frontend's source hashes. Reuse
the admitted native/fixture packages after exact identity checks. Freeze six
fresh processes in selected/split/split/selected/selected/split order, four exact
117489-read requests each, thinking on and zero reuse. Keep first use separate;
compare subsequent-three medians in the three adjacent pairs. All paired time
ratios must be below1 and their geometric mean at most0.995 to advance. Finish
all processes without speed-based retries or exclusions. This is a whole
prefill policy comparison: selected384 versus split128 effective expert slots,
81920 versus117488 maximum rows and96 versus48 MoE windows must be observed
in their respective actual routes. Decode remains1256 slots. A passing screen
still needs repeated decode/handoff,agent,long-thinking,task and physical262144
qualification before adoption. Nine host rejection controls and two fresh
frontend probes pass without a native rebuild. All six processes now close:

| Process/policy | First use,tok/s | Subsequent three,tok/s | Median prefill,s | Median TTFT,s |
| --- | ---: | --- | ---: | ---: |
| 0/selected6e40 | 876.5 | 972.5 / 977.0 / 971.9 | 120.8134 | 121.042 |
| 1/original split | 927.6 | 1000.3 / 1000.6 / 1007.1 | 117.4150 | 117.576 |
| 2/original split | 931.4 | 1000.5 / 1002.1 / 1005.9 | 117.2459 | 117.396 |
| 3/selected6e40 | 906.5 | 976.7 / 976.1 / 972.8 | 120.3666 | 120.528 |
| 4/selected6e40 | 901.2 | 969.5 / 970.0 / 972.4 | 121.1269 | 121.295 |
| 5/original split | 929.8 | 998.4 / 996.2 / 997.7 | 117.7651 | 117.930 |

All24 exact117489-read requests have zero reuse and matched body hashes;
both hold all117488 residual rows on GPU. Actual selected/split placement is
384/128slots,81920/117488maximum rows and96/48windows,with checked split guard
and drain. Source/runtime/validation and owner gates pass with no fresh GPU
fault. Recorded jobs decrease36.1..36.2percent and dequants about43.1percent,
while requested expert H2D bytes differ by less than0.05percent. These are
logical work/requested-byte counters,not physical traffic or resource-saturation
measurements. The three paired time ratios are0.97187067,0.97407337 and0.97224564;their
geometric mean is0.97272942,a2.727percent time reduction. The prospective
performance screen passes. The separate1000tok/s screen fails because the
third split warmed-process median is997.7tok/s;the other two are1000.6 and
1002.1. Neither the target nor adoption is established. Final full hashes of
all15model assets pass. All24Down dispatch records conserve fused+separate
jobs;selected-arm fusion counters are retained for the next work-mix check.
The reverse pair contains an overnight gap before process3;preserve that
limitation rather than treating the samples as contiguous. The separate96K
case below now passes same-request split-prefill to long-decode continuation.
Default v4 generation uses approximately33Kinput and does not exercise this
route. Native/runtime evidence may be reused only after identity checks;
the benchmark change itself adds no native,task,agent or physical262144 qualification. Successful
raw output from all six comparison processes and consumed research dispatch
transactions are retired after verification;individual samples,decisions and
work counters remain. Both native generations stay only for the imminent
repeated decode/task/context qualification;older legacy provenance holds remain separate. The separate
`split-v1-96k` official-client case is implemented. All35active client tests
(including9new split cases) and9fail-fast integration tests pass;the optional
real-tokenizer fixture is unrun. Against old175c1cb98,the default v4 matches
36wire bodies and6full workload results across two output caps and three
calibration settings. The14controller rejection/finalizer checks and two fresh
current-source benchmark/server definition/mock-loopback probes pass,with
exact Python/project/provider identities and normal owner closure. Actual
candidate long-generation checks now pass;no native rebuild or GPU
qualification follows from the host-only checks. The first owned selected-arm
diagnostic observes96204fresh input tokens and3360generated tokens,including
thinking,5496answer characters,853.8PP and13.4decode tok/s under diagnostic
tracing. Draft acceptance is2397/2892(82.884percent);one sample does not
qualify an MTP regime. Actual96203GPU-resident residual rows use384slots,
81920maximum and96windows. Native exit0,quiet validation,ownership closure
and no fresh fault are recorded. The diagnostic scope nevertheless fails:
Main omitted the host_probe_accepted flag required by the shared finalizer.
Preserve that failed status. The separately committed controller fix and
candidate-only v2 scope close successfully,with normal native exit0,quiet
validation,closed ownership and no fresh GPU fault. The exact historical body
hash matches,without requalifying or counting the failed selected sample as a
new pair. Actual96204read/zero-reuse input produces3360tokens including
thinking,5496answer characters and4340reasoning characters. Diagnostic
prefill is106866.7ms(900.2tok/s),decode248857.9ms(13.5tok/s),client TTFT107.014s
and MTP2397/2892. These are one first-long-use traced observation,not clean
speed or repeated decode preservation. All96203BF16 residual rows stay on
the GPU;128slots and48windows use65536primary rows plus30667tail rows,
314030336extra allocation bytes,checked guard/drain and decode_changed=0.
Configured decode slots remain1256. Job counts conserve6113fused plus18109
separate Down jobs=24222. The candidate and historical selected observations
have the same output hashes and logical decode counts;equal generated text
was not required and does not establish task quality. The native long
request also reports201930cache hits/1851360lookups(10.9percent),1649430CPU
entries and1285620unique CPU misses,with zero GPU expert-copy blobs/bytes.
These are logical placement/work observations,not a physical bottleneck proof.
The source-bound117489-read attribution plan now uses the maintained
`tools/sycl/profile_split.py`:exact indexed UnitTrace assets plus the existing
CPU-wait v4 partition,without command-service profiling. Ten host rejection
controls and fresh frontend probes pass. The v2 prefix/environment check also
passes with normal exit,closed ownership,removed session and no fresh fault.
The v1 prefix remains failed:Main omitted the launcher's exact version and
preload bookkeeping variables from the expected environment. The prospective
fix checks all111fields without relaxing safety checks. No native model ran
in either prefix probe. The one committed117K profile now closes normally:
117489read/zero reuse,24outputs with thinking,quiet validation,removed session and
no fresh GPU fault. Prompt time131729.7ms is1.042499times the median126359.5ms
of three clean first-use samples,passing the frozen1.15observer screen.
The374052790-byte timeline has280298GPU operations and703250host API events,
with unique native-source joins and checked raw-clock bounds. Observed GPU
interval union is117.193s inside134.659s of controlled request time;this is
neither arithmetic utilization nor a full critical path. The main prefill
scope records129.983s wall,123.440s main-thread CPU,79.641s known wait wall
and44.202s CPU outside those waits. The latter includes submission/runtime
work and12.583s unclassified CPU;it is not proof of CPU inference arithmetic
or CPU saturation. The largest single full-shape QSA prefix kernel group
takes26.262s and the GDN recurrence-columns4 group8.541s of nonexclusive
service. The closed shape/family analysis now reproduces the original trace
reduction and matches all7432fused GU and7432fused Down launches to native
dispatch counters. QSA prefix totals27.081s;its main normalized compute
kernel has180calls/26.950s. Observed kernel union is103.908s. M2D copy payloads
total49.443GB/11.557s of summed service;this includes the controlled request's
copies,not exclusively expert transfers or measured PCIe wire bytes. Observed
GPU gaps total17.467s. One3.132s gap overlaps3.131s of`zeModuleCreate`;other
large initial gaps precede the PLE-row H2D stage and require source attribution.
The status wrapper's6.656CPU seconds split mainly into4.438s inside enqueue
and2.216s in its known drain;the enqueue includes called runtime work,not
proved CPU inference math. The native expert path requests38.562GB in18888
uploads,usually1.9712MB or2.6624MB each,in the same in-order queue as its
consumers. This motivates a transfer-granularity screen,not a PCIe-saturation
claim. QSA matched arithmetic/VRAM ceilings remain unknown. The source/family
and next-priority questions are now recorded;retire the complete temporary
timeline after retaining any still-required PLE configuration fixture. Older97cfa/6e40 coverage or
observer cost does not qualify9a79. Physical PCIe/DRAM bytes and arithmetic
saturation remain unknown. Repeated clean decode,task/agent checks and
physical262144 fill/tails/RESTORE remain pending;no inference adoption follows.

Two independent upload screens remain rejected. The extra-queue prototype's144
batches have median candidate/reference ratios1.571..2.212 and zero observed
copy/reader overlap. The single-queue128-expert pre-upload screen passes all432
source/output/bank/guard/drain checks,but fails its prospective performance
gate:the36 group medians range0.437..2.197. Three comparison processes and the
small compat/cancellation/boundary cases close without a fresh fault. See
`prefill_copy_pipeline_v5` and `prefill_copy_batch_v1` in the linked result.
These are synthetic integer-reader screens under the official2026.2 component
runtime,not model speed or a PCIe/cache/staging explanation. V1 creates a new
queue/workspace and performs full CPU checking between timed batches;these
costs are outside its timer. Its original timing eligibility remains passed.
The separately frozen bounded2/4/8-expert comparison now also closes without a
fresh fault. All10small traced CPU-oracle cases and864comparison batches pass
correctness,ownership and timing eligibility. No group passes its36prospective
process/shape gates:geometric-mean candidate/reference time ratios are1.000179,
1.003995 and1.008300 for groups2,4 and8. All observed copy/reader overlaps are
zero. These results reject bounded grouping for this persistent synthetic
fixture;they do not identify PCIe saturation or explain the older V1 variation.
Keep all864individual timing vectors and the compact decision in
`prefill_copy_batch_v2`;no production-consumer or model trial follows. The
component build,normal logs and consumed audit/review are now retired:
44files/3889527bytes. Its677compiled inputs recover as12original Git sources
plus665dependencies matching the retained original official SDK manifest;the
exact source and recipe remain in pushed history. The unused batch helper,
probe,controller and tests are also removed. The host-transfer probe source
remains a required configuration input for the maintained model build recipes.
Prefill and decode implementations are unchanged.
The closed transfer-calibration executable and two consumed receipts are also
retired:3files/360664bytes,after recovering11original Git sources and633frozen
dependencies with the original recipe. Its individual bandwidth samples remain;
legacy SDK/runtime provenance holds are unchanged.
Existing report review now confirms that the actual-model SGLang QSA path at
`877be2974aa151a984e52822c028276c2b1e9303` uses FP32`tl.sum` without prescribing
an exact reduction tree. Main also checked that revision's BF16 attention
comparison test:it uses output`rtol=atol=0.02`,rather than defining an FP32
normalizer/state allowance. The next candidate may change only the32term
denominator fold. Commit the specific comparator,task limits and performance
plan before building it. The earlier exact-target proposal remains deferred
without a GPU run;this source evidence does not inherit numerical/model
qualification or resolve the physical262144/decode-preservation gates.
The isolated source05b870a68and prospective V2plan are committed. Its nine
host evidence-gate checks pass. All eight compiler/link commands exit0,but the
post-link inventory gate rejects an unfrozen GNU linker and13system link inputs;
V2remains build-failed with no GPU execution. All726actual compiler inputs
match the frozen source/SDK. The corrected V3controller queries the linker and
CRT/archive/shared-library inputs under the exact build environment and freezes
their bytes/package versions before a new build. The fixture,generated math,
output criterion and old failure remain unchanged. V3now passes all eight
compile/link commands with726compiler inputs and25link inputs matching the
frozen inventory. Independent source review accepts the bounded sequence.
The host canary,four compiled-property queries,traced17-cell smoke,23numerical
cases and three comparison processes all close normally without a fresh fault.
All96timing batches pass eligibility;the shortest is6.418070seconds. Despite
correct final outputs and selected-state checks,the candidate is rejected:
wall,attention and pack-plus-attention geometric-mean time ratios are1.001793,
1.001782and1.001779. Every one of the12process/scenario medians is slower in
each metric;none meets the frozen no-regression/1percent-gain gate. No model
trial or inference adoption follows. Preserve individual timings in
`qsa_parallel_normalizer_v3`;production prefill/decode remain unchanged.
The rejected official component and consumed normalizer audit/reviews retire
after exact203project/726compiler/25link input recovery and pushed results:
125files/7669913bytes. Keep the96individual timing vectors and compact
numerical/fault/source decision;no normalizer build or raw trace remains.
The next independent candidate shortens Mapping1Top-k radix scans after the
selected set is fixed. A whole selected bucket permits a prefix stop;one
positive-weight block requires its exact key to preserve partial cell budgets.
Source4c99e4198passes four host tests:4752small rank cases,16physical-endpoint
rank cases and uniform-tie controls match independent per-cell numerical sorting.
4140small cases stop before eight passes. This is CPU model/constructor evidence,
not a compiled-kernel speed or correctness result. Production/decode stay unchanged.
The official-SDK normal/counter fixture and controller are now committed at
21c39e7c3. Seven host tests pass in3.995seconds,including paired-sample decision
controls. The frozen component plan covers145numerical cases and144timing
batches across32K/64K/117488contexts in three processes. The official2026.2build
passes all seven compile/link commands with719compiler and25link inputs covered.
The reviewed audit supports exact selection,uniform collectives and fixture
lifetimes,but blocks GPU admission until the controller rejects comparison
processes with no saved radix passes in any timed scenario. Mixed positive/zero
scenarios must remain eligible;keep tie cases and all144samples. Commit this
prospective correction and its all-zero/mixed host controls before execution.
V1remains unexecuted;its original build/source/review hashes stay intact.
The prospective V2controller44636e26c implements the per-process savings gate
and passes10host tests in4.013seconds,including all-zero refusal and mixed
positive/zero acceptance retaining all18groups/144samples. Only the controller
and its host tests change;all other project/native/generated inputs match V1.
Planccfd29440 and the corrected controller now have native Luna-max source
admission. Retire unexecuted V1after original-source recovery and make a fresh
official V2build;commit its actual input/ELF record before properties,traced
tiny,numerics and timing. No GPU execution or adoption follows from host tests
or the inherited source proof.
V1and its consumed source-audit/reviews now retire:74files/2059966bytes after
recovering204project sources and three generated sources,matching719compiler
and25frozen link inputs,and checking all seven command/environment owners closed.
Original build success and blocked device admission remain in the compact result;
the pushed source/recipe commits stay reachable. V2has not inherited its ELF or
any device qualification.
The fresh official V2build passes all seven commands with719compiler and25frozen
link inputs covered. Source44636e26c,planccfd29440 and build record519cd056a now
complete the bounded device sequence:host canary,four properties,traced tiny,
145numerical cases and three comparison processes,all closing normally without
new validation warnings or driver faults. Numeric selected IDs and actual round
counts match;the physical262144selector endpoint passes,not full-model context.
All144timing batches remain eligible;minimum wall/service are0.399063/0.395548s.
Wall/service geometric-mean ratios are0.960609/0.961596 and all18median ratios
are below1,so the prospective component screen passes. Positive-varied scores
save535/482/436radix rounds per256queries at32K/64K/117488;the117488tie case
saves zero and remains in all three process cohorts. Counts exclude final and
unique-key-recovery scans;they do not establish physical memory traffic.
Keep every individual timing vector;reconstructing the original full decision
from the compact cohorts gives an exact match. Next is a prefill-only production
entry bridge and fresh official whole-engine progression,with unchanged decode,
handoff/RESTORE and full physical context gates. No model-wide speed/adoption or
hardware-saturation claim follows from this private component screen.
The closed V2root now retires94files/2344132bytes,including its ELF,objects,
JIT cache and successful traces. Source204/generated3/compiler719/link25input
recovery and all seven build/seven native command/environment owners match.
Keep the144timing samples and source fixtures for the imminent production bridge;
neither selected6e40 nor split9a79engine/required fixtures changed.
The five closed pipeline roots and consumed resource-profile outputs are now
retired after the compact result is pushed:164files/392368585bytes. The final
component's677 compiler inputs recover as12 original Git sources plus665
dependencies matching the retained original official SDK manifest. The partial
failed build's680 dependency paths and exact recipes also recover. Keep only
two generated configuration fixtures totaling3754bytes for the prospective PLE
attribution plan,then retire them when that plan commits the effective settings.
Selected6e40 and candidate9a79 are unchanged. Original failed receipts and the
rejected prototype's source remain recoverable from their recorded Git commits.
Both retained native build generations and their fixtures remain for the named
repeated comparison/qualification;independent legacy provenance holds remain.

The bounded [`metric_capability.c`](../bench/sycl/metric_capability.c) direct-C
check now enumerates24groups and1414metric entries on PCI0000:05:00.0.
All API calls succeed,the three environment/identity gates pass,17observed
runtime ELFs match their independently pinned system closure,and the process
exits0 after1.579s with40.9MB peak RSS,no survivor and no fresh GPU fault.
The first attempt's omitted-system-provider failure remains unqualified at
`899d5ee6d`;the corrected plan uses system IGC2.41.15,its OpenCL-Clang dependency
and the loader tracing layer. This direct-C environment differs from the SYCL
engine environment and does not qualify model or collector interoperability.
Neither comparison engine includes this tool.
The independently authored
[`controller`](../tools/sycl/metric_capability.py) has separate prepare,build and
run actions,with a commit required between them. It pins actual system headers,
compiler/link inputs and the direct-C loader/provider closure independently of
the SYCL allowlist. Native execution has60s wall,45s CPU,1GiB RSS and8MiB text
budgets,three identity/environment/provider gates and no implicit retry.
The probe selects PCI0000:05:00.0 and enumerates group/metric properties without
creating application queues or activating metric groups. Preserve API errors,
permission failures and successful zero-group enumeration as distinct outcomes.
ComputeBasic,MemoryProfile,VectorEngineProfile and VectorEngineStalls each expose
event-based and time-based groups in domain3;names alone do not select a group.
Actual properties include XVE_ACTIVE,XVE_STALL,GPU_MEMORY_BYTE_READ_RATE and
XVE_INST_EXECUTED_XMX_BF16. Their presence does not establish permission,coverage,
physical PCIe traffic or arithmetic saturation. Before using counters for the
MoE resource hypothesis,qualify one documented metric/method with a tiny known
workload,an idle control and observer-cost/coverage checks. The selected83event
properties remain in the query's bound selection receipt. After that selection,
the full1414-entry inventory and superseded enumeration executable retire;
only its provider-closure manifest remains required by query admission.
Enumeration alone does not qualify collection or measure utilization.
Installed flag definitions are EVENT_BASED=1 and TIME_BASED=2;the ComputeBasic
event entry is group3,domain3,83metrics. Select by properties and flags rather
than the index alone. The independently authored
[`metric_known_work.cpp`](../bench/sycl/metric_known_work.cpp) is prepared but
unbuilt and unexecuted. It offers one65536-element exact integer kernel and a
matched-copy control,with checked bounds,drain and physical-device selection.
The copy-only control is not a globally idle GPU measurement. Qualify the
collector's initialization,permissions,coverage and observer cost separately;
this fixture does not establish model throughput or hardware saturation.
For PTI4b6d742 observer-off controls,omit UNITRACE_MetricQuery and
UNITRACE_KernelMetrics entirely:its tracer tests whether those values are
nonempty,so the string0 enables their paths. Query-mode qualification must
also check query/pool cleanup against context destruction;the available CLI
option and the non-counter collector's prior success do not qualify that mode.

The independent direct-C
[`metric_query.c`](../bench/sycl/metric_query.c) and
[`controller`](../tools/sycl/metric_query.py) now build the exact65536-element
OpenCL-C integer workload through pinned clang22 and llvm-spirv. The work-off
process passes every payload/guard check,all four immutable environment/provider
gates and explicit owner cleanup,then exits0 in2.850s. The separate work-query
process successfully activates ComputeBasic EVENT/domain3/83,but
`zetMetricQueryPoolCreate` with PERFORMANCE/count1 returns
`ZE_RESULT_ERROR_INVALID_ARGUMENT` before BEGIN or kernel submission. It
deactivates the group,destroys its owners and exits10 in2.629s;the frozen rule
omits the empty-query arm after this refusal. Both processes are reaped,with
empty sessions and no fresh GPU fault. Preserve this original refusal;it does
not prove a permissions problem or qualify collection. The separately committed
root/child inventory now closes normally in 1.945s with 45.1MB peak RSS. It
returns one root, zero children and 24 groups, including 12 event groups. All
three immutable provider gates pass; MD and ML are mapped after initialization,
all API calls succeed and no fresh GPU fault occurs. Nine host rejection controls
and the Werror build pass. This rules out a root-versus-child trial with the
returned handles; it does not expose ML's internal status or qualify collection.
The topology mode has separate 45s wall,20s CPU,256MiB RSS and4MiB text bounds.
See `metric_query` and `metric_topology` in the compact result. Hardware utilization
and physical bus traffic remain unknown; independent tuning and transfer
measurements continue while counter qualification is pending.

The current-runtime transfer refresh now passes14fresh processes:two traced
2MiB controls,then three processes per allocation/size with one first copy and
three subsequent copies in each direction. All104full payload/source/guard
checks pass,all owners close and no fresh GPU fault occurs. Rates include the
production-compatible copy API and completion wait,with parameter validation
and the qualified copy/direct-submission settings retained. The table gives
the range of the three process medians in decimalGB/s:

| Payload/allocation | H2D | D2H |
| --- | ---: | ---: |
| 2MiB/ordinary RAM | 2.681..4.712 | 3.455..3.869 |
| 2MiB/Host USM | 4.219..4.318 | 4.231..4.380 |
| 320MiB/ordinary RAM | 5.934..6.011 | 5.537..5.546 |
| 320MiB/Host USM | 5.091..5.096 | 5.647..5.648 |

The nine ordinary2MiB H2D samples span0.360..4.771GB/s;keep that variation and
every sample. Do not assume Host USM is a faster H2D placement. Calling-thread
CPU/wall medians are about0.99even without LLM arithmetic. In the320MiB Host-USM
cases,about65.75ms H2D and59.32ms D2H CPU time is inside the explicit completion
wait;ordinary D2H instead spends about60.15ms inside submission. These caller
clocks do not distinguish internal staging from polling,or prove a model's
CPU arithmetic bottleneck. They do not measure physicalPCIe bytes or the
overlapped model critical path. Runtime and compat sources match selected6e40;
the inactive command-profile helper is current source,and this independently
built tool usesC++17 rather than the root engine'sC++20. Its exact recipe,
644compiler inputs,all96comparison copy samples and limits are recorded in
`transfer_refresh` in the compact result. The selected engine and split-MoE
candidate remain unchanged;use these rates for resource budgeting and qualify
the complete split-MoE path before another hardware-saturation claim.

A20261028 source-only check of selected`6e40afc1b` also rules out treating
head release as a configuration-only way to fund the additional accumulator.
`run_prompt_with_native_dense_phase` explicitly rejects
`STRATA_PREFILL_RELEASE_HEADS=1` before reader retirement;the verifier's
`can_release_prefill_head` separately rejects an already-retired dense owner.
The earlier target-head component releases521472000B and restores its weights
and logits,but does not qualify a combined dense/head/MTP/verifier transaction.
Such a candidate needs a joint reader-retirement and restore design,including
canonical/PLE publication,lazy graph recapture and failure unwinding,before
actual allocation or performance tests. Do not remove either guard merely to
make the capacity calculation executable. No source or measurement changed
for this inspection;the existing head-release path remains off.

A20261027 source inspection of the generic
[vLLM Triton expert path](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)
finds separate per-expert outputs in the hidden-state dtype;its
[top-k reduction](https://github.com/vllm-project/vllm/blob/main/csrc/moe/moe_align_sum_kernels.cu)
converts those values to FP32 and rounds the final sum once. The inspected
[SGLang Triton kernels](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py)
also use FP32 reduction. A16-bit serial partial accumulator would round after
each expert addition,so it does not inherit that numerical contract. This
inspection uses moving main references,not pinned model-specific dispatch or
task evidence;pin the relevant path before implementation. Serial16-bit storage
remains a separate task-quality experiment,not an accepted equivalent. Keeping
FP32 partial sums through workspace reuse avoids that new rounding boundary.

One passive current candidate1/request0 sample records96 windows,33225dequant
pairs,37353jobs,7485fused GU jobs/49290450rows,and18937 expert uploads requesting
38673152000H2D bytes. The logged`source_H2D_bytes=225576960` is mislabeled:the
source increments it by`rows*top_k*4` after router readback,so it accounts requested
router-ID D2H bytes. The two uploaded route maps request451153920bytes. Neither
counter measures physical bus traffic;do not add the mislabeled field to expert
H2D or infer PCIe saturation. This single passive sample is retained for the next
window/accumulator question,not a matched baseline work comparison.

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

## Selected fused MoE service

The [current service comparison](../bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json)
uses unchanged selected6e40,1256 decode slots and the indexed PTI aggregate,
without another engine build. Source/controller96bd6c987 and frozen pland7fd4944d
recover the cohort. The39260-token smoke and all three117489-token baseline/profile
pairs close normally with no fresh fault. Both fused families reconcile all46
launch geometries against their fresh request-local successful counts and row
histograms; older census totals are not substituted.

For the first long profile, each family has7485calls and49290450logical rows.
GU/SwiGLU summed device service is4.390441542s and Down/scatter5.580851356s;
projection FLOPs divided by the whole fused service are73.5757 and28.9409TFLOP/s.
These are measured equivalent rates,not arithmetic utilization or hardware peaks.
The three paired prompt-wall ratios are1.02791864,1.04033888 and1.04460423;
their1.04033888median passes the prospective1.05cap. Individual baseline clocks
are130171.1,129815.2,129963.0ms and profile clocks133805.3,135051.8,135759.9ms.
Both arms retain validation,trace and coarse
source counters,so this screens added PTI cost rather than clean production speed.

QSA closed-data analysis at1073485e1,using parser/source43f50485b,separates
one pack and one attention launch per typed call. Both KV heads are inside
each grid. All180calls in each family reconcile against12QSA layers and15
chunks,with two attention and15pack geometries. Attention service is
26.957134826,26.941729303,26.943576593s;pack service is0.131541070,
0.131508993,0.131530757s. Repacking the whole prefix is not a useful priority.
These are exact-family service and source/shape conservation,not an independent
per-execution phase join,full-model critical path or hardware saturation result.
Selected Down already
uses direct FP32 FMA accumulator writes; removing output atomics is not a
candidate. No inference change or speed gain is adopted from this diagnostic.
A CPU-only B570 offline compiler probe at `bae3c5e76`, with frozen plan
`513637504`, now closes the simple regular/native-exp2 comparison. With the
selected compiler and strict math flags, both tiny kernels have the same27
instructions through EOT, including one `math.exp`. The conditional x<-126 zero
seam adds a compare and predicated move. Reject the simple native-exp2 rewrite;
no device program ran and no inference change is adopted. This is not actual
whole-QSA JIT, numerical/domain/FTZ, hardware saturation or speed qualification.
Prior failed GRF/partial-cache/cell/SLM/sum outcomes remain. Adjacent overlap
alone does not establish compatible cross-query groups.
The closed actual-ID census uses the unchanged engine and official thinking-on
39260-token zero-reuse request. Source `afd7fd025`, plan `ba542c888`, host12case
controls, all15model hashes, normal benchmark/native/parser closure, silent
post-closure parameter log and no fresh fault qualify the capture. All12layer
final tails contain6491rows from position32768, with no repeated complete ordered
selection list among77892queries. This rejects exact whole-list packing for those
tails, not partial overlap or the dominant8192-row full chunks.

A CPU-only same-online-slot32cell census at `bce4b8580`, plan `ac7d68b70`, then
conserves all159756492selected-ID reads. Its three finite host controls pass.
Ideal read-once-per-equal-class savings across consecutive4/16/64query windows
are0.0650/0.1586/0.2506percent in aggregate. The largest layer/window case is
1.4857percent, so the prospective1..10percent rule **defers** this direction;
it does not satisfy an all-layer<=1percent rejection. No GPU prototype follows.
These are logical topology bounds, not physical cache or VRAM-traffic savings.

For the117488prefill rows in the service cohort, the source model sums
238865613selected IDs per layer. BF16 K+V payload is2048bytes per ID across
both KV heads; active24-head QK+PV work is24576FLOPs per ID, or12FLOPs per
nominal byte. Across12layers this gives5.870361305088TB nominal reads and
70.444335661056TFLOPs. Dividing by26.943576593s service gives217.876GB/s and
2.61451TFLOP/s equivalents. These exclude cache reuse,physical transactions,
padded-head/tail work,softmax,SLM and spills;they do not establish saturation.
An expert/dense GEMM rate is not this sparse topology's hardware ceiling.
Next source preflight considers subgroup-only K staging, with old row-coalesced
and pitch failures retained;no new inference candidate is selected. One initialized
53252180-byte layer0 ID record serves this immediate source-preflight/comparison;
review it before building and retire it if no distinct legal comparison survives.
The closed full-list/tile census retires83raw/research files (659171689bytes),
its two one-use controllers and the ended exact-list reader mode after verifying
source/results recover from reachable Git. The maintained generic reader/extractor
and the named K-staging fixture remain;this is not a new native qualification.

The [closed K-staging comparison](https://github.com/yqYo1/Strata/blob/ad17ad64b5c4677b5a388caf2526e7e82e9180aa/bench/results/2026-10-15-pr6-recovery-audit/qsa-k-staging-current.json)
now builds from source`e1e9f3bde`:all15commands close at exit0 and699actual
compiler inputs match their pins. The candidate groups K loads within each
subgroup and transposes registers before writing the existing local layout;
online32cell order,attention math and production/decode paths stay unchanged.
The linked SPIR-V contains the intended16bit subgroup exchange. Independent
source/harness review is conditional;Main corrects target-device binding and
exact oracle coverage,with35host controls passing at`2f6b2aec0`. Compiled IR
has eight local pointers;20032bytes is their expected source-modeled payload,
not an observed launch. Preserve the conservative20036byte declaration and
4096byte reserve. The first property-only plan rejects Main's incomplete
marketing-name expectation;its failed status remains,with zero kernel enqueues
and verified child exit. Corrected plan`782d0502f` matches the actual
`Intel(R) Arc(TM) B570 Graphics` and the sole eligible Intel PCI device.
All three kernels pass SG16/WG128/18argument and conservative local-budget
checks:24164bytes against131072per group. Loader-object/symbol checks pass,
validation is quiet and the owned process closes normally without a fresh fault.
Actual-ID smoke and translated final-slot boundary then pass on the B570,
including all18432BF16 output words per shape,immutable inputs,guards,repeats and
the actual20032byte local payload. The independent candidate oracle passes
both smoke cases and all11component cases,including masking,missing/aliased
pages,NaN diagnostics,align2scratch and physical262144bounds. This is not
full-context model fill or state-handoff qualification.

Frozen speed plan`52197a50b` closes all three processes and108eligible samples
without a fresh fault. Every private/original wall ratio passes0.99..1.01.
Candidate/private wall ratios for512queries are1.147429,1.144576,1.147547;
for6491queries they are1.118442,1.118312,1.120846. Attention-only service is
also slower in all six groups. **Reject this exact candidate and do not run a
model trial.** The lower reported spill descriptor12480to3008 is not executed
scratch traffic or a speedup;the pinned API header does not state its units.
No hardware-saturation conclusion follows. The next distinct source proof
packs two16bit values per exchange,aiming to reduce subgroup collective
instances from1024to256 per WG/32selected cells over all256Kdimensions.
It needs a fresh build and qualification;production/decode stays unchanged.
Source`c3bf8dea1` now passes36symbolic masks and216numeric packing cases,
with8192unique source/destination elements per case and three broken variants
rejected. All35host controller controls also pass. The new source model uses
256subgroup exchange instances per WG/32selected cells across all256Kdimensions;
this is not observed device instructions or traffic. Independent source/harness
review now conditionally permits an isolated compile:the new combined audit
and native Luna-max curation verify all nine frozen source inputs. Main rehashes
the eleven installed API headers and compiler before recording that contract.
The property-only sole-device inventory is workstation containment,not native
BDF proof;the first actual-kernel gate must bind the owned child's DRM fdinfo
before release. All15 compile commands now close normally with699 actual
compiler dependencies verified. Intermediate LLVM contains two static i32
subgroup shuffle calls with XOR masks4 and8;this is not final machine-code or
executed instruction evidence. The property-only child closes normally with
zero kernel submissions,quiet validation and no fresh fault. All three kernels
pass WG128/SG16/18argument checks and the conservative24164-byte local-memory
budget against131072bytes per group. Reported spill descriptors are12480 for
original/control and3008 for the candidate;their units are unestablished,and
the rejected eight-u16 candidate had the same3008 value. Actual-ID smoke and
translated final262143-slot checks now pass,with every18432-word output exact
against original32,immutable inputs,guards and observed20032-byte local payload.
The independent candidate oracle passes its2smoke and11component cases,
including masked/missing/aliased pages,nonfinite diagnostics,align2scratch and
physical262144address bounds. These are component checks,not full-context model fill.

Frozen plan`34179e461` then closes all3independent processes and108eligible
timing samples. Original/private control calibration passes in every group,
within0.259%. The packed-u32 candidate is **rejected on speed**:

| Actual queries | Candidate/control wall ratios by process | Attention-service ratios by process |
| --- | --- | --- |
| 512 | 1.154549,1.157028,1.156964 | 1.163768,1.166450,1.166348 |
| 6491 | 1.121215,1.123343,1.122226 | 1.121806,1.123962,1.122858 |

All numerical,ownership,validation and fresh-fault checks pass,while all six
predeclared speed groups fail. **No model trial or production adoption follows.**
Packed uses256subgroup instances per WG/pass versus1024in the earlier
subgroup-u16 candidate. Its4096lane calls are the same256instances in another
unit;the original/private controls instead use direct loads/stores with no
staging collectives. Lower source counts versus that earlier candidate and lower
spill descriptors did not predict a faster kernel. This is not a paired eight-u16-versus-packed-u32
comparison or a physical bandwidth/saturation measurement. Before another
K-staging variation,obtain a source-grounded way to compare the actual selected
NEO-JIT kernel images and their lowered scratch/message/shuffle/local operations.
The component fixture and admission inputs served that bounded diagnostic;
the closed generation now retires after original source/recipe recovery. The source-contract review confirms that
`zeModuleGetNativeBinary` needs a live module,but the installed headers have no
`zeKernelGetModule`;neither a SYCL bundle nor the fixture's property-only
`--admit` proves the dispatched module. The installed loader also exports
`zeKernelGetBinaryExp`,which could retrieve GEN ISA from an actual kernel
handle. Exact SDK source review confirms that its v2 getter selects the first
device. The host-tested observer now binds a sole kernel-program device
to the successful launch's queue device,checks the binary-extension version
and avoids holding its state lock across driver calls or writes. Its recursion
guard requires it to be the sole XPTI subscriber. It proposes copying both
kernel ISA and the same program's module;matching the module text and raw ISA
must precede any instruction claim. Classic Level Zero launch callbacks alone
may miss the adapter's with-arguments path. The host-only observer gate now
passes both builds and21packet cases,including18specific refusals,reentrant
callback bypass,unlocked native queries and exact copied bytes. All24owned
commands close;202source pins and269compiler dependencies are checked. The
packet binary links no UR/LevelZero/SYCL runtime. Actual no-device activation
now passes:one same-PID load/subscription,exact observer/dispatcher init in
loader namespace0,expected46with unchanged version,and a normally closed
owned process with quiet validation and no fresh fault. The following unchanged
three-query GPU smoke also passes all native numerical/input/guard checks and
copies three GEN ISA images plus their modules,each bound to its successful UR
launch and sole program/queue device. Both launches per arm,live B570
`0000:05:00.0` binding,quiet validation,normal closure and no fresh fault pass.
GEN sizes are72000,72000 and74048bytes;spill descriptors remain12480,12480 and3008.
The CPU-only offline gate now matches all three direct GEN copies byte-for-byte
to their exact named executable ELF sections. Installed ocloc1.0 disassembles
unchanged modules through its memory-output API,avoiding the273/303/305-byte
kernel names as filesystem components. All three owned children close0;output
release succeeds,and only the selected assemblies and kernel ZeInfo are kept.
An earlier parser failure remains failed:kernel names also occur in the separate
`kernels_misc_info` root;the corrected parser selects only the `kernels` root.
All three actual ZeInfo blocks report128GRF/SIMD16. Their scratch allocations
are16384/16384/4096,despite the packed candidate being slower. These are static
resource descriptors,not measured scratch traffic or occupancy. Fresh captures
do not retroactively prove historical timed native images. The two bounded reviews now close. Packed adds41static UGM sites and57moves
versus the captured controls,despite smaller scratch allocation;these are not
dynamic traffic or stall counts. Next map the original direct-K source loop to
its native block before selecting a distinct lifetime/unroll change. No timing
or model adoption follows. Closed host mocks
and their one-use source recover from`6c7a3e05d` with frozen plan`21046228a`.
The27closed generation/capture/report files retire (6783939bytes),including
both component executables and the observer. Nine one-use source files restore
from`77e0d370f`;original component/observer recipes restore from`3a3f22931` and
`21046228a`. The subsequent direct-K source audit selected an independent loop
change. The initialized53MBactual-ID fixture now serves the imminent same-build
production/private source qualification for the model candidate. The old
ASM/ZeInfo/LLVM and all auxiliary component binaries are retired;their useful
source-bound facts and original outcomes remain in reachable Git.

The [direct-K comparison](../bench/results/2026-10-15-pr6-recovery-audit/qsa-codegen-current.json)
hoists each lane's stable row read,validity test and row-base calculation out
of16dimension iterations. The finite204-case address/writer/order proof and
independent source review pass. Two separately frozen CPU IR compiles confirm
the row read moves outside the natural loop. The earlier missing-header admission
remains failed;the later probe pins three exact official LLVM504headers.
The native component builds15commands from source`a2f54c3e5`,frozen plan`5c612e3bd`,
with702actual dependencies. All three exact18argument/SIMD16kernels pass a
separate property-only B570 check,with no kernel enqueue or fresh fault and
normal owned-process closure. The candidate spill descriptor increases from
12480to23808;this is allocation metadata,not executed spill traffic.
The traced actual-ID3query and physical262143-address checks pass exact output
bits,guards and immutable inputs. Independent2/11-case tensor oracles also pass,
including masked/missing/aliased pages and align2packed storage. These component
addresses do not qualify full262144-token model processing.

Source/controller`45883c784`,frozen performance plan`a3f64348f`,then closes all108
eligible samples across three independent processes,six rotated rounds and two
actual-ID query sizes. Candidate/control wall ratios at512queries are
0.942146/0.943884/0.942517,and at6491queries0.959880/0.962747/0.960464.
All six groups pass the declared1percent wall/service improvement gates;
original/private calibration stays within0.242percent. Validation is quiet,
ownership closes normally and no fresh fault appears. Thus time falls about
5.6..5.8percent and3.7..4.0percent respectively despite the larger descriptor.
This qualifies a model comparison,not whole-model speed or hardware saturation.

The production overlay constructor`d354a9e40` applies the identical source block
only under compile-time`OrderedExp&&!CaptureState`,retaining the original block
for other instantiations. The next isolated build starts from selected6e40Git
inputs plus this single inference-file change;the current tree's17unselected
native differences are excluded. A committed adapter directs the same-build
independent oracle through the actual production ordered-prefill entrypoint.
The isolated model build then passes18owned commands from source`cd2e308bc`
(controller`06235253b`),frozen plan`8f3ae3dbb`:1411actual compiler inputs,zero uncovered files,four
verified imported link providers and three exact18argument/SIMD16kernels.
Nine auxiliary6e40source files are also recovered solely for CMake target
generation;those targets are not built. The original missing-file configuration
failure remains failed in the linked result and its565closed files retire
(8839659bytes)after source/recipe recovery. The same-source property check and
actual-ID3query/final262143-address gates pass,as do the production-entrypoint
independent2/11-case oracles. Their exact selected attention counts are23/41;
all output bits,guards and immutable inputs match,with normal ownership closure,
quiet validation and no fresh fault. The first property attempt remains failed:
the new controller confused its UR loader with the Level Zero property provider;
source`534a3540a`corrects that and requires a new plan. This does not qualify
whole-model execution. A subsequent prospective link check from source
`2004951af`,frozen plan`de562a846`,matches all232 effective compile commands
and target recipes,with zero uncovered final link inputs. All five relinked
executables are byte-identical to the native-gate artifacts,so those native
checks need no repetition solely for identity. Two fresh CPU-only frontend
definition probes also pass from source`b227036ab`,plan`e16ea9df7`,covering the
changed benchmark/server source and frozen external providers. The older
Python-source refusal remains a refusal. The first production diagnostic then
passes from controller source`ba90ed6bb`,frozen plan`f810dfc29`:39260 fresh
tokens,thinking on,24 outputs,unchanged1256 decode slots,normal prefill-to-decode
handoff and owned exit,quiet parameter validation and no fresh fault.
Observer-charged prefill is57.7638s with57.891s client TTFT;this is not clean
performance evidence. The clean comparison uses six independent processes in
balanced original/candidate order,first long use separate from three subsequent
requests. Controller`e0dafb405`,source snapshot`dc6efcd6b`,frozen plan`8d19a4b0c`
closes all24fresh117489-token requests with quiet validation,normal ownership
closure and no fresh fault. All15 model assets pass full post-run hashes.
Individual samples retain prefill time,PP and first-thinking/content TTFT.

| Independent pair | Original subsequent median PP | Candidate subsequent median PP | Candidate/original time |
| --- | ---: | ---: | ---: |
| Original then candidate | 975.83 | 989.06 | 0.986622 |
| Candidate then original | 971.76 | 989.78 | 0.981790 |
| Original then candidate | 970.16 | 983.60 | 0.986336 |

Each median uses three requests within its independent process. All three time
ratios are below1,and their geometric mean0.9849135073 passes the prospectively
declared0.995 screen. Prefill time falls1.50865percent;the corresponding throughput
gain is1.53176percent. Subsequent TTFT medians are120.561..121.261s originally
and118.919..119.674s for the candidate. First-use samples remain separate;
this is neither cold-disk performance nor a decode measurement. The candidate
does not reach1000tok/s. No physical hardware saturation follows from this gain.

The source-bound small-gain screen passes,but the candidate is **not adopted**.
Controller`8382987d1`,frozen V2plan`a28e950c6`,closes all six independent
`bench.sh --long --long-generation --think --runs 1` processes in balanced
original/candidate order. All108requests,final source/runtime/model-asset checks,
owned native exits and quiet validation pass,with no fresh fault. Thinking stays
on and1256decode expert slots remain unchanged. The strict long-generation cases
read33212/33286fresh tokens and produce2933/8169tokens. All three samples per arm
confirm strong2107/2481(84.9percent)and weak4323/11541(37.5percent)MTP acceptance.
The nominal upstream32K case reads32089tokens and remains supplemental. The prior
plan's refusal for that count remains failed and recoverable from`848896c6f`;
the fresh V2plan does not relabel or pool its diagnostic speeds.

| Process order | Arm | Strong-MTP TG | Weak-MTP TG | Agent-followup median TG | Agent turn4 TG |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Original | 13.7 | 8.6 | 11.2 | 10.8 |
| 1 | Candidate | 13.9 | 8.7 | 10.9 | 8.8 |
| 2 | Candidate | 13.8 | 8.7 | 11.3 | 8.8 |
| 3 | Original | 13.9 | 8.2 | 10.8 | 10.6 |
| 4 | Original | 13.9 | 8.7 | 11.2 | 10.5 |
| 5 | Candidate | 13.9 | 8.7 | 10.8 | 8.7 |

The frozen paired-rate geometric means are1.002427(strong),1.023864(weak)and
0.993931(agent followups). The agent preservation screen therefore fails;the
prefill gain does not justify adoption. All six turn4samples have identical
request/output/reasoning hashes,11722input tokens,114outputs and70/135drafts;
this records the repeated decrease without establishing its cause. Keep the
selected6e40engine,twelve fixtures and1256-slot decode layout. Main stops further
detailed regression forensics and advances the independent selector. Necessary
further cause work belongs to Luna only when concrete new evidence opens a
bounded eligible question. Physical262144fill/tail/handoff/RESTORE is still
pending;successful execution here does not qualify that capacity.

The one-use cohort continuation has closed and retires to source`f27ec72ea`.
After matching the exact final executable's reachable source and recipe,417
closed executable/object/relink and successful observer/cache files retire
(167596328bytes),with a path/hash/reason manifest in the linked result.
The next selector builder's source-only admission passes again after deletion.
It needs the original compile/source/SDK receipts and selected runtime/selector
sources only;that hold is not another retained engine. All twelve generated
cohort configurations/plans reconstruct byte-for-byte from the committed
configuration,plan identities and constructor. The selected engine is unchanged.

Two independently authored batched radix selector candidates are rejected;the
selected6e40engine and decode placement remain unchanged. They replace the
sixteen per-digit workgroup reductions with subgroup partials and two explicit
workgroup barriers. Rank,tie/NaN/zero/tail semantics and ascending IDs pass the
independent weighted-cell oracle. Source-only proof is not performance adoption.

The SG16V2cohort at`2a530e125`passes all native/correctness/event checks and18
independent processes/432eligible batches,but four shape/pattern comparisons
against the original fail. Its full source/recipe/decision restores from
`de8cb3378`;the rejected generation was retired after exact source recovery.
The earlier last-event proxy failure stays failed in`0309ca952`;the replacement
captures actual submitted events directly. Main does not investigate the
provider or historical decode decline further.

SG32V3uses eight partial groups per bin,two bins per subgroup and bounded
lane%8loads,with576B declared accessors. Source`216c4db2c`,build plan`285b7c98a`
closes15commands and695actual compiler inputs,with no uncovered input.
Normal/captured device IR matches after three verified source-identity labels.
Both property processes report SG32for control/candidate,4096B static local
memory and spill descriptor0;these are not occupancy or physical traffic.
All six17-case correctness processes and three small actual-event probes pass.
The frozen performance plan`ff8c65d1a`then closes18processes/432eligible batches,
identical selected IDs,normal ownership and no fresh fault. Full individual
samples and exact recipes restore from`89406c824`.

| Context cells | Scores | SG32candidate/original clean time | GPU service |
| ---: | --- | ---: | ---: |
| 32768 | Varied | 0.975991 | 0.979006 |
| 32768 | Equal | 0.951697 | 0.950650 |
| 78453 | Varied | 1.026957 | 1.027052 |
| 78453 | Equal | 0.988370 | 0.987081 |
| 117489 | Varied | 1.079761 | 1.077841 |
| 117489 | Equal | 1.050277 | 1.051529 |
| 262144 | Varied | 1.074303 | 1.074521 |
| 262144 | Equal | 1.038658 | 1.039340 |

Ratios are geometric means of three paired process medians,each from three
rounds. Eleven of16original/control-by-shape gates fail,so **no model trial or
adoption** follows. The prospectively committed >=1percent all-case clean-wall
and GPU-service rule is unchanged. This is a rejected performance proposal,
not a regression in the selected engine.

In the SG16fixture's117489varied case,original total179.438ms includes179.054ms
GPU service and177.003ms thread CPU inside completion wait,versus1.299ms in the
public invoke region. CPU polling is not host inference arithmetic. Timed work
contains no PCIe copy;physical VRAM traffic,arithmetic saturation and the
whole-model limiter remain unmeasured. The existing7.689s production selection
queue interval includes scoring and host gaps,so it is not selector GPU service.

The CPU-only union census`5e581f6e8`conserves13,313,041selected incidences in
the initialized53MB layer0-tail fixture. Exact two-query unions reduce ideal
logical K/V reads23.304percent,with1.139801times the prospective padded matrix
work. Four queries save39.399percent but require1.798209times that work,so the
first implementation uses two. These are logical costs,not physical traffic,
all-layer coverage or speed predictions.

Main-authored source`75aa272bb`adds a GPU sorted union and a separate two-query
BF16 consumer,with independent query masks and online softmax. WG192/SG16
packs24heads into three8-row matrix blocks and declares21856local bytes.
Typed K/V4staging and ascending FP32 tile sums remain;union tile boundaries
change BF16 probability rounding. Bad packed rows stage zeros while only their
member queries receive diagnostic NaNs,preventing nonmember0*NaN propagation.
Independent Orca Luna and native Luna-max source review finds no algorithmic
blocker under the explicit producer/packed-generation/lifetime contract.
The clarified preflight explicitly caps live KV at262144;the existing validator
limits query rows per call to65535. Neither is full-model context qualification.

The first native attempt refused misaligned byte-mask guards before any kernel
launch. That failed outcome remains in Git. Typed alignment fix`e4aaa758e`
preserves the GPU bodies; corrected native source`e135cfbd2` and recipe`2632db271`
pass nine CPU commands and verify699actual compiler inputs. The subsequent
native property query,38union cases,three eight-case small processes and three
19-case extended processes pass independent tensor/set oracles,guards,actual
launch ABI,provider,validation and fault checks. Extended components reach
physically backed final addresses at262144cells;they do not fill a model context.

The frozen full-path comparison`6f98ef6ea`,result`b842ddecb`,passes36eligible
samples across three fresh processes. Both arms include prefix packing;the
candidate also pays for union generation. Candidate/original wall medians are
0.879607/0.879491/0.879217;their geometric mean is0.879438. The corresponding
GPU-service mean is0.879478. All processes pass the prospective1percent-gain
screen,clock/100ms batch gates,exact constant-V output checks and fresh-fault
checks. This is about12.1percent less component time,not model PP. Inputs use
actual6491-query selected IDs but synthetic zeroQ/K,constantV and Page16;
absolute times must not be substituted for earlier model-profile intervals.
Physical memory traffic and hardware saturation remain unmeasured.

Source`4e6dbe25a`adds the opt-in long-prefill `bf16-prefix-pair` route to the
actual feature tree. Production packing,union and attention share the existing
in-order GPU queue. Its checked maximum8192-query/2051-cap metadata needs an
additional84025344bytes(80.133MiB),reserved before expert-bank planning.
Decode and short-prefill selection remain unchanged. The actual production
objects pass bounded GPU component qualification;the model route remains
**unselected: the clean model comparison below rejects its speed**. The independent Orca Luna audit found a missing pair-only
device check at admission. Correction`ddb0df149`adds one reusable no-submit
predicate at admission and the wrapper;native Luna-max accepts the corrected
source gate. The selected6e40-plus-pair-only559-file snapshot and its frozen
model/production-object fixture recipe compiled successfully and verified1410actual
compiler inputs. Prospective linker tracing resolves33/19/19/20frozen inputs
with no uncovered file,and all four final ELFs match the original build exactly.
The property-only query and38union cases pass. Page4andPage16each pass
three eight-case small processes and three19-case extended processes,including
physically backed262144cell final addresses,independent numerical/set oracles,
production-wrapper equivalence,guards,actual launch ABI,immutable providers,
parameter validation and normal owner closure without fresh faults. These
components do not fill a model context. The first official thinking-on model
diagnostic also passes39260fresh input tokens/39259engine rows and24outputs.
It reconciles60pair calls over12layers/five chunks and471108queries through a
checked drain,with source/model/provider/owner checks and no fresh GPU fault.
Actual placement keeps all39259residual rows on the GPU and the384-slot prefill
bank;the prefix reservation is164743904bytes including84025344pair bytes.
The corrected per-request placement parser passes eight CPU rejection controls
and replay of that actual closed log. This traced first use is not clean speed.
The frozen clean comparison now closes all six fresh processes and24thinking-on
requests,each117489fresh client tokens/117488engine rows and24outputs. All
source/model/provider/owner/validation/fault gates pass. Both arms keep384prefill
slots and all residual rows on the GPU;the pair prefix adds84025344bytes.
First long use remains separate from each process's subsequent-three median:

| Adjacent process pair | Original median PP | Pair median PP | Pair/original time |
| --- | ---: | ---: | ---: |
| 0/1 | 974.7 | 839.8 | 1.160577 |
| 3/2 | 975.4 | 839.8 | 1.161417 |
| 4/5 | 973.3 | 837.7 | 1.161859 |

The time-ratio geometric mean1.161284fails the predeclared small-gain screen;
the pair route remains rejected and original6e40 stays selected. The component's
12.1percent gain did not transfer to actual Page4 model data. No speed sample
was retried or excluded. Generation24does not qualify decode/task quality,
agent sessions,handoff/RESTORE or physical full context. The next finite tuning
discriminator has now closed one matched first-long request per arm with the
existing coarse+fine QSA observers. Source7791d3124corrects a controller-only
acknowledgement omission;the earlier plan remains failed in Git. All12CPU
controls and the fresh v2 source/model/provider/owner/fault/coverage gates pass.
Observer-time ratios to same-arm clean first-use medians are1.004288and1.007279,
inside the prospective[0.95,1.05]sanity screen;these are not causal overhead.
The attention composite grows30.7010to46.5444seconds,while whole first-long
prefill grows130.2063to146.5777seconds. GDN25.1741to25.1782andMoE40.6923to40.7419
stay near each other. This localizes the material new cost to QSA preparation,
union and attention together;queue spans include host gaps,not exclusive GPU
service or hardware saturation. No more Main regression archaeology follows.
The bounded pair comparison is finished. Its original559native,35controller,
9311dependency and1410actual compiler inputs and effective recipe recover from
pushed feature history;all four final ELFs match the traced link proof. Unused
pair exports,targets,fixtures and controllers retire to`6c5212f56`;the
pre-pair tree preserves independent safety fixes. Successful raw output and the
extra generation retire2034files(338393262B),with path/hash/reason manifest
in the linked result;all four roots are verified absent before the next build. Selected6e40and its twelve fixtures remain unchanged. A property-only fresh B570 query now confirms BF16/BF16/FP32/FP32
M4N16K16(andM8),SG16,maximum WG1024and131072local bytes,without any kernel
submission or new fault. Its28files(13445974B)retire after exact Git/source/recipe
recovery. The independent constructor`76366e1c7`implements an isolated uniform
M4/WG96/SG16variant:three real4-head blocks,32cells,and128dimensions per PV
subgroup. All12head writers are unique;the two K/V4staging loop bounds stay4.
Declared local payload falls20036to19188B. This removes25percent of padded
matrix-row work,not measured GPU service or model time;actual compiler resources,
numerical correctness and performance remain unqualified. The selected engine
and decode are untouched. Reused independent tensor/FP32state/guard checks
coverPage4/16and real-head block edges before any timing gate. The CPU build
passes163native sources and696actual compiler inputs with zero uncovered. The
first compiled-property process then refuses the deployed M4coordinate-apply
helper before any kernel submission. Native exits1and is reaped with an empty
session;the incomplete protocol/failed gate remains failed. No fresh GPUfault
appears. Main closes this source variant and prepares public matrix load/store
with explicit FP32SLMrow scaling,accepting its added local-memory cost only if
a fresh full-path comparison wins. No private fragment-layout assumption or
runtime/compiler patch is introduced.
See`qsa_pair_model.clean_prefill_comparison`and`qsa_pair_proposal`in the
[current service decision](../bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json).

After exact source/recipe recovery,the closed SG32generation retired264files
(346618850B). Its source and samples restore from`89406c824`. The completed
pair-component build and53MBinput have also ended their native consumer and
retired292external files(64394955bytes)before the sole imminent model candidate. The original6e40
engine and twelve fixtures remain the validated fallback.

## Attribution limits and outstanding gates

Any necessary detailed follow-up on the agent-turn decline stays with Luna. Main
implements and measures distinct prefill candidates after the rejected selector.
Keep phase-preservation and
adoption checks,plus input-length/time approximations and route-boundary
characterization;further root-cause detail is not Main's next tuning task.

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

The current selected-QSA diagnostic uses the unchanged original6e40 and existing
PTI collector,with official`bench.sh`,thinking enabled and39260fresh input tokens.
The native process exits0,all15model hashes match,and no fresh fault appears.
The original cohort remains **failed**:the controller incorrectly assumed that
`IncludeKernels` also filters the aggregate device table. Closed-data replay
conserves the full126-row table,then independently checks the seven exact
ordered-exp rows:60attention calls covering471108query rows take8.264100184s;
60prefix-pack calls take0.016599873s. These are summed GPU-service diagnostics,
not clean speed,independent UR joins,a lossless execution ledger,observer-cost
qualification or complete phase attribution. The full aggregate includes
startup,warmup and decode. See`selected_qsa_service_v2` in the
[resource result](../bench/results/2026-10-15-pr6-recovery-audit/prefill-resources-current.json).

The same attention kernels report SIMD16,128GRF registers and12480 in
PTI's spill-memory field. The collector labels it as bytes,but the pinned API
header does not establish the descriptor's units or per-work-item scope. It reads
`ze_kernel_properties_t.spillMemSize`;this is not measured physical memory
traffic. Its32-byte
reported static SLM field is separate from the source's20036-byte dynamic
local payload. Register pressure is now a concrete candidate mechanism;its
critical-path cost and the cause of the historical1003-to975decline remain open.
The earlier same-math GRF-only comparison already removed12480-to0scratch
and slowed2.5..9.5% across three shapes;do not repeat that unchanged candidate.
The distinct eight-of-sixteen Q-fragment cache is now **rejected**. It preserves
arithmetic/order and passes all three arms' oracle,guard,input-immutability and
physical262144-address component cases. The reported spill descriptor falls12480to192
at the same128GRF,while dynamic local storage rises20036to28224B. Nevertheless,
three clean processes and162individual latencies show5.76..6.91% longer time
across all three synthetic shapes;the geomean is1.06395846 versus original and
1.06424522 versus the namespace control. No model trial or adoption follows.
See[the compact comparison](../bench/results/2026-10-15-pr6-recovery-audit/qsa-partial-current.json).
This is counterevidence to using spill allocation alone as the optimization
priority. The reviewed cache-working-set/reuse design is conditional on an exact
source-bound production selected-ID,width and page-map fixture;that fixture is
not established by the closed service observation. Recover or capture it with a
bounded committed plan before a production-shaped reuse experiment. Do not repeat
the completed service/counter capability checks. Device-side prefetch remains
indexed knowledge:the configured Xe2 overload/capability and lowering must be
qualified before use. Neither report supplies a throughput gain or physical
bandwidth/critical-path attribution. Preserve decode and global JIT flags.

The source-bound capture now passes using the original engine's existing
`STRATA_QSA_DUMP` hook,without a native rebuild. Nine host corruption controls
pass;the74-token warmup's12records validate at a gate before the long request.
Official`bench.sh` processes39260fresh tokens and24thinking-on outputs. The
646213056-byte transient file contains exactly24expected headers,159788904valid
causal/ordered IDs and1764264uninitialized cap entries excluded from parsing.
Native and benchmark exit normally;all15model hashes match,parameter validation
is silent,and no fresh fault appears. This observed first-use run is diagnostic,
not clean speed or a decode qualification.

The retained layer0 final-chunk fixture has6491queries at position32768,width2051,
and53252180bytes. It selects39127distinct logical cells out of39259live cells.
Adjacent-query overlap has median45.295%;16-query windows have median11651
distinct cells and2.8151logical reuse factor. Their nominal packed BF16 K+V
footprint has median23861248bytes (22.756MiB). These are logical sensitivity
inputs,not measured cache hits,physical traffic or saturation. Whole-chunk
compaction could omit only132live cells for this layer/chunk;this does not justify
another whole-prefix packing optimization. Next compare fixed real-query subsets
across bounded rotating copies,with pack and attention costs separated.
Page identity remains source-derived from qualified full-resident Mode0 admission,
not a captured device table. No Q/K/V values were captured;declare synthetic
tensor values in that calibration. Retain this one initialized fixture outside
Git only until the named working-set comparison closes or is superseded.
See`selected_qsa_capture_v1` in the linked resource result for hashes and limits.
Its77closed raw/log/generated files (658887608bytes) and the one-use controller
are retired after exact204-input recovery;the full result recovers at`ea556dd31`
and the controller at`9d94b2293`. The fixture is the capture root's sole remaining
file. The selected engine/twelve fixtures and shared runtime admission still pass.

The fixed-kernel working-set diagnostic is now closed. Source`2d3e948c4`,
recipe`c30c754fb` and fixture SHA prefix`6d48fddc` retain the actual attention
submission event directly. Three traced smoke,three larger component,one resource
and three timing processes all close normally with exact reference outputs,
guards,input/replica immutability,positive event clocks and no fresh driver fault.
Component cases include physical262144 final addresses,not full-model fill.
Original,full-control and prepacked kernels have equal SIMD16,18arguments,
128GRF,12480-byte compiler spill and32-byte raw-static-SLM descriptors;source
dynamic SLM remains20036bytes. These are not traffic or occupancy counters.

The180individual timing samples use actual selected IDs but synthetic finite
Q/K/V,source-derived identity pages and rotating byte-identical packed KV copies.
The repeated-query256counterfactual and all6491actual-query geometry stay separate.
For full6491queries,the median one-copy wall/service times are124.1591/123.9911ms.
Across the three processes,eight-copy/one-copy wall ratios are0.998122,1.000605
and1.001025;service ratios are0.999161,1.000668 and1.001009. Repeated-query wall
ratios are1.015429,1.031701 and1.021319. Neither shape passes the prospectively
required5%increase in all processes and both metrics. Every timing batch passes
100ms/timer-floor gates;all5940attention intervals pass clock/order
checks. Event-observer overhead is unqualified;these are component diagnostics,
not clean model speed. Per-call rotation changes inter-call reuse;flat curves do
not exclude intra-launch reuse,memory latency,bandwidth or synchronization limits.
No model trial or inference adoption follows from this diagnostic.

The earlier experimental queue-last lookup's invalid-clock failure remains
failed at`117ee9b28`;raw operands were absent and its cause is still unknown.
Direct-event success is a useful avoidance result,not a root-cause diagnosis.
The closed fixture/IR/log generation has no further native consumer;195project
and691actual compiler-input identities and the exact original recipe/executable
are recovered from Git before retirement. Its86component files and seven consumed
research/diagnostic transactions (93files,6777612bytes total) are removed,along
with four one-use source files. No v1/v2/v3 QSA component root remains. The
selected original engine and twelve fixtures still pass shared admission. Keep
only the53252180-byte initialized selected-ID fixture for the named next
same-math barrier comparison,then review it again.
See[the compact diagnostic and180samples](../bench/results/2026-10-15-pr6-recovery-audit/qsa-working-set-current.json).

The same-math tile-barrier fusion is now **rejected**. Source`5463e9736`
kept ordered-exp arithmetic/layout and moved one loop-end barrier after the loop.
Compiled CFG cycles contain five barriers in original/control and four in the
candidate;seven static sites remain in every arm. Independent source-contract
review and three small plus three extended oracle processes pass,including
14scalar cases per extended process,physical262144final-address components,
exact three-arm output,guards and immutable inputs. These are component tests,
not full-model context fill. Three actual-ID smoke processes and one resource
process also pass. All arms retain SIMD16/WG128/18arguments/128GRF,
12480-byte spill and32-byte raw-static-SLM descriptors;source dynamic SLM is20036B.

The frozen comparison uses three independent processes,six rotated rounds and
three arms on512/6491actual-ID query shapes:108individual samples. All outputs,
event intervals and timing-batch gates pass,with no fresh fault. Namespace
control stays within1percent of original in every process and both shapes.

| Actual queries | Candidate/control wall ratios by process | Attention-service ratios by process |
| --- | --- | --- |
| 512 | 0.993953,0.993144,0.995118 | 0.993709,0.993057,0.994245 |
| 6491 | 0.994348,0.996337,0.992612 | 0.994041,0.996331,0.992570 |

The0.37..0.74percent reductions miss the prospectively required1percent in
every process and both metrics. No model trial or adoption follows;selected
engine,decode and dispatch remain unchanged. Keep the108samples once in
[the compact result](../bench/results/2026-10-15-pr6-recovery-audit/qsa-barrier-current.json).
The full source/input/recipe/qualification evidence recovers at`208c591b0`.
All remaining69generated/build/timing files (8095768B) and six one-use source
files (66155B) are retired after source recovery. Selected engine and twelve
fixtures remain unchanged. GPU service includes stalls;it does not prove
arithmetic saturation. The next source-only question is isolated kernel-local
GRF-mode control,with no assumption that removing spills improves speed.

The closed kernel-local GRF API discriminator completes three compilations and
five metadata host controls,with665actual dependencies matched. Source`40aa674b7`
and plan`243977e2c`emit exactly one SIMD16 attention request of128/256per arm,
with no pack request. Their LLVM is byte-identical after namespace/source-path,
GRF request and opaque RegisterAllocMode1/2normalization. This is front-end
request evidence,not native lowering,B570 legality,occupancy or speed.

Historical source`b4beb8d53`already tested the same current-body explicit128/256
handler-property overload,independently of the older functor trial. Installed
`nd_launch(handler,launch_config,kernel)`forwards to that overload. Changing API
spelling supplies no new backend mechanism;no duplicate GPU trial follows.
The independent audit's native review is conditional and keeps exact target
legality/per-kernel admission unqualified. Full source/recipe/LLVM evidence
archives at`52947b6b4`;the failed occupied-name preparation at`7d3e9a3c0`
remains a precompiler refusal. All20raw/generated/IR files (2534695B),the three
one-use source/test files and the checked-out completed result are retired after
recovering164project and665actual dependency inputs. The selected engine and
twelve fixtures remain unchanged. The compact closure in the research index
pins the external deletion manifest;separate legacy holds remain open.

The isolated64-cell QSA candidate is **rejected on speed**. The same BF16
8x16fragments,SG16/WG128 and cached-Q route reduce the width2051outer loop
from65to33updates,but increase active dynamic local memory20032to39744bytes.
The changed online32/64recurrences are checked independently;old32output bits
are not a criterion for64. Full source/build/runtime qualification and the frozen
plan recover at`4f2fffe03`;native kernels are`9b59e9b51`,harness`20927a11e`,
recipe`d05e23ca3`and execution controller`e87dd625a`. See the
[compact comparison](../bench/results/2026-10-15-pr6-recovery-audit/qsa-cell64-current.json).

Three smoke and three extended native processes pass,the latter with51main
case samples and zero final-output error against each own scalar recurrence.
Coverage includes mixed/tail widths,aliases/missing pages,guard/input checks and
physical262144final-address components. The first local-argument observer remains
failed in history:it expected the score pointer at3rather than its actual ABI
index15. A host replay and separately committed corrected plans pass without
changing native kernel math. Property admission queries exact identities and
namespace-0loader bindings;actual traced WG128/18argument launches pass.
The32reported local bytes and4096heuristic reserve are distinct from dynamic
local storage and hardware residency. Spill descriptors12480/13952are not traffic.

The actual-IDfixture then passes three fresh smoke processes and all108timing
samples in three further independent processes. Each shape has six rotated rounds
for original32,private32 and private64,with64/4calls per512/6491-query batch.
The minimum batch is496686737ns. All scalar spots match their own recurrence;
full outputs are finite,guards/inputs are intact and no fresh fault appears.
Across all three processes,the6491shape changes171881of39880704old32output
words,max1.525878906e-5,relativeL2 0.0001828117462. Sparse spot checks and these
deltas do not establish whole-model numerical quality.

| Actual queries | Candidate/control wall ratios by process | Attention-service ratios by process |
| --- | --- | --- |
| 512 | 1.605281,1.604244,1.606402 | 1.641414,1.640282,1.642925 |
| 6491 | 1.560674,1.558906,1.558670 | 1.563451,1.561659,1.561248 |

The private32control stays within0.26percent of original in every group. Both
metrics decisively fail the predeclared1percent-gain gate;no model trial or
adoption follows. Service time increases inside the GPU too. The same valid
K/Vsource-read byte count and only1.54percent more padded matrix work do not
explain the55.9..64.3percent cost increase by themselves;physical traffic,SLM
bank behavior,register effects and occupancy remain unmeasured. This is not a
PCIe/GDDR saturation claim or an explanation of the historical1003-to975decline.

The next pitch-only QSA candidate is also **rejected on speed**. It keeps
32cells,cachedQ,global loads,FP32 arithmetic/online order and barriers fixed,
changing only the V local row stride256to264. Source`39044cc68`,build-source
`57fb85a80`,recipe`ecbdcc1e1`and frozen performance plan`2342c9dcc`recover from
`cfabc1ee2`. See the [compact108-sample comparison](../bench/results/2026-10-15-pr6-recovery-audit/qsa-slm-pitch-current.json).

One property-only child,three traced actual-IDsmoke,three physical final-address
components and three timing processes all pass execution,numerical,guard,
input/provider/clock and fresh-fault checks. Every pitch output matches original32
bits;all scalar spot errors are zero. The boundary case actually selects
262141..262143in the last physical page. It does not fill the whole context
through the model or qualify handoff/RESTORE,model quality or decode preservation.

| Actual queries | Candidate/control wall ratios by process | Attention-service ratios by process |
| --- | --- | --- |
| 512 | 1.002227,1.004882,1.003071 | 1.002581,1.005172,1.003292 |
| 6491 | 1.004809,1.002584,1.003273 | 1.004835,1.002602,1.002874 |

Private32calibration stays within0.138percent of original in all six groups.
All108batches pass the100ms/timer-floor gates;the minimum is496956766ns,and
2448direct attention intervals pass ordering/clock checks. The candidate is
0.223..0.488percent slower in wrapper wall and0.258..0.517percent slower in GPU
attention service. It fails the prospective1percent-gain gate in every group;
no model trial or adoption follows. Selected prefill/decode and the original
engine remain unchanged.

Actual traced dynamic local payload is20032/20032/20544bytes;source allocation
also retains an unused4-byte Qaccessor. All three queried kernels report32static
local bytes and12480in the compiler spill field. The extra512local bytes and
unchanged descriptors do not establish physical traffic,bank conflicts or
occupancy. The actual compiler reports7.1.0pre-release/clang22.1.0 at Intel LLVM
`504366f4b82ff00bbac7b0c956635eed8bc599d8`;all seven installed matrix headers
match that revision. Its BMG matrix restrictions require8-byte-multiple strides
and4-byte-aligned matrix pointers. Source/SDK,harness and API reviews are closed,
including the correct inclusive current-token endpoint and the distinction
between active payload,source allocation and the generator's default layout.

After recovering195original project inputs,31controller inputs,the697compiled
dependency closure,8700compiler assets,generated sources and the exact recipe,
the rejected generation's96external build/raw/transaction files retire
(9104951bytes). Its five unused prototype files also retire (88843bytes);
original source and commands remain reachable in Git. No prototype exports or
build targets are adopted. The selected engine and twelve fixtures still pass
shared admission. Separate legacy provenance holds remain open. Existing geometry
and counter references support the following bounded address-locality diagnostic;
it does not measure physical DRAM traffic or bandwidth saturation.
The source/input audit now proves all6491query counts are2051from the immutable
producer/caller/helper and validates13313041initialized actual IDs. These counts
are source-derived,not a device steps capture;32is the tile size. The ten-command
locality fixture build and both exact-kernel property queries pass,but its first
traced smoke fails the host's raw uniform-V bitwise oracle. The child exits1and
the anchored owner scan finds no survivors;no fresh GPU fault occurs. The failed
cohort remains archived at`16686a2eb`,with retirement proof at`09e3e1c94`.

A source audit finds288negative-zero oracle counterexamples in the three-query
input:raw negative-zero V need not retain its sign through the FP32 accumulator
starting at+0. Whether this explains every observed mismatch is still unproven.
Main corrects the expected recurrence,adds bounded mismatch diagnostics and
separates failed observation/protocol from physical process closure in a new
prospective source/build/plan. Observed GPU bits are never canonicalized. No
model speed gain or hardware-resource conclusion has been adopted. See the
[current locality decision](../bench/results/2026-10-15-pr6-recovery-audit/qsa-locality-current.json).

The repaired fixture passes its own property-only native admission, three traced
smoke processes and three timing processes, with silent validation, closed owners
and no fresh fault. Each of four arms compares 18432 output words in the smoke
and 39880704 in the full shape, including 288 and 623136 expected zero-sign
corrections respectively. Observed bits are never canonicalized. Full outputs
are checked before timing and after every timed batch; guards, inputs and packed
operands remain intact. The original failed cohort remains failed.

All 96 timing batches pass the 100ms and timer-floor gates. Each batch contains
eight calls, with eight balanced four-arm rounds per process and two untimed
warmups. The original uses the captured IDs; the private control differs only in
namespace and host event export. Hot and spread arms use synthetic, sorted,
distinct causal IDs with the same 2051 counts and equivalent packed values.

| Process | Private/original wall | Hot/actual attention service | Spread/actual attention service |
| --- | ---: | ---: | ---: |
| 0 | 1.00113952 | 0.88872091 | 1.09066100 |
| 1 | 1.00116830 | 0.88859290 | 1.09053270 |
| 2 | 0.99976716 | 0.88903499 | 1.09143513 |

All three calibration ratios pass the frozen 0.99..1.01 bounds. Concentrating
addresses reduces attention service by 11.10..11.14%; spreading increases it by
9.05..9.14%. This passes the prospective 3% address-sensitivity screen. It does
not prove cache misses, physical traffic, hardware saturation, the model's
limiting resource or an achievable 11% model speedup. No inference implementation
is adopted and no model speed trial follows from this diagnostic alone.
The compact result retains all 96 samples; the source, generated inputs and ten
build commands recover from the pinned Git recipe. The diagnostic generation,
sole ID fixture and consumed reviews have ended their consumers. After source,
recipe and remote-history verification, 82 external files (61583388 bytes) and
five unused prototype source files (84069 bytes) retire. Both closed research
transaction rows also retire. The original engine and twelve fixtures still
pass shared admission; separate legacy provenance holds remain open.

After this serialized diagnostic closes, the next model attribution
step is one bounded 117489-token selected 6e40 route capture, reconciling planned
work, actual API outcomes and device executions through phase drain. Existing
UR/PTI and oneMKL fixtures qualify their direct calls, not complete model coverage.
The completed fused GU/Down observer-cost gate remains scoped to its original
diagnostic workload; changed instrumentation needs its own cost qualification.
The current raw UR subscriber caps output at 65536 records and its parser at
16MiB. Prove the selected route's planned counts and collection budget before
launching that capture; a budget-stopped capture cannot qualify complete coverage.
The initial whole-model collection in the resource result has native source
`bb71e78db`, rather than selected `6e40`; its histogram is not a qualified
selected-route distribution. Later nested fixtures retain their separate scopes.
For the next selected-source census, `6e40:src/prefill/moe_window.inc` already
prints planned job histograms after the router readback and host route build,
before the job loop, when `STRATA_PREFILL_LAYER_MAJOR_PROFILE=1` is set before
exec. Shape census alone does not enable this output. The route header pins
512 experts, top-k 10 and jobs of at most 8192 rows. Actual counts remain dynamic;
these constants do not establish the selected request's job distribution or
record budget. Reuse `bench_engine.extract_planned_moe_work` for ordered window,
row/expert conservation and planned-versus-retired totals before extending the
API/device ledger. This census is planned work, not complete device coverage.
The reader is now independent of optional CPU/GPU observer frames. Source
`3115b670a` passes six host-only planned-work tests and nine existing resource
admission tests, run serially with `/usr/bin/python3.12 -I -S -B` on
`tests/core/test_planned_moe_work.py` and `tests/core/test_resource_engine.py`
under the shared lock and a 60-second deadline. Controls include missing,
duplicated and reordered windows, malformed histograms, ambiguous retired
counters and configured capacity larger than actual rows. These synthetic
controls do not supply a selected-model histogram or qualify native execution.
The selected-model census keeps204 pinned controller inputs,58 unchanged child
sources and the original engine plus12 same-build fixtures. Three initial cohorts
remain failed in Git:wrong acknowledgement (`55b0ae1b0`),normal-summary refusal and
incorrect short extent/capacity (`2c11c2689`),then a nonexistent official sample field
(`5eaab6d99`). All native children exit0 and no fresh GPU fault is recorded. These
are controller failures;the completed model requests do not retroactively pass
their frozen plans. The original report reviews missed these contracts.

Controller`cc59c96f0` now uses the actual newline-terminated gate command,drains the
bounded human report while rejecting extra gates,and reads the official sample
schema. Five host tests pass:real begin/end/summary handling and rejection controls,
plus full replay of the actual closed v3 request and five corrupted sample variants.
The original failed run stays failed. The unchanged short request prospectively
has39260input/39259prefillrows and capacity39259;the unchanged117489-input case
has117488rows in81920+35568 windows. The fresh v4 smoke now passes all48windows,
18796jobs and18844320routed rows,with matching retired counters,actual native/Python
provider checks,closed owners,silent parameter validation and no fresh fault.
Requested router D2H is75377280bytes and map H2D150754560bytes;these are logical
extents,not bus measurements. The short histogram ends its validation consumer;
the next capacity planner needs the long distribution. The117489-input case now
passes its final source/model/provider/owner/fault checks:96windows,37353jobs
and56394240routed rows. Both actual native and benchmark children exit0;
parameter validation is silent and no fresh fault appears. Requested expert
weight H2D is38673152000bytes;router D2H225576960bytes and map H2D451153920bytes.
These are requested extents,not measured bus traffic or additive transfer time.
The source-planned routed GU/Down work is554377936896000FLOPs with two operations
per FMA;this does not supply an attained or maximum arithmetic rate.

Across second windows,the permanent bank hits14288of14981expert encounters.
Only693misses can include a repeated upload,and some are new experts. Even a
loose bound using the largest2662400-byte packed expert is1845043200additional
bytes. The source uses a permanent prefix plus scratch,not LRU;the counts do
not support a large LRU-thrashing explanation or establish a cache speedup.
The exact4550-bin long job-size distribution remains only for the imminent
API-capacity/matched-operator planner. Per-layer lists and the short histogram
have ended their consumers. Frozen full inputs recover from303d012e5;
the closed one-shot controller and consumed historical replay tests recover
fromcc59c96f0. Keep the generic planned-work parser and its maintained tests.
See the [current census](../bench/results/2026-10-15-pr6-recovery-audit/model-work-census-current.json).

The job loop requests at least171795GPU producer operations and its expert
dequants at least66450more. If each producer generated one UR begin/end pair,
that alone would produce476490records,before initialization,dense/QSA/GDN,
decode,memory validation and lifecycle calls. This is a source-based planning
estimate under an explicit API-mapping assumption,not a complete API count or
upper budget. Do not run the existing65536-record subscriber on the whole
model unchanged. Plan finite loss handling,phase coverage and observer cost
before collecting the actual API/device ledger.
The new subscriber loss handling passes39parser tests and21native CPU controls
from source47567e9ab. Seven caps,including2/3/4/5/8/511and the65536default,
stop recording with one explicit overflow marker while four callback threads
and the application continue. A late invalid argument is not read after stop;
the parser refuses even a paired prefix. Both original metadata fixtures,four
environment-write guards and eight invalid-cap controls pass. This qualifies
host callback behavior only. The candidate observer has not been loaded into a
GPU/model process,and disk/format errors still retain their old fail-fast behavior.
Static review then found a missing boundary:the first omitted callback still
decoded metadata before checking the cap. The original21passes remain valid for
their recorded controls,but do not cover that boundary. This candidate's four
native/manifest files are retired after source/recipe recovery;no GPU/model
process used it. The replacement reserves its row under the output lock before
reading callback data,and adds invalid first-omitted begin/end controls.
The replacement from023c7cc2b passes all23native CPU controls,including invalid
argument data on the very first omitted begin and end. Controller552a93efa also
passes two actual host UR canaries with complete capture and cap3. Both preserve
the expected error46 and UINT32_MAX output sentinel,with before-main validation,
silent parameter checks and closed exit0 processes. The complete arm records
two actual rejected API pairs;the truncated arm emits one loss marker and is
refused by the reader. The independent UR trace contains no device/queue/enqueue
work. These actual return/output controls still do not qualify a GPU/model
observer or its cost;ordinary runtime teardown remains unqualified.
Keep the qualified v3 observer until separate device and observer-cost gates
select a replacement;the65536maximum and16MiB parser still do not permit a
complete whole-model capture. The selected engine and decode are unchanged.
This reader is specific to the pinned6e40 schema. Neither host replay nor the
planned census qualifies API/device coverage,physical traffic,clean speed,decode,
quality,physical262144context or model adoption.

Complete prefill phase attribution requires three independent ledgers: route-owner
planned operations, actual API results and collector device executions.
[`work_ledger.py`](../tools/sycl/work_ledger.py) implements the host reconciliation
gate, including graph-child expansion, lifetime generations, rejected submissions,
host-task completion, shapes/requested bytes and drained phase bounds. Its
18 adversarial host controls pass using synthetic records only. Actual UR/PTI
adapters, model-route coverage and observer cost still require qualification
before complete-ledger claims. On current NEO, the retained6e40 submission fixture
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

The selected original6e40 rejects command-service profiling with GU/Down fusion.
The separately qualified observer below extends those actual fused kernels;an
unfused profile changes processing and cannot substitute for it. Its component,
small-model and observer-cost checks are already closed. The library-ledger gap
above remains separate from this scoped event attribution;do not rebuild or
repeat the completed observer merely because an older paragraph calls it pending.
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
pinned generated header. The later source022652cc5 diagnostic engine passes its small-model gate and
a six-process/twelve-request observer-cost comparison. Time-on/off ratios are
1.00501452,1.00539514 and1.01639250,geomean1.00892029;the frozen gates pass for
that binary/workload only. Scoped96-window CPU/status and sampled copy/GEMM
attribution is already recorded under`attribution_after_observer_cost` in the
resource result. The diagnostic engine is retired after exact source/recipe
recovery. This does not qualify complete device coverage,physical traffic,
hardware saturation or a new observer build. Use those closed results and a
new concrete hardware/whole-model counterfactual before collecting again.
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
2026-10-29T21:31:36.563996Z at`46c152a1310f77cd2f0ce2c7cd213962df14fe83`.
The committed upstream-status.json tracks the partial functional reviews and
ports,including CPU-name/missing-x86-flags reporting and stream-parser scan
slices from this interval. Required tracking/recovery/alias/API/CLI functions,
earlier functional obligations and hardware-tuned unmodified-upstream
measurements remain incomplete. Check again around2026-10-31T21:31Z using
the project-only sync skill. These dates reconcile the existing committed
status;this document edit is not another remote check or a completed import.

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
