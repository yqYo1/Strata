# Native CPU streaming service calibration

Root has passed standalone compile/link/import checks for this target. Cohort execution and statistical qualification remain pending. The prior `native-service-host` actual-weight smoke is a separate correctness result and its tight-repeat samples are not reused here. All execution, payload access, qualification and cleanup belong to root under the shared measurement lock.

The program handles one exact GU/Down/NT cell, one task policy and one homogeneous pool batch size per fresh process:

```
native_service_calibration PACK_DIR PRIMARY_GGUF COHORT_TSV GU DOWN NT TASKS BATCH SEED
```

GU is 18/21/22/23; Down is 20/42; NT is 1/2; TASKS is 0/6; BATCH is 1/6; SEED is a decimal uint64. The TSV is at most 16384 bytes, requires a final LF, and has the exact header `cohort<TAB>layer<TAB>expert`. Each following line is `train<TAB>layer<TAB>expert` or `holdout<TAB>layer<TAB>expert`. There must be exactly 192 unique train and 192 unique holdout IDs. Layers are 0–47, experts 0–511. Duplicate IDs are rejected globally, including cross-cohort overlaps. No automatic selection, replacement, format pooling, or holdout promotion happens. Root freezes the list from source/route evidence before looking at timing; eligibility for an exact GU/Down cell is not inferred from broad Down-only sufficiency tables.

Use immutable pinned production pack metadata and primary/shards. The production layout and NativeRolePlan validate all 48 layers' role extents, then selected owned full blobs are assembled. No engine parser is copied. Selected rows must have identical exact GU/Down types, H=2560, FF=640, row sizes, activation types/sizes, offsets and blob geometry. Each cohort's GU Gate+Up bytes and Down bytes must individually exceed 64 MiB. All 384 owned full blobs together must fit within 1 GiB. Root binds SHA256 identities and selected role extents before launch and rechecks original pack identity afterward. The program records extent paths/selected offsets/bytes and FNV fingerprints; these do not replace SHA256 provenance. GGUF mappings close after assembly. No expert dump is saved.

Inputs are fixed by layer/token only: `((i*37+layer*11+token*101)%509-254)/257.0f`. They are finite and NT2 inputs are distinct. All experts in one layer share the exact same quantized input bytes, across cohorts. Production input quantization and its pinned GGML meaningful-byte comparison happen once per used layer/token before service timing. Input quantization is explicitly excluded from every measured service interval.

Buffers are preallocated and bound before measurements. Owned weights, inputs, FF, quantized intermediates and outputs are naturally first-touched during assembly/untimed correctness preparation. No cache drop/reset/prefetch policy is installed. A streaming working-set floor is not proof that all subsequent loads come from DRAM. NT1 GU and Down20 require byte-exact independent per-row GGML outputs; production act and Down20 FF quantizers match pinned GGML meaningful bytes, with untouched tail guards. Down42 exactly mirrors production `act_quant_any` + `q2_rows_any`; ActQ meaningful fields and bitplane fields are checked against the selected production quantizer and the initial reference, excluding padding. NT2 and Q2 differences from independent GGML row controls are characterization only, with no new tolerance. The GGML Down control uses actual direct FF converted into GGML activation bytes; the Q2 ActQ route differs. This is not an independent NT2/Q2 full-FFN qualification.

All 384 experts pass untimed direct/reference and actual homogeneous pool partition checks before timed rounds. Each subsequent round checks finite full-row outputs/intermediates, sentinels, meaningful quantized fields and byte-exact repeats/partition results against immutable prepared references after the end clock. Input bytes are reverified outside every interval. Checksums and output happen after timing. Fixed buffers are unique per expert/token; pool job pointers and batch arrays are prepared before the start clock. No harness per-expert clock/check/hash/printing occurs inside a measured round.

Five distinct arms:

- GU: exactly one production GU call per expert; no FF quantization or Down call.
- FFquant: exactly one full FF quantization per expert/token, using fixed previously validated FF; no GU/Down.
- Down: exactly one full Down row call per expert, using fixed previously validated quantized FF; no GU/FFquant.
- direct_complete: one GU, FFquant and Down sequence per expert; one measured outer round duration, not a sum of medians.
- pool: `run_split_multi_native(f,jobs,n)` for each homogeneous batch, n=1 or6. Retain whole-round wall time and deltas of production `ms_multi_gu`, `ms_multi_q`, `ms_multi_down`. Production internal phase clocks/joins are unchanged. Never divide batch duration by jobs and label it job service.

Each arm runs three whole-cohort warmups, then 25 full shuffled rounds, separately for train and holdout. Each cohort ID appears exactly once per round. The program groups TSV rows into train then holdout while retaining within-cohort file order. Schedule generation is SplitMix64 (uint64 wrapping addition/multiplication with constants in source), with one shared state initially equal to SEED. Visit cohort0 then1, rounds0..27; initialize indices cohort*192..cohort*192+191, and Fisher-Yates i=191..1 with `j=next64 % (i+1)`. This is a bounded deterministic modulo shuffle, not a claim of unbiased random sampling. Identical round orders are reused across all arms. Order FNV hashes serialize global TSV indices as four little-endian bytes each. ID rows bind those indices to actual layer/expert. First three orders are warmups; remaining25 are samples. Root freezes the seed before execution.

The pool is created once, with five pinned workers, host participation and TASKS chosen by root. TASKS=0 means18 logical GU/Down row tasks per batch for six participants; TASKS=6 means6. BATCH=1 gives192 pool calls per round; BATCH=6 gives32. Pools with different TASKS/BATCH profiles run in separate fresh processes. Production host affinity pinning covers all calls and RAII restores it after pool destruction. Planned CPUs, actual host CPU and Linux thread allowed CPU-list snapshot are emitted outside round timing. The snapshot does not associate worker index with TID or prove actual CPU residency; root reconciles it with the declared topology. Null GPU-release callback is required throughout. No engine callback is registered.

CSV record tags: META, ENV, EXTENT, ID, REFERENCE, WORKING_SET, PLACEMENT, ROUTE, TASK_PLAN, WARMUP, ROUND, RESULT. ROUND has cohort, arm, round0..24, GU, Down, NT, tasks, batch_jobs, batches, logical_jobs192, order_fnv, outer_ms, pool_GU_ms, pool_Q_ms, pool_Down_ms, output_fnv, host_cpu. Direct arms label batch_jobs1/batches192 as calls, not pool batches. Pool phase values are zero for direct arms. Exactly250 recorded ROUND rows and30 WARMUP rows are expected. stderr gets production `pool.diag` after untimed admission and after pool rounds, outside round intervals. RESULT/correctness_pass describes the program's finite/reference/quantizer/repeat/partition checks only. It does not self-certify statistical acceptance, representativeness, full lifecycle or adoption.

Root-only recipe, not executed:

```
cmake -S "$WROLE/sycl/tools/native-service-calibration" -B "$CAL_BUILD" \
  -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DSTRATA_ROOT="$WROLE" \
  -DGGML_SOURCE_DIR=/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml
cmake --build "$CAL_BUILD" --target native_service_calibration --parallel 6
"$CAL_BUILD/native_service_calibration" "$PACK" "$PRIMARY" "$COHORT_TSV" \
  "$GU" "$DOWN" "$NT" "$TASKS" "$BATCH" "$FROZEN_SEED" \
  > "$PRIVATE_RUN/rounds.csv" 2> "$PRIVATE_RUN/pool-diag.txt"
```

New private result/build directories; no v1 overwrite. CMake copies v1's exact production CPU closure/per-TU ISA, IntelLLVM precise FP policy, pinned static native GGML/Threads, IQ AVX-VNNI=0, IQ2S GCC absent, and no parent SYCL project. Q2 compile support must be1, matching the qualified host binary. No compiler substitution, global Strata march/SYCL flags or GPU backend. Root must inspect actual build flags and source hashes, clean pinned ggml, and runtime imports for this new binary; prior binary qualification is not inherited automatically.

Root first reviews/builds and runs source-negative admission cases (wrong TSV/header/count/ranges/global overlap/format/geometry/byte floor), then a separate correctness/no-device observed run. Timing runs must use a clean frozen environment: no profiler, LD_PRELOAD, LD_AUDIT, LD_DEBUG, strace, diagnostic histogram or fake ISA environment. The program refuses named loader/debug/feature-forcing variables but cannot detect an external tracer universally; root owns this separation. It never starts profiling/tracing. Use the shared measurement lock and root finite wall/CPU/RSS/address-space/FSize/NOFILE supervision. Proposed output caps: stdout2MiB, stderr2MiB. Owned blobs<=1GiB plus roughly tens of MiB of reference/work buffers, ggml/pool overhead and full-shard virtual mappings during assembly; source caps are not RSS guarantees. Existing production pool60s stall watchdog may abort; root records nonzero/signal/fault/cleanup status and keeps a closed failure receipt. Normal exceptions return1.

Finite work: 2 cohorts *5 arms *28 rounds *192 experts, plus bounded initial correctness. No per-round order generation or allocation enters timing. Root owns R46 bootstrap/relative CI/holdout postprocessing and cannot substitute the three-warmup/25-round schema for statistical acceptance. Disjoint expert holdout is not independent workload data; A/B route fixtures and repeated prompts remain dependent. No mixed callback NT, GPU overlap, weighted/shared expert, actual decode activations, DRAM bandwidth, end-to-end time, LRU/policy, performance adoption or full262144 qualification is claimed. Preserve R77 and R78 correction history; v1 remains unchanged.

Within each cohort the implementation visits round0..27, and executes all five arms in a rotating block per round, instead of finishing all rounds for one phase first. The base arm sequence is GU, FFquant, Down, direct_complete, pool; position is `(arm_index + round_index + SEED%5)%5`. This balances each phase across five starting positions over25 measurement rounds. Both cohorts use that same phase rule and the precomputed expert order for each round. Root must still observe boost/thermal/load and counterbalance fresh-process profile/cell ordering; this finite rotation alone is not proof that time drift is absent.

For root's separate non-timing loader/syscall correctness observation, append `--correctness-only`. This mode runs all384 initial direct/GGML/quantizer and homogeneous pool checks, affinity snapshot and output checks, then returns before any WARMUP/ROUND intervals. It permits exactly `LD_DEBUG=libs` for R78's loader observation, still forbids LD_PRELOAD/LD_AUDIT, and emits `RESULT,correctness_only_pass,no_calibration_rounds`. Timing mode always rejects LD_DEBUG. A traced correctness-only process cannot supply timing samples. Neither mode starts a tracer/profiler itself. Owner admission must require normal exit0 and verified complete outputs; a parseable RESULT before a failed final flush/close is not sufficient.


Root records `STRATA_IQ_PREFETCH` in environment output. Its default is 2048
bytes; zero disables the production IQ prefetch. Pin the same value or absence
in every compared process. The initial compile-only receipt remains historical;
adding this metadata does not change the kernel or task policy.

Root found the first streaming timing runs placed all workers on host CPU0:
the fixture narrowed caller affinity before constructing the production pool,
whose topology detector then saw only that CPU. Those original output-correct
receipts remain, but their timings are invalid for service calibration or task
comparison. Construct the pool under original caller affinity, then pin the
host. Root must require actual singleton thread CPU sets to match the declared
host plus five distinct workers before interpreting any new timing. This fixes
the standalone fixture only; production initialization order is a separate audit.

## Opt-in actual-cohort IQ2S three-arm check (source-only, unexecuted)

Configure a fresh private build with the recipe above plus
`-DSTRATA_CALIBRATION_IQ2S_INDEX_CHECK=ON`. The default is OFF; an OFF
binary rejects this mode before layout/model payload opening. Root-only invocation:

```
"$CAL_BUILD/native_service_calibration" "$PACK" "$PRIMARY" "$COHORT_TSV" \
  22 20 1 0 1 "$FROZEN_SEED" --iq2s-index-correctness-only \
  > "$PRIVATE_RUN/index-check.csv" 2> "$PRIVATE_RUN/index-check.stderr"
```

This mode preserves the existing bounded TSV, production layout/NativeRolePlan,
frozen order, full owned blob assembly and production layer-input preparation.
It requires exactly384 IDs, H2560/FF640, GU22/Down20, NT1/tasks0/batch1 and
756940800 owned bytes. Down20 K640 is not sent to the index dot.
It branches before full FFN correctness, pool creation, scheduling, warmups and
timing. Default calibration and `--correctness-only` keep their original arms
and finite/reference/partition requirements. The new mode is a separate dot
check, not a substitute for those gates.

Each of245760 Gate/Up row pairs uses identical prepared production Q8_K bytes,
rechecked against the pinned GGML quantizer per ID. The three arms are the
IQ2_S CPU trait, the copied noinline direct control and the copied noinline
register-index candidate. Each computes Gate, Up and the existing scalar finish
`(g/(1.f+std::exp(-g)))*u`. All IEEE bits, including signed zeros, Inf and NaN
payloads, enter equality; finite/Inf/NaN counts are separate. No finite filter
removes actual rows. There are491520 role rows,1474560 dot calls and1474560
comparisons against the baseline (two other arms times three values).

`INDEX_ID` records split, global index, layer, expert,640 row pairs, mismatch
count, result FNV and activation FNV, followed by27 counts ordered
arm(baseline/control/register), value(Gate/Up/finish), class(finite/Inf/NaN).
`INDEX_SPLIT` contains split,192 IDs,122880 row pairs, mismatches, FNV and those
counts. FNV consumes each FP32 word as four little-endian bytes in row/arm/value
order. At most one `INDEX_FIRST_MISMATCH` records index/layer/expert/row/arm/value
and both uint32 words. `INDEX_COMPLETE` is a coverage marker; only zero mismatches,
successful flush and normal child exit0 support the final pass. Existing EXTENT/ID
records remain; root must verify384 ID rows and1152 extents. No raw expert, Q8 or
per-row result trace is written. Keep root's existing8MiB aggregate text budget,
1536MiB group RSS and finite CPU/wall limits; source caps are not RSS guarantees.

MXCSR is observed at entry, after preparation and completion; control bits
(mask0xffc0) must remain unchanged, without setting modes. The host uses production
affinity pin/RAII restore. Workers are never started, so worker MXCSR is explicitly
not observed. This mode supplies no worker-mode evidence.

`iq2s_index_dot.cpp` retains the MIT notice and copies the helper/table/dot source
tokens from `native-iq2s-nt1-index-spread/iq2s_index_spread.cpp`, SHA256
`c51f580bc72cb303e7fe30b1cfb6f7c09ad82332ffd60d441af0eac2dbe77b89`,
through the end of `index_candidate`, excluding unit fixtures/main. The included
GGML codebook remains pinned to commit3cf03257f219afbe7334045ff7c6a06ac68c627d.
Only this new TU gets AVX2/FMA/F16C/precise; other production TU flags are unchanged.
The retained old out-param helper is unused by the hot candidate; no index array,
gather, scale table, pairing or accumulation change is introduced.

Root must close copied-role identity with the existing frozen1152 per-extent
SHA256 list (`568892bce458dbe96cd63c67e6c9cf89ba9b0dc54b93eac401b138c0c6fc7cc6`)
and manifest/TSV pins before admitting actual-weight results, keep the source
stable/read-only under the serial lock, and recheck source identity afterward.
Printed FNV/paths/offsets do not replace that evidence. The existing mapped reader
is a trusted-stable-source contract, not an adversarial concurrent-writer proof.
Root must independently verify linked trait/control/candidate ISA/arithmetic and
custom `native_quant_act` quantizer FP instruction ordering; the R100 reference
quantizer capture does not establish that custom branch. Release and IntelLLVM
ASan/UBSan, OFF rejection before opening payloads, wrong-mode/cell negatives,
unchanged default correctness/timing behavior, normal exit/flush and no-GPU
observations remain pending. No actual weights were read by the implementer.
Synthetic layer activations are not live model activations. No timing,32K model
parity, performance, adoption or full-context qualification follows from this mode.

## Private actual-weight dot timing (source-only, unexecuted)

The same OFF-by-default `STRATA_CALIBRATION_IQ2S_INDEX_CHECK` option admits
`--iq2s-index-timing-only`. Root uses a new private release build and result
location with the existing IntelLLVM/pinned GGML recipe and exact cohort pins:

```
"$CAL_BUILD/native_service_calibration" "$PACK" "$PRIMARY" "$COHORT_TSV" \
  22 20 1 0 1 "$FROZEN_SEED" --iq2s-index-timing-only \
  > "$PRIVATE_RUN/index-timing.csv" 2> "$PRIVATE_RUN/index-timing.stderr"
```

OFF/invalid-cell rejection occurs before pack opening. Entry and prepared caller
MXCSR control bits must be exactly0x1f80 (nearest-even, FTZ/DAZ off), and
STRATA_NO_Q8K_AVX2 must be absent. Timing always rejects LD_DEBUG, LD_PRELOAD and
LD_AUDIT. Root must keep clean STRATA settings and independently verify the new
binary's linked dispatch/quantizer/dot paths. No modes are changed.

Before any interval, the unchanged384-ID three-arm Gate/Up/finish exact admission
runs. Its correctness RESULT is an admission marker only; timing success also
requires INDEX_TIMING_COMPLETE and the final timing RESULT/normal exit0/flush.
Immutable pretiming expected Gate/Up bits are then saved from the actual trait.
Output buffers and all row/input/output pointers are preallocated/prebound. Each
round verifies every output against these saved bits, including nonfinite bits,
and checks guards and input quantizer parity outside the clocks.

Two fixed strata per split: stream192 visits all192 frozen-order IDs, Gate+Up
201523200 bytes; hot8 selects within-split indices0,24,48,72,96,120,144,168,
Gate+Up8396800 bytes. INDEX_HOT_ID enumerates the IDs; INDEX_WORKING_SET records
count/bytes/order FNV. No cache flush, cold-load or DRAM claim. All384 owned full
blobs remain756940800 bytes; Down payload is not accessed by the timed dot arm.

Each split/stratum runs3 warmup rounds then18 recorded rounds; every round has
trait/direct/register arms with position `(position+round+seed%3)%3`. All arms
traverse identical bound row pointers in frozen ID, row, Gate/Up order. Eighteen
recorded rounds balance six samples per arm position. Splits and strata are
visited in fixed split0/1 then stream/hot order; root must counterbalance fresh
processes separately to assess drift. No automatic long timing or profiler.

The host steady_clock interval surrounds the arm switch plus the bounded dot
call traversal and preallocated output stores. It includes real trait ABI/call
and loop overhead; no dispatch subtraction or kernel-only claim. The trait uses
its real eight-argument ABI; direct/register helpers use their typed three-argument
ABI. No finish, Down, quantization, allocation, metadata, checks, hashes or logging
occurs inside. Expected-check/FNV work follows the end clock. Per-sample output
is outside timing and may affect the next sample's conditions.

INDEX_TIMING fields: split,stratum,arm,round(-3..-1 warmup;0..17 recorded),
warmup0/1,arm_position,IDs,rowpairs,dotcalls,GUweightbytes,order_fnv,output_fnv,
outer_ms,host_cpu. stream has122880 rowpairs/245760 calls per arm; hot has5120/
10240. Output FNV encodes four little-endian bytes per FP32 in ID/row/Gate-Up
order. INDEX_TIMING_COMPLETE requires216 recorded and36 warmup rows. Successful
flush and owned normal exit0 are mandatory; earlier admission PASS alone cannot
qualify this mode. Existing default/correctness-only modes remain separate.

Root requirements: review diff; same-flags fresh release and sanitizer exactness,
absolute MXCSR/clean-environment evidence, opt-out/argument/source/flush negatives,
complete sample/count/hash identity, immutable expected checks,1152 selected
role SHA256 verification and before/after stable-source checks. Preserve existing
8MiB text/1536MiB RSS and finite wall/CPU/AS/FSize supervision, shared serial lock,
no GPU imports/device observations and fresh-process replication. Two384-by1280
float buffers add3932160 bytes plus guards and bounded prebound row tables; these
caps are not RSS guarantees. No raw weights/Q8/per-row output is persisted.

This measures actual selected weights with synthetic per-layer inputs on one
caller, not workers, live model activation, whole-model32K speed, memory bandwidth,
full lifecycle or adoption. Source implementation has not been built or run.
