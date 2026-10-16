# PR6 dedicated prefill API harness v1

**SOURCE_ONLY_UNSAFE_TO_RUN. Implementation delivered; configure, preprocess,
compile, link, tests, providers, GPU, model numerics and performance are UNRUN.**
Worker task `task_565940f6a1ac`, dispatch `ctx_66579ab253c5`, 2026-10-16 JST.
Main retains all execution, independent review, Git, integration and retention
responsibility. No 600 tok/s result or physical262144 qualification is claimed.
No existing failure, receipt, report or approval was changed.

The immutable baseline is `7b1bd9b67d58f2d653a2184b93e5ddb83f7dcc32`, on the
existing `verify/pr6-prefill-restore` branch. The source checkpoint is
`3da01f7148f5536f444c8f471c65e2d47c58bc41`; changes since that checkpoint are
review/report records, not new historical engine implementations. The working
tree was initially clean. This delivery is an uncommitted source snapshot, not
an integrated or approved engine. The JSON pins base/final file identities, the
patch, required input reports and preserved historical inventory. The patch
contains exactly two source paths; these three new delivery artifacts are
excluded to avoid self-reference.

## Exact source delta and origin

- New `validation/pr6-restore/harness/prefill_api.cpp`: a real Linux executable
  entrypoint using retained public engine APIs; no production dispatcher change.
- Append-only, 20-line opt-in target in `validation/pr6-restore/CMakeLists.txt`.
  `PR6_BUILD_PREFILL_API_HARNESS=OFF` by default; when ON, `pr6_prefill_api` is
  still `EXCLUDE_FROM_ALL` and requires an explicit target build. Existing
  `PR6_MAIN_REVIEWED_BUILD`, offline vendor hashes, checked-DPCT transformation,
  native library and source-closure gates remain. The target links
  `strata_prefill` and copies its actual compile/link options: the historical
  child-directory SYCL/AOT/precise flags otherwise do not reach a sibling TU.

Both additions are independently authored project integration under repository
LICENSE. No external engine implementation, helper, test, template, archived
proposal or excluded source was used. SHA-256 is a small newly written scalar
implementation of the digest algorithm, used only on bounded host evidence;
Main must independently qualify its vectors before trusting receipts. No
inference arithmetic, kernel, original198 inventory file, historical
`sycl/CMakeLists.txt`, vendor49 source, source resolver, FP32 residual layout or
layer/chunk loop was edited. No branch switch, Git mutation, extra agent,
upstream check, dependency installation, runtime query, process inspection or
cleanup occurred. Python was used solely as an editor/hash/artifact generator;
read-only Git commands inspected the immutable base and generated its diff.

## Entry, model and call trace

`main` first parses explicit paths/counts, snapshots relevant inherited
environment, and reads **exactly** `count*4` token bytes as little-endian int32.
It hashes those actual bytes and stores immutable int64 IDs for the public
`Prefill::run` API. Missing/trailing/short input, negative IDs and malformed
integers fail. No tokenizer, random/repeated synthetic content, file search,
model generation, network or arbitrary discovery exists. `--shard` is repeatable
and requires explicit distinct absolute files, metadata shard first. The PLE
file is separately explicit; the pack root contains `index.txt` and the native
expert layout. If experts.bin is absent, role files named by the pack must be
in the supplied shard list before `FileExpertSource` may map them.

The first shard must declare 512 experts/10 selected experts, matching
Prefill's actual fixed workspace (48 layers, embedding2560, hc4, hc rank320,
FF640 and the remaining built-in QSA/GDN geometry). Scaled RoPE is held; the
retained unscaled configuration consumes the model's validated frequency base.
The existing layout loader and native expert support gate reject unsupported
formats. `NativeDense::served_names` and `keep_unquantized_ple_key` construct the
same native-serving skip set as the retained native preset, with embedding and
head served by their own loaders. No canonical embedding/head fallback is
silently selected.

After selecting an explicit device, the harness obtains the registered borrowed
DPCT default allocation queue and creates one registered owning compute queue
with `create_queue(true)`. Both must be in-order, on the same GPU/context.
`WeightTable::pool_bytes/load`, `check_all`, `NativeEmbed::load`,
`NativeDense::load(..., include_ple_key=true)` and `NativeHead::load` use the
original loaders and their error checks. Embedding/head shapes and real
vocabulary, every input ID and selected output ID are admitted before run.
The output mix's otherwise caller-dependent shapes/bytes/forms are also checked.

The retained CLI's native preset sets `gr_native_mmvf=true` and PLE native BF16
and postops=true (`generate.cpp:1891-1930`, setters near2385). The harness sets
those relevant public controls before session creation. This matters:
`Prefill::run_impl` only chooses batched PLE when
`ple_native_postops_enabled()` is true (prefill.cpp:2237-2246); default false
would change its arithmetic/schedule. No decode objects are needed to set these
process-local controls. `gr_fp32_activations=false` plus native MMVF matches the
preset and still uses the native path's local FP32 activation contract. PLE key,
value, norms and F16 convolution wiring follows the retained CLI's actual
representations, with stricter bounded shape/byte checks and no PLE-off option.

`session_bytes/init` allocate/carve one complete session. `ss.db` starts null
and must remain null. `session_cells`, `position_span`, the real `Prefill::init`
and `Prefill::run` checks remain in the dependency path. KV storage is explicit
(fp16/int8/q4_0/k8v4), with **all** pages resident and INT8 rotation disabled.
No streamed-KV, split session, cache lending, stage helper or peer is configured.
A real `PleTable` uses mmap with no IO worker/keepalive; this avoids incompatible
simultaneous direct-I/O and native/expert mapping of a shared shard. It is an
explicit I/O scheduling choice, not a claim of historical timing identity.
`FileExpertSource` supplies real pack blobs or checked model role slices;
Prefill's original transient/copy_blob staging leases remain intact.

The only model-body invocation is
`Prefill::init(wt,g,ss,experts,nullptr,nullptr,8192,compute,err)` then
`Prefill::run(ids.data(),count,0,err)`. Native constructor dependencies do **not**
require a split. `Prefill()` only allocates its Impl (prefill.cpp:652).
`STRATA_PREFILL_LAYER_MAJOR=2` selects `run_layer_major` even for one chunk;
compact=1, MMQ=0 and XMX=0 retain the historical compact FP16 operator path with
FP32 residual storage. Its original resident-KV, full-session, stage, provider,
workspace, overflow, state and readiness checks still execute. Original layer
and chunk traversal, short first chunk and residual placement logic are unedited.

`on_chunk` sees only final-layer chunks in layer-major execution. It verifies
contiguous bounded positions/counts and final checkpoint readiness. At the last
row it copies the final **device FP32 residual** into the session's existing
`ss.R`; run itself does not export its final row there. This 40960-byte D2D copy
uses the documented synchronized callback, is retired before returning, and is
included explicitly in run timing. After the entire run is retired, one row is
read back, checked finite and hashed. The unchanged public
`lm_head_mix` plus `NativeHead::run` computes final-token logits on the same
ordered queue. Full logits are temporarily read to a bounded host buffer for
finite/argmax/SHA evidence; they are never saved or printed in full by default.

`Prefill`'s object file has retained `draft_kv` references, so static linking can
include MTP/verifier/session implementations. **Presence in the link is not
construction or execution.** The harness declares no Verifier, MtpDrafter,
Doorbell, SessionGraphs, TokenGraph, SessionLoopScratch, CPU decode pool or
coupled draft object; it calls none of their init/capture/run routines.
`concurrent_doorbell_admitted()` remains constexpr false without an override,
and the harness statically asserts that exclusion. Setting native math flags
is not a decoder initialization. Compiled closure, static initialization and
actual no-launch proof still need Main's binary/provider review.

## Lifetimes, errors and compact receipt

Source IDs, options, environment and host result vectors outlive Runtime.
Runtime publishes each allocation/queue/loader owner directly into its fields;
partially initialized objects remain owned for unwind. Native load/init use a
600-second fail-stop deadline. Existing Prefill init600/request7200 and bounded
retirement/join60 behavior remain. Head+readback has a60-second fail-stop bound.
Runtime close starts its60-second watchdog before provider operations, retires
compute and borrowed allocation queues, destroys Prefill while source/session
owners live, retires again, unbinds native embedding, destroys head/dense/embed,
releases rope registration, deletes host QSA descriptors and frees its arenas.
The mapped sources/PLE stay alive through prefill worker joins. It retires the
allocation queue again and destroys **only** its owned compute queue last.
The borrowed default queue is never destroyed. Normal success calls close
before emitting the JSON completion receipt; unwind follows the same order.

Unknown completion/submission/retirement/deadline exits86 with owners live;
selected DPCT async delivery exits87. A known-quiescent cleanup exception or
legacy noexcept-destructor termination exits88. The entrypoint installs a
terminate handler for that last case; it cannot classify an arbitrary terminate
as a new successful cleanup. Ordinary parse/load/readiness/nonfinite/output
failure returns1, with no completion receipt. Existing engine functions that
call exit(1) remain unmodified failures; an absent receipt or abnormal exit is
never promoted to success. External finite process/log supervision is still
required, including pre-runtime file/metadata work and driver faults.

A completed receipt is `PREFILL_ONLY_COMPLETED_UNCOMPARED`, not numerical PASS.
It records actual token SHA/count/path/encoding, explicit model/shard/PLE paths,
GPU/driver, relevant inherited environment, context/chunk/first/ring, requested
and effective rounded GPU residual token counts, vocabulary and arena sizes,
expert source/read counter, final residual SHA/min/max/L2, full logits SHA,
selected ID/logit, argmax and final PLE predecessor window. Phase timings keep
load/session init, prefill init/run, callback D2D, engine stats, final readback,
head+logit readback and cleanup distinct. No throughput score is printed.
Optional timing controls add their inherited instrumentation overhead; Main
must not subtract callback/validation or engine statistics from whole-path time
without a separate matched measurement. No CPU-versus-wait attribution is
inferred from these clocks.

Only `--capture-final-row NEW_ABSOLUTE_PATH` saves a tensor: exclusive create,
40960 bytes, little-endian FP32 shape[4,2560], same hash as the receipt. No full
KV/recurrent tensor, session snapshot, generated text or large trace is saved.
Capture write failure remains failure and a partial artifact requires Main's
retention review. SHA correctness is UNRUN and must be checked independently.

## Accepted and held cases

| Input/condition | Source behavior; execution is UNRUN |
| --- | --- |
| Count32768..262144, context>=count<=262144 | Admitted host envelope, then actual session/QSA/Prefill capacity checks |
| Chunk8192, first0..8192 | Explicit; matching immutable environment required; first8192 is effectively ordinary8192 |
| Odd long tail, e.g.32769/40959 | Admitted, original tail kernels/order retained; no numerical claim |
| Exact physical count/context262144 | Host-admitted only; physical allocation/page/state correctness UNQUALIFIED |
| Smaller actual allocated QSA capacity | Rejected before Prefill; Prefill repeats retained checks |
| Count<32768, zero, negative, one-past262144, undersized context | Rejected before runtime; no tiny-input path was invented |
| Continuation/restore/rewind/generation | No CLI/API route; fresh position0 only |
| Wrong count/partial/trailing token bytes, negative/out-of-vocabulary IDs | Rejected, actual bytes hashed and immutable |
| Missing/duplicate/relative shards, unsupported512x10 geometry, unsupported native formats or scaled RoPE | Rejected, no geometry/math substitution |
| PLE absent/wrong storage/key, missing output mix, head mismatch | Rejected; PLE cannot silently disappear |
| Unknown inherited STRATA variables, dump/NaN/tuning switches | Rejected before runtime; no unbounded dump side effects |
| Provider queue/context/readiness/callback/statistics inconsistency | Failure or retained fail-stop; no retry over partial state |
| OOM, partial init, async/provider failure | Existing loader controls plus owned unwind; requires independent fault gates |
| Legacy doorbell/verifier/native CLI fullmodel | Still unavailable; no env/build bypass |

The harness's long-input scope deliberately rejects tiny inputs. Main still
needs a separately reviewed tiny/odd-tail/continuation gate before long model
execution; this executable does not turn its32768 minimum into permission to
skip that gate. No MTP path, RESTORE, state export, quality/task or full cold/
warm/restored decode is qualified by final-row/logit evidence alone.

## Main-only recipes and next gates (not executed)

These are parameterized real CLI recipes: Main must supply its reviewed
absolute build/dependency/input paths; no model location was discovered here.
Do not run configure until its compiler probes and offline dependencies have
been reviewed. No recipe is an approval.

```sh
: "${PR6_BUILD_DIR:?new reviewed build directory}"
: "${PR6_VENDOR_ROOT:?absolute pinned historical vendor snapshot}"
: "${PR6_LLAMA_ROOT:?offline llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d}"
: "${PR6_CXX:?absolute coherent icpx path}"
: "${PR6_CC:?absolute coherent C compiler path}"
cmake -S validation/pr6-restore -B "$PR6_BUILD_DIR" -G Ninja \
  -DPR6_MAIN_REVIEWED_BUILD=ON -DPR6_BUILD_PREFILL_API_HARNESS=ON \
  -DPR6_VENDOR_ROOT="$PR6_VENDOR_ROOT" -DSTRATA_GGML_DIR="$PR6_LLAMA_ROOT" \
  -DCMAKE_CXX_COMPILER="$PR6_CXX" -DCMAKE_C_COMPILER="$PR6_CC" \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_SYCL_PARITY=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$PR6_BUILD_DIR" --target pr6_prefill_api --verbose
```

First review the entire patch against the named base and actual public APIs,
especially PLE/native controls, checked queue registration, final-row export and
cleanup. Pin compiler/runtime/UR/LevelZero/oneMKL/ggml/vendor identities and the
private checked-DPCT hash. Inspect actual configured source/forced include,
compile/link commands, AOT/FP32 flags, inherited target options, ODR/ABI and
absence of alternate current-engine sources. Building only the new target
still builds its historical dependency libraries; it is not model-free runtime
qualification. No `strata` CLI execution is necessary.

Next independently qualify the scalar SHA implementation against known vectors
and input/row hashes using trusted host tooling, parser/exact-byte bounds and
ownership source paths. Main owns any new fixtures. Then complete finite
model-free selected-provider checks for checked default/owned queues, actual
pending/complete/event/barrier/error delivery, accepted partial submission,
source leases, queue/context and retirement, exit86/87/88 and partial cleanup.
No-init/late-init/cancel/relayout and ring-wrap/tail provider gates from v3 remain.
Review actual host arithmetic in loader/stager/PLE and oneMKL dispatch/provider
behavior separately; a GPU queue is not proof that every operator executes on
GPU. Preserve FP32 reference and state/quality comparator obligations.

Only after those source/provider and separate tiny-model gates does a supervised
long-input example have this executable syntax (all variables are Main-selected
real paths/values; repeat `--shard` for every supplied shard):

```sh
: "${PR6_MODEL_ROOT:?existing native pack root}"
: "${PR6_SHARD1:?metadata shard absolute path}"
: "${PR6_SHARD2:?second model shard absolute path}"
: "${PR6_PLE_FILE:?actual PLE table file absolute path}"
: "${PR6_TOKEN_FILE:?existing real input, exactly32768 little-endian int32 IDs}"
# Main sets the provider/SYCL/UR/driver environment in this child's env before
# exec, from a qualified recipe. Remove unrelated inherited STRATA variables.
env STRATA_PREFILL_LAYER_MAJOR=2 STRATA_PREFILL_COMPACT=1 \
  STRATA_PREFILL_MMQ=0 STRATA_PROMPT_ATTN_XMX=0 \
  STRATA_PREFILL_FIRST=256 STRATA_PREFILL_LAYER_MAJOR_R_GPU=0 STRATA_PREFILL_RING=8 \
  "$PR6_BUILD_DIR/pr6_prefill_api" \
  --model-root "$PR6_MODEL_ROOT" --shard "$PR6_SHARD1" --shard "$PR6_SHARD2" \
  --ple-file "$PR6_PLE_FILE" --tokens-file "$PR6_TOKEN_FILE" --token-count 32768 \
  --context 32768 --chunk 8192 --first-chunk 256 --residual-gpu-tokens 0 \
  --ring-slots 8 --kv int8 --device 0 --logit-id 0
```

Ring8/host residual/int8/mmap above is a concrete conservative arm, **not** a
reproduction of historical645.85 scheduling. Match the archived configuration
before any historical comparison. For an odd tail use an actual32769-token file
and count/context32769. For physical256K, use an actual262144-token file and
count/context262144 only after the physical-context gates are separately
admitted. For all-GPU residual placement set both immutable env and CLI value
to the actual count; there is no FP16/BF16 residual substitution. Optional final
row capture needs an explicit external current-comparison consumer/path.
Diagnostic `STRATA_TRACE=1`, `STRATA_PREFILL_TIMING=1` and
`STRATA_PREFILL_TRANSFER_TIMING=1` are the only optional engine variables;
separate diagnostic and clean timing processes. Queue/driver environment is
never changed inside the process.

Main records source/patch/binary/compiler/library/model/pack/token SHA,
individual commands/effective environment and separate model numerical/state
comparisons, boot/owned PID+start, process deadlines, normal/abnormal exit,
cleanup and fresh journal fault findings. Default raw text budget64MiB; no GPU
work, cleanup or builds concurrently with measurements. Output0 plus complete
JSON is necessary, not sufficient for correctness; missing receipt, timeout,
budget stop, mismatched hashes or fault is incomplete/failure. Compare final
residual/logits and full KV/scales/GDN/conv/indexer/PLE state before matched
speed samples. Full physical262144 and export/exit/fresh RESTORE plus quality
and excluded full decode obligations remain open.

No run logs, binaries, tensors or deleted artifacts were created in this task.
Main owns retention of the compact source/report/patch, any later40960-byte
capture's named consumer and the next review after independent source/build/
provider gates. This source delivery does not relax v3's native execution hold.
