# One-pass layerwise periodic checkpoint capture v1

**PROPOSAL ONLY. Product application, build, CPU fixture execution, GPU/runtime,
RESTORE correctness, numerical quality and performance: NOT_RUN.** Main's Direct0
and Direct1 full same-92acf71e-binary benchmark arms must BOTH close before Main
applies, builds or tests this candidate. No process, service, Git, build, test,
benchmark, GPU, profiler or cleanup action was performed by this author. Only
reads and artifact generation/hashing inside this new directory were performed.

The project-only `strata-upstream-sync` skill was read and applied to source
provenance, component scope and measurement reporting; this task's proposal-only
restriction overrides its usual integration/build/Git workflow. No CUDA/HIP
engine import or validation is proposed.

The candidate fixes mixed-position intermediate periodic snapshots under exact
LM=1 by capturing each layer while that layer is at the requested absolute
prefix. It preserves a single layer-major sweep for each existing caller segment.
It does not split at 16K. With no existing root/turn/reread segmentation, logical
expert payload remains **50,292,326,400 B per full pass**. Existing root/turn
segments can still cause their existing extra passes. Nothing here measures
physical transfer bytes, elapsed time, or an achievable target token rate.

## Inputs and source finding

Read: AGENTS.md; docs/ARTIFACT_RETENTION.md; project SKILL.md;
research-index.json; R608 and Main's correction; R609; coherence-v1 FINAL report
(`implementation.ja.md`) and source.diff; actual configured root SYCL sources.
`sha256-ledger.json` pins old/new product text and the proposal diff. Main supplied
92acf71e as the current source/binary boundary; Git was not queried. Generator
rejects unexpected source hashes and writes ONLY this directory.

`run_impl` ends each chunk with a checked `cudaStreamSynchronize(m.cs)` before
stage/chunk callbacks (current prefill.cpp:3070..3078). `run_layer_major` calls it
with one owning layer and resets the cursor for each layer. Its stage wrapper
filters early layers/intermediate positions; its final-layer regular callback
forwards every chunk. A whole-session snapshot there can therefore combine early
layers at segment end and the final layer at an intermediate prefix. Existing
checkpoint size/pointer validation establishes allocation compatibility, not
state position.

PLE normalized history is actually advanced in `run_impl` at **layer 1**, both
`native_ple_postops_batch` and tokenwise `ple_history_advance`. `ss.ple_prev` is
reset/recomputed by the layer-major wrapper and is not a valid intermediate
prefix tag. `ConversationCheckpoint` does not save prev IDs; existing restore
correctly derives them from its own last two IDs. Keep that representation.

The preceding coherence-v1 caller segmentation remains a possible safety fallback
with repeated expert work. It is not this implementation. R609's payload volume
analysis is useful. Its divisions by observed transfer rates are **scenarios,
not hardware lower bounds**: the 2.25 GB/s bulk figure is slower than the observed
5.79 GB/s expert copies. No claim that a performance target is impossible follows.

## Concrete proposed source

`source.diff` changes three existing files and adds one header:

- `Prefill::on_layer_checkpoint` is a new distinct internal callback, separate
  from `on_chunk` and `on_stage_chunk`. It runs inline only for active LM after
  the checked native compute-queue synchronization, before existing callbacks
  and before another chunk can advance that layer. It receives layer ordinal,
  absolute done position, and an inline checked-copy capability.
- The copy capability submits D2H to **m.cs**, checks submission, and always checks
  retirement even when submission failed. Retirement failure aborts while host
  owners are alive. It never queues a raw host pointer for later capture. Its
  source pointers and capability cannot escape the inline callback. LM already
  requires copy==compute queue and an idle stager; the issuer is joined before
  this boundary. There is no process-environment mutation.
- `LayerwiseCheckpoints` previews the existing retention decisions using metadata
  only, constructs running-state owners only for final surviving new positions,
  validates the full input/session range, and tracks a 48-bit fixed layer mask.
  Geometry is restricted to the already-supported 48-layer/12-QSA/36-GDN LM
  model. Every retained state is a run-identity + absolute-prefix object, not a
  view into the mutable session. All arithmetic byte products/additions are
  checked against INT64_MAX. Prefix/chunk steps are bounded by `end`; next
  threshold saturates; run/stamp exhaustion is rejected before publication.
- At each relevant owning-layer boundary, it copies exactly one GDN slice, or
  that QSA layer's tail/dead/block-position fields. It copies PLE at layer 1
  only. It also checks the complete layer/chunk schedule, even when a position
  has been evicted in the preview and needs no state bytes.
- The ordinary final-layer callback still executes target/MTP prompt work,
  progress, accounting and ordinary PP output. Once that callback succeeds it
  marks pending MTP completion and advances the same periodic threshold. It
  never publishes an active layerwise transaction.
- Caller `Prefill::run` stays whole. The existing MTP-expert wrapper must return
  (reattachment succeeds or fail-stops), then the borrowed-slot refill must
  succeed. Only then can a preallocated vector be populated by noexcept moves
  and swapped into `checks`. All checks precede those moves; no failure is
  permitted midway through publication. Candidate payload allocations happen
  before the run, not while old checks are being moved.

The narrow feature gate is SYCL, exact existing LM="1", single GPU, no split/
peer, cache>0, interval>0, and a genuinely long batched call (`n>sp.chunk()`).
Cache0, interval0 and current cold benchmark have no installed capture hook or
pending payload. Non-LM and multi-GPU keep their existing callback path. Short
segments under the gate keep the existing coherent full-model `run_impl`
checkpoint behavior. The existing arithmetic expressions outside the gate are
unchanged. CUDA/HIP runtime behavior is outside qualification scope.

`generate_proposal.py` generates full proposed files, source.diff and old/new
hashes, with exact source-hash/anchor checks. It is a generator, not a validator.
The new header template and its proposed copy must have the same hash. Do not
apply if Main source moved; review/rebase against the actual resolver first.

## Schedule, cache semantics and transaction assumptions

First threshold remains resume+interval. At the first real chunk end reaching
that threshold, the original event occurs once; next threshold is that actual
end+interval. This is not an absolute 16K grid and does not create multiple
snapshots when one chunk crosses several thresholds. The planner reads actual
`sp.chunk()` **after lend/relayout**, and the current LM implementation keeps that
chunk constant inside a run. The window path still creates no periodic saves;
if windows cross a threshold, the next batched chunk end is overdue and saves.
A reread INT64_MAX sentinel remains inactive at all supported physical positions.

Before one existing outer segment, preview every periodic event at those exact
chunk ends, followed by the existing root/turn save event if applicable. The
preview executes the actual `conv_cache::eviction_victim` policy: first item
pinned when cap>=2; cap1 takes the newest; all other slots rotate by used stamp;
a duplicate position only changes used stamp and consumes a clock tick. Thus a
periodic event and root/turn at the same end consume TWO stamps and share ONE
state owner. Old states that survive are moved, not copied. New states that the
final policy evicts are never allocated. Stamp values still include all events,
including discarded periodic entries and root/turn duplicates.

This preview is safe only because the serve issuer is serialized, checkpoints
are not externally visible during a synchronous prompt run, callback code cannot
change `checks`, and root/turn destinations and actual chunk are fixed. `commit`
checks the original size/stamps and full schedule. Main must re-audit those
assumptions if callback/thread ownership or chunking changes. Preselection
changes internal temporary residency; post-success published content/stamps are
identical to the existing policy applied to coherent events.

Keep all session positional target/draft KV arrays. This snapshots running
state only, not independent conversations. Preserve existing prefix/image/cvec
filtering and LIVE publication; no sampled-but-unconsumed output token is added.
RESTORE still restores QSA tail/dead/blockpos, rebuilds the pooled dead row at
`ids.size()/idx_block`, derives PLE prev from the matching IDs, and calls existing
MTP `kv_restore(resume)`. Neither QSA nor MTP full KV arrays are truncated or
replaced by this candidate. Parking and conversation-cache serialization retain
the same `ConversationCheckpoint` format.

Cancellation/error drops pending states and never publishes partial candidates.
Already-published earlier outer-segment checkpoints remain. Explicit native
capture/MTP errors cannot be swallowed by a simultaneous stop flag. A cancel
observed after a successful long run/refill drops that transaction; cancellation
never publishes partial LIVE. A queue retirement failure terminates with owners
live. Existing root/turn events in a captured long segment move to the same
transaction's post-refill commit; short/non-LM events retain their old path.

## RAM bound and byte accounting

Let S(P) = GDN + optional PLE + 12*(tail+dead+blockpos) + 4*P + image-key bytes.
Let O be actual capacities of all old running-state/ID/image payload vectors;
let N be the sum of S(P) for **final surviving NEW** positions. Let M cover
bounded metadata/owner/publication vectors. The source conservatively reserves
**B = 2*(N+M)** for new allocations/capacity and separately checks actual vector
capacities after allocation. It first admits bounded metadata preview before it
allocates preview vectors. Unknown memory telemetry or insufficient RAM rejects
before submitting any new prefill work.

Required inequality is **O + B + 8 GiB <= O + available_after_old**, equivalently
B <= available_after_old - 8 GiB. Old checks must remain live until commit; they
are already reflected in available memory and cannot be subtracted as reclaimable
space. Checked `O+B` records the worst old+new payload overlap; publication uses
moves, so there is no third full-state copy. Existing LM host admission executes
later against free memory after pending owners exist, charging residual/upload
storage cumulatively while keeping its own 8 GiB reserve. This is host-RAM
admission, not freed VRAM or full conversation-KV sizing. Allocator overhead and
unrelated concurrent memory users require Main observation; no admission scheme
reserves OS RAM against unrelated processes.

For this geometry, one GDN payload is **117,669,888 B =112.21875 MiB**. Six payloads
are **706,019,328 B =673.3125 MiB**, before IDs/PLE/QSA/metadata/capacities. At 256K,
16 possible 16K events do **not** imply 16 payload owners with cache6: only six
survive the preview; old six + new up to six coexist until commit. For no prior
root, those positions are 16384,196608,212992,229376,245760,262144. An existing
pinned root can replace that first periodic survivor. Physical prompt processing
usually stops at n-1, so use the actual n-1 schedule, not this round-length example.

## Main validation plan — every item NOT_RUN

1. Finish BOTH Direct0/1 same92acf71e binary benchmark arms, including their
   authoritative completion/fault/cleanup receipts. No apply/build/test overlaps
   either live arm. Review source resolver, current hashes, source.diff, queue
   semantics, metadata-preview assumptions and immutable native exec environment.
2. Run the new standalone CPU `state_position_fixture.py` only after that gate.
   It uses token-time boundary scheduling, specification-level root/LRU events,
   closed-form toy prefix states and a provenance/lifecycle auditor. It covers
   32K/64K/256K, 4096/8192 chunks, interval1/16384/16385, nonmultiple tail,
   shifted/resumed positions, windows, de-dup stamps, pinned root, cap1/2/6,
   mixed-position negative control, missing layer/PLE/MTP, wrong run,
   cancel/copy/MTP/run/restore/refill failure, no partial publish and int64 limits.
   This is independent CPU contract evidence, NOT execution of the C++ candidate.
3. Add/run an actual-header host fixture in Main's owned build after integration:
   fake SessionState host buffers plus checked-copy recorder, feed actual layer/
   chunk callbacks into `LayerwiseCheckpoints`, compare final states/stamps to
   this independent oracle, and inject failures at every submission/retirement/
   MTP/refill point. Verify callback sequence, no deferred pointers, old-owner
   survival, peak allocation, overflow/invalid geometry and token IDs. The fixture
   must not mutate environment once any runtime/provider thread has initialized.
   Run normal + sanitizer CPU coverage and the actual configured SYCL build;
   existing checkpoint/cache/setup checks appropriate to touched paths too.
4. Small native cases: cache0/interval0/short<=chunk controls; long cache6 interval
   16384 with identical actual IDs, FP32 saved residual R=4096, MTP4, fixed expert
   placement, no adaptation. Record capture run ID, actual prefix, per-layer
   position/completion, PLE owner, MTP completion, wrapper return and refill
   boundary. Assert no pending owner is selected for RESTORE and no callback
   exposes mixed positions. Verify default/nonLM callback meanings separately.
5. Actual RESTORE-versus-reread at every retained point for matched 32K and64K
   nonmultiple-tail prompts. Compare all 36 GDN rows, PLE history and prev IDs,
   all12 QSA tail/dead/blockpos, reconstructed pooled dead row, positional target
   and draft KV fingerprints, first logits/residuals, fixed output IDs and
   MTP drafts offered/accepted. Use real ID/image prefix fingerprints. A saved
   byte size, successful API response or plausible output does not qualify it.
6. Use separate immutable-environment processes for controlled full reread and
   `STRATA_CKPT_REREAD`. Reproduce actual chunk/root/turn schedule where needed;
   otherwise numerical schedule differences are a separate question. Compare
   fresh read length, RESUME/REUSED and selected prefix, not only generated body.
7. Root/turn same position as periodic, just before/after, resume shifted from
   global grid, interval16385, windows63/64/65, shared root/second chat, rewritten
   suffix miss, LIVE continuation, image identity, cvec, thinking on/off and
   tokenizer-confirmed turn248045. Test short-read0/64 in separate processes.
   Preserve serialized issuer, one-token-ahead consumed/live contract and MTP
   `kv_restore` semantics; parked-conversation behavior remains a separate gate.
8. Cancel before layer0, layer1 PLE capture, middle layer, final-layer earlier
   chunk, immediately after final MTP callback, after wrapper restoration and
   before/after refill/commit; then a new request. Confirm prior published
   checks survive, incomplete pending states are absent, partial LIVE is absent,
   and new requests resume/reread correctly. Inject logical errors first in the
   host fixture; do not intentionally poison a live GPU context. Copy/MTP/refill
   errors remain errors even when cancellation arrives. Test stale run IDs and
   duplicate callbacks without relying on actual concurrent issuance.
9. Allocate the **actual physical --max-context262144** engine and host-backed/
   streamed QSA fallback, then drive real long input close to the physical end
   while leaving legal input/decode space. Record actual n, n-1 batched length,
   physical allocations, page/slot modes, host fallback, per-layer capture and
   retained-point RESTORE through the boundary. Include final-input-token and
   legal subsequent decode. A tiny pool indexed at262143 or a CLI/context number
   is not full physical256K validation. Unrun means unqualified.
10. Only after correctness/cancel/fault gates, measure clean same-setting fresh
    processes at32K/64K and full256K plus real agent turns. Keep all individual
    PP/TTFT, state-copy, MTP detach/restore, refill, decode and restored samples.
    Check one full logical expert sweep per existing outer segment from source
    counters/actual copies, and report state-copy bytes/retained points. Compare
    against current cache0, safe interval0 root/turn and optional segmentation
    fallback separately. No mixed-binary speed adoption or scenario-rate bound.

All native SYCL/Level Zero/driver/experiment variables must be set in the native
child's environment before exec, then remain immutable. Do not reuse legacy
fixtures that use setenv/unsetenv between arms after runtime initialization.
Main owns all execution and integration decisions after reviewing this FINAL.

## Retention

Consumer: Main source review, later integration and host/native validation.
New directory budget: 2 MiB; no raw tensor/session/log/profiler capture exists.
Review point: Main's source review and eventual integration or rejection.
Keep compact diff/generator/fixture/report/hash ledger in Git when Main closes
its work boundary. No artifact deletion was performed; deletion manifest is [].
