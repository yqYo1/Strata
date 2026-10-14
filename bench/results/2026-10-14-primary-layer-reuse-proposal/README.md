# Primary layer reuse proposal v1

2026-10-14; source base `5bf0742913584a343975ea7f792c97cefac488fe`, personal
`feature/sycl`. Niko1221/Strata is the sole upstream. XeStrata is an architecture
reference only; no Xe engine source or CUDA/HIP implementation is imported.

This is an implementation proposal with an applicable preparatory refactor and
host planner/ownership fixtures. It is **not an implemented or qualified
layer-major engine**. Product source, build configuration, running measurement,
Git state, model payload and GPU were not changed. No compilation, test,
`git apply --check`, performance run or profiler was executed. The source was read
and patches generated in this directory only. Main owns application and checks.

## Delivered code and bounded initial scope

`primary-layer-body-refactor.patch` extracts the existing chunk's entire layer
body into a local `run_layer(l, allow_next_layer_fuse)` callable, then calls it in
exactly the existing layer loop with `true`. Current chunk-major behavior remains
the proposed default. This is a small source refactor, not loop transposition:
chunk scratch, issuer, ordinal state and callback remain outside the callable.
The extra parameter gives a future layer-major caller a place to disable the
half-1 norm fusion across layers. Without that change, saving only `R` loses the
next layer's `m.grs`/`m.xn16`/low component computed by `gr_write_norm_rs`.
Within-layer half-0 fusion remains available. Numerical qualification must still
check unfused recomputation; no precision changes are proposed.

`host-planner-and-ownership.patch` adds a header-only host planner and standalone
host fixture under `include/strata/prefill` and `tests/cpu`. The same files are
available under `source/` for standalone compilation without applying anything.
The fixture uses explicit checks, not assertions removable by `NDEBUG`. It covers
64K/262144 bytes, bounded residency, integer overflow, failed admission clearing
its output, checkpoint and scheduler availability gates, exact suffix ranges,
ordinary cache exclusion, never-recorded drains, producer join, stale-generation
rejection and drain-error quarantine. This is a protocol model, not a GPU lifetime
proof. No CMake patch is required for Main's first host check.

A reachable capacity design under the current **stock settings** is a bounded
subset of expert blobs retained through every chunk of one layer. Existing
ordinary ExpertCache residents remain in place. The rest use the original ring,
formats, gather, GU/Down, reduction and expert order. This avoids requiring all
512 experts at once and avoids group-at-a-time accumulation changes. It does not
eliminate remaining expert transfers, and no throughput benefit is claimed.
Choose a deterministic nonresident subset by expert id initially; routing-aware
selection is a separate candidate. The planner computes a conservative maximum
count; the actual selection can be smaller because ordinary resident entries are
excluded. Layer blob stride/bytes and source format must be checked per layer;
`max_blob` must cover every layer visited.

Use fresh `pos0=0`, one GPU, no image/peer/layer split/borrowing/adaptive cache
relocation; existing `spec=4`, MTP callback, prompt_cache=0, KV int8 resident32768,
max_context262144, chunk4096, cache128 are the target. These settings do not
establish the new path is ready: `layer_runner_ready=false` at present prevents
planner admission. MTP is not excluded merely because `on_chunk` exists. Its
specific final-residual/token callback must run in increasing chunk order during
the final-layer pass, synchronously before each `m.R` buffer is overwritten.

## Capacity arithmetic and admission

Primary residual width is `HC*N=4*2560=10240` FP32 elements: 40960 bytes/token.
A single in-place pageable host plane is 2,684,354,560 bytes at 65536 tokens and
10,737,418,240 at 262144. An input chunk can be copied to device, transformed and
written over exactly the same host range after its input copy/read completes:
other token ranges do not need that residual input again. This does not make
stateful attention/GDN chunk-independent; their own state remains sequential.
Do not pin an entire 10GiB plane by default. Either validate pageable transfers
or preallocate bounded pinned bridge buffers; their host/device peaks belong in
`additional_host_peak`/`additional_device_peak`. Enforce finite host admission
using available RAM/cgroup budget after live model/session allocations. Two
pinned chunk bridge buffers alone require `2*4096*40960=335544320` host bytes.
The fixture's zero extra peaks are only a planner arithmetic example, not a
production allocation recipe.

The supplied live-free premise is about1299MiB **with** persistent prompt,
Verifier, MTP and graphs. The largest layer needs
`512*2662400=1363148800` bytes, already exceeding1299MiB before a700MiB reserve.
With zero new device metadata/bridge peak, the same budget admits at most235
expert slots,625664000 bytes (about596.68MiB); real additional peaks reduce this.
The figure is arithmetic from Main's live-memory premise, not a measurement by
this proposal. No idle10GB capacity claim is used. CPU residual admission can
fail at either64K or262144. All allocation/admission failures before mutation
return to the original chunk-major path at the same requested context, KV,
chunk and model settings. Do not cap context to the host-plane budget.

The planner has checked add/multiply/alignment, finite explicit reserves, address
space limits and an empty output on rejection. A valid plan is not allocation
success. Allocate and validate the host plane, bounded bridge buffers, device
slot, metadata and events; initialize source/callback contracts; obtain fresh
free-memory admission before the first embedding/session operation. RAII host
allocation exceptions are pre-mutation fallback. Do not switch to fallback
mid-pass after state mutation unless a separately proven restoration completed.

## Remaining production splice, in order

1. Share the current `run_layer` body through an explicit chunk context rather
   than the local closure: `T,p0,R`, scratch, qsa/gdn ordinals, PLE land operation,
   immutable weights/source and per-layer stream issuer. Keep the current loop
   as the unchanged call path. The supplied refactor only creates the first seam.
2. Add an owned temporary per-layer resident-pointer resolver, separate from
   ordinary cache metadata and Verifier graph pointers. All stream-exclusion and
   consumer sites must use it: current `prefill.cpp` lines1536,2029,2103,2356,
   2372,2529,2565-2566. Never publish a temporary slot as an ordinary cache index.
   At admission select nonresident ids; load them once into aligned slots, then
   retain those exact addresses through every chunk of that layer. The stream_all
   sequence and routed staging jobs exclude this subset. Preserve all existing
   MMQ/fused/scalar dispatch paths, including large-chunk fused GU/Down.
3. Rebuild stream_all sequence/issuer per active layer, not the old sequence for
   all layers of a chunk. Process that layer's chunks in increasing absolute
   position. Compute qsa/gdn ordinal from the layer index for each invocation;
   do not carry the incremented ordinal from the previous chunk. Recreate step
   records, staged QSA pages and routing metadata per chunk at its exact position.
   First pass builds embeddings per chunk; later passes upload its host residual.
   Force `normed=false` on every loaded chunk and disable half-1 cross-layer norm
   fusion. After every layer invocation save FP32 `R` to its exact host range.
4. Recreate PLE rows from immutable pre-request `prev0` and tokens in layer1;
   retain each host buffer until upload completion. Update PLE's convolution in
   increasing position and publish `ple_prev` only with a successful prompt.
   Finish callback/MTP/draft_kv in increasing position during the final layer.
   `draft_kv` applicability is a separate contract: it reads session KV scratch;
   verify full-prompt QSA state does not change which chunk range it extracts.
   If that optimization is not applicable, its existing `mtp.prefill` fallback
   may be viable; this must be proved at the call site and not silently assumed.
5. Add explicit request admission tying `checkpoints_disabled` to the actual
   serving caller. With prompt_cache0, `checkpoint_at` is a no-op. With cache on,
   current `checkpoint_at(done)` copies **all** layers while earlier layers are
   already at full-prompt end: saving those mixed states is wrong. Reject before
   mutation, or explicitly suppress per-chunk snapshots and implement coherent
   final/required checkpoint behavior. Do not merely disable a cache flag after
   a partial pass. Progress during nonfinal layers needs internal layer/chunk
   heartbeats; PP final-residual positions remain increasing on the final pass.
6. Invalidate live continuation before mutation; propagate partial STOP/error as
   failure and retain existing stored conversation checkpoints. Main correction
   to R482: full atomic rollback is not yet proved necessary. Current request
   start has `live_ok=false`; an error can invalidate the fresh session and the
   next GEN can `session_zero`. This must be verified across all exits, including
   MTP/PLE reset and internal RESTORE. Do not turn a partially processed layer
   state into a continuable session or a coherent chunk checkpoint. A caller
   without the invalidation/reset contract is ineligible.

These are missing implementation obligations, not delivered features. Until
implemented, the v1 patch supplies only the reusable layer seam and host safety
contracts and must not advertise an accelerated layer-major capability.

## Expert source and exact release proof

The current `ExpertSource::blob` base contract only guarantees a pointer until
its next call; transient native assembly can reuse storage. To fill a bounded
slot without assuming stable bulk spans, use `copy_blob(layer,e,owned_staging)`
while its CPU read completes; enqueue H2D; keep that staging buffer until its own
DMA completion before the next fill. Failed read cannot publish a resident entry.
Pinned sources may use direct copies only with an explicit source owner lasting
through DMA. Do not retain a vector of base `blob()` pointers across calls.
A future NativeExpertSource stable-span API must return pointer,size,alignment,
source generation and shared lifetime owner, distinguish contiguous arena vs
GGUF assembly/complement fallback, and prove owner survives every last DMA. This
optional API is not added to the broad base contract or required by this proposal.

Before replacing/freeing the layer slot: prevent new producers; join any
issuer/stager/PLE futures which can issue work; complete pending bridge uploads
and downloads; synchronize compute and copy queues with error checks; complete
stager host-DMA releases; establish no remaining final-layer callback/MTP reader;
only then mark this lease generation drained and release/overwrite. SYCL's
same in-order queue can preserve device ordering but does not prove host staging
DMA finished. Preserve each stager release event. Ignore no failed copy, event
record, wait or synchronize result in the new path. On drain failure, do not
free memory still potentially in use: quarantine owners, invalidate request and
terminate the owned engine safely if the runtime cannot prove quiescence. Do
not claim a successful queue proves the other queue drained. `Lease` deliberately
has no freeing destructor and cannot certify backend completion itself.

Phase-release alternatives are unnecessary for the bounded subset capacity
example. If Main wants full-layer residency, release/rebuild dependency is real:
serialize service admission, wait Verifier pending commit and all graph/cache/
MTP/Prefill producers, destroy graphs referencing released arenas, release only
owned Verifier/MTP scratch, preserve KV/session/checkpoints, recreate/rebind
Verifier and MTP and warm/capture required graphs before decode. The normal
chunk scratch remains needed for this layer runner; releasing it wholesale is
not a free extra1GiB. Graph/provider actual memory return and rebuild peak must
be measured. No full-layer fast path is offered until that lifecycle is complete.

## Main reproduction and qualification (not executed)

Standalone host fixture, after Main owns the exclusive build/test window:

```sh
c++ -std=c++20 -Wall -Wextra -Werror -pedantic source/layer_reuse_host_fixture.cpp -o /tmp/strata-layer-reuse-host
/tmp/strata-layer-reuse-host
```

Run from this directory. Main can apply the two patches from workspace root
with `git apply <path>` after reviewing/checking them. They do not add CMake test
registration or change the measurement client. Compile the actual configured
SYCL target after applying; the audited source resolver is
`cmake/sycl_backend.cmake`: `strata_prefill` compiles `src/prefill/prefill.cpp`,
`strata` compiles `src/program/generate.cpp`. Root CUDA generator behavior is
not evidence of a separate Arc implementation.

Before adoption Main owns: real model fixed-history residual/logit and complete
GDN/conv/QSA KV/index/PLE/MTP comparisons against chunk-major;64K and full262144
plus near-boundary requests; continuation and internal RESTORE; STOP first,
middle and final layer/chunk; blob load/copy/callback/drain failures; callbacks
order; prompt_cache gate; exact format pairs. Record original result statuses,
commands/environment, every individual sample and correctness/fault results.
Only then measure speed with unchanged Niko `bench.sh`, stock client and fixed
model/settings. Include completed H2D bytes, host residual traffic, slot hits,
full-path time, peak RAM/VRAM and decode regression separately. The new mode may
lose to chunk-major because a single host plane creates roughly two40KiB/token
transfers per layer; a bounded subset only removes some repeated weight copies.

## References and review point

Required prior reports read: R470 transfer projection, R471 phase-release
candidate, R472 serve lifetimes, R474 progress/stager contract, R482 layer-major
splice; registry `../2026-10-14-primary-model-baseline/research-index.json` and
Main's corrections are authoritative where they contradict the older reports.
R471 standalone CLI ordering must not be applied to serve. Verifier arena is
not357MiB; observed graph/runtime memory change is not measured arena release.
R474 compact bound must use per-expert tile padding (safe49, tighter47 at270
rows), not ceil270/8 alone. No private record or model payload was read.

Owner/consumer: Main's serialized implementation review and host fixture build.
Next review: after extracting the explicit chunk context, before enabling any
layer-major request admission. No raw traces or tensor captures are created;
retain these compact proposal files in Git at Main's next work boundary. Source
proposal is not adoption, history integration or a performance result.
