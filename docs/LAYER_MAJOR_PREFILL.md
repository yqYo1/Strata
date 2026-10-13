<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Experimental restoration of layer-major prefill

`STRATA_PREFILL_LAYER_MAJOR=1` selects reading all chunks for each layer. Its value
is frozen at startup; absent, `0`, `01`, and `true` retain chunk-major processing.
The old `STRATA_PREFILL_COMPACT=2` is not restored. Current kernels, MMQ/FP16
fallback, batching, and scratch/gated aliases use their existing settings.

The route targets fresh standalone and ordinary `--serve` GEN on one GPU. It
accepts all-resident KV and the current context262144/resident32768 streamed-KV
path. Streamed KV retains the existing host-prefix seed, append, identity stage,
and attention for each chunk, without a persistent per-layer stage cache. All QSA
states must have matching KV mode/page/context geometry. Physical capacity,
numerical equality, and speed require separate qualification.

Disable prompt/conversation caches and saves. Each fresh request uses ordinary
checked session zero; actual resume/live/checkpoint/parked/slot state is also
checked. Completed non-reusable request metadata is dropped. Warm-up and later
GEN requests can use the same settings. A one-token prompt skips prefill and uses
the normal token path. Intermediate checkpoints, public stage callbacks, MTP,
split/helper, pipeline, vision, borrowed workspace, adaptive swaps, SAVE/RESTORE,
batch, and VRAM commands are rejected. Phase-cache/head-lease integration is a
later stage. Use ordinary phase0 serve.

It requires `--no-prefill-borrow --prefill N --prompt-cache 0
--conversation-cache-mib 0` and an explicit positive `--vram-reserve-mib M`.
Serve additionally requires explicit `--short-read 0`, preventing switching part
of a request into Verifier windows. Automatic chunks and reserve reduction are
not used. Raw `--spec 4` is accepted; the ordinary first decode window has actual
T=1.

`STRATA_PREFILL_LAYER_MAJOR_R_GPU` is a maximum GPU-prefix token budget. Explicit
unsigned decimal values from `0` through `262144` are accepted; signs, whitespace,
overflow, and trailing text are rejected. It is clamped to actual request n,
then rounded down to chunk boundaries. Ordinary conditioning omits the final
prompt token, so n=P-1. When all n fits, the last short chunk also stays on GPU.
This adjusts a budget to request length; insufficient memory never silently
reduces reserve or prefix. For example n=17/chunk8/budget32768 gives GPU17 rows;
n=33/budget17 gives GPU16/RAM17 rows. Trace distinguishes requested budget from
actual GPU/RAM rows.

Input, pos0=0, context<=262144, shape, byte products, and store boundaries are
checked before allocation. All FP32 residual rows persist in a GPU prefix and RAM
tail; chunks never cross stores. Fresh free VRAM/RAM is sampled after existing
model, workspace, head, session, and ordinary expert-cache allocations. The new
residual store, one-layer expert cache, source staging, residency metadata, and
explicit reserve must fit or the request is rejected. Reserve covers transient
lazy kernel/provider allocations; it is not a physical high-water guarantee.
There is no device-name dispatch.

Layer-major count/carve omits the normal device expert staging ring. Before each
chunk, it checks every active-layer expert's residency in the private cache,
slot count, offsets, and actual blob-byte capacity. Host Stager, pinned sources,
events, and MMQ gather ring/group buffers remain. Missing experts cannot execute
without a cache-miss ring. Streamed-KV stage and current aliases remain allocated.

All expert blobs of each layer upload once into the new cache, reused by all its
chunks. A source blob pointer remains valid until its checked synchronous copy
completes, before the next blob call reuses it. Transient sources use current
`copy_blob`. Asynchronous upload optimization is a separate change.

Current chunk code advances global QSA/GDN ordinals, positions, states, and PLE
history. PLE's two previous tokens reset to the same original request-entry values
for each layer. Decode's cache is preserved. Internal callbacks write intermediate
residual bridges; public on_chunk receives residual rows/output/progress only in
the final layer after checked compute completion. No snapshot at the same position
across all layers is produced; callbacks must not save checkpoints. Tokens/chunks
accumulate once in the final layer; total wall is one outer interval. One-layer
boundaries may disable existing inter-layer norm fusion, so numerical equality
must be checked with complete head/state comparisons.

Success, error, and cancellation check compute/copy drains before restoring
callbacks, cache, stack residency, and residual pointers. Unknown copy/drain
completion causes fail-stop before unwind/free. Failed sessions are not rolled
back; nonfinite or unhealthy requests are not reused.

`STRATA_TRACE` reports layer, role, actual uploaded bytes, slots, computed chunks,
requested/effective prefix and GPU/RAM rows, and checked completion. For speed,
disable trace and transfer/timing diagnostics and alternate matching source/binary,
shape, KV, prefix budget, and MMQ policy. Separately qualify complete vocabulary/
output, session states, subsequent decode, repeated fresh GEN, normal teardown,
and faults. Historical PR #6 numbers are not measurements of this route. Follow
[artifact retention](ARTIFACT_RETENTION.md) to preserve commands, environment,
individual samples, and outcomes.

The standalone `--full-context-probe TOKEN` arm can exhaust exactly 262144
physical target cells with layer-major enabled. It requires exactly 262144 input
IDs, `--max-context 262144 --max-new 0`, int8 KV, resident32768, explicit
`--vram-reserve-mib 700`, phase0/head0, no MTP, and the same fixed owned prefill
and fresh-state restrictions above. The terminal ID must be valid but is unused
in this target-only arm. Checked final-layer callbacks account for every row,
including the final chunk/page; no successor row or generation follows exhaustion.
The existing non-layer-major probe still uses MTP and in-process RESTORE.
Layer-major does not enable checkpoint, SAVE/RESTORE or reuse. This source change
has no full-context, numeric, memory-fit, speed or fault qualification. Near-boundary
GEN and eventual RESTORE need separate qualification; a 32768 resident ring does
not reduce physical context to 32768. Ordinary SERVE routing is unchanged.

This diagnostic can retain the existing scratch/gated alias settings. Prefill's
matching count/carve, gated-alias dependency and in-order queue checks remain.
The completion callback reads R, which is separate from aliased scratch.
Zero output still retains the existing head, logits and decode expert-cache
allocations. The later fresh VRAM/RAM admission includes those retained owners;
the residual bridge, one-layer cache and 700MiB reserve must fit or it refuses.
This does not establish full-span memory fit.
