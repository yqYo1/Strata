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
batch, and VRAM commands are rejected. Phase0 serve remains available. The optional phase-cache/head-lease route requires
serve, explicit fixed 4096 or default fixed 8192 chunks, and exact phase/head-lease opt-ins; layer-major 12288 is
rejected. Existing phase geometry, immutable ranked profile, and fresh GEN guards
still apply. No physical fit or numerical qualification is claimed.

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

The phase layer-major owner sequence is: drain and retire Verifier/session/token
graph readers, release the head, invalidate the complete checked host/device
residency table, detach and destroy all decode-cache physical extents, then create
the prefill workspace. An actual unpoisoned zero-mapped prefill lease is required
at Prefill init and run. The private ordinary all-expert cache still loads once
per layer. Its checked drain and destruction, and workspace retirement, precede
remapping/refilling/checking/publishing the baseline decode cache; head reload and
extra decode mappings then complete before the first Verifier window. The same
sequence applies to later GENs. Any failed transition poisons the cycle and
fail-stops through the existing phase fatal path. The virtual range and residency
table remain owned. Fresh VRAM/RAM admission in run charges current owners plus
private cache, FP32 bridge, and the unchanged explicit operator reserve; no
automatic shrink or static fit claim is made.

The qualified serve-first-row source feature is included as an exact opt-in
diagnostic. Its existing phase/head/effective-T1/max-new1 guards and bounded full
vocabulary publication remain in place; this combined source awaits Main testing.

Explicit `--prefill 4096` also admits the combined phase/head/layer-major route
with scratch alias enabled. It selects exactly 4096 on first and later fresh
GENs; it never rounds, resizes automatically, or falls back after failed admission.
The 8192 default and independent opt-in 12288 contract remain; combined
layer-major/12288 remains rejected. The 4096 route uses the same bounded geometry,
requested workspace count/carve, lazy GEMM/chooser estimates, fresh free check,
and explicit positive reserve. Actual callback T must fit configured capacity
and the persistent session context, including a short final chunk. Residual-prefix
admission still checks actual fresh VRAM/RAM after workspace initialization;
a smaller workspace alone does not prove a larger prefix fits.

Phase prompt statistics additionally print
`experts_host_source_checked_copy_wall_ms`, the existing `ms_experts_host`.
For layer-major it spans host source access/staging and checked expert copies;
it is a host wall interval, not isolated PCIe service time. No new timing events
or waits are added. This source option has no new performance or numerical
qualification: CPU checks, two fresh GENs, complete head/state comparisons,
actual memory admission and the long stock benchmark remain required.

`STRATA_PREFILL_ROUTE_UNION=1` enables a default-OFF diagnostic for fresh layer-major
requests. After the existing checked route-readback wait, it updates one 512-bit
set per layer. Eligibility is 1–64 layers, 512 experts and K=10; other geometries
or flows decline the diagnostic without changing inference eligibility. It adds
no kernel, readback, synchronization or clock. Sets reset per request; token routes
are not retained. The fixed request-owned structure is below 8 KiB, with a 64-byte
temporary chunk set. The structure is not allocated when the flag is OFF.
After the existing request retirement drain, it emits at most 64 layer rows and
one summary. `active_visits` sums distinct active experts in each chunk;
`union_experts` counts distinct experts across the request. Per-layer row/chunk
coverage is checked against expectations, along with order, IDs, extents and
counter overflow. Cancellation, errors and incomplete coverage cannot publish
complete status. `union_logical_bytes` is cardinality times
`layout.blob_bytes(layer)`, not measured PCIe traffic. Upload extents, cache
placement, route choices and math stay unchanged. CPU/GPU validation of this
change has not been run.
