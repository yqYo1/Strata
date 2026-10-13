<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Experimental phase decode cache

`STRATA_PHASE_DECODE_CACHE=1` selects an experiment limited to serial GEN in ordinary `--serve`.
The value is frozen at startup; absent, `0` and `01` use the ordinary path.
An omitted ckpt bit from the stock HTTP client is accepted only when no actual checkpoint,
live session, parked conversation, slot or reusable state exists.
Prompt and conversation cache capacities remain zero.
GENI, SAVE, RESTORE, batch, multi-GPU, MTP, pipeline, vision and remote helpers are excluded.
The original 262144 context admission and generation reserve remain in use.

After successful prefill, each GEN retires Prefill and Gemm scratch, fills and reads back
additional expert mappings at stable virtual addresses, and compares full residency before decode.
The next GEN retires prior verifier graphs, invalidates extra residency, unmaps and drains,
then reinitializes Prefill using the same session buffers. The ordinary fresh session zero follows.
The phase change does not replace owned KV, GDN, PLE or QSA buffers.
A short warm-up uses the same protocol. Cancellation, nonfinite values and uncertain completion
fail-stop the experiment; the HTTP server suppresses automatic restart and reload.
Start the server explicitly for the next experiment.

SYCL supplies the VMM aspect, device recommended/minimum and context minimum granularities,
maximum allocation and actual free memory. Start addresses, reservations and map extents
are checked for alignment, divisibility and overflow; uncertain ownership cannot continue.
The experiment adds a quota of 16 experts/layer to the 3 experts/layer baseline.
This is not a measured optimal value. Insufficient capacity never silently reduces the quota.
Unsupported devices use the default OFF path.

A failed attach or detach leaves the range's mapped/unmapped state unknown.
SYCL unmap requires a confirmed mapped range. The owner becomes poisoned and
rejects further requests and reader publication. Map rollback stops immediately;
repeated close calls perform no unmap, physical destruction or virtual release.
Physical handles, the virtual range and backend/context ownership, including
earlier successful mappings, remain retained until process teardown. Destroying
this owner follows the existing std::terminate policy. There is no automatic
retry, implicit recovery or direct Level Zero reclamation. Known-state failures,
such as failure before physical creation, keep their ordinary cleanup behavior.
No provider guarantee of map/unmap failure atomicity is assumed.

Linux CPU subprocess fixtures cover unknown map results, failed unmaps and a
failed rollback unmap. After removing injected faults, they check that repeated
close retains ownership without unmap/free/destroy or reader publication, then
use _exit without destructors. Synthetic mapped and unmapped post-error states
do not establish actual provider behavior.

The CPU fixture is `phase_cache_owner_test`; the tiny actual-VMM fixture is
`phase_cache_vmm_test --small-vmm`. The latter is not registered with CTest and is run by
the operator serializing GPU work. Pin source, compiler, runtime/provider and model inputs,
then verify two fresh GENs and normal QUIT before performance comparisons.
Slot bytes and residency must match. Identical generated text is not a mandatory quality gate.

With phase ON, `prompt_ms` is elapsed time from just before retiring the previous
GEN, after successful GEN admission, through completion of this prompt's map,
fill, readback, memcmp, residency publication and host refresh. It includes the
previous GEN's quiescence, graph retirement, invalidation, unmapping and Prefill
reinitialization, and this GEN's quiescence and Prefill reset. The first GEN has
no previous-GEN retirement. Short warm-up requests use the same accounting.
Startup base fill, work before GEN admission, decode and QUIT are excluded.
OFF retains its previous clock position and meaning. PrefillStats `ms_total`
records elapsed `Prefill::run` time, including transfers and synchronization.
This separate aggregate does not acquire phase costs.

Phase-only logs record owner counter differences for `begin_gen` and
`successful_prompt`, and actual cycle operation/transition host monotonic elapsed
spans. They distinguish quiescence, graph retirement, Prefill reset,
reinitialization, invalidation, native map/unmap, physical creation/destruction,
drain, publication and checked close. Expert payload H2D/D2H and memcmp are
separate from residency-table H2D/D2H and comparison. `calls`, `success`, `failed`,
`attempt_bytes` and `success_bytes` count actual calls and results, never expected
slot counts as copy counts. Failed operation elapsed spans and attempted bytes
are retained. `host_elapsed_ms` spans marked `nested=1` include nested work: their
sum is not wall time. They are neither GPU event timing nor transfer-only device
time. Compare the upstream client's whole request wall separately.
