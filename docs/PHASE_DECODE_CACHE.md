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


## Native head prefill lease and fixed 12288 capacity

`STRATA_PHASE_HEAD_LEASE=1` defaults to OFF. It defers Prefill initialization until
first GEN, then releases native head weights and scratch after quiescence and
reader graph retirement at each GEN. After the complete Prefill run and reset,
it restores the head with the same upload and conversion from immutable original
raw GGUF bytes copied into CPU RAM before the first upload. Conversion uses the existing
Runtime checked drain to check stored asynchronous errors and include its watchdog. The snapshot holds
only the byte count validated against supported type and shape. Reload does not
reopen the file or accept changed contents with unchanged metadata. Additional
CPU payload storage requires exact `STRATA_PHASE_HEAD_LEASE=1`; no head D2H copy
is made.
Verifier retains the same NativeHead object and captures new graphs after reload.
An upload or conversion failure with unknown submission / completion retains
weights, scratch and the original CPU source owner, pointers and byte counts,
and rejects retry / run / release / reload. An uncertain owner destructor
fails-stop before freeing a DMA source. Only allocation failures before any
submission can safely clean up and retry. A failed release API also poisons the owner
without assuming whether storage was released; further release and retry are refused.
Each successful release clears its pointer before the next release, preventing a later
failure from causing a double free. Failure, cancellation,
nonfinite values and unknown completion fail-stop; reload is not recovery.
Head retirement and restoration time enter prompt_ms.

`STRATA_PREFILL_SCRATCH_ALIAS=1` also defaults to OFF. On an in-order compute queue,
it shares emb / bo / ple_emb and qkv / FP32 y after their last readers complete.
Byte counts, owned / borrowed carve and relayout use the same startup setting.
FP32 y and its normalizer retain their calculations.

Fixed 12288 capacity requires explicit `STRATA_PHASE_PREFILL_12K=1`,
`--prefill 12288`, and both lease and alias above. All other phase restrictions
remain. Without the opt-in, capacity stays fixed at 8192. Auto, other capacities,
BF16X2 and separately owned KV staging are rejected. Initialization and every
fresh GEN reinitialization use the same capacity; insufficient memory never
reduces it to 8192. Full context remains 262144.

After head release, fresh device free is compared against recounted workspace,
token buffer, mode1 identity table, conditional GDN chooser and lazy GEMM bounds,
and existing `--vram-reserve-mib`. KV staging is reported inside workspace without
double counting. Reserve is the existing operator setting (default 700 MiB,
positive required), leaving room for uncertain JIT, backend, provider and
fragmentation costs. No GPU model identifier selects admission. Logs distinguish
requested-byte estimates, the actual shared-region maximum count, configured
capacity and actual run chunk T. This guard does not replace physical peak
measurement. Runtime capacity qualification requires first-use chooser / GEMM
peaks, repeated GEN, the 4095 tail and final token, logits / persistent state
comparisons, full 262144 context and continuation / restore checks.

## Diagnostic capture of the first serve logit row

Both `STRATA_SERVE_FIRST_ROW_CAPTURE=1` and `STRATA_SERVE_FIRST_ROW_DIR`, naming an
existing private output directory, enable this diagnostic. Absent, `0`, `01`, and
`true` are OFF. When OFF, this diagnostic performs no row allocation, copy, or
file IO. Do not use it for speed measurements.

This route requires the existing phase-cache and head lease contracts. It accepts
only ordinary serial `GEN 1` on one GPU, with `--spec 1 --suffix-draft 0
--lookup-chain 0`, and prompt/conversation caches and saves, adaptive swaps,
batch, MTP, pipeline, and `--dump-logits` disabled. Besides startup settings, it
checks actual fresh state, zero reused tokens, and the first Verifier window's
T=1 and position P-1. It rejects incompatible requests and exits without releasing
owners on uncertain copy or phase failure.

After ordinary prefill, head reload, and phase publication, it reads the full FP32
vocabulary row through `Verifier::copy_logits(0)` after the first successful
window without NaN/Inf. It publishes `request-N.bin` only after the request, last
commit, and phase completion succeed. The layout is int32 vocabulary, int32
rows=1, then vocabulary FP32 values. COPIED on stderr means incomplete; COMPLETE
records request index, actual prompt length P, predicted position P-1, source row
0, actual reuse 0, vocabulary, byte size, and the completed filename.

Each process reserves at most 64 requests and 64 MiB in total. Creation, write,
flush, close, byte size, and finite values are checked. It creates `.partial`
exclusively and publishes only completed files through an exclusive hard link,
so the output filesystem must support hard links. It never overwrites an existing
name. Failed partials are not completed captures. Successful partial links also
remain: Main reviews hashes and required evidence after the run closes, then
follows [artifact retention](ARTIFACT_RETENTION.md) for cleanup.

For fixed history comparisons, each request's prefix is `prompt + follow[:i]`.
The generated argmax token does not need to feed the next history. An external
controller binds each prefix, target, completed row, and source/binary identity.
A single captured row does not establish full sequence teacher forcing, NLL,
full quality, 256K, RESTORE, or a speed pass.
