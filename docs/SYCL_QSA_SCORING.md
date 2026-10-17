# SYCL dedicated-prefill QSA scoring

`qsa_block_scores_prefill` is the layer-major long-prefill entry declared in
`include/strata/kernels/qsa_select.hpp`. It accepts the same geometry and output
layout as `qsa_block_scores`, with a separate register-sharing implementation.
The original public scorer remains the decode, short-prefill and reference path.

## Selection and scope

Set `STRATA_SYCL_SCORE_QUERY_TILE` before starting the process and keep the
process environment unchanged after runtime initialization. An omitted value or
`0` selects the original scorer; `8` shares a pooled key across eight adjacent
queries. Other values raise an error. The setting is cached on the first call.
With tile `8`, `STRATA_SYCL_SCORE_HEAD_PARALLEL=1` enables a separate four-head
parallel mapping. An omitted value or `0` keeps sequential heads; values other
than `0`/`1`, or enabling parallel heads with tile `0`, raise an error. This
setting is also fixed before exec and cached. Both optimizations remain opt-in
until actual engine performance, including the first request, is qualified.
The engine's `STRATA_SELECT_OLD` override bypasses this entry and forces the
original scorer. Direct callers of this entry do not read that override.

The optimized geometry is four indexer heads, 128 dimensions and four cells per
block. It uses one subgroup of 32 work-items per work-group. The current device
qualification is Intel Arc B570 with the compiler, runtime and driver pinned in
the measurement receipts; this is not a qualification of other devices.

## Input extent and initialization

For `max_blocks`, query count `nq`, and an active launch bound `active_blocks`:

| Argument | FP32/int32 layout |
| --- | --- |
| `pooled` | FP32 `[max_blocks, 128]` |
| `dead` | FP32 `[128]` |
| `q_idx` | FP32 `[nq, 4, 128]` |
| `steps` | int32 `[nq, kStepCount]` |
| `scores` | FP32 `[nq, max_blocks]` |

`active_blocks=-1` launches all `max_blocks`; otherwise it must be positive and
no greater than `max_blocks`. Each valid query's `n_bid` must be smaller than the
launch bound. Its incomplete block reads `dead`, and a nonempty partial tail
gets the existing bonus. Future score columns are not written.

**Tile 8 may read every launched pooled-key row before inspecting each query's
step.** Allocate the complete declared pooled span, initialize every row in the
launch bound, and keep those rows immutable until completion. This includes a
row that is incomplete or unused for an individual query. Do not pass a compact
live-prefix allocation or publish a concurrent future-row write merely because
that query will not emit a score there. For reusable capture with an omitted
active bound, initialize and retain all `max_blocks` rows.

The engine's main indexer state allocates `max_cells / idx_block + 2` pooled rows
and zeroes the whole allocation. Dedicated-prefill producers and scoring use the
same ordered compute queue in the measured configuration. The ring drafter's
smaller pool has no indexer and is not an input to this scoring path.

## Queue and ownership

Use device allocations belonging to the queue's runtime/context. The writable
score span must not overlap an input. Keep all input/output allocations, the
queue and its runtime alive until a checked completion wait finishes. Shape,
alignment and overlap checks do not establish allocation size or ownership for
an arbitrary raw pointer; supplying the complete spans is the caller's duty.

An explicit stream is an in-order `sycl::queue*`; the function enqueues work and
can return before it completes. Order another producer queue with dependencies,
then wait before host access, reuse or release. A null stream uses the default
runtime and waits before returning. A timeout alone does not cancel device work
or authorize releasing its buffers.

## Arithmetic and validation

The query-sharing path retains FP32 operands, four-term lane-local sums, the
original XOR reduction tree, sequential head sums and tail treatment. The SYCL
build disables floating-point contraction for the scorer.
The head-parallel mapping groups eight lanes per head, combines the original
lane contributions in the same root tree, and sums the four head roots in the
original order. Its fewer subgroup exchanges do not establish lower physical
memory traffic, register pressure or occupancy.
Exact score and ID parity is checked against the original kernel in the actual-entry fixture
`tests/sycl/qsa_score_prefill.cpp`; this does not impose FP32 parity on unrelated
future 16-bit algorithms.

The current fixture covers query counts 1–9, selection boundaries 2051/2052,
and physical KV extent 262144 with 256 and 255 queries. Those component checks
are separate from full-model context, RESTORE, quality and throughput evidence.
See `bench/results/2026-10-15-pr6-recovery-audit/qsa-score-prefill-entry-results-v1.ja.md`
and the individually pinned results for their exact scope.


## Experimental FP16 score GEMM in long prefill

`STRATA_SYCL_PREFILL_SCORE_GEMM=1` selects the FP16-input, FP32-output GEMM
scorer only inside the long layer-major lease. Default0 retains the existing
scorer. Chunk8192, at most256 selection queries and the owning session's block
capacity are required; `STRATA_SELECT_OLD` must be absent. The dedicated short
prefill and decode routes do not allocate or use this workspace.
The explicit opt-in rejects unsupported Level Zero/FP16/SG32 devices, geometry
or arena topology before this request starts model work. It does not silently
switch arithmetic after KV/indexer state has been updated; leave the flag at0
to select the existing path. The direct API repeats its device check before
submitting its own work.

This changes completed-block operand rounding. The dead-key FP32 tree and
partial-tail bonus remain unchanged. The private component experiment passed
its documented rounded-input numerical, guard and physical-context component
checks; those measurements are in
[the result report](https://github.com/yqYo1/Strata/blob/8063aacc6749d18089f7a53db5d3e47b745f348c/bench/results/2026-10-15-pr6-recovery-audit/qsa-score-gemm-results-v2.ja.md).
They do not establish this integrated path's model quality or throughput.

The scorer borrows the existing `attn_h` reservation during selection. At
chunk8192 the reservation is100663296bytes; the maximum workspace including
alignment/guard padding is50611200bytes. All preceding readers and following
writers use the same in-order queue. Attention writes `attn`/`qsa_attn16`, then
gate overwrites `attn_h` before output projection consumes it. `Qf` and
`qsa_Qfull16` remain live gate inputs and are never reused by the scorer.
The carve rejects insufficient/misaligned spans or a status word inside the
arena; the engine validates registered device spans before model work.

Only a separate256-byte status allocation is additional, included in the VRAM
admission budget before residual/MoE placement. It is zeroed once per long
request. Every scorer invocation atomically accumulates invalid-input and
nonfinite-arithmetic bits without clearing earlier failures. After each QSA
pass, the engine copies four bytes into a host owner declared before the
checked lease, drains and checks the status. A nonzero status rejects the
prefill result before decode. Status remains outside the arena subsequently
reused by gate, MoE and GDN. Borrow restoration and all temporary-owner release
follow checked retirement; unproven retirement aborts with owners still live.
