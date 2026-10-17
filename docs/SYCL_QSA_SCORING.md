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
build disables floating-point contraction for the scorer. Exact score and ID
parity is checked against the original kernel in the actual-entry fixture
`tests/sycl/qsa_score_prefill.cpp`; this does not impose FP32 parity on unrelated
future 16-bit algorithms.

The current fixture covers query counts 1–9, selection boundaries 2051/2052,
and physical KV extent 262144 with 256 and 255 queries. Those component checks
are separate from full-model context, RESTORE, quality and throughput evidence.
See `bench/results/2026-10-15-pr6-recovery-audit/qsa-score-prefill-entry-results-v1.ja.md`
and the individually pinned results for their exact scope.
