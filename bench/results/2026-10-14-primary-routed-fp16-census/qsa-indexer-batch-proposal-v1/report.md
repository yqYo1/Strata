# QSA indexer batch proposal v1

Prepared 2026-10-15 JST from current root SYCL source only. Proposal only: no
product file changes, patch application checks, builds, tests, GPU/profiler work,
process/service inspection, reset, cleanup or Git operations were performed.
Main's active owned benchmark was not read or touched. No speed, numerical
equivalence, fault-free execution or adoption result is claimed. Main owns all
serial validation after the benchmark; the PR #6 recovery review remains first.

The project-only Strata upstream-sync skill and `docs/ARTIFACT_RETENTION.md`
were read. Research registry's `previous_registry` chain was read back through
the performance-regression audit registry, together with R550, R552 and Main's
R549/R550 review/corrections. No PR #6, history or donor audit was read. R550's
score-reference table is corrected by Main; it is unrelated to this batch
submission proposal. R552's storage precision experiment remains separate:
this patch keeps the existing F16 raw-tail / FP32 transformed-key storage.

## Concrete change

`unified.patch` changes only `src/kernels/sycl/native_qsa_indexer.cpp`, the batch
API comment in `include/strata/kernels/native_qsa_indexer.hpp`, and a new pure
integer header `include/strata/kernels/qsa_indexer_batch_plan.hpp`. Exact
`STRATA_SYCL_QSA_INDEXER_BATCH=1` selects the candidate. Unset, empty, `0`, `01`,
`true`, `1 ` and other values retain the existing per-token loop. Single append
and its device-position/capture dispatch remain unchanged. The shared math
refactor also serves that baseline; it needs original-source regression parity.
The switch is read on each batch invocation, allowing same-process comparisons
with disjoint state buffers without rebuilding or caching an environment value.

The current prefill caller normally invokes the batch API unless
`STRATA_INDEXER_PER_TOKEN` is present. Its comment promises three launches, but
the current root SYCL implementation actually submits one append per token.
For a 4096-row chunk that is 4096 submissions for each QSA layer. The proposed
path has at most three submissions for any valid batch size. This is a static
source count, not a measured submission cost or latency improvement. The
prefill caller and comment are not edited in this artifact.

| Phase | Condition | Work and output |
| --- | --- | --- |
| Initial serial append | `p0 == 0` | Use the existing token-zero append to establish tail slot 0, pooled row 0 and dead. Leaves block_pos unchanged. |
| Bulk completed blocks | `(p0+n)/4 > p0/4` | One 256-lane work-group per completed four-cell block. Read preexisting tail or half-rounded batch raw. Write each completed pooled row exactly once. |
| Final metadata | Every positive batch | One lane per dimension. Apply the last serial write to each tail slot; if a block completed, write just the final spare from dead and the last completed block_pos. |

Thus fresh batches use two submissions if no block completes, three if one or
more completes; continuation batches use one or two. `n <= 0` preserves the
API's original immediate return. No allocation, host/device transfer, wait,
dependency event, new queue or synchronization is added. Inputs remain required
to stay valid until the already asynchronous queue work completes.

## Source arithmetic and ownership proof

`prepare_artifacts.py` pins the original native indexer SHA256 to
`d87abcf8aece04bc1566f4310e0dee546b7c80e0b711d75d118551b606d38685`.
It moves the original reduction/RMSNorm/RoPE body into `normalize_rotate`, used
by both serial append and bulk append. It does not retain a copied independent
math oracle. The only body edit is the non-writing lanes' `return;` becoming
`return 0.f;`, after the original second local barrier. The serial caller still
returns for those lanes. The helper retains:

- All 256 lanes, subgroup size 32, eight partials, XOR reductions with offsets
  16, 8, 4, 2, 1, both local barriers, `ss / 128.f + eps`, `rsqrt`, the multiply
  by mean followed by gamma, and rotation pair arithmetic.
- Incoming `float(sycl::half(raw[d]))` conversion, four terms summed in row
  order with the same left-associated `sum +=` sequence, and
  `sycl::fma(.25f, sum, 0.f)`. Bulk rows read the same half-rounded values as
  serial tail storage; completed keys remain FP32.
- Zero's repeated four-copy pooling and forced rotary position zero through
  the original append. Complete block b uses `base + 4*b` even when its start
  preceded this chunk. MRoPE uses the same `rp*3 + pair%3` entry; registered
  cos/sin table lookup, out-of-table analytic fallback, YaRN ramp/magnitude and
  frequency construction are unchanged.

For `e=p0+n`, the completed block range is `[p0/4, e/4)`. A block in this range
ends before e, so all its rows at or after p0 are valid raw rows `[0,n)`. Only
the first group of an unaligned start can have rows before p0. Those rows are
exactly tail slots `0 .. p0%4-1`; row 3 cannot be a preexisting-tail read.
All complete groups read tail and raw only and write disjoint pooled rows.
They never write intermediate spare rows or tail, so no complete group's
output can be overwritten by another group's spare write. The initial append
is ordered before bulk on the existing in-order queue; bulk is ordered before
metadata on that same queue. Metadata therefore cannot overwrite the tail
before its partial-block consumers finish. No inter-group ordering is needed.

Tail slot s in `{0,1,2}` receives the largest position `<e` congruent to s
modulo 4 only if it is `>=p0`; otherwise its old value remains unchanged.
This matters for n1..3, no-complete chunks and aligned ends: an aligned end
retains slots 0..2 of the final completed block, rather than clearing them.
Final spare is `pooled[e/4] = dead` only if a completion occurred; block_pos
becomes `base + 4*(e/4-1)`. With no completion, block_pos and all other pooled
rows are retained (apart from initial pooled0). Intermediate serial spare
writes are all overwritten by subsequent completed rows, so their omission
changes no final state and exposes no missing read to a caller: the batch API
permits state reads only after the whole chunk append.

At `p0+n == 262144`, the last valid cell is 262143, the last completed key row
is 65535, final spare row is 65536, and physical extra row 65537 is untouched.
The existing validation requires `(capacity/4+1)*128` floats; the caller
allocates `capacity/4+2` rows (`src/core/layer.cpp:505,612-614`). Both the span
validation and +2 physical allocation are preserved. The same integer logic
handles capacities not divisible by four and validated INT32_MAX endpoints.

## Consumers and lifetimes

The configured source resolver lists the root native indexer in
`cmake/sycl_backend.cmake:83` with `-fno-fast-math` at line 93. The patch stays
on that resolved backend. It imports no CUDA/HIP engine or kernel.

The actual chunk consumer is `src/prefill/prefill.cpp:2069`. Its raw pointer is
`m.idx_raw`, carved as `T*128` FP32 values at line 845 from the persistent chunk
region; projection writes it on `m.cs` at line 2020. Gamma belongs to the
loaded `indexer.k_norm.weight`. Tail/dead/pooled/block_pos belong to session
QSA state and are carved at `src/core/layer.cpp:610-614`. The batch and the
following `qsa_block_scores` calls at prefill lines 2089-2093 use `m.cs`.
`queue_for(stream)` checks this is an in-order queue. Even though the region
is shared with the later MoE half, all its later writers are queued after these
indexer reads on the same queue. The metadata submission is complete before
subsequent same-queue score queries read pooled/dead. It adds no raw consumer
after the current batch/selection phase.

`Prefill::relayout` drains m.cs and copy before recarving (lines 900-910),
`Prefill::release` drains them before freeing (lines 549-552), and successful
`Prefill::run` has its existing completion drain at line 2968. This proposal
does not strengthen, weaken or certify exceptional/sticky-error teardown.
Only the original API pointer-lifetime requirements are claimed; fail-fast
paths and registered-table lifetime continue to require Main's existing gates.

`Angles` (frequencies, scaling args, table descriptors and MRoPE pointer) and
`QsaIndexerBatchPlan` are captured by value in device lambdas. Local accessor
handles are captured by value and refer to per-group local storage. Submit
handler lambdas consume stack arguments synchronously to record each kernel,
as the existing append does; device lambdas do not capture host stack references.
Registered cos/sin and MRoPE memory are borrowed and retain the original
registration/context/replay lifetime contract. No new owned allocation or
temporary host input is introduced.

## Deferred commands and gates

The following are reproduction commands, not results. Run only after Main's
benchmark closes and Main permits serial CPU/build/GPU work. No command below
was executed by this implementation agent. The generator itself was executed
only to write proposal files and hashes; it does not apply or test the patch.

```sh
proposal_dir=bench/results/2026-10-14-primary-routed-fp16-census/qsa-indexer-batch-proposal-v1
sha256sum --check "$proposal_dir/SHA256SUMS"
patch --dry-run -p1 < "$proposal_dir/unified.patch"
c++ -std=c++17 -O2 -Wall -Wextra -pedantic \
  -I"$proposal_dir/proposed/include" "$proposal_dir/fixture/coverage.cpp" \
  -o "$proposal_dir/fixture/coverage"
"$proposal_dir/fixture/coverage"
```

Main should apply to a source-pinned isolated review tree only after the dry-run
and original-source hashes agree; do not apply over an evolved native indexer
without reviewing the new source. Use the established configured SYCL build
recipe and existing QSA parity target, recording actual flags/binary hash.
The existing parity source at `src/kernels/qsa_parity.cpp:1381-1464` checks an
aligned n8 fresh batch with none/linear/YaRN scaling. It is insufficient for
this proposal's entire continuation/tail matrix and must be extended by Main.

`fixture/coverage.cpp` includes the actual proposed planning header. Its oracle
uses independent literal serial state transitions with distinct symbolic rows,
not another planning formula. It compares every pooled row including final
spare and untouched +2 row, three tail slots, dead and block_pos; verifies
completed write disjointness, source bounds and retained tail slots; exhausts
small capacities/start/length combinations; includes n0..3, all remainder pairs,
4095/4096/4097 chunks, 32K/64K/full262144 endpoints, arbitrary preexisting states,
seeded prefixes, single-cell head continuations and a second batch. INT32_MAX
planning is covered without a huge allocation. This fixture is model-free and
symbolic: it cannot establish SYCL numerical or memory-lifetime correctness.
No assertion is compiled away by NDEBUG.

Required runtime correctness gates before any timing/adoption decision:

1. Compare the original untouched implementation against the proposed build
   with flag unset, plus exact-value negative controls; confirm shared-helper
   refactor preserves every state word and single-append/captured replay.
2. Compare flag-off serial batch, flag-on batch and device-position single
   appends bit-for-bit for **all physical pooled rows**, tail, dead and block_pos,
   with guard canaries. Use n1..3, all 16 start/end remainder pairs, no-complete
   batches, p0=0, arbitrary initialized continuation, 4096-sized chunks, odd
   lengths, partial final chunks and p0+n=capacity at 32K/64K/262144. Include
   non-divisible capacities and finite F16 rounding boundaries, signed zero,
   small/large finite magnitudes and cancellation-heavy rows.
3. Repeat under none/linear/YaRN, zero and nonzero aligned sequence base,
   analytic RoPE, matching registered tables and fallback past table extent,
   MRoPE with distinct t/h/w and supported combinations. Preserve the original
   registered-context rejection and input/span validation; n<=0 remains no-op.
4. After each paired chunk compare one-token decode/next-head state and output,
   a second batch, then actual selector IDs/scores and selected-attention/full
   model first logits under identical inputs. The current canonical prefill
   leaves a real final token to nativeT1; include that boundary. Compare capture
   replay/snapshot/restore continuations without changing those mechanisms.
5. Record real submission counts, normal completion/teardown and fresh fault
   receipts using Main's existing supervision. Obtain clean unprofiled timing
   separately from detailed tracing. Only after correctness passes measure
   both small and 4096-row indexer batches and whole prefill. Keep all samples
   and environment/source/binary identities. A launch-count reduction alone
   cannot qualify speed or adoption; PR #6 recovery review may supersede this
   candidate before runtime work.

## Artifact retention

All files are compact reviewable source, commands and deferred gates; no live
trace, model/session/raw tensor or duplicate diagnostic capture was collected.
Owner is Main; consumer is post-benchmark source review and the serial validation
matrix, with PR #6 recovery first. Budget is 256 KiB. Review at that boundary
and on any source change; preserve original hashes/status and retire the
proposal only when its consumer is closed or superseded. No cleanup/deletion
occurred. `provenance.json` records base and candidate hashes; `SHA256SUMS`
covers the entire prepared artifact except itself.
