Native SYCL QSA indexer device-batch candidate v1, 2026-10-15

This is an unapplied, unbuilt, untested source proposal against the actual
unchanged root. Only this proposal directory was written. No compiler, build,
test (including host), GPU, process/environment/private-state inspection, Git
mutation or cleanup was executed. Main continues to own every serial build,
test, GPU measurement and artifact cleanup boundary.

`source.diff` is the complete integration payload. `source-manifest.json` pins
each root base and proposed file SHA. New production planning/fixture sources
are also provided for direct review. Existing translation units are represented
only by the diff. Other proposals, including BF16 v2, head release and KV prefix
reuse, remain untouched and must not be stacked for this comparison.

`STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH=1` selects the candidate through the
existing `native_qsa_indexer_append_batch` API. The immutable local-static
parser accepts only absent/0 (default reference) or exact 1. It reads no flag
inside a kernel and never mutates the environment. Existing single-cell
`native_qsa_indexer_append`, its private `append` body, decode and all default
reference per-row submissions remain intact. Candidate n=1 also calls that
original kernel. A SYCL-only explicit candidate API permits comparisons under
one immutable process environment; the fixture never flips the option.

The device batch has three ordered phases on the same validated in-order queue:

1. If the batch begins at cell 0, invoke the original single-cell append once.
   It initializes the half-rounded first tail row, dead seed and pooled[0]
   with exactly the existing repeated-cell 0 RMS/RoPE behavior. No independent
   workgroup races to initialize or consume dead.
2. One workgroup per completed four-cell block computes that block's pooled
   key. A group reads any pre-batch first-partial-block components from the old
   tail; components in this batch come directly from raw after the SAME
   `float(sycl::half(...))` roundtrip. The group performs the same left-associated
   four-term sum and explicit fma(.25,sum,0), then the unchanged 256-thread /
   subgroup 32 RMS tree, partials[8], rsqrt/multiply order, local values barrier
   and analytic/table/M-RoPE expressions. The reduction/rotation source is
   copied verbatim from the existing oracle. Each group writes only its own
   completed pooled row. Tail and dead are read-only in this phase; no group
   writes a neighbor's spare or publishes block_pos.
3. A separate 128-item finalizer runs after every completed group. Each lane
   writes its component of the latest newly appended tail slots 0/1/2, leaving
   untouched slots whose last write predates this batch. If any block completed,
   it writes dead only to the terminal spare row and lane 0 publishes that last
   block's absolute first position. Intermediate spare writes from the original
   loop would be overwritten by the next completed key; omitting them creates
   no inter-group race. The finalizer is the inter-kernel global ordering
   boundary, using the existing in-order queue rather than a workgroup barrier.

There are no new global buffers, allocations, host readbacks, waits, events or
environment mutations in production. Workgroup-local values/partials match the
oracle. Existing API ownership remains: all buffers/tables survive downstream
queue completion, and asynchronous failures remain subject to the existing
checked runtime/caller retirement. Selection and attention remain after the
complete append call on that queue. No callback or MTP schedule changes.

The checked host plan is used by the actual candidate. It rejects negative or
overflowing ranges, invalid capacities/sequence bases and unaddressable spans
or global ranges. Nonempty work also reuses the existing fixed 128/4/64,
epsilon/scaling, complete disjoint-span, table/context and queue validation
BEFORE even a seed launch. Unordered/null queues and overlapping pointers are
prelaunch failures. N=0 is a no-op after range/base checking; the legacy default
API's previous n<=0 behavior remains intact when the option is off. No input
finite-value readback is introduced: the existing caller contract requires
finite raw/gamma values whose raw values round to finite half.

`p0` and `pos_base` remain separate. A nonzero partial first block reads its
actual saved tail; completed blocks rotate at base+4*block, while the original
cell 0 seed always rotates at 0. Registered RoPE tables, analytic fallback outside
their range, M-RoPE lookup and table-context checks remain the original ones.
No arithmetic is silently changed to BF16, reordered higher-precision reduction
or clamping. Compiler options for the actual native source remain the existing
`-fno-fast-math` and `-ffp-contract=off`.

At capacity 262144 the last completed row is 65535 and the required moving spare
is 65536. The planner includes 65537 pooled rows. No modulo addressing or smaller
scratch substitute is used. These bounds and a later synthetic physical-span
fixture do not qualify a real 262144-context model run.

Source and measurement scope:

- The configured `cmake/sycl_backend.cmake` directly includes
  `src/kernels/sycl/native_qsa_indexer.cpp` in `strata_kernels`; the new explicit
  API has a definition in that actual source, not a fixture-only declaration.
  The manual GPU fixture links that production library. CUDA/HIP sources and
  implementations are untouched.
- R680's one-token/one-kernel mapping is the source basis. Main subsequently
  supplied the completed v2 summary/hash/count checks and independent unitrace
  table check: actual 34603-window/9 chunks, 415284 append calls, about 1.917 s append
  device service among 1,017,611 GPU operations. R680's older pending-v2 caveat
  is historical; no profile was run or reanalyzed here.
- Fewer submissions target dispatch count. Those counts and device service do
  not predict throughput or a reduction in the approximately 98 s visible gap.
  QSA selection/attention, transfers, GDN, MoE and CPU waits remain separate.
  The candidate is default-off and has no speed, quality or adoption result.

New fixture sources and their limits:

| Fixture | Actual coverage when Main runs it |
| --- | --- |
| `qsa_indexer_device_batch_plan_test.cpp` | Uses the production planner/parser. Enumerates short partial ranges and counts completed blocks; checks flag parsing, 8192-row launch geometry, last pooled/spare extent, integer/base/capacity and byte-span rejection. Pure host; no numerical or GPU claim. |
| `qsa_indexer_device_batch.cpp` small | Builds real contiguous oracle prefixes at p0=0/1/3/4/5. Clones into independent candidate allocations, then compares after each split 1/2/3/5/4/1/7/2 chunk. Exact byte comparison covers ALL allocated pooled rows, tail, dead, block_pos and guards. A saved image is copied into fresh candidate allocations midway and republishes the moving spare from dead before further continuation. This is RESTORE-like component coverage, not actual session RESTORE. |
| GPU large | Tests full 4096 and 8192 batches, a seeded nonzero partial start, and final 3-row partial tails. Runs linked current block-score/top-k against both states. |
| Causal selection checks | Query-specific steps include counts 1/3/4/5/2051/2052 and each batch end when available. Both arms must have exact scores/IDs, sorted unique IDs strictly below each query's n_kv, identity below/through 2051, unchanged unused ID lanes, and untouched future-block score masks. Completed future pooled keys may exist but cannot become future causal IDs. |
| GPU physical | Allocates the full 65537-row pooled span in each arm, with full-row canaries before/after it. Seeds a REAL contiguous 262140-cell prefix through the unchanged single-cell oracle, clones it, then compares the final four cells against a genuine candidate batch. Checks spare 65536, block_pos and next-row 65537 canaries. This is deliberately a separate heavier suite; it is not a smaller allocation relabeled as full capacity. |
| GPU invalid | Invalid ranges, bases, capacities, geometry, scaling/epsilon, pointers/aliases/alignment, null/unordered queues and registered-table foreign contexts must reject before state changes. Checks every state byte and guard afterwards, including a zero-length no-op. |

Each GPU invocation uses one fixed explicit none/linear/YaRN scaling, one
analytic/full/short RoPE-table selection and plain/M-RoPE mapping. YaRN enables
the correction ramp as well as magnitude scaling. Short tables (1024 positions) exercise both
table and analytic fallback. Tables register before the append comparisons and
are unregistered only after a checked drain, before their owners are released.
These are explicit table-registration APIs, not environment mutations. Host DMA
operands use checked waits/fail-stop; guarded state and every extra fixture
queue remain live through checked retirement. Successful output is one compact
JSON record with component-only qualification fields set false for full-model
context, RESTORE and performance.

Manual reproduction for Main AFTER source review and idle integration; none of
these commands was run here:

```text
cmake --build <configured-SYCL-build> --target sycl_qsa_indexer_device_batch_test
cmake --build <configured-host-build> --target qsa_indexer_device_batch_plan_test
<configured-host-build>/qsa_indexer_device_batch_plan_test
<configured-SYCL-build>/sycl_qsa_indexer_device_batch_test small none analytic plain direct
<configured-SYCL-build>/sycl_qsa_indexer_device_batch_test large linear analytic plain direct
<configured-SYCL-build>/sycl_qsa_indexer_device_batch_test large yarn short mrope dispatch
<configured-SYCL-build>/sycl_qsa_indexer_device_batch_test invalid yarn full mrope direct
<configured-SYCL-build>/sycl_qsa_indexer_device_batch_test physical none analytic plain direct
```

The host target is registered under the existing
`STRATA_BUILD_PREFILL_HOST_TESTS` option. The GPU target is EXCLUDE_FROM_ALL and
has no automatic CTest execution. Set all runtime/driver/profiler variables in
the owned native child before exec. Table full/short modes additionally require
`STRATA_ROPE_TABLE=1` before exec. Dispatch mode requires exact
`STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH=1` before exec. Direct mode calls the
explicit production candidate without changing that option. Run none/linear/
YaRN, table and mapping combinations in separate owned processes; never flip
an environment arm after runtime initialization.

Main should use its existing exclusive measurement lock and owner/controller.
Suggested finite initial budgets: 90 s for host/small/invalid, 300 s for large and
900 s for the deliberately heavy physical suite; 64 MiB raw text per run, 45 GiB RSS
cap and fresh fault/journal boundaries. These are initial limits to review,
not measured runtimes. Record a timeout/budget stop as incomplete or failed.
Do not silently drop the oracle prefix or physical allocations to fit a budget.
No tensors, sessions or large traces are written by this fixture; comparisons
use temporary in-process buffers and one compact status record.

Remaining Main gates:

1. Verify every base SHA and integrate this proposal alone. Build the actual
   SYCL library/fixture and host helper, retaining compile command proof,
   source/binary/toolchain/runtime/device identities. Run existing native QSA
   fixtures with the flag absent/0, then the new direct/dispatch matrix and
   strict flag tests. This proposal has not even been syntax-compiled here.
2. Require exact state/guard/causal-ID parity after every batch and seeded
   continuation, including registered tables/M-RoPE, all partial boundaries,
  4096/8192 and the physically allocated last block/spare. Preserve every
   failure and its original status. The unchanged single-cell kernel is the
   independent scheduling oracle; numeric equivalence is a gate, not assumed
   from copied math.
3. Compare actual model prefill with identical source/configuration/tokens and
   flag 0/flag 1: selected rows/logits, final indexer/KV/GDN state, callback/MTP
   order, decode continuation, cancellation/fault closure and real RESTORE.
   Preserve the existing separate layer-major checkpoint limitations noted by
   R608/R680; this kernel does not fix their scheduling semantics.
4. Exercise real physically admitted 262144-context correctness and handoff
   before a maximum-context claim. Obtain clean repeated matched long-input
   timings independently of instrumented profiling, with decode regressions and
   fresh driver fault closure. Only then consider adoption. 600/1000 tok/s remain
   goals; no throughput or visible-gap reduction follows from launch counts.

Retention: keep this compact source/fixture/identity proposal for Main's next
idle review. No run artifact was created or cleaned. Future owned runs retain
individual outcomes, exact comparison summaries and fault closure; retire full
successful traces/duplicate logs only after evidence is verified, under the
measurement lock with path/hash/reason manifests. Failed receipts stay failed.
