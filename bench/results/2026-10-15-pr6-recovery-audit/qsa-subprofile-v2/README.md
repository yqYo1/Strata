# QSA subprofile v2 — source-only handoff

V2 extends the immutable v1 proposal; v1 was not edited. Main reports current HEAD04659cea contains report-only changes. This proposal deliberately remains based on the same pinned actual source bytes, not that later report commit.

Base commit: `a39e44a49e5ba38e58221714bd71acae52d3a2e1`. `base-manifest.json` pins the actual byte snapshots captured before edits. `source.diff` is generated exclusively from those snapshots, not from a later live worktree. Only this proposal directory was written. No Git operation, build, test, GPU/runtime launch, process intervention or cleanup was performed. All proposed code is uncompiled and unqualified.

Origin: independent own-fork observer and orchestration based on the pinned own-fork source, AGENTS.md, project-only upstream-sync skill, artifact policy, research-index and R727–730. No XeStrata source was read or used. The closed canonical1/profile1 model receipt supersedes the reports' pending-run statements: actual78453, C8192, FP32/R32768, selected768 exact, native/server0/fault0. Its instrumented QSA103980ms / wrapper235039ms motivates attribution only; neither this proposal nor that instrumented sample establishes an optimization.

Enable only in a separately owned native process whose environment is fixed BEFORE exec/runtime initialization:

```
STRATA_PREFILL_LAYER_MAJOR=1
STRATA_PREFILL_LAYER_MAJOR_PROFILE=1
STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE=1
```

The code only reads environment variables and never mutates them. As with the existing coarse flag, the fine flag is read on entry to the admitted long layer-major call. Short warmups use the existing run_impl dispatch and allocate no observer even when both flags are configured. The new flag requires the coarse flag on an admitted long call; other routes do not construct the observer. Without the new flag there are no fine event owners, marks, clocks or output. No unsupported-observer guard was relaxed, no legacy PfTimer was enabled, and no tensor math, data format, scheduling, chunking or ownership was changed.

Product files in `proposed/`:

- `include/strata/prefill/layermajor_qsa_profile.hpp`: new default-off diagnostic with fixed 48 rows, seven pairs/fourteen event handles, checked failures and new output prefix.
- `src/prefill/prefill.cpp`: nullable borrowed observer, seven whole-QSA-layer/chunk seams, retirement only after existing checked callback drain.
- `src/prefill/layer_major.inc`: admission/reservation, ownership, expected counts, final checked completion/close/emit.

`source.diff` contains exactly these three product changes. Fixtures remain separate handoff sources under `fixtures/`, not an automatic product-tree install.

Seven sections, in output order:

1. projections: K/V/Q and both indexer projections, K normalization and RoPE, plus weight lookup and dispatch host work;
2. KV stage: whole staged host-to-pool operation; resident mode emits an explicit no-op marker interval (not evidence of transferred data);
3. KV append and query normalization/RoPE, including forward rotations and indexer query normalization;
4. indexer append: whole chunk including either existing batch or per-token implementation, with only one outer pair;
5. selection: all selection batches, plus any already-enabled selection diagnostics and pool setup;
6. attention: all attention dispatch/fallback batches and reverse FWHT;
7. output projection: attention gate plus native projection.

Per-section `queue_ms` is the interval between in-order barrier command-end timestamps. It includes host gaps, interference and instrumentation, and is not GPU kernel service. `host_wall_ms` spans begin-marker submission through end-marker submission; it includes helper waits if those helpers block, and is neither CPU service nor GPU service. Both are explicitly NONEXCLUSIVE. The coarse v2 formatter and fields remain byte-for-byte unchanged; new lines start `strata lm qsa profile:` with `v=1`. Instrumentation can change measured coarse durations when enabled; output compatibility does not promise identical timings. The common denominator is the coarse `started_through_checked_lease_retirement` wrapper, excluding teardown; no restore duration is inferred. Individual host/queue intervals and the aggregate are attribution clues, not exclusive fractions or an Amdahl bound. Marker gaps between adjacent sections are outside individual section brackets.

Geometry and bounds (predicted, not measured):

| Geometry | QSA chunks | Samples per section | Total section samples | Extra record markers | Extra event queries | Extra elapsed calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 78452 engine-prefill rows / C8192, 10 chunks × 12 QSA layers | 120 | 120 | 840 | 1680 | 1680 | 840 |
| 262144 rows / C8192, 32 chunks × 12 QSA layers | 384 | 384 | 2688 | 5376 | 5376 | 2688 |

Each QSA layer has ten or32 samples per section; all36 GDN rows have zero. These maxima are for C8192, not all admitted smaller chunk sizes. Smaller chunks use `ceil(n/T)` expected counts without allocating per-chunk arrays; output remains fixed. `finish` checks every row/section against expected counts and rejects any pending pair before setting complete. Queries/elapsed are invoked only after an existing successful checked chunk drain. No pending wait, new drain, per-selectedbatch/query/expert events, or per-chunk output was added. A false/error query disables observation; elapsed is not called for that pair, no reuse follows, and owner retirement remains mandatory.

Output is exactly50 lines (header +48 rows +aggregate), seven scientific-format values in each time vector and seven bounded counts per row. With the appended dispatch fields, conservatively budgeting1024 bytes per row plus2KiB header gives52224 bytes, below64KiB. Storage is fixed; additional RAM admission reserves a further1MiB over coarse's1MiB allowance for the observer/event owners. This reservation is conservative, not measured runtime allocation overhead. Coarse output is separate from this extra budget.

Lifetime: fine owner and borrow are declared after coarse owner/borrow and before Restore/checked lease. Reverse destruction is checked lease retirement, Restore, fine borrow restoration, fine event owner destruction, coarse borrow restoration, coarse owner destruction. On success lease.retire precedes fine close/emit. On exceptions or ordinary early returns the existing lease still drains while all event handles, residual/cache/callback owners remain alive. Failure of that retirement still aborts with owners live. Partial event creation/record/query/elapsed failures mark the observer unusable and keep incomplete output; observer methods are noexcept and cannot bypass retirement. This relies on the pinned own SYCL compatibility EventDestroy remaining delete-only; CUDA/HIP validation remains outside scope.

Host fixture handoff (NOT RUN):

```
c++ -std=c++17 -Ifixtures/shim -Iproposed/include fixtures/qsa_subprofile_fixture.cpp -o /tmp/qsa-subprofile-fixture
/tmp/qsa-subprofile-fixture
```

Run from this proposal directory under Main's serialized build/test ownership. The shim deliberately exposes no wait/synchronize APIs. It asserts no ElapsedTime on pending marks and no event destruction before checked retirement; lifetime arms additionally require Restore before destruction on normal and exception exits. It covers full48-layer ten/32-chunk geometries, exact marker/query/elapsed and actual-dispatch counts, pair reuse after drain, GDN zero counts, missing samples, partial creation at first/middle/last (0/7/13 successful handles), duplicate begin, begin/end record failure, query failure, elapsed API failure, NaN/negative elapsed values, pending-pair finish without wait/query, inconsistent dispatch/query/staged/GDN/cap counters, per-record cap mismatch, and Destroy errors invalidating previously complete output. Every profile remains fifty lines; a 2MiB total text budget comfortably covers these bounded host-fixture cases. It is only a model of caller lifetime; Main must inspect/compile the actual product seams and preserve existing owner tests. It does not test the runtime provider or fail-stop drain behavior.

Main review gates: inspect complete diff against pinned source, verify unchanged coarse formatter and owner declaration order; compile host fixture and product under existing actual SYCL resolver; run old coarse-only gate/parser with the fine flag absent, short warmup, fine/coarse dependency and observer failure arms. Then use a separately owned same-request canonical/profile/fine diagnostic with selected768 FP32 comparison, exact per-layer/aggregate fine and coarse counts, normal graph/MTP/native owner restoration and fresh fault/closure checks. Any observation failure is incomplete, not a model PASS substitute. No clean performance, task quality, physical262144, real RESTORE, or >600/1000tok/s claim is made.


V2 actual-dispatch evidence is recorded by exactly one `qp->dispatch(matrix_used, staged, T, m.cap)` at the existing matrix/fallback decision, OUTSIDE `observe_attn_dispatch`. It uses the returned matrix-path boolean, not eligibility or the requested environment. No new queue activity is associated with counters. New appended row keys are `cap`, `expected_queries`, `dispatch_chunks`, `query_total`, `matrix_chunks`, `fallback_chunks`, `matrix_queries`, `fallback_queries`, `staged_chunks`. Existing v1 prefix/version and vector fields remain unchanged. `query_total` means the sum of actual per-chunk T, rather than request-token count. Stage count means `st.kv_mode==1` at that dispatch; it is not bytes or device activity.

For the current request, actual prompt tokens78453 include the reserved final token: the layer-major observer receives n78452. Therefore each of twelve QSA rows must finish with expected_chunks10, dispatch_chunks10, expected_queries78452, query_total78452 and consistent cap2051 (initialized from actual `m.cap`, checked against each actual dispatch). Aggregate dispatch_chunks120 and queries941424. All36 GDN rows have zero samples, expected/query/dispatch/staged/cap fields. At the physical262144/C8192 theoretical geometry, aggregate chunks384 and queries3145728; this is a source-derived fixture geometry, not a physical-context qualification. Matrix/fallback split and staged counts remain actual observations; no split is predicted.

`finish` checks each row's total actual dispatch chunks and total queries against its expected geometry, staged_chunks<=expected_chunks, all GDN counters/cap zero, and a positive consistent QSA cap across layers. Dispatch rejects excessive chunks, out-of-range T, query overflow relative to expected, and cap mismatch. Default-off/no-allocation behavior and checked owner retirement are unchanged from v1. `fail()` now explicitly clears `complete` so a Destroy error after `finish()` produces complete0 without relying only on `usable` in the formatter. Main must parse/verify the appended counters and cap before inferring any actual matrix/fallback attribution. Tests and compilation remain unrun.
