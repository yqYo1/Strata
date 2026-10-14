# QSA prefill fallback query transpose proposal

2026-10-14 JST. Artifact only: unapplied, unbuilt, untested, unmeasured and
unadopted. Main owns all runtime work. The sole writes are in this directory:
`generate.py`, `unified.patch`, `identity.json`, `provenance.json`, this report,
and a compact manifest. No product edit, branch/Git operation, build, test,
GPU/profiler execution, service/process change, cleanup or reset occurred.
The prohibited private `prefill-gemm-only-contract-root-v1/record.json` was not
read. Commands were bounded source/compile-database reads, hashing and artifact
generation. This personal fork's sole upstream is Niko1221/Strata; XeStrata is
optional reference material and supplied no donor code here.

`unified.patch` SHA256:
`1327bbe1d213ba7ff4ba98560bd1f43846ae36682c7cc997449f118de63e6d04`.
`identity.json` pins both targeted bases and proposed bytes. `generate.py` checks
those bases before writing artifacts and can reproduce the patch with
`python3 bench/results/2026-10-14-primary-routed-fp16-census/qsa-transpose-proposal/generate.py`.
It never applies the patch. The generator's root is derived from its canonical
artifact location. Move the artifact only after adjusting that explicit contract.

## Source and route evidence

Read AGENTS.md, ARTIFACT_RETENTION, the project-only upstream-sync skill and its
integration/evaluation references, the current research index with Main's
corrections, prior compact-MoE index/R515 and current R522. Their exact hashes
and those of reviewed source files are in `provenance.json`. Current prefill is
Main-owned and changing; it is intentionally neither patched nor hash-pinned.

Current configured root SYCL source resolver in `cmake/sycl_backend.cmake`
selects `src/kernels/sycl/decode_attention.cpp` and
`src/kernels/sycl/prompt_attention.cpp`, with root `src/prefill/prefill.cpp`.
Current prefill attempts whole-chunk `qsa_prompt_attn_batch`; only force-bypass
or return-false enters sliced `qsa_decode_attn_batch`. The transpose proposal
adds its own public prefill entrypoint and separate internal launch in the
already selected decode-attention translation unit. No new CMake source is
needed. Existing `launch<Format>`, `qsa_decode_attn_batch`,
`qsa_decode_attn_step`, scratch sizing and matrix selection remain unchanged.
The shared header exposes the added API only under `STRATA_ENABLE_SYCL`.
CUDA/HIP implementation and validation are outside scope.

The frozen own-fork old source is
`.tools/pre-xe-1eb-source-v1`. Actual old compile database
`/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-pre-Xe-1eb-build-root-v2/build/compile_commands.json`
selects `S/sycl/src/prefill/prefill.cpp` and
`S/sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp`; the latter is compiled SYCL
despite its `cuda` directory name. `provenance.json` preserves both exact
selected commands and the database hash. It does not treat root `.cu` as the
compiled old Arc engine. Old flags include icpx 2026.1, `-O3`, C++20, `-fsycl`,
subgroup default32, per-kernel split and `-fp-model=precise`. Current CMake
kernel target uses `-ffp-contract=off`; Main must pin the actual new configured
compile commands, toolchain and binary before comparing.

Old prefill lines933/3951 select layout only inside fallback. Layout1 dispatch
at old attention line631 is `launch_chunk_variant<KV_MODE,32,true,false>`:
64 selected cells, eight SG32 subgroups (256 work-items), 12 Q heads per KV
head, D256. Its 3D group axes are query/KV-head/chunk, chunk fastest. The new
flat group mapping preserves that logical order: chunk=`group % chunks`,
KV-head=`(group/chunks)%2`, query=`group/(chunks*2)`. Grid size is
`nq * 2 * ceil(cap/64)` groups plus the existing-form merge over
`nq * 24 * 256` output elements.

The preserved current diagnostic binary is native68da0262, full SHA in the
long-phase summary. It consumed 34,561 real inputs: 34,560 batched as eight
4096 chunks plus1792, then the final real token on nativeT1. Context was
262144, INT8 KV, resident32768, compact FP16=1, MMQ=0, ring8, MTP4, split off.
Its QSA attention interval was49.455134 seconds, nonexclusive and affected by
existing phase barriers. That interval is not matrix-kernel service time or a
critical-path attribution. Gate/output projection is in the separate QSA
projection phase. Matrix acceptance/decline was still unobserved in this
receipt. Main's new route counters are the prerequisite for deciding whether
this fallback candidate runs at all. No historical old/current PP gap or this
interval establishes a transpose benefit.

## Actual change and numerical contract

The only restored old access pattern is FP32 query-local transpose. For head
`h` and original dimension `d`, store at
`h*256 + (d%8)*32 + d/8`. Score lane `l` reads its eight consecutive original
dimensions via `h*256 + j*32 + l`, `j=0..7`. This is a bijection: inverse
dimension is `8*l+j`, and every query local element is written exactly once.
Global query reads remain linear/coalesced. K stays eight adjacent dimensions
per lane; V reads and the chunk/merge layouts are unchanged. The word cached
here means per-work-group local memory, not a new persistent query cache.

Old `sq` was already float; its INT8 load multiplies float code by a converted
FP16 scale, and its accumulators/partials were float. The transpose introduces
no dtype narrowing. However, wholesale old code would change arithmetic: old
score uses an eight-term multiply/add expression and old softmax uses
`sycl::native::exp`, with a different compiler FP contract. The proposal
instead reuses current `load<1>` (`f32_from_f16`, float code times float scale),
current first multiply followed by seven explicit `sycl::fma`, XOR32 reduction
order16/8/4/2/1, scale0.0625f, current `sycl::exp`, FP32 chunk sums/accumulators,
value FMAs, and the current merge's exp/FMA/division expressions. It retains
current `-INFINITY` masking and fully zero-initialized empty partials rather
than the old `-FLT_MAX`/partial initialization behavior.

For valid metadata and the same compiler, mathematical operation order and
input float values match current fallback. This source argument is not a
bitwise runtime result: compiler code generation may differ across the two
lambda kernels. Main's first gate is bitwise attention/partial, logits and
residual comparison against current forced fallback. Any difference requires
an explained numerical investigation and explicit quality criteria before
adoption; do not silently relax the check or compare diverged generated
histories. Matrix-vs-fallback is a separate, already non-bitwise comparison.

## Admission, bounds, aliasing and lifetime

`qsa_prefill_attn_transpose_batch` returns false before enqueue for unsupported
shape, mode, storage extents or reported device capabilities. Supported shape
is nonempty nq1..65535, cap1..INT_MAX, Q24/KV2/D256 and positive page size up to
INT_MAX. INT8 K and V plus both scales/page table must be populated; FP16,
Q4/hybrid or conflicting pointer families decline. Decline has no writes,
submissions, allocations, device reads or waits. A zero-query call also
declines; unchanged fallback owns its ordinary zero-query behavior.

There are no allocation sizes in `QsaAttnPools`, so the new API requires an
explicit owner-supplied `QsaPrefillAttnStorage`. Zero/unknown extents decline.
`logical_cells` is physical context capacity, not selected-cell cap, current
prefix length or resident count. `table_pages` and `pool_pages` are allocated
logical-table and physical-pool pages. All are positive and at most INT_MAX,
and table entries must cover ceil(logical_cells/page_size). Real full context
therefore permits positions0..262143; cell262144 is outside and is masked
before any lookup. No spare indexer capacity is mistaken for another valid
context cell. Table/pool allocations retain their actual sizes, including any
owner-provided spare, in the span checks.

Capability checks evaluate the same queue's device once per new entrypoint:
SG32 is actually listed, work-group and one-dimensional work-item limits are
at least256, and reported local memory is at least15872 bytes. The arrays are
3072 FP32 query floats (12288 bytes), 768 FP32 score/probability floats
(3072 bytes), and64 int64 rows (512 bytes). The code reuses current prompt
admission's reported-device query pattern; it does not infer SG32 support from
SG16 matrix admission or from the Arc model name. A capability query exception
declines. No new device capability probe kernel is submitted. Main must record
actual returned subgroup/memory/WG values and compiler resource limits during
the small gate; this artifact has not measured them. If counts are instrumented,
use this evaluated outcome, without a second query for reporting. Per-call
host capability-query overhead also belongs in a quiet candidate comparison.

Checked products bound nq*heads*D, nq*scratch stride, IDs, steps, table bytes,
full K/V code/scale bytes, group count and groups*256 before submission. Both
SIZE_MAX and the existing signed-64 checked_count contract apply. Extent
overflow declines; invalid pointer alignment/address range or overlapping
writable spans throws before enqueue, preserving the existing safety error
contract rather than concealing it as a safe fallback. Read-only arrays may
alias each other where otherwise valid. Query, IDs, steps and table use4-byte
alignment; code arrays use1, scale arrays2; scratch/output already require4.
`validate_spans` verifies pointer arithmetic and disjointness, not backing
allocation size or initialized content. Truthful allocated extents, context
ownership and initialized selected rows remain allocation-owner obligations.

Device code keeps the original width rule: kStepWidth in0..cap gives a per64
chunk clamped width; malformed width produces a completely empty partial.
It does not reinterpret kStepPos/NKv, recompute selections or extend causal
width. The existing selector must continue to supply only causally valid IDs.
Negative IDs and negative page entries remain masked. Additional guards mask
cell>=logical_cells before reading the table and page>=pool_pages before
reading either pool. Positive valid pages use exactly
`(page*2 + kvhead)*page_size + cell%page_size`, computed in size_t; admitted
full-pool byte extents bound this row and its final D256/scale4 access. Staging,
identity, streamed nonresident and ring lookup preserve the supplied table;
the candidate creates no residency decision or cell remapping. Malformed
positive IDs/pages have no current valid-output parity promise: current code
could access outside its buffers, while the candidate masks them. Main should
verify this defense with canaries, not deliberately enqueue them into current
fallback as a reference.

All work-items in a group observe the same `here`, so the empty return is
group-uniform and writes all G maxima/sums and G*D accumulators. A nonempty
group writes all query elements, all64 rows, and all G*64 scores (including
masked cells) before their barriers. Eight subgroups each handle cells
warp/warp+8/... uniformly; no lane-dependent branch encloses a barrier or
subgroup reduction. Every head/probability is written before value reads.
All-masked and partial tails write zeros through the same current merge rule.
There is no uninitialized local-memory read or divergent work-group barrier
in this mapping.

Two launches use the same existing in-order queue: chunk partials, then merge.
Prior producers/staging/selection on that queue precede the readers. The merge
finishes before downstream gate/projection or scratch reuse enqueued on that
queue. No added dependency event, queue, host wait, copy or allocation exists.
`finish(stream,event)` retains current semantics: nonnull stream returns after
enqueue; null stream uses the existing runtime wait, with no extra wait. Main
must retain query/IDs/steps/table/pools/scratch/output until completion and
preserve existing cross-queue producer events if its call route uses them.
There is no host mutation or reclamation permission from this API.

## Host hook fragment for Main, absent from patch

Keep the existing `old_attn || !qsa_prompt_attn_batch(...)` expression and its
route counters. Inside its existing `m.attn_batch` loop, replace only the
individual unconditional decode-fallback call with this fragment. The exact
value1 enables; omitted,0,10 and other values are off. The current prefill
already includes `<cstring>`. Check owner dimensions against current source
before wiring: `take_stage` allocates by the primary QSA state's n_pages and
`ident_table` has that length; `qsa_state_init` allocates table by st.n_pages
and ordinary pools by st.n_slots. Do not guess extents from resident32768 or
from m.cap. Shared layer shape/capacity agreement must remain true.

```cpp
bool transpose_taken = false;
#if defined(STRATA_ENABLE_SYCL)
static const bool transpose_enabled = [] {
    const char* v = std::getenv("STRATA_PREFILL_QSA_TRANSPOSE");
    return v && std::strcmp(v, "1") == 0;
}();
if (transpose_enabled) {
    const int64_t staged_pages = ss.qsa_states[ss.qsa_primary()].n_pages;
    const strata::kernels::QsaPrefillAttnStorage storage{
        st.max_cells, staged ? staged_pages : st.n_pages,
        staged ? staged_pages : st.n_slots};
    transpose_taken = strata::kernels::qsa_prefill_attn_transpose_batch(
        m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
        m.steps_dev + t0 * strata::kernels::kStepCount,
        m.cap, s, storage, m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
}
#endif
if (!transpose_taken)
    strata::kernels::qsa_decode_attn_batch(
        m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
        m.steps_dev + t0 * strata::kernels::kStepCount,
        m.cap, s, m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
```

This hook never changes decode calls elsewhere, default prompt selection,
chunk size, selection cap, math, staging, compact mode, KV format, MTP or
ring configuration. Forced-fallback comparison uses the existing
`STRATA_PROMPT_ATTN_OLD=1` seam separately. That legacy variable tests presence,
so setting it to0 does not restore auto selection; omit it for auto arms.

## Main's serialized test and decision matrix

All entries below are pending, not executed here. First finish the owned run
and route observations. Main reviews pinned bases, patch and exact hook,
records actual configured TU/flags/binary, then builds and runs serially under
the campaign's measurement lock. No cleanup/build/test/GPU overlap. Keep
individual failed, rejected and successful outcomes.

| Gate | Arms and required evidence |
| --- | --- |
| Route | Natural prompt auto OFF/ON; separate forced-fallback OFF/ON. Count matrix attempts/accepts/declines/force-bypasses and transpose attempted/accepted/declined calls plus query totals/shape histogram. The new function accepts only nonempty batches; zero calls are no kernel evidence. If all natural calls are matrix-accepted, report candidate unexercised under that route. |
| Capability and decline | Actual SG32/local15872/WG256 and compiler limits. Unsupported Q/KV/D/page/nq/modes, missing/unknown storage and overflow decline before submit; malformed spans throw before submit. Exercise every supported owner layout. Keep default/OFF and existing decode behavior evidence. |
| Short tensor | Same initialized INT8 arrays and selection/steps for baseline/current forced fallback and candidate; nq1/2/3/31/32/33, caps1/63/64/65/133/32768, widths0/1/63/64/65/cap, invalid widthsnegative/cap+1, partial query tail, page sizes1/4 and another legal positive size. Identity/permuted/ring/nonresident maps, negative IDs/pages, all masked, alternating masks, final physical ID262143, distinct scales, tiny/large query magnitudes. Compare full attention and meaningful chunk partials bitwise; guard allocations and finite results. Use invalid positive cell/page only in candidate defense tests, without unsafe current-reference launch. Cover valid owner extents and output/scratch-versus-later-pool alias rejection. |
| Short model | Same frozen prompt/history, compact OFF/ON as required, chunk+tail (e.g.130 real tokens/chunk64 =129 batched64/64/1 plus final nativeT1), matching fallback counts and identical logits/residuals. Unchanged decode kernels still require cold/restored continuation checks; source identity alone does not prove model state equivalence. |
| Long >=32K | Actual >=32768 newly consumed real tokens, including last nativeT1; pin token/body/rendered hashes. Reuse the canonical34561 fixture:34560 batched8*4096+1792 plus final input. Repeat with same context262144/INT8/resident32768/compact/MMQ/ring/MTP/split. Confirm query tails and exact route. A standard suite case with actual32049 tokens is separate and is not >=32768. Keep clean unprofiled reference; existing nonexclusive timing may be diagnostic only. |
| Physical262144 + zero output | Actual262144 input, max-new0, correct supported full-input route, last valid position262143 consumed and committed;262143 batched plus final real nativeT1 where that route applies. Verify actual chunk ranges, page/indexer/spare bounds, finite logits/state, no emitted IDs and target state completion. Entry rejection, allocation success, logical capacity,64K or accounting alone is insufficient. Existing full-input proposal is itself pending and cannot supply a runtime pass. Reject additional output at capacity. |
| Near boundary and RESTORE | Separate262143input/max-new1 boundary case; normal generated continuation where space exists. Cold vs saved/RESTORE fixed histories at short,long and near-full contexts; verify full physical fixture retains all262144 cells and correct tables/indexer/target state. Restore full context with zero output and compare head/state where supported; do not process position262144. Compare resumed ordinary continuation only where valid room exists. Preserve MTP bind/effective-spec and ring residency evidence. |
| Quality | Same-history teacher-forced logits/top1/logprobs/distribution/residuals at short,long and near-full positions plus established needle/task suite. Explain any non-bitwise difference before setting tolerances. Record task scores per arm; varied generated histories are not direct tensor pairs. No precision narrowing is proposed, but no quality result is claimed. |
| Formal stock | Same unchanged Niko0.1.43 `bench.sh` and clients from the pinned pristine checkout; full suite with `--long --runs 3`, identical server/input/client settings and separate quiet arm processes. Run natural auto transpose OFF/ON; forced-fallback OFF/ON are separate settings to explain the component if needed. Keep at least3 individual PP/cold-decode/restored-decode samples, interleaved order, usage/read/reuse, offer/accept counts, median/range and all failures. Natural-route gain requires actual candidate calls. Independent longer decode repetitions remain required;24-output stock decode alone does not settle regression recovery. |

Canonical formal client checkout is
`/home/yayoi/ghq/github.com/Niko1221/Strata/.worktree/bench-upstream-v0143-pristine-20261012`.
Established command form is `./bench.sh --url http://127.0.0.1:18095 --long --runs 3`
(Main chooses the owned localhost port). Reference `stock-comparison.json`
pins bench.sh SHA `7091b38069a62ec315043d720a3e2f98cc82216a3ba4fe72790764fea49db963`
and its dependent clients; Main verifies the actual same unchanged bytes for
both arms. No custom tensor/CLI diagnostic or49.455s interval substitutes
for formal speed samples. Hold input, accepted chunks, context, cache/reuse,
KV residency, compact/MMQ/ring, requested/effective MTP and split fixed.

No adoption decision follows this artifact. Retain the existing implementation
unless the supported prefill component passes safety/numerical/quality/full
input/RESTORE and repeated matched performance gates without decode regression.
Record an inactive or declined candidate honestly. Default remains OFF until
Main makes an evidence-backed component decision.

## Retention and next review

Owner: Main. Consumer: pinned patch review, serial application/build and the
matrix above. Next review: after current route-counter evidence and runtime
work close. Only compact source proposal/provenance files are retained here;
no tensor, session, profiler timeline or raw trace was created or deleted.
Main's future diagnostic jobs need a finite deadline and64MiB raw-text budget
unless a named exception is recorded before launch. Keep a representative
distinct failure mechanism, per-run owner/exit/boot/fault results and original
statuses. Any required full-context/RESTORE capture stays outside Git with a
named consumer/hash until those comparisons close. Main commits artifacts at
the next authorized boundary; this agent performs no Git mutation.
