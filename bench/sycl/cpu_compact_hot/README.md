# Bounded compact GU cache candidate

This is a source-only, CPU-component experiment on
`experiment/cpu-compact-cache`. Nothing is integrated into the production engine
or selected by the ongoing six-process decode comparison. No build, test,
numerical pass, RAM admission or performance improvement is claimed yet.

The isolated child is necessary because Main's current `feature/sycl` engine,
helpers and comparison inputs are frozen. Main owns implementation and all
testing. Integrate useful completed work into `feature/sycl` with its original
commits after the comparison closes, then retire this child. Rejecting this
candidate means removing its unused source after preserving the decision.

## Evidence and provenance

`compact_iq3xxs.hpp` is byte-identical to this fork's Main-authored kernel at
`313e5c4bc2d2e0c0fedb80020daf45662bec8a95`, SHA256
`f246d73832966a6c2cf954b32fd3c32b5ef82aed619f0353032e386474a135b4`.
The pinned GGML tables retain their MIT provenance; the cache and ownership
fixture are newly authored under the repository LICENSE. No XeStrata source,
design or proposal is used.

Original evidence is `cpu-roofline-current.json` at
`286b4a132b960a3a633511173846b57ebef32cee`. It measured all three gains for
single-thread cells and six-thread reuse cells (1.2401/1.2905 geometric-mean
speed ratios). Both six-thread streaming cells failed the repeated-gain gate.
Setup cost and raw/view co-residency remain charged; do not relabel that result
as all-shape or model adoption. The packed98B gather and activation-LUT
successors were rejected and are not restored.

The historical observer-qualified CPU leaf sample had 86.1% NT1 jobs and
65.6..66.5% of leaf CPU time in GU. Those CPU sums include memory stalls;
current critical-path and physical RAM saturation remain unknown. This
previously faster owned path can be prepared without pretending that the
hardware ceiling or its current model benefit is established.

## Component boundary

One owner represents one immutable model/source generation. A single host
thread prepares selected `(layer, expert)` views, then freezes the index. A
lease keeps encoded bytes alive after owner/raw-source retirement. No timed
insertion, eviction, ranking, environment mutation or runtime engine hook is
introduced. NT1 decode on the supported IQ3_XXS GU/Q8_K contract alone is
eligible; prefill, other formats and NT2+ must use their existing path.

The budget covers index capacity, guarded block arrays and Entry objects.
Shared-pointer control blocks, allocator overhead, packed-source/page-cache
residency and total RSS still need measurement. AVX2/FMA/F16C and the exact
2560/640 geometry are explicit constraints. No full 112.5-GiB int8 expansion is
planned. `lifetime.cpp` checks ownership/admission only; comparing calls before
and after retirement is not an independent numeric oracle.

The index's actual vector capacity is checked after construction; the known
byte budget is not a strict peak-allocation limit. Construction, freeze and
publication must be serialized. Native paired-versus-GGML selection and model/
role identity remain caller/bridge obligations, with actual-dispatch numeric
tests required before integration.

`tools/sycl/cpu_compact_hot.py` prepares a CPU-only icpx2026.2 recipe and checks
committed source/plan/tool identities before build/check. It reuses the exact
GGML3cf headers and the coherent build's actual dependency census, rejects
uncovered compile inputs and GPU libraries, and verifies the final executable
before the ownership test. Its shared measurement lock prevents overlap with
Main's GPU work. Build/check are finite, supervised by the existing pidfd
owner, with an8GiB virtual-address cap;the ownership check has a128MiB RSS cap.
Prepare the plan,commit it,then build/check only at a serialized boundary.
No invocation or pass is implied by this recipe's source preparation.

## Pending Main gates

1. Commit the exact CPU-only build recipe, compiler/dependency inputs and
   controller before building. Preserve the coherent engine/fallback; do not
   build, test or clean while another GPU job is active.
2. At a serialized boundary, build/run the bounded ownership fixture. Then
   compare real IQ3_XXS blocks against the actual decode dispatcher with the
   same Q8_K activation, rounding and reduction contract. Include both actual
   Down formats, disjoint row partitions, aliases, finite outputs and guards.
3. Commit a prospective timing plan before speed samples: lookup plus actual
   GU dispatch, setup/conversion and allocation separately, first use versus
   amortized reuse, bounded hot/cold coverage and raw fallback. Keep individual
   matched repetitions, timer floors, known bytes/RSS and observer conditions.
   The old mixed-NT synthetic distribution does not prove model cache reuse.
4. Admit a production bridge only after component gates and measured route/
   reuse/RAM evidence. Keep the bridge decode-only, preserve current prefill
   and collect repeated matched model results with strong/weak MTP. Physical
   context/RESTORE, task quality and default adoption remain separate.

Review point: after the active EARLY0 comparison closes. A prepared prototype
or one component pass does not complete the throughput goal.
