# Root SYCL QSA score grid-stride proposal v1

Prepared 2026-10-15 JST for Main's review after the active benchmark and requested PR6 audit boundary. This is a concrete, default-off experiment proposal. It is **unapplied, unbuilt, uncompiled/unrun, and unadopted**. No tests, GPU work, profiler, processes, services, resets, cleanup, Git operations or private/model data reads were performed. Only this proposal directory was written. Product/client/recipe files, live evidence, and the earlier Stager affinity/error and MTP observer artifacts were left untouched.

`proposal.patch` changes only `src/kernels/sycl/qsa_select.cpp` and adds `include/strata/kernels/qsa_score_grid_stride.hpp`. The `proposed/` directory contains the reviewable resulting files. `prepare_artifacts.py` is a source-pinned artifact generator; its execution only fabricated the patch, proposed source, extracted validator and provenance here. It does not apply a patch, import product code or invoke validation. The two fixture sources have not been executed.

## Scope and motivation

The configured resolver is `cmake/sycl_backend.cmake` → root `src/kernels/sycl/qsa_select.cpp`. `Verifier::record_window` calls the scores API without `active_blocks`; the decoder fast-selection call does the same. No CUDA/HIP source, top-k selector, indexer storage, KV streamer, attention, queue, wait, copy, event or graph policy changes are proposed. The tensor-core score entry point continues to return false. This flag affects any invocation of this root-SYCL score API, including an uncaptured call with an explicit active bound; it is not restricted to a single call site.

R550 and Main's R549–550 review motivate a separate score-grid experiment. With a fixed context capacity of 262144, `max_blocks=65538`. At a 32768-cell endpoint there are 8193 score candidates per query, including the final dead candidate. Current score launches have 65538 groups per query and return early for unused capacity. **These are source counts, not a measured bottleneck or speedup.** Main corrected R550's byte accounting: each query has four 128-wide FP32 heads, or 2048 B; logical references and cached/physical traffic differ. This proposal makes no traffic, cache or timing claim and does not implement the separate byte-radix selector idea.

## Exact policy and default-off proof

The environment flag is `STRATA_SYCL_QSA_SCORE_GRID_STRIDE=1`. Missing, empty, `0`, `01`, `10`, `true`, `yes`, whitespace and newline spellings remain off. The helper checks a nonnull pointer, character `1`, and immediately terminating character. The result is process-latched at the first call that passes the existing geometry and span validation. Set the flag before launching each fresh validation process; changing it after the first valid call cannot change this policy or an already captured graph.

The new branch appears after the existing validation and before the original submission. When off, execution reaches the **unchanged original kernel and its unchanged `finish(stream,event)`**. There is one host policy branch per call and a one-time host environment lookup; no additional GPU kernel, GPU branch, host allocation, clock or output is added to the off path. Adding another kernel to the binary can still change compiler artifacts, so unchanged source is not a measured timing guarantee.

The artifact generator copies the original FP body verbatim with indentation alone changed into the opt-in branch. The original body also remains intact. `fixture/check_source.py`, supplied **unrun**, checks that removing just the new branch and includes gives the exact pinned baseline, that the experimental FP statements match the original statement bytes after removing indentation, and that the extracted validator matches the actual source. This is inspectable source evidence, not a completed structural test or a GPU bitwise result.

## Geometry and ownership

Let `A` be the original validated bound: `blocks` if `active == -1`, otherwise the validated positive `active`. Let `G=min(A,4096)`. The new launch is `nq*G` work-groups of 32 lanes, with required subgroup size 32. A flat group maps to query `qi=group/G` and first block `group%G`; for each query a group visits `b=first; b<=nb; b+=G`. The shared helper implements this actual traversal and is called from the experimental device kernel and the CPU fixture.

The device still reads `n=steps[qi*kStepCount+kStepNKv]` and `nb=steps[qi*kStepCount+kStepNBid]` at execution. It retains the old validity conditions `n>=0`, `nb==n/4`, and **`nb<A`**. It must never compare `nb` to `G`: at full context `nb=65536` is valid with `A=65538`, although `G=4096`. Malformed rows produce no scores, matching the baseline's return behavior; this proposal does not change the caller's error contract. A zero KV count (`n=0,nb=0`) is valid and still computes the sole dead score. `nq<=0` remains rejected by the unchanged host validator.

Every valid `(qi,b)` has exactly one group owner, `qi*G + (b mod G)`. Score writes remain `scores[qi*blocks+b]`; score row stride is capacity, never `G` or `A`. Query stride remains 512 floats, pooled-key stride remains 128 floats, and step stride remains `kStepCount`. Blocks above `nb` and capacity beyond an explicit `A` remain unwritten. Host `active_blocks` validation is unchanged, including rejection of zero, values below -1, and bounds greater than capacity.

All 32 lanes follow the same validity and block-loop conditions because those depend only on the query step and work-group index. Each candidate retains the original four-head/lane dot-product accumulation, XOR subgroup reduction order, ReLU and head accumulation. `b==nb` uses `dead`, including a zero-width tail; only a nonempty tail (`n%4`) receives `+1e9f`. Neither tail skipping nor score approximation is introduced. Kernel arithmetic and ordering are source-preserved; subgroup compiler behavior and exact output equivalence still require Main's SYCL validation.

The constructor takes only the positive `A<=INT_MAX` already returned by the real validator. `G>=1`; actual device counts originate in int32 step fields; indices/loop increments use int64. The loop's final increment cannot overflow int64 under the existing validated bounds. Group count depends on immutable call geometry, not a host read of a changing step. Captured launch shape remains fixed across graph replay; context changes are consumed through the existing device step buffer and existing dependencies.

| Context endpoint, remainder 0 | Useful candidates/query | Old groups/query at capacity 65538 | Opt-in groups/query | Largest candidates/group |
| ---: | ---: | ---: | ---: | ---: |
| 32768 | 8193 | 65538 | 4096 | 3 |
| 65536 | 16385 | 65538 | 4096 | 5 |
| 262144 | 65537 | 65538 | 4096 | 17 |

These are integer source-derived counts. Total score arithmetic and useful score writes are unchanged. Fewer groups may reduce inactive scheduling overhead or may lose useful parallelism, change register occupancy, or worsen cache behavior. No favorable result is assumed. At `A<=4096`, the opt-in launch has the original group count and each valid group visits at most one candidate.

## Unrun fixtures and validation commands

`fixture/coverage.cpp` directly includes the proposed traversal header. Its host validator is the unchanged function byte-extracted from the pinned actual score source into `fixture/extracted_validate.hpp`; it is not a handwritten substitute. Its independent oracle enumerates all groups in the original bound and derives valid candidate counts using multiplication and remainder, without grid-stride traversal. It compares coverage, unique group ownership, query separation, full-capacity score address strides, untouched capacity and final dead-candidate ownership.

The matrix covers capacity 65538; `nq=1..4`; endpoints 32768/65536/262144; every tail remainder, including each query of a verifier window; explicit smallest legal batch bounds; bound edges 1/2/4095/4096/4097/8193/16385/65538; mixed valid/invalid rows; zero KV; negative and int32-extreme counts; inconsistent `nb`; `nb==A`; small capacities; and the actual host validator's empty/malformed geometry cases. The maximum valid host geometry is checked without allocating a corresponding tensor. This fixture checks integer geometry, not FP subgroup execution, device pointer spans, graph capture or model output.

The following commands are **for Main only after the stated boundary**; none was run by the implementation agent. Run from the existing worktree, serially. The CPU fixture uses the artifact's proposed include path and the pinned standalone product `qsa.hpp`; it does not link a product runtime.

```sh
proposal=bench/results/2026-10-14-primary-routed-fp16-census/qsa-score-grid-stride-proposal-v1
sha256sum -c "$proposal/SHA256SUMS"
python3 "$proposal/fixture/check_source.py"
c++ -std=c++17 -O2 -Wall -Wextra -pedantic \
  -I"$proposal/proposed/include" -Iinclude \
  "$proposal/fixture/coverage.cpp" -o "$proposal/fixture/coverage"
"$proposal/fixture/coverage"
```

The checksum file names are repository-root-relative. The compile command is supplied, not an executed build. The proposed files are not part of the current build. Main must recheck the target source hash before considering application; drift requires review/rebase, not force-application. The generator refuses a changed target source or an already existing product helper. No branch/worktree was created and no HEAD identity is asserted; file-byte pins are authoritative.

After source review, CPU fixtures and Main-owned application, use the existing configured SYCL recipe/build checks. Compare fresh-process off/exact-1 arms with identical model/configuration and device/runtime. Compare all live candidate scores bit-for-bit first, then unchanged top-k IDs exactly and selected attention/verifier results. Include all C/T/remainder shapes above, random and adversarial key/query values (equal scores, signed zero, infinity/NaN and tail bias), active-bound calls and malformed rows. Check replay with steps advancing within one fixed captured graph, including crossing 4096 and each tested context boundary; retain the original validation/fault/teardown gates. These GPU checks have no implementation or result in this proposal and must not be inferred from the CPU oracle.

Measure score phase and whole ordinary verifier/decode separately, clean/unprofiled first, at C=32768/65536/262144 and actual T=1..4. Keep selector time, streamer misses/copies and other window work distinguishable. Record each sample and the exact flag, source/binary/compiler/runtime/device identity. Reject adoption if any score/ID/correctness/fault gate differs or the primary shape/mode regresses. A lower group count alone cannot establish a benefit, and a score microbenchmark alone cannot justify decode adoption.

## Source pins, provenance and retention

The base `src/kernels/sycl/qsa_select.cpp` SHA256 is `45084756ca5e8e36085dae46581049c81366a45cdfde23e8c6f38defbba0c809`; proposed source SHA256 is `6dec0d5febfcdb82a7686145c9d0f762194d220977447b07afb2d2e325701969`. New helper SHA256 is `c7ddbfd332de2629e248275ba35b3d568dc61dfc3fe3f757f0bbfa8a9ee14de8`. Patch SHA256 is `fc44bc560960cbe02f79467713d9577b37a5d949a209b8cce4f8f1507ee0b8b4`.

`provenance.json` pins the actual resolver, call-site sources, headers, AGENTS, retention policy, registry, R550 and Main's R549–550 review. The registry is a time-of-read reference, not a future application precondition. R550's source-derived count and byte corrections were read together with Main's review; no literature speedup or alternative selector was adopted. `SHA256SUMS` identifies every compact proposal file except itself. No raw trace, tensor/session snapshot, external progress backup or cleanup was needed.

Consumer and next review point: Main, after the active benchmark and requested PR6 audit boundary, for review of this source-pinned patch and all serial validation. Keep this compact proposal until that decision is recorded; preserve rejection/failure results as well as successful samples under the repository retention policy. Earlier proposal artifacts remain distinct and unchanged.
