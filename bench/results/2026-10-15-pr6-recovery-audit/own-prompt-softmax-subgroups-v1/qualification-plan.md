Status: every command/run below is proposed for Main; **UNRUN**. This package changes no live product files. Main owns integration, builds, tests, GPU jobs, profiling, environment, retention and commits.

1. Apply gate and build identity

Wait for Main's current owned jobs to close and take the existing exclusive measurement lock. First decide whether the unchanged-binary official profile supports testing this hypothesis; a queue window is not a device service measurement. Review the entire proposal.patch. Verify every touched live file against manifest.json base_sha256 before applying; stop on drift and rebase the patch against an explicitly pinned replacement rather than replacing live files from files/. Main may use git apply --check and then git apply on feature/sycl. Do not combine R844 scale or R856 observer proposals in this comparison.

Use a NEW build root and the currently qualified compiler/runtime recipe. Enable STRATA_ENABLE_SYCL=ON and STRATA_SYCL_TESTS=ON; keep CUDA/HIP disabled. Build strata_kernels, sycl_prompt_native16_test, sycl_prompt_k_cellmajor_test and sycl_prompt_attention_test explicitly. The first two fixtures are EXCLUDE_FROM_ALL, so a default build does not suffice. CMakeLists.txt includes cmake/sycl_backend.cmake, which directly names src/kernels/sycl/prompt_attention.cpp in strata_kernels. Both manual fixtures link that library; the K fixture requires its existing STRATA_TEST_PROMPT_MATRIX=1 definition. Without that definition a manual compile uses decode attention and never covers this proposal. Record source/package/patch/binary hashes, exact compile_commands entry, link recipe, device, driver and runtime. Inspect the ACTUAL prompt_attention.cpp compile command for strict FP math (-fno-fast-math or an established equivalent, -ffp-contract=off, no fast-relaxed/native-exp/FTZ override); fixture-only flags do not establish kernel flags. No build/configuration was executed by the worker.

2. Correctness, serialized and separate processes

Use the existing Main-owned bounded process launcher, fresh fault/journal boundaries, immutable child environment installed BEFORE exec and one process at a time. No setenv/unsetenv or route changes inside an initialized process. A failed oracle, diagnostic, guard, validation, drain or fault check stops qualification without a tolerance change. Component deadline: 600 seconds per process; raw text budget: 64 MiB per diagnostic process. A deadline/budget stop is incomplete, never PASS.

Start with exactly two BF16 --smoke jobs, softmax=0 then softmax=1, each with the qualified tracing/parameter-validation environment. Each smoke retains its two original cases and the extended unsupported-span/nonfinite diagnostics. Preserve one representative excerpt per distinct unresolved failure, not complete repeated polling. Then perform the clean full suites:

| Fixture | Child environment per process | Number |
| --- | --- | --- |
| sycl_prompt_native16_test | softmax absent, then 0, then 1; old prompt flags fixed/absent | 3 |
| sycl_prompt_k_cellmajor_test | below C0..C4, each with softmax 0 then 1 | 10 |
| sycl_prompt_attention_test | softmax 0 then 1, direct=0 and other old prompt switches absent | 2 |

Here softmax means STRATA_SYCL_PROMPT_SOFTMAX_SUBGROUP. Absence, 0, any value other than the exact string 1 select false. All pre-existing valid mode/geometry rejection behavior remains unchanged. The BF16 absent-versus-0 runs must agree exactly, and Main must compare candidate false outputs for all original cases against a qualified unchanged binary/reference with the same source/flags/environment. Original dump case order is preserved as a prefix: 112 BF16 cases and 16 K fixture cases; the new full suites contain 116 and 20 cases respectively. Keep the established default regression gate; do not compare old and new complete dump sizes as though the added cases were absent.

| Config | STRATA_SYCL_PROMPT_DIRECT | STRATA_SYCL_PROMPT_K_CELL_MAJOR | STRATA_SYCL_PROMPT_NATIVE_KV_MIN | Covered FP32 specializations |
| --- | --- | --- | --- | --- |
| C0 | 0 | absent | absent | Mode0..3, Direct=false |
| C1 | 1 | absent | absent | Mode0..3, Direct=true, Native=false, KMapping0 |
| C2 | 1 | absent | 1 | Mode0, Direct=true, Native=true; other modes retain their paths |
| C3 | 1 | 1 | absent | Mode1, Direct=true, KMapping1 |
| C4 | 1 | 2 | absent | Mode1, Direct=true, KMapping2 |

Keep STRATA_PROMPT_ATTN_Q4 absent or at its existing enabled setting so Mode2 executes. C4 retains the original >=64976-byte device-local-memory gate; if unsupported, report the specialization unqualified rather than silently relabeling it. Existing K-layout logical_tile checks, all Mode0..3 FP64 comparisons, widths13/67/2051, missing/negative IDs/pages, zero and invalid widths, query scaling, guards, and batch-single checks remain. New FP32 sparse cases retain those counts and put the sole selected resident cell at slots12,0,2050 for queries0,1,2. The unchanged scalar FP64 reference independently yields that cell's V value; this tests both initial and trailing empty tiles in each format.

The BF16 suite retains its original regular width0/1/31/32/33/2051/-1/2052 cases at nq1/3 and page1/4/16, range/zero/smallp/positive-infinity-score/negative-infinity-score cases, causal IDs, table recovery, full-span checks, invalid Q/scales and diagnostic recovery. New sparse widths17/33/49/65 use nq3, page4 with missing pages; head0..23 are compared by the unchanged FP64/BF16 rounding budget. Late-only queries use the final selected slot (including slot16 of a partial CH tile); the middle query contributes at slot0 and then traverses missing tiles. Original finite extreme-scale/range cases already exercise broad logits and overflow-to-infinity behavior; no second oracle or relaxed tolerance was added.

The true route changes FP32 sum order. Require each route to pass the SAME existing FP64 oracles/tolerances, finite/NaN contracts and guards. Record cross-route numeric differences and fingerprints, without demanding bitwise identity between false and true. Existing within-route repeat and batch/single exact checks stay. Keep per-process route/env receipts; a successful default CTest run alone cannot qualify true.

3. Bounded performance screen after correctness

Run 16 clean component processes: dtype FP32/BF16 x nq128/256 x route sequence 0,1,1,0. Invoke the corresponding manual fixture with --bench NQ 1 --dump PATH under its existing validated settings (FP32 C1, BF16 legacy flags absent). Repeats=1 gives one individual measured wall sample per process for both harnesses; do not turn the FP32 aggregate repeat loop into a claimed sample distribution. Record all 16 samples, order, output hashes, exact environments and fault results. The benchmark oracle covers only queries0,1,2,last, so it supplements rather than replaces the full suites. Keep no cross-route exact-output requirement. Runtime initialization/warm-up, submission and completion behavior must be treated identically across arms.

Main may collect exactly one bounded device-service/counter profile per route on the SAME qualified component workload with its official tool and separate collector processes. Keep clean wall samples separate. Check actual compiler lowering, local allocation, subgroup occupancy, spills and device attention service. Preserve end-to-end dependency time as well; neither one busy CPU core nor overlapping queue intervals proves the relevant bottleneck. No device timeline, occupancy, speedup or CPU-bottleneck conclusion is established by this proposal.

4. Product gates and decision boundary

Only after the component screen warrants continuation, run the existing selected-quality controller at the current 32K fixture identity with each route once, then the existing physical262144 correctness/RESTORE boundary with each route once. Keep unchanged numerical/task gates, model/capture pins, shared-state and no-fallback requirements; do not edit controllers or captures to pass. Use a 3600-second owned-process deadline and 64 MiB raw text budget per job, or record a finite justified replacement before launching if the existing qualified fixture requires more. These four correctness jobs are not performance repeats.

If those pass, perform exactly four clean long-input prefill jobs in route order 0,1,1,0 with the same model, prompt/context, placement/chunking and all non-softmax settings. Record each sample and all faults/quality fingerprints; use the existing repeated long-prefill performance gate and compare the whole dependency path. A speedup or the 600/1000 tok/s milestones cannot be claimed from a component win or one run. Leave short-input/decode layout, shared state and physical-context obligations in force. Adopt only after Main has enough actual evidence; otherwise retain default and record rejection/hold with the individual failures/samples intact.

At the boundary Main commits reviewed source/reports and the exact decision. Keep compact metadata and a necessary unresolved failure excerpt; retire successful complete API traces, duplicated polling/timelines, and superseded tensor captures only under docs/ARTIFACT_RETENTION.md after verifying their consumers. This source-only package itself created no runtime captures and authorizes no cleanup of active work.
