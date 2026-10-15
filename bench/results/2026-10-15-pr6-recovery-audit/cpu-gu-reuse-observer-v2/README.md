# CPU GU recurrence observer v2 — proposal only

No product edit, application, compilation, test, timing, Git command, model/service/GPU operation or cleanup was performed by this implementer. Only this directory was written. Main owns review/rebase/application and every build/test/model operation, serially after its current profiling work. The source diff is a proposal against the byte hashes in `pins.json`; it is not adoption of a cache or inference change.

Main reports that the CPU nt1 expanded win repeated across four processes. Actual model job counts were nt1=46,071, nt2=15,757, nt3=4,361, nt4=2,592. These are communicated Main results, not independently verified measurements by this author. Only nt1 expanded performance is measured; multi-token expansion has no qualified speed result. V1's all-job LRU can contain entries admitted/promoted by nt2..4 and therefore cannot evaluate an initial nt1-only cache policy. V2 fixes that consumer issue with independent policy states, not by filtering v1 hit totals afterward. It does not implement a cache or claim a model speedup.

V1 is preserved untouched; Main reports its committed boundary as `089fa933`. V2's source diff targets exactly the **same unapplied product baseline** pinned by v1. `write_proposal.py` asserts equality of all original modified-product input hashes before generating a diff. It does not apply v1 or v2, rebase, fetch or query Git. R595/R598 and v1 remain pinned design references; new evidence supersedes their historical unrun status, not their model/RAM adoption gates.
## Files and Main-only checks

- `source.diff`: unapplied product changes, including the complete new header.
- `cpu_gu_reuse_observer.hpp`: the same header separately, allowing a dependency-free CPU mock before any product application.
- `cpu_mock.cpp`: independent vector-LRU oracle for both policies and all four capacities, mixed formats/unknown types/nt, byte and denominator conservation, exact expert origin grouping, strict nt1-only isolation across 12,000 intervening nt2..4 jobs, request-cold reset, off wrapper, count/fill/bypass overflow latches, base64 width/padding roundtrip and output canaries. It prints exact `sizeof(observer)` and `sizeof(Lru)` when Main runs it; a compile-time assertion caps observer state at 2.5 MiB. Not run here. The off wrapper covers the observer call boundary, not full product numerical parity.
- `write_proposal.py`: regenerates only this directory's diff and input pins from current source; refuses mismatched anchors. It does not apply or run the proposal.
- `pins.json`: input hashes, intended product paths, baseline assertion and owner; `manifest.json` pins proposal bytes.
- `transform_v1_to_v2.py`: disabled historical one-shot authoring transform, retained only as construction provenance. Do not execute; the final header is authoritative and the supported generator is `write_proposal.py`.

When Main is idle and has the exclusive measurement owner, proposed CPU-only commands (not executed):

```
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic cpu_mock.cpp -o /owned/private/cpu_gu_reuse_mock
/owned/private/cpu_gu_reuse_mock
c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer cpu_mock.cpp -o /owned/private/cpu_gu_reuse_mock_asan
/owned/private/cpu_gu_reuse_mock_asan
```

The mock has assertions; do not compile with `NDEBUG`. Use Main's pinned compiler where required; commands above are illustrative, not new compiler pins. Main must review diff, verify input pins, apply only after current work closes, check actual compile commands, and run product OFF/ON matched output/token/work and route/service conservation before interpreting real-model data. Observe standalone correctness separately from performance; the observer adds issuer work and may perturb decode wall. No overhead or equivalence result exists yet.

## Actual source and identity

`CMakeLists.txt` includes `cmake/sycl_backend.cmake` for SYCL; that file's `strata_engine` explicitly compiles `src/core/expert_source.cpp` and its `strata` executable compiles `src/program/generate.cpp`. The latter constructs `DecodeServiceBudgetScope` after prompt setup. The proposal adds no source resolver, CUDA/HIP engine or kernel port. It hooks the existing native CPU multi dispatch also used by native single-token `expert_pool_dispatch` delegation. Main must verify its configured build's effective source command still resolves these pinned files.

`expert_pool_dispatch_multi` computes CPU ownership after GPU/PCIe/peer/remote routing. It calls `d.src->blob(d.layers,e)` only for a new CPU expert, then stores that pointer in `jobs_multi[jo].blob`. At that exact successful assignment the observer copies the integer `e` into a small issuer-local `job index -> expert ID` array. It never derives expert ID from job slot, output slot, pointer arithmetic, mmap location or route frequency. After grouping is complete it visits job-array order `j=0..njobs-1`, with each job's final nt, before native dispatch. This is deterministic first-appearance order within each CPU batch; worker execution order is concurrent and is not modeled as a serial trace.

Existing `FileExpertSource::blob` may select RAM complement, exchange override, staged source or mapped source for the same `(layer,e)`. All resolve to the actual expert identity passed to the source. Source location/residency is not exported and no additional `blob()` call changes source read accounting. The observer excludes every GPU/PCIe/peer/remote-owned route; all CPU jobs count regardless of backing storage. Original route/service ledger code and output remain present. No worker reads/writes the observer; no added worker atomics, clocks, barriers, waits or FP operations exist.

## Lifetime and opt-in

Enable only for an eligible existing joined request:

```
STRATA_DECODE_ROUTE_LEDGER=1 STRATA_DECODE_SERVICE_CPU=1 STRATA_CPU_GU_REUSE_OBSERVER=1
```

Exactly the string `1` enables the new option. Default OFF adds no observer allocation, clocks or output; the pointer guard is false and origin-array bytes are not initialized or read. The environment is inspected only after an existing joined state has successfully been constructed. Observer allocation is nothrow; allocation failure or an ineligible shape yields no observer receipt and is not zero-reuse evidence. A qualified run requires both its normal complete joined receipt and a complete new receipt with matching request identity.

Domain is exactly 48 layers × 512 experts; GU geometry is H2560/FF640. Known types must have their expected packed rows and gate/up/down offsets. Unsupported native GU types remain counted per expert and as `group=3` but bypass simulation; no unknown expanded size is invented. Any shape/type-layout violation, out-of-domain identity, nt outside 1..4 or counter overflow latches invalid. The existing kernel MAXT8 is unchanged; observing nt5..8 makes this four-nt diagnostic incomplete and changes neither dispatch nor context limits.

The observer is allocated after prompt and borrowed through the request's `ExpertDispatch`, owned by `DecodeServiceBudgetScope`. Every request begins with an empty simulated cache and fresh counters, including RESTORE/repeated prompts. Nothing persists across requests/model loads; it cannot establish a model-warm persistent-cache hit rate. It observes decode CPU jobs only, not prompt routes or prefill service. A real model-scoped cache would need source/model identity, invalidation, request/error/cancellation/RESTORE lifetime and warm-start experiments separately.

Submission updates happen after grouping, before workers. Completion accounting happens only after `run_split_multi_native` returns through its existing final parking boundary. At finish the existing pool-bound/fail-stop check is retained; a request is not read out or destroyed while pool ownership is unproved. After normal detachment/snapshot, a separate footer is emitted at request close with the joined request sequence; the original joined ledger output follows unchanged. Error/early exit yields incomplete evidence through the existing ledger's `complete(...)` predicate. No observer changes original result status or adds a synchronization point.

## Exact simulation and bounded cost

Eight independent conventional demand-admitted byte-budget LRUs simulate 256 MiB, 1 GiB, 4 GiB, 8 GiB for each of two policies:

- `policy=0 policy_name=all_supported`: admit and touch every supported-format job, nt1..4. This is a hypothetical future all-nt policy whose nt2..4 speed has not been measured.
- `policy=1 policy_name=nt1_only`: only supported-format nt1 jobs can admit, hit, promote or evict. Every supported nt2..4 job bypasses this policy entirely, including jobs for an expert already resident because of nt1. This is the consumer for the currently measured nt1 expansion candidate.

Unsupported formats bypass both policies. `policy_skipped` reports unsupported jobs by nt separately from supported-format jobs bypassed for policy reasons; these categories are disjoint. First-touch/retouch/per-expert counters remain global over **all CPU jobs**, including unsupported and policy-bypassed jobs. No counter in one policy changes another policy's membership/recency or per-policy statistics. LRU state is not shared across capacities. They track exact membership, MRU/LRU links and payload byte occupancy over **this request's serial issuer job sequence**. A hit moves an entry to MRU; a miss admits the whole GU expert and evicts LRU entries until it fits. Payload sizes are exactly:

| GU type | expanded block B | blocks per GU expert | expanded GU B |
| --- | ---: | ---: | ---: |
| 18 IQ3_XXS | 266 | 2×640×10 | 3,404,800 |
| 22 IQ2_S | 266 | 2×640×10 | 3,404,800 |
| 21 IQ3_S | 262 | 2×640×10 | 3,353,600 |

The representation includes original half-D bits, signed codebook bytes and raw subscales. It matches the probe, including the metadata omitted in R595's earlier estimates. No expanded payload is allocated, prepared, read, cached or uploaded by this proposal. Each policy assumes all three known types are admitted when its nt rule permits. It does not estimate policy subsets, prepare costs, allocator/page overhead, LLC hit rates, bandwidth or time saved. Multi-token jobs count as one cache touch with nt token-expert entries; existing packed GU already reuses decoded weights across nt within a job. Hit-token counts must not be mistaken for new reuse opportunities within nt.

Each supported nt1 touch does eight constant-time unlink/front operations plus evictions; nt2..4 touches do four plus a bypass-counter update. Each admitted entry can be evicted once before its next admission, so total LRU work is amortized O(8×jobs), not a domain scan per touch. No sorting, time sampling, filesystem read/write or allocation occurs per touch. Counters and eight LRU link sets have fixed storage (compile-time cap 2.5 MiB; exact `sizeof` is printed by the unrun Main mock and on each receipt), independent of context/request length. Receipt work is bounded by 98,304 fixed counters, their finite ULEB128 width and 256 KiB output storage. Counter overflow stops further updates; it does not wrap or silently reset. There is no artificial request/context cap.

The 8 GiB ceiling and **8 GiB live available-RAM reserve** remain mandatory gates for any later real admission. These are payload-only hypothetical budgets, explicitly marked `budgets_ignore_live_ram=1`; they do not prove RAM feasibility. No allocation claim can be inferred from them. The observer does not poll MemAvailable or RSS on every batch. Main's matched model run must record live MemAvailable/cgroup headroom, anonymous/file/peak RSS and future cache-owned bytes; a real cache would need admission limited to `min(8 GiB, live available - 8 GiB reserve - metadata)` throughout, with packed fallback. No claim that an 8 GiB cache is safe is made here.

## Receipt and conservation

Separate bounded block, at most 262,143 emitted bytes, no per-call file trace or source pointers. Header includes request ID, complete/overflow, submitted/completed jobs and token-expert entries, unique first touches, retouches, unsupported jobs and simulation caveats. Each policy×budget reports hits/misses, fill payload bytes, evictions/evicted payload bytes, hit-token entries, resident payload bytes and the exact expanded entry sizes for types18/22/21. Every policy×budget×supported format×nt row reports hit/miss counts, including explicit zero rows for nt2..4 under nt1-only. Fill payload is the sum of the exact whole-expert size on every miss, including cold first fills and refills after eviction; it is a logical write-volume requirement, not measured prepare time or DRAM traffic. No hit/miss denominator includes skipped unsupported or policy-bypassed jobs. `format_nt` provides job and token-entry totals for group0=18/group1=22/group2=21/group3=other × job nt1..4. There are no extra format×nt clocks; existing service ledger clocks retain their original meaning.

Per-expert format is the 48-entry `formats` vector indexed by layer. Per-expert nt job counters are one dense `expert_jobs_uleb128_base64` line: standard base64 of unsigned LEB128 integers ordered layer0 expert0 nt1..4, layer0 expert1 nt1..4, ... layer47 expert511 nt1..4. There are exactly 48×512×4 integers. Per-expert token-entry counts are **exactly nt×the corresponding job count**. Zero counters remain explicit; integer IDs and formats reconstruct every expert without address leaks. This encoding bounds common sparse output near 131 KiB; larger values remain exact if the whole block fits. Decode the complete line before using it.

On output capacity exhaustion, discard the entire detailed block and emit only `complete=0 reason=output_cap`; no truncated per-expert table or partial hit claim is presented as complete. Counter overflow/invalid route, missing completion, error status, or failure of original joined completeness makes complete=0. Consumers must reject all incomplete receipts for adoption decisions. Original route/service results are still printed unchanged.

For a complete receipt, observer jobs/tokens equal both its own parked-return completion totals and original joined `cpu_jobs/cpu_rows`. Sum of all expert nt jobs equals jobs; sum of nt×jobs equals tokens; first touches + retouches equals jobs; format×nt sums equal jobs. For every capacity and policy, hits+misses+unsupported+policy-bypassed equals global CPU jobs. For policy0, bypassed is zero; for policy1, eligible hit+miss jobs are exactly supported nt1 jobs. Each format×nt hit+miss bucket equals that policy's eligible global format×nt job bucket; nt1-only nt2..4 buckets are zero. Bucket sums equal aggregate hits/misses and weighted hit-token counts. Fill bytes equal the size-weighted miss buckets; fill minus evicted bytes equals current resident bytes. Per-format resident counts independently reconstruct current resident bytes, which never exceed that budget; misses minus evictions equals the current resident entry count. The unrun mock compares policy1's member/recency/link arrays and all its statistics against a second observer fed only nt1 jobs, despite 12,000 intervening nt2..4 jobs in the first. Overflow-safe updates and sums guard these checks. The integration also compares all four original joined `job_nt` buckets against summed observer format×nt jobs. Main should additionally compare original per-format planned GU rows (640/job) without introducing extra service clocks. Conservation failure is diagnostic failure, not a modification of the product's numerical result.

## Retention and decision

Current consumer: Main's recurrence/working-set question for the initial nt1-only policy, after the current Direct34K GPU model profile closes. Main will adapt its unexecuted v1 build recipe to v2 only after this final proposal boundary. Keep this compact proposal and Main's eventual exact commands, compiler/binary/input pins, individual correctness/overhead/model samples, complete aggregate receipts and decision in Git at the authorized boundary. No raw weight/tensor/event traces are produced. Maximum new emitted receipt is 256 KiB per request; a finite supervised request count/deadline and text budget remain Main's run responsibility. No run or cleanup is launched here.

Next review: Main CPU mock normal/sanitized correctness, then product integration OFF/ON validation and a bounded real decode request. Use policy1 results for an nt1-only cache decision. Policy0 aggregate hits cannot substitute for policy1, and extracting only nt1 rows from policy0 still uses the wrong admission/recency/eviction history. A useful hit rate alone does not pass adoption. Require repeatable component and full-model wall benefit with cold prepare and eviction/lifetime/RAM costs, unchanged output/token/work, fault-free completion and >=8 GiB live RAM reserve before proposing any selective expanded GU cache. If recurrence is absent, retain packed behavior and close this candidate. This proposal never creates model-cache data or production predecode.
