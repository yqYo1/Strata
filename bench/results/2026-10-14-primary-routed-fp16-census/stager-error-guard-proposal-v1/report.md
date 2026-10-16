# Stager ownership error guard proposal v1

2026-10-14, personal Strata `feature/sycl`. Dedicated implementation-agent follow-up, artifact-only. **Unapplied, uncompiled, unrun, unmeasured, unintegrated, unadopted.** Only this proposal directory was written. Product sources, the live screen-v4/native68da, clients, recipes, the original affinity proposal, and its hashes were untouched. No tests/build/GPU/profiler/Git/service/reset/cleanup actions occurred. No other agent was contacted or interrupted. PR6/private Codex logs and the excluded B secret record were not read. Main owns all later application and serial validation after the active benchmark and user PR6 audit boundary.

`unified.patch` changes only root `src/prefill/prefill.cpp` and adds `include/strata/prefill/stage_ownership.hpp`. `identity.json` pins their base/proposed hashes and the patch. `transition-sites.json` records each exact operation and site. `generate.py` fabricates these review artifacts and a concrete, unrun `cpu_fixture.cpp` by extracting the proposed Stager, inline issuer, and release bodies into CPU-only fake-adapter scaffolding. Running the artifact generator performed no product import, patch application, compilation, or fixture execution. `prepare_cpu_fixture.py` is unrun source for Main to regenerate the fixture from the actual applied source in a private directory.

## Confirmed source issue

R534 and Main's `stager-error-ownership-review.ja.md` distinguish normal multiworker ownership from failed adapter results. In `src/sycl/compat.cpp`, `attempt` calls `Runtime::check()` before its body. The runtime retains its first asynchronous exception. Therefore an already sticky error can cause `cudaEventSynchronize` to return an error **without reaching the actual event wait**. `Runtime::wait(event)` doing `wait_and_throw` before checking errors does not bypass that earlier adapter check.

Current `Stager::work` ignores `cudaSetDevice` and `cudaEventSynchronize` results, then may overwrite its host slot and publish `ready=1`. `issued_one` ignores `cudaEventRecord` failure and advances the issue watermark. The inline stream-all issuer, threaded stream-all issuer, and nonstreamed `stage_one` also ignore the primary ring's dependency-wait, H2D-copy, and copied-event results before publishing copy progress. Device-side consumer release events similarly precede `used_of` and consumed/slot acknowledgments without checking results. These are concrete pre-existing source paths. **They do not prove a reproduced race, the origin of an earlier GPU hang, the live source mode, or a present GPU fault.**

The original affinity proposal deliberately leaves these paths unchanged. This follow-up does not widen affinity, alter worker counts, adopt scheduling changes, or repeat R533's inheritance work. R521/R532 also keep prefill staging time distinct from packed-IQ CPU decode work and warn that consumer elapsed time includes wait/submission and overlaps GPU intervals; none is used as causal performance evidence here. R519/R520/R522/R523/R524/R531 cover distinct GDN, benchmark, QSA and GEMM questions and remain separate candidates. The provenance manifest pins all census research reports and both current registries so Main can retain that context without copying their contents.

## Concrete proposal

The helper evaluates each wrapped call once and returns its successful status unchanged. In SYCL builds a non-success result prints operation expression, named site, numeric error, up to 512 bytes of the existing thread-local error detail, source filename/line, and `aborting native`; it explicitly flushes stderr and calls `std::abort`. It does not consume `cudaGetLastError`, allocate a failure queue, alter runtime sticky-error semantics, or replace an error with success. There is no option: if adopted, this is SYCL safety behavior on failed ownership transitions. Healthy calls add only the checked result branch, with no logging. The non-SYCL macro expands to the original call expression; CUDA/HIP implementation, compilation and validation are outside this work and are not claimed.

The 31 guarded source transitions are:

| Scope | Calls | Publication/reclamation protected |
| --- | ---: | --- |
| Stager device identity/worker selection, host-slot event wait, DMA event record | 4 | Worker selection; host overwrite/`ready`; `issued` watermark |
| Inline and threaded issuers, plus threaded device selection | 9 | Dependency before reuse; H2D before host DMA acknowledgment; copied event before `stage_live`, stream counters and issuer acknowledgment |
| Nonstreamed `stage_one`, pinned and staged branches | 5 | Dependency/copy/event before new copy progress and slot mapping |
| Fused, grouped, individual-MMQ, FP16, streamed/nonstreamed consumer leases | 9 | Copied-event wait before read; used-event record before `used_of` and consumed/slot acknowledgments |
| Existing final run drains and existing release drains | 4 | Successful run result; completion before event/buffer reclamation |

The threaded issuer currently uses `core::OnDevice`, which silently tolerates failed selection and, for SYCL device 0, does not select anything after reading the current ordinal. The SYCL branch instead checks `cudaSetDevice(m.device)` before entering that issuer loop. SYCL's adapter only supports ordinal 0 and has no thread-local current-device switch, so this performs the explicit selection/check without adding a queue, wait, or restoration duty. Other builds retain their original scoped `OnDevice` path; the shared OnDevice class and all other callers are untouched.

No host-slot copy or ready store occurs after its failed wait returns in that worker's control flow. No `issued.store` occurs after a failed DMA-event record returns. An H2D failure stops before `issued_one` and any copied-event/global issuer acknowledgment. Failure of the later copied-event record stops before device-copy/issuer publication; the earlier host-DMA watermark may already have advanced after its own successful record, which is the correct separate lease, not a false DMA failure acknowledgment. Failed consumer release records stop before the following slot-map/consumed publication.

The release guards wrap the two drains already present, in their original order, before any existing free/destroy loop. A failed drain is not proof of completion, so this path aborts before the C++ code reclaims the buffers. The final run drains also fail-stop immediately. Other existing checked early-return drains remain unchanged; eventual release is guarded. This is scoped ownership protection, not a complete engine-wide audit of every unchecked API call, kernel failure, allocator failure, peer path, diagnostic timer, PLE path, or library lifecycle.

## Stop policy and lifetimes

Fail-stop was chosen over cooperative cancellation because the existing worker/issuer loops wait on `ready`, `issued`, generation and consumed counters, and `finish` waits for active host workers. A graceful error protocol would need to solve every one of those waits and retain any queued H2D/source-buffer lease whose completion is unknown. A failed wait or record cannot supply the missing DMA drain. This minimal change adds no cancellation state and never claims a failed completion released a slot.

`std::abort` terminates the native process abnormally (normally SIGABRT on this Linux host). It does not unwind Stager/Prefill destructors, fabricate a normal reply, restart the native process, repair its sticky runtime, reset the GPU, or prove the GPU is recovered. The owner/client must record the abnormal exit and apply the existing fault/restart review before a fresh native instance is considered usable. OS/driver process cleanup is not measured here and is not presented as a successful GPU drain or recovery. Other threads can have already completed independent work while the diagnostic is emitted; this is process fail-stop, not an atomic cross-worker rollback.

On success the existing calls, ordering, queues, copy streams, event objects, in-order alias policy and skip behavior remain. No extra copied-wait is inserted for skipped entries; `copy_events` still gates precisely the original waits/records, while Stager's host DMA wait remains unconditional. No event or synchronization is added. Generation/head claims, active-job counts, `ready`, `issued`, modulo host/device ring selection, layer planning, source-pointer/transient `copy_blob` contract, memcpy bytes, dequant/GEMM math, and normal join/destruction order are untouched. Existing `copy_blob` false still calls its existing abort path. This proposal is independent of the original affinity patch and requires explicit composition/rebase review if Main chooses both.

## CPU fixture source, not results

`cpu_fixture.cpp` contains the proposed Stager definition and actual inline-issuer/release bodies extracted without algorithm edits. Only release's function signature is adapted to a fake Impl pointer; the body is preserved. The fixture includes the actual proposed helper. Fake CUDA-shaped functions return deterministic statuses and use CPU memory, with no SYCL headers, runtime libraries or GPU calls. The fake failed event wait deliberately performs no completion action, representing the precise pre-body failure concern.

The fixture source intends to check these ten cases, **none compiled or executed**:

| Case | Expected outcome |
| --- | --- |
| Failed worker host-slot wait | SIGABRT plus site/error diagnostic; slot bytes unchanged, ready 0, no H2D/issue progress |
| Failed worker device selection | SIGABRT before a wait/copy/ready publication |
| Failed Stager DMA event record | SIGABRT; issued watermark still 0 |
| Failed issuer H2D | SIGABRT; no DMA record, device bytes unchanged, global issue progress 0 |
| Failed issuer dependency wait | SIGABRT before H2D or event record |
| Failed copied-event record after successful H2D/DMA record | SIGABRT; host watermark 1 permitted, global device-issue progress 0 |
| Failed release compute drain | SIGABRT before any free |
| Successful worker | Same expected bytes, ready, event record and original finish/destructor watermark behavior |
| Successful inline issuer | Same expected copied bytes, event records and issue/counter progress |
| Successful release | Both original drains precede the existing free |

Each case owns one forked child and mapped probe storage, with a five-second child deadline. Expected SIGABRT children disable their own core dumps. Shared probes let the parent inspect ready/issued, slot bytes and fake free/call counters after child exit; the parent does not dereference child-private vector storage or run its destructor. Normal children exercise the actual Stager destructor after releasing the fixture-owned mapped ready array from its unique_ptr. The test harness does not alter product counter algorithms. These are CPU fake-status checks, not successful DMA/event-runtime, GPU math, long-context, driver-health, multiworker throughput, or generated-output evidence. The fixture exercises one worker and inline issuing; other guarded source sites still need review and healthy integration validation.

After Main's active benchmark and user PR6 audit boundary, Main can review/rebase and apply, then regenerate `cpu_fixture.cpp` from the applied source using `prepare_cpu_fixture.py --root <root> --output <private>/cpu_fixture.cpp`. Compile with `c++ -std=c++17 -pthread -I<root>/include <private>/cpu_fixture.cpp -o <private>/cpu_fixture` and execute under Main's serial test ownership. The supplied generator intentionally rejects a changed Stager init signature, so composition with the affinity proposal needs an explicit fixture adaptation rather than silently testing a stale body. Record applied source/helper/fixture/binary/compiler hashes and every case's exact status/diagnostic.

Then validate healthy SYCL runs with identical source/binary/runtime/driver/environment/request identities, output/logit fingerprints, actual context/chunk/ring/issuer/same-queue modes, individual sample order and normal native/owner exits. Exercise existing inline/threaded and stream-all/nonstreamed paths as applicable without injecting faults into a real GPU. Preserve prefill versus cold/restored decode distinction and fresh per-run fault-journal bounds. Assess overhead with quiet matched runs; checked statuses alone establish no speed or numerical result.

## Retention and decision

Main is the sole consumer and owner of this small proposal, its source identities and the unrun fixture. The next review point is the post-benchmark/PR6 decision and serial validation completion. The report, patch, transformation source, site mapping, fixture source/template/extractor, and provenance should remain reachable in committed evidence at the work boundary. No traces, model tensors, profiler timelines, driver captures, or duplicate raw histories were generated; no file was deleted. The original affinity proposal remains byte-identical and separately identifiable.

The decision is **proposal ready for review, not adopted**. No successful build, fake-status test, runtime safety test, fault-free run, output equality, GPU recovery, previous-hang root cause, or performance result is claimed.
