# PR6 safety corrections v3: D1–D5 source delivery

**Status: SOURCE_ONLY_UNSAFE_TO_RUN.** Worker task_08d5d970525c, dispatch
ctx_fe4210560094, 2026-10-16 JST. Source changes are confined to
`validation/pr6-restore`; this report, its JSON and patch are the only new result
artifacts. Main owns independent review, every execution, commit/push and
integration. No configure, preprocessing, compiler, syntax check, build, fixture,
test, runtime/device/process query, GPU, profiler, service/package/permissions
operation or cleanup was performed. Read-only Git/source/hash inspection and
Python file editing/artifact generation were used; no Git mutation occurred.
The shell name `python` was absent; the editing heredoc was then executed with
`python3`. This was an editor invocation, not a fixture or project test.

The immutable base is `185fc2d2642071f8fff43a7b426f936f3a974a09` on the existing
user-requested `verify/pr6-prefill-restore` branch. Its only differences from
source checkpoint `2776c37b095e2b84a28f575ad43d888c1c42d254` are the independent
v2 review MD/JSON. The initial working tree was clean. The companion patch is
the complete binary-capable source/fixture/README diff against this exact base,
including all three new source files; only the three v3 delivery artifacts are
excluded to avoid self-reference. JSON records each changed file's base and final
hash/blob identity, all original report checks, unchanged-source identity checks
and the complete patch hash. It is an uncommitted snapshot, not a new commit.

## D1: completion boolean and accepted work

The shared host-callable `core/validation_control.hpp::ring_progress` stores
`ext_oneapi_empty()`'s **bool** separately from exceptions. It consumes async
errors before accepting a ring observation, and distinguishes pending, reached,
complete-without-ring and query error. The two inherited `DPCT_CHECK_ERROR`
boolean-discard sites in session token replay and verifier polling now use it.
Pending work continues polling; a genuinely completed queue with no required
ring is a failure, never success. A provider/query exception fail-stops with 86;
a nonempty async list delivered through the existing checked DPCT handler still
exits 87. No handler, registry, forced-header generation or ODR gate was changed.

Both held graph routes arm the existing independently authored DraftUploadGuard
before graph submission. An early return or callback exception retires actual
accepted queue work with the checked barrier/event path, or exits 86 with caller
owners alive; it does not treat an error receipt as completion. A completed queue
with a missing ring is retired before returning false. Pending ring abandonment
at the existing 20-second bound now exits 86 directly instead of unwinding or
using the unproved volatile release handshake. A 60-second deadline begins
before launch, and final acceptance requires checked retirement. Verifier includes
its copy queue in retirement. These changes do not authorize enabling the route:
D3 still excludes it, and its remaining legacy capture/debug/library behavior
has not received a whole-route lifetime certification.

`pr6_validation_control` uses this actual helper for ordinary pending/completed
queries, completed-without-ring, a reached ring with later queue work pending,
query/async-consumer exceptions, and 10,000 logical pending observations followed
by valid completion. This is finite host source, not a measured slow GPU test.
It cannot prove actual status semantics, provider progress or async delivery.

## D2: position, pointer, actual capacity and lookahead admission

The real shared `position_span`, `prefill_admission` and `draft_admission`
helpers check negative values and use subtraction against the minimum of
physical 262144 and the supplied actual capacities. Admission never forms an
unchecked first+count or p+a. Draft admission checks the T catch-up rows and the
separate maximum emitted chain from accepted row a, reserving one verifying
position for its final produced token. It mirrors the original mandatory first
draft and min(max_t-1,max_drafts) later-chain limit; max_t=1 is now refused for
draft calls, which always write draft[0], while one-row prefill is retained.
The bound is conservative when min_p could terminate a chain early.

`session_cells` admits the SessionState QSA ordinal/allocation span against the
geometry's real QSA array count before indexing, and takes the minimum capacity
of every allocated state, including a split stage's primary. It presumes a
caller-owned initialized SessionState and valid backing allocations; integer
admission cannot authenticate arbitrary pointers or corrupted objects.

Both MTP prefill branches pass through the same admission before OnDevice,
record allocation, upload, capture, addition or narrowing. Nonempty input
requires residual and token pointers, actual session/QSA and draft-state spans,
and the existing actual-vocabulary gate. A zero-length request is an admitted
no-op only at a valid nonnegative endpoint (including exactly the capacity),
with an initialized/bound object and valid vocabulary; null residual/token
pointers are then allowed and no device selection/capture/upload follows.
`draft` and `draft_first` require residual/window and output pointers and check
catch-up plus chain lookahead before graph capture or first-residual staging.
The private capture helpers and warm reject invalid/uninitialized row geometry;
record_forward checks row spans before negation/indexing. kv_restore validates
its endpoint before its rounded page calculation. No private prefill graph route
can bypass the public position admission at submission.

The alternate `Prefill::draft_kv` consumer previously bypassed MTP::prefill. It
now admits required residual/token pointers and actual session/QSA/draft-state
capacity before its own additions and uploads, and shares the empty no-op policy.
It retains its prior vocabulary/physical span checks and upload leases.
Verifier::run rejects malformed pointers, T and positions before pos0+t,
capture or device selection; window_logprobs also gates its public position span.
Verifier execution remains excluded independently by D3.

The host fixture invokes these production helpers for INT64_MIN/MAX, negative
positions/counts, huge spans, null pointers, exact physical endpoint/one-past,
empty endpoint policy, differing smaller session/state limits, catch-up versus
lookahead near the boundary, malformed T/a/max_t and required output pointers.
The fixture does not read a large tensor: pointer extents, device residency,
actual row kernels and full physical256K execution remain Main gates.

## D3: explicit exclusion, with original source retained

`concurrent_doorbell_admitted()` is an inline constexpr false with **no environment
or build-option override**. The legacy device atomic/CPU volatile/fence code is
preserved for comparison, clearly marked held. No matching host atomic protocol,
capability/alignment/scope or payload-publication proof has been invented.

The allocator refuses before allocating any mapped field. Session capture with
a nonnull doorbell refuses before allocating graph arrays or recording; moe_route
with a doorbell refuses before any projection/kernel work. session_loop,
session_capture_token and session_run_token refuse at entry. Every verifier mode
is refused at init before device selection, registration, allocation or recording;
private record/capture/boundary routes, run/commit/warm/result access and legacy
release/trace readers are held too. This includes eager, host-boundary,
no-host/all-resident, split and asynchronous commit modes. The host-boundary
source still contains doorbell publication/other paths without a complete safe
fallback proof; it is not silently promoted to a safe original substitute.
Verifier queue members start null, so construction/refused init/destruction
cannot initialize or destroy borrowed DPCT default queues.

The CLI refuses unsupported decode/verifier/serve modes immediately after option
parsing, before its device queries, allocation and graph recording. The retained
original `--no-capture --no-pool` path calls session_token without a doorbell.
Its existing contract explicitly lacks a CPU expert pool hook: it is only a
GPU-floor decode diagnostic, not full-model decode. Its main SessionState keeps
`db=nullptr` and no doorbell is allocated. No automatic mode switch conceals this
change. Direct original no-doorbell session/token/replay source remains where
its caller-provided-parts contract applies; this does not certify arbitrary
external graph handles or raw low-level kernel use.

**Fallback impact:** ordinary Prefill::run and its CPU expert/stager/PLE path,
FP32 residual math, original weights/layout and QSA/GDN/MoE kernels remain
source-available. No ordinary Prefill run dispatch was disabled. Main needs a
separately reviewed prefill API harness for full native long-input validation;
this worker did not create or qualify that harness. The restored native-pack
CLI requires verifier/spec mode and is therefore unavailable under the exclusion.
There is no safe original full-model decode fallback established by this work,
no full historical600 reproduction and no all-engine qualification. Removing
D3 later requires a separately scoped proof and implementation, not flipping
this constant. The fixture's static_assert prevents accidental host-helper
build acceptance after such a flip without revisiting its expectations.

## D4: one attempt before embedding metadata mutation

NativeEmbed owns the real OneAttemptLoad gate and calls begin as the first load
statement, before dimension validation, model parsing and any row/bytes/pointer
metadata mutation. Every first failure consumes the attempt, including invalid
dimensions, architecture/tensor/payload refusal, allocation/upload failure and
caught exceptions. Every later call returns false without touching prior table
metadata or ownership. Thus a successful IQ table followed by a supported
truncated BF16 reload cannot acquire a new stride. The prior good table remains
consistent; initial failures expose no loaded vocabulary. Retrying requires a
fresh owner; cleanup ownership and gather arithmetic are unchanged.

The host fixture uses the actual gate for an initial failed attempt, a successful
attempt followed by repeated refusals and a fresh retry owner. It proves only
the source/helper contract, not a parser/model/allocation/DMA result. No fake
model fixture was created. Concurrent use/mutation is outside this one-attempt
serialized-object contract, as before.

## D5: the same bounded helper on both issuer joins

`prefill/retirement.hpp::bounded_join` starts the existing 60-second watchdog
before joining a joinable worker and exits 86 on join/deadline uncertainty.
Both explicit normal issuer join and IssuerJoin's unwind/destructor join call
it; Stager worker joins call it under the existing total destructor deadline.
An already joined/absent worker is a no-op. Issuer stop and all borrowed state
outlive its guard; the guard is destroyed before StagerDone and source owners.
No owner mutex is held at either issuer call. Stager releases its mu lock and
notifies quit before joining, preserving worker access to that mutex.

RetirementDeadline's own notified control-thread join remains its existing
condition-variable shutdown, not a provider/issuer join or a recursive watchdog.
This change makes no blanket 60-second claim for unrelated engine CPU pools,
other futures or all public runtime operations. The enclosing prefill request
still has 7200 seconds; initialization retains 600. OS scheduling and external
process supervision remain assumptions. No actual hanging provider was run.
The ordinary host fixture calls the production bounded_join for absent, normal,
already-joined and exception-unwind cases, with live borrowed result/mutex/stop
storage and an unlock-before-join handoff; it cannot qualify provider deadlock.

## Preservation, provenance, timing and Main gates

All 13 existing result files, including original/v1/v2 reviews, v2 patch,
historical evidence/closure and both fixed/current pin records, are compared
byte-for-byte to the immutable base and remain unchanged. Prior v2 delivered
source pins match the base; their new intentional corrections are separately
pinned. The faithful197-file anchor and completed original198-file/84-TU inventory
remain distinct and unedited. The missing peer stub, original/vendor inventory,
outer/private DPCT generation and license/ODR/registration gates are preserved.
All kernel files and every unmodified validation file match the base; only the
listed source/control/admission/docs/fixtures differ. Root/current engine files
are unchanged. No BF16 storage/scheduler/alternative math or bench implementation
was imported; no XeStrata implementation/helper/test/archive was read or used.

New helpers and fixtures are independently authored under the repository MIT
LICENSE against the pinned base. Existing upload-guard reuse retains the v2
independent own-fork provenance; no external implementation/template/header was
copied. The retained build-private Intel dependency transformation and notices
were not edited. No installed dependency was changed, no new upstream check was
made, no nested agent was dispatched and no source was committed or pushed.

Timing deltas are recorded separately from historical arithmetic: new checks and
empty/refused paths, the bounded normal issuer join/watchdog, held-route boolean
queries/error consumption, pre-launch deadlines/accepted-work leases/final
retirement, removal of the discarded initial flush queries, and hard exclusion
of mapped handshakes. V2's same in-order copy=compute queue and event/lease/order
contracts remain. All valid-input weights, arithmetic kernels, FP32 residual
storage, layer/chunk order and upload source leases are preserved in source;
numerical equivalence has not been executed. Historical 645.85/600 claims remain
about their historical binary/schedule only. This delivery contains no timing
samples and cannot promote any historical performance result.

Main's required next gates are:

1. Independently review the complete patch and source admission/exclusion,
   especially the CLI fallback impact and session/QSA/draft capacities. Preserve
   the SOURCE_ONLY_UNSAFE_TO_RUN hold. Main already ran four older ordinary
   helpers with PASS_HOST_HELPERS_ONLY (`A/own-pr6-host-fixtures-v1.json` per task);
   that receipt is neither rerun nor requalified here. Build/run the two new
   ordinary C++ fixture targets separately under finite host supervision; do not
   equate them with actual DMA/provider/graph tests. New CTest timeouts are 15s.
2. Pin the offline llama/vendor source and coherent compiler/libsycl/UR/Level
   Zero/oneMKL stack. Audit configure probes, actual generated private DPCT hash,
   source closure, effective compile commands and representative forced include/
   ODR/ABI/default/private/kernel/GEMM preprocessing before accepting a build.
3. Before any native execution, Main must separately authorize and supervise
   finite model-free provider/event/queue/owner/capture/cancel/cleanup gates:
   partial accepted submission/publication failure, pending/complete status,
   async87, retirement/deadline86, allocation-cleanup88, ring wraps/tails, source
   immutability, no-init/late-init/relayout readiness and borrowed-queue lifetime.
   D3 stays excluded in every model-free or model test. Check generated/compiled
   rejection, including environment-selected verifier variants, before claiming
   that an entrypoint is unavailable in the actual executable.
4. Qualify a separate no-doorbell prefill harness, then tiny/odd-tail/continuation
   and both MTP prefill branches plus draft/subset/round/step admission and
   numerical/state/quality gates. Compare logits, residuals, KV/scales,
   recurrence/conv/indexer/PLE state. Full physical262144, export/exit/fresh
   RESTORE/continuation/rewind/branch and separate cold/warm/restored decode remain
   unqualified; the excluded full decode route needs separate later repair.
5. Only after the relevant correctness/fault gates, take clean matched individual
   prefill samples. Use immutable child environment before exec, owned PID/start/
   boot, finite process and 64MiB log budgets, exit/cleanup/fresh fault receipts.
   Timeout, missing receipt and budget stop remain incomplete failures.

Main owns artifact retention and the next review point. No raw run logs, tensors,
binaries, profiler captures or cleanup/deletion manifest were created because
no experiment or deletion occurred. The JSON and source patch are compact source
evidence, not a safety/build/runtime/600-performance pass.
