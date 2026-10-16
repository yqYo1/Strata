# PR6 safety corrections v2: source-only delivery

2026-10-16 JST. Worker `task_1ceb5bd1e1c2`, dispatch `ctx_af2da9c517b4`.
**Status: `SOURCE_ONLY_UNSAFE_TO_RUN`.** R1–R7 have separately reviewable source
corrections. No configure, compile, build, fixture/test execution, native/device
query, GPU work, profiler, process inspection, service operation or permission
change was performed. This is neither a safety certification nor a new speed
result. Main owns every execution gate and integration decision.

The starting HEAD was `eb700276c8cd100eea5b70fe3767f1e02d0ae04c`, with a clean
working tree on the existing `verify/pr6-prefill-restore` child. The original
faithful anchor `efba8d47817811d493e28f396ebd0e14e161ca75` and seven-fix checkpoint
`2170747a3d79e0fd40b9e86736dc1785435e630f` remain original reachable ancestors.
The companion JSON pins the final source HEAD, every changed validation file,
all fix commits and the complete binary-capable patch against that starting
HEAD. The patch excludes this delivery report itself. Original review files,
`report.ja.md`, `manifest.json`, original timing evidence and original fixed-source
pins were not rewritten. The corrected `historical-closure.json` explicitly
distinguishes the incomplete faithful anchor from the complete historical source
inventory; the old inventory remains reachable at the checkpoint.

## R1: complete the actual resolver closure

Trigger: historical CMake requests `src/core/peer_experts.cpp` but the earlier
snapshot omitted both resolver candidates. Commit `9d38d9f1` restores the exact
original SYCL stub from `d0c5dda0dc2e8d82458c4b0ff31c4cef44c40b32`, blob
`63656fd6866d15670a940a9e25e7750f3b071ebc`, SHA256
`5708f78a7571bb8e6ed883f5d582298fe238145d2a1d224939becc0332cf0360`.
It still explicitly refuses the unsupported peer tier. No CUDA implementation
was substituted and the historical inner CMake was not edited.

The full inventory is **198 project files, 4,007,615 original bytes, 84 TUs**:
groups 3 spec, 4 CPU, 4 native CPU, 6 core, 47 kernels, 13 engine, 3 prefill,
1 program, plus the three explicit port sources. Source inspection independently
parsed the inner CMake requests and resolved every candidate; those selections
equal the corrected inventory. Every original inventory blob/SHA256/size was
checked against the original Git objects. That is a source identity observation,
not an executed configure. Commit `7bf1783f` additionally makes the outer wrapper
compare the actual configured target SOURCES against the complete selection,
including the optional native CPU group when enabled. This check is authored,
not executed. The faithful anchor still has its original 197 files.

## R2: event retirement, buffer ownership and finite failure paths

Trigger: GPU ordinary fills and CPU volatile reads of host-USM completion words
had no atomic synchronization contract; failed signals and host gates could
spin indefinitely. Commit `ca1aebce` removes **both** Stager and PLE completion
counter allocations, fills and volatile polling. Follow-ups `dbd24092`,
`da12a24c` and `1804cd68` close issuer/destructor deadlines, publish actual DMA
events and fail-stop on bad staging reads.

Stager retains each actual `memcpy` event in its ring slot, then release-publishes
`issued`. A wrapped worker acquire-observes that publication before reading the
event, retires it before touching the source buffer, and release-publishes ready
only after its host copy finishes. The issuer acquire-observes ready before
overwriting that slot's event. Finish first cancels claims and releases host
issue gates, waits for active workers to leave, then retires all recorded events.
A late old-generation claim cannot enter a new generation because the packed
head checks the generation; the 16-bit job count and generation wrap are now
admitted explicitly. Cancellation does not authorize a buffer write before its
old DMA completes. Partial-vector initialization examines only constructed slots.

There is no separate completion-signal submission anymore: the returned DMA
event is the completion evidence. Transfer submission, event publication or
post-submit recording uncertainty stops the native process with owners alive.
PLE similarly saves its actual upload event and retires the other buffer's
previous event before launching its refill task. The outer request guard retires
accepted work on success, callback refusal, cancellation and exception; failure
clears readiness and requires reset/RESTORE. Layer-major retains its earlier
inner Restore guard, before local residual/cache owners die. Failed source reads
can no longer publish fabricated zero expert data as successful input.

`retirement.hpp` and `sycl_retirement.hpp` are new independent project helpers.
The event observer polls status with 50-microsecond sleeps, accepts only complete,
then invokes `wait_and_throw`; queue retirement submits the already-used oneAPI
barrier extension and consumes async errors. Host gates, event/queue retirement,
transfer submission, issuer joins and stager destruction have 60-second process
deadlines. Initialization has 600 seconds; a whole prefill request, including
cleanup of local futures, has 7,200 seconds. A watchdog bounds a provider call
which itself blocks. Deadline or uncertain completion exits **86**, without
stdio locks, C++ unwinding, source frees or residency publication. A deadline
exit is an incomplete failure, never a correctness result. Standard C++ thread
scheduling/OS process execution remain assumptions; this cannot cure a stopped
OS or driver. Existing waits outside the corrected prefill/MTP retirement
ownership path still require Main's external supervised process deadline.

The current-project separate-copy/compute timeout report was read in full; it is
a different source/runtime route with unresolved cause, not causal evidence
against PR6. The corrected prefill explicitly assigns **`copy = cs`**, admitting
one selected in-order queue in the allocation device/context. Layer-major already
loads each layer serially before its consumers. Chunk-major's existing event and
slot dependencies remain, with copies now ordered on that queue too. Main must
qualify both inline and threaded issuer cases. No current native16 algorithm,
weight bank, precision kernel or modern scheduler was imported.

Arithmetic, weights, layer/chunk sequence and FP32 residual storage are unchanged.
The queue choice, completion events, checks, finite deadlines and final retirement
barriers are **intentional synchronization deltas**. They add host watchdog and
event overhead. Old scheduling/performance claims do not qualify this route.

## R3: the selected DPCT queues must fail-stop on async error

Trigger: historical DPCT logs and consumes asynchronous errors. A local handler
on a new private queue would leave the many default-queue users unchecked.
Commit `e11f0bb2` implements an owned fail-stop contract and changes the existing
project event/queue waits to checking waits at their original wait points;
`dbd24092` adds the generated dependency's modification notice.

`checked_dpct.cmake` operates only after all 49 external historical vendor hashes
pass. It generates **one build-private modified dependency header** from the
exact original `dpct/device.hpp`. It replaces only that original handler span and
adds registration at its unique owning queue-creation return. It retains the
original Intel notice/guard, adds a modification notice and records the generated
SHA256. The outer wrapper force-includes it for `strata_includes` consumers before
vendor relative includes can load the original body. No shared vendor/runtime
installation is written, and no vendor header is copied into Git. The external
ggml targets do not receive this project's forced include.

The actual historical `init_queues` installs this handler on default in-order and
out-of-order queues. Its `create_queue(true)` users also install it: generator
main/stage/adaptation streams, MTP, verifier copy/compute, session captures,
remote experts, pinned bandwidth and fused-GR checks. New prefill aliases the
checked compute queue. Kernel launchers use `q_of`; oneMKL's DPCT descriptor is
bound through `q_of`, and the underlying DPCT descriptor/default helpers therefore
see the patched defaults too. JSON records source locations/counts. Queues with
this contract are registered by owning shared_ptr and tracked by weak_ptr;
`q_of` rejects arbitrary unregistered queues, including stale reused addresses.
Its compile guard refuses a direct inner build that bypasses the wrapper.

Any delivered nonempty async error list exits **87**, with no unwind or automatic
owner destruction. Checked waits precede residency acceptance, PLE state acceptance,
callbacks/results and host readback; init readiness follows checked retirement.
MTP's A/C/F record and borrowed-residual guards now use bounded checked retirement.
Synchronous submission uncertainty in the transfer helpers exits 86. Existing
ordinary synchronous error-code paths elsewhere remain their original contracts;
this is not a whole-engine synchronous-error or allocator audit.

**UNRESOLVED qualification:** no preprocessing/build has proved the forced header's
effective inclusion/ODR/toolchain behavior, and no provider-error injection has
proved delivery through the selected installed runtime/library stack. Main must
verify both before any model diagnostic. A working source contract is not an
observed async failure result. Hidden library queues/fallback arithmetic and
general graph destruction remain separate audits; no all-GPU arithmetic claim
is made from a queue parameter.

## R4: owned queue versus borrowed default

Commit `d2c6165b` initializes `cs` and `copy` to null and records copy ownership.
No-init destruction or an early init failure cannot destroy DPCT's default queue.
Release destroys only an owned queue and clears its ownership/pointer. With R2,
`copy` aliases the borrowed `cs` and is never destroyed by Prefill. Normal-input
math is unchanged; no-init construction also avoids initializing a device just
to fill these pointers. Peer stub restoration does not claim a working peer tier.

## R5: publish readiness after complete initialization

Commit `d2358ec4`, followed by `dbd24092`, adds explicit readiness and a one-attempt
init contract. Readiness is cleared before init, reset/release and relayout,
and published only after all allocations, GEMM binding, carving and checked
retirement succeed. Reinitialization requires reset, as the existing retry callers
already perform. Run and draft reject incomplete init; failed relayout cannot
reuse a partly carved workspace. Failure after submitted request work clears
readiness; admission refusal before submitted work leaves an existing good
workspace intact.

Before any allocation or queue selection, init admits positive chunk <=262144
and session capacity (preserving the historical two reserve cells), fixed artifact
geometry, bounded QSA ordinals/allocation counts, capacities and page/slot counts.
This prevents late-failure fields from impersonating a complete initialization.
No normal arithmetic or storage dtype changes. Raw externally supplied page IDs,
all allocator paths, concurrent use of one Prefill object and session rollback
are not certified by this flag; Main must test the actual late-failure paths.

## R6: actual embedding vocabulary before gathers

Commit `6a47d65c`, followed by `0f21f150`, `da12a24c`, `c94826b2` and `2fec70bf`,
exposes the successfully loaded native table's actual vocabulary and validates
nonnegative, in-range, int32-representable IDs. Prefill validates its complete
host span before submitted work, including native batched input; explicit image
row overrides retain their existing fallback and int32 admission. Draft-KV,
MTP prefill/draft/first-draft and verifier host inputs check actual vocabulary
before their uploads/graphs. Direct one-row and `embed_row` calls reject one-past
and int32-overflow values. Native embedding load also rejects invalid dimensions
and handles nonthrowing allocation nulls through its existing fallback/refusal.

The public direct device gather requires the producers and gather to share an
in-order queue and the caller to keep IDs immutable through completion. It
downloads/checks device IDs before submitting an embedding kernel; the required
checked D2H transfer itself is a synchronization delta. If its submission or
retirement is uncertain, the host admission vector stays alive through exit 86.
That API must not be used inside graph capture.

Existing graph recording uses a private admitted method, accessible only to
Prefill, MTP and Verifier. Source proof: Prefill uploads checked host IDs; MTP's
two prefill graph forms upload checked next-token records; verifier copies checked
window IDs. MTP's round/step `mtp_select` reads admitted input or head-selected IDs.
Bind/init require the actual native head vocabulary to fit the embedding table;
draft subset file IDs and subset count are admitted before upload/direct gather,
so subset mapping also stays in range. Sampling/argmax ranges and graph node/token
flow are unchanged. Main must verify captured/replayed token producers and table
lifetime/rebind; public raw low-level kernel APIs still require caller ownership
and valid dimensions. No arbitrary concurrent mutation safety is claimed.

`fixtures/token_admission.cpp` is new independent host source covering zero,
last valid, one-past, negative and int32/int64 bounds, invalid vocabulary, null
spans and the final physical262144 token. It was not compiled or run. Valid
embedding math, quantized blocks and internal graph launch sequence are unchanged.

## R7: transactional doorbell initialization

Commit `b5991013` replaces the startup allocator with independent project helper
`allocated_fields.hpp`: every allocation must be nonnull before memset, and all
five fields must finish before host/device aliases and geometry are published.
Null or throwing allocation releases each earlier field exactly once; cleanup
failure exits **88**. Size arithmetic is admitted before allocation. An existing
doorbell is refused without replacing pointers that may be in an active graph.
`fixtures/allocated_fields.cpp` injects null and exception at each of five fields,
checks cleanup and checks actual zeroed memory on success; it is unexecuted.
Healthy allocation order, aliases, bytes and inference math are unchanged. This
fix addresses initialization, not the separate doorbell signaling protocol.

## Provenance and Main's remaining sequence

New source/helpers/fixtures were written independently under this repository's
MIT LICENSE against the pinned child. No current 16-bit code, XeStrata code,
helpers, tests or archived proposals were read or adopted. The sole restored
implementation addition is the exact original fork peer stub. The DPCT change
is a surgical transformation of the already-required historical Intel dependency,
not an external engine implementation reference. Its original SPDX is
Apache-2.0 WITH LLVM-exception; the primary [LLVM license](https://llvm.org/LICENSE.txt)
was checked and the generated dependency retains its notice. New Git project
files do not receive a copied vendor license header. JSON pins exact local
source objects/hashes and input report hashes. No new upstream check was made.

Only primary API documentation informed the new event/error protocol:
[SYCL event](https://github.khronos.org/SYCL_Reference/iface/event.html),
[async handler](https://github.khronos.org/SYCL_Reference/iface/async_handler.html),
[USM allocation](https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html),
and Intel's [queue barrier extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc).
These were used for contracts, not copied implementation examples. They do not
identify or qualify the installed runtime. Main's sequence is:

1. Review the complete pinned patch, source identities and original review;
   inspect the build-private DPCT transform and generated header/license. Verify
   the pinned offline llama.cpp HEAD/diff, all 49 dependency hashes and a coherent
   compiler/libsycl/UR/Level Zero/oneMKL stack. Preprocess representative default,
   private-queue, kernel and GEMM TUs to prove one effective checked handler,
   ownership registration, q_of and profiling definitions. Inspect configured
   target closure (84 with native CPU support) and effective compile commands.
2. Main alone configures/builds/runs the four standalone host fixtures: existing
   draft/range plus token/allocated-fields. Preserve original tests and numerical
   gates. Add/inject no-init destruction, every pre/post-queue and late-carve
   failure, relayout failure and reinit-after-reset; verify default queue ownership.
   Check finite timeout/exception/cleanup exits 86/87/88 in owned child processes.
3. Before model work, qualify actual event retirement with ring2, >2 wraps, partial
   final ring, cross-generation reuse, cancellation while waiting/reading/issuing,
   accepted DMA then publication failure, synchronous partial submission, provider
   async failure, never-complete status and blocked provider API. Verify pinned
   source contents stay immutable until completion and no owner destructor runs
   on uncertain retirement. Inject errors on both actual DPCT default and private
   queues; independently check normal completion does not report a failure.
4. Only after source/dependency/failure review, Main may schedule finite model-free
   copy/graph diagnostics and tiny 2/257-token runs. Fix all environment values in
   the native child's environment before exec. For a small traced run use the
   existing qualified UR tracing/parameter-validation recipe and finite log budget.
   Cover both issuer modes, PLE double-buffer reuse and valid/one-past IDs; inspect
   graph prefill, verifier and MTP direct/round/step/subset producers separately.
5. Progress to 4096 and 8087 odd-tail/continuation cases, FIRST0/256,
   chunk-major/compact/layer-major, host/mixed/GPU residual placement and both MTP
   fallbacks. Compare complete logits, residuals, KV/scales, recurrence/conv,
   indexer state and PLE history against independently checked FP32/state evidence.
   Preserve numerical/quality gates; natural-language equality is not acceptance.
6. Run actual physical262144 page/chunk boundaries and full export/exit/fresh
   RESTORE/continuation, including rewind/branch and failed/cancelled-session
   discard. Allocation-only historical capacity evidence is not a full-context
   pass. Resident historical KV and FP32 residual memory admission remain required.
   Cold/warm/restored decode are separate gates; general graph cleanup and legacy
   mapped doorbell communication remain **UNRESOLVED whole-route qualifications**.
7. Only after the prior gates, measure original-condition paired 32769-file-token /
   32768-batched-token workload and matched official >=32K thinking workload in
   separate clean owned processes. Preserve all individual samples, failures,
   ordering, source/binary/env/input hashes, startup versus prefill clocks and PLE
   warmth. Neither historical 645.85 nor current unmatched 403 is a measurement
   of this changed synchronization route. **No new 600 tok/s claim is made.**

Every Main run needs owned PID/start/boot, finite process deadline, exit/cleanup
and fresh journal/fault evidence. Default raw text budget is 64 MiB; retain only
representative unresolved diagnostics after compact evidence is extracted.
This worker created no raw measurement traces, tensors, builds or cleanup
manifest because it ran no experiment and deleted no artifact. Main alone
reviews/tests/integrates/pushes and retires this child; no branch switch, push,
PR, merge, nested worker or shared installation write occurred here.
