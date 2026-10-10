# Opt-in SG32 GDN quad pipeline

Source-only implementation: no configure, build, import, syntax check, runtime/device query, GPU/model test, profiler or disassembly was performed by the implementation agent. Root owns all review and serialized execution. This candidate is separate from the CPU WY equation oracle and does not change GDN's token recurrence.

`STRATA_GDN_QUAD=1` opts into a separately named `strata::prefill::GdnRecQuadPipelineSG32` kernel. Unset or literal0 keeps the old launch path; other strings are rejected. The setting is frozen by the existing process-static selector pattern. Start a fresh process for each profile. The old default kernel bodies and launches, keyhead kernel, output norm and existing parity source are unchanged. No new GRF, fast-math, AOT, global FP, graph or runtime configuration is installed.

The normal selector considers quad only for variant0, pipeline enabled, REC_HEADS absent, KEYHEAD0/unset and KEYHEAD_TUNED0/unset. Existing explicit serial/nonpipeline/keyhead/tuned/nonzero variant choices retain their old paths. This includes the existing HEAD-selected nonzero variant. If quad admission is denied, the normal caller executes its old path once. If submission is attempted, errors propagate and the legacy path is never retried against possibly modified state. A quad-specific empty T0 call returns handled/Empty before queue lookup and submits neither recurrence nor output norm; negativeT throws. Disabled/default or conflicting-selector T0 retains the existing path and is not covered by the new no-op contract.

The kernel is a remap of the current pipelined column recurrence. A128-item workgroup has four required32-lane subgroups. Actual subgroup IDs give `rg=lane/8`, `c=lane%8`, `col=cb*32+subgroup_id*8+c`; no assumption is made about how ND local coordinates map into subgroup membership. Each lane owns the same32 key-row values for one value column. Each of the48 heads uses `head%16` Q/K. Q/K staging uses the independent `item.get_local_linear_id()`0..127, providing exactly one write per shared Q/K row. The first workgroup barrier protects prior reads before overwrite; the second publishes all current Q/K writes. Next-token loads remain software-pipelined after publication. Only the three shared-red barriers and redSLM array disappear.

All lanes convergently gather KV partials from source lanes c,c+8,c+16,c+24 in that order, then combine `((p0+p1)+p2)+p3`. The same gather/order is used for output partials. Native exp, each32-step FMA projection loop, `(v-g*kv)*beta`, stateFMA with separately computed k*delta, each32-step FMA output loop and rsqrt(128) multiplication match the old source operations. Only rg0 writes each output column; all state elements have one owner. The candidate lambda directly carries `[[sycl::reqd_sub_group_size(32)]]`. The old recurrence lambda has no such attribute and remains unchanged.

The dedicated `gdn_recurrence_quad_variant` checks admission on the same queue/context/device before any mutable submission: in-order queue; deviceSG32; device/max-dimension support for local(1,4,32); exact type-level kernelID; executable bundle containing that ID for that device; compiled subgroup exactly32; kernel maximum workgroup at least128. The actual submission uses `handler.use_kernel_bundle` with that same executable bundle. No string search or unrelated kernel capability is substituted. Private/spill values are queried opportunistically and reported as known/unknown, never a performance threshold. A capability-query exception may return false before submission. The report marks Submitted at the attempt boundary, not at completion; submission/norm/check/async errors are not caught for fallback. Resource information is not a register count or speed prediction.

The public header adds a bounded GdnQuadReport and explicit candidate/reference APIs. `diagnostic_deny=true` is only a pre-submit test seam: it returns false before queue lookup, not a simulated hardware support test. The reference API invokes the unchanged legacy pipeline body and launch geometry without a new subgroup attribute, then the unchanged norm helper. It bypasses all environment selectors, so the test cannot accidentally compare the candidate to itself. This separately named reference image is not a claim that compiler-generated binary instructions equal the original dispatcher image; root must check the actual differential and default path separately.

New correctness target (OFF by default):

```
cmake -S "$WQUAD/sycl" -B "$QUAD_BUILD" \
  <root's frozen production compiler/ggml/AOT options> \
  -DSTRATA_GDN_QUAD_PARITY=ON
cmake --build "$QUAD_BUILD" --target gdn_quad_parity --parallel 6
python3 "$WQUAD/sycl/tools/debug-run.py" "$QUAD_BUILD/gdn_quad_parity"
```

The angle-bracket line is a placeholder for root's already frozen build options, not shell syntax to execute. Keep existing precise FP/AOT settings, `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1`, `EnableDirectSubmission=0` and the admitted device selector/runtime pins. CTest registers only the short new discriminator with240s timeout and QUAD0/KEYHEAD0/TUNED0/PIPELINE1/REC_HEADS unset. The test directly calls the dedicated candidate, so QUAD0 does not disable it. Existing parity/test definitions are untouched; root must explicitly keep QUAD0 when running those existing default-reference tests. These commands are unexecuted recipes; all root activity uses the shared measurement lock and private owned receipts.

Root added `gdn_quad_parity --host-only` after source review: it exercises only the empty, forced-denial, negative-length and ordered-partial host contracts before queue lookup. This mode has not been executed yet; shared-library loading and the resulting process still require supervised closure. It does not qualify compiled SG32 admission or GPU parity.

`gdn_quad_parity --host-fail-stop-probe` injects an ordinary C++ exception into the same wrapper while a host endpoint and destructor marker remain live. Expected process status is2, a FAIL-STOP line, and no destructor-marker line. It calls no GPU API and tests only the host fail-stop control flow, not device-loss handling. A supervising parent must require the exact status and closed process; this mode is not registered as a GPU PASS.

Default short suite:

- Before queue lookup, host marker arrays test empty/no-submit, negativeT rejection and forced pre-submit denial. A host cancellation control verifies that `(2^24+1-2^24)+1=1` while the reassociated grouping gives2.
- T1/3/4/5/17/257, zero and nonzero initial state, two patterns: distinct48-head/key-row/column/token inputs and a controlled four-partial cancellation state. Every case makes three successive calls (T,3,5), comparing after each, including two carries. This is24 cases /72 actual candidate calls, not a timing loop.
- Every positive call requires candidate status Submitted and booltrue; an unsupported candidate is a test failure with its denial reason, not a legacy-vs-legacy PASS. Compare all786432 FP32 recurrent cells, all active FP32y and FP16y bit-for-bit; finite and complete-write checks;32 outer guards and entire unused output capacity remain sentinel.
- The cancellation nonzero state places2^24,1,-2^24,1 at the first row of the four row groups for every head/column. Only those key/query rows are1, beta=.25, g0 and v0. Changing KV association changes delta and the small state cells; output partial ordering is sensitive too. The directed fixture separately distinguishes all interleaved head groups; cancellation Q/K alone are intentionally shared. No CPU tolerance substitutes for the GPU bitwise comparison.
- A device empty test compares full state bits/hash and all output sentinels before/after T0. Forced-denial test snapshots full device state and checks all output remains untouched before caller explicitly makes one legacy reference fallback; full differential parity then checks that fallback against the independent reference state. Logged counts are caller API invocation counts, not hardware event or per-kernel submission counters.

Optional bounded synthetic prefix carry, never part of initial default execution:

```
python3 "$WQUAD/sycl/tools/debug-run.py" "$QUAD_BUILD/gdn_quad_parity" \
  --prefix 32768 --chunk 2048
python3 "$WQUAD/sycl/tools/debug-run.py" "$QUAD_BUILD/gdn_quad_parity" \
  --prefix 262144 --chunk 2048
```

Only totals32768/262144 accepted. Chunk is1..2048 and ceil(total/chunk)<=2048, otherwise reject before queue lookup. Buffers are reused; at most2048-token h/z/y exist, never full262144 raw h/y. Every prefix handoff compares full state/active outputs. With2048 chunks, there are16 or128 calls. Initial state is controlled nonzero, inputs use cumulative token offset. This is synthetic recurrence+norm kernel coverage, not physical model-prefill/session/conv/full-lifecycle qualification. Owner must first approve the short gate and budget the longer commands independently; no automatic escalation.

Root resource/closure plan: suggested short deadline240s, longer owner-specified deadline<=1800s initially, stdout4MiB/stderr64MiB diagnostic caps, finite RSS/AS/FSize/NOFILE controls and free device memory. At max2048 chunk, device allocations are roughly278MiB plus compiler/runtime resources; host live preparation/comparison buffers are roughly130MiB plus loader/runtime overhead. Those are source counts, not measured RSS/device peaks. USM allocations are tracked. Partial construction has submitted no commands and attempts to release tracked allocations, reporting cleanup failures while retaining the original error. All host endpoints of asynchronous memcpy live outside `gpu_stage` lambdas. The lambdas include submission and its completion wait; any exception there logs FAIL-STOP and calls `std::_Exit(2)` before the endpoint vectors or USM owner can unwind. Candidate/reference submission errors use the same process policy, with no retry. Normal teardown drains before releasing USM; a failed final drain also exits without C++/static destructors or explicit release. Free failures after a successful drain retain ordinary failure reporting. This conservative disposable-test policy is not engine recovery or safe session reuse. It prevents the R95 host-vector unwinding defect by source construction; independent review and runtime qualification remain required. No post-submit fallback/reset or device/process cleanup command is issued. Parent supervises faults, normal exit, PID/ticks, forced cleanup and survivors. Require final output flush and normal0; a printed PASS before later teardown/closure failure is not authoritative.

Not implemented: expected NaN/Inf propagation masks, per-head/per-tile interior red-zones, deliberate post-submit async-fault injection, a complete automatic-selector integration matrix, compiled instruction/disassembly verification, captures/model/logit/logprob/ID/MTP/all66-state checks, whole-model performance samples, profiling, GRF experiments or adoption. Outer guards and full differential arrays detect many cross-slice errors but are not a proof of every write's ownership. Private/spill reports do not prove profitable occupancy. R89's existing test gaps are documented rather than silently labelled covered. Root must complete build/support/bitwise tests, then separately validate default/selector/fallback integration and model/full-context gates before any performance or adoption decision. All existing thresholds and failed C/D/r5 statuses remain unchanged.

Reviewed frozen source `kernels.dp.cpp` SHA3a5077a284048b4fde24a4146e3b02e8d84d259480f755e4b0e94cfc52e54f1c; existing parity SHAd4ca2e5e428751252033f53038f2d43c1814e8ccedaaa8ac5e0ebccd1992a6e4. R84/R86/R87/R88/R89 were read before implementation; R88 correctly notes the old recurrence has no source subgroup attribute. Installed header source confirms compile_sub_group_size(uint32_t); no query was run. Exact changed-file hashes and source-only status are in the companion implementation report.


## Explicit quiet repeated prefix service mode (source-only addition)

Root reports the frozen earlier short and32768/262144 prefix bitwise gates passed.
Those are synthetic correctness evidence, not a timing result. The current SG32
image's reported spill5056 versus legacy0 remains a resource observation, not a
measured slowdown/speedup or permission for a GRF-256 property. This addition changes
only the parity executable and this README. Kernels, selector, API header, CMake/FP
flags, norm arithmetic and GRF modes remain unchanged. New timing mode has not been
built or executed by the implementation agent. Ordinary short, --prefix, --host-only
and --host-fail-stop-probe paths retain their previous behavior.

Only the exact nine-argument layout below enables timing:

```sh
./build-sycl-gdn-quad-v2/gdn_quad_parity --timing-prefix 32768 --chunk 2048 \
  --order legacy-first --samples 5 > quiet-legacy-first.csv 2> quiet-legacy-first.stderr
./build-sycl-gdn-quad-v2/gdn_quad_parity --timing-prefix 32768 --chunk 2048 \
  --order quad-first --samples 5 > quiet-quad-first.csv 2> quiet-quad-first.stderr
```

Root must run these as separate supervised fresh processes under the shared lock,
with no competing GPU work, clean timing environment and frozen runtime/build/device
identity. These are recipes, not executed commands. Do not use debug-run.py or the
diagnostic_environment() tracing profile for timing. The ordinary correctness mode
still uses its existing diagnostic policy. No automatic profiler starts here.

Bounds: total32768..262144 integer, chunk exactly1024 or2048, samples1..9. Missing,
extra, reordered/duplicate options, invalid order, overflow/nondecimal values and
known trace/validation/profiler/compiler-dump variables fail before queue lookup.
Samples are explicit; default mode never times. Root should start with32768/chunk2048
and5samples; longer prefixes require a separate owner decision/deadline. At the
maximum there are11full paired prefixes (2warmup+9recorded),256chunks each,5632total
recurrence+norm arm calls,2816comparisons; no loop is unbounded. No full262144 h/y
array is allocated. Fixed chunk buffers and existing state allocation are reused.

Both arms start each paired prefix with identical separately allocated nonzero
state and guards, initialized and full-bitwise checked outside timers. Every live
chunk uses exactly the same directed48-head synthetic input/tokenoffset/finite
parameters in both arms. Each arm carries its own entire prefix; it never reads or
copies the other arm's intermediate state. Arms are interleaved at corresponding
chunk boundaries so full state/FP32y/FP16y/finite/outerguard/unusedcapacity checks run
after EVERY matching chunk. This measures two independent carried prefixes with
interleaved service intervals, not uninterrupted single-arm throughput. Readback
and input transfer between chunks may change cache, clocks and residency; they are
excluded from service sum but remain experimental context, not claimed free.

Two warmup pairs run the complete selected prefix for both kernels and norm in both
execution orders, resetting state between pairs. Their service samples are labelled
warmup and excluded from recorded sample counts. Recorded sample1 uses --order,
sample2 reverses it, and alternation continues. All chunks within one pair use that
pair's order. Even sample counts are balanced; odd counts differ by one. Run both
initial-order fresh processes if a balanced aggregate is needed. Position/order are
reported source-visible labels, not shuffled/nonreproducible order.

Clock definition: host std::chrono::steady_clock from immediately before gpu_stage
calling the public explicit recurrence function through successful queue.wait_and_throw.
Summed service includes wrapper/host-call overhead, candidate exact bundle/device/
SG32 admission and resource queries on EVERY candidate chunk, legacy setup, submit,
recurrence, existing output norm, the no-op DPCT check() wrapper and queue completion wait.
check() performs no validation; wait_and_throw() supplies completion/error reporting. Same
in-order queue is idle at each start. This is host service including admission;
there is no separately measured admission estimate and none is subtracted. It is
not a SYCL event time, GPU kernel-only time, JIT-free assertion or whole-prefix wall
clock. Full-prefix warmup reduces first-use effects without proving their absence.
Existing queue profiling property is recorded and unchanged for BOTH arms; no event
profiling values are consumed or profiling queue mode toggled.

Excluded: state initialization, input generation, guard fills, all H2D transfers,
readbacks, full checks, digesting, allocation and CSV output. Host transfer endpoint
vectors are allocated outside gpu_stage callbacks and remain alive across their
submit+completion fence. Submit/wait errors use existing fail_stop/_Exit2 without
stack/static destruction or explicit USM release. No candidate-denial fallback is
accepted in timing; every candidate must report Submitted/compiledSG32 and complete.
After both arm waits, numerical mismatch fails normally after known completion;
Buffers final drain/free remains mandatory. Teardown/output failure cannot be
accepted because a preceding sample or summary line says pass.

Output: TIMING_META and TIMING_HEADER; four labelled warmup arm rows;2*samples
recorded arm rows; final TIMING_SUMMARY only after Buffers normal drain/release.
Individual arm rows retain phase/pair/order/position, length/chunk count, summed
service seconds, total tokens, logical recurrence/admission/norm call counts,
initial/final state FNV and output chunk-chain FNV, candidate last-call SG/private/spill
known/raw values, and exact paired-chunk status. Legacy resource fields are0/unknown,
not a new query or assertion that its spill remains0. Counts are public caller
invocations, not measured hardware events. No speed ratio/median/tolerance or
adoption threshold is computed; root interprets individual samples separately.

Digest definition: raw guarded float/half allocations including unused sentinel
capacity, using existing FNV1a; output chain joins prior FNV with canonical
little-endian uint64(offset,live,chunk_fnv) for each chunk. Final state FNV is last
full guarded state. Both arms' raw data are independently read and hashed after
full bitwise comparison; cross-pair digests must repeat. FNV repeat checks are
reproducibility checks, not a collision-free bitwise repeat proof or SHA provenance.
Actual paired comparisons are full arrays, not hashes. No raw tensor files emitted.

quiet_environment rejects presence (even0) of its finite named trace/validation,
LD_PRELOAD/LD_AUDIT/LD_DEBUG and IGC/compiler-option hooks. This guard is not proof
that unknown environment hooks, system-level sampling or an external profiler are
absent. Root must freeze the entire environment, confirm clean owner timing profile,
no attached profiler/tracer and no other GPU client, retain process/fault/health
closure and source/binary/flags hashes. EnableDirectSubmission=0 and the existing
copy-offload/device/adapter settings remain owner pins, not modified by this mode.

Root review/build/testing checklist: compile exact unchanged kernel/FP closure;
rerun host modes and default short regression; reject malformed timing args and
known dirty env before queue activity; quiet32768 warmup+5pairs with full comparisons
and normalexit0/no forcedcleanup/survivors/new faults; retain individual CSV plus
repeat/order/count evidence; rotate initial order in another fresh process. Root
may approve longer totals afterward. Initial proposed wall<=1800s, CSV1MiB/stderr4MiB
and finite RSS/AS/file/NOFILE supervision are owner bounds, not measured requirements.
Host scratch is still one-chunk bounded (largest additional hash pair ~96MiB, freed
between components), existing device capacity unchanged. No model/lifecycle/
prefill+conv/postprojection/decode/end-to-end performance/adoption is qualified.


## Test-build host/event discriminator (source-only addition)

R104's completed quiet synthetic service comparison is order-sensitive and includes
candidate admission every chunk. It does not identify a spill, GPU-execution or
admission cause. This separately gated diagnostic observes host sections and command
events to help separate those hypotheses; no diagnosis or new run is claimed here.
No new hardware counter/profiler, GRF/property/global FP flag or kernel arithmetic,
name, launch geometry, selector/default change is introduced.

`STRATA_GDN_QUAD_EVENT_DIAGNOSTIC` is a new CMake option, OFF by default and rejected
unless `STRATA_GDN_QUAD_PARITY=ON`. Its macro applies privately to strata_prefill and
the parity executable. Normal option-OFF builds compile out the capture struct,
thread-local sink, event saving and host-section clocks. Existing public production
signatures remain identical; extra binding API exists only in diagnostic builds.
Use a separate diagnostic build, keeping the already qualified quiet build and its
receipts intact. Existing --timing-prefix formatting and checks are unchanged and
never bind a sink. In an option-ON build a null sink disables clock/event recording,
but compiled branches/local returned-event lifetimes can still perturb host cost;
clean uninstrumented service evidence must use the separate option-OFF build.

Root-only recipe, unexecuted by the implementation agent:

```sh
# Add only these options to root's exact frozen same-flags configure recipe,
# with a NEW build directory, not over the qualified quiet build:
# -DSTRATA_GDN_QUAD_PARITY=ON -DSTRATA_GDN_QUAD_EVENT_DIAGNOSTIC=ON
cmake --build build-sycl-gdn-quad-event-v1 --target gdn_quad_parity
./build-sycl-gdn-quad-event-v1/gdn_quad_parity --diagnostic-prefix 32768 --chunk 2048 \
  --order legacy-first --samples 1 > event-legacy-first.csv 2> event-legacy-first.stderr
./build-sycl-gdn-quad-event-v1/gdn_quad_parity --diagnostic-prefix 32768 --chunk 2048 \
  --order quad-first --samples 1 > event-quad-first.csv 2> event-quad-first.stderr
```

Args retain exact9position layout, total32768..262144, chunk1024or2048, samples1..3,
strict decimals<=6bytes and existing known dirty-env rejection before queue lookup.
Option-OFF --diagnostic-prefix rejects before queue lookup. Require pre-existing
in_order AND enable_profiling; no queue property is changed. If property is missing,
DIAGNOSTIC_UNAVAILABLE is emitted and diagnostic stops before allocations/submission.
Missing captured events, unavailable get_profiling_info or decreasing timestamps
report PROFILE_QUERY_FAILED after the known drain and stop with normal final
drain/free; no substitute host data or accepted partial samples. Valid equal
timestamps are retained as zero device duration, without inferring timer precision.
Candidate admission denial emits DIAGNOSTIC_DENIED with original reason/observed
admission duration and fails; it never runs legacy fallback on that state.

Same full-prefix two-order warmup pairs, separately initialized matching state,
matching directed input/chunks, alternating trial order, service timer and full
state/FP32/FP16/guards/finite checks after every matching carried chunk are retained.
No all-token arrays. Diagnostic rows pre-reserved and held in memory; all numerical
checks and Buffers final drain/free finish before any success CSV rendering. At max
5pairs*2arms*256chunks=2560rows, modest bounded host metadata; event objects remain
only in the current arm's capture and are released after successful wait/query.
No output formatting occurs between diagnostic service intervals (failure logs only).
New diagnostic_pair duplicates the frozen fixture flow rather than changing the
quiet-mode function; root should review both for drift whenever one is changed.

Capture sink ownership: thread_local pointer, scoped bind to one fresh host capture
for one explicit public legacy/quad call. Nested bindings refused and prior binding
restored; null outside each diagnostic call. No workers/device code write the sink,
no new event/queue/command is submitted by instrumentation. Existing same-call
recurrence q.submit/parallel_for and anonymous launch_gdn_out_norm parallel_for
return their actual sycl::event, saved after the unchanged submissions. Captured
recurrence identities are GdnLegacyPipelineReference and GdnRecQuadPipelineSG32;
norm identity is existing gdn_out_norm_keyhead_kernel_43e92c. The binding API is a
test-only contract, not production interception of unrelated kernels. Do not use it
around a different wrapper or nest/widen its lifetime to concurrent calls.

Host steady nanoseconds: prevalidation from public function entry through existing
T/ld16/diagnostic prechecks (legacy also its props construction); q_of lookup alone;
quad admission from immediately before get_device through exact bundle/kernel/SG/WG/
private/spill queries and their existing read-only catches, ending before recurrence
submit; legacy admission0/not observed. Queue-property/optional construction gaps
are other host work and are not silently classified. Recurrence enqueue host time
from immediately before the existing submit/parallel_for through return, excluding
event storage. Norm enqueue host time from immediately before existing q_of+parallel_for
through return (includes that norm queue lookup; properties prepared beforehand).
Event storage/clock overhead perturbs total service. Existing check() is a no-op;
success/error boundary remains explicit q.wait_and_throw within gpu_stage.

After each existing successful wait, read recurrence and norm command_start/end
in a separate profiling-only scope outside gpu_stage. Preserve four raw device timestamp ns plus independent end-start
command duration. Event query time is outside service clock, but can affect the next
arm's cache/host scheduling context. Same queue profiling property is preserved.
Command elapsed is runtime event observation of that submitted kernel, not an
in-kernel barrier time or hardware-counter result. Do NOT add host subintervals and
device intervals into a service decomposition, subtract them to infer wait time,
or compare absolute host/device epochs. Host enqueue and device execution overlap.

Endpoint/failure policy unchanged: all memcpy host vectors outside gpu_stage and
live across their submission+successful wait. Timed wrapper and wait remain
inside gpu_stage; submission/completion exceptions fail-stop without stack/static
destruction or explicit USM release. Profiling-only errors happen after a successful
drain, report a separate status and unwind through the normal final drain/free.
Any failure of that final drain retains fail-stop behavior; no pass summary is emitted. No post-submit retry/reset/reuse. A numerical/hash mismatch
is ordinary failed process only after known drains; final drain/free/output failures
prevent authoritative normal0. Success lines alone never override process/fault/
owner closure. No active receipt or model capture is queried by this helper.

CSV: DIAGNOSTIC_META includes PID/length/config/norm identity; DIAGNOSTIC_HEADER
identifies each perchunk row's phase/pair/order/position/arm/recurrence identity,
offset/live, existing service seconds, five host ns fields, raw command timestamps
and two independent command durations. After those rows DIAGNOSTIC_PREFIX_HEADER
and existing TIMING-schema rows (diagnostic_warmup/diagnostic_sample phases) retain
prefix summed service/digests/caller counts. Final DIAGNOSTIC_SUMMARY follows all
checks and normal buffer release, then main flush checks. For32768/chunk2048/1sample:
96event rows,6prefix arm rows,4metadata/header/summary lines =106lines. Owner must
require exact expected counts, same prefix digests, fullpaired gates, profile
availability and normalexit0/no forcedcleanup/survivors/newfaults. FNV retains its
reproducibility-only status. Process PID is association, not PID/startticks ownership
proof; root receipt retains that responsibility.

Pending root gates: inspect source/compile-out and kernel-body identity; reject
optionON without parity; fresh same-flags OFF and diagnosticON builds; normal host/
short/quiet regressions; malformed args/dirtyenv OFF CLI rejection before queue;
two separate clean initial-order32768 diagnostics first; event identity/availability/
row/order/count consistency plus complete paired bits and owner health/closure.
Preserve exact source/object/binary/compiler/runtime and device-image provenance.
Bound initial wall<=1800s, text<=4MiB, finite RSS/AS/FSize/NOFILE and shared lock.
No event availability or component cause is proven by source. Diagnostics may change
quiet service; use only for attribution, then separately remeasure clean OFF service.
No model/lifecycle/performance/adoption or GRF support conclusion follows.

Root R106 review corrected the original source-only handoff: post-drain timestamp errors are distinct from unknown completion. The original implementation report remains unchanged. The reused default in-order queue installs dpct::exception_handler, which calls strata::rethrow_sycl_errors; it reports errors and rethrows the first, so the normal wait boundary does not silently consume async failures. No replacement queue or handler is installed. These are source findings; root build and runtime profiling remain pending.
