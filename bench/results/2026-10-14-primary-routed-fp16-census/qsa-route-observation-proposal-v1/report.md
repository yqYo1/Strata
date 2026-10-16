# Host-returned QSA route census proposal v1

2026-10-14 JST. Main artifact only: unapplied, unbuilt, untested, unmeasured,
unadopted. Live v4 screen/native68da source/recipes are untouched. No build,
test, GPU launch, profiler, process/service/reset operation or cleanup ran for
this proposal. The generator writes only this directory and pins current
src/prefill/prefill.cpp6b4b7a71. Kernel/API/CMake files do not change.

The 34K diagnostic had a49.455s nonexclusive QSA-attention interval, but it
never observed natural matrix acceptance/decline. This caller-only census
resolves the prerequisite: which current implementation actually ran, how many
host admission returns and query rows it covered, and how many32-query fallback
slices were called. No interval becomes exclusive service time or critical path.
R531 is the source review; R532 separately maps decode CPU host/pool work.

The patch preserves the exact short-circuit expression old_attn || !prompt(...)
and the returned bool selects the same fallback loop. One prompt call at most,
no duplicate device capability query. The old variable is a presence flag:
setting STRATA_PROMPT_ATTN_OLD=0 still forces bypass; omit it for natural routing.
Counts are opt-in via exact STRATA_PREFILL_QSA_ROUTE_OBSERVE=1, defaultOFF.
KV mode is inferred once from the same host pool pointers as current callee.
Rows distinguish accepted/false-declined/forced, current T, mode, staging;
fallback rows distinguish actual nb, mode and staging. cap/page-size are fixed
per admitted run and reported. Each QSA layer has its own completed-chunk count.

Observer bounds: primary unsplit SYCL24/2/256 geometry, <=512QSA layers,
0<n<=262144 within position range, positive chunk/capacity<=8192, positive
fallback batch<=capacity. Host counters allocate at most2101504 bytes only when
requested. Invalid geometry or observation indices turn the observer ineffective,
without selecting a different inference path. Allocation failure declines the
observer; current inference calls continue. Serialized output limits2048 active
rows and256KiB. A later allocation failure while formatting output can still fail
this diagnostic; no all-failure partial-record guarantee is claimed by this patch.

Emission occurs only after the existing successful compute/copy synchronization.
A host accepted return is submission evidence, and the separate existing final
sync plus native/owned/fault receipts establish completion. A thrown capability,
span or enqueue exception propagates as before; it is never converted to false
or fallback. No ledger is emitted for a run that aborts before successful return.
The outer failure receipt remains authoritative. Detailed admission-decline
reasons and partial exception-ledgers from R531 are explicitly not implemented.
Add those only if returned-route results leave a concrete unresolved question,
inside the existing evaluated guards, without querying properties twice.

Counts invariants are checked before effective output: accepted+declined+forced
must equal chunks*QSA layers; their summed T must equal input*QSA layers;
fallback query sum must equal declined+forced query sum; slice count must match
sum ceil(T/batch), and exact histogram totals must match scalar counts.
For canonical34560batched/8*4096+1792/12QSA layers, source-derived checks are
108returned layer-chunk decisions and414720query rows. All-fallback/batch32
would be12960slice calls. These are expectations, not observed runtime counts.
The last real input consumed by nativeT1 is outside this batched census.
Selected-cell occupancy, physical traffic and layer timing are not measured.

Main's next gates remain pending and serial: finish v4screen plus standard
32K/agent/long-output validation; perform the user-requested PR#6/Codex public
history/safety recovery audit; recheck this source base if that changes prefill;
apply only then under the measurement lock. First verify exact-flag defaultOFF
and rejection scope; build with current configured root/toolchain; compare OFF/ON
full residual/first logits at chunk+tail; run canonical34K observation and quiet
reference with source/binary/input pins and native/server/fault closure. Check
natural/forced routes separately and retain all results. Counter consistency is
not a numerical/quality/full262144/RESTORE or speed qualification.

If every natural query is matrix-accepted, do not claim that the earlier
fallback-transpose proposal helps this workload. If fallback is used, pursue the
separate pinned transpose proposal with its numerical/lifetime/physical-context
checks and component-level quiet A/B. No candidate is adopted here.

Retention ownerMain, consumer source review and the serial route census after
requested benchmark/PR6 boundary, next review when route results close. Only
compact source/patch/research is kept in Git; no raw capture is created.
