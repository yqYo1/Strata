# 32K output reproducibility and stopped-job follow-up

All model jobs here use the same32,768-token code-review fixture as the parent,
context33024,8192-token chunks, int8 KV, normal MTP4, cache128, five CPU workers,
FIRST0 and RING8 on the same Arc B570 boot. No short-input speed comparison is
used. State/head capture and API logging exclude these jobs from performance
comparisons. No production source/binary was changed by these experiments.

The first fresh production head capture completes64 finite-logprob outputs,
but its first logprob (-1.152473) differs from both previously observed results
(-1.023937 and -1.051243). The next identical head job stops in its first chunk
at layer1. The kernel journal records one CCS engine reset. The main stack waits
in queue::ext_oneapi_submit_barrier, and a runtime host-task thread waits for a
native event in assignKernelEventCompletionData. The code_location fields next
to call return0x59f6b0 identify line4823,column45 of the recorded production
prefill.cpp: compute waits for the staged copy marker. This localizes a wait;
it does not establish which operation caused the engine reset.

A private candidate replaces only expert/PLE host_task acknowledgements with
owned events returned by the actual DMA and status queries before host-buffer
reuse. Ring ownership, event lifetime, failed-query handling, monotonic generations
and concurrent CPU readers pass normally and under ASan/UBSan. Its first logged32K
job nevertheless stops in layer0 while Stager::wait waits for readiness. Three
staging threads are inside queryCounterBasedEventStatus ->
synchronizeTimestampCompletionWithTimeout -> assignKernelEventCompletionData.
No new xe fault is recorded. The candidate is not adopted; CPU tests do not
prove runtime completion or GPU correctness. Its initial relocated-source compile
failure and corrected private build are both retained.

Both stopped jobs are cleaned up with neither owned inferior nor debugger
remaining. Fresh logged H2D/kernel/D2H checks pass on the same boot after each
failure. No reset, rebind, reboot, service or package change is performed.
The private candidate trace uses immediate command lists and successful native
appends; SUBMITTED status alone is not evidence of unflushed regular batches.

The installed NEO source version26.31.39395.14 places both observed native waits
in its profiling counter-event completion path. A process-local experiment with
EnableImplicitConvertionToCounterBasedEvents=0 keeps the production executable,
arithmetic and queue profiling unchanged. Its logged and unlogged state/head32K
jobs both complete normally without xe faults, but their states and outputs differ.
Two successful jobs do not prove prevention, and this setting does not fix the
output discrepancy. Full trace/state/head files stay private with receipts.

The post-prefill captures contain66parts, taken after32767 batched input tokens
and before the held-out input token/first verifier window. PLE history and the
first three QSA K/V/scales/pooled states match completely. The first differing
GDN storage ordinal is11 (model layer14); all preceding GDN rows match. The first
different QSA ordinal is3 (layer15), with the first K/V byte difference at token8200,
in the second8K chunk. Parts1,3 and21..65 differ; dead/block positions match.
All248,320 first-head floats differ (maximum absolute difference1.692817,
RMS0.200383). This proves a discrepancy exists before decode, without identifying
its first producing operation or excluding legal BLAS reduction variation.

The state/head gate therefore stops before a second state repeat and before
either clean timing job is launched. No rate from these jobs is accepted for
tuning. The rejected private candidate and failed equality gate remain explicit;
neither is a fix. Full262,144-cell normal-MTP validation and PP1000/TG70 remain open.
An additional documented oneMKL GPU CNR experiment is being evaluated separately.

Primary references: [NEO event completion](https://github.com/intel/compute-runtime/blob/26.31.39395.14/level_zero/core/source/event/event_impl.inl),
[NEO event conversion](https://github.com/intel/compute-runtime/blob/26.31.39395.14/level_zero/core/source/cmdlist/cmdlist_hw.inl),
[SYCL events](https://github.khronos.org/SYCL_Reference/iface/event.html),
[oneMKL GPU reproducibility conditions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/reproducibility-conditions.html).
Source URLs/hashes and observations versus inferences are recorded in analysis.


The [oneMKL GPU CNR follow-up](cnr32k/README.md) also leaves the equality gate
open: two fresh32K captures agree in every state/head byte, but a third differs
in later state parts and output. All three exit normally without xe faults;
neither clean speed job is launched. CNR is a tested diagnostic condition, not
an adopted fix or explanation of all failures.


The [actual-DMA candidate with CNR](event-ack-cb-cnr/README.md) matches the
production control's complete state/head in its logged32K run, then stops in
native timestamp status queries in the next unlogged run. It remains unadopted;
owned cleanup and same-boot GPU health pass without a manual recovery action.


The [nonprofiling private copy queue](event-ack-no-profile-cb-cnr/README.md)
completes three32K comparisons with identical prefill state, full head and
output, without new xe faults. It is not adopted: its directly owned queue is
outside the existing global-wait registry. A registered factory is separately
rebuilt and validated before considering any clean timing or production use.


The [registered copy-queue candidate](event-ack-registered-copy-cb-cnr/README.md)
preserves the existing global-wait contract and passes three matched32K full
state/head/output checks with no new xe faults. Every translation unit uses
the same changed header and production compile settings match. This is still
a private correctness candidate; clean timing, default-environment repeats and
full262,144-cell serving remain separate gates, with no adoption claimed.


The [no-CNR and prefill scheduling follow-up](registered-copy-no-cnr-and-prefill-scheduling/README.md)
adds six complete exact32K state/head checks and a clean32K ABBA comparison.
The registered-copy candidate is reproducible in these checks without MKL CNR;
the prefill scheduling candidate changes only14 root-sync properties. Both
remain private while default-backend and full262,144-cell serving gates are open.


The [default counter-conversion check](registered-copy-default-counter-conversion/README.md)
then passes another three exact32K full state/head/output controls on the same
private scheduling binary, with both implicit-conversion override and MKL CNR
absent. All exit normally without new xe faults. These captured runs are not
clean timing evidence; full262,144-cell serving remains unvalidated.
