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


The [host-only32K profile](host-api-profile/README.md) then passes three exact
full-state/head/output checks while capturing native host API counts without
GPU event instrumentation. Concurrent API durations do not identify PCIe or
CPU dominance. A separate [attention128/layout1 experiment](qsa-batch128-layout1/README.md)
passes three full-state/head controls and a clean32K ABBA comparison on the
unchanged private scheduling binary. Every performance input is32,768 tokens.
Full262,144-cell serving and production adoption remain separate pending gates.


The [processing-order/GPU-residual follow-up](processing-order-and-gpu-residuals/README.md)
adds nine exact 32K main-state/head/output controls and a clean three-condition
default/RAM/GPU/GPU/RAM/default comparison. GPU residuals use MTP decode-only
release and immutable-RAM restoration, with full payload checks before timing.
All comparisons read 32,768 tokens; full262,144-cell serving remains unvalidated.


The [main-cache release/restoration controls](main-cache-release-and-restore/README.md)
add actual-code ASan/UBSan offset/rollback tests with a patterned negative
control and completed 32K kept/partial/full lease checks. Clean timing and
full-context serving remain separate gates; see the checkpoint's completed list.

The [same-process full-RAM repeats](same-process-main-cache-repeat-32k/README.md)
pass nine complete32K raw-state/head/output checks and trace release of
already-used UR command buffers before physical unmap. Capture durations
are excluded from speed comparisons; full262,144-cell gates remain open.


The [private KV-streaming source candidate](kv-streaming-source-candidate/README.md)
fixes possible shared hit writes and host-counter type-punning before
exercising the existing RAM-KV route. It compiles/links but is not GPU-tested;
no speed, full-context or adoption result is claimed.


The [RAM-KV32K processing-order comparison](kv-streaming-32k-and-layer-order-v3/README.md)
records defined KV representations/page claims, three exact used-state/head
reads per configuration, and matched quiet ABBA runs with initial versus
repeated32768-token measurements. Full262144 occupancy and adoption remain
pending.

The [visible-output commit follow-up](visible-output-commit-capacity-gate-v1/README.md)
reproduces two invisible committed tokens at a64-output limit and retains a
CPU-tested private fix. Its initial32K state/head/output and exact saved-prefix
gate passes; the same process is continuing full256K repeat/restore/refusal
checks. Logged correctness durations are excluded from speed comparisons.

The [first full-capacity image and polling candidate](visible-output-commit-full-first-and-poll-backoff-v2/README.md)
freezes the completed first262140-input/4-output read through cell262143,
all13 saved KV layers through262144 cells, and its following exact32K
control. The parent sequence was active at capture. A separately compiled
host polling candidate passes CPU sanitizer checks but is not GPU-tested;
neither this partial archive nor synthetic polling counts claim speed.

The [full-capacity repeat and indexer spare follow-up](indexer-spare-full256k-rejection-and-candidate-v1/README.md)
records two matching fresh full reads and every saved state/KV byte, plus
actual32K disk-resume parity. Full disk roundtrip is rejected because the
last pooled indexer row differs in each main layer. The process exits0
normally without a kernel fault. A private kernel candidate restores the
spare after partial speculative commit; its new capacity sequence is
separate, and no full-gate/adoption or speed result is claimed here.

The [repaired first full image and profiling preparation](profile-definition-audit-and-poll-rebase-v1/README.md)
records the first repaired full read/image plus its exact interposed32K
control. The second full read is active at that historical capture. Actual preprocessing
also exposes inconsistent DPCT class/inline definitions across compiled
translation units. A private uniform-header build replaces only seven
objects, while a separate polling rebase and local VTune help checks prepare
later measurements. Those preparation snapshots precede GPU checks and
do not establish adoption.

The [terminal watchdog abort and separate32K definition check](full256k-watchdog-abort-and-uniform32k-v1/README.md)
records the subsequent host watchdog stop during the second full read,
the exact stopped stacks and pending DMA submissions. No new kernel fault
is recorded; post-exit device/runtime health passes. The cause remains
unresolved. The uniform-header executable's separate four fresh32K reads,
actual disk-restored continuation and normal owned exit pass. The complete
physical256K and quiet speed gates remain pending.

The [subsequent GPU timestamp rejection and polling gate](device-profile-rejection-and-poll32k-v1/README.md) retain the actual earlier VTune startup failure on this Ryzen processor and a failed logged 32768-token unitrace GPU timestamp run. The application watchdog stopped at layer 26 of chunk 16384; no new kernel fault was recorded and post-exit GPU/runtime health passed. Its empty/incomplete trace and durations are excluded from performance conclusions. The DD5-based polling candidate subsequently passed four fresh 32768-token reads, all head/used-state/output/logprob/MTP comparisons, actual disk continuation and normal exit. It is still private: quiet matched inputs of at least 32768 tokens and the complete physical 256K gate remain required. First and repeated full reads must be reported separately; the underlying pending-counter cause is unresolved.

The [subsequent quiet polling comparison](poll-matched-quiet32k-v1/README.md) completed sixteen fresh 32768-token reads in control/candidate/candidate/control order on the same B570. First process reads are separate from later full reads. Subsequent prefill was 427.510 tok/s for DD5 and 427.225 tok/s for polling backoff (-0.067%); decode was 16.486 and 16.373 tok/s (-0.686%). All output IDs, logprobs and visible MTP counts matched, with RESUME 0 and REUSED 0, normal QUIT and no new kernel fault. There is no useful prefill gain, and this small sample does not establish a decode benefit or regression. Polling backoff is not adopted. At that quiet 32K boundary, the complete physical 256K gate and pending native counter diagnosis were unresolved; diagnostic/profiled durations are excluded.

The [later complete DD5 physical 256K sequence](uniform-full256k-and-qsa-source-v1/README.md) passed two fresh full reads through cell 262143, all 13 saved KV layers with 262144 cells, every saved state/KV tensor byte on repetition and actual disk restoration, a clipped two-token tail, both capacity refusals and a later fresh 32768-token input. Head/used-state/output/logprob/MTP gates and normal owned exit passed without a new GPU fault or reset. Diagnostic durations are excluded; this does not resolve the earlier pending-event cause. A separate twelve-head QSA reduction has CPU arithmetic/routing checks and source-only preparation, with no engine compile, GPU numerical result, speed gain or inherited capacity proof at this archive boundary. Every future performance comparison uses at least 32768 input tokens and separates first/later full reads.
