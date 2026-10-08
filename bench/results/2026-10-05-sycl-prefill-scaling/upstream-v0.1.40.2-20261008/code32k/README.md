# 32K comparison of upstream v0.1.40.2 and the SYCL integration

The unmodified upstream tag and integrated fork each process exactly 32,768
input tokens and generate 64 greedy tokens with normal MTP4. The input is an
expanded real-source code-review prefix, followed by a completed review question
and assistant prefix. `coding-review-32k-fixture.json` records the original fixture,
all suffix IDs, round-trip check and final shared SHA256. Prompt/conversation
caching is disabled; every process reads the entire input once.

Hardware and runtime are the same as the parent integration record: Arc B570 10 GiB,
Ryzen 5 5600X, 128 GiB RAM, installed NEO 26.31.39395.14, oneAPI 2026.1.1, Level Zero V2.
Both request context33024, int8 KV, 8192-token prompt chunks, cache128 per-layer,
five CPU workers and pcie0. Common FIRST0 and RING8 disable the short initial
chunk and set the streaming ring to eight blobs. No layer-major/compact/weight-release
tuning flags or extra phase waits are forced. Record actual allocation because
upstream packs 144 mixed-size cache slots into 281 MiB while the fork preserves
128 explicit-count slots into 325 MiB. These are comparisons of complete
configurations; different residency prevents attributing a difference solely
to the lifetime or launch changes.

The initial logged/validated control for each executable is functional evidence
only. Clean timings have no API tracing, validation, additional waits or logits
dump. Each is a cold first request in a fresh process; the engine's DONE time
includes request JIT/graph preparation and PLE, and excludes model startup.
An external owned GDB/PTY observer remains attached without interrupts in all
successful runs. Clean order is upstream/fork/fork/upstream (ABBA).

| Clean configuration | Repetitions | Prompt tok/s | Decode tok/s | Accepted tuning reference |
| --- | ---: | ---: | ---: | --- |
| Unmodified upstream | 2 | 393.65 | 17.14 | yes |
| Integrated fork | 2 | 334.93 | 16.96 | no, output mismatch |

All six processes finish normally with 64 finite-logprob tokens, complete owned
cleanup and no new xe fault. Both independent upstream source-status checks
remain clean before and after execution. Per-run protocol, project messages,
health probes and immutable executed controllers are included here. Full diagnostic
API logs remain private; their sizes and SHA256 are in each record. Logged
control rates are excluded from the table and retained in the raw CSV.

The table retains observed elapsed times, including the rejected fork rows.
Only the upstream is accepted as a reproducible tuning reference in that sequence:
its logged control and both clean repetitions match all64 token IDs and every
protocol logprob. The fork's logged control and second clean repetition agree,
but its first clean repetition has a different first logprob and diverges in
IDs at output index34. The first token's logprob is -1.023937 versus -1.051243.
This is a difference within the same binary, input, cache placement and requested
settings, so mixed cache sizing between configurations cannot explain it. It
shows unresolved output reproducibility without identifying the resource or
proving logging caused the difference. The fork times cannot support an accepted
speed comparison or tuning decision before this is resolved. Finite output,
normal exit and no xe fault do not establish model correctness.

These records establish completion of these32K jobs. They do not establish the performance
of larger contexts, repeated/restored conversations, cache release/recreation,
the unresolved earlier process waits, or the PP1000/TG70 target. Full262144-cell
normal-MTP gates remain open. Short37-token preflight timings remain excluded
from performance comparisons.


Later32K state/head follow-up localizes an output discrepancy to the prefill
state, before the first verifier window. An identical head job records a CCS
reset; a rejected private event-ack candidate stops without a new xe fault.
Both are cleaned up and subsequent same-boot logged GPU health passes. Disabling
implicit counter-event conversion permits two completed32K state captures but
does not make their states/outputs equal. The equality gate prevents clean timing
jobs from launching. See [full records and limits](repro32k/README.md).

The later [registered nonprofiling-copy comparison](repro32k/registered-copy-no-cnr-and-prefill-scheduling/README.md)
passes three exact32K state/head/output checks without MKL CNR, then another
three on a scheduling-only candidate removing14 independent prefill root-sync
properties. A dump-free32K ABBA comparison passes all four exact-output,
normal-exit/cleanup and no-new-xe checks. Registered copy averages406.78 PP /
16.96 TG tok/s; without the prefill root-sync properties it averages408.52 /
16.06. The prompt difference is smaller than the control's repeat spread;
the decode results vary within the no-root arm. This does not establish a
useful speed change. These private candidates are separate from the original
production/upstream comparison and remain unadopted while full262,144-cell
serving validation is incomplete. The later [default counter-conversion check](repro32k/registered-copy-default-counter-conversion/README.md)
passes three exact32K full state/head/output captures on the same scheduling
binary without either the counter override or MKL CNR. Those captured timings
are excluded from the speed comparison.

The [subsequent GPU timestamp rejection and polling gate](repro32k/device-profile-rejection-and-poll32k-v1/README.md) retain the actual earlier VTune startup failure on this Ryzen processor and a failed logged 32768-token unitrace GPU timestamp run. The application watchdog stopped at layer 26 of chunk 16384; no new kernel fault was recorded and post-exit GPU/runtime health passed. Its empty/incomplete trace and durations are excluded from performance conclusions. The DD5-based polling candidate subsequently passed four fresh 32768-token reads, all head/used-state/output/logprob/MTP comparisons, actual disk continuation and normal exit. It is still private: quiet matched inputs of at least 32768 tokens and the complete physical 256K gate remain required. First and repeated full reads must be reported separately; the underlying pending-counter cause is unresolved.

The [subsequent quiet polling comparison](repro32k/poll-matched-quiet32k-v1/README.md) completed sixteen fresh 32768-token reads in control/candidate/candidate/control order on the same B570. First process reads are separate from later full reads. Subsequent prefill was 427.510 tok/s for DD5 and 427.225 tok/s for polling backoff (-0.067%); decode was 16.486 and 16.373 tok/s (-0.686%). All output IDs, logprobs and visible MTP counts matched, with RESUME 0 and REUSED 0, normal QUIT and no new kernel fault. There is no useful prefill gain, and this small sample does not establish a decode benefit or regression. Polling backoff is not adopted. At that quiet 32K boundary, the complete physical 256K gate and pending native counter diagnosis were unresolved; diagnostic/profiled durations are excluded.

The [later complete DD5 physical 256K sequence](repro32k/uniform-full256k-and-qsa-source-v1/README.md) passed two fresh full reads through cell 262143, all 13 saved KV layers with 262144 cells, every saved state/KV tensor byte on repetition and actual disk restoration, a clipped two-token tail, both capacity refusals and a later fresh 32768-token input. Head/used-state/output/logprob/MTP gates and normal owned exit passed without a new GPU fault or reset. Diagnostic durations are excluded; this does not resolve the earlier pending-event cause. A separate twelve-head QSA reduction has CPU arithmetic/routing checks and source-only preparation, with no engine compile, GPU numerical result, speed gain or inherited capacity proof at this archive boundary. Every future performance comparison uses at least 32768 input tokens and separates first/later full reads.

The [later QSA twelve-head build, numerical gate and matched 32K comparison](repro32k/qsa-reduce12-build-numerical-and-quiet32k-v1/README.md) replaces one kernel archive member on DD5. Four fresh 32768-token numerical reads and actual disk continuation passed, then all sixteen quiet A/B/A/B reads matched IDs/logprobs/MTP and exited normally. Initial process PP was 392.213 versus 395.194 tok/s; later full-read PP was 427.237 versus 432.984 tok/s (+1.345%). Later TG was 16.730 versus 16.410 tok/s (-1.916% in this small sample); no decode gain is claimed. Logged numerical durations are excluded. The QSA binary remains unadopted pending its own complete physical 256K lifecycle, which cannot be inherited from DD5. Source-only HC/HIP knob applicability checks do not enable unsupported SYCL flags.

The [later complete QSA physical 256K sequence](repro32k/qsa-full256k-and-gemm-phase-preparation-v1/README.md) passed two fresh full reads through cell 262143, all 13 KV layers with 262144 cells, every saved state/KV tensor byte against DD5 and on repetition and actual disk restoration, a clipped two-token tail, both capacity refusals and a later fresh 32768-token input. All twelve request gates, six session operations and normal owned exit passed without a new GPU fault. Diagnostic durations are excluded. Earlier statements about uncompiled or capacity-pending QSA describe their respective archive boundaries. The later archive also preserves separate source-only GEMM dispatch and unexecuted compute-phase profiling preparations, including CPU rejection checks; neither inherits QSA qualification. Every performance comparison uses at least 32768 input tokens with first/later full reads separate.
