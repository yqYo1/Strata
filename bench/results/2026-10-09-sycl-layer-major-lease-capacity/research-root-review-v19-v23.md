# Parallel research and root validation, 2026-10-10 JST

The optimization goal remains open. No new speed improvement was adopted.
The original C full-context repeated-prefill numerical rejection remains
unchanged. This boundary contains CPU/build evidence, not a new GPU run.

## Work in parallel

Luna researchers completed rounds 19–22 in separate scopes: capture ownership,
native CPU dispatch, long-prefill boundaries, cross-engine Gate/Up methods,
and profiler source/protocol. They wrote reports and performed no builds,
model tests or profiling. Round21 used source/web research after the preceding
reports returned. Root shared the previous report directory, immutable registry
and Git archive with every assignment; no running researcher was interrupted.

Sol 6.1 separately implemented the native-dispatch histogram and native CPU
profiler protocol. Root reviewed their source, corrected issues identified by
Luna, and exclusively managed execution. The histogram remains unbuilt and
untested. A separate Sol source assignment is implementing the bounded capture
reader while root reviews and archives results. Research and source work may
overlap; builds, tests, profiling, model runs and cleanup use the same exclusive
measurement lock and are serial.

The committed registry and report files are the resume record. External report
files are the researchers' working outputs, rather than separate resume notes.

## Actual completed root checks

| Check | Result | Scope |
| --- | --- | --- |
| Debug `9c2ebde8` real SYCL build | Pass, 231.338 s | Compile only; binary `4fa49449…` |
| Actual target compilation flags | Pass, 115 objects / 114 sources | All reachable objects compiled; unchanged flags |
| Ordinary libc timer controls | Pass, 5 separate processes, 1.722 s | Four clock/notification combinations and ITIMER_PROF |
| Real capture header with fake queue, v2 | Pass, 12 cases, 2.322 s | ASan/UBSan framing, ownership and budget; no real SYCL |
| Five-GEN capture controller preflight | Pass | CPU-only pins/history checks; model admission still held |
| Native gprofng collection | Rejected, 4.710 s | Native target/protocol normal; no usable profile |

The capture source fix retains one bounded readback destination until process
exit if both the primary wait and fallback drain throw. Later captures reject
before another copy, and the original error remains primary. This was an error
path found by inspection; it has not been shown to cause C's numerical defect.
The mock lets the delayed copy complete after capture destruction. ASan stays
enabled; LSan is disabled only for the two intentional quarantine cases.
It cannot prove actual device retirement, USM or kernel behavior.

The initial mock v1 was rejected because root used `floor(pos/4)` instead of
`floor((pos+1)/4)` for completed blocks. Its file was 133,495,680 bytes, 192 bytes
short of the intended geometry. No production change was needed for this error.
The corrected v2 emitted 690 frames / 133,495,872 bytes, leaving 721,856 bytes
under the 128 MiB cap, and rejected one additional large record before queue
work. Original v1 receipt, source and failure status are preserved. Its large
synthetic payload has no remaining consumer; the retirement plan pins it.

The native profiler target has no application timer or SIGPROF handler. Compile
and collector exited 0; READY/DONE/BYE completed with no forced cleanup or live
owned processes. Root's first reader incorrectly rejected NUL padding followed
by a final newline. Closed-only extraction, without recollection or changing
the original failed receipt, confirms independent collector cerror 9,
`itimer could not be set`, no ptimer profile schema and zero-byte frameinfo.
The target mapped the installed collector and libc; their hashes and filtered
target-effective environment are preserved. The failure also occurs without
Python, but the internal failing API/errno or resolved function pointer remains
unknown. The five ordinary timer controls do not qualify gprofng.

## Research reconciliation

Round21 boundary corrections: cell 98328 **is divisible by four** and starts
pooled block 24582, which completes at 98331. QSA ordinal 2 is model layer 11.
The unchanged C values were four generated token IDs; internal QSA selected IDs
were not captured. Preserve the original report and use these corrections.

The root verified the external QSA reports in their primary issue threads.
The tile-union RFC's later discussion corrects its original baseline and narrows
the server gain; these are another GPU's results, not B570 forecasts. The top-k
report distinguishes order, tied sets and a separate unresolved visible-text
symptom; do not use it as proof of Strata's cause. SGLang's issue illustrates
that a fault surfacing at a readback wait need not originate at that wait.
[vLLM tile-union RFC](https://github.com/vllm-project/vllm/issues/55394),
[vLLM top-k report](https://github.com/vllm-project/vllm/issues/54521),
[SGLang asynchronous-fault report](https://github.com/sgl-project/sglang/issues/37633).

The new Zen3 candidate is conditional high-NT register-pressure inspection.
`row_dot` holds `accf[NT]`, `acci[NT]` and decode temporaries; this is source
evidence, not proof of spills or a bottleneck. First measure actual format/NT
exposure, then inspect generated code and test the affected specialization.
Token chunking repeats weight decode and can lose. Cache-line task boundaries
are lower priority and require actual worker-tail/coherence evidence.

H `96bd5bb4` classification predicates match statically, but histogram-off still
uses the resolver refactor. Counts are expert-job row exposure, not task-fragment
calls, timing, physical DRAM traffic or worker imbalance. Enabled collection
affects host timing intervals. Build/parity/model and clean timing gates remain
open; the four T passes do not qualify H.

Round23 found no demonstrated installed collector switch for the missing timer
details. Static strings support `SP_COLLECTOR_TRACELEVEL`, whereas the documented
debug resolver/call strings are absent from this installed release library.
An isolated, opt-in collector build could record pointers/versions/objects,
immediate errno and actual period. No such build or collection was performed.
Keep this lower priority than the numerical prefill blocker.

The researcher's round19 public-source HTTP read was within the user's request
for web exploration. Root's earlier ambiguous “no-fetch” restriction meant no
Git fetch/clone or source-control synchronization. Later assignments explicitly
allow read-only HTTP. The original disclosure remains in its report.

## Next root work

Before a GPU capture, integrate and CPU-test the bounded reader: parse all
headers and payload bounds, require exact layer/phase/row coverage and five-GEN
byte-ledger assignment, retain per-record hashes, and pin the reviewed launch
controller. Source preflight alone does not satisfy this gate. Use the original
control/save/resume/full1/control/full2 history and fresh health/fault scope.

The bounded diagnostic deliberately omits clipped-tail, RESTORE, refusal and
later-control lifecycle checks. Even two equal captured repeats cannot qualify
the full lifecycle or clear the original C failure: extra waits may mask it.
Keep `full_lifecycle_passed`, `performance_eligible` and `adopted` false.

After locating the numerical defect, return to separate prefill/decode tuning.
Research remains iterative after new evidence, with shifted scopes and shared
past reports; root keeps managing all execution serially.
