# Borrowed queue destruction and orderly exit, 2026-10-06

The real Strata engine reproduced a CPU SIGSEGV after four completed normal
MTP requests and the message `all queues completed`. No new xe fault was
recorded in that run. Restoring ordinary C++ destruction alone therefore
failed; this was not evidence that the Level Zero context had already shut
down. See [the failed run](model-before/record.json).

The exact same frozen engine, SHA-256
`600815fb7c3294eb54779aa2e5dee9198117f60888201abff9ef7644a281d38e`,
then reproduced the crash under GDB. Its main-thread stack is
`urQueueFinish -> sycl::queue_impl::wait -> Verifier::~Verifier -> main`.
The return address `0x4d59dd` disassembles to the second `copy_->wait()`, after
`destroy_queue(cs_)`. See [the stack](model-before-gdb/gdb-shutdown.txt).
GDB's wrapper returns 255 for the signal; the plain engine returned -11.
Both completed the four requests first. Neither run recorded a new xe fault.

An unused `ver_same` is constructed even in a single-GPU serve session.
Migration had changed its owned `cs_` and `copy_` initializers from CUDA's
`nullptr` to the same borrowed `get_in_order_queue()` pointer. Its destructor
removed that queue from the device's owning vector, then waited through the
other dangling pointer. The device's own default-queue pointer also became
dangling. MTP's unloaded path, prefill's unused/partial-init copy path and
remote-expert partial initialization had the same ownership mistake.

The candidate keeps owned queues null until creation, selects the owning
device during teardown, and checks asynchronous errors before reclamation.
Prefill distinguishes its borrowed compute queue from its owned copy queue.
Peer compute, output and copy queues are drained before freeing their shared
buffers; peer output is also finished before freeing its primary destination.
The peer tier is currently rejected by this SYCL port, so that ordering fix
is not attributed to this single-B570 incident.

The common DPCT `destroy_queue` helper now rejects either borrowed default
queue and queues owned by another device. This adds a local failure boundary
if a future caller violates ownership, before it corrupts runtime references.
It cannot replace the queue waits or prove all GPU memory accesses valid.

Production-code CPU tests use queue/USM stand-ins and ASan/UBSan:

| Evidence | Result and scope |
| --- | --- |
| [Verifier destructor and actual initializers](verifier/record.json) | Old unused and partial-init paths reproduce two UAFs. Five normal candidate cases pass, including separate owning device and partial initialization. A sixth injected failed-wait case reaches termination before reclamation; the CPU observer exits 86 deliberately. |
| [Actual common queue destruction method](compatibility/record.json) | Old in-order and out-of-order default destruction reproduce two UAFs. Six candidate cases pass: both defaults rejected, owned queue destroyed, foreign queue rejected, null and repeat calls accepted. |
| [Registry/callback regression](registry/record.json) | Five candidate cases pass; two old callback UAF controls still reproduce. |
| [Production Linux pipe reader](serve-input/record.json) | Four ASan/UBSan groups pass, including an open writer on shutdown, STOP/CRLF/EOF and an entire 262,144-token input line. This is protocol validation, not full-context GPU execution. |

The failed-wait case originally expected a swallowed exception. That
expectation was corrected: a destructor function-try-block rethrows when its
handler falls through, and a noexcept destructor terminates. No successful
reclamation is claimed after a failed queue wait.

The Linux serve exit also joins its stdin reader and watchdog before the
global drain, then uses ordinary destruction on success. This permits NEO's
internal direct-submission ring teardown instead of skipping it with `_Exit`.
Failed global completion retains the nonzero immediate exit, avoiding unsafe
reclamation. The Windows branch remains unchanged; this port builds on Linux.

The first real model candidate with those ownership fixes, engine
`dda41b86e83036ac36bd1810a0a9d289bb2bad6f9fb0a135f7f9faec76b39435`,
passes four normal MTP requests, checkpoint repeat/restoration and ordinary
exit 0 with direct submission disabled. All generated IDs and printed
logprobs match the retained normal-MTP report. No new xe messages appear.
See [the successful run](model-no-direct/record.json) and
[its protocol summary](model-no-direct/protocol-summary.json).
The final engine includes the common-helper guard, SHA-256
`4c4961439c89ced05df526ac0e39104fa528bcf25fa9276afdace03673452601`.
It passes the same four MTP/checkpoint requests and exit 0 with default direct
submission and default V2 copy offload. All four IDs/printed logprobs match the
historical report, and no new xe fault appears. Actual ring/semaphore allocation
logs confirm that direct submission is active. The second internal semaphore
again starts at `0xffffd556aa2e0000`, matching the lower 48 bits of the original
fault address, now inside an actual healthy Strata process. See
[the final default-settings run](model-default/record.json).

The final engine also passes four no-MTP/no-cache requests, same-mode repeated
IDs/printed logprobs and ordinary exit with default direct submission and no
new xe faults; see [the unloaded-MTP check](model-no-mtp/record.json).
Its printed logprobs differ from the historical MTP mode, so cross-mode
logprob equality is not claimed. The first no-MTP controller was rejected
before READY because it requested conversation caching, which this explicit
diagnostic mode prohibits. That rejected configuration and its no-fault kernel
record are preserved [separately](model-no-mtp-rejected/record.json).

The final full header rebuild exposed a missing oneMKL include path in the
shell environment. Rebuilding with explicit installed MKLROOT/CPATH succeeds;
no compiler arithmetic flags or math sources were changed. Commands, source
and failed/successful log digests are in [the final build record](final-build.json).

All model checks here use context 128, 37 prompt tokens and four generated
tokens per request. Allocation logging and persistent cache off are explicit
process settings. They do not prove full logits/state equality, 262,144-token
execution, long-load stability, a failure rate, or the speed target. They prove
a project ownership bug and the short shutdown regression it caused. Its link
to the first BCS semaphore faults remains the inference documented in
[the shutdown associations](../direct-submission/README.md), not a captured
PID/ASID attribution or a complete cause of the later kernel recovery hang.

Protocol summaries retain IDs, printed logprobs, protocol control lines and
the digest/path of the complete private allocation log. The pinned model
binary and unabridged controller output remain in the user's private state
directory. Evidence files contain no binaries or third-party issue snapshots.
