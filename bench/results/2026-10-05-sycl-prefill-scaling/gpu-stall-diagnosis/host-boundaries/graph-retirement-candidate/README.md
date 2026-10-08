# Graph retirement candidate after the repeat crash

This candidate changes resource lifetime ordering. Its build and CPU checks
pass. It has not run on the GPU and does not establish recovery, model-output
equality, improved speed or prevention of the recorded fault.

MTP release completes the device's queues, verifies its immutable RAM source,
discards every prompt/round/step executable graph including coupled variants,
and only then unmaps physical memory. The next use captures fresh graphs after
successful restoration. A partially failed unmap keeps the suspended state;
failed restoration never marks the weights ready. The target expert cache
also requires a retirement callback before releasing backing and rebuilds
verifier graphs after restoration for both VMM and ordinary allocation.
Keeping all backing skips the callback work.

[Thirteen CPU cases](graph-ordering/record.json) compile the actual production
release/restore and graph-drop functions with queue, allocation and graph
stand-ins under ASan/UBSan. The old-source control retains all 54 graph handles
across release; it does not submit them to a GPU or manufacture an ASan UAF
as proof of the real crash. The candidate checks repeated release/restore,
disabled release, failed completion, missing RAM, byte verification failure,
partial unmap, failed map, verifier retirement/rebuild, failed commit/wait,
unsupported multi-GPU use and missing restored address. The stand-ins check
that every graph is destroyed while its backing still exists.

The first CPU harness mislabeled an MTP payload error as `verify-error`;
its prefix dispatch selected the verifier test instead. That harness assertion
is retained in [cpu-harness-routing-control](cpu-harness-routing-control/record.json).
The corrected harness uses `payload-error`. It is not a production GPU failure.

[Six controller cases](protocol/record.json) run the real serving checker
against CPU-only children. They cover normal exit, an unexpected exit code,
an unrequested termination signal, differing repeated output, a protocol
timeout and a child that ignores QUIT/SIGTERM. Startup and shutdown each
produce more than a pipe's capacity. Raw stdout is written as it arrives;
timestamped events and partial results survive every failure. Graceful QUIT
drains stdout before waiting. Escalation records TERM/KILL separately and all
waits are bounded. The signal control uses SIGTERM, not a GPU process crash.

[The build](build.log) exits 0. The frozen candidate SHA-256 is
`2b8323a218b6e7cd2f401353da6abb6b72a28d27c1bd9b40de4be60749793b8d`.
[record.json](record.json) lists source digests and the private frozen binary.
CPU tests cannot prove a backend residency hypothesis, safe queue progress,
model parity, full-context operation or the benefit of graph recreation.
