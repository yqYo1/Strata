# Expert-phase pacing checks after unitrace profiling

Measured on 2026-10-07 on the same Arc B570, Ryzen 5600X and runtime as
the [unitrace profiles](../profiling-20261007/README.md). Production sources
and the production executable were unchanged. These are correctness and
execution checks of a private candidate, not validated throughput gains.

The private executable SHA256 is
`08cf63a3642242307372860ca6d09cc2971f8acc8f367e70d119fb8ac3d65ac1`.
Only the prefill object was replaced. The candidate can group expert-phase
waits while retaining every dense, attention, routing, combine and chunk
boundary. The setting counts phases, not experts. A value of one retains
each original phase wait. The same pacing can run without per-wait prints.
Build receipts verify that production sources, archives, objects and binary
were unchanged.

The first run used context 128, FP16 KV, 32-token chunks, layer-major mode
2, 600 cache slots and normal MTP. Phase batch 16, phase prints, Level Zero
parameter validation and flushed API logs were enabled. First, repeat,
other-input and restored-input requests all matched the released control:
four IDs, all printed logprobs and all 248,320 finite float logits for each.
Six verified MTP release/restore pairs completed. Printed phase completion
waits fell from 73,692 to 9,665; every non-expert phase count was unchanged.
The maximum positive mark gap was 16. The process exited normally with no
forced cleanup, survivor or new xe fault. Its 174.507 s duration includes
heavy diagnostic logging and is not a clean throughput measurement.

The next run used context 4,096, INT8 KV, a 2,048-token input, 1,024-token
chunks, layer-major mode 1, 128 cache slots and normal MTP. It used phase
batch **one**, with neither phase prints nor API logging, validation or a
profiler. Identical fresh requests had zero reused tokens. The first two
requests matched the prior control exactly; the third differed in IDs,
printed logprobs and the complete head. The supervisor then terminated its
owned process. No survivor or new xe fault remained. The cleanup receipt
has a null exit code because its final reap raced process exit; the PID was
separately confirmed gone. This run fails the correctness gate, so its
times are not accepted as an optimization result. Quiet phase batch 16
was not run, and this candidate was not adopted.

A repeated-input check of the unchanged workspace-reclamation executable,
SHA256 `3f3ed0526848f6c7273a70da8802cbe00389b67791629c000c26399bc50823f2`,
with its original per-phase synchronization and full diagnostic API logging
stalled during the first request at layer 35, chunk position 1,280.
The last API entry was `zeCommandListHostSynchronize`; it had no return
before the application's 60 s watchdog stopped the process with SIGABRT.
No new xe fault appeared. The subsequent bounded health probe passed three
rounds of exact H2D, kernel and D2H comparisons without a reset.

Another unchanged-executable check retained per-phase synchronization and
phase prints but removed the diagnostic API logging and validation.
All four identical fresh 2K requests matched the prior IDs, logprobs and
complete finite head and exited normally with no new xe fault or survivor.
This does not establish the cause of the quiet candidate's difference or
the logged control's wait stall. Neither a general hang-prevention claim
nor any full-context correctness claim is supported by these checks.

Receipts and executed controller/candidate sources are retained here.
Large private logs are represented by SHA256, byte count and bounded tails;
complete raw logs and binary heads remain at the recorded private paths.
