# IQ4NL component profiling attempt

Source 9c28dec59b04b3d92e1dc8a4862103089f908f25. Incremental build completed two steps in 10.905812498 seconds; the existing kernel library and parity binary retained exact hashes. Original builder accidentally stored parity bytes/hash beside the service path. That record is preserved; separate offline identity revalidation established the actual service binary before any GPU execution.

Logged qualification passed: 512 exact-event receipts, 132 batches, 768 wrapper calls, 268 waits, full independent-RNE pre/post replay on 64 inputs, guards and immutability. The native trace contains 384 generic and 384 private launches, all 6400 groups / local32, 512 START and 512 END queries, 1024 native kernel timestamp queries; only successful UR/ZE results.

The clean service process failed during untimed prephase batch3, before any timed cell. Raw command_submit=301019766336958, command_start=301019766335625, command_end=301019766364687: start is 1333 ns earlier than submit. Metadata/receipt counts were valid and completion known. Normal exit1; owned session empty and reaped; no new GPU fault or dump. Preserve FAILED status. No retry, offset correction, relaxed order validation, speed result or adoption follows.

A separate host-wall-only fixture will measure inclusive submission-to-known-completion time without event profiling; it is a different metric and protocol. It cannot establish pure device service. This original experiment and validator remain unchanged.

Original protocol used four cyclic orders repeated twice and full-cohort D2H readback after each cell. These balance positions while confounding predecessors and condition cache history. Results would concern instrumented direct helpers only, not cold-cache behavior or model speed. Actual model minimum32768 batched positions, full262144 lifecycle and independent repeated decode remain unqualified.
