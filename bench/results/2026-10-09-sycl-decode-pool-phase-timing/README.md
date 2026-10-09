# Serve CPU pool phase diagnostics

This candidate starts from the integrated v0.1.41 pool-tasks branch `a2c8846`.
Under the existing `STRATA_DECODE_TIMING` gate it differences existing cumulative
GU, intermediate FF-quant and Down phase counters at request boundaries.
A separate flushed stderr line reports host phase milliseconds per verification
window. These overlap GPU work and must not be added into serial latency.
Packed expert-blob submitted bytes are accounting, rather than DRAM traffic.
With zero windows the line reports raw deltas and no per-window division.
Existing stdout protocol and timing/profile formats are preserved. No new
timers, clocks, atomics, waits, queues or numerical operations are introduced.

Root source comparison verifies that removing the diagnostic changes restores
the entire original generate source. Compilation and runtime qualification are
pending; there is no performance claim or adoption. Tests/builds are managed
only by root. See [source proof](source-review-v1.json) and
[implementation report](implementation-report.txt).
