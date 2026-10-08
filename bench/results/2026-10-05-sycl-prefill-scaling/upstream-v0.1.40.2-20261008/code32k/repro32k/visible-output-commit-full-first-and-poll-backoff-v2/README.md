# First full-capacity image and a CPU polling candidate

This archive freezes two completed portions of the still-running capacity
controller: the first full read and the following fresh32K control. It does
not contain the controller's eventual terminal receipt. The same ArcB570
10GiB / Ryzen5600X /128GiB host and pinned runtime are used throughout.
All performance comparisons require at least32768 input tokens; these
logged correctness checks provide no speed measurements.

The first262140-token input generates four visible outputs with RESUME0.
Verifier windows reach physical cell262143. SAVE contains exactly262143
consumed IDs, equal to the prompt plus visible outputs except the last.
The4239591088-byte image contains all12 main layers and the MTP layer,
each with262144 physical cells. No saved tensor byte is excluded. The
following fresh32768-token input matches the accepted control's complete
head, used prefill state,64 output IDs/logprobs and visible MTP41/66.
See [the completed-portion snapshot](full-first-v4-snapshot.json).

At capture, the second full input was active. Repeated full-image equality,
actual disk restore, clipped-tail parity, capacity refusals, later32K parity
and owned exit/fault checks remained pending. The candidate is not adopted.
Large head/state/session files remain private with their hashes in
[private-artifacts.json](private-artifacts.json).

Separately, a private executable changes only the prefill host wait policy:
the first32 failed readiness checks yield; later failures request a10us
host sleep. Readiness queries, ownership generations, acquire/release,
cancellation priority, the five-minute budget and timeout exit behavior
remain unchanged. Only prefill.cpp.o is replaced; the actual dependency
file proves that its shadow header was selected. Compilation and CPU
ASan/UBSan checks pass for immediate readiness, cancellation and timeout
without unsafe cleanup. The candidate has not run on the GPU.

Synthetic query counts are not CPU utilization, DMA or inference timings.
The runtime event-status query can enter UR/LevelZero; a host deadline
cannot preempt a blocked query. The public header comment is corrected
without changing behavior. Compiled original and candidate headers, build
and source-review receipts, dependency evidence and controllers are retained.
See [summary.json](summary.json) for the capture scope and pending gates.

Production and accepted binaries remain unchanged. No reset, rebind,
reboot, service, package or global setting is changed.
