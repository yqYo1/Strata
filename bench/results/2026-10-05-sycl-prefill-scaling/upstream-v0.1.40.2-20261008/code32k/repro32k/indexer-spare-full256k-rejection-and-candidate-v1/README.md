# Full-capacity repeat and the rejected disk roundtrip

On the ArcB57010GiB / Ryzen5600X /128GiB host, two fresh262140-token inputs
with four outputs reach physical cell262143. A different fresh32768-token
input separates them; both full reads report RESUME0. Their complete heads,
used prefill states, output IDs/logprobs and MTP counts match. SAVE records
262143 consumed visible-prefix IDs and all13 main/MTP KV layers through
262144 physical cells. Every saved state/KV byte matches between the two
full reads, including the last physical cell; no tensor byte is excluded.

Actual RESTORE of the earlier32K saved session succeeds and reproduces all
64 continuation IDs/logprobs and the complete head. Actual RESTORE of the
full session passes engine checksum/compatibility validation, but its next
SAVE differs from the original. Each main layer changes only its last
512-byte pooled indexer row65535. All other pooled bytes and all other
state/KV parts match. The restored row equals the live idx_dead key; the
original saved row does not. See the exact file-offset and byte evidence
in [the source review](checks/indexer-spare-commit-v01402-source-review-v1/record.json).

This is a rejected full-capacity correctness gate. The process receives
QUIT, exits0 normally, needs no forced cleanup and produces no new kernel
fault. Restored clipped-tail parity, capacity refusals and later32K input
are not reached after the rejection. The separately retained read-only
health check passes. See [the terminal receipt](full256k-r4/record.json).
The earlier [first-image archive](../visible-output-commit-full-first-and-poll-backoff-v2/README.md)
remains a historical partial snapshot, not a claim that this sequence passed.

The verifier replays accepted indexer positions and marks the others -1.
When the speculative window completed a four-cell block but commit stops
before its end, that completed block key can remain in the live spare row.
The private candidate tracks the final valid position in the existing
append kernel and copies idx_dead into its corresponding spare after replay.
It leaves completed blocks, arithmetic, local memory, existing barriers,
subgroup32, launch geometry, queue order and root properties unchanged.
Only native_qsa_indexer.dp.cpp.o is replaced in the accepted kernel archive;
all other archive members and link inputs are verified unchanged. The host
polling candidate is separate and is not included in this executable.

Compilation/link and the source/byte audit pass. The separate r5 controller
passes its initial fresh32K gate: complete head, used state,64 IDs/logprobs
and visible MTP41/66 match the accepted control. SAVE contains exactly32831
consumed visible-prefix IDs, and every main spare equals idx_dead. A live
32832-token continuation reports RESUME32831. See the later
[initial32K snapshot](initial32k-v5-snapshot.json); the timestamped summary
predates this completed portion. The parent is still running its first full
read. Repeated-full, actual disk-roundtrip, clipped-tail/refusal and later32K
gates remain pending; the candidate is not adopted.
Compiled original/candidate source, complete diff, controllers, dependency
file and receipts are retained. Large payloads and binaries remain private
with hashes in [private-artifacts.json](private-artifacts.json).

All performance comparisons require inputs of at least32768 tokens and
separate initial from repeated full reads. These logged correctness durations
are excluded from speed. No reset, rebind, reboot, service, package or global
setting is changed. Production and the accepted binary remain unchanged.
