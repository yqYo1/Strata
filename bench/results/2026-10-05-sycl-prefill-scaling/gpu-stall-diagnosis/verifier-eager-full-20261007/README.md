# Eager verifier full-context comparison on 2026-10-07

On the same B570 10 GiB / Ryzen 5600X / 128 GiB host and boot, the unchanged
3f3e executable with the existing eager-window option completes a normal-MTP
request with 262,140 input and four output tokens. Windows [262139,1],
[262140,4], [262141,3] execute through the last KV cell 262143, and one
verified MTP release/restore pair completes. All printed logprobs and all
248,320 head floats are finite.

The four IDs [40,3172,1151,539] match the earlier lazy candidate, but all
248,320 first-head floats differ, with maximum absolute difference
0.412168025970459. The eager head hash is 26eb4436…992b720a, versus
ce388272…86f7516 in the lazy run. All four logprob lines differ. Their
same-setting 2K comparison had matched, which did not establish full-context
parity. The source of the full-context difference remains unresolved.

The strict comparison stops the controller before the second request. It
removes both owned processes; no new xe fault is observed. The following
logged H2D/kernel/D2H probe passes all three 16,384-word rounds without a
reset. This termination is a comparison failure, not an observed GPU hang.
The repeated-prefill, clipped-tail, overflow-refusal and later-valid gates
remain incomplete. The original eager path is not adopted as a full-capacity
fix. Diagnostic durations are not clean throughput.

[Assessment](assessment.json), [terminal record](full-run/record.json),
[memory checkpoints](memory-checkpoints.txt), health receipt, source/controller
hashes and bounded log tails preserve this result. Full heads and large logs
remain private with recorded SHA-256 hashes.
