# Integrated dequant launch-property check after upstream refresh

On B570 10 GiB / Ryzen 5600X / 128 GiB RAM with the recorded oneAPI, NEO and
Level Zero versions, a private candidate replaces exactly one kernel archive
object in the updated ceff2d8a control. Its SHA is 9f43b89e…02766ecf. Exactly
the iq_dequant_f16 and iq_dequant_gu_f16 wrapper property lists drop
use_root_sync. Kernel names, formulas, ND-ranges, device capability checks,
other objects, phase waits and all production source/link inputs are retained.
The candidate source hash a5ece3f8 matches the earlier 144-pair actual-wrapper
check; this archive adds integrated model checks after shared-header refresh.

The first fully Level Zero/UR logged context-128 normal-MTP process completes
all four requests with exact IDs, every logprob and all 248,320 first-head
floats versus the updated control. Six MTP release/restore pairs complete;
normal exit and owned cleanup succeed with no new xe fault.

A following pair disables API logging and validation, retaining the same
phase waits/progress and context-4096/int8 KV/layer-major-1 settings. Each
binary receives one 2048-token request split into two 1024-token chunks,
followed by four normal-MTP outputs. The updated control's prompt is 35.8242 seconds, versus 35.6255 seconds in the candidate (-0.555% time change, 1.00558x).

These are single samples with synchronization/progress still present, not
unsynchronized throughput or repeated statistical evidence. See the
[assessment](assessment.json) and terminal receipts for exact whole-head,
ID/logprob and exit/cleanup results. Large diagnostic logs and whole heads
remain private with hashes and bounded tails. No GPU reset, service change
or production kernel change was performed. The candidate remains unadopted;
repeated full262144-cell CLI/serve, clipped-tail, refusal and later-valid gates
are incomplete. PP1000/TG70 is not established.
