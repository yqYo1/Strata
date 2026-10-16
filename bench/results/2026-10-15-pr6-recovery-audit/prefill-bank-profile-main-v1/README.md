# Main source-only canonical bank and coarse profile combination

Immutable base: `9177ad4fb7cc1587255db7235be474aca3eb4e2a`.
Unapplied, uncompiled and untested. Direct measurements used the unchanged 8fdc
reference binary and unchanged product source. Their cohort stopped at its first
failure: explicit VRAM headroom admission, engine exit1, no fresh fault/dump and
all owners retired. Three samples passed; the six-sample performance comparison
is incomplete and no setting is adopted. No build, test, GPU operation or artifact
cleanup occurred during this assembly.

`assemble_prefill_bank_profile_v1.py` reconstructs the independent own-fork
canonical bank v3 and coarse profile v2 proposals from pinned Git blobs in memory.
It checks every canonical/profile base hash against the immutable base and each
hunk against one exact context. It never writes complete existing-source mirrors.
XeStrata is a performance reference only; no XeStrata source was used or restored.

The two explicit context merges are recorded in pins.json. The entry timer ends
after canonical enter and persistent PLE rebinding, and the run timer starts with
the combined native/canonical private overlay callback intact. No fuzzy hunk or
replacement from stale full source is used. Main reviewed both complete input
diffs and these two merged seams. This is source review, not runtime evidence.

The 15-file diff includes four manual EXCLUDE_FROM_ALL targets. The host policy
test does not establish allocation correctness. The canonical component uses the
actual loader/owners, checks exact PLE field pointers and all canonical payloads,
and exercises Prepared cancellation. Its payload oracle shares loader metadata;
it is not an independent GGUF metadata parser. Separate canonical allocation and
native reservation fault targets exercise confirmed OOM rollback and uncertain
allocation fail-stop. Existing native payload fixtures remain necessary. None
has been compiled or executed for this candidate.

After the audited first-failure cohort closure is recorded, Main may check/apply
this exact diff, then use build_prefill_bank_profile_v1.py for a fresh ABI build.
The script permits compilation after that specific closed preallocation failure;
it does not turn the failed performance comparison into a pass. It refuses missing
audited closure or source hashes.
Before device use, qualify profile owner/error/output bounds and the actual host
and component contracts, including the already-retired event timestamp APIs.
Retain the existing safety environment before exec; never change process
environment after runtime initialization. All builds/tests/GPU/cleanup remain
serial under Main's measurement lock, with finite supervisor budgets.

Both new flags stay default OFF. Short warmup/short prefill still use their normal
layout with PROFILE=1. No speed, bottleneck, task-quality, full physical 262144,
RESTORE or graph-lifetime qualification is claimed by this source artifact.
Stage-specific inference and safety decisions remain separate from source
integration. Required real model and fault gates are described in the canonical
v3 and profile v2 recipes and Main's review report.
