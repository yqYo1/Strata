# Experimental layer-major prefill with streamed INT8 KV

Existing SYCL layer-major prefill refused streamed KV. This experiment admits
plain INT8 streamed KV through `STRATA_PREFILL_LAYER_MAJOR_STREAMED_KV=1` together
with the existing layer-major selector. It keeps one existing full-capacity
identity stage for each QSA layer's whole chunk loop: seed the resumed prefix
once, then reuse the same layer's appended prefix across chunks. Host KV remains
authoritative. Resident mappings, indexers, GDN/conv, PLE and all existing
append/attention arithmetic stay in their current paths.

Admission checks the single-GPU compact path, full stage geometry, buffers,
capacity and in-order compute queue before changing resources or session state.
The mode and local stage extent do not enter SAVE/RESTORE. Competing next-QSA
prefetch is disabled only inside the opt-in. Partial-page seeds copy valid
cells only; the current INT8 geometry has 1056 bytes per cell. That is a
calculation, not a measured copy rate or allocation peak.

A separate gpt-6.1-sol agent wrote code only. A read-only gpt-6-luna agent then
reviewed the submitted source. The root agent moved primary-QSA access into
the opt-in branch to preserve zero-QSA/default behavior, and disarmed the error
compute-drain fallback after the existing successful compute fence. This removes
the extra successful per-layer wait identified in the independent review. The
original reports are kept and refer to the submitted source; root-source-review
pins the corrected source.

This branch is unqualified: no compilation, CPU tests or GPU execution have yet
been run on this change. No speed, memory-fit or full-context claim is made.
Root exclusively schedules all builds/tests after the active 32K comparison.
Before adoption it needs exact prefix/owner checks, logged state/logit/LP/MTP
comparisons, resumed/over-resident/prefetch/cancellation cases, physical 262144
cells and SAVE/RESTORE/capacity boundaries, then separate repeated quiet prefill,
fresh decode and restored decode comparisons. Context capacity is not shortened.
