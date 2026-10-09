# Standalone streaming native CPU calibration

Root built the source-only Sol handoff in 24.255s with the same standalone
CPU-only closure and per-TU flags as the qualified smoke tool. This receipt
covers compile/link/import checks only: no 384-ID cohort, statistical gate,
performance result or model inference was run by this build.

R81 identifies an omitted metadata setting, STRATA_IQ_PREFETCH. Root launchers
already use a recorded minimal environment, so this is not evidence of a
nondefault existing run. Record the setting in tool output before comparing
future NT2 timing profiles. Keep all earlier source/build/research statuses.

Root added the environment metadata field, rebuilt incrementally in 4.937s and
re-audited the full target compile/link closure. Both build receipts passed,
with normal exits and no owned survivors. The original v1 source remains in
commit 64f189c6 and the original build receipt remains unchanged. No kernel,
task policy or production engine source was changed; neither build executes
the 384-ID streaming cohort or qualifies timing, statistics or full context.

The two receipts, target commands, actual compile flags, direct imports and
root supervisors are retained as compact reproducibility evidence. Full
tracked-file listing is redundant with each receipt's hashes and is not copied.
No build binary or model payload is committed.
