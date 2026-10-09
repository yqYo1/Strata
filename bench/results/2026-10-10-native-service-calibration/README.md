# Standalone streaming native CPU calibration

Root built the source-only Sol handoff in 24.255s with the same standalone
CPU-only closure and per-TU flags as the qualified smoke tool. This receipt
covers compile/link/import checks only: no 384-ID cohort, statistical gate,
performance result or model inference was run by this build.

R81 identifies an omitted metadata setting, STRATA_IQ_PREFETCH. Root launchers
already use a recorded minimal environment, so this is not evidence of a
nondefault existing run. Record the setting in tool output before comparing
future NT2 timing profiles. Keep all earlier source/build/research statuses.
