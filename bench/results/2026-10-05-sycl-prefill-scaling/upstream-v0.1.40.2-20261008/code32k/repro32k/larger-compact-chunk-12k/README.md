# Larger compact chunk: rejected 32K control

This uses the unchanged private e82fc5 executable on Arc B570, 10 GiB VRAM,
Ryzen 5600X and 128 GiB RAM. Input stays at 32,768 tokens; only the compact-hc
chunk grows from 8,192 to 12,288. Main cache and MTP decode-only payloads
restore from immutable RAM, with complete payload checks. Context is 33,024,
normal MTP is 4, all 32,767 prefill residual rows stay on GPU, and 128 main
expert slots use the matched 64 MiB segmented allocation.

The [offline accounting](accounting/v2/record.json) extracts actual sizing
helpers. Its 8K accounted/shared sizes match the executed log after the
owned single-ring padding correction. At 12K the accounted workspace is
2,738,729,216 B, and reused residual scratch reduces the new GPU-row storage
to 838,819,840 B. The engine's page-budget estimator gives a net increment
of 708,837,376 B over 8K. Those page estimates are not a measurement of SYCL
physical granularity. The 16K increment is 1,417,674,752 B; it is not run.

The logged/validated 32K check exits normally, receives 64 outputs, performs
owned cleanup and records no new xe fault. Main and MTP payload checks pass,
but the mathematical gate rejects 12K: 63 of 66 main-state parts differ;
all 248,320 first-head floats differ (max absolute 0.873567, RMS 0.180161).
The first generated ID differs at zero-based index 2, and logprobs differ
at index 0. MTP counts are 38/75, versus the accepted 8K reference's 43/66.
This is a normal engine exit with a rejected correctness gate, not a GPU
hang. The later [exact-word health check](health/record.json) passes.
No clean timing follows and diagnostic durations are not speed evidence.
The changed chunk/GEMM/callback boundaries have not been isolated as the
cause; neither a rounding-only explanation nor a code bug is established.

The [allocation analysis](analysis/record.json) records a 921.152 MiB drop
in reported free VRAM between prefill entry and restored weights. During
main restoration plus graph warm, the main mapping grows by 384 MiB, while
reported free drops by 1,221.324 MiB. Only 1,114,112 B of new device USM is
requested in that interval. Native logs show 680 new command lists and 61
new modules; all device USM and native modules/kernels/command lists are
destroyed at process exit. This does not isolate runtime-private heap,
pooling, allocation rounding, or prove a leak.

The [source review](review/serve-first-capture-memory.json) verifies that
this serve path does not call verifier warm at startup. Released arms warm
all legal window sizes at cache restoration; a kept-fraction-zero arm
captures on first use. Consequently the restoration interval includes
initial graph capture and kernel loading in these fresh processes. The
matched cache-release means include those different capture schedules;
do not attribute decode variation solely to transferring or releasing
cache bytes. Repeated recapture needs a separate warm-process comparison.

The [output-head lifetime review](review/output-head/record.json) retains
source facts for a possible later phase lease; no head lease is implemented.
Any future lease must retain immutable GGUF bytes and restore output backing
before verifier capture, including partial-failure cleanup.

Full 262,144-cell occupancy/repeat/restore/clipped-tail/refusal/later-valid
gates remain mandatory and open. The older repeated-prefill capacity failure
is not resolved by this 32K observation. Production binaries/defaults remain
unchanged. Raw API/state/head payloads stay private with exact hashes; no
reset/rebind/reboot/service/package/global change occurs.
