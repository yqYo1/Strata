<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Update recheck and grouped GEMM events on Arc B570

On2026-10-12 the remote main branches and release targets still match the versions
already evaluated: official v0.1.43 (`7695859`) and XeStrata xe0.1.42 (`da3d426`).
No new engine source delta was found. The prior32K unmodified-source evaluations
and selective upstream functionality import (`0691bcdf`) remain the update evidence.
The original workspace branch is not the integration branch.

The prior official settings were diagnostics, not an optimized performance baseline.
R278 confirms that gap. R277 found no new omission among the imports listed in v160;
known queued SSE keepalive/cancel and READY test closure remain to be handled.
This report does not claim complete server parity, physical262K or SAVE/RESTORE qualification.

The isolated diagnostic source now builds in free and contrib-llvm with warnings as
errors. Mandatory lints report no new source findings against d787a19b; three existing
CMake findings remain. The first build and lint failures are preserved as failures.

Seventeen finite real GPU cells passed. All selected XMX, using device-USM inputs,
weights, outputs and checked actual device bounds. Every output matched the independent
host reference; all16 tail canaries survived. Fresh kernel-journal/device checks passed,
as did the final integer health check. One contrib-llvm diagnostic cell was also run.

| Maximum expert rows | Phase | Event microseconds, three clean processes | Mean |
| --- | --- | --- | ---: |
|128|gate/up|347.292,258.854,273.333|293.160|
|128|down|210.521,166.875,165.625|181.007|
|512|gate/up|762.292,785.104,820.937|789.444|
|512|down|370.000,369.896,366.875|368.924|

Each command includes16 groups, of which13 are nonempty; total active rows are1630
or6622. The larger cells achieved53-59 useful FP16 TFLOP/s. This is a synthetic
service measurement, not the device peak or a model throughput claim. Smaller cells
varied by14-16% sample CV; all individual samples are retained. Host intervals include
cold dispatch/kernel-bundle queries and remain separate from the exact GPU event.
Initialization copies and full-output validation are outside that event interval.

The selected32K model recipe still uses0691bcdf with native dequant batching enabled;
its existing three clean samples averaged376.42 prefill t/s and8.27 decode t/s.
This diagnostic has not replaced that model binary or qualified production model output.
Next measure actual production issuing and kernel/transfer/GDN/QSA phases, then assess
larger chunks from the complete arena calculation rather than only GDN scratch.

See updates-recheck.json, grouped-event-results.json and the exact receipt index.
Research reports are preserved with their identities and Main corrections in registryv164.
