<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Upstream bench.sh phase OFF control

Unchanged upstream v0.1.43 bench.sh --long --runs3 completed on Arc B570 /
Ryzen5600X, free/source-built DPC++7.1.1, native IQ3_S, source0f112782.
Phase cache and native COPY were OFF; chunk8192/group16, workers5/tasks18,
no borrow/MTP/reuse, int8 KV resident32768/context262144. Tracing and validation
were disabled during performance measurement.

| Run | Actual input/read | Output | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- | --- |
| 1 | 32049 | 24 | 384.8 | 7.3 |
| 2 | 32049 | 24 | 383.1 | 7.8 |
| 3 | 32049 | 24 | 386.9 | 8.6 |

Each row read32049 inputs with zero reuse and generated24 tokens. Prefill median
is 384.8 tok/s and decode median is 7.8 tok/s.
Short chat and multi-turn results remain standard-suite supporting results.
All requests succeeded; original QUIT, exit0 and owned process closure passed.
Source/binary/model/provider identities matched before and after with no new
GPU fault, reset, service change or retry.

This is one OFF control campaign, not an ON effect or adoption claim. Assess
prefill and decode separately and repeat unclear effects. The stock approx32K
input/24-output case does not qualify full physical256K, sustained decode or
quality. Exact generated prose identity is not a universal quality condition.

Registry v189 adds5 research reports. R376's incorrect mode0 assumption is
preserved together with corrected v2 and Main's errata. Full-context host KV is
authoritative, with a bounded resident GPU page cache. Requested source sizes
are distinct from free-memory telemetry. R377 optional hash diagnostics, R378
longer benchmark settings and R379 full-context restore checks are source-only
research. Closed duplicate logs retire only under the verified deletion manifest.
