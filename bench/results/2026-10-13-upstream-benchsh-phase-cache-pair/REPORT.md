<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Upstream bench.sh phase cache comparison

Unchanged upstream v0.1.43 bench.sh --long --runs3 ran OFF then ON on Arc B570 /
Ryzen5600X, free/source-built DPC++7.1.1, native IQ3_S. Binary/argv/model/client
match; the intervening source commit changed method documentation and reports.
COPY0, chunk8192/group16, workers5/tasks18, no borrow/MTP/reuse, int8 KV
resident32768/context262144. Tracing/validation were disabled. The only changed
setting is STRATA_PHASE_DECODE_CACHE.

| Phase cache | Run | Actual input/read | Output | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 32049 | 24 | 384.8 | 7.3 |
| 0 | 2 | 32049 | 24 | 383.1 | 7.8 |
| 0 | 3 | 32049 | 24 | 386.9 | 8.6 |
| 1 | 1 | 32049 | 24 | 362.4 | 7.1 |
| 1 | 2 | 32049 | 24 | 375.4 | 8.2 |
| 1 | 3 | 32049 | 24 | 383.0 | 9.1 |

Every row read32049 tokens, reused0 and generated24. Prefill medians are
384.8 and 375.4 tok/s (-2.44%).
Decode medians are 7.8 and 8.2 (+5.13%),
but ranges 7.3–8.6 and
7.1–9.1 overlap. ON values rise in
run order, and24-output decode continuations vary. No decode benefit or adoption
is established; retain default OFF and repeat matched arms before that decision.
Assess prefill and decode independently.

Both campaigns passed every request, original QUIT, exit0, owned-process closure
and before/after source/binary/model/provider checks with no fresh GPU fault,
reset, service change or retry. This stock approx32K case does not qualify full
physical256K, sustained decode or quality. ON upstream results.json is retained
byte-for-byte; OFF is in the preceding report. Chat/agent are supporting results.

Registry v190 adds7 research reports. Input activation quantization is already
shared across experts; no redundant cache is proposed. A separate source-only
head-lease/FP32 scratch-alias/explicit12K candidate is archived as deltas; build,
runtime, fit and numerical quality are unqualified. Main owns checks after this
measurement. Production fixed-history logits capture and later gated overlay /
Dm streaming also need qualification. Exact prose identity is not a universal
quality gate. Closed duplicates retire under the verified deletion manifest.
