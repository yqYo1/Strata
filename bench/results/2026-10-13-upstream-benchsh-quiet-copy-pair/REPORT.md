<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Quiet upstream bench.sh COPY comparison

Main completed both arms using the unchanged pristine upstream v0.1.43 client,
7695859, with `bench.sh --url http://127.0.0.1:18095 --long --runs 3`.
Arc B570 / Ryzen5600X, source410904c/freeb54579b, source-built DPC++7.1.1,
UR Level Zero v2, DirectSubmission0, native IQ3_S, chunk8192/group16,
workers5/tasks18, cache128/effective144, no borrow/MTP/prompt reuse,
int8 KV resident32768/context262144. Validation and tracing were disabled.
The same binary/client/settings ran COPY0 first, then COPY1; the only arm setting
is STRATA_COPY_ENGINE. Actual COPY1 group1 selection was observed before bench.

| COPY | Run | Actual input/read | Output | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 32049 | 24 | 407.5 | 9.7 |
| 0 | 2 | 32049 | 24 | 408.0 | 8.6 |
| 0 | 3 | 32049 | 24 | 403.0 | 7.9 |
| 1 | 1 | 32049 | 24 | 478.0 | 9.3 |
| 1 | 2 | 32049 | 24 | 470.8 | 8.6 |
| 1 | 3 | 32049 | 24 | 474.8 | 8.7 |

Every row read the whole32049-token prompt, with no reuse and24 generated tokens.
Prefill medians are 407.5 and 474.8 tok/s
(+16.52% in this campaign). Decode medians are 8.6 and
8.7; ranges are 7.9-9.7
and 8.6-9.3. Keep phases separate:
three short decode continuations with varying stock tags/histories cannot settle
a small decode effect or prove sustained throughput. No optimization adoption is
claimed by this report. Earlier validation/trace timings remain diagnostics.

All official requests succeeded. Each arm passed fresh GPU health/fault checks
and final source, binary, model and provider identity checks. Original QUIT was
written/flushed, engine/server exited0, owned anchors were reaped with empty
sessions, and no fallback/reset/service action occurred. The complete COPY1
upstream result is byte-identical; COPY0 is retained in the preceding report.
Actual roughly32K stock rows are eligible for the declared stock comparison;
historical strict32768 fields remain false. Generated-text identity is not a
universal quality test; quality/full physical262144 remain unqualified.

Registry v185 contains 894 completed reports. The new separate
scratch worktree implements default-OFF lifetime aliases while retaining FP32
raw y and its normalizer. Three source files are archived for durable review;
no candidate build, parity, physical fit or performance is claimed. Main owns
all such checks after model closure. Phase-cache CPU/ASan/server successes from
the preceding report do not establish real SYCL VMM validity. R360/R362 examine
uncertain-map failure handling before its first GPU admission. R361 reviews
existing decode counters without summing overlapping intervals into wall time.
Its prefill numbers labelled ms are TPS, and38293 is a shell session identifier.
R362 did not find the existing tests/core/phase_cache_owner_test.cpp; Main's
preceding report preserves its actual Release/ASan successes. A CPU mock cannot
establish provider failure atomicity, so the API-contract question remains.
Main implemented fail-stop retention for unknown map/unmap state, then compiled
and ran the corrected CPU owner in Release/NDEBUG and host ASan/UBSan. Both
passed, including11 child-process failure cases checking no retry/unmap/free
after uncertainty, retained ownership and rejected readers/requests. The first
failure remains a failure; removing a mock fault cannot authorize recovery.
Source updates and current CPU receipts are preserved separately from previous
passes. No real-provider map-error injection or SYCL/model qualification follows.
The explicit server child-env implementation matches Popen inheritance; the
suspected empty-dictionary inheritance mismatch was not found.

Next: judge prefill and decode separately, retain all first samples, and repeat
matched quiet decode campaigns if needed. Serially build/review candidate source
with sourceLLVM free/contrib and test bounded lifetime/state/actual memory gates
before model use. Successful full polling/API/journal duplicates are retired
only after verified extraction, keeping original pinned receipts/results.
