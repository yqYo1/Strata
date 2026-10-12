<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Repeated32K-input,128-token worker comparison

Arc B57010GiB, Ryzen5600X, free build/sourceLLVM7.1.1, same qualified binary,
ordinary persistent serve, noMTP, segmented handoff, int8KV32768resident,
context262144,144profile-filled expert slots,18tasks,8192prefill, greedy.
Nine fresh-process samples complete the predeclared5,3,4,4,3,5,3,5,4order.
Each worker count has three individual samples, retained without outlier removal.

| Sample | Workers | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- |
| 1 | 5 | 391.515 | 3.9993 |
| 2 | 3 | 388.948 | 7.7539 |
| 3 | 4 | 375.444 | 8.4747 |
| 4 | 4 | 373.208 | 9.4521 |
| 5 | 3 | 388.954 | 9.0246 |
| 6 | 5 | 390.216 | 9.2385 |
| 7 | 3 | 388.454 | 9.2103 |
| 8 | 5 | 388.690 | 9.3186 |
| 9 | 4 | 392.385 | 9.4080 |

All nine requests are complete, finite, normally closed with QUIT, observed and
reaped with empty owned sessions, fresh health/provider checks, unchanged boot
and no relevant kernel/device fault. All reported128token/LP5 fingerprints happen
to match. Quality remains unevaluated: neither equality nor differing free-running
strings are general quality acceptance. Input/binary/source/env/commands and
individual times stay pinned in the receipts. Startup is separate.

The complete matrix still has decode variability. summary.json records pooled,
mean/median/range and sample variation, with within-block paired time ratios.
No worker setting is adopted or claimed optimal. Prefill/decode remain separate
decisions; a fast phase does not justify a regression in the other phase. The
slow first worker5 sample remains valid evidence, not silently discarded.

R332 uncovered ambiguous rawUR borrower cleanup if SYCL queue construction
throws after native import. The new copy packet fail-stops before owners unwind
and before a live imported queue can be destroyed on insertion failure. It does
not repair or qualify URinternals. Shared production CPU failure seams and the
tinyGPU source still need Main build/lint/sanitizer/provider/role/event checks
before any model treatment. DefaultOFF remains the proposal's default.

R333 identifies Gemm lazy tc_w/tc_x buffers outside Prefill::owned/bytes_needed;
reset destroys that owner after drains. Actual freed bytes need measurement.
The phase-cache owner packet implements checked actualVMM storage/reader
bookkeeping and fault fixtures but is not wired into persistent serve yet. A
separate caller implementation is in progress; no cache-growth engine, speed
gain, quality, RESTORE or full262144-input result is claimed.

New source-only unified patches and hashes are archived for durable review.
Raw logs and duplicate inputs for the five newly closed samples are retired only
after useful evidence is verified. Original private receipts remain byte-identical
for current provenance/prefix consumers. Source packets/model/state fixtures stay
under named finite consumers; the retention manifest records every deletion.
Next: isolated free/contrib-llvm build, CPU ownership/sanitizer checks and small
debugGPU qualification for the native copy queue; then quiet>=32K comparison.
The1000prefill/70decode objective and optimized official/fullcontext/quality
comparisons remain open.
