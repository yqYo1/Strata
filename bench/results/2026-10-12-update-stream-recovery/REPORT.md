<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Current upstream identities and remaining server imports

The fresh remote HEAD and peeled-tag checks resolve official main/v0.1.43 to
7695859aa13afb248bdb4d36f9b2e95707db079d and XeStrata main/xe0.1.42 to
da3d426079019daa6e01388d03b0d94ae67070b7. Both match the exact sources already
built and measured without local engine patches on 32768 input tokens.
The pristine worktrees still have those HEADs and no tracked changes.
Their measured executable and original qualification hashes still match.
The current curated source matches all 413 engine/build/setup/tool/data pins
from the previous integration proof. No new upstream engine revision was found.

The existing pristine Xe diagnostic measured 217.61 prefill and 10.41 decode t/s.
Official cache128 measured 19.07 / 6.29 and cache512 17.77 / 8.48.
These are individual diagnostic samples, with validation/tracing enabled,
different compilers and different prompt routes. Official used serial windows8;
Xe used batched8192. These results do not rank optimized implementations.
The separately selected curated recipe averaged 376.42 / 8.27 over three clean
32K samples; it is a different source/binary and remains a separate result.
See the [original update evidence](https://github.com/yqYo1/Strata/blob/61b78394/bench/results/2026-10-12-latest-updates-pristine-builds-host-regression/UPDATE_RESULTS-v160.md)
and [selected tuning](../2026-10-12-decode-criticalpath-native-dequant-batch/selected-tuning.json).

The renewed functional audit found four remaining shared server gaps despite
the unchanged remote versions: an answer reserve when limiting reasoning,
a short-period reasoning-tail detector, keepalive comments while a running
stream withholds tool arguments, and CPU/I/O-aware monitoring of a silent engine.
Main independently checked all four against the pinned official source.
All four are now implemented and passed Main's host qualification.

The early stall monitor must retain the normal silence allowance when GPU
activity is unavailable or cannot be read. Missing telemetry is not evidence
of an idle GPU. Only the service's own engine process may be ended; no GPU reset,
unbind, other process adoption or service operation is part of this port.
The display gauge's zero is also unknown: Xe can report zero when it observed
no client rows. Current real telemetry therefore supports warning/observation,
and does not authorize early termination. The decision path remains separately
tested with an explicit qualified-idle mock callback. Slow sampler publication
does not refresh old observations; sample age starts before the read.

## Completed shared server qualification

The new focused CPU suite passed all 15 cases, including budget boundaries,
periodic reasoning, actual held tool arguments, process/counter identity,
missing or zero GPU readings, slow telemetry and STOP/DONE cancellation.
The complete fake-engine/HTTP suite ran 581 tests: 570 passed and 11 existing
optional-dependency skips. The preceding unchanged baseline had 566 tests:
555 passes and the same 11 skips. No real engine/model or GPU ran in these checks.

Mandatory Python, spelling, secret and license checks have no new findings.
The original first lint receipt remains FAILED for a new nullable-process type
finding. Main fixed the explicit liveness guard and independently rechecked it.
All engine/build/setup files remain unchanged in this shared-server port.
No speed gain is attributed to the server tests or to the reused diagnostics.

R302 additionally identifies opt-in multi-GPU replica serving and Linux physical
CPU grouping. These are being adapted in a separate worktree. Windows-only
prevent_sleep has no Linux effect. Replica functionality must receive its own
Main-owned host/mock qualification; it is not covered by the 581-test result.

## Research corrections and remaining qualifications

R301 incorrectly calls B570 an NVIDIA GPU. The target is Intel Arc B570.
Its references to an older corrected 8K executable do not replace the exact
latest pristine 32K receipt: that run completed 16 output tokens, normal QUIT
and fresh fault checks. Its single diagnostic still cannot establish a clean
repeated speed or mathematical-equivalence comparison.

R298 incorrectly describes DONE field13 as integer bytes. Current generate.cpp
prints file-read bytes divided by 1e6 with one decimal place, so a nonnegative
fractional value is valid. Main also found a separate GEN128 admission defect:
the old leaf-service build proof pins nine changed files and does not contain
generate.cpp; the proposed direct lookup would raise KeyError before GPU launch.
Neither proposal nor fixture has been admitted for a long model run.

R299 records why ordinary pristine official batched prefill is excluded when
host-USM atomics are absent. Its conditional single-chunk/fully pinned route
needs actual effective chunk and complete pinned-source receipts before a
request. A CLI setting alone is insufficient. The optimized-official-parameter
TODO remains open. Physical 262144-token input, RESTORE, full-vocabulary logits
and whole-model task quality remain unqualified.
