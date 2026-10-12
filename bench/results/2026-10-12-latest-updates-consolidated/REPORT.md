<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Latest updates consolidated into the tuning source

The 2026-10-12 remote fetch still resolves official v0.1.43 to 7695859 and XeStrata
xe0.1.42 to da3d426, exactly the sources already measured without local engine
patches on 32K. No new engine revision was found, so this check reuses those
individual measurements rather than run identical revisions again.

This merge joins the already qualified 501dd9ef server branch to the 78817e6d
prefill diagnostic branch. It brings queued SSE keepalives, cancellation and
READY failure handling, stable tool order, readable refusal parts, spaced tool
names and streaming parser fixes into the current tuning source.

Main verified all 413 tracked engine/build/setup/tool/data files are byte-identical
to 78817e6d and all 55 server files are byte-identical to 501dd9ef. Fresh server
regression ran 555 tests: 544 passed, 11 existing optional-dependency skips.
Fresh mandatory Python lints introduced no findings. No GPU or model ran for
this merge and no performance promotion follows from these host tests.

The prior pristine 32K diagnostics measured XeStrata 217.61 prefill / 10.41 decode
t/s, official cache128 19.07 / 6.29 and cache512 17.77 / 8.48. Official used serial
windows 8 with icpx 2026.1; XeStrata used batched 8192 with LLVM 7.1.1 free. These
logged single samples are not an optimized or controlled speed ranking. The
selected integrated 32K recipe remains 376.42 prefill / 8.27 decode t/s across
three clean native-dequant-batched samples, identified by its original binary.

The functional coverage recheck found a missing conversation-cache Monitor
view. It is now imported at 440e0851 with bounded, serialized log observation
and conservative unknown state after lost history. Its 566 host tests have
555 passes and 11 existing skips; seven actual-renderer/DOM states also passed.
See [the Monitor qualification](../2026-10-12-upstream-monitor-cache/REPORT.md).
Windows child containment is outside the selected XeStrata Linux platform.

The same-binary 32K calibration ran OFF/ON/ON/OFF in four fresh processes.
Prefill samples were 90072.2/88155.3/91066.3/88075.0 ms. Every run closed normally
without a fresh device fault; all 16 output IDs and reported LP5 matched.
Mean profiling ON/OFF difference was +0.60%, but the two pairs had opposite signs,
so the perturbation is inconclusive. Keep profiling OFF for normal speed.
Individual decode timings, conditions and selected-leaf aggregates remain in
profiling-calibration.json; no physical DMA bandwidth is inferred.

The upstream-parameter TODO is confirmed open: the old official runs explicitly
used --short-read 65536 on 32K and literal --prefill 8, taking serial windows.
They cannot rank optimized batched prefill. The source/runtime safety gate must
be satisfied before any new pristine large-batch run. The staged setting matrix
is retained in R286. Full physical 262144-token input and RESTORE remain unqualified.

Resume tuning from this combined source. The native Copy2D cohort source remains
unapplied and unverified in its isolated worktree; R287 lists the allocation-owner,
credit/publication and fault-path fixtures Main must close before GPU/model tests.
The conditional 16K alias/GDN design in R284 is a proposal, not a fit guarantee.

Source and commands, exact environment, ownership, individual test results,
source equivalence and remote identities are retained in the receipts. All
existing failed qualification statuses remain in their original reports.

Earlier model evidence: [pristine update results](https://github.com/yqYo1/Strata/blob/61b78394/bench/results/2026-10-12-latest-updates-pristine-builds-host-regression/UPDATE_RESULTS-v160.md) and
[selected tuning](../2026-10-12-decode-criticalpath-native-dequant-batch/selected-tuning.json).
