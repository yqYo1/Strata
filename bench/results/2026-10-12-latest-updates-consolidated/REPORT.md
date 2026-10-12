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
view, which is the next source change. Windows child containment applies to
the Windows upstream; the selected XeStrata platform is Linux. Full physical
262144-token input, RESTORE and optimized pristine parameters remain unqualified.

Source and commands, exact environment, ownership, individual test results,
source equivalence and remote identities are retained in the receipts. All
existing failed qualification statuses remain in their original reports.

Earlier model evidence: [pristine update results](https://github.com/yqYo1/Strata/blob/61b78394/bench/results/2026-10-12-latest-updates-pristine-builds-host-regression/UPDATE_RESULTS-v160.md) and
[selected tuning](../2026-10-12-decode-criticalpath-native-dequant-batch/selected-tuning.json).
