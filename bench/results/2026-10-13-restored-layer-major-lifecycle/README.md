<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Restored layer-major prefill candidate qualification

Fresh 1K and32K GEN requests completed in one process followed by normal QUIT.
All512 experts were uploaded once per each of48 layers;32K processed four chunks.
Pre/post16384-word GPU checks, kernel fault checks and full SHA checks of12 model inputs passed.

The candidate used phase0, all-RAM FP32 residuals,8K chunks, configured context262144, int8 KV resident32768 and explicit700 MiB reserve.
Each request used max_new1, raw spec4, effectiveT1 and no reuse. Actual prefill processed1023/32767 rows, excluding the final conditioning position.
The32K residual allocation was1342136320 bytes and the layer cache1363148800 bytes.
Admission used fresh reported free memory; the normal device miss ring was omitted.
Individual completion records and diagnostic times are in summary. They are not speed samples.

Free/contrib-llvm builds, four CPU CTest configurations, ASan/UBSan, server phase lifetime and actual-parent static checks passed.
The initial signedness build failure, two new static findings and reserve-option controller preflight failure retain their failed status.
Main corrected the type, redundant Impl boolean, widening before multiplication and explicit reserve option.
No checks were disabled and the reserve was not reduced.

Numerical quality, full physical256K, RESTORE, bench.sh speed comparisons and adoption remain unqualified.
Matching output IDs do not replace a quality comparison. The existing serve profiler suppresses the eligible mixer side branch, so its fractions cannot establish the quiet execution critical path.
The prior6.45 GB/s sample used Host-USM H2D; the pageable H2D control attained about4.61 GB/s. Neither directly measures this bridge.

Separate full decode-cache/head lease and typed decode route ledger worktrees remain source candidates.
Main added a two-layer/two-cycle actual-device full lease fixture and is qualifying its builds.
Next: actual lease, fixed-history numerical comparison,256K boundaries and unchanged upstream bench.sh --long --runs3.

Registryv195 references958 previous and9 new reports.
Verified structured evidence replaces duplicate successful text/source listings; a manifest records path/hash/reason.
Original hash-pinned receipts stay unchanged for current controllers. No physical storage gain is claimed.
The skill discovery link exists only under this project's.agents/skills; both user-wide paths are absent.
