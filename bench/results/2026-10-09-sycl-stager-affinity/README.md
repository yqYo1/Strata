# Prefill stager CPU affinity experiment

`STRATA_STAGER_CPU_LIST=1,2,3` places the existing RAM-copy workers on the
requested logical CPUs on Linux. The list cycles if there are more workers than
CPUs. Invalid lists or failed affinity calls fail initialization. With the
variable unset, the original worker launch is retained.

On the Ryzen 5600X, the default thread count is three (`12 / 4`). A read-only
observation during a fresh 32K prefill found three newly created threads on CPU 0,
with matching start times, while the decode pool's workers were parked on CPUs
1 through 5. Source inspection identifies the stager launch after the host has
been pinned to CPU 0. This suggests contention worth measuring; it is not a
cause diagnosis for the separate numerical failure of the resource-release
experiment.

This branch starts from the integrated v0.1.41 review head `2326895`, whose
compiled engine baseline is `1eb8948`. It changes only prefill worker placement,
without the experimental native copy queue or resource retirement. CPU expert
kernels and decoder sources remain the baseline sources.

Unqualified: compare fresh inputs of at least 32K, first logits, all live prefill
state, output logprobs and MTP counts. Assess prefill and decode separately.
Keep full 262144-cell coverage before adoption. No speed improvement is claimed.

CPU checks use the actual extracted Stager with delayed CPU DMA events. With
the test's host pinned to CPU 0, the default workers inherit CPU 0; the opt-in
workers run on CPUs 1, 2 and 3. Both configurations preserved all 6144 delayed
reads across 128 generations and repeated stager recreation under ASan/UBSan.
Six invalid or unavailable CPU settings were rejected and released cleanly.
These tests do not run SYCL, Level Zero or model inference. See
[host test receipt](host-test-receipt-v1.json) and
[source comparison](source-review-v1.json), which verifies that removing the
two Linux opt-in blocks reproduces the entire original prefill source.
