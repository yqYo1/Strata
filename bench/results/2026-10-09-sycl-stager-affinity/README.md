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

Held: the completed comparison did not establish a useful improvement.
This candidate has not been adopted or qualified at the physical 262144-cell
capacity. The compiled source is commit `ba1494d`; later commits in this branch
only archive evidence.

CPU checks use the actual extracted Stager with delayed CPU DMA events. With
the test's host pinned to CPU 0, the default workers inherit CPU 0; the opt-in
workers run on CPUs 1, 2 and 3. Both configurations preserved all 6144 delayed
reads across 128 generations and repeated stager recreation under ASan/UBSan.
Six invalid or unavailable CPU settings were rejected and released cleanly.
These tests do not run SYCL, Level Zero or model inference. See
[host test receipt](host-test-receipt-v1.json) and
[source comparison](source-review-v1.json), which verifies that removing the
two Linux opt-in blocks reproduces the entire original prefill source.

## Completed 32K comparison

Measured on Intel Arc B570 10 GiB and Ryzen 5600X with 128 GiB RAM, Linux
7.0.0-38, oneAPI compiler 2026.1.1-325 and the pinned Level Zero v2 adapter.
Both engines use the same Qwen3.8-Flash-Next IQ3_S pack, 8192-token prompt
chunks, physical context capacity 262144, INT8 KV with 32768 resident cells,
MTP spec4, expert-cache128 and no prefill cache borrowing. Host CPU 0 and five
CPU pool workers on CPUs 1 through 5 are unchanged. Three stager workers use
CPUs 1, 2 and 3 only in the opt-in mode.

The first logged comparison ran four fresh requests in A/B/A/B order with
32768 input tokens and 64 output tokens. First heads, all 66 used live-state
parts, output IDs/logprobs and MTP counts matched the qualified reference.
Full UR/Level Zero logs were retained; these instrumented times are excluded.

The quiet comparison ran six order-balanced blocks of three independent
processes: the reference engine, candidate with the option unset, and candidate
with placement on CPUs 1, 2 and 3. Each process measured one fresh 32K prompt
and decode, then restored the same checkpoint 13 times. The first restored
request is excluded as warm-up; 12 subsequent 64-output requests are measured.
This is 18 independent processes, 216 measured restored requests, 18 restored
warm-ups and 18 fresh requests. The independent sample count is six per mode,
rather than 72 per mode. All 18 processes passed exact output ID/logprob/MTP
checks, normal exit and GPU health checks; no forced cleanup was used.

| Phase | Reference tok/s | Candidate unset tok/s | CPUs 1,2,3 tok/s | Placement vs reference, paired change and 95% interval |
| --- | ---: | ---: | ---: | --- |
| Prefill | 411.590 | 411.108 | 412.863 | +0.309% [-0.124%, +0.744%] |
| First decode after prefill | 16.466 | 16.330 | 16.485 | +0.083% [-2.978%, +3.240%] |
| Decode after checkpoint restore | 21.686 | 21.686 | 21.644 | -0.196% [-1.453%, +1.078%] |

Rates are total phase tokens divided by mean phase duration. Paired changes
use geometric log-latency ratios across six independent blocks with Student-t
intervals (five degrees of freedom). These are different summaries. A
confidence interval crossing zero leaves the direction unresolved and does
not prove equivalent speed. Prefill uses 32768 protocol tokens; 32767 are
batched and the last input enters decode. Fresh and restored MTP acceptance
are 41/66 and 46/51 respectively, so the two decode phases are kept separate.

See [phase summary](phase-summary-v1.json),
[all 18 terminal receipts](repeat-sequence-v1.json),
[first logged numerical check](first-logged-diagnostic-v1.json),
[uniform actual 115-object build flags](uniform-build-flags-v1.json) and
[archive hashes](comparison-archive-v1.json). Raw logs and checkpoint data stay
in the local state directory named by the receipts. CPU background activity
and physical variation remain possible; repetition does not eliminate them.
The candidate is held because no meaningful gain was established, and full
physical 256K qualification was not spent on this result.
