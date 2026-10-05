# Host expert pages and PCIe split on the Arc B570

These runs use an Intel Arc B570 10 GiB, Ryzen 5 5600X, 128 GiB RAM,
native IQ3_S Qwen3.8 Flash Next and oneAPI 2026.1 with precise math. The
selected compute runtime and environment are recorded in each manifest.
The separate embedding service stays stopped at the user's request.

## Transparent huge pages

Upstream's CUDA arena has a 2 MiB-aligned anonymous allocation fallback
with `MADV_HUGEPAGE`. The SYCL port did not carry this fallback. The new
optional `STRATA_SYCL_ARENA_THP=1` requests it when a reserved hugetlb
allocation fails. The existing allocator remains the default. Shared-file
arenas retain their original layout, and `STRATA_NO_LARGEPAGES` takes
precedence. Prefix and tail trimming use complete system pages. Advice
does not guarantee huge-page backing; the arena still reports normal
page backing rather than claiming a reserved hugetlb allocation.

The [frozen JIT build](build.json) changes only the host arena allocator
from PR6. Both page modes use this same binary. Linux's THP policy is
`madvise`, and the reserved huge-page pool has zero pages. The
[three paired runs](thp-paired/summary.json) reverse order in pair two.
Each process runs one 16-token warmup and two 64-token writing requests.
They use a 37-token chat prompt, context 2,048, K8/V8, 600 expert slots,
five CPU workers, PCIe fraction 0.55, MTP window four, zero probability
floor, split verification, no suffix draft and greedy sampling. The
small prefill chunk is 32; these are decode checks, not PP measurements.
Repeated requests reuse the same prompt checkpoint. Decode diagnostics
and five printed logprobs per token are enabled identically, with no GPU
phase instrumentation. Startup and memory observations are outside the
request decode timers.

| Page mode | Median decode tokens/s, six 64-token requests | Observed huge backing of anonymous memory |
| --- | ---: | ---: |
| Original fallback | 7.3058 | 0 GiB |
| THP requested | 7.3436 | 46.771–46.840 GiB |

All six complete first heads have 248,320 finite values and identical
bits. Every generated ID and printed logprob matches across the six
processes, including the warmups. No process swap is observed. Huge
backing is measured through `/proc/PID/smaps`; it is not inferred from
the successful advice call. The median speed difference is about 0.5%,
with overlap between individual observations, so no repeatable end-to-end
speed gain is established in this PCIe-heavy condition. The switch stays
optional.

The initial decode diagnostic reports about 13 distinct PCIe experts per
layer-window. This motivates testing smaller PCIe shares before deciding
whether CPU-side page changes help the actual bottleneck. The printed
phase times cover host submission and waits; they are not independent
GPU kernel times. PP 1000 and TG 70 remain unachieved targets.
