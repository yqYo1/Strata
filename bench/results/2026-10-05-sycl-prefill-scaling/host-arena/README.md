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

## Request-level PCIe share sweep

The [PCIe sweep](pcie-sweep/summary.json) keeps the same binary and model
settings. One process per page mode accepts a 16-token warmup followed
by ten 64-token requests. The request-level shares are 0.55, 0, 0.1,
0.25, 0.4, then reverse order. Startup is excluded, and all requests
reuse the same warmed prompt checkpoint. Each table entry is the median
of two requests, not an independent multi-process confidence estimate.

| PCIe fraction | Original pages, decode tokens/s | THP, decode tokens/s |
| ---: | ---: | ---: |
| 0 | 14.817 | 15.359 |
| 0.1 | 12.774 | 12.998 |
| 0.25 | 11.724 | 11.625 |
| 0.4 | 7.757 | 7.875 |
| 0.55 | 7.303 | 7.412 |

All requests generate the requested length and finite printed logprobs.
For each fraction, all IDs and printed logprobs match between repetitions
and page modes. Both initial complete finite heads match. Different PCIe
fractions change which experts use CPU or GPU arithmetic and are not
required to have identical logprob bits to each other. The checkpoint
was initially warmed at fraction 0.55; this sweep is not a fresh-prompt
comparison for every fraction.

Zero PCIe share roughly doubles generation throughput in this writing
fixture. In the original-page zero-share final request, the diagnostic
reports 156.71 ms/window, including 136.44 ms of CPU jobs, 0.12 ms of
planning and zero PCIe experts. Returning to 0.55 reports 323.34
ms/window, including 71.56 ms of CPU jobs, 79.23 ms of planning and
14.68 distinct PCIe experts per layer-window. Reducing CPU jobs by
transferring more experts therefore loses overall time on this card.
This is a decode result; layer-major FP16 prefill still uses GPU expert
work and is not switched to CPU by this request option.

## CPU-heavy paired comparison

The [nine-process comparison](cpu-paired/summary.json) fixes the startup
PCIe share to zero. Three rounds compare original pages, THP, and THP
plus the existing `STRATA_IQ256_GATHER=1` flag. Round two reverses their
order. Each process has a 16-token warmup and two repeated 64-token
requests, with all other settings and instrumentation as above.

| Mode | Median decode tokens/s, six measured requests |
| --- | ---: |
| Original pages | 15.3910 |
| THP | 15.8294 |
| THP and AVX2 gather | 15.3843 |

All nine complete finite first heads match bit for bit, and every ID
and printed logprob matches across processes and repetitions. THP's
median is about 2.8% higher in this CPU-heavy condition, with overlapping
individual timings and a 12.228 tokens/s outlier retained in the first
original-page process. This is a small local observation, not a general
throughput guarantee. The gather flag gives no improvement and is not
selected. Prefill performance is not measured by these short prompts.

## Allocation and AOT validation

The [24 mapping checks](mapping-check/run.log) exercise original pages,
THP and the `STRATA_NO_LARGEPAGES` override at sizes immediately around
system and 2 MiB page boundaries, including a 64 MiB plus 17-byte arena.
They observe the actual mapping boundaries and use `mincore` after
destruction to verify that both ends are unmapped. All checks pass.

The current-source B570 [AOT check](aot-check/summary.json) also passes.
Its complete finite first head and all IDs and printed logprobs from
one 16-token warmup and two 64-token requests match the JIT original-page
control. The AOT result is a validation observation; it is not a paired
AOT speed comparison. Both build logs and the allocation probe are
preserved. The embedding service remains inactive.
