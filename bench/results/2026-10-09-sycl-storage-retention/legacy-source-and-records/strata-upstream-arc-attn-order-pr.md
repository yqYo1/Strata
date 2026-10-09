Longer coding prompts spend more time in the original split-K attention. Add opt-in `STRATA_PREFILL_ATTN_LAYOUT=4`, which retains layout 3's subgroup 16 and shared query transpose and puts neighboring queries on the innermost workgroup dimension. The floating-point expressions, reduction order, 64-cell split and merge remain the same; the default remains layout 0.

On Arc B570 10 GB, the standalone FP16 attention median improves by 5.81%. Three matched normal CLI pairs at 8,087 tokens reduce median prefill from 17,034.7 to 16,924.2 ms (0.65%), or 474.74 to 477.84 overall tok/s. The 4,096-to-8,087 incremental rate improves from 1,128.58 to 1,146.41 additional tok/s. All twelve complete finite 248,320 first heads and generated IDs match the accepted reference. The record also contains negative aligned-FP16 padding and GEMM-wrapper studies; they produce no production GEMM change.

Validation:

- Standalone attention: all 32 complete-output bit checks pass, with unchanged independent FP64 reference errors.
- JIT: all 30 registered tests pass, including the existing layout-3 test and the new query-order test.
- Normal MTP: short and long first/repeat/other/restored-checkpoint requests pass; both layouts' long first heads, generated IDs and logprobs agree exactly.
- B570 `bmg-g21` AOT: all 30 tests pass with no skips, both full 4K/8K model heads match the original, and all four short/long MTP request outputs and logprobs match JIT/original references.

Measurements use Ryzen 5 5600X, 128 GB installed RAM, oneAPI 2026.1 and compute runtime 26.35.39758.10. Batch 128, cache 128 and context/chunk 8,192 stay fixed, with phase/transfer/preload markers off for the CLI comparison. Raw logs, immutable binary/source hashes, matched ordering and measurement limits are recorded in the [measurement report](https://github.com/yqYo1/Strata/blob/9c0255b/bench/results/2026-10-05-sycl-prefill-scaling/attention-query-order/README.md).
