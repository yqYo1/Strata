# Quiet32K native expert-copy comparison

On2026-10-09 JST, Arc B57010 GiB / Ryzen5 5600X /128 GiB RAM ran baseline/off/on/on/off/baseline, four fresh32768-token A/B/A/B inputs and64 visible outputs per process. Context262144, actual8192 chunks,128 cache slots/325 MiB and resident KV32768 match. Every argument/environment value matches except the binary and explicit native-copy flag. No API logs, validation layer, profiler, head/state dumps, event-timestamp queries or extra phase waits were enabled. Existing completion checks and ring barriers remain.

| Mode | First prefill tokens/s (2 reads) | Later prefill tokens/s (6 reads) | Later decode tokens/s (6 reads) |
| --- | ---: | ---: | ---: |
| baseline | 405.260 | 449.782 | 16.701 |
| nativeoff | 408.712 | 449.707 | 16.706 |
| nativeon | 483.911 | 541.376 | 16.572 |

Baseline is qualified binary86972697. Off/on are uniformly rebuilt binary2721f8ef with `STRATA_PREFILL_COPY_ENGINE=0/1`. Rates are pooled token/time ratios. First process reads are separate from later fresh reads. Native-on later prefill improves by20.364% against baseline and20.384% against candidate-off. The native queue factory is the source difference under test; all114 compiler commands match the baseline after normalizing worktree/build/dependency-output paths.

| Mode | Later prefill per-read range | Later decode per-read range |
| --- | ---: | ---: |
| baseline | 449.561–450.052 | 15.733–18.536 |
| nativeoff | 449.608–450.034 | 15.326–18.142 |
| nativeon | 541.154–541.580 | 15.245–18.457 |

Native-on decode changes by-0.770% against baseline; the per-read spread and sample do not establish a small decode gain or regression. All modes request five logprob alternatives for every output token. Decode time includes their readback and CPU scoring; no-logprob generation is a separate unmeasured condition. Every raw read and process-level pooled rate is preserved in JSON/CSV.

All24 outputs, printed logprobs, MTP counts, finish reasons and repeats match the qualified baseline. Every read has RESUME0/REUSED0. All six processes finish with normalQUIT/exit0, no new GPU fault, no forced cleanup and no owned survivor. These quiet checks concern visible references; [the separate first diagnostic](../first-diagnostic/README.md) supplies four-fresh first-head/all66-live-state proof and a bounded native-queue transfer trace.

The change routes expert copies through a validated copy-only immediate queue while retaining the existing compute queue, math, CPU pool, ring reuse barriers and DMA-event ownership. This throughput comparison does not measure DMA active time or an exclusive PCIe wait fraction. The bounded diagnostic trace confirms captured UR/native expert copies use the imported ordinal1 queue; it is not a trace of every quiet transfer.

Candidate272 is not adopted and its full262144 lifecycle is pending. Baseline869 full qualification does not qualify this different binary. Terminal receipts and exact measured controllers bind binaries, source, flags, runtime, fixtures, protocol and cleanup. Large logs, binaries and tensor/session states remain private.
