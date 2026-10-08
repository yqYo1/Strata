# Quiet32K host accounting comparison

On2026-10-09 JST, Arc B57010 GiB / Ryzen5 5600X /128 GiB RAM ran baseline/off/on/on/off/baseline, four fresh32768-token A/B/A/B inputs and64 visible outputs per process. Context262144, actual8192 chunks,128 cache slots/325 MiB, resident KV32768 and every requested argument/environment value match except the measured binary and explicit host-counter flag. No API logging, validation, profiler, head/state dump, native event query or extra phase wait was enabled.

| Mode | First prefill tokens/s (2 reads) | Later prefill tokens/s (6 reads) | Later decode tokens/s (6 reads) |
| --- | ---: | ---: | ---: |
| baseline | 407.814 | 449.923 | 16.674 |
| hostoff | 411.976 | 449.778 | 16.508 |
| hoston | 411.931 | 449.748 | 16.414 |

The baseline is qualified binary86972697; hostoff and hoston are private instrumented binary494cf4be. Rates are pooled token/time ratios. First process reads are separate from later fresh reads. Hostoff later prefill differs by-0.03235% and hoston by-0.03887% from baseline. These differences are below the per-read prefill spread. The sample does not establish a useful speed change or a small decode regression: baseline decode spans15.637–18.571 tokens/s, off14.991–18.396 and on14.747–18.651. Full medians/ranges and all24 per-read values are in the JSON/CSV.

All24 complete outputs, printed logprobs, MTP counts, finish reasons and repeats match the qualified baseline. Every read has RESUME0 and REUSED0. All six processes exit normally with no new GPU fault, forced cleanup or owned survivor. These quiet runs check visible numerical references; [the separate first diagnostic](../first-diagnostic/README.md) supplies four-fresh first-head and66-part live-state proof for this private binary.

Eight host reports repeat the exact diagnostic transfer counts: A92998 copies/190240998400 bytes and B92979/190199219200. Over the six later reads, host submission totals average1217.629 ms, GPU grouping waits29353.892 ms, CPU grouping47.744 ms, issuer-publication waits0 and RAM-worker copy sums10124.776 ms; prefill wall duration averages72853.172 ms. Submission and grouping CPU work are small compared with wall time. GPU grouping waits include preceding queue work and dependencies; they are not measured PCIe-only waiting. RAM-worker sums may overlap with GPU work and with one another, so these durations cannot be added into a wall-time breakdown.

Exact expert byte counts make transfer reduction and overlap concrete next experiments. The bounded original API log maps the captured expert ring to an engine supporting compute and copy; a separate native copy-only queue candidate is being prepared on another branch. It has no performance or GPU-safety result here.

The accounting binary remains private and unadopted, with its full262144 lifecycle pending. Only the baseline has full-context qualification. Nothing in this archive makes that qualification transferable to modified code. Small terminal receipts, source/runtime/controller hashes and project/protocol logs are preserved byte for byte; large API logs, binaries and state/session tensors remain private.
