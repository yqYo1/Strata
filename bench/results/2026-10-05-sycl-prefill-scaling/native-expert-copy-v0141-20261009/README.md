# Native expert-copy queue trial

On2026-10-09 JST, Arc B57010 GiB / Ryzen5 5600X /128 GiB RAM completed [four first-use32K head/live-state/output checks](first-diagnostic/README.md) and [24 quiet32K reads](quiet32k/README.md). Native copy-only selection improves later prefill from449.782 to541.376 tokens/s (+20.364%). Candidate-off provides a separate uniformly rebuilt control. Decode variation does not establish a small speed change; logprob scoring is included in these timings.

The change is opt-in and retains existing math, CPU work, completion checks and ring barriers. All owned runs finish normally. The candidate remains unadopted until its own full262144 lifecycle qualification; that gate is pending. The first diagnostic archive preserves its historical first-use scope.
