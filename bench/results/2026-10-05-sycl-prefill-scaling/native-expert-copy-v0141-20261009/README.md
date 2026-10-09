# Native expert-copy queue trial

On2026-10-09 JST, Arc B57010 GiB / Ryzen5 5600X /128 GiB RAM completed [four first-use32K head/live-state/output checks](first-diagnostic/README.md) and [24 quiet32K reads](quiet32k/README.md). Native copy-only selection improves later prefill from449.782 to541.376 tokens/s (+20.364%). Candidate-off provides a separate uniformly rebuilt control. The two-process-per-condition decode comparison does not establish equivalence; its mixed-prompt ranges are not a noise estimate. Logprob scoring is included in these timings.

The change is opt-in and retains existing math, CPU work, completion checks and ring barriers. All owned runs finish normally. Its own [full262144 lifecycle qualification](full256k/README.md) now passes: two fresh full inputs, all saved tensor bytes, actual restoration, clipped tails, capacity refusals and a later32K input. The opt-in engine remains on its tuning branch; this archive does not change defaults. The first diagnostic archive preserves its historical first-use scope.

[Separate repeated decode comparison](decode-repeat/README.md) uses six independent process blocks and72 fixed-state measured decodes per condition. Prefill and decode adoption are judged separately; a confidence interval including zero is not proof of equivalence.

The repeated fresh32K comparison finds an ON-versus-baseline first-decode regression of7.985% (95% speed-change interval −12.806% to −2.897%). The candidate is held back despite its prefill gain; phase separation is required before adopting it.
