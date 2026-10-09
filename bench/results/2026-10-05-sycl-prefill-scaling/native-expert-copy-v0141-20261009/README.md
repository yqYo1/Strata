# Native expert-copy queue trial

The [first diagnostic](first-diagnostic/README.md) uniformly builds candidate6caa1421 and runs four fresh32K inputs with64 outputs on Arc B570 / Ryzen5600X /128 GiB RAM. First logits, all66 live-state parts and generated/MTP results match qualified updated baseline869; normal exit and kernel-fault checks pass. A bounded trace pairs880 expert transfers with the imported copy-only native queue.

This experimental opt-in path changes the expert-copy queue factory only. It retains the existing ring barriers, DMA events, CPU pool and math. Logged times are excluded from speed. The separate quiet baseline/off/on comparison is pending; full262144 qualification and adoption remain pending. Native profiling is disabled for this trial.
