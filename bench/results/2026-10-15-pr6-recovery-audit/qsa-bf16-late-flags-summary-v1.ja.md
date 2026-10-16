# BF16 QSA: diagnostic aggregation scheduling control

Same BF16 operand/FP32 online arithmetic and matrix geometry. Diagnostic-only workgroup collectives: three per 32-cell tile -> two after all tiles. Explicit local accessor payload 28168 -> 28160 bytes; actual compiled hardware barriers/allocation are not measured.

| Path | Independent process medians (ms) | Median (ms) |
| --- | --- | --- |
| base | 627.616498, 627.444672, 627.245598 | 627.444672 |
| late | 617.088252, 617.114474, 616.768229 | 617.088252 |

Latency reduction: 1.6506%; speedup: 1.01678x. 各3process×10 samples、2 warmups。順序: base-0,late-0,late-1,base-1,base-2,late-2.

Each variant checks four full queries against independent BF16 contract; full output finite/guards/repeat and all inputs immutable. Full late-flags boundary/diagnostic suite passed separately. Cross-variant exact output is a control result for this fixture, not an adoption precision requirement.

Component host timing only. No model PP/quality, full-model 262144 context, RESTORE, production selection, GPU arithmetic utilization/occupancy or physical traffic qualification.

Retain as experimental scheduling result; defer production selection until whole-prefill gate. This particular diagnostic aggregation does not explain most component latency. Next discriminate repeated per-query KV decode/local transpose from one-time GPU BF16 pack, charging pack time and extra memory; use pinned SGLang prefix branch as design reference, not a speed claim.
