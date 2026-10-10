# Current native pack metadata census

CPU-only header inspection of the local IQ3_S pack on 2026-10-10. All 144 role descriptors across 48 layers match the primary GGUF header's types, shapes and absolute offsets. Per-role block equations sum to every declared expert blob and the contiguous packed total. No tensor payload or GPU state was read, and no engine/model run was performed.

| GU type | Down type | Layers | Bytes per expert |
| --- | --- | ---: | ---: |
| IQ3_XXS (18) | IQ4_NL (20) | 14 | 2,176,000 |
| IQ3_XXS (18) | Q2_0 (42) | 3 | 1,715,200 |
| IQ3_S (21) | IQ4_NL (20) | 9 | 2,329,600 |
| IQ3_S (21) | Q2_0 (42) | 1 | 1,868,800 |
| IQ2_S (22) | IQ4_NL (20) | 15 | 1,971,200 |
| IQ2_S (22) | Q2_0 (42) | 5 | 1,510,400 |
| IQ4_XS (23) | IQ4_NL (20) | 1 | 2,662,400 |

The pack contains 512 experts per layer, 24,576 expert blobs, and 50,292,326,400 total packed bytes. The maximum blob is 2,662,400 bytes, giving 1,363,148,800 bytes for the frozen layer-cache slot-size scenario. This verifies current local metadata matching the historical size arithmetic; it does not prove that a future run loaded the pack or that its simultaneous VRAM allocations fit.

The previous 384-expert CPU capacity fixture covers selected IQ2_S/IQ4_NL experts in 15 layers. It cannot stand in for all paired formats. The production FP16 path writes 6,553,600 bytes of GU weights and 3,276,800 bytes of Down weights per dequantized expert, but layer-weight reuse alone does not prove that dequantization is reused: the source has a two-slot dequant ring and actual routed calls still need counting. Fused native/MMQ paths have different service and eligibility.

[The receipt](receipt.json) preserves all per-layer descriptors, source/manifest/header hashes, unchanged file identity and interpretation limits. The whole large shard was not hashed; the header hash covers only metadata plus padding. This is a next-control admission input, not an actual row-routing census, service result, payload qualification or full-context proof.
