# Raw-layer and bounded-group staging budget

Use this reference only if Main considers copying contiguous packed experts by whole layer or bounded group on the long-prefill route. It complements docs/research/prefill-packed-transfer-topology.md, which describes a separate per-expert bank result at source commit 8b57207dc5ccfe1101156a294224d27e374ca64c. This note is pinned to d5ea1cb8e7b7e8feaa89c0af93df9a02e82ef509 and covers N=2560,F=640 raw-layer/group size arithmetic. Crosswalk route, source revision and shape before using that other result; this note does not select a placement.

## Origin and seven-pair payloads

Curated from knowledge-collection report SHA-256 9b47510b306ae239a716db72cad577af3fef83e73e7ff045028a6e9a756e0c95, pinned to Strata source d5ea1cb8e7b7e8feaa89c0af93df9a02e82ef509. Full source pins are in source.json SHA-256 db8e5ecb82f4648010159d170cc11b2964861da4c86bf1b6ce4484b8c3da8fda; the exact seven-pair descriptor is native_experts.txt SHA-256 d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d.

Each layer contains 512 experts. For pairs GU/Down, per-expert raw bytes B_l and layer counts are: 18/20: 2,176,000 (14 layers); 18/42: 1,715,200 (3); 21/20: 2,329,600 (9); 21/42: 1,868,800 (1); 22/20: 1,971,200 (15); 22/42: 1,510,400 (5); 23/20: 2,662,400 (1). The 48 layers total 50,292,326,400 bytes. A whole layer's payload is 512*B_l; the largest is 1,363,148,800 bytes for 23/20 (1.2695 GiB). Those sizes do not establish device admission or fit under max_mem_alloc_size.

The earlier 32K diagnostic records read=32,769, while 15,728,640 assignments at K=10 across 48 windows imply 32,768 routed token rows/window. Keep those quantities distinct. It reports 22,320 accepted uploads, zero recorded hits and 45,841,868,800 requested upload bytes; per-row actual misses are unqualified, and requested bytes are not PCIe wire traffic or latency.

## Alternatives and lifetime constraints

- Whole-layer copy reduces upload commands but copies unselected experts too. Retaining all 48 would require 50.29 GB before other allocations.
- For maximum B_l, two buffers for group widths 8/16/32/64 use 42.6/85.2/170.4/340.8 decimal MB; one buffer uses half. With u selected experts in width g, excess payload is (g-u)*B_l, worst case (g-1)*B_l. Selection-driven groups and eager full-layer coverage have different payload costs.
- The existing bank uploads B_l on a miss and may reuse a slot; its behavior is not equivalent to a contiguous layer allocation. A prior global host-staging 640/1280-row screen failed; those row limits are not a raw-layer or group threshold.
- The layer-major source slab uses padded expert stride S=ceil(max(B_l)/256)*256. GGUF tensor offsets alone do not prove one tight contiguous expert-layer view. A provider must prove 512 experts, B_l stride, and an extent of exactly 512*B_l.
- Keep host source bytes unchanged until each copy completes, and each destination unavailable until its last dequant/GEMM reader completes. Cross-queue copy, consume, reuse, and error-drain edges need explicit dependencies.

At physical context 262144, a fully GPU-resident BF16 residual of width 10,240 alone is 5 GiB, before KV, decode allocations, expert data, activations and scratch. Actual free VRAM, maximum allocation, simultaneous owners and candidate admission remain unknown. No new residency result is established here.

## Consumer and provenance

The next consumer is a Main-owned placement decision after a completed representative census supplies exact per-layer routed IDs/counts and a full-context snapshot records free/max allocation plus all co-resident allocations. Only then should Main pin a source/build and compare per-expert, bounded-group and whole-layer options with a lifetime-safe plan.

Key source pins: src/prefill/moe_window.inc SHA-256 be2937ca8f2b817f195da4aa944288837d59a0bc90fece269917186928cd0fa5; src/prefill/layer_major.inc 1fd34995c5255ed655b049596d795760c31b88055b5b572d13456162d507737e; include/strata/prefill/expert_bank_plan.hpp 7b3874a8eabd78fa48dac1902ce287f0954a23c98333de9424cd50ea80c7f156; include/strata/prefill/expert_upload_staging.hpp 8f3bbbb1ade5e52a16704a3d469cd99e7fc2f05559cd7018bcd11bf983196b81; and src/core/expert_cache.cpp 4330db0592bf7d860647fda5a7db9a7d2a2d2d35fcb28800bd2f918912abaf6e. The source snapshot is MIT; the report cites SYCL 2020 revision 13 and Level Zero 1.20.41 specifications, not external implementation code. XeStrata was not used.
