# Production QSA prefix reuse guardrails

Verified against local source commit `cd1d7e06c3eb876430924b8b19a2edb4b816ac1b`; relevant source pins match the manifest. This is design context, not performance or runtime qualification.

## Current boundary

The production `bf16_prefix_ordered_exp` callsite in `src/prefill/prefill.cpp` passes per-query selected IDs and `live_cells = p0 + T`. In each call, `src/kernels/sycl/qsa_bf16_prefix.cpp` prepares the full live prefix for both KV heads into caller-owned transient BF16 scratch, then runs attention on the selected IDs. Packing checks the logical page mapping before reading codes/scales, writes zero for unmapped or padded cells, and converts int8 code × decoded FP16 scale in FP32 before BF16 rounding. The same in-order queue orders pack before attention. The API contract explicitly refreshes scratch on every call; it does not provide a persistent packed-prefix cache.

Within the call, Q is reused across selected-cell tiles. A packed K/V row can be reused by more queries only where their selected lists overlap. Across chunks, `p0 + T` changes the live extent; QSA layers use separate KV pools. The separate bulk KV mirror lifetime does not establish that this BF16 scratch can be reused.

## Exactness and decision gates

A selected-row compact pack must preserve each query’s selected ID order and duplicates, bounds/missing-row behavior, invalid-scale propagation, code × scale FP32 evaluation, BF16 RNE operands, FP32 online-softmax updates, and the current ordered-exp sum. Do not merge rows in a way that changes multiplicity or the sequence of attention updates.

Before considering either selected-row packing or a query tile that shares staged K/V, measure the union of selected logical rows versus the full live prefix across representative chunks, and include duplicates, invalid/missing rows, near-dense unions and low-overlap cases. The added ID maps, gather work, synchronization, local storage and register pressure can erase the data movement saved. This prerequisite already exists under `sparse_cross_query_reuse_scope`; reuse that conditional scope rather than repeating a literature survey.

Cross-call hoisting needs evidence of an unchanged same-layer live extent and page-map generation. Current production calls refresh scratch and extend the live prefix chunk by chunk; without a qualifying repeated consumer, reject this seam before implementation.

## External context and limits

[FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691) are algorithmic context for tiling and online softmax, not implementations of Strata’s ordered sparse selection.

The collection cited the current [vLLM paged-attention wrapper](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/paged_attn.py) and [SGLang kernel guide](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/README.md), with the [SGLang license](https://github.com/sgl-project/sglang/blob/main/LICENSE). The vLLM file has an Apache-2.0 SPDX header; the SGLang repository license is Apache-2.0. These moving `main` links were not pinned to source commits. They provide cache/layout and ownership context, not an exact sparse reuse implementation or code provenance. The collection did not identify an Intel exact sparse-attention implementation.
