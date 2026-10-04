# CUDA translation-unit checklist

Every upstream `.cu` file under the inventoried scope is listed, including diagnostic
programs. All remain unverified as a complete translation unit. The common headers
and host paths are separately included in `source-inventory.json`.

| Upstream source | Listed literally in root CMake | Complete SYCL semantic/runtime parity |
| --- | --- | --- |
| `src/core/device.cu` | yes | unverified |
| `src/core/pinned.cu` | yes | unverified |
| `src/kernels/cuda/bf16_gemv.cu` | yes | unverified |
| `src/kernels/cuda/cvec.cu` | yes | unverified |
| `src/kernels/cuda/dequant_bf16.cu` | yes | unverified |
| `src/kernels/cuda/dequant_s2.cu` | yes | unverified |
| `src/kernels/cuda/elementwise.cu` | yes | unverified |
| `src/kernels/cuda/fused_gdn.cu` | yes | unverified |
| `src/kernels/cuda/fused_gr.cu` | yes | unverified |
| `src/kernels/cuda/gdn.cu` | yes | unverified |
| `src/kernels/cuda/gr.cu` | yes | unverified |
| `src/kernels/cuda/iq_kernels.cu` | yes | unverified |
| `src/kernels/cuda/kv_q4.cu` | yes | unverified |
| `src/kernels/cuda/kv_q8.cu` | yes | unverified |
| `src/kernels/cuda/kv_stream.cu` | yes | unverified |
| `src/kernels/cuda/native_bf16.cu` | yes | unverified |
| `src/kernels/cuda/native_flash_attn.cu` | yes | unverified |
| `src/kernels/cuda/native_gdn.cu` | yes | unverified |
| `src/kernels/cuda/native_gdn_preprocess.cu` | yes | unverified |
| `src/kernels/cuda/native_gr_norm.cu` | yes | unverified |
| `src/kernels/cuda/native_gr_postops.cu` | yes | unverified |
| `src/kernels/cuda/native_mmvq.cu` | yes | unverified |
| `src/kernels/cuda/native_moe.cu` | yes | unverified |
| `src/kernels/cuda/native_ple_postops.cu` | yes | unverified |
| `src/kernels/cuda/native_qsa.cu` | yes | unverified |
| `src/kernels/cuda/native_qsa_indexer.cu` | yes | unverified |
| `src/kernels/cuda/native_qsa_score.cu` | yes | unverified |
| `src/kernels/cuda/native_rope.cu` | yes | unverified |
| `src/kernels/cuda/native_router.cu` | yes | unverified |
| `src/kernels/cuda/ple.cu` | yes | unverified |
| `src/kernels/cuda/qsa.cu` | yes | unverified |
| `src/kernels/cuda/qsa_decode_attn.cu` | yes | unverified |
| `src/kernels/cuda/qsa_prompt_attn.cu` | yes | unverified |
| `src/kernels/cuda/qsa_select.cu` | yes | unverified |
| `src/kernels/cuda/quantize_act.cu` | yes | unverified |
| `src/kernels/cuda/rope.cu` | yes | unverified |
| `src/kernels/cuda/router_top10.cu` | yes | unverified |
| `src/kernels/cuda/s2_expert_grouped.cu` | yes | unverified |
| `src/kernels/cuda/s2_gemv.cu` | yes | unverified |
| `src/kernels/cuda/s2_gemv_fast.cu` | yes | unverified |
| `src/kernels/cuda/s2_gemv_q8.cu` | yes | unverified |
| `src/kernels/cuda/s2_gemv_quads.cu` | yes | unverified |
| `src/kernels/cuda/s_gemv.cu` | yes | unverified |
| `src/kernels/cuda/sampler.cu` | yes | unverified |
| `src/kernels/cuda/shared_expert.cu` | yes | unverified |
| `src/kernels/cuda/verify_kernels.cu` | yes | unverified |
| `src/prefill/gdn_rec_parity.cu` | yes | unverified |
| `src/prefill/gemm.cu` | yes | unverified |
| `src/prefill/ggml_cuda_host.cu` | yes | unverified |
| `src/prefill/kernels.cu` | yes | unverified |
| `src/prefill/moe_fused.cu` | yes | unverified |
| `src/prefill/moe_fused_iq.cu` | yes | unverified |
| `src/prefill/moe_mmq.cu` | yes | unverified |
