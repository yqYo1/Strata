# Optional vLLM Xe2 native16 GEMM

This experimental backend uses the Intel BSD-3 vLLM XPU Xe2 tile to read FP16 or BF16 matrices, accumulate locally in FP32 and store FP16 or BF16 output directly. It creates no global FP32 output tensor or CPU GEMM fallback. The build option defaults OFF. Standalone component qualification is recorded in the native16 audit; engine, quality and full-context qualification remain separate.

## Build and select

Use the existing qualified SYCL compiler, MKL and ggml dependencies. The optional dependency is the existing Intel sycl-tla checkout at `87f6850680a580654b9ea2c80dbc01aeb36ad231`. No dependency is downloaded by this option. Add these settings to the normal SYCL configuration:

```text
STRATA_ENABLE_SYCL=ON
STRATA_SYCL_VLLM_XE2_NATIVE16=ON
STRATA_SYCL_DEVICE_ARCH=
STRATA_SYCL_AOT_DOUBLE_GRF=OFF
SYCL_TLA_SOURCE_DIR=/absolute/path/to/sycl-tla
```

The module checks the exact bytes of 207 consumed TLA headers. The new source alone requests `spir64_gen`, `bmg-g21-a0`, GRF256 and the split-barrier/block2D/subgroup-matrix translator extensions. TLA declares SPIR-V builtins with `SYCL_EXTERNAL`, which forbids `-fno-sycl-rdc`. An ordinary RDC object is device-linked separately with plain `-fsycl-link`; the host slice is extracted with `clang-offload-bundler`. Both resulting host objects are direct final-link inputs. Native relocatable device IR and AOT/GRF options do not enter the existing engine's generic JIT link. This uses the documented [Intel LLVM device-link stage](https://intel.github.io/llvm/design/CompilerAndRuntimeDesign.html#device-link). Actual installed-compiler mixed-image acceptance and runtime selection must be checked; source flags alone are not binary proof.

Set the immutable child environment before starting the engine:

| Setting | Effect |
| --- | --- |
| `STRATA_SYCL_PREFILL_NATIVE16=bf16` | Existing long-input native16 chain |
| `STRATA_SYCL_PREFILL_NATIVE16_BACKEND=vllm` | Use the vLLM adapter for public out16 GEMMs |
| Backend unset or `own` | Retain the original joint-matrix out16 path |
| `STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE=auto` or unset | Initially compact for T<128, wide otherwise |
| Tile `compact` or `wide` | Explicit tile for controlled experiments |

The automatic threshold is an initial choice, not a tuned selector. A requested backend missing from the build, malformed setting or unsupported queue rejects explicitly. One process-level route line distinguishes `engine=0` from `engine=bf16`; selecting a backend alone does not enable the long-input chain. Existing short-input and decode routing remains available.

## Tensor and lifetime contract

`f16_out16` reads FP16 X/W; `bf16_out16` reads BF16 X/W. The output format is independent. X is `[T,K]`, W is `[N,K]` and Y is `[T,N]` with row stride `ldy` (zero means N). The vLLM backend requires positive dimensions, beta +/-0, N%64=0, K%32=0, ldy>=N, ldy%8=0 and 64-byte base addresses. All complete X/W/Y rectangles must fit live registered Device allocations in the queue's context/device; Y cannot overlap an input. R810's surface, coordinate and INT_MAX byte/work-item limits remain. Ragged T uses real logical surfaces, with no padded caller memory assumption. Unsupported cases reject instead of silently falling back. The own path retains its original broader contract.

Each Gemm retains the original allocation owner's Runtime and borrows the original stable queue. It adds no device selector or second Runtime implementation. The caller still owns the queue, original async handler and X/W/Y buffers until completion. The registry proves allocation extent/domain, not logical suballocation identity. In-order dependencies and existing checked cleanup remain required; a failed wait does not authorize freeing an in-flight owner.

Long-prefill HC and routed expert constants satisfy the static N/K granularity. Actual WeightRef dimensions and pointer alignment still govern. Column slicing in `native_proj16` can violate N granularity, alignment or the conservative full Y rectangle guard. Such sites require explicit examination; arbitrary sliced model support is not claimed. This backend does not change the existing MoE host grouping schedule.

QSA's separate opt-in is `STRATA_SYCL_PREFILL_NATIVE16_QSA=1`. It requires the long native16 chain, resident unrotated INT8 K/V, Mapping0 and session text RoPE tables (`STRATA_ROPE_TABLE=1`). Projection, norm/RoPE, KV append, attention and gate use typed16 boundaries. The indexer, local norm/softmax/accumulation and recurrent state retain explicit FP32 arithmetic. QSA's original arena reservation remains unchanged. Unsupported cache/position modes reject before layer work.

## Qualification and provenance

Manual targets `sycl_vllm_native16_test`, `sycl_native16_paired_gemm_bench`, `sycl_native16_qsa_test` and `sycl_prompt_native16_test` are excluded from the default build. The first two preserve the independent numerical fixture and five-route paired benchmark from the reviewed standalone component. Main owns finite, serialized GPU jobs, fresh fault checks and actual image/entry attribution. A component PASS or lower sampled NLL is not a general quality or full262144-context acceptance. Do not use exact generated-text equality as a quality requirement.

The core derives from vllm-project/vllm-xpu-kernels `6b30c9b92cc525c32729df988b5f918e7c5661a5`, `csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp`, through the recorded R810 adaptation. Intel's 2025 BSD-3 notice is retained in full in `src/prefill/sycl/vllm_xe2/xe_gemm_bsd.hpp`; that file is not relicensed MIT. New adapter/glue/tests follow Strata's MIT license. External TLA's consumed closure contains 192 BSD-3 files and 15 Apache-2.0 WITH LLVM-exception files; preserve the applicable Intel/NVIDIA and Apache/LLVM notices when distributing source or binaries. No XeStrata implementation, helper or test is used.
