# FP16-X packed expert math contract

Use this reference only if Main selects a direct packed-GGUF projection seam for long prefill. Its consumer is the source and numerical plan for that candidate, before implementation. It does not establish a production route or a performance case.

## Origin and exact formats

Curated from knowledge-collection report SHA-256 d141bdc2f2a6f85161b415a8759c3177e695d141cf91dac743836ea5afbd2145, pinned to Strata source d5ea1cb8e7b7e8feaa89c0af93df9a02e82ef509. The full implementation pin map is in the report's source.json (SHA-256 c322209b743487e3f151e15e5f01db67cf635eefec1741c2e5f86d292aaf6c6e); native_experts.txt is pinned as d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d.

The pinned shapes are N=2560 and F=640. GU is two F-by-N matrices; Down is N-by-F. The seven observed pairs and exact per-expert blob sizes are:

| GU/Down type | Layers | Bytes per expert |
| --- | ---: | ---: |
| 18/20 | 14 | 2,176,000 |
| 18/42 | 3 | 1,715,200 |
| 21/20 | 9 | 2,329,600 |
| 21/42 | 1 | 1,868,800 |
| 22/20 | 15 | 1,971,200 |
| 22/42 | 5 | 1,510,400 |
| 23/20 | 1 | 2,662,400 |

GGUF block geometry is type 18=(256 values,98 bytes), 21=(256,110), 22=(256,82), 23=(256,136), 20=(32,18), and 42=(64,18). GU row-byte strides for types 18/21/22/23 are 980/1100/820/1360; their Up offsets after 640 rows are 627,200/704,000/524,800/870,400 and Down offsets are 1,254,400/1,408,000/1,049,600/1,740,800. Down row strides are 360 bytes for type 20 and 180 for type 42. No K block tail occurs at these dimensions; a generic API must still reject or safely mask partial blocks.

## Numerical and storage contract

The materialized route dequantizes packed weights, rounds them to FP16, and uses FP16 activations with FP32 accumulation. A direct packed candidate must decode the exact per-format codebook/sign/scale, round each decoded weight to FP16 RNE, widen that half value for multiplication, and accumulate in FP32. Using the unrounded decoder float changes the algorithm.

- GU output rounds to BF16. SwiGLU consumes BF16 gate/up, evaluates in FP32, clamps finite hidden values to [-65504,65504], then stores FP16 hidden.
- Down consumes FP16 hidden and FP16-rounded weights, accumulates in FP32, and outputs BF16 before routed weighted FP32 combination. The ordinary Down plane is two-byte storage with BF16 semantics: production calls pass output_bf16=true. Do not label this plane FP16.
- Preserve routing/order and FP32 combination or recurrent state separately from tensor storage. A changed reduction tree needs an explicit prospective tolerance.
- The current packed SIMT API takes Q8_1 activations and returns FP32 projections. It is not an equivalent replacement for the FP16-X/BF16-output boundary.

A candidate can load each packed K block once per output-row/K tile and reuse its decoded weights over M FP16 rows while keeping M accumulators in FP32. This is a mapping hypothesis, not a speed claim. Check all address extents, format-specific indices, M tails, and any generalized K tail.

## Minimum evidence before use

Exhaustively compare each supported block decoder with scalar decode after FP16 rounding across code, sign, scale, extrema and half-way cases. Compare M=1, full and partial-M GEMMs with the materialized route, then check GU BF16 rounding, SwiGLU/clamp, Down BF16 output and combine/handoff. First prove production dispatch can reach the selected path for the exact pair set. Set tolerances and safety gates in a committed plan; collect long-input performance only after path attribution selects this cost.

## Provenance

Core pins from the report include src/prefill/moe_window.inc SHA-256 be2937ca8f2b817f195da4aa944288837d59a0bc90fece269917186928cd0fa5, src/kernels/sycl/native_mmvq.cpp 2db527d87fce1a24ffb7998d87701c5cd0f5d0c2aa1a20a46904d1262b0dca58, src/kernels/sycl/iq.cpp 414872755b46a01b3cdb940e39fe65c2bc5595ca62ff970c0465e6d79c5f49ef, and include/strata/sycl/gguf_decode.hpp be53c2b1d612433720599999ea0bc4902d2d550e16270a9eebe8e6868fb91c50. The BF16 Down correction is verified against the same source revision's pinned src/prefill/sycl/gemm16.cpp SHA-256 4549995b96fbacc42d2640d31ad33767a25ec575d5c72ce822a0fd18038ed626 and the call sites in moe_window.inc.

The report records the existing packed kernel's GGML lineage as commit 3cf03257f219afbe7334045ff7c6a06ac68c627d, MIT; the fork snapshot is also pinned to its MIT LICENSE. The note proposes no copied code. XeStrata was not used.
