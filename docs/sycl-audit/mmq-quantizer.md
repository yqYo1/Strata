# MMQ activation quantizer: source-derived validation

Measured on Intel Arc B570, Linux, oneAPI compiler 2026.1.1, Level Zero, on
2026-10-04. This is component correctness validation, not a PP/TG measurement.

The new implementation preserves the pinned GGML source's 128-value, 144-byte
block layout; K-block/row transposition; D4 FP32 scales, DS4 half scale/sum pairs,
and D2S6 half scales/partial sums; four adjacent input values per lane; XOR
reduction order; reciprocal/multiply/round/reciprocal expression order; input row
selection; 512-column padding and 128 trailing guard blocks. The API is the 2D
specialization called by Strata `prefill::mmq::quantize`. GGML's separate scatter,
3D/4D and FP4 entry points are not implemented by this component.

The layout selector is checked for every `ggml_type` against the original selector
extracted verbatim from fixed GGML `mmq.cuh`. Twenty-two types have a Q8 metadata
layout in that helper; this does **not** mean all 22 have Strata MMQ products or
that FP4 uses this quantizer in the engine. Configure checks hashes of `mmq.cuh`,
`quantize.cu`, `quantize.cuh`, and `ggml.h`; a modified quantizer source is rejected.

The byte oracle is a host translation of pinned `quantize_mmq_q8_1`, using raw byte
offsets and the existing independent CPU half-bit conversion, rather than the GPU
struct or old SYCL Q8_1 implementation. Source extraction validates the selector;
it does not automatically extract or verify the host arithmetic translation.

## Results

Both JIT and `bmg_g21` AOT builds pass 546 cases, comparing 70,640,640 bytes including
guards. Cases cover all three metadata layouts; widths 4, 32, 128, 512, 1280 and
2560; 1, 7 and 17 rows; direct and selected/repeated input rows; noncontiguous row
stride; zeros, varied finite values, cancellation and exact/adjacent half-way
values. Six additional cases use 4007 rows and width 1280/2560. Leading and trailing
guards must remain unchanged. [Machine-readable record](mmq-quantizer-results.json).

The first 432 small cases passed before adding full-size inputs. The 4007-row
fixture then failed at type IQ3_S, byte 372833: expected code 3, actual code 2.
The observed value was `0x1.2afa64p+7`, maximum `0x1.daa0b4p+12`, reciprocal
`0x1.12p-6`, FP32 product `0x1.4p+1` (2.5). Separately storing `round(product)`
returned 3, while the original multiply/round/int8 expression produced code 2 in
the quantizer. This happened with `-fno-fast-math -ffp-contract=off`. Replacing the
multiply with explicit `fmul_rn` preserved the FP32 intermediate and resolved the
failure. The exact positive and negative values are now dedicated regression
inputs. This evidence does not identify which compiler transformation caused the
original expression's result; no disassembly-based attribution is claimed.

## Limits

No CUDA device result has been compared. Upstream compiles the MMQ target with
`-use_fast_math`; this component explicitly rounds divisions and multiplication.
The source expressions and layout are preserved, but CUDA reciprocal approximation,
flush-to-zero behavior, and cross-vendor numerical differences still require an
actual CUDA comparison. Extreme subnormals, NaN and infinity inputs are not covered.
For all-zero blocks the source reaches NaN-to-char conversion (`0 * inf`); the new
component writes zero codes explicitly. Those bytes still need CUDA confirmation.

This quantizer alone is not a full MMQ product, a full engine port, or a speed
improvement. A separate [product component](mmq-product.md) now consumes its layout. The
CUDA translation-unit checklist remains unverified for complete `moe_mmq.cu`.
The root build and existing upstream CPU/host processing have not been modified.

## Reproduction

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S tests/sycl_upstream -B build-upstream-sycl -G Ninja \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_GGML_DIR=/path/to/pinned/llama.cpp
cmake --build build-upstream-sycl -j 2
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ctest --test-dir build-upstream-sycl -V
```

For AOT use a separate build directory and add
`-DSTRATA_SYCL_DEVICE_ARCH=bmg_g21`. These builds only produce the component test.
