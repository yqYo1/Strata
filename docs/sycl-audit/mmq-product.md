# Routed MMQ product: first upstream operation port

This component consumes the upstream 128-value transposed activation blocks from
`mmq_quantize`. It implements the nine formats enabled by default in the pinned
Strata CUDA MMQ build: Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL,
IQ4_XS and Q8_0. It is isolated from the engine while the upstream audit continues.
It is not yet the complete `prefill::mmq::Context` adapter.

## Source correspondence

`tools/sycl/extract_mmq_loaders.py` checks fixed source hashes and extracts all nine
`ggml_cuda_mmq_load_tiles_*` functions from pinned GGML `mmq-load-tiles.cuh`.
Their arithmetic and indexing bodies are retained. The extraction removes CUDA
function qualifiers and adds the logical thread index as an explicit parameter.
The CUDA MMA shared-memory branch is selected; the SYCL shim supplies byte
permutation, per-byte comparison/subtraction, popcount, half conversion and logical
thread coordinates. Codebooks and packed block structs come directly from pinned
`ggml-common.h`, using its existing SYCL declarations.

The Intel tile has 16 weight rows and 64 activation rows, 128 work-items, and
16-lane hardware subgroups. Loader indexing retains logical 32-lane CUDA thread
coordinates. The shared-memory row stride is 84 words, large enough for both
8-scale and 16-scale formats. These are explicit device tiling choices, not claims
that CUDA uses the same tile.

| Upstream operation | New component |
| --- | --- |
| Load 256 weight values and metadata per row | Original extracted CUDA loader into local memory |
| Load MMQ activation blocks | Original transposed K-block/row addressing with D4 FP32 metadata |
| Q2_0/IQ2_XXS/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS/Q8_0 integer dot | Intel XMX signed-int8 product, 32 reduction elements |
| IQ2_XS/IQ2_S integer dot and two scales | Two separate 16-element products (zero-padded to the Intel 32-element instruction), then scale and combine |
| `C*dA*dB` and running FP32 result | Explicit FP32 multiply and FMA epilogue |
| Grouped expert bounds and destination IDs | Device bounds, empty groups, identity or permuted output IDs, caller's output stride |
| Packed weight storage and guard | Read-only original packed weights; original 4096-byte tail requirement; no global FP16 weight expansion |

The original scale application is in `mmq-vec-dot.cuh`, functions
`ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma` and
`ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma`. IQ2_XS/S use the latter, combining two
16-value sums under one activation scale. The remaining seven formats use the
former. This distinction is tested rather than flattening every format into one
32-value weight scale.

## Independent checks

The same extraction tool separately takes the nine original `dequantize_row_*`
functions from GGML's CPU `ggml-quants.c`. The test uses those scalar decoders and
a double-precision dot against the quantized activations. It does not use the GPU
loader to compute the matrix reference. Half conversion is adapted to the SYCL
block declaration; the decoder bodies and codebooks are original.

The acceptance bound is `abs(error) <= 3e-6 * sum(abs(products)) + 1e-6`. This is
numerical agreement with an independently decoded mathematical reference, not
bitwise equality with CUDA. The quantizer has its separate byte-exact test.

There are 90 small cases across all nine types, 1/16/19 weight rows and valid
widths among 256/640/1280/2560. Every active output is numerically compared.
Cases include empty, one-row, 65-row and four-row experts, identity and permuted
output IDs, output padding, and unused leading/trailing activation rows.

Five additional cases use 4007 activation rows:

- IQ2_S, IQ3_S and IQ4_XS: 1280 weight rows, reduction width 2560.
- Q2_0 and IQ4_NL: 2560 weight rows, reduction width 640.

These large cases check every active output for a finite written result, all
inactive output/stride/tail guards, unchanged input weight bytes, and numerical
samples across weight and activation tiles and their boundaries. They do not
numerically compare every large output. In total there are 86,652 numerical
comparisons and 187,165 output guard checks per build.

JIT and B570-specific AOT results and exact source/binary hashes are recorded in
[mmq-product-results.json](mmq-product-results.json). This is correctness evidence;
no end-to-end PP/TG or speedup is claimed.

## Portability failure found by these checks

The first IQ2_XS nonzero multi-block case failed with default strict aliasing:
product -580.319 versus reference -1159.74. A separate diagnostic reconstructed
the CUDA loader's output without doing a matrix product and found 756 mismatching
weights among 1280 values. Disabling strict alias optimization eliminated those
loader mismatches and passed the product comparisons. The flags are applied to
the extracted-loader component and its test, not to the upstream engine globally.
The original CUDA code deliberately reinterprets packed fields and local storage;
these accesses must not be compiled with incompatible alias assumptions. The
exact miscompiled instruction has not been identified by disassembly.

## Remaining fidelity work

The initial full-K implementation recorded above has since gained the
[upstream stream-K partition and fixup algorithm](mmq-stream-k.md), with both
numerical and exact addition-order checks. The original CUDA automatic tile/config
selection, scratch pool, optional Q5_0/Q4_K/Q5_K/Q5_1 formats, gather/SwiGLU wrapper,
and full host/runtime integration remain open. No CUDA device comparison has
been run, including CUDA fast-math, FTZ, exact FMA contraction and stream-K results.

The original host processing and root build remain unchanged. No old SYCL tuning
preset or implementation is imported as the new baseline.

## Reproduction

Use the same isolated CMake commands as [the quantizer](mmq-quantizer.md).
`ctest -V` now runs both `mmq_quantize_upstream_source` and
`mmq_product_upstream_cpu`. The product requires Intel XMX int8 support as tested
on the B570; it is not a generic-device fallback.
