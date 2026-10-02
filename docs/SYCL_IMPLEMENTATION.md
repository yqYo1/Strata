# SYCL backend implementation

The backend is under development. The runtime and device arena run on Intel Arc
B570; the SYCL build does not yet provide the `strata` inference executable.
The implementation follows [the port research](SYCL_RESEARCH.md) and
[the operation inventory](SYCL_BATTLEMAGE_OPERATIONS.md).

## Build and validate

On Linux with Intel oneAPI DPC++ installed:

```sh
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
cmake -S . -B build-sycl -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_ENABLE_SYCL=ON -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF
cmake --build build-sycl --target strata-device sycl_runtime_test sycl_kernels_test sycl_gdn_test sycl_quantize_test sycl_bf16_test sycl_mmvq_test sycl_gr_test sycl_rope_test sycl_kv_test sycl_attention_test sycl_qsa_index_test sycl_decode_attention_test -j2
ctest --test-dir build-sycl -R '^sycl_' --output-on-failure
build-sycl/strata-device --list-devices
```

`STRATA_SYCL_TESTS` controls the independent SYCL tests. The native CPU expert
library is disabled in this runtime-only command to avoid fetching ggml before
the inference path is connected. CUDA, HIP and SYCL are mutually exclusive.

For the separate large-address check, with at least 5 GiB of free VRAM:

```sh
timeout 60 build-sycl/sycl_runtime_test --large
```

This allocates 4 GiB + 64 KiB, writes different 64-bit values below and above
the 4 GiB boundary, and reads both back. It does not measure memory bandwidth.

## Runtime contract

- Compute and transfer queues use the same device and context. Each queue is
  ordered; dependencies between queues use explicit events.
- CPU access to host USM follows completion of its producing GPU command. No
  concurrent CPU/GPU atomic access to host/shared USM is required.
- Device and bounded host-staging allocations own their context lifetime and
  complete pending work before freeing storage. Bulk expert RAM will remain
  ordinary host memory rather than a shared-USM allocation.
- Asynchronous errors are retained and rethrown by runtime checks and waits.
  A timeout reports incomplete work; it does not cancel commands or authorize
  releasing their buffers.
- `DeviceArena` preserves the engine's bump-allocation interface, with checked
  arithmetic and alignment of actual pointers. Memory planning queries free
  VRAM rather than treating total capacity as available memory.

## Validation recorded on 2026-10-02

Intel Arc B570, DPC++ 2026.1.1, Level Zero driver `1.17.39395+14`, normal driver
settings; see the research document for the machine configuration:

- 64 changing-input CPU → transfer queue → compute queue → CPU round trips,
  each with 4,097 integer elements: all results matched.
- Invalid subranges, impossible device allocation size and move ownership:
  expected rejection/ownership behavior.
- A deliberately throwing host task: the asynchronous error reached the caller
  and remained observable on subsequent checks.
- Bounded completion wait and the existing `strata-device --selftest`: passed.
- Distinct 64-bit writes below and above 4 GiB: both values matched.
- FP16/BF16 conversion over 73,748 inputs, including every FP16 encoding,
  rounding boundaries, random FP32 bit patterns and NaNs: passed against host
  `_Float16` and oneAPI BF16 conversions (NaN payload identity is not required).
- Weighted RMS normalization with row tails/canaries, batched GDN gate and
  elementwise arithmetic: passed against double-precision CPU calculations.
- Generic and subgroup-32 native routing: batched stable-ID and weight checks
  passed, including ties and widely separated logits. The CPU reference restores
  the default floating-point environment so subnormals participate in sorting.
- Packed 2/4/8-bit embedding gathers, expert row exclusion and event-completed
  payload publication: passed. GPU polling for CPU-written flags is explicitly
  rejected; the scheduler must use events.
- Generic and native GDN recurrence: six changing steps checked against a
  double-precision CPU oracle with a different state layout. This includes the
  real 128-wide, 16-key-head/48-value-head geometry, modulo head pairing and
  native readout scaling. Convolution history, canaries, L2 normalization and
  the sigmoid-gated output norm also passed.

- Canonical Q8_0, CPU-expert scaled Q8_0 and Q8_K activation quantization:
  2,048 blocks per variant matched CPU reference bytes, including zero blocks,
  halfway rounding cases and opposite-sign maximum ties. FP32 expert scales
  also matched bit for bit. Inverse conversion, output canaries and invalid
  shapes/overlaps passed. Quantization uses explicitly rounded division: the
  ordinary SYCL FP32 division differed by one ULP on the scale of test block 14.

- BF16-weight/FP32-activation MMVF: 73 shapes matched an ordered-FMA CPU
  reference bit for bit, including all adaptive work-group sizes, padded
  activation/output strides and 1, 2, 3, 4, 5 and 8 tokens. Each batched output
  also matched its separate single-token call and a double-precision dot
  product within the test tolerance. Fourteen legacy BF16-activation shapes,
  including odd reduction widths and split sizes 1 through 256, passed the
  double-precision reference and canary checks.

- Composed gated residual: twelve layer/head cases passed CPU references,
  including the real 2,560-wide, four-stream, rank-320 geometry in BF16,
  FP32 and native MMVF modes. Checks include per-stream normalization,
  activation rounding, all projections, absent head injection, saturated/zero
  gates, exact in-place writes, workspace sizing and output canaries.
- NEOX RoPE at positions 0 through 262,144, widths 128/256, unscaled,
  linear and YaRN scaling, text/image positions and in-place updates: table
  rotations matched bit for bit. Analytic rotations used 32 host-libm float32
  frequencies captured by value and strict host floating-point compilation;
  the maximum absolute difference from the float32 CPU reference was
  `2.385e-7`. Standard SYCL device `pow` had produced up to `0.02439` on the
  same inputs. This is a correctness measurement, not a speed comparison.
  Registered angle tables currently belong to runtime device 0.
- Paged FP16, INT8 and Hadamard-Q4 KV storage: 15 format/page/width
  combinations passed byte-exact CPU checks, with permuted page tables,
  nonresident pages, host-USM authoritative copies and output canaries.
  Coverage includes stored-FP16-scale INT8 rounding, Q4 maximum ties,
  changing device steps, empty gathers, Q4 prompt/step equivalence and
  prompt staging. The 256-point Hadamard transform matched the CPU
  butterfly sequence bit for bit and its inverse recovered the input
  within `4e-7` absolute error.
- Gathered GQA attention with the real 24-query/2-KV-head, 256-wide
  geometry and a 6-query/3-KV-head, 68-wide fixture: counts 0, 1, 7, 257
  and 2,051 passed double-precision softmax/value references. Normal-input
  maximum absolute error was `3.358e-7`; a fixture with query/key magnitudes
  up to 30/10 had `7.169e-5`. Scalar and device-step entry points matched
  bit for bit. Empty selections, weight/output canaries, FP16/FP32 gates,
  in-place native gates and weighted RMS at widths 66 through 2,560 passed.
- Generic indexer pooling: eight geometry/scaling/image-position cases,
  each with 19 changing steps, matched CPU pooled/spare keys bit for bit.
  Sixteen score cases passed double-precision per-head ReLU references.
  Twenty-four selection cases through 32,768 cells matched independently
  sorted IDs exactly, including ties, signed zero, NaNs and infinities.
  Signed zero is canonicalized by bits because arithmetic `+0` was optimized
  away by the compiler and initially broke the ascending-ID tie rule.
- Split attention directly over paged FP16, INT8, Q4 and hybrid K8/V4:
  twelve format/page cases passed the double-precision reference with maximum
  absolute error `3.541e-7`. Three queries with 13/67/129 selected cells and
  capacity 133 matched their single-query calls bit for bit. Checks include
  permuted pages, missing pages, all-masked/empty selections, chunk tails and
  scratch/output canaries. Quantized direct reads retain FP32 dequantization;
  they do not insert the FP16 rounding used by the separate gather path.
- Native Q8_1 and MMVQ for Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K,
  Q6_K, IQ4_NL and IQ4_XS: 150 synthetic cases passed, including small/large
  reduction widths, row tails, 1/3/8 columns and both multi-column layouts.
  Q8_1 bytes matched the independent host quantizer; the exact layout matched
  separate single-column calls bit for bit. The largest
  `abs(error)/(1 + sum(abs(reference terms)))` was `5.099e-8` against scalar
  dequantization and double-precision dot products. The reference includes the
  original-input-sum correction required by affine Q4_0/Q5_0.
- Real Q2_0 GGUF: first/middle/last rows from all 448 tensors of those ten
  formats passed with three activation columns. The same normalized maximum
  error was `4.699e-8`. This samples weights, not a forward pass.

The real-weight check is optional and reads the model in place:

```sh
build-sycl/sycl_mmvq_test --gguf /path/to/model-00001-of-00002.gguf
```

The recorded model is `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, Q2_0,
revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. Both GGUF shard sizes
and SHA-256 hashes were checked before use. The model and native pack live
outside the repository, under `$XDG_DATA_HOME/strata-sycl` (or
`~/.local/share/strata-sycl` when unset). The pack contains 1,079 index entries;
its tokenizer passed all 16 strings in `strata_tokenizer.py --check`.

These checks establish runtime and kernel behavior, not model correctness or inference
performance. Remaining work includes the model kernels, CPU expert scheduling,
prefill, cache/sequence state, verification and launch integration, followed by
layer/logit comparisons and real-model inference validation.
