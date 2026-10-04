# Original native products and grouped experts

The complete pinned `native_mmvq.cu`, `iq_kernels.cu`, `native_bf16.cu` and
`bf16_gemv.cu` now compile into native SYCL components. Their original host
entry points, argument guards, format dispatch, old/new IQ paths, multi-exact
native layouts, grouped v1/fused choices and launch geometry remain in the
translated files. The original files are unchanged. This closes these component
bindings; full engine execution and new PP/TG measurements remain open.

The JIT and B570 AOT builds at this checkpoint each pass 20 related CTest programs. The new
product test performs 2,420,678 numerical comparisons, 6,932,160 byte comparisons,
2,168,088 bit comparisons and 1,997,284 guard comparisons per build. Full logs,
source/object/binary hashes and earlier failures are in
[native-products-results.json](native-products-results.json). The earlier
[GEMM checkpoint](native-gemm.md) remains historical; its complete synthetic
GEMM/dequant regression also passes with these new IQ bindings.

The following [native primitive phase](native-primitives.md) extends launch
argument capture and local-memory support, regenerates these four product files,
and passes their complete regressions in both 21-program suites. Hashes and
logs below remain the earlier product checkpoint.

## How the original sources execute

`tools/sycl/lower_cuda_products.py` checks the source hashes against the pinned
inventory and checks the pinned GGML common header. It translates all four
files, covering 26 kernel definitions and 45 launch sites. Generated sources
stay in the build directory. All four complete objects link into the product
test without section garbage collection or placeholder kernel implementations.

`cuda_kernel.hpp` maps CUDA launches to native `nd_range<3>` kernels on the
owning Runtime stream. CUDA x/y/z axes map to SYCL dimensions 2/1/0. The exact
original shared-array extent becomes a local accessor. Kernels without shared
memory have no local accessor. Warp-dependent kernels require native subgroup
size 32 and retain each XOR/down reduction step. Naive BF16, non-warp split
BF16 and elementwise SwiGLU kernels retain their original workgroup geometry
without requiring that subgroup size.

[SYCL subgroup algorithms](https://github.khronos.org/SYCL_Reference/iface/group-algorithms-library.html)
provide XOR permutation and lane shifts. The adapter explicitly restores CUDA's
self value for an out-of-range down-shuffle target, since the SYCL result in
that case is unspecified. The original integer dot operation becomes four
signed-byte products added in order. This is an arithmetic binding, not a
measurement or assertion of Intel hardware DP4A instruction generation.

The complete IQ translation unit now supplies its own original flat dequant,
embedding and interleaved gate/up entry points. The preceding separately
written IQ wrappers were removed. Dense dequantization continues to use the
previous original arithmetic helpers. The existing GEMM test independently
checks these entry points against actual pinned GGML CPU dequantizers.

## Quantizer rounding found during validation

Initial product tests failed before reaching the grouped pipeline. At a
half-tie, native Q8_1 stored 63 while the independent CPU scalar-expression
oracle stored 64. Removing fast-math did not resolve it. Adding
`-foffload-fp32-prec-div` also did not resolve it on this installation.

A diagnostic kernel repeated the actual subgroup maximum tree and stored the
intermediate values. For input `0.0588379`, both sides obtained maximum
`0x1.e2p-4` and scale `0x1.e5cb98p-11`. Plain GPU input/scale division returned
`0x1.fbfffep+5`; the CPU and explicit Intel `fdiv_rn` returned `0x1.fcp+5`
(63.5). `roundf` therefore selected different integer codes. A separate probe
without the subgroup reduction reported no differences; it did not reproduce
the failing product. Both probe results remain recorded.

The lowering now substitutes explicit `fdiv_rn` for the two Q8_1 divisions,
retaining their original operands and order. The same original IQ quantizer
helper serves separate and fused stages. Final tests compare all Q8_1 metadata
and code bytes against the independent CPU tree, including zero blocks,
signed inputs and half-ties.

The [compiler option reference](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2026-0/foffload-fp32-prec-div.html)
describes a correctly rounded division mode. The failure above is a measured
result with this compiler/driver combination; the explicit operation is the
validated binding. Installed intrinsic header hashes are recorded too.

The original native CUDA sources use `--use_fast_math`. This SYCL component
uses `-fno-fast-math`, enables fast FMA contraction for native MMVQ/MMVF only,
and explicitly rounds Q8_1 division. NVIDIA approximate-division, FTZ and
exponential bit identity have not been measured. Agreement with the CPU
expression oracle does not establish CUDA device bit identity.

## Coverage on the B570

Measurements use the Intel Arc B570 and Ryzen 5 5600X, oneAPI compiler 2026.1.1,
compute runtime 26.35.39758.10, IGC 2.41.5, GMM 22.10 and Level Zero loader 1.32.
The recorded environment selects the Level Zero GPU and disables the SYCL
persistent cache. Driver/runtime paths, resolved targets and hashes were
rechecked against the prior phase. No new library loader trace was taken here;
other driver/compiler caches may still exist.

| Original path | Cases per build | Inputs and branches |
|---|---:|---|
| Native MMVQ | 320 | Ten formats; widths 512/8192; every 1–8 column count; five output rows; exact and generic multi layouts |
| IQ product | 504 | Fourteen formats; widths 256/1280/2560; columns 1/3/4/5/8/9; old and decode-once paths |
| BF16 | 184 | Thirteen widths from 2 to 2560, every 1–8 token count, padded strides; split sizes 1/16/32/64/128/256; ordinary naive/warp branches |
| Grouped experts | 371 | All 72 supported gate/up × down pairs; old/new × v1/fused; group stride, empty calls and asynchronous submission |
| Graph replays | 75 | 24 two-kernel product graphs replayed three times; one three-kernel grouped graph replayed three times |

The native formats are Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K,
IQ4_NL and IQ4_XS. IQ products cover IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL,
IQ3_S, IQ2_S, IQ4_XS, IQ1_M, Q2_0, Q4_K, Q5_K, Q5_1, Q5_0 and Q8_0.
Together these exercise 17 distinct quantized product formats.

The independent product reference uses actual pinned GGML CPU `to_float`
functions and FP64 accumulation with the stored Q8_1 scale/codes. Q4_0/Q5_0
also include the original-input-sum correction used by the original GPU dot
formula. The numerical bound is `3e-5 * L1 + 2e-6`; the largest observed
absolute error divided by L1 is `1.83106e-5`. This is an input-specific error
measurement, not a whole-model error bound. Exact native multi-column results
also match separate single-column calls bitwise. Old/new IQ results match
bitwise on these inputs.

Grouped tests use two GPU expert blobs, seven group slots, thirteen entry
slots, repeated/permuted activation tokens and scatter to eight distinct output
rows. Active group entry counts are 1/0/5/2, with unused prefixes/suffixes.
All 72 format pairs use synthetic H=256/FF=256 weights. Additional Q2_0/Q2_0
and IQ3_S/IQ4_NL pairs use synthetic H=2560/FF=640 weights. No real model
weights are loaded by this product test.

Gate/up/down values are checked against CPU references. Separate SwiGLU is
checked with an independent exponential on the stored GPU gate/up values;
its Q8_1 intermediate is checked bytewise against the CPU tree. Old/new,
group stride and separate/fused active Q8_1 bytes and outputs also match each
other bitwise. Guards cover unused output rows, scratch tails and fused
intermediate entries outside the call. Expert blobs and index arrays remain
byte-identical after execution.

All ten native F32 convenience compositions and fourteen IQ products are
captured as quantize-plus-product graphs. Instantiation leaves output untouched;
replay reads changed input values after the host launch-choice flags change.
The grouped graph similarly retains its fused three-kernel choice while replay
reads changed input values, expert pointers and device group counts 0/2/4.
A warmed whole grouped call returns within two seconds behind a held CPU
callback while an independent stream completes GPU work. That verifies host
asynchrony, not an overlap speed or bandwidth claim.

| Final related suite | CTest programs passed | Total wall time |
|---|---:|---:|
| JIT | 20/20 | 14.59 s |
| B570 `bmg_g21` AOT | 20/20 | 8.94 s |

These are correctness-suite durations, not inference throughput measurements.
AOT compilation reports register spills in some IQ multi-column kernels; the
logs preserve those warnings. Unchanged standalone large MMQ tests and the
previous six-format real-model dequant sample target were not rerun here.

All 256 inventoried original files still match their pinned hashes. The host
C++ syntax checkpoint remains 19 passes/one failure; the four new native kernel
translation units are separate component builds. Remaining kernel families,
native program/architecture metadata, complete original engine linking and
whole-model state validation remain open. Fixed vendor BLAS workspace parity,
Windows and physical multi-GPU execution remain unestablished. PP1000/TG70
has not been measured or achieved.
