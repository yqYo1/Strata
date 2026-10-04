# Original GEMM and quantized weight expansion

The unchanged `src/prefill/gemm.cu` now compiles and links as a complete component
through a source cuBLAS frontend and native oneMKL. Its `init`, `init_external`,
`rebind`, `bf16`, `f16`, `native` and destructor paths run on the B570. The original
row-slicing orchestration is retained. This target does not use section garbage
collection or replacement test implementations. Full prefill and engine linking
remain open; the HIP/hipBLASLt branch is not exercised.

Both JIT and B570 AOT builds pass all 19 related CTest programs. The new GEMM
program makes 2,525,030 numerical comparisons and 15,556 padding checks in each
build. These are component correctness checks, not PP/TG measurements or a CUDA
device differential. All 256 original manifest files remain byte-identical.
Full logs, source/object/binary hashes, selected library hashes and earlier
failures are in [native-gemm-results.json](native-gemm-results.json).

## Native BLAS binding

`cublasGemmEx` resolves the opaque frontend stream to its owning Runtime queue.
It submits a real column-major oneMKL USM GEMM with BF16 or FP16 operands and
FP32 output. The original `W^T * X` view, leading dimensions, alpha and beta are
retained. Real `N`, `T` and `C` operands are supported, with `C` equivalent to
transpose. Handle device ownership, stream validity and device allocation ranges
are checked before submission. Unsupported type/compute/algo combinations return
an error. The zero-K case applies beta on the GPU, leaving row padding untouched.

The binding explicitly selects `compute_mode::standard`. The relevant native
interfaces are documented in Intel's [GEMM reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2024-2/gemm.html)
and [compute-mode reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2024-1/compute-modes.html).
Local installed headers and actual B570 tests are recorded separately from those
documentation versions. The source frontend is not the cuBLAS binary ABI.

`cublasSetWorkspace` returns `CUBLAS_STATUS_NOT_SUPPORTED`. The oneMKL interface
used here has no external-workspace argument. Original optional setup reports
status 15 and continues; `Gemm::init` still allocates its original 32 MiB buffer.
The library does not use that buffer. Original fixed-workspace allocation-budget
parity is therefore **unreproduced**, and internal vendor allocations are not
bounded by this frontend. Passing cold graph capture does not close that gap.

## Original dequantization arithmetic

`tools/sycl/extract_dequant.py` verifies pinned source hashes and extracts the
literal original `group32`, geometry, IQ conversion/dispatch and row-byte helper
bodies. It lowers CUDA qualifiers and half intrinsics to SYCL syntax. Original
block structs and IQ codebooks come from the existing pinned GGML shim. Generated
helper files are identical between JIT and AOT builds. The previous port's
quantization layout or scale precision is not used as the oracle.

The dense dequant family covers Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K,
Q6_K, IQ4_NL, IQ4_XS and Strata Q2_0. FP16/FP32 expansion also covers IQ2_XXS,
IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ1_M through the original IQ dispatcher.
The IQ flat and gate/up entry points cover their original 15 formats. The
embedding dispatcher additionally covers BF16, including padded source rows and
repeated token IDs. Each IQ work item retains its original logical lane within
a 32-lane block; no physical 32-wide SYCL subgroup is assumed by these helpers.
At this checkpoint, other IQ products, native expert products and Q8 activation
quantizers remained separate open families. The later
[original product binding](native-products.md) compiles the whole IQ/native
files and replaces these separately written IQ wrappers. Its tests cover the
product and grouped branches; CUDA device arithmetic identity remains open.

The numerical oracle links the actual pinned `ggml-base` CPU dequantizers and
FP16/BF16 conversions. It does not reuse the generated GPU arithmetic. Tests use
finite synthetic blocks with zero and signed scales, selected source rows, and
guarded outputs. FP16/BF16 conversions are checked bit-for-bit. Whole GEMM
outputs are checked against independent FP64 products of the rounded input
values, with tolerance `2e-5 * sum(abs(terms)) + 2e-6`. This establishes the
observed error bound, not CUDA accumulation-order identity.

## B570 validation

Measured on Intel Arc B570 with Ryzen 5 5600X, oneAPI 2026.1.1 and isolated
compute-runtime 26.35.39758.10 / IGC 2.41.5 / GMM 22.10.0. Persistent SYCL binary
caching is disabled. The AOT target is `bmg_g21`; vendor oneMKL device code is
loaded dynamically and is not rebuilt by Strata's AOT flags. The installed MKL
version header reports `INTEL_MKL_VERSION=20260100`; its hash is recorded rather
than deriving a release version solely from the installation directory name.

| Check | JIT | B570 AOT |
| --- | ---: | ---: |
| Related CTest programs | 19 pass | 19 pass |
| Dequant slice cases | 51 | 51 |
| All dequant / IQ compared values | 1,513,728 | 1,513,728 |
| IQ flat / gate-up / embedding cases | 32 | 32 |
| Original GEMM cases | 59 | 59 |
| Low-level BLAS orientation / zero-K cases | 10 | 10 |
| Graph cases | 18 | 18 |
| GEMM numerical comparisons | 1,011,302 | 1,011,302 |
| Padding checks | 15,556 | 15,556 |
| Maximum product error / sum of absolute terms | 7.6064e-08 | 7.6064e-08 |
| Original real-model sample types | 6 pass | 6 pass |

The dense products include asymmetric dimensions up to `T=129, N=1280, K=640`,
beta 0/0.5/1, NaN-initialized output for beta zero, and padded output strides.
Quantized native products cover all 17 formats with scratch holding either two
or 19 rows, exercising the original sliced and unsliced paths. Borrowed scratch
is rebound from two to three rows. Seventeen graphs each replay three times
with changed quantized weights. The eighteenth graph captures the process's
**first oneMKL operation**, before any BLAS warmup; instantiation leaves output
untouched and first replay produces the expected product.

A held callback verifies that warmed GEMM host submission returns within two
seconds while the source stream remains pending and independent GPU work
completes. Releasing the callback produces the expected matrix. Invalid leading
dimensions, unsupported mixed input types, undersized interior pointers and
stale handles are rejected; invalid calls leave output unchanged. This verifies
dependency and lifetime behavior on the tested inputs, not overlap throughput.

The unchanged `src/kernels/dequant_bf16_test.cpp` also reads both local
Qwen3.8-Flash-Next IQ3_S GGUF shards. It checks the first four rows of the first
eligible 2D tensor for six supported types: Q6_K, IQ4_XS, Q4_K, IQ4_NL, Q5_K
and Q8_0. FP32 differences are zero and BF16 checks pass in both builds. This
original BF16 test does **not** cover the IQ3_S expert blocks or the full model.
Current model sizes and modification times are recorded; complete shard hashes
were not recomputed in this phase.

A separate successful JIT run under `LD_DEBUG=libs` records the actual loaded
oneMKL, SYCL, GGML and selected driver libraries. The AOT suite uses the same
environment but has no separate loader trace. Initial extractor/CPU-oracle build
errors, an erroneous legacy-stream capture test and a duplicate-main model
target failure are retained alongside their resolutions in the results record.
Legacy default-stream GEMM is tested ordinarily; graph capture uses a named
stream as required by the frontend's CUDA capture rules.

## Reproduce and remaining integration

```sh
source /opt/intel/oneapi/setvars.sh
# Select the isolated driver library prefix documented in runtime-memory.md.
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export SYCL_CACHE_PERSISTENT=0
cmake --build build-upstream-sycl -j3
ctest --test-dir build-upstream-sycl -V \
  -R 'original_|mmq_public_stages|mmq_context_lifetime|runtime_'
cmake --build build-upstream-sycl-aot -j3
ctest --test-dir build-upstream-sycl-aot -V \
  -R 'original_|mmq_public_stages|mmq_context_lifetime|runtime_'
cmake --build build-upstream-sycl-aot --target original_dequant_model_test -j3
build-upstream-sycl-aot/original_dequant_model_test SHARD1.gguf SHARD2.gguf
```

The component CMake configuration requires oneMKL headers and BLAS libraries;
`STRATA_MKL_ROOT` defaults to `/opt/intel/oneapi/mkl/latest`. This phase does not
change the historical 19/20 host C++ syntax census: the fully linked original
GEMM CUDA translation unit is separate evidence. Native architecture admission,
remaining kernel families, full engine linking, model state parity and new
PP/TG measurements are still required. PP1000/TG70 has not been established.
