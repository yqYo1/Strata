# Isolated IQ2_S NT1 index-spread fixture

Source-only, unexecuted. This standalone CPU fixture changes no production dispatch,
quantizer, pool, calibration, parent CMake, or model configuration. No timing loop,
weights input, model loading, GPU dependency, or performance/adoption gate is present.

Three distinct arms: normal pinned IQ2_S CPU trait pointer, a noinline direct-control
copy of its AVX2 dot, and a separate noinline index-only copy. Both copies use the pinned CPU FP16 conversion macro (x86 lookup table), not the public conversion function. The latter replaces
only the grid-index construction with the existing HiSpread/table/interleave
mechanism. It retains four scalar codebook reads per vector, lane order, signs,
scale nibble expansion, maddubs/madd, two int32 accumulators, ascending block FMA,
GGML horizontal tree and final `0.125f` multiplication. Gate and Up are separate
calls followed by `(g/(1.f+std::exp(-g)))*u`. No scale table, gather, paired activation
loads, or accumulator fusion. Both copies safely memcpy packed sign words to local
uint16 storage instead of the C source's uint16 pointer cast. This shared aliasing
adaptation is explicit; its emitted code/cost must be checked, not assumed free.

## Source identity

Frozen Strata base assigned by owner: af12591ca840deb4649b703cbb7469fbc2417fc9.
Pinned GGML commit: 3cf03257f219afbe7334045ff7c6a06ac68c627d.
Copy source: `ggml/src/ggml-cpu/arch/x86/quants.c`, AVX2 branch of
`ggml_vec_dot_iq2_s_q8_K` lines 3075–3158, exact scale shuffle and hsum helpers.
SHA256: 3c489fdc77e3ab484a5f188bff9c60c0e9483c951ed65c1a562a04c249d5628e.
Codebook/layout source: `ggml/src/ggml-common.h`, SHA256
0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c.
HiSpread/grid_indices source: `src/kernels/cpu/iq_avx2.cpp`, SHA256
58e9c7e49b8158f698cda59aff06c245f62aaecd33556b1e88e234140890fded.
Original GGML license applies to copied code; pinned GGML repository LICENSE is
MIT. Generated standalone code keeps its tables internal to isolated_iq2s.

## Finite fixtures and output

No arguments are accepted. 256 qh bytes × 256 low bytes × four positions = 262144
index comparisons, all 1024 indices, plus 65536 decoded four-qword lane comparisons.
The literal scalar index oracle is independent of HiSpread construction.

96 profiles: widths 256/512/2560/8192, three literal uint32 seeds, eight patterns.
Each tests 16 synthetic independent Gate/Up pairs (1536 pairs, 9216 dot calls total).
LCG is unsigned32 `state=1664525*state+1013904223` modulo 2^32. Inputs are zero,
constant positive, alternating endpoints, half-step boundary inputs with a block
anchor of +127 (iscale=-1), random finite ordinary inputs, two finite magnitude
scales, and nextafter neighbors of the half-step inputs. Packed weights have all
legal byte fields, nonnegative finite fp16 scales, zero/max/asymmetric/random scale
nibbles and all-positive/all-negative/random sign streams. The maximum width is
8192 and maximum row count is 16; no file input or caller-controlled allocations.

Activations are quantized with pinned production `quantize_row_q8_K`; buffers are
prefilled zero including all-zero block bsums. Before quantization, finite input,
nonzero max/finite scale and bounded scaled products are checked; after it, finite
d and exclusion of raw -128 are required. Zero blocks are separately valid. No
NaN/Inf/raw-Q8/denormal robustness cases are mixed into the ordinary-domain proof.
No synthetic input is claimed to represent live engine activation distributions.

Every Gate, Up and final GU must be finite and bitwise equal across all three arms.
Failure throws and exits nonzero, with no success summary. Compact CSV lines report
exhaustive counts and one profile result with an FNV1a checksum of inputs, prepared
weights, Q8 bytes and dot results. FNV is reproducibility identity, not cryptographic
provenance. stdout flush/ferror failure exits nonzero; partial output is never a pass.
Owner must require child normal exit0 plus final summary and all expected profiles.

## Owner-only recipe (not executed here)

Use the shared serial measurement lock and separately bounded stdout/stderr. Freeze
clean dependency commit/status, compiler version, full configure flags, source and
binary SHA256, CPU placement and FP environment. Require native Zen3 AVX2/FMA/F16C,
not simulated CPU feature flags. CMake hashes central copy/table/trait/quantizer
sources; owner remains responsible for complete dependency/build provenance.

```sh
cmake -S sycl/tools/native-iq2s-nt1-index-spread -B build-iq2s-index-spread \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release \
  -DGGML_SOURCE_DIR=/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml
cmake --build build-iq2s-index-spread --target iq2s_index_spread
./build-iq2s-index-spread/iq2s_index_spread > fixture.csv 2> fixture.stderr
```

Standalone IntelLLVM C++20 host recipe, precise FP, AVX2/FMA/F16C on candidate TU,
GGML_NATIVE=ON, static CPU GGML/Threads, no OpenMP/SYCL parent. No dependency download
is implemented. Owner supplies finite wall/CPU/RSS/file budgets (initial wall120s,
RSS512MiB and output1MiB are proposed supervision limits, not measured requirements).
Repeat an owner-approved sanitizer build. No long-running timing is automatic.

Before any performance experiment inspect linked direct-control, candidate, baseline
and complete callers, bound to binary/object hashes. Confirm active baseline AVX2;
same scalar loads/signs/scales/madd/FMA/reduction/postscale; no gather, no reassociation.
HiSpread adds a data-dependent 2KiB table load and two eight-element temporary index
arrays per inner iteration. `grid_indices` loads eight low bytes; only four are
active, and even the final second-half load remains inside the block's sign bytes.
Sign memcpy may introduce stack scratch in both copies. Reject if index shifts are
not reduced, hot-loop scratch/spills/helper calls increase, or same operation order
is not established. Exact outputs alone do not prove identical instructions or cost.
There is no claim of call-target removal benefit: inspect trait call devirtualization.

External real-weight cohort compatibility is deliberately not implemented: no payload
is necessary for this synthetic index-only fixture. Future actual640-row expert,
train/holdout timing, live-activation, numerical model/fullcontext and adoption gates
remain independently required. C/D/r5 and other previous failure statuses are unchanged.
