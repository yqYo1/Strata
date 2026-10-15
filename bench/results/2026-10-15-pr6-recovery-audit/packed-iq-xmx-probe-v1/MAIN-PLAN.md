# Packed IQ FP16 XMX component proposal v1

Status: **all builds, tests, GPU launches, comparisons and timing NOT_RUN**.
Only this directory is new. No product edits, Git/branch operations, process
control, compiler invocation, tests or GPU calls were performed by the author.
Main owns serial execution after the live diagnostic closes and its gates clear.
No performance or numerical qualification, engine integration or adoption claim.

This applies the project strata-upstream-sync skill, AGENTS.md and artifact
retention policy. Read R610, Main's R609/R610 review, R616 and R617. R616's active
format list is wrong: use GU 18 IQ3_XXS, 22 IQ2_S, 21 IQ3_S, 23 IQ4_XS; Down 20
IQ4_NL and 42 custom Q2_0. No type17-only implementation is proposed.

## Component and source contract

`probe.cpp` contains a real standalone joint_matrix kernel. One SG16 workgroup
owns one M8/N16 output tile, streams K16 fragments, stages FP16 A [8,16] and B
[16,16] in local memory, accumulates FP32, then masks padded M on store. All
matrix operations and local barriers are subgroup/workgroup uniform. The last
barrier in each K iteration prevents staging overwrite before matrix reads.
SLM is 1280 bytes per WG (A256+B512+C512), before implementation overhead.
No candidate allocation is proportional to N*K FP16. Packed loads and decoder
lookup work repeat across M tiles; this is an intentionally simple prototype,
not a coalescing/occupancy claim. No SwiGLU, routing, grouped experts, peer,
streaming engine or product dispatch is included.

Tile types/layouts match `tools/sycl/matrix_probe.cpp` and configured
`src/kernels/sycl/prompt_attention.cpp`. Runtime checks fp16, ext_intel_matrix,
SG16, WG/SLM and advertised FP16/FP32 matrix combinations as that path does.
The max-size admission follows current code; it is not proof every intermediate
shape is supported. Compile/runtime rejection closes the experiment, preserves
the error and leaves current product fallback unchanged. No retry on a fault.
Only six formats, 1<=M<=1280, 16<=N,K<=2560, N%16=K%16=0,
K divisible by format block width, and even GU N are admitted. CLI decline=77;
shape/type/matrix decline is not silently measured as a different algorithm.
Pointers are privately allocated USM; this is not a public pointer-validation API.

The decoder is the exact existing proposal header (not a second transcription):
`../parallel-iq-dequant-v1/proposed/include/strata/sycl/iq_parallel_decode.hpp`.
R617 reviewed scalar semantics, not device qualification. Every B weight uses
repository `f16_from_f32` RNE bits before bit_cast to half. X is already row-major
half [M,K], packed weights are output-row-major [N,K], Y is float [M,N]. GU
input storage is [640 gate rows][640 up rows]; logical columns 2r/2r+1 read
row r and row r+N/2, matching current interleaved GU materialization. A GU
640x2560 pair therefore uses CLI N=1280,K=2560, not N=640.

The independent reference kernel calls current scalar `decode_block` into
float[256], converts with repository RNE and expands W; then calls exactly the
row_major/nontrans/trans FP16 -> FP32 oneMKL GEMM orientation, alpha1/beta0,
LD=K/K/N in `src/prefill/sycl/gemm.cpp`. Reference allocates full dense W for
one component only (current Gemm::native can chunk scratch); this tests full
buffer arithmetic, not current chunk scheduling/performance equivalence.

## Serial Main adapter and proposed commands

Do not run these until Main releases its active diagnostic. Use the owned
controller/exclusive measurement lock, native child environment before exec,
no setenv/unsetenv after provider initialization. One case/arm per process;
compare offline after both normal exits. Use clean identical environments with
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu` and one explicit B570 device selection
if multiple GPUs appear. Record complete environment, device/driver/runtime,
boot ID, owned PID/start, start/end journal cursors, deadlines, exit/reap and
fresh kernel-fault receipt. Runtime/driver faults, timeout, budget stop or missing
receipts are incomplete/failed, even if numeric output happens to compare.

Pin source hashes from source-manifest.sha256 and current product revision in
Main's receipt; resolve and hash actual compiler, oneMKL and linked runtime files
before configure. Paths below pin configured versions, not claimed binary hashes:

```
D=bench/results/2026-10-15-pr6-recovery-audit/packed-iq-xmx-probe-v1
cmake -S "$D" -B /tmp/strata-packed-iq-xmx-v1-build \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx \
  -DMKL_DIR=/opt/intel/oneapi/mkl/2026.1/lib/cmake/mkl
cmake --build /tmp/strata-packed-iq-xmx-v1-build -j1 --verbose
python3 "$D/cases.py" quick
```

CMake only builds this executable and uses actual project's MKL_SYCL::BLAS
link target, sequential host threading, strict math and size_t SYCL ranges.
First use JIT as declared; AOT is a separate later build/arm with full recipe,
never quietly change the binary during a sweep. Budget compile 300s, 16MiB text.
Record successful actual compile/link flags and executable SHA256. Compilation
has not been attempted; local matrix pointer overload support is an explicit
first gate. Fix only this directory if needed, rehash and restart comparisons.

For each emitted case, Main supplies private existing capture directory and
unique prefix; example small GU case:

```
env ONEAPI_DEVICE_SELECTOR=level_zero:gpu /tmp/strata-packed-iq-xmx-v1-build/packed_iq_xmx_probe ref 18 gu 3 32 256 /PRIVATE/ref
env ONEAPI_DEVICE_SELECTOR=level_zero:gpu /tmp/strata-packed-iq-xmx-v1-build/packed_iq_xmx_probe packed 18 gu 3 32 256 /PRIVATE/packed
python3 "$D/compare.py" /PRIVATE/ref /PRIVATE/packed
```

These illustrative commands require Main's supervision wrapper; they deliberately
do not invent a second process-control framework. 30 quick cases cover all six
formats and M1/3/4/8/9 including partial tiles. Then `cases.py full` emits quick
plus 42 actual-role cases: GU rows640/K2560 (N1280), Down rows2560/K640,
M4/8/16/32/64/128/1280. Per-case arm deadline 30s quick, 120s full;
full sweep wall budget 30min, stop when exhausted. Always compare once per arm,
no timing repetitions in v1. Negative CLI checks (separate processes) require
77 for type17, M0, M1281, N17 and K255; ensure no output captures are created.
A declined matrix combination cannot be counted as passing negative shape coverage.

Bound each pair to 64MiB tensor captures, each job to 64MiB text, retained summary
<=1MiB and one distinct failure excerpt <=256KiB. Retire successful pair .w/.x/.y
only after comparison/report hashes are recorded and verified and owned processes
are gone under Main's lock. Record path/size/SHA256/replacement/reason deletion
manifest. Keep first useful divergent buffer only for named unresolved numerical
consumer; owner Main, review at next correctness decision, budget64MiB. Do not
clean an active pair, existing diagnostic, or last required fixture.

## Correctness and interpretation

Both arms have independent input/output guards, unchanged W/X checks and output
NaN initialization; all output values must become finite. Each captures full
W/X/Y, metadata and cold completed wall; comparator verifies identical full input
bytes and reports SHA256, full-buffer max absolute/RMS/relative error and range.
The finite bounded synthetic corpus varies blocks/scales/codes deterministically.
It is not a model-tensor corpus, all-scale-bit decoder qualification or NaN/Inf
contract test; retain the separate parallel-dequant fixture for those obligations.

FP16 operands, FP32 accumulation, changed summation order: exact float bits or
model token IDs are not required here. Comparator screens against the conservative
absolute bound 2*gamma_K*K*max|X|*max|W|+1e-6, gamma_K=K*u/(1-K*u),u=2^-24.
This assumes ordinary FP32 summation and finite products; it is deliberately
loose and is not a model quality tolerance. Record measured error, range and
bound even when it passes. Later gates must use actual GU/Down tensors, narrower
per-output sum(abs(XW)) bounds/quality tolerances, existing SwiGLU/FP16 boundary,
then model quality and current-call-histogram row-weighted performance separately.

Cold wall includes reference allocation/free, first dispatch/JIT and synchronization;
it is diagnostic only, cannot justify a speed claim. No clean repeated timing,
register/spill/occupancy or end-to-end benefit is measured by this proposal.
Default product dispatch remains unchanged. Main must separately audit configured
resolver, callback admission and integration if later gates justify it.
