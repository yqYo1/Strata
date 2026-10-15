# Native BF16 selected-cell attention source proposal

Status: source proposal only. Build, compile, tests, GPU execution, tracing,
numerical results, faults, performance, model quality and physical 262144-context
qualification are **UNRUN**. No product files, branches, services or process
environment were changed. Main owns integration and all execution.

Base: `375a6d6ee905859ef176997262e3baef6e739ad2`. Production inputs were read with
`git show` at that revision. The attention source SHA256 is
`c88bd802d17ec5f9e1df05eee5581c9f50be86e87b1a454783b5c75aca55be3d`.
`proposal.patch` is generated against these exact bytes, rather than the moving
worktree. Apply/reconcile that patch after complete review; do not copy whole
files over Main's concurrent changes. The unchanged root `CMakeLists.txt` copies
are resolver context and are not in the patch. The configured SYCL source list
and the added manual fixture target are in `cmake/sycl_backend.cmake`.

## Interface and integration boundary

The patch adds `qsa_prompt_attn_bf16_batch` under the SYCL header guard and a
separate implementation. Existing `qsa_prompt_attn_batch`, its helpers, all
modes and environment dispatch remain a byte-identical prefix of the source.
No caller, prefill, GEMM, HC, native16 helper or GDN file changes. Only an
explicit future native16 route can call this entry; it does not select itself.

Query and output are BF16 bit patterns in `uint16_t[nq,24,256]`. Cache layout is
the existing Mode1 INT8 K/V plus FP16 scales per 64 dimensions. Mapping0, 32-cell
chunks, 12 active plus four padded head rows, one WG128 per (query, KV head),
subgroup16, and reuse of one KV tile across all 12 heads are retained. There is
no split into separate head8 work-groups, no pending query-register proposal,
and no global FP32 query, output, intermediate tensor or scratch allocation.

The new API requires `QsaPromptBf16Bounds{logical_pages,physical_pages}` before
`stream`. This is an explicit new integration obligation: the old pool API has
no allocation extents, so it cannot validate the full cache span. Main must
supply actual accessible page-table entries and physical pool capacity from
owners, rather than infer them from width or logical context. Different logical
and physical counts are supported. Each physical pool holds
`physical_pages*2*page_size*256` bytes; each scale pool holds
`physical_pages*2*page_size*4*2` bytes. Dimensions and byte products are checked.

Before submission the supported path checks alignment, every output/input
span overlap, and USM context registration at the first and final byte of each
full span. Device/shared allocations must belong to the queue device; host USM
requires the device's host-USM aspect. SYCL pointer queries do not expose an
allocation-size/identity query: this is **not** proof that caller-supplied extents
are truthful or that two registered endpoints belong to one contiguous owned
allocation. Main must retain the allocation-owner size/lifetime guarantee; no
raw Level Zero query, new queue or runtime registry change is introduced.

`nq==0` is a successful no-op. Otherwise negative/too-large nq, cap<=0,
cap>INT_MAX, unsupported geometry/Mode1 pointers/Q4 combinations, invalid bounds,
or missing device capabilities return false before submission. On supported
geometry/device, invalid pointers/spans throw `std::invalid_argument` before
submission. Width is read from each device step, not the capacity: negative or
width>cap becomes zero as in the baseline, including width2052 at cap2051.
Negative selected IDs/pages and IDs/pages outside declared extents are missing.
Valid IDs and pages preserve baseline mapping. Causal masking is already in
selected IDs; the kernel does not infer an additional mask from step position.

## Arithmetic and rounding contract

1. Each subgroup visits two head rows, reads contiguous BF16 lanes, widens
   individual values for FP32 per-head max reduction, and records one bounded
   power-of-two shift. Q is written to a single scaled FP16 local tile. No second
   low part or global conversion is used. Per-head scaling avoids a large head
   suppressing a small head in the same KV group. It cannot preserve arbitrary
   exponent differences within one head; FP16 underflow there is a numerical
   gate, not an accuracy claim.
2. Exact INT8 codes enter FP16 XMX operands. Each 64-dimension QK partial uses
   FP32 accumulation, FP32 decoded K scale and the inverse power-of-two shift
   with 1/sqrt(256)=1/16. `ldexp` is applied last so a separately stored inverse
   scaling factor cannot underflow before multiplication. Scores, online max,
   exponential, sum and rescaling coefficient remain FP32.
3. For V scale group g and cell c, let m=max(abs(vscale[c,g])) for the current
   chunk. A single FP16 probability tile stores
   `half((exp_score[c] * (vscale[c,g]/m))*16384)`, with zero for m=0.
   Division precedes multiplication. For finite scales, abs(vscale/m)<=1 and
   0<=exp_score<=1, so the operand is bounded by 16384 even at FP16's minimum
   nonzero scale 2^-24 or maximum finite scale 65504. There is no infinite
   reciprocal or high+low probability reconstruction. Small probabilities can
   underflow in this one FP16 part; the FP32 normalizer still uses unrounded
   probabilities. This differs numerically from the reference by design.
4. PV uses FP32 matrix accumulation, applies `m/16384` per group in FP32, and
   rescales the persistent local/register accumulator with the online alpha.
   Final normalization is FP32; output is stored directly as BF16 using
   round-to-nearest-even, including quiet-NaN handling. BF16 and FP16 storage
   each use two bytes; this design makes no universal accuracy or speed claim.

All-zero scale groups produce zero. All-missing/zero/invalid-width selections
produce zero, including when Q is nonfinite. Nonfinite live Q diagnoses BF16
quiet NaN on its affected head; nonfinite live K/V scales diagnose NaN across
the sharing KV group, without passing infinity to the FP16 probability cast.
A NaN from FP32 QK overflow cancellation is also diagnosed. Equal infinite
score maxima share mass without exp(inf-inf); finite scores supersede negative
infinity mass and positive infinity scores supersede finite mass. The fixture
checks uniform overflow-score cases against FP64, but arbitrary logits that
exceed FP32 range are not numerically qualified.

## Resource and collective reasoning

Declared per-WG SLM, calculated from accessor extents (not a measurement):

| Store | Bytes |
| --- | ---: |
| Single Q FP16 tile [16,256] | 8192 |
| Reused K/V FP16 tile [32,256] | 16384 |
| Single P FP16 tile [4,16,32] | 4096 |
| FP32 scores [16,32] | 2048 |
| FP32 K and V scales [32,4] each | 1024 |
| FP32 max/sum/alpha [16] each | 192 |
| FP32 group factors [4] | 16 |
| INT64 resolved rows [32] | 256 |
| INT32 shifts and query flags [16] each | 128 |
| Total declared | 32336 |

A static assertion checks the formula. The gate requires local_mem_size>=36432
(32336+4096 conservative allowance), max WG/item size>=128, subgroup16, GPU,
FP16 and Intel matrix aspects, and an actual supported tuple
A=FP16/B=FP16/C=FP32/D=FP32/M8/N16/K16. Exact discrete sizes and continuous
maxima are handled separately. This does not replace compilation/device-kernel
resource validation by Main. No BF16 XMX tuple is required because BF16 I/O is
scaled to FP16 locally.

Matrix objects are ordinary scoped `joint_matrix` objects, never copied or
indexed through private fragments. Loads use typed local multi_ptrs and
row-major strides. Intel coordinate-aware apply supplies matrix row/column;
generic binary apply combines matching accumulator coordinates. QK is executed
by four whole subgroups, PV/output by all eight; every matrix operation is
uniform within its subgroup. All work-group barriers and group reductions are
encountered uniformly, including missing/invalid-width and padded rows. Before
K overwrites V, the previous chunk has a full work-group barrier; after QK,
there is a barrier before the reused tile becomes V. Bounds ensure every matrix
load covers allocated padded local storage. Output coordinates cover each
active head/dimension once; padded heads do not write.

Only input/output/cache/selection allocations scale with actual chunk/context.
Temporary storage is fixed at the above size, rather than proportional to
logical 262144 context or selected width. No physical full-context test was run.

## Fixture and Main-owned gates

`tests/sycl/prompt_native16.cpp` adapts the explicitly authorized own FP64 oracle
input with SHA256 `31cb331094fc42b77fa5d7babdfb41e7eaacb06c9e83f135ad3bee72f4cc484e`.
Its scalar reference uses actual quantized BF16 queries, exact INT8 codes and
stored FP16 scales; it does not reproduce online/XMX arithmetic. It reports raw
FP64 error, error versus a BF16-rounded FP64 result and the worst ratio to a
predeclared component budget:
`0.001*weighted_abs_V + 2*BF16_spacing(reference) + 1e-12`.
This screens single-part probability and final BF16 rounding; it is not a bound
on general Q compression or a model quality acceptance gate. No byte equality
with the old FP32 math is expected. Batch/single and repeated candidate outputs
are required to match BF16 bits for identical inputs.

Coverage in source: nq1/3; page1/4/16; widths0/1/31/32/33/2051/-1/2052; mixed
missing and fully missing table; recovered valid table/output; causal selected
IDs with populated future cache entries; every output finite and guarded; every
input immutable; per-head very small/large BF16 query ranges; zero scales; tiny
probabilities with FP16 min/zero/max/negative scales; uniform positive/negative
FP32 score overflow; live infinite K/V scales, infinite/NaN Q, missing rows with
nonfinite backing scales/Q, and recovery. Extended smoke also rejects cap0,
invalid nq/geometry/modes/bounds, null/unregistered/misaligned/overlapping spans,
and checks out-of-bounds IDs/physical pages as missing. Rejected calls preserve
the full output sentinel. The new target is EXCLUDE_FROM_ALL with no CTest run.

Main should compile with the existing qualified SYCL toolchain, then first run
`sycl_prompt_native16_test --smoke --dump <owned-bf16-dump>` in a finite owned
process using the already qualified immutable pre-exec environment:

```
UR_ENABLE_LAYERS=UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION
UR_LOG_TRACING=level:info;flush:info;output:stderr
```

Keep the layer value comma-separated; shell-quote the logger value if a shell
constructs the environment. Record full effective environment, source/binary
hashes, device/driver/toolchain, bounds, outcomes and fresh fault status. First
trace budget is 64 MiB, with a finite deadline set by Main's existing supervisor.
A budget/deadline stop is incomplete. No trace or tensor was created here.

After a successful traced smoke, Main can run the full fixture, then separate
clean processes with `--bench 128 5 --dump <owned-path>` and
`--bench 256 5 --dump <owned-path>`. Repetition count is explicitly bounded1..20.
Each repetition prints an individual submission-to-completion wall sample;
validation/hashes run outside those timed samples. All benchmark rows have
width2051 and valid/masked selections, not mostly empty queries. Full output
payloads are dumped as BF16 [nq,24,256], and shape, byte offset/size, input and
guarded-output FNV1a64 fingerprints are printed. Main must compute/pin external
SHA256 for dumps and retain individual samples; FNV is a quick fingerprint,
not a cryptographic artifact hash. Benchmark FP64 coverage is queries0/1/2/last,
while finiteness/guards/input identity cover every output/input. Cross-process,
old-route and whole-model comparisons remain Main obligations.

Before adoption Main must reconcile allocation bounds and the explicit typed
producer/consumer route, verify old FP32 math/dispatch preservation after merge,
review actual device arithmetic and kernel resources, inspect numerical error,
measure complete dependency timing, and gate teacher-forced/task quality plus
physical 262144-context/shared-state/RESTORE behavior. All remain UNRUN.

## Provenance and retention

`provenance.json` pins the implementation base, explicitly permitted external
fixture, design-only reports and official API documents. The official matrix
document records include immutable Intel LLVM revisions and retrieved byte
hashes. Only this project's own MIT source plus original worker code implement
the change. SGLang/vLLM reports inform dtype/FP32-accumulation boundaries only;
no Apache/BSD source or sample implementation was copied. No XeStrata source,
helpers, tests, archives or pending head8/query-register proposal was read or
used. New source follows the repository MIT license.

The package is bounded to 16 MiB. It contains source/base/patch, compact
provenance and review notes only, with no builds, binary, tensor, measurements,
profiler data, service state or cleanup activity. Main is the consumer and the
next review point is complete patch review before integration/build. Once a run
is performed, Main follows docs/ARTIFACT_RETENTION.md and retains compact
commands/environment/samples/results plus one diagnostic for each unresolved
mechanism. This package's absence of results is never a pass.
