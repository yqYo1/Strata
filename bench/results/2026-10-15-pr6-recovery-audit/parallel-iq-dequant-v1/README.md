# Parallel IQ dequant-to-FP16 proposal v1

2026-10-15. **Proposal only: UNAPPLIED, UNBUILT, UNTESTED. Every runtime
qualification is NOT_RUN.** Only this new directory was written. No product,
CMake, existing fixture, binary, build, test, CPU benchmark, GPU, server,
profiler, process/service inspection, Git operation or cleanup was performed.
Python was used only to author text, transcribe the proposal and hash files;
neither fixture, comparator, syntax check nor application check was executed.

Main must close **both** current Direct0 and Direct1 full clean benchmarks of
binary SHA256 `92acf71e6b4fd914f68c8cefcae7e03f170adf62b5e3b94cf31af747344183f7`,
and verify all server/client/controller owners retired, before applying this
patch or starting any build/test (including CPU). A Direct0 completion alone
does not release the freeze. This proposal does not assert their current state.
Main serializes every subsequent build, test, fault check and GPU measurement.

The project-only strata-upstream-sync skill, AGENTS.md, artifact retention,
research-index.json, R610 and main-review-round609-610.ja.md were read first.
This implements R610's exact FP16 materialization candidate by transcribing this
fork's scalar formulas. It imports no reference engine or dot-product decoder.

## Change and eligibility

`source.diff` changes only `src/kernels/sycl/iq.cpp` and adds
`include/strata/sycl/iq_parallel_decode.hpp`. Full proposed file bodies are in
`proposed/`. `source-pins.json` fixes old/new SHA256 and decoder dependencies;
`artifact-sha256.json` fixes this delivery. Main must verify the old pins and
review the diff before applying it. There is no CMake edit.

`STRATA_SYCL_PARALLEL_IQ_DEQUANT` is read once. Only the exact value `1` opts in;
unset, empty, `0`, `01`, `2`, `true` and whitespace variants remain OFF.
Set it in each native child's environment before exec. No in-process environment
mutation, flag setter, diagnostic, cache or allocation is added.

The new half kernels are selected only after all existing type, shape,
alignment, checked_count, span, overlap and address-overflow validation. Generic
F32, embedding, MMVQ and all unsupported half types retain their original kernel
bodies. OFF performs the old single launch and the old finish call. ON performs
one replacement launch and the same finish semantics: no new queue, event wait,
barrier, host copy, scratch allocation or downstream GEMM change.

| Type | Format | Block elements / bytes | Actual target role |
| --- | --- | --- | --- |
| 18 | IQ3_XXS | 256 / 98 | GU |
| 22 | IQ2_S | 256 / 82 | GU |
| 21 | IQ3_S | 256 / 110 | GU |
| 23 | IQ4_XS | 256 / 136 | GU |
| 20 | IQ4_NL | 32 / 18 | Down |
| 42 | custom Q2_0 | 64 / 18 | Down |

All six types are admitted by both public half entry points when their existing
shape checks pass. Eligibility does not depend on routed row count, compact
mode, N=2560, FF=640, or ne<=64. The fixtures include the actual dimensions;
these are synthetic operands and establish no model quality. Other types
2/6/7/8/11/12/13/14/16/17/29/30 explicitly decline to the old half path. All
invalid types still throw before dispatch.

One work item owns one output `uint16_t`. Adjacent IDs write adjacent half
elements rather than whole-block-strided stores. The generic index maps to
`block=i/B, j=i%B`. GU maps `r=i/width, col=i%width`, reads gate for even r and
up for odd r, source row `r/2`, and preserves `[gate row0, up row0, ...]`.
There is no per-item array, local memory, subgroup collective or fixed local
size. A plain range lets SYCL select workgroups; no new device admission is
needed. Coalescing is a source address property, not measured memory traffic.

The tradeoff is repeated codebook/scale/packed-byte reads by adjacent work
items, more work items, and integer index arithmetic. Compiler commoning,
cache behavior, actual register/spill use and occupancy are unknown. A modest
lane-group decoder may later beat this baseline; none is claimed here.

## Arithmetic and bounds review

The helper uses the same tables, scale conversion and left-to-right FP32
multiplications as decode_block. IQ3_S keeps `db * grid * sign`, IQ3_XXS keeps
`d * (0.5 + scale) * 0.5`, IQ2_S keeps `d * (0.5 + scale) * 0.25`, IQ4_XS keeps
`d * (ls-32)` then `* codebook`, and Q2_0 keeps `float(code-1) * d` in that
operand order. Signs are never folded into scale or grid. The final conversion
is the existing f16_from_f32, preserving signed zero, subnormals, finite
overflow, infinity and its canonical NaN payload/sign policy. The equivalence
is a source argument pending CPU and device bitwise gates, not a result.

| Type | Last byte read | Largest variable shift | Signed interpretation |
| --- | --- | --- | --- |
| 18 | 97 | u32 aux assembly 24; sign field 21; grid 24 | final FP32 sign multiply; parity from existing helper |
| 22 | 81 | u64 grid 56; packed high 6 | final FP32 sign multiply |
| 21 | 109 | u32 grid 24; sign/high 7 | positive odd scale then sign multiply |
| 23 | 135 | 16-bit scale high 14 (promotes to int) | ls converted to int before subtracting 32 |
| 20 | 17 | nibble 4 | int8 codebook converted to float |
| 42 | 17 | code 6 | int code minus 1 before float conversion |

Each j is formed by modulo B, so these shifts are in range. Table indices are
bounded by 255/511/1023 or 15. Packed reads are bytes; no stronger alignment is
introduced. Generic source offsets are below the already-validated packed
span. GU width/B*S equals checked rowbytes, and row/2 is below rows. The
existing count<=SIZE_MAX/4 check bounds 2*count and all output addresses. The
actual configured size_t SYCL range option is retained. Caller allocations
must still match the spans they claim; this patch adds no allocation lookup.

## Actual source resolver and context

The current native build receipt `../cpu-gu-reuse-build-v3.json` identifies the
92acf71e binary and native build directory. Its current CMakeCache,
build.ninja and compile_commands were read without invoking CMake or Ninja.
CMAKE_HOME_DIRECTORY is this worktree; STRATA_ENABLE_SYCL=ON;
CMakeLists.txt includes cmake/sycl_backend.cmake. That file builds
strata_kernels with this worktree's `src/kernels/sycl/iq.cpp`, links the SYCL
prefill library, and builds `strata` from `src/program/generate.cpp`.

The native `build.ninja` entry resolves this exact iq.cpp and has
`-O3 -DNDEBUG -std=c++20 -ffp-contract=off -fsycl
-fsycl-id-queries-range=size_t -Wall -Wextra -Werror`. Compiler is the pinned
Intel LLVM v711 clang++ path recorded in source-pins. Device arch is empty/JIT;
double-GRF is OFF. The similarly named checkout `build-sycl/` is a different,
older icpx/Make configuration and is not the active 92acf71e build resolver.
These are manifest observations, not a fresh binary reconstruction.

The native expert fallback in src/prefill/prefill.cpp invokes GU dequant then
generic Down dequant and the same FP16 operand/library GEMMs. Generic half is
also used by quantized projection in src/prefill/sycl/gemm.cpp; its profile time
must not all be attributed to Down. R610/Main's profile records GU 91,880 calls
and 22.4007 s, generic half 94,580 calls and 12.0643 s. These are aggregate kernel
durations over the instrumented process, not exclusive prefill wall time or an
end-to-end improvement ceiling. No bandwidth/performance estimate is adopted.

The existing Arc path already implements this scalar GGUF materialization.
The proposal changes that component only. It neither imports a root CUDA
generator nor addresses or closes any remaining shared CLI/API port obligation.

## Files and status

`cpu_fixture.cpp` compares the new scalar element helper with the original
decode_block then f16_from_f32. It crosses 1,024 pattern IDs with 18 edge
scales/type and separately sweeps all 65,536 half scale bit patterns/type,
including every NaN payload/sign, negative zero and both subnormal extremes.
Every output index is checked, all actual grid indices/sign bytes/scale fields
are visited, and GU row mapping is checked at rows1/2/3 and widths256/512/2560.
Source and output guards are checked. This is structured coverage, not the
Cartesian product of every packed bit pattern. CPU offsets0..3 exercise byte
loads; valid product offsets and invalid dimensions/spans are tested by GPU
fixture entry calls rather than a copied CPU validation implementation.

`gpu_fixture.cpp` runs one immutable OFF/ON arm. Quick mode has 42 half cases:
all six candidate types at source offsets0/2, GU interleaving, and generic/GU
declines for every other supported type. It also checks all six F32 paths and
20 invalid calls (shape, custom Q2 width, alignment, null, overlap, size/address
overflow). Full mode adds four full 640x2560 GU and two 2560x640 Down cases.
It captures every half output plus guards, every input plus guards and the
CPU oracle. Normal paths explicitly drain and retire USM owners. Failure exits
without freeing possibly live USM or unwinding the heap-owned queue; Main must
supervise and record ownership/driver outcome before continuing.

`compare_captures.py` is offline and requires complete case sets. It compares
full OFF/ON output bytes including NaNs, all inputs, guards, CPU oracles and
F32 outputs. CPU/device NaN sign latitude is recorded separately while requiring
canonical payload; there is **no** OFF/ON NaN exception. It reports array parity
only: Main must attach normal exits, source/binary/env identities, owner
retirement, actual route and fresh fault-check receipts separately.

`main-validation-plan.md` gives serial commands, identity/retention rules and
model/performance/physical256K/RESTORE/cancel gates. Every listed gate is NOT_RUN.
No default change, numerical qualification, speed adoption or quality claim.
