# R810 standalone vLLM Xe2 native16 proposal

Source only, not integrated or adopted. All compile, link, load, GPU, numerical,
binary DPAS proof, performance, natural-input model quality, and physical
262144-context checks are **UNRUN**. Main's active natural 32K FP32-a/b/BF16 screen
and the current qualified v2 build are untouched. No old v1 ABI snapshot is used.

`xe_gemm_bsd.hpp` is an actual adaptation of Intel's BSD-3 `MoE::xe_gemm`, retaining
block-2D register loads/prefetch, fragment reorder, `XE_DPAS_TT` MMA, split barriers,
and direct typed output. `native16.cpp` adds an independently authored dense grid,
policies and host guards. There is no new joint_matrix kernel. The unmarked vLLM
policy header, grouped scheduler, Torch wrapper, scales/bias, FP8/INT4, clocks,
reference GEMM, event manager and device-0 helper were not copied. TLA can include
generic compatibility utilities transitively; this adapter calls none of their
queue/device manager, host math, reference, fallback or emulation entry points.

## Contract for source review

`Gemm(runtime, queue).beta0(problem, tile)` borrows the supplied Runtime and queue,
returns the actual `sycl::event`, and submits one dense GEMM. X is compact `[T,K]`,
W is compact `[N,K]`; both have `problem.input` storage. Y is `[T,ldy]` with its
independent `problem.output` storage. All four FP16/BF16 combinations are present
as real template instantiations. Public BF16 output is directly consumable by a
second BF16 input call. The FP32 accumulator stays in register fragments; final
reorder converts it once to 16-bit storage. No global FP32 tensor, partials,
scratch allocation, host inference arithmetic, transpose, pack, host transfer,
environment mutation, extra queue, automatic retry or alternate library exists.
The independent output combinations still require compile and numerical proof.

The original `Gemm::f16_out16/bf16_out16` and their beta semantics remain unchanged.
This separate API rejects nonzero/NaN/infinite beta before enqueue and offers no
bias/activation/alpha. Positive T is dynamic, including 1/3/17/160/8192. V1 accepts
N multiples of 64 and K multiples of 32; other N/K tails are explicit errors.
No full-K tail claim is made. `ldy=0` means N, otherwise `ldy>=N` and `ldy%8=0`.
Inputs and output bases must be 64-byte aligned. Output must not overlap either
input; read-only X/W aliasing is allowed if both spans are valid. The full output
rectangle, including last-row padding, must be registered and live.

Block-2D surface width is `2*K` for X/W and `2*N` for Y; pitch is `2*K` or `2*ldy`.
The adapter requires width/pitch/height `<=0xFFFFFF`, width multiple of four,
and pitch multiple of 16 bytes. It additionally limits complete byte rectangles
and global work-item count to INT_MAX to bound CuTe's int coordinate products.
Multiplication and pointer-span arithmetic are checked. `validate_spans` and
`validate_device_span` check alignment, overlap, actual registered allocation
extent, device and context, rejecting ordinary/foreign/host-USM pointers. Raw
USM ownership is not inferred from pointer type; callers must keep Allocation
owners alive until completion and prevent concurrent destruction/reallocation.

The constructor checks the supplied Runtime domain, in-order property, Intel
Level Zero GPU, device USM, actual `architecture::intel_gpu_bmg_g21/g31`, SG16,
and at least 128 work-items per group and last local dimension. Queries occur
once per object; unsupported architecture/query throws. FP16 support is cached
and checked where used. No object or global owns a runtime or a hidden queue.
Runtime/queue references must stay at their original addresses. SYCL does not
expose which Runtime installed a queue's async handler: using a queue created
by that Runtime is a caller obligation in addition to the checked domain.
Linux LP64 is explicit because the pinned TLA payload builtin takes `long`.
Windows LLP64 is blocked by static_assert, not silently truncated.

Ragged M uses real logical `(T,K)` and `(T,N)` surfaces in TLA. Tile coordinate
tensors may extend past M; hardware block-2D zero-fill/suppression handles them,
with no padded allocation fiction. The pinned source documents that behavior
and binds width/height/pitch to the original tensor. All work-items take identical
K loops and split barriers; no early per-lane return is introduced. Initial
prefetch is bounded by `min(3,K/32)`, so K=32/64 never indexes a nonexistent
prefetch tile. Upstream's unbounded three-prefetch behavior is not retained.
These are source arguments, not executed tail/fault validation.

Two explicit geometry variants use the public TLA helper: compact 16x64x32 with
four subgroups (64 work-items), and wide 128x128x32 with eight subgroups (128
work-items). The caller selects the variant, without a tuned dispatch threshold.
The compact geometry is a reported public shape fact; the wide choice is an own
qualification alternative. No policy implementation was copied. Nominal FP32
accumulator payloads are 4 KiB and 64 KiB per workgroup respectively (64 and 512
bytes per lane), before operands, reorders, replication or compiler temporaries.
The core requests no SLM. These estimates do not establish register allocation,
spill traffic, occupancy, numerical behavior or speed; GRF256 is a request.

## Main-only reproduction recipe (not executed)

Use the already qualified official LLVM toolchain. Do not use XeStrata source
or any helpers outside its approved `.tools` compiler installation. All environment
variables, including provider/driver/profiler settings, must be fixed in the native
child before exec; never switch process environment during runtime lifetime.
The commands below do not install/download dependencies or alter any production
build. Put the build directory under Main's owned external artifact root and
serialize it with active tests/GPU work.

```sh
P=/home/yayoi/orca/workspaces/Strata/sycl
R810_PACKAGE="$P/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-xe2-native16-v1"
R810_TLA=/home/yayoi/ghq/github.com/intel/sycl-tla/.worktree/strata-native16-87f
R810_CXX=/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install/bin/clang++
R810_BUILD=/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-r810-vllm-xe2-build-v1
cmake -S "$R810_PACKAGE" -B "$R810_BUILD" -G Ninja \
  -DCMAKE_CXX_COMPILER="$R810_CXX" \
  -DSTRATA_ENABLE_VLLM_XE2_PROPOSAL=ON \
  -DSTRATA_VLLM_XE2_BUILD_FIXTURE=ON \
  -DSTRATA_SOURCE_DIR="$P" -DSYCL_TLA_SOURCE_DIR="$R810_TLA" \
  -DSTRATA_XE2_AOT_DEVICE=bmg-g21-a0
cmake --build "$R810_BUILD" --parallel 1
```

CMake is default OFF, uses only explicit existing paths, and checks manifest
source hashes plus exact TLA revision. It never adds TLA's CMake project,
FetchContent, an install target, network work or an engine hook. Only TLA's
`include` tree is needed by the conservative include inventory. The separate
fixture compiles the pinned current Runtime source into its executable; do not
link that duplicate Runtime registry into the engine. An eventual engine
consumer should link only `strata_vllm_xe2_native16` with its existing Runtime.

vLLM sets `DPCPP_SYCL_TARGET=intel_gpu_bmg_g21` by default when Xe3p is off.
Its actual Xe2 shared-library recipe uses `-fsycl-targets=spir64_gen`, backend
`-device ...`, split-barrier/block2D/matrix translator extensions and
`-cl-intel-256-GRF-per-thread`. This package keeps those relevant flags and
`-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_ENABLE_HEADERS_ONLY`, C++20,
and the kernel subgroup16/GRF256 properties. It adds strict FP compilation for
qualification. `CUTLASS_VERSIONS_GENERATED` is intentionally omitted because
this header-only extraction neither uses `cutlass/version.h` nor generates its
version include. Torch, `-lze_loader`, `-flink-huge-device-code` and parallel
device-link jobs are not needed by the extracted API and are not imported.
Compiler/ocloc acceptance of these flags and TLA builtins is still UNRUN.

Under Main's bounded native-child supervisor, run suites separately with an
explicit device ordinal; no fixture chooses device 0 implicitly. For example,
the child argv is `strata_vllm_xe2_numerics --device 0 --suite smoke` after Main
has selected ordinal 0. This document does not authorize execution during the
active model screen. Record actual compiler/runtime/driver and binary identity,
commands, environment and outcomes with the usual fault/journal boundaries.

- `guards`: malformed spans, overflow/shape/type/beta errors, wrong context,
  host USM, alignment/overlap and output immutability.
- `smoke`: both tiles, all four IO combinations, T=1/3/17/160, K=64 and padded Y.
- `edges`: K=32 and 64, exact FP32 cancellation, even/odd positive/negative RNE
  midpoints, NaN/Inf/zero, min normal and subnormal classification with raw bits.
- `chain`: same-queue FP16/BF16 -> BF16 -> independent FP16/BF16 output, with
  no intermediate host read, FP32 materialization or conversion kernel.
- `full`: both tiles and four IO combinations at T=1/3/17/160/8192 and
  (N,K)=(1280,2560)/(2560,640)/(640,1280), compact and strided outputs.

The dyadic oracle is CPU FP64 closed form, separate from the implementation's
tile/layout/reduction. The edge oracle decodes and rounds bits independently;
finite chosen sums fit FP32 exactly and need no broad numerical tolerance.
NaN compares by class, Inf by sign, zero by magnitude (dot accumulation sign
is not fixed). Subnormals are initially compared against **strict gradual
underflow**; actual DPAS/input/conversion FTZ behavior is **UNESTABLISHED**.
A mismatch fails and prints expected/observed classes and bits; do not convert
that to a pass or weaken the old test suite. Resolve the target contract before
adoption. No additional host FP32 reference GEMM is substituted for this kernel.

Each asynchronous fixture declares its checked drain after all USM/host-vector
owners. It drains before success or exception unwinding; a failed drain exits
without freeing owners. A process timeout does not authorize buffer release.
No source-only review substitutes for this lifetime/fault check on hardware.

## Remaining gates and retention

Main must review the complete source/provenance, compile every real IO/tile
instantiation, inspect binary/device-code proof of DPAS and block2D (including
absence of emulated host arithmetic), run the suites and unchanged old 68-case
GEMM regression, and resolve underflow behavior. No DPAS proof file exists yet.
Only then compare same-queue native resident X/W semantics with the current
native16 and old oneMKL FP32 routes, including conversion/residency, all samples,
CPU submission vs time in GPU waits, device timing and whole dependency path.
`Gemm::beta0` is the source call hook for that operator benchmark; no benchmark
controller, timing loop or environment switcher is included here.

Mixed output is source-instantiated, not upstream-public-dispatch support.
TLA default selectors for row-major W, padded Y and both layouts still need
compiler and numerical qualification. No compile issue is disguised by a
permanent-false support gate. Model/task quality, physical 262144-context
correctness, scheduling/placement, and >600 then 1000 tok/s long-prefill goals
remain independent. The current FP32 reference/default stays unchanged.

The only persistent deliverable is this <=400 KiB package plus the authorized
official external clone and detached source worktree. No binaries, traces,
tensors, logs or duplicate source archives were created or cleaned. Keep the
TLA worktree until Main finishes this source qualification; then review its
consumer before retirement. Record future results in compact committed reports
through Main; source availability and history are not performance adoption.
