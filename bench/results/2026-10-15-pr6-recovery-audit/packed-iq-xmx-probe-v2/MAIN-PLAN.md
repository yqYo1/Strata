# Packed IQ XMX proposal v2 — Main rejection correction

**All builds, tests, comparisons, GPU work and runtime fault injection NOT_RUN.**
Only this new v2 directory was written. No product edits, v1 changes, branches,
Git operations, compilers, builds, tests, GPU calls or process controls occurred.
Main's live diagnostic remains the sole active GPU work.

Main rejected v1 before execution: Guard's unconditional destructor sycl::free
and stack queue could unwind after an exception with completion unknown. v1 is
preserved byte-for-byte with its original source-manifest. Its successful waits
did not make exception paths safe. Main also identified default ostream precision
rounding operand bounds down and the incorrect first-stack icpx2026.1 recipe.
This is a new proposal, not a retroactive pass or repair of v1's recorded status.

## Compact changes

- `probe.cpp`: heap Runtime/queue and fixed-capacity allocation registry retain
  all USM beyond Guard scope. Guard destructors do no runtime/free work. The
  deleted Runtime destructor prevents accidental stack ownership/deletion.
- One explicit `retire()` performs checked queue completion plus sticky async
  fault check before any free or queue deletion. All successful submissions use
  the same checked queue wait. The async handler stores its first exception and
  a sticky atomic fault flag; it cannot silently consume a failure.
- Any exception after Runtime creation (including allocation, submit, wait,
  oneMKL, output capture, metadata and retirement exceptions) reaches fail-stop:
  original exception printed, flushed, `_Exit(86)`, no queue/USM unwinding and no
  retry/drain/recovery attempt. Queue/USM owners remain live when completion is
  uncertain. Host-only vectors/streams may unwind; they own no device work.
  Queue-construction failure has no submitted work or USM; Runtime still remains
  alive until process exit. Successful retirement is the sole normal free path.
- Failed waits do not establish completion and never proceed to free. Exceptions
  during free/destruction occur only after completion was certified; remaining
  resources stay owned until fail-stop. Main must reap the native process and
  inspect driver/kernel faults; `_Exit` is not a GPU-health or cleanup pass.
- Registry reserve precedes queue/provider initialization; capacity checks
  precede malloc. Registering a successful pointer cannot allocate. No temporary
  owning queue, context or USM smart pointer can unwind behind this policy.
- Metadata uses double `max_digits10` for round-trip max_w/max_x and wall values.
  Files are explicitly closed and checked. Success metadata is published only
  after retirement and contains `queue_retired:true`; the v2 comparator requires
  it. Main still requires independent normal-exit/reap/fault receipts.
- Matrix admission now requires an explicitly advertised exact M8/N16/K16
  combination with FP16 A/B and FP32 C/D, shared USM, FP16/matrix aspect, SG16,
  WG>=16 and SLM>=1280. Max-only advertisements conservatively decline (77),
  even if a shape might actually work. Query exceptions propagate with original
  detail rather than being swallowed as unsupported shape.
- CMake requires explicit baseline compiler/MKL pins and rejects mismatched
  paths. No default alternate stack is selected.

`source.diff` gives the v1->v2 probe, comparator and CMake differences. The
matrix algorithm, six types, RNE conversion, GU interleaving, small/full case
coverage, FP16/FP32 numeric contract and full-buffer oracle remain as v1.
Reuse v1 `cases.py` read-only; use the **v2** probe/comparator/CMake here.

## Source-only matrix API audit

Read the actual custom compiler headers at:
`/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install/include/sycl/ext/oneapi/matrix/`.

`matrix-unified.hpp:232` provides the multiplicand joint_matrix_load overload
for `multi_ptr<T,Space,IsDecorated>`; `:344` provides the accumulator store
counterpart. Both reject private_space, and forward the decorated pointer's
address space to cooperative matrix builtins on the Intel path. The proposal
passes local_accessor.get_multi_ptr<decorated::no>(), preserving local_space,
with row-major strides16. Local A/B are half and C is float. All matrix calls,
barriers, shapes and loop trip counts are uniform within the sole SG/WG.
`query-types.hpp` exposes exact/max M/N/K and A/B/C/D type fields; v2 does not
infer exact support from max fields. No claim of compiled overload selection,
backend lowering, device admission or numerical behavior follows from this read.
The existing configured prompt_attention.cpp and matrix_probe.cpp establish
8x16x16 as the local candidate shape; they do not certify this new local-memory
staging kernel. A decline ends this experiment rather than substituting fallback
and labeling it packed-XMX.

## Baseline-first Main commands and pins

The v1 icpx command is rejected and must not be used for the first comparison.
Main's `current-diagnostics-build-v1.json` and `layer-major-build-v1.json` record
this configured custom compiler path (clang22 stack):

```
/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install/bin/clang++
```

They use `/opt/intel/oneapi/mkl/2026.1` and strict SYCL flags. Main must pin actual
baseline compiler/header/oneMKL/SYCL/UR/Level-Zero loaded paths and hashes in its
receipt before configure, including resolved realpaths and effective include,
library and loader environment. Old `build-sycl`/`build-sycl-aot` caches point to
other compilers; do not use their values as this baseline's proof. Existing MKL
interface, threading, C++ ABI, loader search and device selection must match
baseline. No product or other-stack static archives are linked here. Main may
add missing baseline ABI/compile/link flags to this proposal only after recording
those exact flags; it must not compensate for a missing header by mixing SDKs.

Illustrative configure (Main resolves both baseline variables from its receipt):

```
D=bench/results/2026-10-15-pr6-recovery-audit/packed-iq-xmx-probe-v2
cmake -S "$D" -B /tmp/strata-packed-iq-xmx-v2-build \
  -DCMAKE_CXX_COMPILER="$BASELINE_CXX" -DBASELINE_CXX_COMPILER="$BASELINE_CXX" \
  -DMKL_DIR="$BASELINE_MKL" -DBASELINE_MKL_DIR="$BASELINE_MKL"
cmake --build /tmp/strata-packed-iq-xmx-v2-build -j1 --verbose
```

CMake path equality is an anti-accident check, not an ABI/hash audit. Main retains
responsibility for the pinned effective stack and matching baseline JIT/AOT mode.
A new compiler/oneMKL/runtime experiment comes later, independently owned and
labeled, with new pins and numerical comparisons; never silently substitute it
on compile/runtime failure. Preserve original errors and stop.

Run v2 binaries with the v1 CLI/cases, using v2 compare.py. Before any actual
launch, Main should source-review the retirement/error paths, then add bounded
host-runtime fault injection to exercise submit/wait/malloc/file/metadata paths
under its serial controller if required. No fault injection exists or ran in
this v2 proposal. Error exits have no successful completion receipt; never feed
partial or old output files to the comparator. Use a new private prefix per arm.

All other v1 MAIN-PLAN supervision, immutable separate-process environments,
commands/case arguments, output guards/unchanged checks, numerical interpretation
and retention rules apply, with these compiler and lifetime corrections taking
precedence. Budgets remain compile300s/16MiB, quick30s per arm, full120s per arm,
30min sweep, 64MiB tensor pair,64MiB text/job,1MiB compact summary,256KiB distinct
failure excerpt. No work starts until Main releases the current diagnostic and
its existing gates. No performance/adoption claim is made.
