# PR6 validation LLVM/oneMKL build adapter v1

Status: **SOURCE_ONLY_UNSAFE_TO_RUN**. Worker `task_9a67df5f12ee`, dispatch
`ctx_943bcc6308cf`, 2026-10-16 JST. This delivery adds one default-OFF validation
build mode against immutable base `56d2063fd553f8e1ddd69c44b3990ef6a1f63a5f`
on the existing `verify/pr6-prefill-restore` worktree. Main owns review, every
execution and Git mutation. Configure, frontend, preprocessing, backend,
compilation, linking, tests, native/provider initialization, model and GPU work
are all **NONE** for this delivery. No compiler executable or CMake executable
was invoked. No Git branch, stage, commit or push was performed.

## Inputs and provenance

The assignment was initially delivered as a relative path absent from this
child. A read-only file search found the unchanged spec in the primary Orca
workspace, and the coordinator confirmed that absolute path through the pending
Orca question. No scope change or implementation steering followed. The spec,
existing safety corrections/review v3, harness corrections v2, artifact policy,
wrapper, historical CMake file, checked-DPCT gate, local fork CMake recipe and
installed MKL metadata were read. Companion JSON pins these inputs.

Main's `own-pr6-compiler-driver-probes-v1.json` records driver-only rejection of
exact `-qmkl=sequential` and `-fp-model=precise` by the current LLVM clang++22,
and acceptance of those spellings by the separately unqualified packaged icpx.
Those are prior driver observations, not builds by this worker. The adapter
does not select icpx or alter runtime search paths. Its compiler guard checks
the existing installation's real path and SHA256
`1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb`, representing
Main's IntelLLVM22.1.0 binary receipt. Both C and C++ must use that pinned clang-22 executable
(normally via clang and clang++); an implicit GNU C compiler would reject the
new precise spelling in inherited ggml C options and therefore fails admission.
CMake identification strings are not assumed from the driver banner; Main
must inspect the actual CMake compiler-identification result separately. It makes no additional compiler/provider qualification claim.

The new helper is independently authored under repository LICENSE. The only
local implementation reference is the task-authorized fork
`cmake/sycl_backend.cmake` recipe selecting dynamic, sequential oneMKL and
`MKL::MKL_SYCL::BLAS`. Installed Intel package metadata supplies public target
names, settings and link closure; no vendor implementation/template/header was
copied. The LLVM install path is a dependency path only. No XeStrata source,
helper, test, archive, proposal or protected credential record was read or used.

## Exact source behavior

`validation/pr6-restore/CMakeLists.txt` adds
`PR6_OPEN_SOURCE_LLVM_ONEMKL`, default OFF, includes `llvm_onemkl.cmake` only
when enabled and calls its preparation before historical `add_subdirectory(sycl)`.
It adapts the created targets immediately afterward, before the existing harness
copies `strata_prefill` options. With the mode OFF the historical recipe remains
distinct: original flag spellings and existing historical/source/DPCT gates
are retained. The historical `sycl/CMakeLists.txt` itself is byte-identical to
the assigned base. No harness source or inference math changes.

With the mode ON:

1. Require Linux, the pinned C/C++ compiler and native experts. Refuse enabled
   alternative ggml backends before their configuration, including BLAS, CUDA,
   HIP and SYCL; cache those backend options OFF. Refuse explicit GGML_CPU=OFF.
   This keeps the supplied ggml CPU dependency distinct from SYCL GPU BLAS and
   avoids an additional ggml BLAS provider with a separate threading choice.
2. Select only installed `/opt/intel/oneapi/mkl/2026.1/lib/cmake/mkl`, exact
   package version2026.1.0, and its matching MKL_ROOT. Request dynamic CPU/SYCL
   linking, explicit Intel ILP64 interfaces, and explicit sequential CPU **and**
   SYCL threading. Disable OpenMP offload. Existing MKL targets are refused;
   post-find checks require the selected config path, sequential settings and
   the BLAS/domain, interface, sequential and core targets. No download occurs.
   ENABLE_SYCL_COMPILER=ON recognizes the pinned compiler without adding
   package compiler probes; ENABLE_TRY_SYCL_COMPILE=OFF is explicit. These
   settings are configuration inputs, not validation passes. No process
   environment is readjusted or runtime initialized by the helper.
3. Require exact historical `-fp-model=precise` in COMPILE_OPTIONS on all seven
   engine targets and on ggml, ggml-base and ggml-cpu. Require exact
   `-qmkl=sequential` in all four owning prefill properties: COMPILE_OPTIONS,
   INTERFACE_COMPILE_OPTIONS, LINK_OPTIONS and INTERFACE_LINK_OPTIONS.
   Missing targets/flags fail rather than silently applying an incomplete mode.
4. Traverse the already configured historical directory and descendants,
   including the external ggml subdirectories. In target compile properties,
   replace only the exact list element `-fp-model=precise` with
   `-ffp-model=precise`. Remove only the exact `-qmkl=sequential` element from
   the owning prefill compile/link properties. Finding a separate owner for
   that exact shorthand fails. No substring filtering, unknown-flag stripping,
   source resolver or source-property rewrite occurs. Unchanged properties
   are not reset. Generator expressions and every other list element keep
   their value/order; actual generated-command proof remains Main's gate.
5. Attach `MKL::MKL_SYCL::BLAS` PUBLIC to `strata_prefill`, preserving its engine
   dependency. This carries include/compile usage and the complete package
   domain link closure to final `strata` and sibling `pr6_prefill_api`. The
   harness copies the adapted direct SYCL compile/link options and keeps its
   existing C++20 requirement. It receives the package through its existing
   prefill link. CPU libraries do not acquire the old prefill-only shorthand.

The directory-initialized options matter: changing only `strata_prefill` would
leave the unsupported precise spelling on the other engine libraries and on
ggml C/C++ targets. Changing a directory property after target creation would
not repair those initialized target properties. Recursing over build targets
does not rewrite any external dependency source or installed metadata.
[CMake documents this initialization and dependency-option order.](https://cmake.org/cmake/help/latest/prop_tgt/COMPILE_OPTIONS.html)

Clang documents `-ffp-model=precise` as disabling transformations that are unsafe
for floating-point values while retaining within-expression FMA contraction.
This matches the historical recipe's stated precise-math intent; it does not
claim icpx/LLVM numerical identity, fixed denormal state or bitwise equivalence.
No extra fast-math/strict/FP16/BF16 flags were introduced.
[Clang's floating-point model documentation](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior)
and Main's recorded driver suggestion support the spelling.

Intel documents dynamic linking and the ILP64 exception for icpx with -fsycl
when using -qmkl; explicitly requesting ILP64 preserves that interface choice
instead of relying on the package default. Sequential CPU MKL does not imply
that GPU BLAS arithmetic runs on a CPU, or prove where a particular library
operation executes. That remains an actual provider/backend audit.
[Intel's -qmkl guide](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-0/using-the-qmkl-compiler-option.html)
and the installed2026.1 MKLConfig metadata support these settings.

The installed domain recipe contributes `-fsycl`,
`-fsycl-allow-device-image-dependencies`, its MKL RPATH, mkl_sycl_blas,
mkl_intel_ilp64, mkl_sequential, mkl_core and Linux math/dl/pthread support.
Its C++ usage also supplies `-DMKL_ILP64` and MKL includes. This delivery uses
the official domain target rather than reconstructing those link flags. Main
must inspect the resulting commands and resolved shared objects; no actual
link or ABI result was observed. Source search found the restored project's
oneMKL/DPCT BLAS calls only in prefill/gemm.dp.cpp; no project LAPACK/FFT/RNG/
sparse call was found in the inspected restored include/src trees. This finite
source observation is not a proof of every instantiated vendor-header symbol.

## Preservation and delivery identity

Read-only Git-object comparison against the exact base covered243 existing
tracked validation/report files: only the authorized outer CMakeLists differs.
All216 other validation files and26 existing result files remain unchanged.
This includes the entire198-file historical closure, kernel source,84-TU
resolver, DPCT gate and safety/harness source. The 49 vendor files and1390
llama/ggml/cmake/LICENSE snapshot files were separately rehashed against Main's
dependency-preparation receipt, with zero mismatches. Their source is not
edited. Main's receipt remains the origin pin; this is a byte comparison, not
dependency, license, compiler or execution requalification.

The complete source patch includes the wrapper change and new helper against
the assigned base, without delivery reports to avoid self-reference. It is7815
bytes, SHA256 `c28ea9011ef6976ca70eabbf80b3eee99a084781a9f2fdf787136f1247cc7548`.
JSON pins before/after source identities, references and preservation results.
No test script, log, binary, tensor, trace or profiler artifact was created or
deleted. No cleanup was performed. Main owns integration/commit/push and the
next retention review at adapter acceptance or replacement.

## Remaining Main gates

Main should review the entire source patch, then use a new owned build directory
and the existing pinned dependency/compiler environment with
`-DPR6_MAIN_REVIEWED_BUILD=ON -DPR6_OPEN_SOURCE_LLVM_ONEMKL=ON`, explicit C and C++
compiler paths, absolute PR6_VENDOR_ROOT/STRATA_GGML_DIR, and the existing
native-experts/parity/harness settings. This is a proposed configuration delta,
not an executed command. All prior source and private checked-DPCT gates remain.

First inspect the generated commands, both C/C++ compiler identities and
effective math flags for every engine/static-CPU/ggml/harness consumer; confirm
the unsupported flags are absent, all other SYCL/AOT/backend/source-specific
options survive, and no unexpected backend/library appears. Inspect the
checked-header generation/hash/precedence, source closure, dependencies,
symbol/ODR/ABI/link order, ILP64 defines, sequential CPU MKL and installed
libsycl/UR/Level Zero/MKL runtime closure before accepting an actual build.
Successful configuration alone does not qualify compiler commands or a provider.

Actual build, model-free checked-provider/event/queue/partial-submit/async/
retirement/cleanup gates and D3 compiled rejection are separate next gates.
Only then may Main progress through the prior tiny/odd-tail/model/numerical/
state/quality and long-input measurement plan. Main's earlier parser16-host and
SHA33-host cases are unchanged prior evidence. This worker did not rerun them.
Numerical equivalence, physical262144, export/exit/fresh RESTORE/continuation/
rewind/branch and historical600 reproduction remain unqualified; no driver,
provider, whole-engine or performance approval is delivered here.
