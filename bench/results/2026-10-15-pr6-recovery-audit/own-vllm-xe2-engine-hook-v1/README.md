# R821 source-only production vLLM Xe2 hook

The deliverable is `engine-hook.patch`, an actual optional production backend
for `Gemm::f16_out16` and `Gemm::bf16_out16`. It is not applied, compiled,
configured, syntax-checked, linked, executed or adopted. Every GPU, ISA,
numerical, fault, performance, quality and physical-context gate is UNRUN.
The package is below the requested 300 KiB budget. Main owns full review and
all testing. Production files, R810/R816 outputs, R818, libraries and the
active Main build are untouched; no Git operation or cleanup was performed.

## Review and integration model

`engine-hook.patch` contains all existing-file deltas and all seven new product
files. `files/` duplicates only the small new files for direct review; it is not
an alternative complete source tree. Do not copy files and then apply the same
new-file hunks twice. No entire prefill.cpp, root CMake or other large existing
source mirror is present. `manifest.json` pins old and proposed bytes and exact
R810 imports; `SHA256SUMS` seals the whole package except itself.

Every changed production file hash-matched the qualified v2 source receipt
`own-native16-prefill-build-v2.json`, commit
`9c1a2020897a9451ec616a6bb036a661de9d86b0`, before authoring. The inspected
prefill.cpp also matches that receipt and is read-only. No Git HEAD query was
performed. The coordinator reports a later HEAD: that does not change these
explicit production source-byte bases. Review/rebase hunks against the actual
integration tree; do not replace live files from this package. R818 is a separate
component patch to combine later, not an input to this proposal.

Changed existing files: root CMake option guard; cmake/sycl_backend.cmake
source/link integration; Gemm declaration/state; Runtime header/registry owner
accessor; Gemm init config validation; both out16 route entry points.
New files: exact R810 adapter/header/BSD core, small config declaration, optional
CMake module, 207-header external TLA hash inventory, and production usage doc.

## Concrete behavior

- Build option `STRATA_SYCL_VLLM_XE2_NATIVE16` defaults OFF. Enabling it requires
  explicit `STRATA_SYCL_DEVICE_ARCH=bmg-g21-a0`, nonportable AOT, old double-GRF
  override OFF, and absolute existing `SYCL_TLA_SOURCE_DIR`. No download or TLA
  vendoring is added; the include bytes are pinned to official Intel 87f68506.
- `STRATA_SYCL_PREFILL_NATIVE16_BACKEND=vllm` selects the exact R810 dense
  adapter for both public out16 functions and all four input/output combinations.
  Unset or `own` retains the original joint-matrix route and beta math. FP32 GEMM
  source and old short/decode routes remain as before.
- Tile experiment setting `STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE` accepts
  auto/compact/wide. Auto's initial untuned static rule is T<128 compact and
  otherwise wide. No measured selector or performance claim is made.
- Missing build, malformed setting and unsupported initial queue reject. R810
  preserves all synchronous shape/alignment/overlap/registered-allocation
  extent/domain gates before its GEMM submit. No fallback or 16-as-32 aliasing.
- Runtime ownership comes from the existing Allocation registry, not another
  device selector/provider. Per-Gemm state retains that original Runtime and
  borrows the unchanged engine stream at a stable address. Original in-order
  completion/checked-cleanup and buffer retention duties remain.
- At most one concise route/config line is emitted per process. It identifies
  the existing bulk mode separately from the requested backend and tile, so an
  engine=0 run is not misidentified as a bulk native16 benchmark.

`files/docs/SYCL_VLLM_XE2_NATIVE16.md` is the complete usage, call-site inspection,
contract, lifetime, build flag and license acknowledgement for integration.
`provenance.md` records source origin and the review boundary.

## Material limits for Main

The exact R810 compiler/translator/GRF256 requests remain unqualified. In a
whole-engine binary the global GRF backend option may affect decode/other images
without any source or route change. Record that build difference when comparing.
The registry now retains Runtime shared owners and each existing span validator
returns/discards a copied shared owner; this adds host reference-count overhead
whose size is not measured. There are no new host grouping algorithms.

Long-prefill HC constants and individual routed expert shapes satisfy the static
N/K granularity and row-offset alignment from aligned bases. Actual model
WeightRef dimensions and bases still govern. `native_proj16` column slicing can
violate N%64, 64-byte output alignment or the conservative complete Y rectangle
extent, especially for a weight matrix larger than scratch. Such sites reject;
no generalized sliced-model compatibility is claimed. Refer to the production
doc's table for exact T/N/K/ldy mapping. R810's INT_MAX rectangle guard remains,
so there is no new full physical 262144-context claim. No grouped-MoE/Torch port
or R818 attention change is included.

Main's next actions are full source/license/patch review, R810 qualification and
actual enabled whole-engine build/link/flag review, then all type/tile guard,
numerical, chain and unchanged-GEMM gates in serialized owned jobs. Only after
those gates should Main run bounded model/tensor/quality/whole-dependency timing
comparisons, recording route/config and any global GRF effects. Resolve failures
without weakening previous results. Keep the backend opt-in until Main decides.

No generated binaries, captures, profiler data, process inspections, syntax
checks, tests or performance samples exist for this dispatch. Retain only this
compact package for Main's integration/qualification review, then record the
component decision at Main's next committed boundary.
