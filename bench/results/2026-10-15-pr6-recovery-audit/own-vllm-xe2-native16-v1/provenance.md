# Exact source and license boundary

This is a source proposal pinned to qualified Strata v2 source
`9c1a2020897a9451ec616a6bb036a661de9d86b0`, official vLLM XPU kernels
`6b30c9b92cc525c32729df988b5f918e7c5661a5`, and official Intel SYCL-TLA
`87f6850680a580654b9ea2c80dbc01aeb36ad231`. The qualified-v2 build receipt
supplies the Strata base; all six consumed/read ABI and reference source files
hash-match that receipt. No project Git operation, old v1 archive, current build
mutation or project source replacement was performed. `manifest.json` pins each
read file and the source package is fully pinned by `SHA256SUMS`.

The reviewed decision support was [R807](../research-round807-v1.txt) and its
[R808 source review](../research-scope-review-round808-v1.json). The shared library
index is [research-index.json](../research-index.json), hashed for retrieval only.
No unrelated knowledge-collection report bodies were opened. The existing paired
measurement summary is supplied context, not a new result from this dispatch.
This inspection of vLLM/TLA is not a Niko upstream check or an inference adoption.

## Derived code, including exact boundaries

`xe_gemm_bsd.hpp` derives solely from the marked Intel 2025 BSD-3 file
[gemm_xe2.hpp at the vLLM pin](https://github.com/vllm-project/vllm-xpu-kernels/blob/6b30c9b92cc525c32729df988b5f918e7c5661a5/csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp#L95-L247).
The full source file SHA-256 is
`359b1f04dfab3ad0850945b5240ec9b16be02b60fb7ab5bcc10d5cf254af99b4`.
The original copyright, conditions and disclaimer at lines 1–31 are preserved
verbatim. The source template begins at line 95 and ends at line 247. Its
coordinate/tile/copy/fragments/prefetch/MMA/barrier portions are lines 115–173
and 180–203; final register reorder/store are lines 245–247. These actual bodies
are retained, rather than reconstructed from a conceptual description.

The full adaptation delta is small and reviewable in that one file: namespace
and entry-name change, unused TA/TB removal, removal of scale/bias arguments and
FP8/bias epilogues, a K-tile bound on the initial prefetch, and trimmed includes.
The load/reorder/DPAS/barrier loop is the upstream algorithm. No grouped scheduler,
counter, expert rows, Torch, apply_scale, FP8/INT4 path or reference implementation
is present. No speculative upstream prefetch behavior is relied on at K=32/64.

`native16.hpp`, `native16.cpp`, `numerics_chain.cpp`, this CMake recipe and these
reports are newly authored for Strata under the repository MIT LICENSE. They are
not copied from a third-party adapter, policy, test, template or XeStrata file.
The host adapter uses public TLA APIs (`TiledMMAHelper`, `MMA_Atom`, `XE_DPAS_TT`,
shape/stride constructors and block-copy selectors). The two geometry declarations
are own qualification choices expressed as shape facts, not a copied policy class.
No upstream-derived code has been relabeled MIT: the actual core stays BSD-3.

The unmarked `gemm_xe2_policy.hpp` at SHA-256
`054d9df282baa42af69e25715cf27735a14d9c53564254b21f7b293c18c091ab`
remains license-unresolved and **its implementation is not copied**. It is not an
include or build dependency. Its public compact geometry is a fact also recorded
in R807. The package does not depend on resolving that policy's origin. The
BSD-marked launcher was read to verify the native MMA, properties and layout seam;
the Torch API and device-0/512-thread launch calculation were not copied. The
marked grouped source was read for physical B layout and actual shape bounds;
its scheduler and tensor helper were not copied. The CMake files supply build
facts only, not copied build templates.

## External TLA source, notices and API evidence

The official `ghq` root is
`/home/yayoi/ghq/github.com/intel/sycl-tla`, on remote-default `main` at the actual
clone revision `7b56b936b87f799f80dde60c2fd8d11c1128cebb`. It has no tracked
changes. Its sole untracked root item is the authorized `.worktree/` container;
no ignore rule or source/config file was changed to conceal that. The consumed
detached worktree is `.worktree/strata-native16-87f` at the exact requested commit.
`git rev-parse HEAD` independently verified both pinned dependency revisions.
No dependency content was changed or built. No TLA file was vendored into this
package, and no source was downloaded from XeStrata.

vLLM [CMake lines 341–368](https://github.com/vllm-project/vllm-xpu-kernels/blob/6b30c9b92cc525c32729df988b5f918e7c5661a5/CMakeLists.txt#L341-L368)
sets `87f6850680a580654b9ea2c80dbc01aeb36ad231` as its default TLA revision;
an environment/path override exists in vLLM, so that setting alone would not pin
a built binary. This package uses an explicit existing worktree and verifies
its commit and all inventoried header hashes before optional compilation.

The TLA root [LICENSE.txt](https://github.com/intel/sycl-tla/blob/87f6850680a580654b9ea2c80dbc01aeb36ad231/LICENSE.txt)
is BSD-3-Clause with NVIDIA 2017–2025 and Intel 2025 notices. That root label
does not describe every included file. A recursive literal-include inventory
over the five direct TLA headers found 207 dependency files, conservatively
including inactive conditional branches. All 207 have checked SPDX headers:
192 BSD-3-Clause and 15 Apache-2.0 WITH LLVM-exception. Paths, actual SHA-256,
sizes, copyright lines and notice hashes are recorded per file in the manifest.
All notices remain in the original external headers, unchanged. If Main later
vendors/distributes headers or binaries, preserve the relevant BSD and Apache
LLVM-exception terms and notices; do not substitute only the root BSD text.

The explicit direct TLA includes are `cute/tensor.hpp`, `cute/atom/mma_atom.hpp`,
`cute/atom/copy_traits_xe_2d.hpp`, `cute/util/xe_split_barrier.hpp` and
`cutlass/numeric_types.h`, all under TLA's `include` directory. Broad tools/util
and applications include directories from vLLM are unnecessary for this source
closure. No generated TLA headers or CMake configuration are requested.
The conservative inventory also sees platform/SYCL/C++ and inactive CUDA include
names. Those are toolchain/platform dependencies or excluded preprocessor arms,
not a CUDA/HIP import. A missing `copy_traits_xe4_dma.hpp` appears only under
`SYCL_INTEL_XE4_TARGET` in `copy_traits_sm90_tma.hpp:47–49`; that macro is not set
and that file is not synthesized, fetched or treated as a Xe2 dependency.

The exact pinned API evidence for the contract is:

- [xe_rearchitecture.md](https://github.com/intel/sycl-tla/blob/87f6850680a580654b9ea2c80dbc01aeb36ad231/media/docs/cpp/xe_rearchitecture.md#L58-L78)
  documents block-2D bounds/zero-fill, 64-byte base, 16-byte stride, four-byte
  width, and width/pitch/height limits. This is the pinned document actually read,
  replacing R807's loose reference to current documentation.
- [copy_traits_xe_2d.hpp](https://github.com/intel/sycl-tla/blob/87f6850680a580654b9ea2c80dbc01aeb36ad231/include/cute/atom/copy_traits_xe_2d.hpp#L111-L173)
  constructs width, height and pitch from the original tensor and initializes
  the block-2D address payload, with optional runtime assertions. It retains Y's
  N-width and ldy-pitch independently. The adapter enforces the strict source
  bound `<=0xFFFFFF` and the stricter 16-byte pitch before enqueue.
- The same file at lines 960–1178 selects copies from the input/output tensor
  value types and strides; lines 1698–1707 construct related prefetch payloads;
  lines 1794–1830 are the actual `get_block_2d_copy_A/B/D` helpers used by the core.
- [mma_xe.hpp](https://github.com/intel/sycl-tla/blob/87f6850680a580654b9ea2c80dbc01aeb36ad231/include/cute/arch/mma_xe.hpp#L44-L65)
  defines the native DPAS atom/register types; its FP32/BF16/BF16/FP32 and
  FP32/FP16/FP16/FP32 specializations are at lines 232 and 237. `mma_traits_xe.hpp`
  selects the plain-DPAS path for a non-zip tensor. The adapter uses float C/D
  MMA types and independent typed block-store fragments.
- [reorder.hpp](https://github.com/intel/sycl-tla/blob/87f6850680a580654b9ea2c80dbc01aeb36ad231/include/cute/algorithm/reorder.hpp#L82-L119)
  dispatches by source/destination types and layouts. Type conversion and layout
  changes stay in fragments. `reorder_xe.hpp:878–962` supplies float-to-half and
  float-to-BF16 vISA conversion cases; generic same-layout conversion uses the
  typed constructor. `cutlass/bfloat16.h:125–140` uses the Intel BF16 conversion
  builtin in device compilation, explicitly avoiding an unconditional devicelib
  fallback. Which branch/layout is emitted still needs the actual compiler and
  device-binary inspection; there is no claim of measured instruction lowering.
- [xe_split_barrier.hpp](https://github.com/intel/sycl-tla/blob/87f6850680a580654b9ea2c80dbc01aeb36ad231/include/cute/util/xe_split_barrier.hpp#L51-L70)
  supplies the original split barrier builtins used by the retained loop. K count
  and control are workgroup-uniform; no per-lane return encloses a barrier.

The trimmed source does not explicitly include `cute/util/compat.hpp`, but
`cutlass/gpu_generics.h` pulls it transitively. That compatibility family carries
Apache-with-LLVM-exception notices and includes device/queue/default-selector
helpers. Inspection of `compat/device.hpp:717–966` shows its singleton manager is
function-local/lazy; this adapter never calls it. It uses the supplied queue,
`this_work_item` ID and block/MMA intrinsics. No library GEMM arithmetic, queue
manager, CPU grouping, CPU matrix fallback, oneMKL or oneDNN is called. Source
call-path inspection does not substitute for binary/DPAS/emulation evidence.

The only toolchain source inspected was official LLVM's installed
`sycl/ext/oneapi/experimental/device_architecture.hpp`, its `.def` enumeration,
and `sycl/info/ext_oneapi_device_traits.def` under the approved `.tools` tree.
They support the real device architecture query used in the gate. Their paths,
hashes and enclosing Apache-2.0 WITH LLVM-exception license context are recorded
in the manifest; the included `.def` files do not have their own SPDX notices.
No compiler binary was invoked and no library was loaded for this worker task.

## Build facts and qualification boundary

The pinned vLLM root CMake at lines 376–414 sets C++/TLA definitions and the
`intel_gpu_bmg_g21` default, while lines 282–314 and `cmake/utils.cmake:553–615`
show the actual Xe2 compile/link flags. The proposal keeps the relevant flags
with explicit BMG AOT target, SG16, GRF256 and no network dependency population.
The build recipe is optional, default OFF and has not been configured.

Remaining compile risks are bounded to compiler/backend acceptance of the pinned
TLA payload builtins and vISA, split barriers and properties, and the actual
copy/reorder/MMA instantiations for row-major W and independent strided output
at both shapes. All eight type/tile combinations exist in source, with no fake
support gate or whole-global-FP32 cast. Main must resolve actual compile errors
and inspect generated device code before any numerical or performance claim.
No existing functionality, fast route, old regression or quality requirement is
weakened by this isolated proposal. Numerical, fault, performance, model quality
and physical 262144-context results are all UNRUN.
