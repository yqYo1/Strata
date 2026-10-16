# PR6 validation CPU compile-mode correction v2

Status: **SOURCE_ONLY_UNSAFE_TO_RUN**. Worker `task_d23fb885ebc0`, dispatch
`ctx_1762fae14c30`, source base `4aa4072bbdf785167e48b8dd68a26a1c8b72e2c9` on existing
`verify/pr6-prefill-restore`. Only `validation/pr6-restore/llvm_onemkl.cmake`
changes. The complete additional source patch is `llvm-build-adapter-v2.patch`
(957 bytes, SHA256 `c2facbe95a0d55f71fe5281fc527d3c15b6f268c96c490e12e29b5f9951bf302`). Main owns
source acceptance, configuration, generated-command review, builds, tests,
runtime/provider/GPU execution and Git integration. No compiler, preprocessor,
CMake, build, test, provider, model or GPU operation was invoked by this worker.
No branch, stage, commit, push, process inspection or cleanup was performed.

## Prior Main evidence, unchanged

Main's `own-pr6-harness-configure-v1.json` and
`own-pr6-generated-command-review-v1.json` record successful configuration and
review of 111 generated commands: 84 historical engine TUs, one sibling harness
and 26 external ggml TUs. The checked-DPCT header hash is
`5fbc6198ed723f4949245336963e871ac336b6b4c927eee657e0e71e0a233eb3`.
Those observations precede this correction and are not regenerated-command
approval for v2.

`own-pr6-harness-build-v1.json` remains a failed build, exit 1. The CPU
`expert.cpp` failed in the installed Clang `cpuid.h:444`: static `__cpuidex`
conflicted with a builtin declaration under inherited `-fsycl`. The sibling
harness TU compiled with warnings as another Ninja step; no linked harness,
provider qualification or engine run resulted.

Main's `own-pr6-cpu-host-only-probe-v1.json` records exit 0 for exactly the same
unchanged CPU expert source and generated options, with `-fno-sycl` inserted
immediately after `-fsycl` and a different output path. Its source SHA256 is
`9bcf5d889818ef3cb9619df6536e8844d151714803ac8930eada115a61b62e48`.
The checked header, ISA flags and precise math option were preserved. The
probe warned that `-fsycl-default-sub-group-size=32` was unused, and that the
checked header's `destroy_event` function was unused. These warnings are not
hidden by this change. This is one CPU object compilation, not a numerical,
whole-build, link, provider, queue-lifetime or performance pass.

## Exact behavior and reviewed source boundary

The adapter's existing default-OFF `PR6_OPEN_SOURCE_LLVM_ONEMKL` admission is
unchanged. Inside its existing adaptation function, after the existing option
normalization loop, append the CXX-only option
`$<$<COMPILE_LANGUAGE:CXX>:-fno-sycl>` privately to exactly:

- `strata_kernels_cpu`: 8 C++ TUs, all selected from `src/kernels/cpu/` by the
  historical resolver: expert, pool, expert_layout, q2_avx2, native_expert,
  iq_avx512, iq_avx2 and kq_avx2.
- `ggml-base`: 9 TUs, comprising 6 C++ and 3 C files for tensor, allocation,
  backend, metadata, threading, quantization and GGUF host services.
- `ggml`: 2 C++ TUs, backend dynamic-loader and registry host control. This
  target is present in the configured command inventory even though the prior
  harness link directly uses ggml-cpu/base, not libggml.
- `ggml-cpu`: 15 TUs, comprising 12 C++ and 3 C files for the CPU backend,
  repacking, IQ, host memory, traits, AMX, elementwise/vector operations and x86
  quantization/repacking.

The JSON pins every selected CPU TU. The historical CMake declarations and
Main's existing compile-command inventory agree on these target memberships.
No target contains a selected GPU kernel TU. The host backend registry contains
conditional references to GPU backend APIs; its GPU macros/backends are disabled
by the existing adapter. The adapter already rejects enabled alternative ggml
backends before configuration. The historical recipe forces backend dynamic
loading OFF; the inspected ggml CMake rejects CPU_ALL_VARIANTS with dynamic
loading OFF. Main's configured inventory has no CPU variant or feature-object
target, so no wildcard or speculative variant handling is added. If that pinned
closure changes, Main must review the new targets rather than assume this list
covers them.

This adds 28 C++ compile-mode overrides in the inspected 111-command inventory;
the six CPU C-file commands receive no new option. It does not remove or rewrite
inherited `-fsycl`, its generator expression, SG/device-split/AOT flags, ISA
source properties, precise math, source selection, header precedence or checked
preinclude. The late negative option is the task-requested override; the exact
position differs from Main's manual probe and needs actual generated-command
review. No `BEFORE`, substring filtering, unknown-option stripping or diagnostic
suppression is added.

PRIVATE limits this option to the four targets' C++ compilation; it does not
add a usage requirement to their consumers. `strata_spec`, GPU core, kernel,
engine, prefill, final strata and sibling harness compile/link properties are
not modified by this correction. The existing PUBLIC oneMKL SYCL BLAS link,
sequential CPU/SYCL threading and ILP64 settings remain. The checked preinclude
may still be parsed by host-only CPU compilation as in Main's successful probe;
its implementation and event/queue/retirement contracts are unchanged.

## Source identity and provenance

Read-only Git-object/byte comparison against the immutable base covered
247 existing tracked validation/report files. Only the adapter differs;
all 246 others are unchanged, including the outer wrapper, historical
CMake, all 198 historical files, 84-TU resolver, CPU/kernel/math bodies, checked
DPCT gate, all safety fixtures, harness source and prior evidence. The external
49 vendor and 1390 llama/ggml/cmake/LICENSE files were rehashed against Main's
existing dependency-preparation receipt: zero mismatches. This is byte
preservation, not a new dependency/license/runtime qualification.

The six inserted CMake lines and these reports are independently authored under
repository LICENSE, based on this task's explicit append instruction, existing
adapter API, historical target definitions, pinned ggml CMake target metadata
and Main's recorded compiler evidence. No external implementation is copied.
No XeStrata source, helper, test, archived proposal, protected credential record
or LLVM implementation source was read or used. The already configured LLVM
installation is a dependency path only; its enclosing checkout supplies no
implementation. No queue/lifetime or inference change is authorized here.

## Main's remaining gates and evidence limits

Main must review the full six-line delta and regenerate commands under the
same pinned dependency/compiler recipe before building. Compare all commands:
only the four CPU targets' C++ commands should gain exact `-fno-sycl` after
their inherited `-fsycl`; verify no later inherited/dependency/source flag
reenables offload. Preserve every TU, ISA flag, checked-header path/hash,
precise option, include order and sequential ILP64 link dependency. Check that
all C commands and every GPU/harness compile/link command remain unchanged.
Keep the expected unused SG-option warning visible. Any further CPU or GPU
failure needs its own diagnosis; success for expert.cpp does not establish
that the remaining targets compile or the whole program links.

Compile-mode compatibility keeps the host function source body intact but does
not prove numerical identity with the historical compiler, binary ABI/ODR,
provider behavior, no additional link dependency or model performance. Actual
build/link/runtime closure, finite checked-provider event/queue/async/partial
submission/retirement/cleanup and D3 compiled rejection remain separate gates.
All prior tiny/odd-tail/model/numerical/state/task-quality gates continue.
Physical 262144 is **UNQUALIFIED**; export/exit/fresh RESTORE, continuation,
rewind and branch remain **UNQUALIFIED**; historical 600 tok/s is
**NOT_REQUALIFIED**. No new speed samples or validation approvals are claimed.
The existing D3 exclusions and native CLI/decode holds remain in force.

Retention owner is Main, consumer is adapter acceptance and the next generated
command/build gate, next review is acceptance, replacement or retirement. No
raw trace, tensor, log, binary, test script or duplicate progress backup was
created or deleted. Prior failed receipts remain byte-identical. Main owns
commit/push at the accepted boundary.
