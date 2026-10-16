# PR6 LLVM 22 shuffle compatibility proposal v1

Status: `SOURCE_ONLY_UNSAFE_TO_RUN`. The source change is complete for Main's
review; configuration, compilation, provider behavior and numerical behavior
have not been qualified. This worker ran no CMake, compiler, preprocessor,
build, test, native binary, runtime, device job, process inspection or Git command.

The base is Main's recorded commit
`322e1fd86ac694dd9926998548702d2859bd254f`. The complete source delta is
`llvm-shuffle-compat-v1.patch`: one edit to
`validation/pr6-restore/checked_dpct.cmake` and one new helper,
`validation/pr6-restore/llvm_shuffle64.cmake`. The checked-header script's
preimage SHA256 is
`5757b9a1954558257fdc88d031aaeab7cbe7369d75191af21d547a890d2995e8`.
The JSON beside this report pins both resulting source files and the patch.
No historical project file, original vendor file, external installation,
entrypoint, harness, source resolver or other proposal/report was edited.

## Failure and dependency evidence

Main's `own-pr6-harness-build-v2.json` records a closed build with exit 1;
its failed status remains unchanged. The relevant log is
`/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-pr6-harness-build-v2/owned/build-prefill-harness.stdout`,
SHA256 `873192b0da66853ed85064c37a51533cc25e1b975e0348a6899d77cf7e2ce500`.
Lines 12269–12301 show the vendor `dpct/util.hpp:475` ambiguous call to
`__spirv_GroupNonUniformShuffle`, instantiated with `long long` by
`qsa_prompt_attn.dp.cpp:853`. The surrounding v2 recipe already resolved the
CPU host-mode collision; its source list, host-mode override and precise-math
options are outside this patch. Main must recheck all 111 compile commands,
84 historical translation units and the extra harness in the new build.

Allowed dependency evidence was read only from the historical vendor snapshot
and the installed LLVM dependency's API headers. No XeStrata implementation,
helper, test, archived proposal or enclosing checkout source was read. The
installed compiler path is inherited from the existing `llvm_onemkl.cmake`
adapter; its SHA256 is
`1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb`.

Installed `sycl/aliases.hpp:135` and `:136` define
`sycl::opencl::cl_long = std::int64_t` and `cl_ulong = std::uint64_t`.
`sycl/detail/generic_type_traits.hpp:94–102,164–170` defines the fixed-width
signed/unsigned selection used for integral OpenCL type adaptation, and
`sycl/detail/spirv.hpp:939–940` uses OpenCL adaptation around this same builtin.
These are dependency interface/type definitions, not copied implementations.
The new code uses public `sycl::bit_cast` and public OpenCL scalar aliases;
it does not call a private SYCL conversion helper.

`sycl/__spirv/spirv_ops.hpp:165–168` requires compiler-provided SPIR-V builtin
declarations. It does not enumerate this builtin's overloads in the installed
header. Consequently the exact overload set is a remaining compiler check,
not something claimed to have been printed or compiled by this worker. The
installed definitions support using the native OpenCL scalar type, and the
new static assertions require that it is `long`/`unsigned long` with eight-byte
storage for this Linux recipe. The three dependency definition files are
hash-pinned by the helper; their paths and hashes are in the JSON.

## Exact transformation and provenance

The new switch `PR6_LLVM_SHUFFLE64_COMPAT` defaults to OFF. ON requires Linux,
`PR6_OPEN_SOURCE_LLVM_ONEMKL=ON`, both pinned compiler identities, the three
installed API-header hashes and the original DPCT utility hash. No compiler
or fallback probe is added.

The original `util.hpp` is blob
`033f6c74564f64ca7701c4f4fe68fe75cf7d8702`, 51,840 bytes, SHA256
`c76e74a7d18721523839e30c8d77e89f108423e49044aebb8f50ae1fbd3f424b`,
as recorded by `historical-closure.json`. The helper checks the entire hash,
then the exact 311-byte span `[20690,21001)` (original lines 470–475), its
position, hash and uniqueness before replacement. No search across unrelated
collectives or broad regex replacement is used. A changed input fails
configuration. The replacement and complete generated utility body also have
fixed expected hashes and fail configuration on a mismatch.

Only the `__SPIR__` branch of
`dpct::experimental::select_from_sub_group<T>` changes. For `long long` and
`unsigned long long`, it bit-casts to the corresponding signed/unsigned native
OpenCL scalar, invokes the same `__spirv_GroupNonUniformShuffle` with the same
`__spv::Scope::Subgroup` and `logical_remote_id`, then bit-casts back to `T`.
The branch for every other type keeps the exact original return statement.
NVPTX and host branches, shift/xor functions, arithmetic, memory operations,
barriers, queue selection, events and kernel scheduling remain unchanged.

The source-semantics argument is conditional on the asserted pinned ABI and
the compiler implementing the existing scalar shuffle: integral source and
target are trivially copyable eight-byte types; the cast preserves the entire
64-bit representation; the shuffle routes that representation from the same
source lane; the inverse cast recovers `T`. There is no 32-bit conversion,
floating-point conversion, unsigned-to-signed numeric narrowing or two-word
split. `-1`, signed minimum/maximum, values above 2^32, unsigned high-bit
patterns and unsigned maximum all retain their bits. Signed values use the
signed native type; unsigned values use the unsigned native type. Compiler
and runtime validation of these cases remains Main's gate.

The helper and project changes are independently authored under the repository
MIT LICENSE. The private generated artifact contains the original Intel
Apache-2.0 WITH LLVM-exception notice unchanged, plus a modification comment.
Only a short pinned transformation span is present in the helper/patch; no
complete vendor implementation is added to Git or to the project source tree.

## Include order and preserved safety contracts

The existing forced `pr6-checked-dpct.hpp` still starts with
`STRATA_PR6_CHECKED_DPCT`, profiling setup, the project async contract and the
checked device body. Queue-owner registration and fail-stop async handling
are byte-for-byte the existing transformation. The new suffix follows that
body and first rejects an already-defined `__DPCT_UTIL_HPP__`. It includes
`<dpct/memory.hpp>` before appending the transformed utility body.

This prerequisite is necessary: `util.hpp` uses memory declarations without
including `memory.hpp` itself. The historical `image.hpp:14–15` includes
`memory.hpp` then `util.hpp`, and `dpct.hpp:53` reaches it before its own later
utility include. Loading only an include-search-path replacement could lose
to those relative includes. The forced private header instead establishes
the unchanged `__DPCT_DEVICE_HPP__`, `__DPCT_MEMORY_HPP__` and
`__DPCT_UTIL_HPP__` guards before those includes. Subsequent relative includes
are suppressed without duplicate definitions. The original utility header
does not depend on `image.hpp`; this order preserves its prerequisites.
The 49-file outer dependency hash check is still in force.

OFF adds an empty suffix, preserving the original generated checked-header
content. ON emits the combined build-private header and its usual
`pr6-checked-dpct.sha256`, plus `pr6-llvm-shuffle64-proof.json`. The proof
contains input, source-span, replacement, utility-body and actual combined
header hashes. The utility body is 52,816 bytes with SHA256
`cb93a6d8ecb064bf8d693fad0c9c5e64ce753d6f4b126f5e086969d9b1228609`.
The JSON contains the independently source-constructed combined-header hash
for Main to compare; no generated CMake artifact was executed or installed here.

D3 doorbell/capture exclusion, no environment override and prefill-only
entrypoint restrictions are unchanged. A successful compilation would not
authorize a provider launch or decode/capture route.

## Caller participation and affected types

The only observed changed historical instantiation is `long long` in
`prompt_attn_i8_kernel` at `qsa_prompt_attn.dp.cpp:853`. Its row producer returns
`-1` for an invalid cell or a 64-bit page-based row, so narrowing is incorrect.
`issue` executes four iterations in every calling lane. With `lane` in 0–31,
`cell = (lane + 32*j) >> 2` lies in 0–31 for `j` in 0–3. Each call passes
`0xffffffffu` and the default logical size 32. The launch requests subgroup
size 32 and a workgroup of 128; the helper's start-index and modulo expressions
are unchanged. The initial `issue` is unconditional; later calls are under
`more`, determined from workgroup-uniform chunk metadata, not a lane predicate.
The row-dependent `ok` predicate is used after the shuffle.

This establishes a source convergence argument provided the provider honors
32-lane subgroups, the intended local-lane mapping and valid/uniform metadata.
It does not prove those runtime prerequisites or surrounding historical math.
The SPIR-V branch originally does not consume `member_mask`; this proposal
does not strengthen it into a general arbitrary-mask implementation. The
full-mask caller still requires all named lanes to participate at the same
dynamic operation with valid source lanes. No public full-group collective
replaces the operation, so no new full-group participation assumption is added.

Unsigned `long long` is supported defensively by the same type-only correction;
no such historical caller was found in the selected closure. Other observed
`select_from_sub_group` types are float (elementwise/shared expert/GR), double
(GR/sampler) and int (sampler); they keep the original return. Existing `long`
and `unsigned long` also keep it. This is not a qualification of the other
masked helpers, arbitrary masks, other devices or other backends.

## Main-owned next checks (not executed)

The JSON `main_commands` stores the complete configure/build argv and the
effective v2 process environment. Use Main's existing serialized, bounded,
owned-process controller, retaining its wall/CPU/RSS/log/build-tree limits.
The proposed fresh build directory is
`/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-pr6-harness-shuffle-v1/build`.
The exact configure payload is v2's recorded configure command with that `-B`
and the appended `-DPR6_LLVM_SHUFFLE64_COMPAT=ON`; all other args and environment
are reproduced in JSON. The exact build payload is:

```sh
/usr/bin/cmake --build /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-pr6-harness-shuffle-v1/build --target pr6_prefill_api --parallel 2 --verbose
```

Before build, inspect the new proof and compare generated hashes; inspect
`compile_commands.json` and Ninja link commands for the complete retained
closure, late CPU `-fno-sycl`, checked preinclude, precise math and sequential
ILP64 oneMKL linkage. Hash the untouched original vendor snapshot again. Main
should also configure with the switch OFF in a separate owned build directory
and compare the generated checked-header content against v2, then exercise
span/input/API-hash mismatch rejection using disposable dependency fixtures.
Never change installed headers or a running process environment for these arms.

For a small compile-only gate, Main can supply a private translation unit
outside the historical closure and compile it with the generated engine
flags, checked preinclude and dependency paths, replacing only input/output
filenames. It should include `dpct/dpct.hpp` and `dpct/image.hpp` after the
forced header and instantiate the helper inside a 32-lane SYCL kernel for
signed/unsigned long long, long/unsigned long, int, float and double. The
full `pr6_prefill_api` build command above is the exact existing-TU reproduction
of the failing signed instantiation. No new test fixture is supplied or claimed
executed by this source-only proposal.

Only after the separate provider gate permits execution should Main run
finite, owned 32-lane shuffle comparisons with full masks and valid source
lanes: broadcast each source lane, vary remote lane by destination, and compare
all 64 bits for `-1`, zero, signed limits, `2^32+17`, `2^40+lane`, unsigned
high-bit patterns and `UINT64_MAX`. Include logical widths 8, 16 and 32 to
check unchanged subgroup partition arithmetic. A host reference should route
the bits using the original start-index/modulo formula. Masked-divergent cases
remain held unless their original SPIR-V semantics and caller validity are
separately established. Runtime fixtures and exact launch commands depend on
the still-unqualified provider contract; this report does not authorize them.

Compilation/link, compiler-provided overloads, emitted image/operation,
include-order fixtures and all provider/numerical checks remain open. Model
quality, shared state, physical 262144 context, RESTORE, whole-model behavior
and repeated >600 tok/s are unqualified. No timing sample or GPU-safety claim
is made. Main owns full patch review, verification, integration and Git.

Retention: Main is the owner; the consumer is this compatibility proposal's
configuration/compile qualification. Review the compact report and hashes at
the first new failure or successful qualification boundary. No raw capture,
tensor fixture, cleanup, deletion or independent progress backup was created.
