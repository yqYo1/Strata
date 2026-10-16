# PR6 LLVM 22 shift compatibility source proposal v2

Status: `SOURCE_ONLY_UNSAFE_TO_RUN`. This worker changed only
`validation/pr6-restore/llvm_shuffle64.cmake` and added this report, its JSON
and its patch. No CMake, compiler, preprocessor, build, test, native binary,
runtime, device job, process inspection, Git command or cleanup ran. Main owns
review, fresh configuration, command inspection, compilation and execution.

The assigned immutable base is `ae8f85f1431586353c660c468854164089b2b8fc`,
also recorded by Main's failed v3 build receipt. The helper preimage SHA256 is
`b275253f52d1d54b56e00efc2e5f75891cb41b9a15fff16f175875c9eaf0f898`.
The patch was generated with Python `difflib` against that captured preimage,
without Git. It includes the entire source delta for this assignment; the
new reports describe the proposal rather than form part of the source patch.
The JSON pins the resulting helper, unchanged checked-header generator,
patch, inputs, earlier reports and failed receipt.

## Observed failure and correction

Main's `own-pr6-harness-build-v3.json` records exit 1, `ownership_closed=true`,
and no GPU execution. Its stdout lines 16468–16472 identify an ambiguous
`__spirv_GroupNonUniformShuffleUp` at generated checked-header line 1591,
instantiated as `dpct::experimental::shift_sub_group_right<unsigned long long>`
by historical `qsa_select.dp.cpp:1027`. This is a compile failure, not a GPU
hang. The original receipt and log retain their failed status and bytes.

The added adaptation changes only the `__SPIR__` branch of this masked
shift-right overload. For C++ `long long` and `unsigned long long`, it
bit-casts the input to the corresponding native OpenCL scalar, calls the
same `__spirv_GroupNonUniformShuffleUp` with the same Subgroup scope and
`delta`, then bit-casts the result back. Static assertions require eight-byte
storage and the pinned Linux `long`/`unsigned long` OpenCL ABI. Every other
type executes the original initialized-result, guard and return statements
inside the `else` branch. This preserves their initialization requirements;
there is no new default construction of `T`.

Both branches retain the original calculations of `id` and `start_index`.
Both call the builtin before applying the original
`(id - start_index) < delta` guard, replacing the result with `x` on the same
lanes. No builtin is moved behind a lane-dependent condition. The unchanged
logical size, source-lane choice, unsigned delta and guard therefore remain
part of the original operation. The casts preserve all 64 bits, including
the high half of packed counters and signed negative bit patterns; no numeric
conversion, split-word algorithm or public full-group collective is introduced.
Compiler overload resolution and actual bitwise routing remain unqualified.

The existing select adapter is unchanged. Its original SPIR-V operation,
logical index calculation and intermediate generated-body hash remain pinned.
The two Main-added namespace declarations before `memory.hpp` remain intact:
`dpct::detail::dpct_memcpy_3d_detail` and
`dpct::detail::dpct_memcpy_3d_detail_usmnone`. The checked device body, owning
queue registration, fail-stop async handling, include guards, profiling setup,
device selection, events and all NVPTX/host branches are unchanged.

## Exact source and generated-output pins

The original vendor `util.hpp` is 51,840 bytes, blob
`033f6c74564f64ca7701c4f4fe68fe75cf7d8702`, SHA256
`c76e74a7d18721523839e30c8d77e89f108423e49044aebb8f50ae1fbd3f424b`.
The existing 49-file vendor identity check remains in force; no installed or
historical vendor file is edited. The adapter continues to require the pinned
Linux LLVM/oneMKL mode, compiler identity and three installed API-header hashes.

The shift's original exact span is `[24013,24322)`, lines 555–563, 309 bytes,
SHA256 `cf84272978a99b9556b72a8fb075be116a8068e5de96675d8df42b7d0fa21b13`.
The preceding select adaptation adds 976 bytes, so this span is checked at
`[24989,25298)` in the already checked select-only body. Its first-match
position, length, hash and absence of a later duplicate are checked before
replacement. The full original input hash also rejects any earlier duplicate
or unrelated source change. The select-only intermediate body must still hash
to `cb93a6d8ecb064bf8d693fad0c9c5e64ce753d6f4b126f5e086969d9b1228609`.

The shift replacement is 1,274 bytes, SHA256
`4bad21494f42ef9b7c7edd7270e6efdb315900ceb2521828c382a4cd8aafa00b`.
The final utility body is 53,781 bytes, SHA256
`1f1a2d88b8ed217737c13ccce347b3d802e8ab389925054aafca7ef932407e36`.
Both are fixed expected hashes in the adapter and reject an unknown result.
The proof schema becomes `strata-pr6-llvm-shuffle64-v2`, retaining the existing
select span fields and adding the original/adjusted shift span, count and hashes.
The unchanged checked-header generator appends the actual combined-header hash
to the proof when Main configures the build.

Independent source-string construction predicts an 87,940-byte combined header,
SHA256 `22709e0715e75621996b9eb289d507a11fcb2919770e161221cf74cd1cca9b21`.
This is a prediction from source inspection, not a CMake-produced artifact or
a configuration pass. Main must compare fresh `pr6-checked-dpct.hpp`,
`pr6-checked-dpct.sha256` and `pr6-llvm-shuffle64-proof.json` against these
values. The earlier v1 proposal/report and Main's memory-name correction record
are preserved byte-for-byte; the earlier combined-header prediction is historical.

## Selected historical closure audit

The audit used `historical-closure.json`, whose source revision is
`d0c5dda0dc2e8d82458c4b0ff31c4cef44c40b32`: 198 historical project files,
84 translation units including explicit port sources, and 49 external vendor
files. Source inspection found 113 explicitly qualified masked calls in 33
project files: 39 left, 7 right, 59 xor and 8 select. The JSON groups every
call's line and source value type and pins each participating file. This is
a source inventory, not a compiler instantiation trace or execution coverage.
Search of the vendor closure found these helper definitions in `util.hpp`;
no additional vendor caller of a masked shift/xor was identified.
An additional name search found no unqualified or aliased occurrence of these
four helper names in the project closure. All 49 original vendor hashes match.
The original historical manifest matches 171 of the 198 current project files;
it is not the separate fixed-source identity record. This worker changed none
of those 198 bodies, and Main retains the fixed-source acceptance gate.

The integer64 right-shift sites are `qsa_select.dp.cpp:1027` (`x`) and `:1050`
(`z`) in `cl_excl_scan64`. The first is the observed compiler instantiation;
the second uses the same unsigned-long-long overload. The other five right
sites pass `int`: `kv_stream.dp.cpp:73,93` and
`qsa_select.dp.cpp:646,668,1204`. No signed-long-long right caller was found;
the signed arm follows the existing select adapter's paired ABI handling and
remains source-only, without a claimed reached or executed signed instance.

No integer64 masked left or xor caller was found. Existing double callers
are deliberately distinguished from the problematic C++ long-long overloads:
left in `gdn.dp.cpp:164,204`, `ple.dp.cpp:104,129`, `gr.dp.cpp:89` and
`shared_expert.dp.cpp:126`; xor in `qsa.dp.cpp:382`; select in
`gr.dp.cpp:101`, `shared_expert.dp.cpp:138` and
`sampler.dp.cpp:717,773`. They remain unchanged. Other calls pass float, int
or uint8_t. In particular, `s2_expert_grouped.dp.cpp:518,532,546` passes float
with mask `0xffffffffu`, logical size 8 and deltas 4,1,2; its left operation
and reduction order are not touched. Integer64 left/xor, non-full-mask use,
divergent participation and other utility overloads remain source-limited holds;
there is no blind compatibility patch for them.

### Actual shift caller masks, ranges and participation

Both affected calls pass the literal `0xffffffffu`; logical size defaults to 32.
The source computes `lane = local_id(2) & 31`, `warp = local_id(2) >> 5`, and
each loop uses `o = 1,2,4,8,16`, converted to the original unsigned `delta`.
The first loop is unconditional inside the scan. Its subsequent `lane >= o`
addition does not condition the collective. After a workgroup barrier, the
second loop executes under `warp == 0`, which is uniform for each aligned
32-lane source warp. It reads 32 warp sums, writes their exclusive prefixes
and places the total in `s_warp[32]`. `CL_T=1024` and a 33-element unsigned64
shared array express the source's 32 source warps. Inputs pack two uint32
counts, `gt | (eq << 32)`; addition and unpacking stay unchanged.

For each loop, if physical subgroup lanes map as the source assumes, all
32 lanes named by the full mask encounter the same five calls and deltas.
The original SPIR implementation does not otherwise consume `member_mask`;
this patch neither adds mask handling nor claims arbitrary-mask semantics.
The source guard keeps `x` when the lane's logical offset is below `delta`.
These are source conditions, not evidence of provider subgroup mapping.

Crucially, this helper is inside `#if STRATA_SEL_CLUSTER`, controlled by
`DPCT_COMPATIBILITY_TEMP >= 900`, and v3 proves its definition was compiled
far enough to instantiate unsigned64. However,
`qsa_block_topk_cluster` at lines 1575–1577 has an unconditional SYCL `#if 1`
branch returning false. Its cluster launch is in the inactive alternative.
The ordinary top-k caller therefore falls back. The dormant cluster body
also retains migrated cluster assembly outside this task's scope. No physical
subgroup-size guarantee or working cluster execution is inferred from this
compile failure; the adapter does not enable cluster execution or delete its TU.

The existing signed64 select caller remains `qsa_prompt_attn.dp.cpp:853`,
full mask, logical size 32, four issue iterations and `cell=(lane+32*j)>>2`
in 0–31. Its initial issue is unconditional and later issue uses the source's
uniform chunk condition, as recorded in v1. No new execution qualification
of that caller is claimed here.

## Provenance, Main gates and retention

This is an independently authored project adapter under the repository MIT
LICENSE. Allowed dependency evidence consists of the pinned historical DPCT
utility and installed LLVM API/type definitions (`aliases.hpp`,
`generic_type_traits.hpp`, `__spirv/spirv_ops.hpp`). The pre-existing installed
toolchain happens to live beneath a XeStrata-named checkout; no enclosing
XeStrata implementation, helper, test or archived proposal was read or used.
The generated private dependency retains its Intel notice. No new vendor copy,
license template, native TU exclusion or inference implementation is introduced.

Main should review the complete patch against the pinned helper preimage,
regenerate its fresh isolated recipe and inspect the actual compile commands
and historical closure. Configuration and full compile/link are the next
unperformed gates. After Main's provider gate, an owned bitwise oracle should
cover both signed and unsigned64 values, high-half/negative/boundary patterns,
the observed deltas and guard boundaries, and logical sizes 8/16/32 under a
verified full 32-lane mapping. Unchanged int/double fallback coverage should
remain separate from long-long compatibility. This worker supplies no fixture
and executes none. A compile pass alone would not qualify arbitrary masked
helpers, the dormant cluster implementation or historical model math.

Nonempty async-87 handling, pending-event behavior, full-model numerical/task
checks, shared state, physical 262144 context, RESTORE and repeated prefill
above 600 tok/s remain open. The coordinator-reported model-free provider
passes are prior evidence, not validation of this new shift adapter.

Main owns the compact report and source proposal. No raw tensor, trace,
profiler or runtime capture was created or deleted. The next retention review
is Main's first fresh configuration/compile result or a superseding failure;
earlier failed receipts and v1 provenance remain unchanged.
