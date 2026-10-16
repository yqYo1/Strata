# Prefill API harness corrections v2

Status: **SOURCE_ONLY_UNSAFE_TO_RUN**. H1–H6 are corrected in source against `5f16e726e8f3a000e4f9ae23fcb6747cff592ae9` on the existing `verify/pr6-prefill-restore` worktree. No configure, preprocessing, compile, link, syntax check, host test, provider/runtime/device/model/GPU operation or Git mutation was performed. Main owns every execution and integration decision.

The complete correction patch changes only `validation/pr6-restore/CMakeLists.txt` and `validation/pr6-restore/harness/prefill_api.cpp`. New artifacts are this report, its JSON and its patch. Independent-review v1 and all previous evidence remain untouched. No external implementation, excluded source/helper/test/archive, protected credential record or LLVM implementation source was used. New code and reports follow repository LICENSE. Project-only upstream-sync and git-workflow guidance do not expand this assignment's explicit source-only/no-Git-mutation scope.

## Exact corrections

- H1: The sibling executable explicitly requires C++20, with standard fallback disabled and extensions off. Existing SYCL compile/link option propagation, historical resolver and gated build admission are unchanged.
- H2: Before either native or unquantized key wiring, require `ne0=2560`, `ne1=10240`, `elements=26214400`. Existing native pointer/type and BF16 pointer/bytes/kind checks stay in force. This connects the caller's fixed operator extent to NativeDense's matching allocation metadata without changing a kernel.
- H3: Require an opened PLE table with `rows()==kernels::PLE_TABLE_ROWS` (`320001536`) before PLE wiring/inference. No table substitution, hash-domain or dequantization change.
- H4: Add the **bare**, value-free `--validation-small-input` flag. Duplicate flags fail. With it, count must be1–32767; without it, the existing32768–262144 admission remains. All inputs still require context>=count, context>=8192, context<=262144 and chunk8192. All other explicit options, real input/model checks, position0, immutable environment and48-layer geometry remain. The immutable parsed options drive receipt `validation_small_input` and `measurement_scope`; the short scope is `safety_correctness_only`, so its timing is diagnostic evidence only. Flagged long counts fail, keeping these modes disjoint. The single-token expected PLE window is `[-1,last]`; other counts retain their previous check. The receipt's normal first length now uses `min(chunk,count)`, while the retained optional short-first condition and GPU-prefix rounding are unchanged and match the actual layer-major/run_impl source. No continuation call or algorithm was added.
- H5: After the existing Prefill teardown, repeated compute/default retirement and `session_release`, visit every allocated QSA state and free nonnull `host_step`/`host_pos`, clearing them before descriptor deletion. Use the live borrowed allocation queue, the same DPCT default in-order queue used by `qsa_state_init`'s `malloc_host`. No legacy source or session API was changed.
- H6: After successful capture write/flush, call exactly one checked `fclose(file.release())`. The stream is no longer owned when close status is checked, preventing double close on failure. There is no fsync or new persistence promise.

## Source ownership and exception review

The actual `session_init` sets QSA allocation metadata, allocates value-initialized descriptors and initializes states in sequence. If the array was never allocated, cleanup skips it. If initialization returns early or throws after partial allocation, cleanup inspects every allocated slot: nonnull fields are freed and later untouched fields remain null. This harness always creates the full session (`qsa_ord0=0`); the new loop uses `qsa_ord0+j` consistently with initialization, without claiming that unrelated historical split-session cleanup is repaired. The borrowed queue is established before session initialization; no later device switch exists on this path, and that queue outlives descriptor/arena teardown. Both queues are retired before host storage is freed; compute alone is destroyed last.

The retirement watchdog already encompasses teardown and the new frees. Unknown completion keeps existing exit86 semantics; delivered checked async failures retain87. A cleanup/free exception exits88 before descriptor deletion/success. This is a safe failure boundary by source control flow, not proof that an actual provider reclaims all allocations on failure. Partial accepted submissions, provider waits, allocation/free and loader unwind semantics remain unexecuted uncertainties.

Capture open failure has no stream. Write/flush failure retains unique ownership for one destructor close and remains failure even if that close also fails. On the successful write/flush path, explicit close consumes the released stream once; a close-only failure throws before completion/receipt and then Runtime retires/cleans up. A failed partial capture remains on disk for Main's retention decision; this delivery performs no cleanup. No success path discards a close-only error.

## Identity and preservation

The source patch is `7902` bytes, SHA256 `2bb6da22961869a3506ed7fe8c5138cd4509bc5fe9bd998b3774fb4399678fdc`. It is the full-index binary diff against the exact assigned commit. Final source identities are:

- `validation/pr6-restore/CMakeLists.txt`: 7109 bytes; SHA256 `5f7a282bd672cbbcf2fafee896df5daff6645acc5e66c80559ec8fe78094bd6f`.
- `validation/pr6-restore/harness/prefill_api.cpp`: 35576 bytes; SHA256 `c6d9684f87a86ea1da3a947eb2130dd88baadc2cdbbb101321d10da10153bd46`.

Read-only byte comparison found 215 other tracked validation files identical to the assigned commit, with 0 unexpected differences. Historical kernels/math, FP32 residuals, layer/chunk loops, QSA precision, source resolver, vendor/llama dependencies, D3 exclusion and root engine are untouched. JSON records before/after blobs/hashes, consulted contracts and input-report hashes.

The exact SHA function remains byte-identical (2566 bytes, SHA256 `e6d0e91307560bffab8432eba71793fcbeb117c9eace93941a64733710ecc6f7`). Main's supplied `own-pr6-harness-host-sha-v1.json` records33 successful ordinary-host cases for that same extracted function, including oversize rejection; this worker only read/pinned that receipt and did not rerun or widen its result. The supplied receipt SHA256 is `99a2afb95aeb72d937f563e320f311fe2ed7259fecfd8be5583e13dbe09cd050`. Old report statements predating this receipt remain historical evidence.

## Remaining gates

Main must review the complete patch and source contracts, then qualify actual configure/compiler/dependency/generated checked-header/target closure/ODR/ABI/link behavior. Explicit C++20 target properties are source settings until the generated compile command is inspected. No binary or provider approval follows from this delivery.

Before model loading, Main still needs the separately admitted finite model-free checked-provider queue/event/async/partial-submission/retirement/cleanup gates, with D3 excluded. After those, the new admission makes a real1-token then small odd/tail fresh-position-zero model check expressible; these checks remain UNRUN. Continue separate residual/logit/full KV/scales/GDN/conv/indexer/PLE state and task-quality comparisons. A small token count does not support reduced model geometry, continuation or RESTORE.

Long correctness and clean matched individual speed measurements follow the applicable earlier gates. Physical262144 remains UNQUALIFIED, numerical equivalence UNRUN, export/exit/fresh RESTORE/continuation/rewind/branch UNQUALIFIED and historical600 NOT_REQUALIFIED. Both MTP branches and draft-KV/draft/round/step/subset obligations remain separate. The native full-model CLI and full decode are still unavailable under D3; no override was introduced. CPU PLE arithmetic, host initialization/grouping and library/backend/wait CPU attribution questions are unchanged.

No measurement samples, logs, tensor captures, profiler output or binary were created or deleted. Main retains responsibility for immutable pre-exec environment, finite owned supervision,64MiB default raw-log budget, fault/cleanup receipts, integration and retention review. Timeout, budget stop or missing receipt cannot become a pass.
