# R816 Main-owned R810 qualification and paired GEMM

**SOURCE ONLY / UNRUN.** No configure, compiler/linker, syntax checker, binary
load, test, GPU query/job, process inspection, profiler, cleanup, package change,
service change, environment change, Git operation or production edit was run by
this worker. Only source/policy/metadata reads, file hashing, package writes and
Orca lifecycle commands were used. Main must review the complete package before
executing it. Source availability is not qualification, adoption or speed.

The package writes only here. Main's future controller writes fresh explicitly
named private roots under the existing external B state directory, under the
existing exclusive `owned-v0141-measurement.lock`. No other package is modified.
No R810 derived kernel is copied here. New files follow Strata's MIT LICENSE;
R803's MIT benchmark and owner/controller interfaces are the implementation
references. R810's original MIT adapter and BSD-3 Intel core retain their actual
licenses in their original package. XeStrata implementation is excluded: only
the previously approved official LLVM `.tools` installation is referenced.

## Build and ABI

`CMakeLists.txt` invokes the pinned existing R810 CMake project as a private
subdirectory, with its original fixture ON and exact TLA worktree revision
`87f6850680a580654b9ea2c80dbc01aeb36ad231`. It adds an evidence-only fixture
entrypoint: the original `numerics_chain.cpp` has only `main` renamed by a
per-source compile definition; `fixture_entry.cpp` calls it and retains its
original return and stdout. Numerical comparisons, cases, guards and checked
drains are unchanged. The original fixture still compiles current pinned
`src/sycl/runtime.cpp` into its own executable.

The benchmark links the R810 component static library plus the **existing v2**
Gemm/Runtime/project/ggml archives in their original link order and the same
four oneMKL libraries. It never links the fixture Runtime object or rebuilds
an archive. The qualified v2 source revision is
`9c1a2020897a9451ec616a6bb036a661de9d86b0`. Its full 419-file source-pin receipt
and unchanged 68-case receipt are required PASS gates. Current source drift
fails closed; changing a pin requires a new reviewed source proposal.

Native TLA uses R810's exact `spir64_gen`, BMG G21 `bmg-g21-a0`, split-barrier,
block2D and subgroup-matrix translator extensions, strict FP options and GRF256
backend request. No arbitrary compiler/runtime/provider substitution, install,
fetch or download is implemented. Configure and build are private and serial;
**build never launches a smoke test**. Offline AOT/ocloc may query hardware, so
build is conservatively device-touching and has journal/dump guards too.
The approved installation has `llvm-ar` but no `llvm-ranlib`; this package does
not invent that path. Existing system ar/ranlib and LLVM ar are pinned, and
CMake's actual selected commands are retained for Main's review.

The controller preserves `--verbose` actual compile/link output, actual Ninja
recipes, compile entries and actual Ninja include dependencies, with file hashes.
Header-root tree digests cover official LLVM headers/resource headers, system
headers, MKL headers, Strata includes and the existing ggml includes before and
after work; R810's 207-header TLA closure is separately pinned through its sealed
manifest. Compiler, helper tools, current runtime/driver libraries, linked
archives and DSO identities are pinned before/after. Missing tools or changed
headers/libraries fail; there is no pin refresh, fallback or automatic retry.

Tool acceptance, actual device ISA and all execution remain **UNRUN**. A symbol
named DPAS/XMX does not establish generated instructions or absence of spills or
emulation. A bounded optional Main-owned CPU-only first inspection is the pinned
`/usr/bin/readelf -S -W` on the closed build's executable/component objects under
Owner (for example, 30 s, 256 MiB RSS, 1 MiB text). It inventories ELF sections;
it is **not** DPAS proof. The controller does not force an unverified ISA tool
option or fake an ISA acceptance gate. Main must separately inspect actual
images before any production adoption.

## Explicit run sequence

Every invocation requires an explicit fresh `--root`, reviewed
`--manifest-sha256`, explicit latest closed `--prior` and `--prior-sha256`, plus
`--confirm-latest-closed-prior`. No mutable latest path/hash is baked in. Main
checks the supplied latest receipt rather than substituting an old PASS.
Every dependency receipt argument has a separate explicit SHA256 argument.
Use `/usr/bin/python3 -B main_controller.py MODE ...` after source review.

| Mode | Additional receipt arguments | Native work |
| --- | --- | --- |
| `build` | none | Private configure/build only; no smoke |
| `smoke` | `--build-record PATH --build-record-sha256 SHA` | First R810 fixture, `--device 0 --suite smoke`, flushed UR tracing + parameter validation |
| `check` | build pair and `--smoke-record PATH --smoke-record-sha256 SHA` | `guards`, `edges`, `chain`, `full`, each a separate bounded owned process |
| `measure` | build/smoke pairs and `--check-record PATH --check-record-sha256 SHA` | Tiny traced paired smoke for both formats, then 18 clean measurement processes |

Roots must be nonexistent direct children of B named `main-r816-*`; each mode
gets its own root. All receipts and products must match this manifest and exact
binary hashes. All component suites must PASS before **any** paired benchmark.
The explicit ordinal 0 is Main's qualified B570. The benchmark calls
`runtime_for(0)` and R810 rejects a non-BMG architecture, non-Intel Level Zero
GPU, unsupported subgroup/geometry or foreign queue before submission.

A normal failed-but-closed preceding receipt is accepted solely for ownership
continuation when Main additionally supplies all three:

```
--closed-failure-receipt SAME_PRIOR_PATH
--closed-failure-sha256 SAME_PRIOR_SHA
--manual-reviewed-continuation-reason 'Main-reviewed reason describing the original failure and permitted next action'
```

It must have `complete=true`, `active=false`, the same boot, no fault/dump or
incomplete final check, every command's normal ownership observation/reap/session
closure, and all recorded PID/start identities absent now. `module.closed` is
not exit-zero success. A normally closed compile exit1 is distinct from a hung
process that required termination or lost ownership, which this controller
refuses. Original native exit1, correctness failure or metadata failure remains
FAIL; none can qualify build/smoke/check. The continuation reason and original
receipt SHA are recorded. There is no retry loop or failed-kernel automatic
rerun. The oneDNN smoke-v2 receipt is known **overall FAIL**, despite native0,
finite oracle, normal closure and fault0, and must never be consumed as a PASS.
The proposed DSO allowlist explicitly pins installed TBBmalloc; doing so neither
rewrites that prior failure nor qualifies the separate oneDNN component.

Each native case has fresh incremental kernel-journal boundaries from a fresh
`journalctl -k -b -n 0` cursor, current xe/vendor/dump checks, unchanged boot and
pins, exact direct_owner PID/start/session evidence, and normal exit requirements.
Only the pinned Owner may terminate its own pinned identities. The model-owner
scan is read-only and happens only during actual Main execution; it cannot
prove hidden foreign GPU use absent. No signals to discovered processes,
reset, dump removal, simultaneous job or cleanup is implemented.

The original fixture's declared/required actual counts are smoke32, guards26
rejects, edges16 calls/272 printed edge rows, chain8, full120. Complete original
stdout and case records are retained. Guards require their original checked
drain/readback PASS, original return zero, wrapper return marker, normal owned
exit and clean fault window. Merely observing no asynchronous errors after
synchronous rejects is insufficient. Strict gradual-underflow/denormal failures
remain failures and block measurement. A separate explicit Main contract decision
requires separately reviewed source; there is no FTZ waiver flag here.

Own-process `/proc/self/maps` snapshots before Runtime and after checked return
pin actually observed DSO paths without the PLT `dladdr` ambiguity. Unlisted or
deleted mappings fail. Runtime/provider/driver mappings are required, and paired
processes additionally require the four oneMKL mappings. Before/after snapshots
do not inventory a library unloaded between them; no stronger tracing claim is
made. If a library unloads before the post-return snapshot, the missing-required
mapping gate fails and needs source review, never silent relaxation.

## Paired operator semantics

| Route | Producer | Stored output | Global F32 intermediate |
| --- | --- | --- | --- |
| `old_f32_direct` | Existing Gemm FP16/BF16 oneMKL | FP32 | Only its FP32 output |
| `old_f32_plus_gpu_bf16_cast` | Same oneMKL plus one GPU RNE cast | BF16 | oneMKL output |
| `native_bf16_output` | Current Gemm `f16_out16`/`bf16_out16` | BF16 | None |
| `vllm_compact16x64_bf16_output` | R810 `beta0(compact_16x64)` | BF16 | None |
| `vllm_wide128x128_bf16_output` | R810 `beta0(wide_128x128)` | BF16 | None |

All use same immutable resident X `[T,K]`, W `[N,K]`, same input format, beta0,
`ldy=N`, one Runtime and same in-order compute queue. FP32 accumulators remain
local in native kernels. The process also allocates baseline output, but native
routes cannot access a global FP32 pointer. No oneDNN route, pack, transpose,
CPU inference arithmetic or environment switching is added. All spans use
registered `Allocation(Device)`, with dynamically aligned 64-byte live starts
and at least 64-word prefix/suffix guards. Borrowed Gemm/runtime/queue and all
host/device owners stay alive through checked drain; failed drain `_Exit`s
without unwinding owners. Readback's host destination is retained until checked
completion or fail-stop, including an exception path.

Shapes `[T,N,K]` are `[160,1280,2560]`, `[160,2560,640]`,
`[8192,1280,2560]` for FP16 and BF16 inputs. Each gets 3 independent native
processes. Every process runs 3 warmups and 3 timed submissions **per route**;
each round rotates the five-route order, offset again by process index. Timed
round rotation is `(j + round + 3 + process_index) % 5`. The controller retains
all 540 invocations (270 timed submits), their process/owner identities,
individual wall samples and validations; inner submissions are not independent
process replicates. The two tiny traced processes use `[3,128,64]`, three
warmups per route, and no timed performance samples. Clean measurement has no
UR tracing/validation layer or profiler environment. Fixed safety/provider
values are inherited from qualified v2/R803 before exec for every child.

Wall units are seconds from immediately before selected route submission through
same-queue `wait_and_throw` and sticky-error check. They include entry guards,
submission, GEMM, cast when selected and completion. The two thread CPU deltas,
also seconds, separate submission from wait/check, including completion polling;
they do not imply CPU inference or establish a latency bottleneck. Small clock
query overhead is present. Uploads, poison reset, oracle, guards, hashes,
readbacks, DSO snapshots and immutable-input comparison are outside wall.
No device-event/profiler time or tok/s is claimed.

R803's independent full-output rank-two/dyadic oracle is retained:

```
A=r%7-3; B=(r/7)%5-2; D=c%11-5; E=(c/11)%7-3
s(k)=eight +1 then eight -1, repeating
X=(2*A+B*s)/32; W=(2*D+E*s)/64
Y=(K*(4*A*D+B*E)+sum(s)*(2*A*E+2*B*D))/2048
```

All inputs and finite FP32 products/sums are exact in the selected bounds.
Every FP32 element must match exactly, and every BF16 element must match the
independent host FP64 mathematical RNE encoder (either exact-zero sign allowed).
Every invocation checks all outputs finite, complete output guards and bytewise
immutable X/W including guards. BF16 output is direct storage or one explicit
GPU BF16 conversion as declared; no global FP32 materialization in native paths.
FNV1a word-stream hashes are diagnostics; SHA256 pins sources, tools and binaries.
This low-rank pattern supplements the original R810 suites and unchanged old68;
it is not model quality, adversarial accuracy or full262144-context acceptance.

## Limits, retention and remaining work

Per child: build1800 s / 32 GiB RSS; native/utility600 s / 8 GiB RSS; combined
text64 MiB, individual file64 MiB. Whole modes: build2400 s, smoke360 s,
check2400 s, measure14400 s. Root budgets: build2 GiB, other modes512 MiB.
Budget-triggered termination is failed/incomplete, never PASS. There are no
background jobs or cleanup. Header hashing is bounded to100000 files/2 GiB per
root and65536 directory/alias records; tree drift requires review, not regeneration.

Main keeps original samples, commands, frozen environments, source/binary/library
pins, actual case counts, ownership/fault results and original failures. Full
normal UR traces remain only until Main verifies compact proof; Main then applies
ARTIFACT_RETENTION.md in a separate serialized step with path/hash/reason manifest.
Keep a representative unresolved failure diagnostic and its actual mechanism.
This package deletes nothing and creates no raw tensors or model captures.

The research index's native16 status metadata and named past-report paths are
recorded in `manifest.json`; no knowledge-collection body was read. Existing
FP32-v2 68-case and 32K three-arm natural-screen mechanics PASS are unchanged;
BF16 quality remains UNACCEPTED and current native16 performance regressed.
Main still owns build/tool acceptance, all suites, strict denormal contract,
actual ISA/image review, paired operator execution, quality, whole dependency
path performance and physical262144-context gates. **All new execution: UNRUN.**

Main source review correction before the first compile: the historical v2 build
receipt intentionally has GPU_executed=false and only configure/build/two host
checks, without GPU fault fields. Only that exact hash-pinned receipt is accepted
as build-only evidence. The independent 68-case GPU receipt, latest closed prior
and current fresh fault windows still require clean GPU evidence. R816 build-v1
failed at preflight with no owned command and no GPU/compile; its FAIL is preserved.
