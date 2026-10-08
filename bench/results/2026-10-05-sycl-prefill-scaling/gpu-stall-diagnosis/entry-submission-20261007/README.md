# API-entry diagnostics and a stalled submission, 2026-10-07

Arc B570 10 GiB, Ryzen 5 5600X, 128 GiB RAM, xe kernel 7.0.0-38-generic,
NEO 26.31.39395.14, Level Zero loader 1.32.0 and oneAPI 2026.1.1.
Direct submission, UR V2 copy offload and persistent SYCL caching are disabled.
No reset, rebind, reboot, service interruption or driver/firmware update is
performed. Both model runs leave no live inferior or debugger and record no
new xe fault in their kernel-journal windows. Real logged GPU health probes
pass after each cleanup.

## A smaller diagnostic log

The short first-check profile in `diagnostic_environment()` continues to
record Level Zero entry/results, parameter validation, flushed UR tracing and
Strata progress. The long runs here start from that helper and suppress
successful argument dumps (`ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT=0`) and
the additional `UR_LAYER_TRACING` layer. UR loggers flush warning messages.
The Level Zero pattern is `[thread-id:%t] [%l] %v`; the supervisor records
monotonic elapsed time. API entries and error results remain enabled.

The tagged [loader documentation](https://github.com/oneapi-src/level-zero/blob/v1.32.0/README.md#logging-api-calls)
describes this successful-result switch. Before the first long run, the
compact profile's actual B570 probe passes three H2D/kernel/D2H rounds of
16,384 exact words. Its 22,937-byte stderr contains 263 API entry lines.
These are diagnostic settings, not a timing profile. Omitting the timestamp
from the text does not remove all logger work: the second model run's stack
still includes libc time-zone conversion inside the validation library.

The two model runs use frozen executable
`3a83e2c0c8b916225669275217b9a194ebabbe6101faf0ea7a110913d061e9b9` and
the original full CLI's tuning, checked before launch. Both request exactly
262,144 capacity, with 262,142 input tokens and two output tokens, normal MTP
spec 4, prefill 1024, forced layer-major processing, compact workspace 2,
in-place residual reuse, cache restoration from immutable RAM and verified
draft-weight leases. Neither completes the prompt or emits a token.

| Case | Elapsed observation | Private API stderr | Terminal result |
| --- | ---: | ---: | --- |
| `full-context-entry-capacity` | 966.239 s | 344,377,090 bytes | Unchanged submission in repeated stacks; owned process stopped |
| `full-context-entry-csr` | 300.049 s | 1,096,144,595 bytes | API and I/O progress continues; finite diagnostic deadline |

The full flushed API logs remain private. `api-log.json` stores their SHA-256,
entry/result counts and Strata progress lines; `api-log.tail.txt` contains the
last 64 KiB. The archived GDB/MI streams and snapshots are complete.
The only numeric non-success results in these two API logs are respectively
514 and 2,306 graph-capture queries returning documented `ZE_RESULT_QUERY_FALSE`
(`0x78000023`). These query results alone do not establish a fault.

## Stalled host submission

In the first case, the recorded API-write, read and page-fault signature last
changes at 92.683 seconds. It stays unchanged through the last 964.228-second
sample. The main thread consumes CPU; the device busy counter increases.
Thus a changing busy counter alone is insufficient evidence of workload
progress. The original automatic observer included that counter in its
signature and missed the stalled API log; the subsequent controller uses
API-log size and I/O/fault observations instead.

Two requested snapshots and the final stop snapshot retain the same main
thread path: `Prefill::run_layer_major` / `run_impl`, expert SwiGLU submission,
UR V2, Level Zero validation, NEO CSR flush, `IoctlHelperXe::execBuffer` and
`ioctl`. All three show submission/completion argument 732773 and syscall
return register -11 (`EAGAIN`). This is the argument being submitted, not a
measurement of the number of outstanding jobs. The last API entry is
`zeCommandListAppendLaunchKernelWithArguments`.

The matching [NEO ioctl loop](https://github.com/intel/compute-runtime/blob/26.31.39395.14/shared/source/os_interface/linux/drm_neo.cpp)
retries according to the [helper predicate](https://github.com/intel/compute-runtime/blob/26.31.39395.14/shared/source/os_interface/linux/ioctl_helper.cpp),
which includes `EAGAIN`, without a retry count or sleep in that loop.
Upstream [xe execution](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/xe/xe_exec.c)
returns `EAGAIN` when the queue's job count reaches the
[1,000-job limit](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/xe/xe_exec_queue_types.h).
Read-only disassembly of the installed module corroborates the comparison
against 999 and an `EAGAIN` return branch. Source URLs and digests, module
digests and disassembly are retained under `upstream-source-receipts`.
The loaded module's sysfs build ID matches the disassembled installed module.

This makes job-count pressure a lead, not a measured cause: no live kernel
job count, CSR completed counter or blocking GPU command was obtained.
Strata already drains at the router and chunk boundaries, and GEMM uses the
configured compute queue. This evidence does not justify claiming that a
new periodic wait prevents the problem. It also does not prove that the
earlier unlogged two-hour CLI failure had the same cause.

## The bounded repeat does not reproduce the stall

The second run continues changing its API/I/O signature through its last
297.836-second sample. At the 300-second observation deadline, its stack is
in validation-library logging/time conversion under oneMKL expert GEMM and
layer-major prefill. The stalled-submission condition is not seen in this
window. Its planned read-only CSR script, with inferior function calls
disabled, is consequently not executed; no CSR counter result is claimed.
The deadline snapshot resumes its own interrupt before bounded cleanup.
Debugger pauses and logging affect execution timing.

The old chunk traces repeat positions for successive layers and identify
chunk starts, not completed full-context processing. The candidate now adds
the zero-based half-open layer range `[begin, end)` to each start trace.
Its build-only receipts are separate from these two executed artifacts.
No arithmetic is changed. Completed verifier windows remain the required
proof that the final allocated KV cell executed.

The final trace build passes and freezes executable
`b81a7d6fbc1c6d524cf3b64e196f5866c91ec56c76e460a66b63b28ce3aed461`.
Its final archive/link preserves all 148 object-file digests observed before
that step; native CPU task-factor remains 0. The earlier failed build receipts
are retained: imposing the GPU library path on Nix binutils causes a libc
ABI error; the attempted archiver reconfiguration also changes one `ggml.c.o`
digest and is recorded; removing that library path lets archives complete but
exposes the missing MKL link search path. The successful final link explicitly
sets installed `MKLROOT`/`LIBRARY_PATH` and leaves `LD_LIBRARY_PATH` unset.
The new trace binary has not run on the GPU. The two model diagnostics above
use the earlier frozen executable and their original source versions.

Full 262,144-cell CLI and normal-MTP serve correctness, cancellation and
restoration-failure integration, repeat performance and general prevention
remain unvalidated. No PP 1000/TG 70 achievement is claimed.

## GDB serving transport checks

`owned_gdb.py` can now place a newly owned serving child's stdin/stdout on a
caller-owned PTY while keeping GDB/MI separate. The exec shim redirects only
the inferior's stderr to private `inferior.stderr` before executing it.
No shell quoting, root access, ptrace-policy or GUI change is required.

Two actual CPU/GDB cases pass in `owned-gdb-pty-cpu-fixed-path`: exact small
and 1,260,008-byte requests/replies, stderr isolation, stack capture/resume,
normal exit, and forced cleanup of a blocked child with no survivor.
The original seven CLI lifecycle cases and overwrite refusal pass again
with the same new helper in `owned-gdb-with-tty-cli-cpu`.

`owned-gdb-pty-cpu` preserves the initial failed harness receipt: the compiled
fixture path collided with the output-directory name, before GDB launched.
The corrected fixture is named `protocol-probe`. This is a harness error,
not a GPU failure. These CPU checks do not establish physical serving or
full-context correctness.

`sources-used/owned_gdb-before-tty.py` is the exact helper loaded for the
first model run. The second run and the successful CPU transport/CLI checks
use the separately frozen current helper. `manifest.json` records each
file's digest and maps recorded source digests to these versions. Unrelated
kernel-journal rows and full environment credentials are not copied.
