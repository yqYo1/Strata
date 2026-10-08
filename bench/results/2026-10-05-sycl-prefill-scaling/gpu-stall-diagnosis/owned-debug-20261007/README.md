# Owned debugging and full CLI capacity, 2026-10-07

Arc B570, Ryzen 5 5600X, 128 GiB RAM, xe kernel 7.0.0-38-generic,
Level Zero loader 1.32.0, NEO 26.31.39395.14 and oneAPI 2026.1.1.
Direct submission, UR V2 copy offload and persistent SYCL caching are disabled
in these GPU diagnostics. No reset, service stop, firmware or system setting
change is performed.

## Original full CLI did not complete

The frozen `79a4b363d33f66f41d910be6274e609b4eb73f62afb0dc49bca72bf2a538d223`
candidate attempted exactly 262,144 capacity with 262,142 input tokens and two
requested output tokens. Both overflow refusals passed before model loading.
The CLI did not finish within its 7,200-second job deadline. Its memory/progress
observations cannot establish token processing: device busy cycles and read/
fault counters stopped changing while the main CPU thread remained active.
The supervisor records no new xe fault in the job's kernel-journal window.

The controller's short cleanup grace records PID 428740 as surviving KILL.
A later identity check finds the process absent; the earlier failed receipt
and stalled-writer marker are kept unchanged. After checking that the old
process was gone, a separate unprivileged probe performs H2D, a kernel and D2H
in three rounds and matches all 16,384 words each time. It exits zero with no
new xe fault/reset messages. Its complete Level Zero and UR logs, settings and
receipt are in `post-capacity-timeout-health`.

This establishes small GPU execution after cleanup without a reset. It does
not establish full-context capacity, prevention or the original wait's cause.
The original job did not enable API tracing, which cannot be added after
launch. The original controller and its unsuccessful terminal receipt are in
`original-full-cli`. The kernel subset records the complete private journal's
hash and row count; unrelated messages are not copied here.

## Owned debugger checks

`sycl/tools/owned_gdb.py` starts a fresh diagnostic process as GDB's child
rather than attaching to an existing sibling. With this host's ptrace policy
unchanged, seven real CPU-only cases pass: exit 0, 7 and 127; exact arguments
including spaces, quotes, backslashes and a literal dollar sign; two stack
captures with resumes and normal exit; SIGSEGV capture without resuming the
fault; and a blocked child with bounded cleanup. Debugger logs cannot be
overwritten. No live child remains after the checks. The actual backtraces,
GDB/MI streams and source hashes are in `owned-gdb-cpu-exact-argv`.
The final helper extends backtraces to 48 frames and copies only the new log
range without reading the entire old stream into memory. All seven cases and
the overwrite refusal pass again in `owned-gdb-cpu-extended-stack`.

The three earlier failed harness checks remain recorded. The first mixed
Nix compiler helpers with the host libc and failed before starting GDB. The
next two exposed GDB argument quoting mistakes. The final helper uses a
private JSON/exec shim to preserve argv without a shell; the inferior PID
stays the same across exec and GDB follows the real executable. These were
harness errors, not GPU failures.

The caller supplies a finite deadline and private output directory. Cleanup
checks both the debugger and its inferior, pinning the latter with a pidfd
before signalling it. A surviving owned process is recorded and blocks a
new GPU job. All-thread debugger pauses can affect host-dependent GPU work;
they must be recorded when interpreting the subsequent execution.

## Last allocated KV cell

CLI now prints `strata trace: window POSITION COUNT` after a successful
verifier window when `STRATA_TRACE` is set. The capacity controller requires
`(262141, 1)` followed by `(262142, 2)`, in addition to finite complete head,
two outputs and normal process exit. Thus a logical input/output count alone
does not pass; the last allocated KV cell must actually have executed.

The new build succeeds with no arithmetic change, producing frozen candidate
`3a83e2c0c8b916225669275217b9a194ebabbe6101faf0ea7a110913d061e9b9`.
Its build receipt and output are in `completed-cli-window-build`.
Nine CPU stub cases in `cli-capacity-trace-cpu` pass: valid CLI at 64/262,144,
rejection of missing/short/oversized/wrong-position traces, a boundary check,
and valid serve at 64/262,144. Enabled diagnostics retain Level Zero and UR
settings. These checks validate pass/fail guards, not physical GPU capacity.

Full CLI and normal-MTP serve execution with the last-cell check remain
required. Diagnostic timings are not throughput measurements.

## First model diagnostic had a tuning omission

`initial-diagnostic-tuning-omission` preserves an unsuccessful diagnostic of
the traced build at the same context and argv. Its baseline environment file
did not contain the Strata flags that the original capacity controller added.
The diagnostic inadvertently used default chunk-major prefill and a 256-token
first chunk. It reached chunk starts 0, 256, 1280 and 2304, rather than reproducing
the original layer-major path. This is a harness error; its observed progress
does not resolve the original wait.

Full Level Zero plus verbose UR argument tracing produced 2,147,745,910 bytes
by the 159.18-second log limit. The supervisor captured all CPU threads,
resumed its interrupt, then ended the diagnostic and verified that the inferior
and GDB were gone. No new xe fault occurred. A further logged small GPU probe
passes in `post-owned-debug-health`. The original 2 GiB stderr remains private;
its digest, exact non-success-result counts and last 64 KiB are retained here.

The logger's 61,594 `ERROR (2013265955)` lines are graph-capture queries with
`ZE_RESULT_QUERY_FALSE` (`0x78000023`), whose
[documented meaning](https://github.com/oneapi-src/level-zero/blob/v1.32.0/include/ze_api.h)
is that the list is not capturing. Their enclosing UR query returns success.
Those lines alone do not identify a fault.

The corrected diagnostic overlays the original executed controller's `env`
and checks every tuning value before launch. It retains full Level Zero API
logging and parameter validation, while disabling the additional verbose UR
argument/profiling layer to bound log growth. The unchanged original receipts
remain distinct from this correction.

## Corrected layer-major diagnostic

`corrected-layer-major-diagnostic` overlays all original executed tuning values
and asserts they match before starting the same 262,142-token fixture at
262,144 capacity. It keeps Level Zero API tracing and parameter validation,
with the additional UR argument tracing disabled. The layer-major path is
confirmed both by progress messages and the actual CPU stack.

The job reaches its finite 600-second observation deadline while device busy
cycles and processing progress continue. It does not finish the prompt or
emit the two required outputs. Its incomplete diagnostic receipt remains
`healthy: false`. The full 5,988,142,559-byte API log remains private; this
directory preserves its digest, result counts, Strata progress and last 64 KiB.
The all-thread snapshot includes `Prefill::run_layer_major`, `run_impl`,
`gather_rows16`, UR submission, the Level Zero validation/logging layer and
glibc time conversion. This records an active path; it does not establish a
stalled API or explain the original unlogged wait.

The supervisor resumes its requested interrupt, then closes the diagnostic
at the declared deadline. Neither GDB nor the inferior survives, and the
kernel window has no new xe fault. A separate small GPU execution check after
cleanup passes with complete API logs in `post-owned-debug-l0-health`.
Full CLI last-cell, normal-MTP serve, repeated correctness and performance
remain pending. Do not use this diagnostic's timing as normal throughput.
