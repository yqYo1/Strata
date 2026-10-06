# First-check API logging, 2026-10-07

Host: Ryzen 5 5600X, Arc B570, installed Level Zero loader/validation layer
1.32.0, NEO 26.31.39395.14, oneAPI 2026.1.1, Level Zero V2. Direct submission
was disabled in the enumeration probes. No device reset, service change,
GPU context creation, device allocation or GPU command was issued by these
probes. This is a logging check, not GPU arithmetic or full-context validation.

`debug-run.py` and the recovery health check now share
`diagnostic_environment()` in `recover-xe.sh`. They enable Level Zero API
entry/results, successful results, parameter validation, UR tracing and
Strata progress on stderr. The wrapper preserves the caller's runtime and
tuning choices. Clean `health_environment()` removes inherited `ZEL_` flags
as well as `ZE_`/`UR_` flags, so they do not silently contaminate timing runs.

The first `loader-only.py` probe called `zeDriverGetApiVersion` before
initializing the loader and used a null driver handle. It exited with SIGSEGV
and no trace output. That was an incorrect probe call sequence, not evidence
of a Strata or GPU failure. Its original receipt and empty output files are
retained rather than overwritten.

The corrected `loader-enumeration.py` initializes the GPU driver, obtains
valid driver handles, then queries the API version. All four calls returned
success. stderr contains timestamp/thread-ID entries and successful returns.
The driver reports API 1.17. No context, allocation or command APIs are used.

`live-flush-and-ur.py` also initializes the UR loader, enumerates/releases the
V2 adapter and tears the loader down. It then waits for one byte on stdin.
The parent observes both Level Zero return logs and UR API logs while the
child is still alive, snapshots stderr, then lets it exit normally. This
verifies that useful logs are present before process exit. See
`live-flush-record.json` and `before-child-exit.stderr`.

The existing fake-backend recovery tests passed: 18 core and 30 display
test methods. Shell syntax and wrapper Python syntax passed. These do not
submit GPU work. The updated helper was installed without invoking it;
`installation.json` records before/after hashes. The already running 256K
CLI retains its original environment and remains outside this logging check.

The settings follow the
[Level Zero 1.32 API logging documentation](https://github.com/oneapi-src/level-zero/blob/v1.32.0/README.md#logging-api-calls)
and [UR tracing documentation](https://oneapi-src.github.io/unified-runtime/core/INTRO.html#tracing).
Parameter checks and a successful enumeration do not prove dependency,
memory-lifetime, kernel or full-capacity correctness. Do not use these
diagnostic timings as performance measurements.
