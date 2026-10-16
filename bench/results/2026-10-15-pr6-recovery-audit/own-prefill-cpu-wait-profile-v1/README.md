# Own prefill CPU wait profile v1

Source-only, unbuilt and untested proposal prepared 2026-10-16. No test, build,
GPU, profiler, process inspection/control, or Git command was run. Only this
proposal directory was written. Main owns review, integration and all validation
when the clean measurement cohort has closed. This is not a performance result.

Base supplied by Main: `617647659f21487ff3bf61fa1fd4ba87ba2520bf`. All seven
existing touched source files were copied and SHA256-pinned before preparation;
`pinned-base.json` preserves that initial record. `manifest.json` adds proposed
hashes and the new header. `source.diff` is generated against those same copies,
never a later live version. Header and preparation helper are newly authored;
the root MIT LICENSE applies. No XeStrata source, archived observer/helper
implementation or third-party implementation was read or used. Active source
and the R751 research report were read. Existing native_mmvq.cpp attribution to
llama.cpp remains byte-preserved; the proposal only wraps its existing waits.
The project-only upstream skill was read for provenance/scope constraints; this
is not an upstream check or implementation adoption.

## Behavior and boundary

Only exact `STRATA_PREFILL_CPU_WAIT_PROFILE=1` enables this diagnostic. Its
getenv occurs on the calling thread in Prefill::run before creating the scope,
only after long layer-major dispatch was selected. There is no runtime-init
dependency, process environment mutation, async signal handler, sampled stack,
wait-policy change, new GPU event/command, allocation or change to arithmetic,
placement, synchronization or scheduling. CUDA/HIP source behavior is preserved.
The short run/warmup and decode paths never open a scope.

The scope surrounds the entire selected run_layer_major call. It includes
argument qualification, source copies, allocation, pass execution, callbacks,
entry/final/lease drains and local-owner destruction (including frees/cache
retirement). The scope closes AFTER run_layer_major returns, including its
function-try-block handling. Successful completion is the actual returned bool;
false/exception returns remain incomplete. It emits one stderr key/value line
with caller wall/CPU, union wait wall/CPU, CPU outside waits, aggregate counts,
counts and CPU/wall by compute queue, transfer queue, event, compat queue, and
direct queue. It does not use the existing timing observer or add its events.

Header-only inline thread_local state is shared across host translation units.
Only this calling thread is observed; the production caller is expected to be
the native main thread. It does not identify or enforce the OS main TID. A future
caller on another thread would report that caller's CPU under main_cpu_s; Main
must retain the invocation/TID proof with each run. Workers never inherit scope.

Each explicit wait is wrapped immediately around wait_and_throw, with queue
lookup and event submission outside the interval. The existing exception,
sticky-error, drain-both-queues and fail-stop cleanup behavior is preserved.
An RAII wait guard accounts thrown waits and rethrows through the unchanged
caller; nested waits increase depth, but only the outer interval contributes,
so there is no double counting. Current call graph has no naturally nested
wrapped wait. Recursive profile scopes are rejected as unavailable, and invalidate
the parent diagnostic. Destructors are noexcept. Fatal abort/_Exit cannot unwind
or promise a report; an absent line is an incomplete diagnostic, never zero.

Clock sampling uses steady_clock wall and Linux CLOCK_THREAD_CPUTIME_ID CPU.
Clock failure, backward values, open waits or an impossible union invalidate
ALL reported durations to -1 and set available=0/clock_or_accounting_error=1.
Non-Linux CPU measurement is unavailable, never a fabricated zero. Count and
throw fields remain diagnostic evidence when durations are invalid. errno is
preserved around clock sampling. Disabled waits perform a TLS pointer check
and ordinary trivial object setup only; no time syscall, lock, getenv or output.
Scope construction still zeroes fixed stack storage and reads the opt-in at long
dispatch. Enabled sampling overhead contributes partly inside wait brackets;
outside_wait_cpu is therefore a residual, not an exact CPU-work taxonomy.
Reporting itself is excluded from measured time and may use libc stderr locking.

## Explicit coverage and limits

Source review found these actual wait_and_throw sites and wraps all of them:

- Runtime::wait(): compute and transfer separately; Runtime::wait(event): event.
- compat wait_all(): each registered queue; StreamDestroy and StreamSynchronize:
  compat queue. compat memcpy/memset/event/query/elapsed/graph retirement paths
  already route their blocking event waits through Runtime and are covered there.
- prefill/sycl/postops.cpp route fallback: direct queue.
- kernels/sycl/native_mmvq.cpp: all three direct queue sites.
- kernels/sycl/iq.cpp: both direct queue sites.

launch.hpp finish, device.cpp, native_q2, s2_experts, gr, cvec and kv_stream use
Runtime::wait and are covered transitively. Runtime::wait_for has no active
callers per R751; its final blocking event wait would be covered, but its
status-poll/yield loop would count outside waits. No cause is assigned to it.
Prefill's host Stager condition waits are not SYCL waits; long layer-major rejects
live Stager jobs and disables stream_all. Host source condition waits, future
get/join and CPU work remain outside. Implicit blocking inside submissions,
alloc/free, queue/context/event destructors, get_info/get_profiling_info, runtime
error delivery, graph finalize, mutexes or provider initialization is NOT separately
attributed to explicit SYCL waits and stays outside the wait brackets. No internal
UR/Level Zero function or busy-poll instruction is established by these timings.
No CPU/wall ratio alone proves call-stack attribution or scheduling policy.

## Reproduction and integration (Main only, not executed)

Inspect the complete diff and these limitations first. Independently verify that
the current revision is the supplied base and each touched file matches its
pinned SHA before applying. Preparation helper never applies the patch and
refuses an output beneath the source tree or an existing output directory:

```
python3 bench/results/2026-10-15-pr6-recovery-audit/own-prefill-cpu-wait-profile-v1/prepare.py \
  --source-root /home/yayoi/orca/workspaces/Strata/sycl \
  --output /tmp/strata-own-prefill-cpu-wait-profile-v1-review
```

It checks live base and stored base/new hashes, regenerates source.diff and
materializes only a proposed overlay. A mismatch means rebase and full review;
never replace active files from a stale snapshot. After review Main can use its
normal checked patch application workflow. No CMake changes are required.
Actual root resolver is CMakeLists.txt -> cmake/sycl_backend.cmake: runtime and
compat compile into strata_sycl_runtime, direct waits into strata_kernels and
strata_prefill, Prefill::run into strata_prefill, actual executable strata uses
src/program/generate.cpp. Existing configured root tree is build-sycl-aot
(CMAKE_HOME_DIRECTORY points here, Release, icpx 2026.1). Once serialized and
using the normal toolchain environment, build the existing production target:

```
cmake --build build-sycl-aot --target strata -j2
```

Record full command/environment, actual dependency/compile proof and binary
SHA; an unchanged binary cannot validate header integration across libraries.
Do not configure/build vision or the ggml oracle as a substitute. Do not use
this untested proposal's measurements as clean throughput samples.

## External validation plan (not executed)

1. Main reviews production header on host only first. Compile a small standalone
   consumer using this real header, not duplicated accounting code. Check scope
   off produces no line and no clock calls (linker clock_gettime wrapper counter);
   simple scoped CPU work plus sleep yields nonnegative totals and residual.
   Check repeated waits, nested Wait guards, nested scopes, thrown wrapped calls,
   false completion, scope unwinding, two separate threads and successive scopes.
   Verify count/throw totals and union accounting; worker activity must not enter
   caller totals. Use explicit per-consumer parameters, no environment mutation.
2. With a host-only link wrapper for clock_gettime, force first/read-end failure
   and backward CPU timestamp; require available=0 and every duration=-1. Compile
   a non-Linux preprocessor arm on an appropriate host and require unavailable.
   Check no exception escapes destructors, original exception survives and errno
   remains unchanged by clock sampling. Also exercise deliberate open-Wait misuse
   only in an isolated host harness to confirm invalid/open_wait reporting.
3. Main builds the production target after the cohort. Run existing relevant SYCL
   runtime/compat/prefill checks through Main's normal serialized ownership/fault
   harness. Use separate processes with opt-in absent, 0, 1 and malformed values.
   No getenv mutation after runtime initialization. Verify only exact 1 emits;
   short prefill/warmup/decode emit no line; long selected run emits exactly one.
4. Run one bounded long-input diagnostic through the existing owned harness with
   the same chunk/context/model settings as the clean cohort. Compare outputs,
   state and status against baseline; preserve existing physical 262144 correctness
   requirements and original failures. Verify completed vs cancelled/exception
   evidence, all wait categories and aggregate arithmetic. Check measured caller
   CPU against independently bounded /proc caller CPU sampling over matched phase
   endpoints, accounting for clock/observer overhead and read-window differences.
   Optional failure injection remains a separate process and must not weaken
   checked queue retirement. Fatal abort may have no summary; record incomplete.
5. Retain each compact line, source/binary/environment/input identity, independent
   samples, original correctness/fault outcomes and interpretation. Do not infer
   waiting consumes the hot core until material cumulative wait CPU explains the
   caller CPU. High wait CPU/wall supports on-CPU time inside blocking API calls;
   precise internal polling attribution remains unresolved without stack evidence.
   Review added overhead separately with clean disabled/enabled processes; do not
   claim zero overhead. Follow ARTIFACT_RETENTION, 64 MiB raw log ceiling per job,
   and Main's measurement lock. No log cleanup during active GPU work.

No host harness is included to avoid introducing unreviewed test hooks or copied
production logic. All checks above are outstanding; no compilation, functional,
numerical, or performance pass is claimed. Next review owner: Main, before patch
application and again after external validation.
