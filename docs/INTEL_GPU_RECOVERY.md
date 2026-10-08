# Arc B570 recovery and failure investigation

On the investigation host, run this one command:

```sh
~/.local/bin/strata-gpu-recover
```

No arguments or separate sudo command are needed. The script asks for the sudo
password when necessary, checks the GPU's users, tries function-level reset
(FLR), and checks execution. If FLR fails without an unresolved safety condition,
it tries a bus reset only when the GPU is alone on that secondary bus. It stops
after the first verified recovery. The installed entry point is built from
[sycl/tools/recover-gpu.sh](../sycl/tools/recover-gpu.sh).

The script does not stop SDDM, Xorg, Orca or Codex. Xorg can hold B570 descriptors
and mappings even when every B570 connector is disconnected and another GPU
drives the monitor. Such ownership stops recovery before any reset. The earlier
GUI shutdown coincided with termination of this host's headless Orca service;
it cannot be treated as isolated from the running work environment.
The earlier automatic GUI-shutdown path is disabled, including its worker CLI.

Before a new reset, the entry checks for an interrupted, previously approved
display restoration from the same boot and caller. It can request SDDM startup
and check GPU health, but performs no reset or GUI shutdown in that path. It
does not reuse old-boot approvals. A running recovery or a stuck writer prevents
competing restoration. This compatibility path repairs only the earlier
root-owned 0644 plan inside its private 0700 directory; writable or linked plans
are refused. New plan replacements explicitly use 0600 regardless of umask.
Starting SDDM is a bounded request, not proof that the login screen appeared.

Other remaining clients, an attached B570 display, incomplete inspection and
interruption stop the sequence. The embedding service remains inactive and
disabled as requested; the script does not change it.
Client detection uses descriptors, mappings and blocked xe close stacks because
`xpu-smi ps` does not show every owner.

The result is printed on the terminal and a summary is saved at
`~/.local/state/strata-sycl/gpu-recovery/latest.json`. Detailed logs stay under
`/var/log/strata-gpu-recovery`. Exit 0 means the small GPU probe passed without
new xe fault/reset messages. Exit 3 meant the old detached GUI recovery was handed
off; it did not mean the GPU recovered. A health receipt must have
`healthy: true`, after another GPU probe covering the display-start request and
fresh kernel messages. A queued or interrupted receipt has `healthy: false`.
Exit 4 means recovery was declined or refused because a blocking condition
remains. Other failures report the step and its log. A successful
probe does not establish that a full Strata workload is correct.

The helper does not save devcoredumps, flash firmware, change xe's timeout/reset
policy, or reboot the host. It never stops services. It may start SDDM only to
finish the same caller's previously approved interrupted restoration. Each sysfs writer and GPU probe
runs in a separate process with a deadline. A writer still in uninterruptible
sleep after KILL is recorded by PID and start time; subsequent runs refuse to
compete with it. KILL cannot make a kernel D-state wait immediately disappear.

The internal helper is [sycl/tools/recover-xe.sh](../sycl/tools/recover-xe.sh),
installed alongside the entry point as `strata-xe-recover-core`. Its inspection
and developer controls are not needed for normal recovery. The legacy restoration helper is
[sycl/tools/recover-xe-display.sh](../sycl/tools/recover-xe-display.sh), installed
as `strata-xe-display-recover`; it cannot start a new GUI-shutdown worker. The probe is
[sycl/tools/xe-health.cpp](../sycl/tools/xe-health.cpp), installed as
`strata-xe-health`. Rebuilding it with oneAPI only compiles it:

```sh
icpx -fsycl -fp-model=precise sycl/tools/xe-health.cpp -o ~/.local/bin/strata-xe-health
install -m 755 sycl/tools/recover-xe.sh ~/.local/bin/strata-xe-recover-core
install -m 755 sycl/tools/recover-xe-display.sh ~/.local/bin/strata-xe-display-recover
install -m 755 sycl/tools/recover-gpu.sh ~/.local/bin/strata-gpu-recover
```

The host's four installed files are already prepared. The probe selects PCI
`0000:05:00.0`, uses an in-order Level Zero queue and checks all 16,384 integer
words after H2D, a kernel and D2H in three rounds. It uses no Strata model,
graphs, virtual memory or mapped polling words. The helper runs it as the
ordinary sudo caller using the installed Level Zero V2 adapter, persistent
caching off and copy offload off. Its library search path includes the installed
UMF 1.1 library required by the V2 adapter. An isolated dependency-load check
runs before any reset; a missing runtime refuses recovery. New kernel-journal
faults/resets invalidate a data match.

## Logs for first correctness checks

Use `python3 sycl/tools/debug-run.py EXECUTABLE ...` for initial SYCL checks and
failure investigation. It preserves the caller's runtime, device and tuning
settings, enables Strata progress, and writes UR and Level Zero API traces to
stderr. The recovery health probe uses the same diagnostic settings and saves
its environment in `health-environment.json` beside `health.stderr`.

The installed Level Zero loader is 1.32.0. Its
[API logging documentation](https://github.com/oneapi-src/level-zero/blob/v1.32.0/README.md#logging-api-calls)
requires `ZEL_ENABLE_LOADER_LOGGING=1`, `ZEL_LOADER_LOGGING_LEVEL=trace` and
`ZE_ENABLE_VALIDATION_LAYER=1`. We also enable successful results with
`ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT=1`, parameter checks with
`ZE_ENABLE_PARAMETER_VALIDATION=1`, and stderr output with
`ZEL_LOADER_LOG_CONSOLE=1`. Entries include time and thread ID, so a call with
no matching return is useful evidence of a wait inside that call.
The old `ZE_ENABLE_LOADER_DEBUG_TRACE` prints loader diagnostics and is
deprecated in this version; it is insufficient for tracking all API calls.

The [UR tracing layer](https://oneapi-src.github.io/unified-runtime/core/INTRO.html#tracing)
is enabled with `UR_ENABLE_LAYERS=UR_LAYER_TRACING`. Its logger flushes each
info-level call using `UR_LOG_TRACING=level:info;flush:info;output:stderr`.
Loader and Level Zero adapter debug messages also flush to stderr. Capture
stderr directly to a file when supervising a probe, preserving the last
entry even when the child never returns. Keep stdout separate for the serving
protocol. Parameter validation alone does not establish memory-lifetime,
dependency or device-kernel correctness.

These are diagnostic runs. Do not report their timings as normal performance.
Start a separate timing process with the trace and validation settings removed;
the clean `health_environment()` also removes inherited `ZEL_` settings.
Environment variables must be set before launch. They cannot turn tracing on
in a process that is already running.

For a long model investigation, a smaller log can retain Level Zero API
entries, error results and parameter validation while suppressing successful
argument dumps and the additional UR tracing layer. Record the exact settings
and elapsed time in the supervisor. This still changes execution timing; it
does not replace the detailed profile for short first checks.
Prefill's chunk-start trace also identifies its zero-based layer range,
`[begin, end)`, so repeated positions in layer-major processing are distinguishable.
These start messages do not establish completed token processing.

After the exact executable passes its short detailed check, a long capacity
diagnostic can further reduce logging to
`ZEL_LOADER_LOGGING_LEVEL=warn`, with parameter validation and
`STRATA_TRACE=1` retained, and UR loggers set to warning without the additional
UR tracing layer. Save the exact environment and elapsed time. This profile
does not retain every API entry; an owned debugger can inspect a stopped
submission. Validation and the debugger still affect timing.

For a wait that needs a CPU backtrace, `sycl/tools/owned_gdb.py` starts a new
diagnostic process as GDB's child. It can capture all threads and resume its
own requested interrupt without root or a change to the host's ptrace policy.
The caller must provide a finite deadline and a private output directory.
Its cleanup checks the debugger and inferior separately; a surviving process
blocks another GPU job. A debugger pause can affect host-dependent GPU work,
so record the pause and do not treat a fault during it as an untouched
reproduction. This helper is for diagnosis, not performance measurements.

Snapshots preserve the initially selected thread's registers and separately
locate the main LWP by the owned inferior PID. They also read the main
registers, 24 stack words and 16 instructions, then restore the original
thread selection. A watchdog abort may select a different thread from the
blocked submission; its syscall register must not be attributed to the
main thread. Unavailable memory is recorded without discarding the other
reads. The [four real CPU/GDB checks](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/main-thread-debug-20261007/README.md)
cover that distinction, requested-interrupt resume and a partially unreadable
crash snapshot. These are read-only diagnostic checks, not GPU capacity proof.

A serving diagnostic can pass a newly allocated PTY slave descriptor as
`inferior_tty_fd`. The caller owns the PTY and its stdin/stdout protocol;
the helper puts the inferior's stderr in `inferior.stderr`, separate from
GDB/MI and protocol replies. The
[CPU protocol checks](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/entry-submission-20261007/README.md)
cover a 1,260,008-byte request, exact replies, stack capture/resume and bounded
cleanup. These checks establish transport and process ownership, not GPU
serving correctness.

The [real model PTY check](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/layer-trace-20261007/README.md)
then completed four normal-MTP requests with the layer-range trace executable.
All output IDs, printed logprobs and complete finite heads matched the preceding
released-weight control. Six release/restore pairs and a normal exit were
recorded, with no new xe faults or surviving processes. This establishes the
short real-model diagnostic path.

The subsequent [full-cell CLI check](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/full-capacity-20261007/README.md)
completed on the same executable with 262,144 allocated context cells. It
executed windows `[262141, 1]` and `[262142, 2]`, used the final cell 262143,
emitted two tokens with a complete finite head and exited normally, without
new xe faults or surviving processes. It retained warnings, parameter
validation, Strata progress and an owned debugger; it is not a throughput
measurement.

The subsequent [normal-MTP full-context diagnostic](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/serve-submission-20261007/README.md)
stopped in layer 15 at the chunk starting at token 91,136. The first stopped
main-thread snapshot showed NEO's ioctl retry path returning `EAGAIN`. A
read-only CSR inspection found a submission counter 1,101 ahead of its
completion tag. That difference is not the kernel's live job count, and the
tag was sampled only once. The application watchdog aborted; owned cleanup
removed the engine and debugger. No new xe fault/reset was recorded, and a
logged three-round GPU probe passed afterward without a reset. Serving
capacity and general stall prevention remain unproven.

`STRATA_PREFILL_EXPERT_WAIT_BATCH` optionally waits for the existing in-order
SYCL queue after groups of direct-FP16 experts, before submitting the next
group. Values are integers from 0 through 256; the default 0 keeps the
existing schedule. MMQ grouping is unchanged. This diagnostic setting
changes submission timing, not the weights, row order or reductions, and
does not establish a limit on the driver's live jobs.
The [32-expert short check](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/expert-wait-20261007/README.md)
passed four actual normal-MTP requests. All output IDs, printed logprobs and
complete finite heads matched the released-weight control; 377 queue waits
and six release/restore pairs were recorded. Detailed API logging and
parameter validation were enabled. Its subsequent [full normal-MTP check](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/expert-wait-capacity-20261007/README.md)
passed the preceding stall position but stopped at layer 19, token 216,064,
and the application watchdog aborted. The main-thread stack was not
sufficiently resolved to identify the stopped call. No new xe fault/reset
or surviving child was recorded, and the logged small GPU probe passed
afterward without reset. Waits every 32 experts are insufficient for this
workload; the full-cell CLI gate remains pending. This is not a prevention
or performance result.

The [one-expert short comparison](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/expert-wait-one-20261007/README.md)
then passed the same four actual normal-MTP requests on the same executable,
with 16,677 waits and matching output IDs, printed logprobs and complete
finite heads. Detailed API logging and parameter validation were retained;
the updated owned debugger's CPU checks passed first. This condition still
needs its own full-cell CLI check. Its subsequent [full normal-MTP diagnostic](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/expert-wait-one-capacity-20261007/README.md)
stopped at layer 10, token 139,264. The requested main-thread snapshot showed
NEO waiting for completion target 8,582,366 through UR V2 `queueFinish` and
the added SYCL wait, without an observed main-register `EAGAIN`. The old
CSR reader could not select a completion-wait frame, so tag/counter values
are unavailable. The application watchdog aborted after the resumed pause;
both children were removed and the logged small GPU probe passed without
reset or new xe faults. Waiting after every expert is also insufficient.
This is not a performance or general prevention result.

The [installed legacy-adapter short comparison](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/legacy-l0-20261007/README.md)
passed the small three-round GPU probe and all four actual normal-MTP
requests, including complete head equality. The same executable used the
legacy adapter's documented copy-engine-off setting and no expert waits.
Its subsequent [full-capacity diagnostic](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/legacy-l0-capacity-20261007/README.md)
stopped while finishing layer 1's chunk from token 83,968 in an event
completion wait. The first watchdog abort was inspected without a preceding
requested pause; no live CSR/tag values were available. Both children were
removed and the logged V2 small probe passed afterward without reset or new
xe faults. A backend change alone is also unproven as a prevention method.

The [CPU-code-matched source mapping](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/prefill-wait-map-20261007/README.md)
locates that prefill frame at the compute queue wait before its chunk
callbacks. It does not identify which preceding operation was unfinished.
In this layer-major case all experts of the layer are loaded and the copy
queue drained before its chunks; the routed expert-copy ring is inactive.
The existing phase-mark synchronization can investigate smaller intervals,
but changes scheduling and needs its own short logged output check.
The [phase-sync short check](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/phase-sync-20261007/README.md)
completed four actual normal-MTP requests with matching IDs, printed
logprobs and complete finite heads. It captured 73,692 completed marks with
full API logging and parameter checks, without new xe faults or surviving
children. These marks name the next phase after waiting for earlier work;
they do not establish completion of that next phase's later kernels.

The subsequent [full-prefill phase diagnostic](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/phase-sync-capacity-20261007/README.md)
processed all 262,139 prefill tokens through all 48 layers. It then failed
allocating an 8 MiB physical segment when restoring MTP decode weights,
after verifier graph recapture. The main-thread stack records the lease's
restoration-failure `std::terminate`, not a completion wait. No requested
pause preceded the abort and no new xe fault/reset was recorded. Both owned
children were removed; the logged three-round GPU probe passed without
reset. This exposes a full-context restoration failure: no generated token,
complete head or capacity refusal case was verified in that run. Its
2,680.60 seconds and 20,839,201 synchronization marks are diagnostic data,
not throughput or general stall prevention.

The [typed K/V vector-load check](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/qsa-vector-load-20261007/README.md)
removes unrelated vector-pointer reads from the FP16 and INT8 eight-value
helpers. SYCL `vec::load` reads through the element type; `vec::as` provides
the defined bit conversion before the same pairwise half-to-float conversions.
CPU and B570 fixtures each passed 6,291,456 comparisons against the original
helpers and an independent IEEE binary16 reference. Four real normal-MTP
requests also matched complete finite heads, IDs and logprobs, with six
verified release/restore pairs, normal exit and no new xe fault/reset.
This covers those helpers and short output parity. It does not establish
their involvement in a hang, clean throughput, other attention casts or the
new candidate's full-context gates.

The [unused SYCL workspace check](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/workspace-reclaim-20261007/README.md)
removes 32 MiB from owned prefill storage and adds trace-only free-memory
queries around release and restoration. Four short normal-MTP requests
matched all IDs, printed logprobs and complete finite heads; six verified
release/restore pairs completed without force or new xe faults. Two separate
allocation-pressure probes freed a 1,300 MiB temporary buffer and restored
1,280 MiB of physical backing successfully. They did not reproduce the
full-model failure and do not establish that allocator caching caused it.
The subsequent [completed 256K check](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/profiling-20261007/README.md)
finished prefill but still failed an 8 MiB physical allocation during MTP
decode restoration. The new trace showed 1,411,559,424 bytes free after
temporary buffers were released, 131,244,032 after main-cache restoration
and verifier recapture, and 5,410,816 at the MTP allocation failure. No new
xe fault/reset occurred, and the exact-word probe passed after owned cleanup
without a reset. This remains a capacity/restoration failure; its output
and remaining full-context gates were not reached.

The same [profiler checks](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/profiling-20261007/README.md)
attempted installed VTune 2026.4. Its CPU software collection refused the
current ptrace restriction, and XPU Offload refused the Ryzen host's
microarchitecture. Intel PTI/unitrace was then built privately without
changing drivers or global settings. A fully logged instrumented GPU smoke
check passed, followed by exact-head model profiles and unprofiled controls.
For a 2K input using layer-major mode 1 and 1024-token chunks, explicit H2D
copies ran at 6.163 GB/s of copy execution time, but transfer intervals alone
did not account for the observed request duration. Host submission/wait calls
and expert dequantization are candidates for the next tuning measurements.
Hardware counters were disabled, and the existing per-phase synchronization
remained enabled. These profiles do not establish clean throughput or the
earlier hang's cause; the profiler itself increased that single prompt time
by 30.0%.

The [dequant launch-property preparation](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/dequant-launch-properties-20261007/README.md)
uses that profile to prepare a private comparison with only the flat/GU
dequant wrappers' `use_root_sync` declarations removed. Their kernel bodies
use no root-group synchronization; earlier API logs show cooperative launches.
CPU compile/link passed with unchanged production inputs and only one archive
member replaced. A shared host fixture was also linked against the original
and changed kernel objects for 144 guarded whole-FP16-output cases per binary.
The subsequent [actual-wrapper check](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/dequant-actual-wrappers-20261007/README.md)
passes all 144 whole-output comparisons with intact guards and finite values.
Both processes exit normally without new xe faults; API logs show 144
cooperative launches in the original and zero in the changed wrappers.
The private engine has not run a model. The older work-group probe changed
other wrapper details, so its times do not isolate this property. A stall
cause, clean speed gain and full-model correctness remain unproven.

The subsequent [expert-phase pacing checks](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/expert-phase-pacing-20261007/README.md)
reduced short-input printed phase completion waits from 73,692 to 9,665
while preserving four complete heads. A quiet 2K repeated-input check with
every phase wait retained then differed on the third request. The private
candidate was not adopted and its timings are not accepted as a speed gain.
An unchanged-binary control with full API logging also stalled in a queue
completion wait and exited through the application watchdog; the following
exact-word GPU probe passed without a reset. Four unchanged-binary repeats
with phase prints and without API logging subsequently matched all outputs.
The cause of these timing-sensitive failures remains unresolved.

The private [lazy verifier restoration check](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/verifier-lazy-restore-20261007/README.md)
defers eager capture of every verifier window until the existing run path
needs each window. With all 262,144 KV cells allocated and a 2K input,
main-cache restoration left 1,008,906,240 bytes free and MTP restoration
completed, leaving 69,369,856 bytes. Four generated IDs, all printed
logprobs and the complete finite head matched an unchanged-binary eager
verifier control at identical settings. Both exited normally without new
xe faults or a reset. This exercises restoration at the full KV allocation
size, but does not validate full context occupancy or clean throughput;
the candidate remains private.

The [completed lazy full-context attempt](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/verifier-lazy-full-20261007/README.md)
then generates four finite outputs from 262,140 input tokens and executes
through KV cell 262143. Its second prefill fails before the temporary layer
allocation checkpoint, then aborts while restoring a 64 MiB main-cache
physical segment. After main/MTP release, only 1,244,880,896 bytes are free,
less than the preceding successful layer allocation's 1,363,152,896 bytes.
The earlier return error is obscured by destructor restoration failure;
the specific retained allocation is not identified. Owned cleanup and the
logged GPU health probe pass without a reset or new xe faults. The clipped
tail, refusal and later-valid gates remain incomplete; lazy capture alone
does not satisfy the full suite.

The [2026-10-07 logging check](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/runtime-tracing-20261007/README.md)
confirmed Level Zero entry/results and UR traces before the child exited,
using driver/adapter enumeration without submitting GPU commands. The 48
existing CPU recovery test methods also passed. This validates log delivery,
not GPU execution or full-context capacity.

The [owned-debug checks](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/owned-debug-20261007/README.md)
also cover real CPU/GDB lifecycle checks, CLI last-cell trace parsing and a
small GPU execution probe after the unsuccessful two-hour full-context run.
The probe passed without a reset; full-context correctness remains pending
for that candidate.

The [v0.1.40.2 32K follow-up](../bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k/README.md)
records an original integrated-fork state mismatch and native counter-event
waits, including one xe CCS reset. A private candidate retains actual DMA
events until source reuse and uses a registered in-order copy queue without
profiling unless transfer timing is requested. The registry preserves
device-wide/default-queue waits. Every translation unit is rebuilt against
the same header and production compiler/configuration settings match.
Three fresh32K full state/head/output checks pass both with and without
MKL CNR. Disabling implicit counter conversion alone had not passed this gate.

The [no-CNR scheduling comparison](../bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k/registered-copy-no-cnr-and-prefill-scheduling/README.md)
also checks a candidate removing only14 root-sync properties from independent
prefill kernels. Three32K full state/head/output checks match the registered
control. Four subsequent fresh32K timing jobs omit debug logs, validation,
state/head dumps, transfer profiling and extra waits. On B57010GiB /5600X /
128GiB, the registered control averages406.78 prompt /16.96 decode tok/s;
the scheduling candidate averages408.52 /16.06. The prompt difference is
smaller than the control's repeat spread; decode varies within an arm. These
two repetitions per arm do not establish a useful gain or an attributable
decode regression. All ten no-CNR/state/scheduling/timing jobs finish normally
with exact controls, complete owned cleanup and no new xe fault. Both remain
private. A subsequent [default counter-conversion check](../bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k/registered-copy-default-counter-conversion/README.md)
passes another three exact32K full state/head/output comparisons without either
the counter-conversion override or MKL CNR. All exit normally without new xe
faults; captured timings are excluded. Full262,144-cell serving remains a
separate gate. No short-input timing supports these performance comparisons.

The [host-only32K profile](../bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k/host-api-profile/README.md)
uses pinned Intel PTI unitrace with host timing and Chrome call logging only.
A CPU check of its actual option derivation and callback guards verifies that
device/kernel tracing and metrics are disabled. The actual engine environment
is captured after exec. Its first logged run, quiet profile and unprofiled
control all match the full state/head/output and exit normally without new xe
faults. The quiet trace records about2.23million native event-status queries
across three workers and503,650 kernel appends in the request envelope. Native
wait, query and append intervals overlap GPU work and one another; their sums
do not measure CPU utilization or PCIe DMA duration. Boot/raw clock alignment
and its approximate phase boundaries are documented. Captured/profiled times
are excluded from clean speed comparisons.

The [attention batch128/layout1 experiment](../bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k/qsa-batch128-layout1/README.md)
keeps the same private executable, subgroup32 and ordered arithmetic. Three
fresh32K full-state/head/output checks pass before a clean ABBA comparison.
On the same B57010GiB /5600X /128GiB host, two repetitions per setting average
408.30 prompt /16.35 decode tok/s for default32/layout0 and408.36 /16.72 for
128/layout1. The prompt change is only+0.014%; it does not establish a useful
gain. Decode varies within the default arm, and a prefill setting does not
establish the cause of that variation. Every performance input is32,768 tokens;
all seven runs match output, exit normally and record no new xe fault. The
candidate remains unadopted, with full262,144-cell serving still a separate gate.

The [processing-order and GPU-residual comparison](../bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k/processing-order-and-gpu-residuals/README.md)
uses the same private executable and 32,768-token fixture on B570 10 GiB /
5600X / 128 GiB. Compact storage and attention layout 1 first pass three
full-state/head/output controls. Layer-major processing with residual rows in
RAM passes another three; keeping all batched rows on GPU while releasing
MTP decode-only weights passes three more. Each matches all 66 main-prefill
state parts, the complete first head, all 64 IDs and every logprob. The GPU
condition also verifies every released MTP expert/head payload byte against
immutable RAM before and after restoration. These captured timings are excluded.

Six fresh clean runs use default/RAM/GPU/GPU/RAM/default, with two repetitions
per condition. Prefill averages 408.28 / 338.11 / 422.59 token/s; decode averages
17.10 / 16.96 / 17.32. GPU residuals average 3.5% above default, but that pair's
prompt spread is about 11%; this does not establish a reproducible gain. RAM
residuals add uploads/downloads and measure 17.2% below default. The GPU arm
unmaps 939,524,096 physical MTP bytes in about 22.5 ms and restores 931,016,700
payload bytes from RAM in 247.955 and 248.489 ms, including mapping and copying.
Graphs are retired before unmapping and recaptured when needed; verification
is off during clean timing. The main decode cache remains backed in all arms.
This measures MTP restoration, not restoration of all VRAM or a comparison of
snapshot and RAM sources for the main cache.

All 15 requests match output and MTP acceptance/offered counts, exit normally,
complete owned cleanup and record no new xe fault. The full-context repeat,
restore, clipped-tail, refusal and later-valid gates remain open. A fixed
32,767-row GPU allocation is not validated for 262,144-cell context, and the
older second-prefill memory failure remains unresolved. The result is retained
for tuning; it is not adopted as a production default.

The [main-cache release/restoration controls](../bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k/main-cache-release-and-restore/README.md)
add a matched 64 MiB segmented-cache allocation on the same private binary.
Keeping it backed, releasing half and releasing all with immutable-RAM
restoration each pass a logged 32K request and two fresh full-state/head
controls. All nine match the 66 main-state parts, complete first head, all
64 IDs/logprobs and MTP counts, exit normally and record no new xe fault.
The half condition unmaps 201,326,592 physical bytes and verifies all
130,731,008 occupied tail bytes before and after restoration. Full release
unmaps 402,653,184 bytes and verifies all 281,651,200 occupied bytes. The
reserved addresses and slot metadata remain stable; graphs are retired
before unmapping and recaptured after remapping/copying/verification.
These are fresh processes: retirement is called before the first graph use.
The later [same-process full-RAM checks](../bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k/same-process-main-cache-repeat-32k/README.md)
exercise already-used graphs across three complete 32K rereads per process.

CPU checks extract the actual cache-lease struct and pass 19 ASan/UBSan cases
covering a mid-expert boundary, adaptive residency and rollback/retry paths.
The same cases also pass with byte-varying payloads; a pinned wrong-offset
control is rejected by the pre-unmap comparison. These stand-ins do not
prove SYCL mapping or absence of runtime UB. Snapshot restoration also passes
three complete state/head/output controls. All 12 captured checks precede
eight fresh clean 32K timings, ordered kept/half-RAM/full-RAM/full-snapshot
and reverse. No payload checks, diagnostics, validation, dumps, profiler,
transfer timing or extra waits are enabled. Two-run prompt/decode means are
443.28 / 16.89, 443.76 / 17.70, 433.31 / 17.36 and 442.24 / 17.83 token/s.
Half release is only 0.11% above kept backing; full RAM is 2.25% below and
snapshot is 0.23% below. This fixed-chunk comparison establishes no useful
prompt gain from release alone. Decode varies within an arm, so these
prefill settings do not establish its cause.

Clean full-RAM restoration takes 51.588 and 52.091 ms excluding its graph
callback; this interval includes remapping, copying, zero fill and waiting,
not only PCIe DMA. Graph callbacks range from 537.714 to 1004.037 ms in the
release arms. Full-snapshot suspension takes 220.314 / 221.539 ms, while
restore excluding graphs takes 61.198 / 60.502 ms. These measured intervals
must not be substituted for full-VRAM bandwidth. All 20 jobs retain exact
output and MTP counts, exit normally and record no new xe fault. Larger
chunks using freed memory remain a separate comparison. The fixed 32K row
allocation and the older full-context repeat failure remain open gates.

The [larger compact-chunk control](../bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k/larger-compact-chunk-12k/README.md)
keeps input at 32,768 tokens and increases the chunk to 12,288 on the same
private executable. Its measured accounted workspace is 2,738,729,216 B;
reusing the first chunk's residual scratch leaves 838,819,840 B of new GPU
rows. The request fits, receives 64 outputs, exits normally and records no
new xe fault, but fails the mathematical gate: 63 of 66 main-state parts and
all 248,320 head floats differ. The first generated ID differs at index 2;
MTP counts are 38/75 versus the accepted reference's 43/66. No clean timing
follows. Chunk/GEMM/callback boundaries have not been isolated as the cause.
The subsequent exact-word health probe passes without recovery actions.

Source and native API logs also clarify the fresh-process comparison above:
the serve path does not warm verifier graphs at startup. Release arms capture
all legal window sizes at restoration, while kept backing captures on first
use. The callback interval includes initial capture/kernel loading, and the
means include these different capture schedules. In the 12K diagnostic,
reported free VRAM drops by 921.152 MiB between prefill entry and restored
weights. Main restore/capture requests 384 MiB of physical mapping and only
1,114,112 B of new device USM, while reported free drops by 1,221.324 MiB;
native logs show 680 new command lists and 61 modules. All logged device USM
and native modules/kernels/command lists are destroyed at process exit.
This does not establish a leak, an isolated allocation cause or a solution
to the older full-context repeat failure.

The same-process full-RAM checks pass all nine 32,768-token requests across
three fresh processes. Every raw 66-part main state, complete first head,
64 IDs/logprobs and MTP counts43/66 matches the hard control. Every request
reports resume0; each new dump is renamed and must be absent before the next
request. All processes exit0 normally without forced cleanup, survivors or
new xe faults. Captures, payload checks and phase tracing remain enabled,
so these durations are excluded from performance comparisons.

Successful UR calls release680 main command buffers and6 MTP buffers before
the second and third physical unmaps; main restoration creates/finalizes680
buffers each time. Native LevelZero command-list counts remain704 across
those releases, while all UR references and tracked native resources reach
zero at normal exit. Thus native object counts alone do not establish that
old executable graphs remain live. Source checks separately cover queue
drain, stable-address remapping and recapture after payload restoration.

The two processes without API logs have identical reported-free markers,
within64 KiB of the logged process at matching later phases. Their prefill
entry drops from2153.496 MiB on read1 to1131.254 MiB on read2 and1120.297 MiB
on read3. Main warm initially creates61 modules and680 regular command lists;
later main warm creates no new native module/kernel/list/device-USM objects,
but384 MiB remapping accompanies reported-free drops of394.707 and389.449 MiB.
These counts and requested sizes do not attribute resident runtime heaps,
prove a leak or establish steady state after more requests. Only the captured
full-RAM32K repeat gate is closed. Matched quiet repeated performance and
full262,144-cell occupancy/repeat/restore/clipped-tail/refusal/later-valid
gates remain open. All performance comparisons use at least32,768 input
tokens; first-use loading/capture and later full rereads must be distinguished.

No software reset is guaranteed to recover every firmware/driver wedge. There
is an [upstream B570 report](https://github.com/intel/compute-runtime/issues/962)
where both rebind and PCI reset failed; that report is not proof of this host's
cause. The recovery flow follows the kernel's
[DRM recovery prerequisites](https://docs.kernel.org/gpu/drm-uapi.html#device-wedging)
and [xe recovery guidance](https://docs.kernel.org/gpu/xe/xe_device.html).

## Firmware audit on 2026-10-06

The actual PCI IDs are `8086:e20c`, subsystem `172f:0102` (SPARKLE). All seven
display connectors were disconnected; `boot_vga=0`. The advertised reset
methods were `flr bus`. Firmware was queried without flashing or changing
system metadata, using fwupd's igsc plugin:

| Part | Installed version | Latest matching stable LVFS release |
| --- | --- | --- |
| FWCODE | 21.1182 | 21.1182 |
| OptionROM Code (VBIOS code) | 23.1066.0.0 | 23.1066.0.0 |
| FW Data | 203.1 | No matching public release found |
| OptionROM Data | 23.1051.0.0 | No matching public release found |

The [recorded firmware audit](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/firmware-audit-20261006.json) contains the device GUID matches.
The live stable LVFS metadata was fetched at 2026-10-06 06:34:42 UTC, last modified
October 5. Its SHA-256 is
`3e84ee0f9a44cd8bf6867d07ff4f9544536ed209a9d43bae3dce1686bcaf4c82`.
The FWCODE and OptionROM releases match the device GUIDs reported for this card.
No newer applicable public stable firmware/VBIOS was found. A generic B570
product name does not justify using another manufacturer's OptionROM Data.
[SPARKLE's download center](https://www.sparkle.com.tw/en/Download) does not list
a separate B570 VBIOS update. The physical board SKU is not established from the
subsystem ID alone. OEM-only updates are not ruled out.

Intel's older [Linux firmware article](https://www.intel.com/content/www/us/en/support/articles/000096950/graphics.html)
says Linux driver packages do not update card firmware; that does not mean the
current fwupd/igsc firmware path is unavailable. This host actually exposes all
four parts to fwupd. The GuC firmware loaded by xe (`70.44.1`, recommended
`70.54.0`) is a separate host firmware file, not the card's VBIOS.

For future read-only checks:

```sh
fwupdmgr get-devices
fwupdmgr get-releases 310f45f1f223064b5c16bf6dff31146755a64480
fwupdmgr get-releases 66ce30cd03cc094ee6b0a2cc78519376f1de4707
```

Do not infer a new firmware release from its CAB filename: the current file is
named `intel-arc-bmg-21.1180.cab` but its release metadata says FWCODE 21.1182.

## Scope of validation

The [original validation record](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/recovery-validation-20261006.json)
records the earlier 14-test version. The
[GUI handoff validation](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/display-recovery-20261006/record.json)
records the earlier 39-test implementation. The
[incident and revised validation](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/recovery-incident-20261006/record.json)
records 48 passing CPU test methods, shell syntax and the revised installed
digests. These cover reset ordering, method restoration, child timeouts,
misleading health results, no-argument shell sequencing, explicit consent,
changed/hidden owners, display ownership, borrowed locks, stranded writers,
service-stop failures, interrupted-worker receipts, manager umask changes,
same-boot restoration, runtime dependencies and disabled GUI-shutdown entry points:

```sh
python3 sycl/tools/test_recover_xe.py
python3 sycl/tools/test_recover_xe_display.py
```

The service/reset backends are harmless fakes. Separately, a real temporary user
service proved that its launcher exits before the manager-owned worker finishes
and its `ExecStopPost` runs. The confirmation itself is also tested with a real
private controlling PTY: its prompt appears before input, only `yes` proceeds,
and no/Enter or a detached process without a terminal cancels. This caught and
fixed an earlier `r+` open that requires seeking and fails on real terminals.
Those tests used no root service, GPU or GUI operation;
it does not validate the root SDDM recovery transaction. The probe builds with
oneAPI 2026.1.1.
Read-only inspection ran on this host. A root recovery attempt was refused
before reset because Xorg held the B570 open in an active local X11 session.
The revised entry refuses that case without stopping the GUI. No display
shutdown or device reset was performed by the agent during these tests.
Successful no-reboot hardware recovery remains pending.
The full Strata arithmetic/context/cancellation/graph checks remain separate
from this 64 KiB GPU health check.

## GUI recovery incident on 2026-10-06

The human ran the earlier entry from the GUI at 18:57 JST and reported Orca and
Codex hanging, then rebooted the PC. The previous-boot service journal records
SDDM stopping at 18:57:59 and the recovery worker and its backup both exiting at
18:58:03 with `Unsafe recovery plan ownership/permissions`. The plan-save code
depended on the launcher's 0077 umask. A manager worker using 0022 instead writes
a 0644 replacement, which its own reader refuses; the backup cannot restore
SDDM either. Actual old save/read definitions reproduce the same exception with
CPU temporary files. The changed definitions preserve 0600 across replacements.
The privileged original plan has not been read, so its exact mode is inferred
from this reproduction and the journal, not directly observed.

The journal also records xe reinitialization during both attempts. The saved
kernel interval contains no kernel-panic/lockup signatures; that does not disprove
the reported whole-PC hang. Ending the GUI session and failing to restore it are
established defects. This is why normal recovery no longer stops the GUI.

Orca is actually a user `orca-headless.service` running with Xvfb; it is not
running on SDDM's display. Current unit dependencies have no `PartOf`/`BindsTo`
on the graphical session, and user lingering is enabled. The previous journal
records GNOME restarting the user D-Bus at 18:57:59, followed in the same second
by the Orca application scope exiting and its headless service killing remaining
Xvfb/crashpad processes. This contradicts an explanation based solely on Orca
using the display. The exact headless-service shutdown mechanism is being
investigated separately, without stopping that service or D-Bus.

After the human reboot, the first small probe exited with no SYCL GPU available.
Read-only Level Zero enumeration still found one device. The UR loader identified
the missing dependency `libumf.so.1`: the script's sanitized library search path
omitted `/opt/intel/oneapi/umf/1.1/lib`. Including that path let the unchanged
probe pass all three integer rounds at 20:25:31 JST, with no new xe fault/reset
messages. That check performed no reset or service operation. This establishes
small-probe health after reboot, not recovery without reboot or full-model health.

The verifier watchdog registry also had a CPU lifetime race: a callback could
load an atomic raw pointer, then use the verifier after another thread removed
and freed it. Registration now occurs after initialization, and a mutex covers
each diagnostic/release callback and removal before mapped flags are freed.
This follows the C++ rules for
[object destruction](https://eel.is/c++draft/class.cdtor) and
[thread synchronization](https://eel.is/c++draft/intro.races); an atomic pointer
does not provide ownership of the pointed-to object.
The [CPU regression record](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/verifier-registry/record.json)
extracts the actual registration/callback/removal code, substituting CPU payloads
for the verifier's mapped flags and queues. The pinned old control reproduces
ASan heap-use-after-free for both diagnostic and release callbacks. Five
candidate ASan/UBSan cases pass those races, failed initialization/retry,
16-stage capacity/gap reuse and 2,048 ownership cycles across four owner threads
with concurrent diagnostics/releases. Run it without a GPU:

```sh
python3 sycl/tools/test_verifier_registry_host.py --output /tmp/strata-verifier-registry-check
```

Both reference and CPU task-factor-9 engines compile/link with the registry
change. Their measured production CPU archives remain byte-identical to the
earlier exact-output/timing evidence. The
[link environment audit](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/verifier-registry/link-environment-audit.json)
preserves an initial failed identity check: configuring pinned ggml without
`ONEAPI_ROOT` adds `-lm` and changes the executable. Explicit oneAPI/MKL paths
make both builds use the prior recipe; restoring factor 0 reproduces the initial
reference hash. Neither new engine ran on the GPU. This regression proves a
possible CPU use-after-free, not the trigger of the original GPU fault or a
speed improvement. Other source-review and full-context gates remain pending.
