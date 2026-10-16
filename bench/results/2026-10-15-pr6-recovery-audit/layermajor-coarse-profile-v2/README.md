# Layer-major coarse profile v2: source-only proposal

Base: immutable Git commit `9177ad4fb7cc1587255db7235be474aca3eb4e2a`.
Unapplied, uncompiled, untested. Main owns complete review, builds, all tests and GPU work.
Only this proposal directory was written. No live product source, shared index,
branch, worktree, process, service, environment or measurement artifact was changed.
No XeStrata source was read or used. The implementation is new orchestration
instrumentation against the pinned fork's own source and its compatibility API.
R726 incorrectly calls the SHA256 of `layer_major.inc` a Git HEAD; this proposal
uses the actual immutable commit supplied by Main, not that statement.

`source.diff` changes three existing files and adds one header. The header is also
provided separately. `author.py` with `native_dense_edits.py` reconstructs the exact diff from Git blobs held
in memory; it never writes full existing-source mirrors or live product files.
`pins.json` identifies before/after source hashes and the diff. Main must review
and merge the patch against any intervening edits, not replace live files from
this proposal. Canonical dense bank v3 is neither included nor integrated.

Enable only in the native child's immutable pre-exec environment with
`STRATA_PREFILL_LAYER_MAJOR_PROFILE=1` alongside all existing admitted route
settings. No setting is set or switched by this code. Short requests (`n <= T`), including the official 74-token warmup, and default
route requests retain their original inference/dispatch with no collection or
skip log. V1's short-route rejection is removed entirely in this separate v2 proposal. With the
flag absent/other than `1`, existing route selection, math, queues, callbacks,
owner retirement, statistics and output remain unchanged; no profile allocation,
clock calls, events or logs execute. New pointer checks and environment reads are
present. `STRATA_PREFILL_TIMING` remains rejected; PfTimer is never enabled or
modified, and there are no per-expert or per-kernel marks.

## Measurements and labels

One header, 48 layer rows and an aggregate (`layer=-1`) are emitted per admitted
request, including an incomplete record after safely retired early exits.
Each row uses the orders named in the header; source requested byte totals are
integer fields. A successful wrapper wall denominator runs from the existing
`started` point through checked lease retirement. It includes allocation,
observation, source work, GPU work and caller callbacks. It excludes early
admission/validation and final restoration/destruction; those tails are explicitly
unmeasured. Incomplete wrappers have `wrapper_ms=-1`, never a claimed denominator.

Host wall fields cover source `copy_blob` preparation, upload submission, its
existing drain, the existing native dense activation callback including its
existing preactivation drain, residual input submission by source class, residual
store through its existing drain by destination class, callback chunk drains,
and wrapper-owned drains. Activation is zero when no callback exists; it applies
to the current native overlay hook and does not claim to profile an unapplied
canonical bank. Input has no dedicated baseline drain, so only submit wall and
queue interval are reported; the existing chunk drain covers more than input.
Wrapper drain totals omit the initial entry drain before a layer is selected and
include final drains in layer 47. These wall fields overlap: activation/store/
upload drains also occur in wrapper-drain totals. They are not additive exclusive
fractions.

Six coarse queue intervals use command-end timestamps between ordered barrier
marks: input, QSA block, GDN block, MoE block, expert upload, residual store. QSA starts at
its existing block and ends after output projection; MoE starts at its existing
block and ends after combine and any existing phase observation. GDN begins immediately before the existing `pt.mark(kPfGdn, cs)` after
weight validation, before the QKV/gate/alpha/beta projections, and ends
immediately after `ssm_out.weight` output projection. It includes gates,
convolution and recurrence, excludes the following optional output audit, and
adds no per-token or per-kernel marks. Surrounding
hyperconnection norm/write, PLE and other setup are not included in those
block intervals. Input/store also have host/device class interval totals. FP16
input/store includes unpack/pack work. They are **NONEXCLUSIVE queue timeline
intervals including host submission gaps**, not isolated kernel durations,
exclusive critical-path fractions, verified copy endpoint placement or PCIe wire
time/traffic. Requested bytes describe source intent; physical traffic is not
measured. Profiling adds a few ordered barriers per chunk and affects scheduling,
so this is diagnostic evidence, not a clean speed result or performance adoption.

## Bounded owners and checked retirement

There are exactly twelve event owners (six pairs), 48 fixed rows and no trace
vectors. Input/QSA/GDN/MoE pairs are consumed and reused only after the existing
checked callback stream drain in `run_impl`; the layer-major wrapper always
installs that callback. Upload/store pairs are consumed only after their existing
checked wrapper drains. There are at most eight marks per layer/chunk plus two
per layer upload, never marks per expert. At C8192 and the physical 262144-token
maximum, there are 32 chunks; actual markers are bounded by
`2*48 + 2*(47+12+36+48+47)*32 = 12256`. Smaller chunks retain the same constant event
owner count and few marks per chunk. Layer-major output stays 50 lines regardless of chunks. NativeDensePhase adds
one bounded host-only line when it is selected, for at most 51 diagnostic lines
per current single-invocation request; short warmup adds none.
All floating output uses bounded significant-digit formatting; conservative
1024-byte/layer-row and 1024-byte/header bounds plus the generator line give 52224 bytes, below 64 KiB.
No log truncation is used. Profile allocation admission reserves a further 1 MiB
alongside existing observer reservations and the unchanged 8 GiB RAM margin;
this is a bounded host-owner allowance, not a bound on driver internal allocation.

Own-source compatibility audit at the immutable pin:

- `cudaEventCreate` calls the checked allocation wrapper and defaults timing on.
- `cudaEventRecord` submits one barrier on the chosen queue. All calls here use
  the already-admitted `m.cs`, whose runtime compute queue has profiling enabled;
  the unchanged route rejects `m.copy != m.cs`.
- `cudaEventQuery` checks completion and calls the runtime's wait helper only
  when completed. `cudaEventElapsedTime` internally invokes the same wait helper
  before reading command-end timestamps. **Query is called only after the existing stream drain succeeded;
  ElapsedTime is called only after both queries confirmed completion**. There is no
  explicit event synchronize, stream-wait-event or extra stream synchronize.
  Main should specifically review this already-retired API behavior against the
  no-new-wait invariant; this proposal does not introduce a new pending-event
  dependency or a new queue drain.
- `cudaEventDestroy` deletes the event owner and returns success; it does not
  submit work, wait, or release model tensors. Status is nevertheless checked.

Create/record/query/elapsed/destroy results are checked. The first non-success
status and error count are reported; internal observer invariant/numeric errors
can set incomplete without a CUDA status. An observer error disables further
marks/timestamp reads but retains all event handles. It does not return through
borrowed owners or change engine error status. Profile owners are declared before
Restore and the checked lease: early return/exception drains with all owners
alive, restores borrowed model state, restores the profile pointer, then destroys
events and emits incomplete output. If the existing checked drain cannot prove
retirement, the original fatal guard aborts with owners alive; no profile is
claimed. Successful event destruction/output occurs only after the existing
checked final retirement. Early construction/admission failure may have only the
engine error and no profile object/output.

## NativeDensePhase host-only boundaries

This v2 also instruments only `run_prompt_with_native_dense_phase` in
`src/program/generate.cpp`. Its fast path remains unchanged: disabled NativeDensePhase
or short requests return through the original `run()` without a timing object.
The same exact new PROFILE=1 flag enables a local host-only observer inside the
SYCL NativeDensePhase path. With the flag off, observer methods perform no clock
reads, logging or new CUDA calls. There are no generator event operations, new
GPU calls or new synchronizations even when enabled.

`entry_ms` starts at this boundary after its existing fast guard and includes
eligibility/transaction checks, the existing entry drain, verifier/session graph
retirement, native PLE withdrawal and `enter_layer_bank`, ending immediately after
successful Bank entry. Any later native/canonical preparation must remain inside
these seams when Main integrates it; canonical v3 is not included here. Snapshot
preparation performed before this function, initialization, process startup and
outer request work are unmeasured. Bank entry is distinct from the 48 per-layer
activation callback walls in the layer-major record.

`run_ms` brackets the original `run()` invocation only. That invocation can include
MTP expert release, prompt prefill and callbacks, destruction of prompt-local
owners and MTP expert restoration. It is not pure prefill. The native entry and
restore intervals are separate from the layer-major denominator; the native run
interval contains the layer-major interval. They are labeled NONEXCLUSIVE and
must not be added as independent fractions or called device service time.

`restore_ms` starts just before the existing restore drain, after the original
`if(!retired) return` guard. It ends after the original callback removal,
full-owner restoration, native PLE rebind, graph recapture, verifier resume,
retired-flag clear and existing restored TRACE observer. It includes that
observer's existing telemetry when TRACE is enabled. Original normal, exception
and rollback guard restore calls stay in their original order. The profiling
object is declared before the rollback guard, so incomplete return/exception
summary output occurs only after that guard has restored owners. None of the
TRACE observer's original source is modified; pins record its unchanged hash.
Generator text outside this function is byte-identical, including GpuStage and
MTP helpers. Raw prompt/source-prefix ownership paths remain untouched.

The generator emits at most one bounded line per invocation. `complete=1` means
entry reached Bank, the run call returned and restoration completed; it is not
an inference correctness result. `outcome=success` or `run_false` distinguishes
the run result even when timings are complete. Entry return failure gives
`entry_incomplete`, `complete=0`; missing intervals are -1. Exceptions inside the
existing try are labeled `exception`, with elapsed run-through-throw when that
call was active; they remain incomplete and the original exception is rethrown
after the original restore. Exceptions before that try have incomplete entry
status. Existing fatal exits emit `complete=0 outcome=fatal` before the original
fatal message/abort, retaining original owner fail-stop behavior. An in-progress
restore does not acquire a successful duration. No profile error is converted
into inference success, and no existing inference failure/cancellation status is
changed. Fatal process termination outside this boundary may prevent its summary.

V1 remains immutable and is superseded by this proposal; v2 is the
complete replacement diff against the original 9177 base, not an incremental v1
patch. Full source diff and header were reviewed as text; no compilation,
execution validation, benchmarks, fault injections or GPU operations were run.

## Main qualification required

Review the complete patch, current source resolver and compatibility pins;
confirm no conflicts with newer Main changes. Perform host compilation and
meaningful observer OFF/ON, short/default-route warmup preservation, event-failure/incomplete,
lifecycle and log-bound checks before a serialized GPU diagnostic. Validate the
existing request's correctness/fault receipts and pin source, binary, input,
environment, runtime/driver and individual result under artifact retention.
Keep prefill separate from decode. Neither selected-row parity, full-context
correctness, RESTORE, quality, speed nor bottleneck identification was measured
by this source proposal. It enables the bounded evidence collection for Main's
next request; its timings alone cannot establish exclusive service or adoption.
