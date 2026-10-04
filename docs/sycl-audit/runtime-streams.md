# Runtime stream and event ordering

The new `strata::sycl_upstream::Runtime` is a per-device execution domain with
legacy-default, ordinary blocking and nonblocking streams. It implements queue
ordering, event recording/waiting, copies, byte fills, host functions, completion
queries, synchronization and asynchronous stream retirement. It is a component
core, not a complete `cuda_runtime.h` replacement; the original engine and MMQ
adapter have not yet been switched to its stream handles.

## Original processing inventory

`python3 tools/sycl/audit_runtime.py --output docs/sycl-audit/runtime-inventory.json`
scans the pinned upstream Git tree, not the new implementation. The inventory
records 61 function-like CUDA identifiers and 2,764 candidate occurrences in 108
files, with file hashes and line locations. It masks comments and quoted strings
but does not preprocess or construct a call graph. Counts include declarations,
compatibility code, harnesses and inactive branches; they are not runtime counts
or evidence that all paths are covered. Kernel launch syntax is outside the scan.

Call sites that drive this component's requirements include:

| Source | Required behavior |
| --- | --- |
| `src/program/generate.cpp` stream creation | Generation and adaptation use nonblocking streams. |
| `src/core/session.cpp` capture setup | Temporary capture streams also use ordinary stream creation. They cannot be treated as nonblocking by default. |
| `src/prefill/prefill.cpp` `copied`/`used` ring events | A slot's next copy waits for its previous consumer; a consumer waits for the corresponding copy. Reusing an event object must preserve already-issued waits. |
| `src/core/verify.cpp` `fetch_dma` | A host function publishes the flag after asynchronous expert transfers complete. |
| `src/core/graph.cpp` and session/verify/MTP capture | Capture/replay and reusable completion events are covered by the subsequent [graph component](runtime-graphs.md). |

There is no `--default-stream per-thread` option or
`CUDA_API_PER_THREAD_DEFAULT_STREAM` definition in the pinned source/build scan.
The baseline therefore needs legacy-default semantics. CUDA's [stream ordering
specification](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)
distinguishes blocking streams, which participate in default-stream ordering,
from explicitly nonblocking streams. CUDA's [runtime stream API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__STREAM.html)
also specifies asynchronous destruction. [Event recording semantics](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html)
preserve the recorded state used by an already-issued wait when an event is later
recorded again. These are source/specification references; no CUDA device was
executed for this validation.

## Implementation

All streams share one SYCL context and have in-order queues. Stream handle zero
is the legacy stream. Its submissions depend on the preceding tails of blocking
streams, while a blocking stream submission depends on the preceding legacy
tail. Nonblocking streams acquire no such implicit edges. Host submissions are
serialized only while recording these dependencies; the mutex is never held for
an explicit completion wait. Nonblocking work without an explicit event wait
needs no dependency marker.

An event records a native SYCL event value. A stream wait copies that value when
issued. Re-recording or releasing the event object cannot replace the dependency
already queued. An unrecorded event is complete. Explicit synchronization takes
a snapshot before releasing the host mutex and waiting. Device synchronization
includes queues retired by stream destruction. Retired queues are reclaimed at
a later stream-creation boundary once their tail is complete, or at domain
teardown; stream destruction itself performs no host wait.

Submissions use a callback returning the last SYCL event of the queued operation.
It must not retain the queue, submit outside the domain, or reenter the domain.
If a callback throws after partially submitting work, a marker retains that work
in the dependency chain before the exception propagates. Pointers remain
caller-owned and must stay valid through their queued users. Cross-context event
waits are explicitly unimplemented rather than converted to host waits.

## A host-blocking barrier found during validation

The first implementation used `ext_oneapi_submit_barrier(dependencies)`. On B570,
with one queue held by a host task and a kernel queued after that task, submitting
a dependent barrier in the legacy queue did not return during a two-second
observation. Releasing the task allowed the submission to return. The diagnostic
identified the first legacy dependency as the blocking phase, before the
independent stream could even be submitted. This is a submitting-host stall,
not evidence that an already-submitted independent GPU kernel could not execute.

The [SYCL barrier extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc)
describes enqueued barriers as nonblocking to the host. The exact driver/runtime
instruction responsible for this observed violation has not been identified.
The port now represents a dependency marker as an empty GPU command with
`handler::depends_on`. With this change, the same test returns before releasing
the held task and independent work completes. Event markers use the same path.
The small kernel's cost has not been benchmarked; this avoids silently accepting
a host wait as faithful asynchronous behavior.

## Evidence and limits

Intel Arc B570, Linux, oneAPI 2026.1.1, Level Zero, 2026-10-04, JIT and `bmg_g21` AOT:

- Seven runtime scenarios pass: legacy→blocking ordering, both directions of
  nonblocking exclusion, ordinary streams remaining independent, event generation
  snapshots and event-object release, asynchronous destruction with pending work,
  the prefill ring/host callback pattern, and partial submission followed by an
  exception. The first scenario also checks blocking→legacy ordering.
- The two-slot transfer ring checks every batch's sum across 40 distinct pinned
  host inputs and device slot reuses. It queues all batches without intermediate
  host synchronization.
- Held host tasks make ordering/independence observable. Test gates are released
  before cleanup, including failures of bounded submission observations.
- The retired-stream test confirms both immediate host return from destruction
  and that device synchronization still waits for its pending work. A recorded
  event also remains usable after that stream handle is invalidated.
- Copies, host callback and later device work are checked in queue order.
  The host callback test does not establish a running GPU spin loop observing a
  host atomic flag; that is a separate mapped-memory visibility requirement.

[Runtime results](runtime-streams-results.json) contain source/binary hashes,
JIT/AOT logs and the original barrier-stall diagnostic. Test elapsed time is not
an inference benchmark. All 256 files in the original shared-source manifest
remain byte-identical.

[Native capture/replay](runtime-graphs.md) extends this component. Graph update,
full introspection, timing events, per-thread
implicit streams, device selection, multi-device/cross-context waits, allocation
and mapped-host registration, CUDA-compatible error codes and thread-local error
state remain open. The new runtime uses SYCL exceptions and explicit USM pointers;
asynchronous failure injection and teardown recovery have not been established.
It is not yet a complete runtime parity verdict or an end-to-end speed result.
