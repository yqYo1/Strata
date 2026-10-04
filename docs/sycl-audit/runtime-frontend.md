# Original host code connected to SYCL

The new CUDA source frontend compiles and executes the original
`src/core/graph.cpp`, `src/core/pinned.cu` and `src/platform/memory.cpp` without
changing their processing. This is a Linux, single-device component integration,
not a complete CUDA runtime, a CUDA binary ABI, or a full-engine build. The original
shared-arena self-test also compiles and passes unchanged.

The frontend is selected only by targets that add
`include/strata/sycl_upstream/cuda` to their include path. It forwards to the new
Runtime and Memory components, rather than the previous SYCL port. Ordinary CUDA
and HIP targets keep their existing headers. `pinned.cu` contains host code and is
compiled as C++ for this target.

## Ownership and API processing

| Area | Binding and checked behavior |
| --- | --- |
| Allocations | Device and mapped-host ownership, native host registration, interior aliases, matching free/unregister |
| Streams | Null legacy stream, blocking/nonblocking creation, asynchronous destruction, query/synchronize and event waits |
| Events | Record/query/synchronize and destruction without waiting; stale handles rejected; [timed records and elapsed time](runtime-timing.md) now also supported |
| Graphs | ThreadLocal capture, separate definition/executable handles, node count, instantiation, replay and asynchronous release |
| Errors | Per-thread last error; successful calls preserve it; peek preserves it; get consumes it; NotReady does not set it |
| Copies | H2D/D2H/D2D/H2H and default direction inference; 1D and pitched 2D; byte-valued memset |
| Host functions | Enqueued after preceding stream work through the Runtime host-task implementation |

Handles are monotonic opaque tokens looked up in a registry. They are not raw
`sycl::queue*` values. Kernel adapters use `cuda::submit`, which resolves the handle
and submits through the Runtime dependency/capture path. The [engine-facing MMQ adapter](mmq-frontend.md) now resolves its public entry
points through these handles; its earlier raw-queue variant remains for component
tests. Full prefill binding remains open.

A graph definition now retains a completed recording separately from executable
state. Each instantiation finalizes its own SYCL executable and tracks its own
last launch. Two instances of the same definition can run independently; repeated
launches of one instance retain the existing serialization rule. Destroying the
definition does not invalidate either executable. Destroying an executable returns
while its pending launch is retained by Runtime. Original comments claiming that
CUDA executable destruction always blocks remain unchanged; the prior
[graph audit](runtime-graphs.md) explains that discrepancy with CUDA's contract.

Empty definitions expose a zero node count, so the original `CapturedGraph::end`
executes its own zero-node rejection. The lower-level Runtime convenience method
continues to reject empty captures. Capture status now distinguishes active and
invalidated recordings. The original `GraphRegistry` body-error handling consumes
the thread's error, abandons recording, and does not cache a failed graph.

The public error strings are stable descriptions. A separate
`cuda::backend_error_detail()` retains the last caught backend exception text.
Argument/handle checks, bad allocation and several SYCL errors are mapped, but
this is not an exhaustive CUDA error-code or asynchronous-fault recovery mapping.
The non-error treatment of NotReady follows the
[CUDA error handling contract](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf),
with last-error consume/peek behavior defined by the
[runtime reference](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__ERROR.html).

## Copy scheduling and the 2D graph limitation

Following CUDA's [API synchronization rules](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html),
the synchronous spelling of D2D copies and device memset does not add a host wait.
Pinned H2D and D2H copies wait for completion. H2H copies are host-synchronous,
including the Async spelling. Synchronous pageable H2D first synchronizes the
legacy stream, stages a CPU snapshot into owned host USM, and queues DMA while
retaining that staging allocation until completion. A later pageable copy reclaims
completed staging allocations; the remaining staging is released at domain exit
after synchronization. Async copies submit through the selected stream. Pageable
async transfers retain the backend's possible staging/synchronization behavior;
no universal asynchronous-return claim is made for them.

`cudaMemcpy2DAsync` uses `ext_oneapi_memcpy2d` outside capture. Direct use of that
SYCL extension during graph recording fails with an explicit unsupported-feature
exception on the installed oneAPI 2026.1.1 runtime. Its documented
[native graph-command alternative](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_codeplay_enqueue_native_command.asciidoc)
was also tried. Finalization hangs inside the UR Level Zero V2 adapter while its
native-command callback requests the command-buffer handle. A captured backtrace
shows `urCommandBufferAppendNativeCommandExp` calling the callback and
`urCommandBufferGetNativeHandleExp` waiting for a mutex on the same call stack.
This is evidence of the observed lock stall, not a claim about all UR versions.

The implemented recording path expands the rectangle into ordered 1D copy
commands, retaining copy operations, source/destination pitch and untouched padding.
The complete graph remains recorded and replayed by SYCL; no host work splits its
replay. **A rectangle with seven rows uses seven copy nodes.** This differs from a
single CUDA rectangular-copy node and its performance cost has not been isolated.
The direct path still uses the native 2D SYCL API. The frontend rejects invalid
width/pitch and overflowing extents instead of truncating sizes.

`graph_copy2d_probe` is an opt-in reproducer. Its default mode exits 1 with the
SYCL recording rejection. Its `native` mode prints that native handle lookup has
started, then reaches the external five-second timeout in the tested V2 adapter.
The earlier integrated native attempt reached its CTest timeout too. Both failure
paths, the stack trace and adapter hashes are preserved in the results file.

## Connected validation

The measured system is the same B570/Ryzen 5600X machine as the preceding audit:
oneAPI 2026.1.1, isolated compute-runtime 26.35.39758.10, IGC 2.41.5 and GMM 22.10.0,
with system driver packages unchanged. External registration needs that fixed
runtime; see the [memory audit](runtime-memory.md).

Both JIT and `bmg_g21` AOT pass seven connected scenarios with **47,130 exact value
checks** in `cuda_frontend_test`:

1. Error preservation/consumption and host-thread isolation; pending query does
   not poison last error; stale event rejection; original registry body-error and
   zero-node paths; allocation invalidates capture and its status reports that.
2. Original GraphRegistry captures four `(layer type, token count)` keys exactly
   once and performs 32 replays with changed inputs, checking all 128 outputs each
   time. Original CapturedGraph move ownership is also exercised.
3. Two executables instantiated from one definition are independent: one remains
   behind a held host function while the other increments a device atomic counter
   and completes. Destroying the definition and pending executable preserves the
   work; the final counter is exactly two.
4. Held host functions distinguish asynchronous D2D/device memset from synchronous
   pinned H2D and H2H. Eight pageable H2D copies preserve the source snapshot after
   the original CPU vector is overwritten.
5. Pitched H2D, D2D and D2H use different source/intermediate/destination strides.
   Two direct runs and four graph replays check copied bytes and untouched padding.
   Three seven-row copies plus three initializations record 24 graph nodes.
6. Original PinnedArena registers two adjacent unaligned slices up to its cap,
   shares their native edge page, and releases all imports during teardown.
   Original `load_experts_ranges` reads three ranges with three workers and a
   257-byte chunk; bytes, per-layer FNV checksums, an interior device alias and a
   GPU round trip match the fixture. A truncated fixture triggers the original
   short-read refusal. Whole-arena registration is also checked.
7. Original CapturedGraph runs two connected eight-stage publish/wait/reply-copy
   recordings. Four replays perform 32 CPU/GPU handshakes with changing data and
   advancing sequence numbers. Two bounded runs prove an active GPU wait; two
   runs use the upstream unbounded wait. CPU service completes before any query
   or synchronization by the servicing caller. No bounded timeout is accepted.

The unchanged `src/core/pinned_shared_test.cpp` passes separately. Its assertions
cover CPU-visible shared-file contents and pack/size mismatch refusal; it is not
by itself proof that GPU registration succeeded. The connected frontend test
explicitly checks registered bytes, backing ranges and GPU results.

Six CTest programs pass in each build: both original-host tests and the existing
stream, graph, handoff and memory suites. These regressions cover the graph
representation change; unchanged standalone MMQ arithmetic suites were not rerun.
JIT disables persistent binary caching. All 256 original manifest files still
match their upstream hashes. Test elapsed time is not inference throughput.
Full logs and source, original-object, binary and native-adapter hashes are in
[runtime-frontend-results.json](runtime-frontend-results.json).

```sh
# First select the isolated driver libraries as in runtime-memory.md.
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
cmake --build build-upstream-sycl --target cuda_frontend_test original_pinned_shared_test
SYCL_CACHE_PERSISTENT=0 ctest --test-dir build-upstream-sycl -R 'runtime_|original_' -V
# Diagnostics only; the second command is expected to time out on this adapter.
cmake --build build-upstream-sycl --target graph_copy2d_probe
build-upstream-sycl/graph_copy2d_probe
timeout 5 build-upstream-sycl/graph_copy2d_probe native
```

Still open: the other original host translation units, kernel entry-point bindings,
interprocess events, graph upload/update/full node introspection, capture
modes beyond ThreadLocal, per-thread default streams, device capability/selection
and peer APIs, write-combined/I/O registration, asynchronous allocators, complete
CUDA error behavior, Windows and multi-device execution. Unsupported flags and
node enumeration return explicit errors. Full-model state parity and PP/TG
measurements remain unestablished.

The [MMQ stream binding](mmq-frontend.md) now connects the original public MMQ
API to this frontend. It also fixes process-exit cleanup of an unfinished capture,
so a fatal launch diagnostic is not hidden by a teardown synchronization error.

[GPU event timing](runtime-timing.md) now compiles the original layer/session host
files and runs the original stage timer. Complete engine linkage remains open.
