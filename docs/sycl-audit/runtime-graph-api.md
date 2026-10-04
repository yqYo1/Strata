# Native graph preparation, inspection and concurrent waits

The source frontend now provides graph upload, node enumeration and native node
types. The unchanged original `mtp.cpp` and `verify.cpp` compile with native
experts, MMQ and fused declarations enabled. A component test executes the
original MTP capture-finishing helper, then replays its graph. Full engine
linking, model execution and new PP/TG results remain open.

This integration also exposed a wait problem in the installed SYCL runtime:
a CPU event waiter could stop another thread's DMA or kernel submission. The
runtime now waits for completion without retaining that scheduler lock.

## Preparation and upload

SYCL `finalize()` prepares each executable during instantiation. A native probe
records UR program build, kernel creation, command-buffer append and finalization
before the first graph launch. Its shared value, copy destination and callback
counter remain unchanged after finalization. The first launch then produces the
expected values and exactly one callback. This is evidence from the installed
B570 stack, not a promise about every SYCL backend or version.

The frontend's upload queues a completion marker after this preparation; it does
not replay the graph. Uploads and launches of one executable share a dependency
chain across streams. The marker follows both preceding stream work and the
previous operation on that executable. Other nonblocking streams remain free to
submit work. Pending uploads retain the executable after its public handle is
destroyed. Capture-time uploads are rejected and invalidate the recording.

Recorded buffers remain caller-owned. CUDA graph-owned allocations, stream-backed
allocation caches, device-side graph launch flags and graph updates are outside
this binding. The original MTP helper's explicit upload and synchronization use
the prepared native executable without executing any recorded model operations.

## Node inspection

Definition creation snapshots SYCL's real `get_nodes()` and `get_type()` results.
Enumeration supplies stable opaque tokens, respects the caller's capacity, and
fills excess entries with null. Destroying the definition invalidates its node
tokens while independent executable instances remain usable.

Native kernel, memcpy, byte memset, host-task, subgraph and empty/barrier kinds
have source-frontend counterparts. Other native kinds return an explicit
unsupported error. The mixed-command test observes four native kinds; it does
not replace all recorded commands with an empty node or guess their types from
CUDA spelling.

The public SYCL node API does not expose CUDA kernel function pointers or argument
arrays. `cudaGraphKernelNodeGetParams` therefore returns an unsupported error for
a kernel, leaving the output untouched. The original verify file now compiles,
but its optional `STRATA_VERIFY_NODES` kernel-parameter/name diagnostic remains
unsupported; its interaction with sticky launch errors has not been established.
No CUDA SDK version is fabricated to enable its name-query branch. Full parameter
inspection remains open.

## Wait problem and correction

The unchanged small-slot stager initially failed its held-DMA check. More precise
diagnostics showed that the first two DMAs had not both been issued before the
two-second deadline, while the protected slot remained unready and the producer
was still pending. The earlier generic message said buffer reuse had failed;
these observations distinguish lack of submission progress from early overwrite.

Diagnostic stores and UR tracing changed the timing and could hide the failure.
After restoring the exact original test, two of eight runs failed. Attached
backtraces in both failures show a stager CPU worker inside
`Scheduler::waitForEvent` / `ExecCGCommand::enqueueImpQueue` /
`event_impl::waitInternal`. The producer is waiting for the scheduler's write lock
in `Scheduler::addCG`, once while submitting a kernel and once while submitting a
copy. A separate deterministic test also fails before the correction: an event
waiter behind a held callback prevents an unrelated host-command submission from
returning before that callback is released. This was not a cold-compilation
timeout; the backtraces identify the lock wait.

`Runtime::synchronize_native` now polls the nonblocking completion status with
host-thread yields, then calls `wait_and_throw` on the completed event. Event,
stream and device waits, plus the frontend's synchronous native copy/host memset
waits, use it. Waiting happens outside the runtime mutex, retaining the existing
event and stream snapshots. The final native wait still delivers asynchronous
errors. This uses CPU polling while pending; it is not a measured inference
optimization or a complete CUDA scheduling-policy binding.

The concurrent-wait regression checks event, stream, device and synchronous D2H
waits separately. Each waiter remains pending while another thread successfully
submits a host command; after release both finish. An injected host-task exception
is reported through event synchronization and thread-local error state. The same
queue then completes a byte-checked GPU fill. This does not establish recovery
from a device fault.

## B570 validation

Hardware: Intel Arc B570, AMD Ryzen 5 5600X, oneAPI 2026.1.1,
compute-runtime 26.35.39758.10, IGC 2.41.5, GMM 22.10.0 and Level Zero loader
1.32.0. JIT and `bmg_g21` AOT use the isolated driver libraries described in
[runtime-memory.md](runtime-memory.md), with persistent SYCL cache disabled.
All eighteen related CTest programs pass in both builds.

The added program covers six graph scenarios, four concurrent-wait variants and
host-task error delivery. It checks stable/partial/oversized/empty enumeration,
native node kinds, stale handles, upload without execution, asynchronous upload,
upload/launch serialization, pending executable destruction, legacy-default
ordering, invalidated capture and four state-changing replays through the original
MTP finisher. The original stager tests remain unchanged and pass at both 65,536
and 1,382,400-byte blob strides. The small test also passes eight consecutive
post-correction JIT runs. These repeats address the earlier intermittent failure;
their wall times are not PP/TG or bandwidth measurements.

All seventeen previous related regression programs were run. The unchanged large
standalone MMQ arithmetic suites were not repeated. The
[host syntax inventory](host-graph-compile-results.json) now has nineteen passes
and one failure among twenty original host files. The remaining failed file is
`generate.cpp`, with device/runtime metadata blockers. Successful syntax does not
establish linking or execution; remaining device/GEMM families, full prefill and
session state, optional formats, multi-GPU behavior and measured PP/TG are open.

[Results](runtime-graph-api-results.json) preserve the initial failures, diagnostic
backtraces, before/after wait reproducer, final build/test logs, native preparation
trace and source/object/binary/library hashes. All 256 original manifest files
remain byte-identical to upstream.

```sh
# Select the isolated driver libraries as in runtime-memory.md first.
cmake --build build-upstream-sycl -j3
SYCL_CACHE_PERSISTENT=0 ctest --test-dir build-upstream-sycl -V \
  -R 'original_|mmq_public_stages|mmq_context_lifetime|runtime_'
# Repeat in build-upstream-sycl-aot for bmg_g21.
cmake --build build-upstream-sycl --target graph_prepare_probe
SYCL_CACHE_PERSISTENT=0 SYCL_UR_TRACE=2 build-upstream-sycl/graph_prepare_probe
python3 tools/sycl/audit_host_compile.py --compiler icpx \
  --ggml /path/to/pinned/llama.cpp --output /tmp/host-graph-compile-results.json
```

Primary references: [CUDA graph management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html),
[SYCL graph and node APIs](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc),
[SYCL scheduler source](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/scheduler/scheduler.cpp),
[host dependency waits](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/scheduler/commands.cpp)
and [event status queries](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/event_impl.cpp).
The source links describe the upstream implementation; the attached stacks and
native traces are the evidence for the installed runtime.
