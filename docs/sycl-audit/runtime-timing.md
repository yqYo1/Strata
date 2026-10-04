# GPU event timing and original host callers

The source frontend now implements `cudaEventCreate` and `cudaEventElapsedTime`,
and accepts timing-enabled events through `cudaEventCreateWithFlags`. This removes
the compile blockers in the original `src/core/layer.cpp` and `src/core/session.cpp`.
Both files compile unchanged. Their complete engine behavior and inference speed
remain unverified; only the original layer stage-timer functions are linked into
the connected timing test.

## Timestamp source and semantics

Timed records use Intel's
[`submit_profiling_tag`](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_profiling_tag.asciidoc).
On a device with `ext_oneapi_queue_profiling_tag`, this submits a timestamped marker
without enabling profiling for the whole queue. The returned start/end timestamps
are equal and expressed in nanoseconds. On the tested B570 this capability is
present, and the runtime queues retain `enable_profiling=false`. No CPU timestamp
or device-cycle-to-time estimate substitutes for these GPU timestamps.

The [CUDA event API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)
defines the returned interval in milliseconds, NotReady for incomplete records,
and invalid-resource errors for unrecorded or timing-disabled events. The frontend
checks both completion states before querying their timestamps. Failed or pending
queries leave the output unchanged. NotReady does not set the thread's last error.
An event rerecord replaces its current completion/timestamp; previously enqueued
waits keep their earlier event snapshot. Pending event destruction returns without
waiting. Default-stream ordering still follows the Runtime dependency rules.

Creation with `cudaEventBlockingSync` is accepted. Interprocess events remain
unsupported. A device without native profiling-tag support gets an explicit
unsupported error when creating a timed frontend event; queue-wide profiling is
not silently enabled as a fallback.

The original `layer.cpp` stage timer explicitly excludes captured graphs, because
its CUDA event records did not expose completed timestamps there. The SYCL
frontend keeps captured records as internal graph dependencies and refuses elapsed
time for them. Recording the event again outside capture restores ordinary timing.
The standalone native profiling-tag probe can record/finalize a graph, but this
has not established exportable per-replay event timestamps and is not used to
claim that capability.

## Connected validation

Intel Arc B570, Ryzen 5 5600X, Linux, oneAPI 2026.1.1, Level Zero and isolated
26.35.39758.10 / IGC 2.41.5 driver stack, 2026-10-04. JIT and `bmg_g21` AOT both
pass eight timing cases:

1. Default creation, blocking-sync flag, unsupported interprocess flag, null
   arguments, unrecorded events and disabled timing.
2. A held host callback separates two GPU records. The second record and elapsed
   query return before release; the query returns NotReady without touching its
   output or sticky error. After a 50 ms hold, the interval is at least 45 ms and
   lies inside separately submitted native GPU marker timestamps. Identical and
   reversed event pairs produce zero and the negated interval respectively.
3. A GPU kernel performs 262,144 dependent xorshift steps. The output equals the
   CPU calculation (`1505064235`), and its positive interval lies inside native
   marker bounds. The measured test interval includes command scheduling and is
   not a model benchmark.
4. Cross-stream event dependency, common timestamp epoch and rerecording while an
   earlier wait has already completed.
5. Destruction of a pending timed event before gate release, with stale-handle
   rejection afterward.
6. A default-stream timed record waits for a held blocking stream.
7. Captured records retain two dependency nodes, cannot be queried for elapsed
   time, and can subsequently be rerecorded for ordinary timing.
8. The original `stage_timing_enable`, stage marks and report execute unchanged,
   creating their original 2,560 event handles. The report measures a synthetic
   20 ms host-callback hold between GPU markers; it is not a model stage result.

The timing test links `layer.cpp` with function/data sections and discards its
unused unresolved engine functions. `session.cpp` also compiles into the host
archive but its inference functions are not linked/executed. This distinction is
necessary: these tests do not establish a complete engine link or session parity.

All eleven related CTest programs pass in both builds, including the previous
stream, graph, memory, mapped handoff, original host and MMQ frontend checks.
The unchanged large standalone MMQ arithmetic suites were not repeated.
[Results](runtime-timing-results.json) contain exact build/source/binary hashes,
full logs and measured synthetic intervals. All 256 original manifest files remain
byte-identical. These are correctness tests, not new PP/TG measurements.

The updated [host syntax inventory](host-timing-compile-results.json) reports
13 passing and seven failing translation units, compared with 11/9 before timing.
At that checkpoint, prefill's compile blockers were device get/set, free-memory
query and peer copies. [Device selection and free VRAM](runtime-devices.md) now
resolve the first three API families and run the original expert cache.
[Peer APIs and the original stager](runtime-peer.md) subsequently resolve
prefill's syntax blockers and verify pinned-ring reuse. The subsequent
[graph API integration](runtime-graph-api.md) connects upload and native node
types. Device metadata, full kernel-parameter inspection and remaining kernel/GEMM
bindings are open, as are complete engine linking and model results.

```sh
# Use the driver environment from runtime-memory.md first.
cmake --build build-upstream-sycl -j4
SYCL_CACHE_PERSISTENT=0 ctest --test-dir build-upstream-sycl \
  -R 'original_|mmq_public_stages|mmq_context_lifetime|runtime_' -V
# Repeat with build-upstream-sycl-aot for bmg_g21.
cmake --build build-upstream-sycl --target profiling_tag_probe
build-upstream-sycl/profiling_tag_probe
python3 tools/sycl/audit_host_compile.py --compiler icpx \
  --ggml /path/to/pinned/llama.cpp --output /tmp/host-timing-compile-results.json
```
