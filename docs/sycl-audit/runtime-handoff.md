# Mapped-memory CPU/GPU handoff

The isolated SYCL component now reproduces the upstream doorbell protocol on
Intel Arc B570 with an x86 host: the GPU publishes inputs and a sequence number,
the CPU computes a reply and raises a flag, and the running GPU resumes to read
that reply. The whole sequence can stay inside one native graph. This is a
measured device-specific implementation, not a claim that standard SYCL host
atomics are supported. The original engine is not yet connected to it.

## Source mapping

The reference is upstream commit `99f3dbd0b21d1401b3769e0c0d963913607f380b`.

| Reference | Ported processing |
| --- | --- |
| `src/kernels/cuda/elementwise.cu` `doorbell_ring` | Read and increment the current mapped sequence on the GPU, including on graph replay. Null sequence is a no-op. |
| `doorbell_publish` | One 1,024-thread group copies activation, IDs and weights, fences, joins the group, then publishes the next sequence. |
| `doorbell_wait` | Read the required sequence on the GPU and wait for equality with the host flag. Null flag/sequence is a no-op. |
| `copy_from_mapped` | Four-float loads/stores, 256 threads per group, at most 64 groups, grid stride, 16-byte alignment and a multiple-of-four extent. Nonpositive extents are no-ops. |
| `copy_i32_from_mapped` | One 128-thread group with strided volatile reads; nonpositive extents are no-ops. |
| `src/kernels/cuda/verify_kernels.cu` `wait_flag_ge` / `wait_flag_ge_or` | Wait for a flag at least as large as the required value; the resident-plan skip value is checked once before waiting. `UINT32_MAX` releases a GE wait. |
| `src/core/session.cpp` `session_run_token` | Host polls the sequence, reads inputs, fills the reply, issues a CPU fence and `_mm_sfence`, then stores the flag. |
| `src/core/verify.cpp` flag publication/cancellation | Monotonic GE flags and the high-water cancellation value must remain distinct from the equality doorbell. |

`include/strata/sycl_upstream/handoff.hpp` exposes queue-based components; the
original CUDA entry points are unchanged. Invalid size/alignment arguments use
C++ exceptions rather than the upstream process-exit/error API. No-op operations
return the queue's existing event through `ext_oneapi_get_last_event`, so they
neither erase ordering nor insert a node into an otherwise empty capture.

The float copy deliberately follows the actual CUDA body: its `float4` source
load casts away the parameter's `volatile` qualifier. It is not replaced with an
asynchronous copy-engine operation. The integer copy retains volatile reads.
Row-selective copy, scatter and conditional-copy entry points remain to be ported.

## Device capabilities and the failed direct translations

On the tested B570/Level Zero configuration, both
`usm_atomic_host_allocations` and `usm_atomic_shared_allocations` are false.
Khronos's [USM specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)
makes concurrent host/device atomic access conditional on these capabilities.
Selecting `memory_scope::system` alone does not grant that support. Host USM is
used for pinned, device-visible storage; no unsupported atomic RMW is issued to
that storage. The protocol follows the upstream volatile/fence approach and is
validated empirically for this compiler, driver and device.

A small bounded probe publishes 16 words, notifies a CPU thread already polling,
and waits for a reply. The CPU delays its reply by one millisecond. There are no
driver calls during that CPU service. Both variants have publication/completion
fences; only the polling-loop fence changes:

| GPU access | Without a fence inside the polling loop | With the polling-loop fence |
| --- | --- | --- |
| SPMD volatile load | Wait exhausted its 1,000,000-iteration bound | Reply observed; all words correct |
| ESIMD explicit L1/L2 uncached gather | Wait exhausted its bound; reply payload also incorrect | Reply observed; all words correct |

Checking payloads alone was insufficient: an early probe could observe the reply
bytes after its failed wait. The committed probe therefore requires both correct
payloads and a zero timeout result. Initial probes without a deliberate CPU delay
could also get a lucky first read; the delayed probe exercises an actual wait.
Submission times printed by the diagnostic include compilation/runtime startup
and must not be read as handoff latency.

For the ESIMD variant, the compiler's pre-backend LLVM IR still contained a load
inside a loop. IGC's optimized IR and vISA reduced the no-fence loop to **one
uncached load**, with no back edge. Adding the loop fence retained the load, fence
and backward branch. Uncached access alone did not prevent the compiler from
reusing a value. The [ESIMD memory/fence documentation](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd_functions.md)
describes hardware cache controls and fence scopes; these are not interchangeable
with a compiler-visible changing memory dependency.

The SPMD failure is different: optimized IR and native code still contain
repeated volatile loads. The exact cache/coherency cause of that failure has not
been established. Its source-level system-scoped fence lowers on this device to
an SLM group fence and a UGM GPU-scope invalidate; it is not evidence of a native
system-release instruction. The ESIMD source's explicit system fence, by contrast,
lowers to `lsc_fence.ugm.none.sysrel`. These findings are retained separately rather
than attributing both failures to loop removal.

The component uses the SPMD version with a fence on every unsuccessful poll and
an additional fence after notification. This preserves the original GPU wait
and host service schedule. It does not split the graph or queue a host callback
in place of the GPU wait. The fence cost has not yet been isolated or tuned.

## Validation on B570

Linux, Ryzen 5 5600X, Intel Arc B570, oneAPI 2026.1.1, Level Zero driver
`1.17.39395+14`, IGC `2.40.13`, 2026-10-04: JIT and `bmg_g21` AOT both pass
`runtime_mapped_handoff`.

The connected test uses 48 stages. Each stage publishes 4,096 floats, 10 IDs and
10 weights; the CPU checks them, changes every reply float and ID, then raises
the flag. The GPU copies the reply into the next stage's input. This checks both
directions, not just notification words. A captured sequence contains 192 nodes
and is reused with new input values and an advancing sequence counter.

Each build checks:

- 8 direct pipelines and 16 graph replays: **1,152 handshakes** and **4,840,176
  exact payload/final-output comparisons**, including data on multiple host pages.
- Twelve pipelines first use a bounded diagnostic wait. A deliberately delayed
  first reply records a positive GPU poll count in all 12, proving that the GPU
  entered its wait before receiving the reply. The remaining 12 pipelines use
  the default unbounded wait, matching the original protocol.
- GE comparison, `UINT32_MAX` cancellation value, and three graph replays with
  changed resident-plan skip/flag values. A deliberately unsatisfied bounded wait
  reports exhaustion, and a later replay with new host input succeeds.
- Empty/null operations produce no capture nodes. A real ring followed by no-ops
  preserves its dependency and remains exactly one native graph node.
- A 65,540-float copy exercises the second grid-stride iteration beyond the
  64-group cap; four extra guard elements remain unchanged (65,544 comparisons).

A bounded wait is explicitly a diagnostic extension: exhaustion writes its
status and exits the wait, so the caller must reject all dependent results. The
component default remains unbounded. The fixture reads diagnostic status only
after GPU completion. It never accepts a successful later copy as proof that the
wait succeeded. During host service it issues no event query, stream query or
synchronization; final synchronization occurs after the service has returned.

[Results](runtime-handoff-results.json) record source/binary hashes, full test
logs, capability reports, failed/successful probes, and relevant generated-code
excerpts/hashes. Unchanged runtime/MMQ suites were not rerun for this new component.
All 256 original shared-source/build files remain byte-identical.

## Reproduction and remaining work

With the existing isolated build configured:

```sh
source /opt/intel/oneapi/setvars.sh
cmake --build build-upstream-sycl --target handoff_test handoff_probe_0 handoff_probe_1 -j 4
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ctest --test-dir build-upstream-sycl -R runtime_mapped_handoff -V
# Expected diagnostic failure without the polling fence:
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./build-upstream-sycl/handoff_probe_0
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./build-upstream-sycl/handoff_probe_0 esimd
# Expected success with the polling fence:
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./build-upstream-sycl/handoff_probe_1
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./build-upstream-sycl/handoff_probe_1 esimd
```

The probes are opt-in build targets and are not CTest cases. For native code,
set `SYCL_CACHE_PERSISTENT=0`, `IGC_ShaderDumpEnable=1` and
`IGC_DumpToCustomDir=/path/to/dump` when running the JIT probe. The compiler dump
can change timings; evidence includes ordinary and dump-enabled executions.

This establishes the selected mapped handoff on the measured machine. It does
not establish general host-atomic support, arbitrary registered memory, portable
multi-device mappings, complete cancellation/fault recovery, or whole-model
inference parity. Allocation/registration APIs and the CUDA frontend binding
still need to connect the original host processing to these components. No new
prefill/decode throughput result follows from this test.
