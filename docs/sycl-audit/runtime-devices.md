# Device selection and free VRAM

The source frontend now provides `cudaGetDeviceCount`, `cudaGetDevice`,
`cudaSetDevice` and `cudaMemGetInfo`. It links and runs the original
`src/core/expert_cache.cpp`, including its capacity refusal and slot fills,
without changing that source. Full engine linking and model execution remain
open. No new PP/TG measurement follows from these component tests.

## Execution domains

Device ordinals come from SYCL's visible Level Zero GPUs. OpenCL aliases of the
same card are excluded. Enumeration respects the process's SYCL visibility
selection; an OpenCL-only selection reports zero frontend devices. The count is
not hard-coded to one.

Each host thread starts with current device zero. A valid `cudaSetDevice`
initializes the selected domain and then changes that thread's selection; an
invalid ordinal leaves it unchanged. Default-stream operations use the selected
domain. Named streams, events and graphs retain their creating domain. Event
recording and graph launch reject mismatched owners. Query, synchronization,
destruction and allocation release resolve the resource's owner.

Visible devices in a Level Zero platform share a SYCL context. Host allocations
and registrations use one ledger per platform, preventing duplicate native page
imports after a device switch. Host release drains all live runtimes in that
context before releasing storage. Portable host flags are rejected when visible
devices span platforms with separate contexts. Cross-device USM operations were
unsupported at this checkpoint; the subsequent [peer binding](runtime-peer.md)
now uses native capability and directed enable/disable operations. No peer
capability or CUDA compute capability is invented.

An uncaptured event can be copied under its owner's runtime mutex and waited on
by a different runtime in the same context. A later record cannot retarget the
snapshot. Captured dependencies remain inside their original runtime. Timing
requires events with the same creating device; sharing a context does not prove
that GPU clocks share an epoch.

## Memory query

The query uses standalone Level Zero Sysman initialization (`zesInit(0)`),
finds the Sysman device by the compute device's UUID, and reads
`zesMemoryGetState().free` for device memory modules. Subdevice UUIDs select
their corresponding module IDs. A root reports its distinct modules; mixed root
and tile module reporting is rejected to avoid counting memory twice. Total
capacity comes from SYCL's `global_mem_size`; the deprecated Sysman `state.size`
is not used by the frontend.

The implementation never subtracts its allocation ledger from total capacity.
It observes allocations made outside that ledger, other processes and runtime
pools. It does not synchronize queued GPU work to answer the query. Missing or
inconsistent native memory information returns an unsupported error and leaves
both outputs untouched. Device initialization can also fail independently.

Free physical memory does not guarantee that an allocation of that size will
succeed. The original cache retains its allocation-error handling. The native
probe also records the usable-memory extension separately: on this B570 it
reported 9,899,540,480 bytes, while SYCL global capacity reported 10,666,115,072.
These are different driver quantities; the usable extension is not substituted
for physical free memory.

The frontend does not set process-wide Sysman environment variables. If the
legacy `ZES_ENABLE_SYSMAN` remains enabled when the query is reached, the
standalone path is rejected. The installed UR changes that variable to zero
during its own device initialization; the rejection test deliberately restores
it after initialization to exercise the frontend's error path.

## Validation on B570

Hardware: Intel Arc B570, AMD Ryzen 5 5600X, oneAPI 2026.1.1,
compute-runtime 26.35.39758.10, IGC 2.41.5, GMM 22.10.0 and Level Zero loader
1.32.0. Both JIT and `bmg_g21` AOT builds use the isolated driver libraries
from the [memory audit](runtime-memory.md), with persistent SYCL cache disabled.

All fourteen related CTest programs pass in both builds. Added checks cover:

- Native visible-device count, invalid ordinals, unchanged selection after an
  error, and eight concurrent host threads starting with device zero.
- A GPU-touched 128 MiB allocation made outside the frontend ledger changes
  the native free-memory result. In the isolated JIT check, reported free memory
  went from 10,422,456,320 to 10,283,704,320 bytes. Pool and driver allocations
  also affect these counters; their difference is not an allocation-size oracle.
- The unchanged expert cache rejects a request larger than device capacity
  before creating a frontend allocation. Four slots exercise blocking, queued
  and stream fills, residency/bounds and the original byte-by-byte verifier:
  16,384 bytes checked, followed by release.
- Two real runtimes on this GPU share a context. A held host callback verifies
  that event snapshot/wait submission returns before completion and preserves
  the dependency; captured event export is rejected.
- Empty Level Zero visibility and unsupported legacy Sysman initialization
  preserve explicit errors and untouched query outputs.

The optional second-GPU resource-switch case was **not executed**: exactly one
Level Zero GPU is visible here. These tests do not establish multi-GPU behavior.
The unchanged standalone MMQ quantizer/product arithmetic suites were not
repeated. Existing stream, graph, handoff, memory, public MMQ, original host and
stage-timer regressions were run.

The [results](runtime-devices-results.json) record source/binary hashes, original
source identity, complete build/test logs, the native probe and the earlier test
failure that exposed UR's environment-variable change. All 256 original manifest
files remain unchanged. The [host syntax inventory](host-device-compile-results.json)
at this checkpoint has 14 passes and six failures; prefill's four syntax errors
are `cudaMemcpyPeerAsync` calls. The subsequent [peer/stager validation](runtime-peer.md)
resolves those errors and compiles original peer/remote host files, bringing the
[latest inventory](host-peer-compile-results.json) to 17 passes and three failures.
Graph upload/inspection, device metadata, remaining device/GEMM bindings, full
linking, model-state parity and PP/TG results remain open.

```sh
# Select the isolated driver libraries as in runtime-memory.md first.
cmake --build build-upstream-sycl -j3
SYCL_CACHE_PERSISTENT=0 ctest --test-dir build-upstream-sycl -V \
  -R 'original_|mmq_public_stages|mmq_context_lifetime|runtime_'
# Repeat in build-upstream-sycl-aot for bmg_g21.
cmake --build build-upstream-sycl --target sysman_probe
build-upstream-sycl/sysman_probe
python3 tools/sycl/audit_host_compile.py --compiler icpx \
  --ggml /path/to/pinned/llama.cpp --output /tmp/host-device-compile-results.json
```

Primary references: [CUDA device management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html),
[CUDA memory queries](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html),
[Sysman initialization](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/sysman/PROG.html)
and [native peer access](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_peer_access.asciidoc).
The subsequent native peer binding is documented in [runtime-peer.md](runtime-peer.md).
