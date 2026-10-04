# Peer APIs and the original prefill stager

The source frontend now binds peer capability, enable/disable and asynchronous
peer copies to native SYCL operations. The unchanged original `prefill.cpp`,
`peer_experts.cpp` and `remote_experts.cpp` compile with native experts, MMQ and
fused declarations enabled. The original prefill stager also runs with separate
copy and compute streams. Full engine linking and model execution remain open;
these tests provide no new PP/TG result.

## Peer access and copies

`cudaDeviceCanAccessPeer` queries SYCL's native directed capability for distinct
visible devices. Enable/disable use `ext_oneapi_enable_peer_access` and its
disable counterpart, with a directed ledger updated only after native success.
The enable path reserves ledger storage before the native call and removes the
entry on failure. Repeated enable/disable requests report the corresponding peer
state errors. The reverse direction requires a separate enable operation.

Device allocations keep their creating ordinal. A foreign device allocation is
accessible only in a shared context with native access enabled from the queue's
device to the allocation's owner. Device release drains all live runtimes in that
context before releasing storage, including queues that could use peer memory.

`cudaMemcpyPeerAsync` validates ordinals, stream ownership, device allocation
owners and byte ranges, then submits the existing native asynchronous D2D copy.
It preserves default/nonblocking stream ordering and graph capture. A zero-byte
copy accepts null pointers after ordinal and stream validation. The CUDA
automatic staging path without peer access has not been implemented: such
cross-device transfers return an explicit unsupported error. Cross-platform
contexts are also unsupported. Original prefill checks peer admission before
using its peer-copy path; original peer experts enable both directions.

`cudaInitDevice` initializes without changing thread-local device selection.
Zero device flags and host mapping on a device with native host-USM support are
accepted. Spin scheduling has no native per-device binding and returns an
unsupported error. The original remote preflight already clears this optional
policy error and continues with default scheduling. This adapter covers the
original call pattern, not every CUDA initialization flag or the newer CUDA
`cudaInitDeviceFlagsAreValid` semantics. No CUDA architecture or runtime version
is invented.

## Original stager execution

The test includes the complete unchanged `src/prefill/prefill.cpp` to access its
internal `Stager`. Native experts, MMQ and fused declarations remain enabled.
Original expert-source and CPU layout objects provide their real key function,
type information and layout definitions. Function/data sections and linker
garbage collection discard unused engine functions whose device/GEMM bindings
remain open. This is a component link, not a full-engine link or a replacement
implementation of the stager.

Each run uses three CPU workers, the original ring-size setting of two pinned
buffers, eight generations and seventeen jobs per generation. Even jobs read a
stable pageable source. Odd jobs exercise the original virtual `copy_blob`
branch through a test `ExpertSource` that assembles three slices from patterned
input; all 68 such calls are checked. No model file is loaded.

An initial held copy-queue callback lets the first two DMAs be issued, then
checks that job two cannot overwrite buffer zero before DMA completion. After
release, every compute checksum waits on the original DMA event for its job.
There is no GPU synchronization between generations, so the original previous
generation's DMA events must also protect ring reuse. All payload and padding
bytes are checked after execution, along with 136 independently calculated GPU
checksums per run.

## B570 validation

Hardware: Intel Arc B570, AMD Ryzen 5 5600X, oneAPI 2026.1.1,
compute-runtime 26.35.39758.10, IGC 2.41.5, GMM 22.10.0 and Level Zero loader
1.32.0. JIT and `bmg_g21` AOT use the isolated driver libraries described in
[runtime-memory.md](runtime-memory.md), with persistent SYCL cache disabled.
All seventeen related CTest programs pass in both builds.

The peer test checks invalid/invisible devices, self-peer refusal, unsupported
spin policy, pointer/range/stale-stream errors, asynchronous submission behind
held work, independent-stream progress, and four changed-input graph replays.
It verifies 20,480 same-GPU copy bytes and executes the original remote
preflight's invisible-device refusal. Exactly one GPU is visible: actual
distinct-device capability queries, enable/disable and physical peer transfers
were **not executed**. The conditional second-GPU test does not establish
multi-GPU support on this machine.

| Original stager test | Blob stride | Jobs | CPU workers | Pinned ring | Payload and padding bytes checked |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small slots | 65,536 | 136 | 3 | 2 | 8,912,896 |
| Original CPU expert blob size | 1,382,400 | 136 | 3 | 2 | 188,006,400 |

The larger case uses the original `cpu::BLOB` constant. These arena sizes include
padding and do not represent bytes transferred or DMA bandwidth. Checksum and
byte verification time is not inference throughput. These tests establish
dependency and reuse behavior; a transfer/compute overlap trace and full-model
prefill parity remain open.

The [host syntax inventory](host-peer-compile-results.json) now reports seventeen
passes and three failures among twenty original host files. Remaining syntax
blockers are graph upload/inspection in `mtp.cpp` and `verify.cpp`, and device/
runtime metadata in `generate.cpp`. Compiler recovery can hide additional calls
behind unknown types or constants. Remaining device/GEMM families, full linking,
model-state comparisons, optional formats, multi-GPU execution and measured
PP/TG remain open. The unchanged large standalone MMQ arithmetic suites were
not repeated; all fourteen previous related regression programs were run.

[Results](runtime-peer-results.json) contain complete build/test logs and hashes
for source, binaries, original objects and selected driver files. All 256 original
manifest files remain byte-identical to upstream.

```sh
# Select the isolated driver libraries as in runtime-memory.md first.
cmake --build build-upstream-sycl -j3
SYCL_CACHE_PERSISTENT=0 ctest --test-dir build-upstream-sycl -V \
  -R 'original_|mmq_public_stages|mmq_context_lifetime|runtime_'
# Repeat in build-upstream-sycl-aot for bmg_g21.
python3 tools/sycl/audit_host_compile.py --compiler icpx \
  --ggml /path/to/pinned/llama.cpp --output /tmp/host-peer-compile-results.json
```

Primary references: [CUDA peer access](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__PEER.html),
[CUDA asynchronous peer copies](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html),
[CUDA device initialization](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html)
and [native SYCL peer access](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_peer_access.asciidoc).
