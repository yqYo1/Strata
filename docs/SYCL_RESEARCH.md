# SYCL port research: Arc B570 / Ryzen 5 5600X

Investigated 2026-10-02, Strata `1678de3`, branch `feature/sycl`.
This is a port design and small device probes, not a working inference backend.
The target is this Linux PC and Intel Battlemage or newer. Other vendors,
Alchemist, Windows packaging and multi-GPU support are outside the initial scope.
Newer Intel generations still need their own validation; a B570 result does not
establish compatibility with every future GPU.

## Machine and working toolchain

Observed locally using `lscpu`, `free`, `lspci`, sysfs, `xpu-smi`, `sycl-ls`,
and the probes in `tools/sycl/`:

| Item | Observation |
| --- | --- |
| GPU | Intel Arc B570, PCI ID `8086:e20c`, BDF `0000:05:00.0`, kernel driver `xe` |
| GPU memory | SYCL reports 10,666,115,072 bytes (10,172 MiB); max allocation 10,132,807,680 bytes |
| CPU | Ryzen 5 5600X, 6 cores / 12 threads, AVX2 / FMA / F16C, no AVX-512 |
| System RAM | `free -h`: 125 GiB total, 72 GiB available at inspection; availability changes with other workloads |
| OS | Ubuntu 24.04.5 LTS, kernel `7.0.0-28-generic` |
| Compiler | Installed oneAPI DPC++/C++ 2026.1.1, `2026.1.1.20260724` |
| Runtime | Level Zero V2 via Unified Runtime, driver string `1.17.39395+14` |
| GPU subgroups | 16 and 32 reported |
| Local memory / workgroup limit | 128 KiB / 1,024 work-items reported |
| USM | Host and shared allocations supported; **atomic host and atomic shared allocations not supported** |
| Graph | `ext_oneapi_graph` supported; single-kernel graph replayed 100 times successfully |
| Matrix | FP16 8x16x16 `joint_matrix`, subgroup 16, FP32 accumulation: all 128 outputs correct |

oneAPI is already installed under `/opt/intel/oneapi`. It is not initialized in
the default shell. Running `sycl-ls` by absolute path alone failed to load the
Level Zero adapter because `libumf.so.1` was not found. Sourcing `setvars.sh`
resolved this. No driver or package installation was needed.

Select `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` for reproducible experiments:
the B570 is also enumerated through OpenCL, and an OpenCL CPU device exists.
There is also a GT 710 using nouveau; it is not an inference target.

The matrix query reports FP16, BF16, INT8 and TF32 combinations. Only the FP16
tile was executed here. Do not infer the numerical behavior of TF32, BF16,
quantized expert kernels or full GEMMs from that test. Likewise, the reported
FP64 aspect does not establish fast native FP64 throughput.

## PCIe and memory observations

The upstream path through `03:00.0` reports **16 GT/s x4**. The GPU endpoint and
its immediate internal bridge report 2.5 GT/s x1, even for their maximum link
fields. Those endpoint values are not a useful estimate of this machine's
effective host transfer bandwidth: the copy measurement below is much faster.
The GPU sits downstream of the chipset switch; competing I/O may matter.
Do not assume the reference PC's approximately 26 GB/s PCIe path.

First warm transfer probe, 64 MiB USM host/device buffers, 32 queued copies per
direction, one warm-up copy and a final queue wait, host wall-clock timing:

| Direction | GB/s (decimal, payload bytes only) |
| --- | ---: |
| Host to device | 6.41 |
| Device to host | 6.56 |
| Device to device | 541.04 |

This is a copy microbenchmark with repeated buffers on a running desktop, not a
model benchmark or a sustained VRAM bandwidth measurement. In particular, the
D2D result may reflect caching/copy-engine behavior and must not be used as a
GEMV bandwidth estimate. No dedicated measurement of DRAM bandwidth, concurrent
CPU work plus DMA, storage bandwidth or small expert transfer latency was made.

Implications to test, rather than performance promises:

- Keep frequently used experts compressed in VRAM and preserve CPU/GPU overlap.
  Approximately 10 GiB VRAM must also cover dense weights, KV, workspaces and
  eventually MTP; it is not all available for expert caching.
- Start with CPU AVX2 native experts. This CPU cannot execute the canonical
  pack's AVX-512 expert path (`src/program/generate.cpp`, native-pack check).
  Native Q2_0 is distinct from canonical Q2_0 and is a possible first fixture.
- Tune CPU worker count around six physical cores, then compare SMT. Do not
  copy throughput figures measured on Zen 4 / AVX-512 into this plan.
- Calibrate missed-expert CPU versus DMA/GPU assignment. The existing native
  `pcie_frac` heuristic would give about 0.14 at 6.4 GB/s, but this is only a
  starting point: it does not measure the B570 plus 5600X workload balance.
- RAM capacity permits retaining many experts on the host, but current free
  memory is the budget. Do not assume that a large Unsloth pack plus its dense
  weights, KV and staging allocations fits beside current workloads.
- Delay long-context KV streaming until correctness and transfer overlap are
  measured. Host KV traffic shares the same limited link as expert transfers.

## Existing implementation and port boundaries

`src/` contains 50 `.cu` files, 19,326 lines and 243 textual `__global__`
occurrences; five `.cu` files contain `asm`. These counts include alternatives
and helper paths, not 243 kernels all required by the first inference path.
CUDA API calls also occur throughout `.cpp` and public headers.

| Area | Existing sources | SYCL work |
| --- | --- | --- |
| Build | `CMakeLists.txt`, `cmake/hip_backend.cmake` | Add an explicit SYCL build, mutually exclusive GPU backend selection, `icpx -fsycl`, Intel library discovery and backend-specific sources |
| Device and allocations | `src/core/device.cu`, `pinned.cu`, `expert_cache.cpp` | Device selection, USM device/host ownership, bounded staging, memory budget and error handling |
| Scheduling | `src/core/graph.cpp`, `session.cpp`, `verify.cpp`, `mtp.cpp` | Replace CUDA streams/events/capture; explicit dependencies across compute and copy queues |
| CPU/GPU handoff | `src/kernels/cuda/elementwise.cu`, `src/core/session.cpp` | Replace mapped-memory doorbell polling with completed transfers/events initially |
| Decode | `src/kernels/cuda/`, particularly native MMVQ, MoE, GR/GDN, QSA, PLE, router, RoPE, KV and sampler | Port the selected model's complete path, including tensor layout, reductions and quantization contracts |
| Prefill | `src/prefill/gemm.cu`, `kernels.cu`, `prefill.cpp`, `moe_mmq.cu` | SYCL GEMM adapter and batching; replace CUDA-specific MMQ integration |
| CPU experts | `src/kernels/cpu/native_expert.cpp`, `iq_avx2.cpp`, `kq_avx2.cpp`, `pool.cpp` | Reuse native AVX2 implementations and tune scheduling on the 5600X |
| Launch/server | `setup.py`, `serve/server.py` | Later add SYCL backend recognition, runtime environment and device selection; do not rely on CUDA/HIP detection |

HIP's shim works because HIP retains CUDA-shaped launches and APIs. SYCL does
not: renaming runtime functions cannot translate `<<<...>>>`, `threadIdx`,
warp shuffles, dynamic shared memory, PTX or graph semantics. Use a small typed
runtime layer and explicit SYCL kernel implementations. SYCLomatic can help
translate simple kernels, but its output still needs review and parity tests.
Avoid a wholesale inference-graph rewrite to ggml just to obtain GPU kernels.

### Scheduling is the first architectural issue

The current token graph launches a GPU wait kernel while the host computes
experts, with mapped `volatile` flags and system fences. The session code also
documents CUDA-driver-dependent visibility requiring event queries. `volatile`
is not a portable CPU/GPU synchronization contract.

This B570 runtime reports `usm_atomic_host_allocations=false` and
`usm_atomic_shared_allocations=false`. A direct replacement with system-scope
atomics on `malloc_host` / `malloc_shared` allocations is therefore not a
supported basis for this port. This does not exclude every system-memory
allocation path (see the follow-up below). Initial correctness baseline:

1. Compute the router and export the small activation/IDs/weights payload.
2. Wait for its D2H completion before reading it on the CPU.
3. Run GPU-resident experts concurrently with the CPU expert pool and, where
   beneficial, stage missed experts through a separate copy queue.
4. Transfer the CPU contribution back, join dependencies and combine outputs.

Use a common SYCL context for related queues and allocations; do not depend on
implicit CUDA default-stream ordering. Keep addresses stable for graph replay.
First validate this without graphs, then capture the GPU-only sections between
handoffs. The successful 100-replay probe is not validation of the existing
full-token graph, host tasks, external events or oneMKL/oneDNN graph integration.
The graph extension permits host tasks, but documents submission/performance
limits; host tasks are an experiment, not the initial overlap mechanism.

#### Follow-up: what the atomic capability results actually mean

Direct `zeDeviceGetMemoryAccessProperties` queries on the same B570 returned:

| Level Zero allocation category | Read/write | Atomic | Concurrent | Concurrent atomic |
| --- | --- | --- | --- | --- |
| Host | yes | yes | no | no |
| Device | yes | yes | no | no |
| Shared, single device | yes | yes | no | no |
| Shared, cross device | no | no | no | no |
| Shared system (ordinary system allocations) | yes | yes | yes | yes |

A separate SYCL query also reports `usm_system_allocations=true` and includes
`memory_scope::system` in the atomic scope list. Scope support alone does not
establish support for every allocation category. These results distinguish
device-side atomic operations from concurrent host/device atomic operations.

The installed driver's matching source tag, `26.31.39395.14`, has the default
host capabilities set to access plus atomic access; concurrent support is a
separate decision. Single-device shared concurrent support also depends on
kernel-driver migration support. Linux BMG explicitly advertises all four
shared-system capability bits. Thus the two false SYCL aspects are consistent
with driver policy, not evidence of a broken oneAPI installation or a missing
BIOS toggle. No relevant process environment overrides were found. This is not
proof that the hardware fundamentally cannot perform CPU/GPU atomic sharing.

Ordinary system allocations are a **separate optimization candidate**. Verify
the applicable SYCL/Level Zero memory-model contract and test bounded two-way
publication, changing payloads, graph replay and page-fault behavior before
using that path for the doorbell. It was queried, not execution-tested here.
Do not force capability bits using driver debug overrides and treat that as
validation of coherence or forward progress.

The initial event-based baseline may add host submission/completion latency and
break a full-token graph into smaller sections, particularly affecting decode.
It does not prohibit concurrent CPU experts and independent GPU work, GPU-local
atomics or graph execution. Host USM can also be accessed in alternating phases
with proper completion synchronization, so missing concurrent atomics alone
does not require copying every shared payload. Compare that option with explicit
DMA rather than assuming the latter is always faster. There is no measured
inference slowdown attributable to this limitation yet.

Sources: [driver capability defaults](https://github.com/intel/compute-runtime/blob/26.31.39395.14/shared/source/os_interface/product_helper.inl),
[Linux BMG system-memory capabilities](https://github.com/intel/compute-runtime/blob/26.31.39395.14/shared/source/xe2_hpg_core/linux/hw_info_extra_bmg.cpp),
and [SYCL USM access rules](https://github.khronos.org/SYCL_Reference/iface/usm_basic_concept.html).

Large expert RAM arenas also need care. SYCL has no general portable equivalent
of `cudaHostRegister` for arbitrary existing mappings. Keep bulk CPU data in
ordinary host memory initially and use bounded `malloc_host` staging for DMA;
measure the extra host copy and compare supported Intel-specific alternatives
before deciding to pin tens of GiB. Never replace the entire expert store with
`malloc_shared` and assume it remains in RAM without migration costs.

### Kernel and numerical choices

- For CUDA algorithms whose lane indexing assumes 32, an explicit supported
  subgroup of 32 can be a correctness starting point. Then compare a redesigned
  subgroup-16 implementation. Changing a warp constant alone is insufficient.
- Use FP32 reference kernels for GR/GDN, QSA scoring, normalization and reductions
  first. CUDA `mma`, `ldmatrix` and `cp.async` require different implementations.
  Native QSA scoring even depends on the pinned CUDA TF32 arithmetic contract.
- Prefer oneMKL or oneDNN for initial dense prefill GEMMs; select by correctness
  and timing at Strata's actual shapes. Check strides, transposes, BF16 input,
  FP32 accumulation and workspace memory. Use `joint_matrix`/XMX for targeted
  optimizations after these references exist, not for every single-token GEMV.
- Preserve native quantization block layouts and codebooks. Test signed packed
  byte dot products, rounding ties, saturation and Q8 scales independently.
  Keep strict floating-point behavior initially; the HIP port already needed
  contraction disabled for its Q8_K quantizer. Avoid global fast-math initially.
- Test both subgroup sizes, local-memory use, register pressure and workgroup
  sizes per kernel. Device maxima are limits, not recommended launch sizes.
- Start with JIT SPIR-V. A BMG AOT build can later reduce startup compilation;
  a BMG-only binary alone does not satisfy future-generation compatibility.
  Check architecture and required aspects explicitly at startup.

### llama.cpp reuse

An existing ghq checkout was inspected read-only at
`/home/yayoi/ghq/github.com/ggml-org/llama.cpp`, commit
`42d958167a748f2c04b1f888e84e7a58f609ddcb` (2026-10-01). No clone was needed.
Strata pins `3cf03257f219afbe7334045ff7c6a06ac68c627d`, also available locally.
Between these commits, `ggml/src/ggml-sycl` differs in 15 files, with 1,136
insertions and 122 deletions. Treat the pin and newer SYCL code separately.

The existing backend provides quantized matrix/vector kernels, codebook handling,
Intel library integration and graph examples. Its inspected CMake uses subgroup
16 and has handling for allocations above 4 GiB and optional AOT. Study these
paths and adapt narrowly with MIT notices retained. Do not simply enable
`GGML_SYCL` and expect Strata's CUDA MMQ adapter or custom QSA/PLE to use it.
Do not update the global llama.cpp pin without checking CPU expert behavior,
quantization layouts and existing oracle tests.

Use llama.cpp as a correctness and API reference, not an optimization ceiling
or a presumed optimal kernel design. Optimize against Strata's measured shapes,
memory traffic and CPU/GPU scheduling on this B570/5600X machine.

## Implementation order and completion checks

1. **Runtime foundation:** add SYCL build/device/memory/queue support. Prove
   asynchronous exceptions, allocation failures, cross-queue copies, repeated
   handoffs with changing payloads and bounded waits. Validate large (>4 GiB)
   addressing separately; the current 64 MiB probe does not cover it.
2. **One native model, short-context decode:** native AVX2 CPU experts, SYCL
   dense and expert kernels, routing, all required attention/GR/GDN/PLE paths,
   FP16 KV and sampling. Disable MTP and KV streaming initially. Use event-based
   layer scheduling; start with a small fixed cache and deterministic placement.
3. **Prefill and overlap:** implement correct batched GEMMs, expert staging and
   cache budgets. Then compare CPU misses, streamed GPU misses, subgroup choices
   and graph replay. Reuse the existing calibration intent, not CUDA timings.
4. **Verification and longer contexts:** add MTP/verify, quantized KV and streaming
   one feature at a time. Test graph replay with changing positions/inputs,
   conversation save/resume, cancellation and multiple requests in one process.
5. **Personal launch workflow:** explicit Linux SYCL configuration and server
   launch, then optional setup integration. Vision and distribution packaging
   can be separate work after text inference is validated.

Adapt existing parity checks for quantization, native experts, router, RoPE,
GR/GDN, QSA, PLE, KV and sampler. Compare against CPU references or saved oracle
fixtures, then compare layer outputs and logits using identical token IDs. A
successful compile or plausible generated text is not sufficient. Cross-vendor
float reductions need defined tolerances and routing/top-k stability checks;
do not promise byte-identical generated text from CUDA versus SYCL.

For performance, record exact model/pack, context, prompt IDs, generated length,
cache placement, worker count, MTP setting, versions and background workloads.
Measure TTFT, prefill tokens/s and decode tokens/s separately, with warm/cold
conditions and several repetitions. Record CPU expert time, transfer time,
GPU time, cache hits and peak memory. No inference tokens/s is established here.
The checkout had no `models*` directory; locating/selecting a real fixture and
its model-specific required kernels remains part of the first implementation.

## Reproduce the probes

Run from the repository root in bash with the installed toolkit:

```sh
source /opt/intel/oneapi/setvars.sh
mkdir -p /tmp/strata-sycl-research
icpx -fsycl -O2 tools/sycl/device_probe.cpp -o /tmp/strata-sycl-research/device-probe
icpx -fsycl -O2 tools/sycl/matrix_probe.cpp -o /tmp/strata-sycl-research/matrix-probe
ONEAPI_DEVICE_SELECTOR=level_zero:gpu timeout 40 /tmp/strata-sycl-research/device-probe
ONEAPI_DEVICE_SELECTOR=level_zero:gpu timeout 30 /tmp/strata-sycl-research/matrix-probe
```

The device probe checks replay result 100. The matrix probe checks a single
constant-input tile against the exact expected result 32. These are standalone
research programs, outside the engine build, not an inference validation suite.

To inspect the underlying Level Zero memory capabilities using the installed
system headers and loader (no oneAPI compiler needed):

```sh
/usr/bin/g++ tools/sycl/level_zero_caps.cpp -lze_loader -o /tmp/strata-sycl-research/ze-caps
/tmp/strata-sycl-research/ze-caps
```

## External primary references

- [Intel Arc B-series specifications](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc/desktop/b-series.html): B570 product context; local runtime limits above were queried separately.
- [SYCL 2020 specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html): USM aspects, memory model and event ordering.
- [Intel command graph extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc): capture, host-task and event restrictions; this evolving specification is not proof of installed runtime behavior.
- [Intel joint matrix extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc): query supported types/shapes rather than assuming CUDA MMA tiles map directly.
- [llama.cpp SYCL documentation](https://github.com/ggml-org/llama.cpp/blob/42d958167a748f2c04b1f888e84e7a58f609ddcb/docs/backend/SYCL.md) and [backend sources](https://github.com/ggml-org/llama.cpp/tree/42d958167a748f2c04b1f888e84e7a58f609ddcb/ggml/src/ggml-sycl): reusable implementation references, not a Strata performance baseline.

Local background: [engine details](DETAILS.md), [paper](paper/Strata-Paper.pdf),
[HIP port and parity caveats](AMD_HIP.md), and [native MMVQ contract](native-mmvq.md).
