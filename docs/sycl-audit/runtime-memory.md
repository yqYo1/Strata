# Allocation and mapped host registration

This component implements owned device/host allocations and external host
registration in one `Runtime` context. It is not yet bound to the CUDA frontend
or the full engine. The original host sources remain unchanged.

## Source processing retained

| Original caller | Required operation | Component behavior |
| --- | --- | --- |
| `src/core/pinned.cu`, `PinnedArena` | Register anonymous or shared-file arena before touching pages; register bounded slices; unregister slices before releasing the arena | Native external mappings of existing pages, with separate logical slice metadata |
| `src/core/expert_source.cpp` | Owned pinned storage or partial registration; device alias at an interior offset | Owned host USM and registered-memory aliases retain the host address and offset |
| `src/core/layer.cpp`, `verify.cpp`, `mtp.cpp` | Mapped payload/flag buffers, ordinary device buffers, release during teardown | Host/device allocation kinds and matching release paths; pending Runtime work completes before storage is released |

`Memory` uses aligned device USM (256 bytes) and host USM (64 bytes). Zero-size
allocation returns null. Null free is a no-op. Lookup covers interior pointers;
release requires the original pointer and matching kind. Duplicate/overlapping
logical registrations, conflicting permissions on shared pages, wrong-kind frees,
interior frees, double frees and overflowing ranges are rejected.

External storage is imported through `ZE_extension_external_memmap_sysmem`, using
`zeMemAllocHost` with `ze_external_memmap_sysmem_ext_desc_t`. This creates a native
mapping of the original storage; it does not allocate a second payload buffer.
The returned address must equal the supplied address and the driver must report
host/imported-host allocation metadata. Read-only mappings use the corresponding
native allocation flag. A successful SYCL `prepare_for_device_copy` call alone was
not accepted as evidence of native registration: the initial probe returned
unknown native allocation metadata despite successful GPU access on this
system-USM-capable device.

Native imports must not overlap. Logical slices can share an edge page, so the
registry imports uncovered page ranges and retains each backing mapping until
its last referencing slice is released. A coarse backing can include unused
prefix pages after the first slice is unregistered. **Keep the backing arena
mapped until all its registrations have been released.** This matches
`PinnedArena` teardown; independent partial unmapping of that arena is outside
this component's contract. Adjacent slices with different permissions on a shared
page are rejected. Native free is followed by a metadata query to confirm release.

Free/unregister marks its entry as retiring, releases the registry mutex, waits
for Runtime work (including retired streams), then releases storage. A concurrent
allocation remains possible while the wait is pending. The caller must prevent
new submissions that use a pointer while that pointer is being released. The
Runtime must outlive Memory. Native release failure during teardown is fatal;
recovery from partial driver failures has not been established.

Allocation/registration/free on the thread owning an active capture invalidates
that capture before modifying storage. Allocation on another thread is allowed
under the supported ThreadLocal capture model. Device synchronization still
rejects active capture. This is the selected lifetime policy, not a claim that all
CUDA allocation/free scheduling and error codes have been reproduced. CUDA's
[synchronous/async copy rules](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
remain a separate frontend task; synchronizing every copy is not the planned binding.

## Driver lifetime failure and correction

On B570 with compute-runtime **26.31.39395.14**, native registration and GPU reads
and writes succeeded, but `zeMemFree` returned success while allocation metadata
still reported host storage. A following file mapping at the reused address then
collided with that retained allocation. The standalone `memory_import_probe`
checks release, not just the kernel result:

| Driver | Imported type | Free result | Type after free | Probe result |
| --- | --- | --- | --- | --- |
| 26.31.39395.14 | HOST (1) | success (0) | HOST (1) | fail after 2,048 correct GPU values |
| 26.35.39758.10 | HOST (1) | success (0) | UNKNOWN (0) | pass, including same-address reimport; 4,096 correct values |

Intel's [NEO-19425 fix](https://github.com/intel/compute-runtime/commit/fa22fc6e08db1a98381917cf46187007c21cf802)
excludes external mappings from the USM reuse cache. The tested newer release
contains it. A [subsequent compatibility change](https://github.com/intel/compute-runtime/commit/2b6562547d7d463848826bf159fec61d0dabbb95)
keeps reporting HOST, whereas the current
[extension specification](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/EXT_ExternalMemMap.html)
calls this HOST_IMPORTED. Therefore the component accepts either type and checks
that metadata is absent after release. An intermediate test requiring only
HOST_IMPORTED failed on the fixed release; that overly strict check was corrected.
Attempts to disable the old release's allocation cache through debug environment
variables did not resolve the reproducer and are not part of the solution.

The component rejects Intel driver builds below **39758** before importing caller
storage. The tested minimum is **26.35.39758.10**; older custom backports are not
qualified. The native version encoding used by this guard is Intel's
`0x01030000 + NEO_VERSION_BUILD`. Owned USM allocation remains usable on the older
driver. The old-driver rejection test passes and confirms no registration record
or imported backing is created. The opt-in native probe bypasses this guard to
retain an independent reproducer of the driver defect.

The newer runtime, GMM 22.10.0 and IGC 2.41.5 were extracted from official release
packages into a dedicated user directory. Package checksums match Intel's release
checksums. Only the test processes select these libraries; system packages were
not replaced. The measured library paths and hashes are in the results file.

## Validation

Hardware: Intel Arc B570, Ryzen 5 5600X, Linux 7.0.0-38-generic, oneAPI compiler
2026.1.1, Level Zero loader 1.32.0. Runtime reports `1.17.39758+10`. Both JIT and
`bmg_g21` AOT tests pass five memory scenarios with **13,464 exact value checks**:

1. Owned host/device allocation, alignment, interior aliases, copies and GPU work.
2. Adjacent unaligned registrations sharing an edge page, four graph replays with
   changed inputs, early release of one slice, preservation of caller storage,
   and eight same-address reimports with changed inputs and pending GPU work.
3. Shared-file mapping with GPU writes, followed by read-only mapping and GPU reads.
4. Free and unregister wait for work on a destroyed stream; another allocation
   proceeds while both releases are waiting. Copied results remain correct.
5. Capture restrictions preserve existing allocations; an allocation by another
   thread leaves a ThreadLocal graph valid.

All nine component CTest programs pass under the new runtime in both builds,
including the quantizer's 546 source-reference cases, the product's 475 cases,
seven runtime-ordering scenarios, graph replay, and the 1,152 CPU/GPU handshakes.
JIT regression disables persistent binary caching. The AOT run includes existing
MMQ binaries plus the rebuilt runtime/memory programs; it tests their execution
on the new driver, not a clean rebuild of every MMQ binary with the new IGC.
All 256 original manifest files remain byte-identical. Test duration is not
inference throughput. Full logs, package/source/binary hashes, expected old-driver
failure and guard rejection are in [runtime-memory-results.json](runtime-memory-results.json).

Reproduce after extracting the official
[compute-runtime release](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10)
and its specified IGC/GMM dependencies to a directory:

```sh
source /opt/intel/oneapi/setvars.sh
strata_driver_dir=/path/to/extracted/26.35.39758.10
export LD_LIBRARY_PATH="$strata_driver_dir/usr/lib/x86_64-linux-gnu:$strata_driver_dir/usr/local/lib:$LD_LIBRARY_PATH"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
cmake --build build-upstream-sycl --target memory_test memory_import_probe
SYCL_CACHE_PERSISTENT=0 ctest --test-dir build-upstream-sycl -V
build-upstream-sycl/memory_import_probe
```

Build configuration follows the preceding component records. For the old-driver
rejection check, select the system libraries and run
`build-upstream-sycl/memory_test --expect-registration-rejected`. The standalone
probe intentionally exits 1 on that driver and keeps any retained mapping's
system storage alive until process exit.

The [CUDA source frontend](runtime-frontend.md) now compiles and tests the original
`PinnedArena`, expert loader and graph manager. Still open: large real-model arena
pin limits and residency behavior, multi-device portable
registration, asynchronous allocators, frontend error/flag compatibility, and
full-model state and speed measurements. These small tests do not qualify all
heap/file mapping kinds, NUMA policy, pinned capacity, or PCIe transfer bandwidth.
