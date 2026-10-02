# B570 USM atomic support: driver and kernel investigation

Investigated on 2026-10-02 on the B570 / Ryzen 5 5600X machine described in
[SYCL_RESEARCH.md](SYCL_RESEARCH.md). This supersedes the initial suggestion
that false USM atomic aspects alone establish a hardware restriction.

## Findings

The false `usm_atomic_host_allocations` and `usm_atomic_shared_allocations`
aspects originate in the user-mode driver's allocation-specific capability
policy. They do not establish an inherent inability of B570 to synchronize
with the CPU. The inspected installed and newer driver versions both report
false by default. The kernel already exposes the faulting/system-memory path.
No supported configuration change that turns those aspects on and gives a
working low-latency handoff was established.

Ordinary system allocations are a separate path: a small two-way atomic
publication test passed without overrides. On this machine it was much slower
than event-based access to host USM in the particular tests below. This supports
an event-based baseline for Strata; it does not establish an optimal final
implementation or an inference performance ratio.

**An internal KMD-migration override was also tested and caused a failure.**
It changed the advertised shared capability but SYCL queue creation did not
complete. Subsequent normal probes reported device loss and a glibc failure;
the kernel logged job timeouts and automatic GPU resets. All test processes
ended, and no permanent settings were changed. Further execution tests were
stopped. `xpu-smi` still listed the device as normal, but a later minimal
default-setting health check (one device integer, one kernel writing 42, one
copy back) also reported `UR_RESULT_ERROR_DEVICE_LOST` and timed out after ten
seconds. Compute recovery was **not** established. Do not repeat the override test on
a working machine or treat its capability changes as a usable configuration.

## Versions and isolation

| Component | Inspected or tested version |
| --- | --- |
| Running kernel | Ubuntu `7.0.0-28-generic`, `7.0.0-28.28~24.04.1` |
| Installed, not booted kernel | `7.0.0-38-generic`, `7.0.0-38.38~24.04.4` |
| Installed Intel compute driver | `26.31.39395.14-1~24.04~ppa1`, Kobuk Intel graphics PPA |
| Level Zero loader | `1.32.0-1~24.04~ppa1` |
| gmmlib | `22.10.1-1~24.04~ppa1` |
| Compiler | oneAPI 2026.1.1 |
| Newer driver loaded for capability queries | Official `libze-intel-gpu1_26.35.39758.10-0_amd64.deb` |
| Additional driver source inspected | Tag `26.35.39758.13`; master `724697d73af22685918fcad4575c6bc267589ca5` |
| Intel LLVM/UR source reference | `16fff99e152f8485cda1905fcd9bbd18b347849c` |

The newer driver package was extracted with `dpkg-deb -x` under
`/tmp/strata-sycl-research/driver-26.35`, not installed. A child process's
`LD_LIBRARY_PATH` selected its library. `LD_DEBUG=libs` confirmed loading
`/tmp/strata-sycl-research/driver-26.35/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1`
and the system gmmlib. Its SHA256 matched the official release:
`c19a641b953d55aebbf1d51bec364a84bf629f985e02fbbe6dc70224c0e88470`.

This was an isolated capability-query comparison using existing dependencies,
not a complete driver-stack upgrade or an inference test on the newer release.
GitHub marked `.10` as the latest packaged release; `.13` was a newer source
tag. APT's existing cache listed installed `.14` as its candidate. No package
installation, reboot, BIOS change, module reload or persistent environment
change was made.

The compute-runtime repository was obtained with `ghq` after checking existing
repositories. Its root remains clean on `master`; older tags were read using
`git show` / `git grep`, without switching the root branch.

## Capability decision chain

The inspected LLVM source maps both SYCL atomic aspects to UR's
`ATOMIC_CONCURRENT_ACCESS` flag, not merely `ATOMIC_ACCESS`. UR's Level Zero
adapter maps the driver's `CONCURRENT_ATOMIC` bit directly. Thus a device may
support GPU-side atomics while these two SYCL aspects remain false.

At compute-runtime tag `26.31.39395.14`:

1. `level_zero/core/source/device/device.cpp`, `getMemoryAccessProperties`,
   obtains host and shared capabilities separately from `ProductHelper`.
2. `shared/source/os_interface/product_helper.inl`, `getHostMemCapabilities`,
   starts with access plus atomic access. Concurrent bits require the explicit
   concurrent-access override; there is no BIOS/kernel feature query in this
   concurrent-bit decision.
3. `getSingleDeviceSharedMemCapabilities` adds concurrent bits when
   `isKmdMigrationAvailable` is true or the concurrent-access override is set.
4. `DrmMemoryManager::isKmdMigrationAvailable` calls
   `Drm::hasKmdMigrationSupport` in `shared/source/os_interface/linux/drm_neo.cpp`.
   Its normal result is `hasPageFaultSupport() && isKmdMigrationSupported()`.
5. `ProductHelper::isKmdMigrationSupported()` returns **false**. A repository-wide
   search of `shared/source` at this tag found no product specialization that
   enables it, including BMG. Kernel fault support alone cannot make it true.

The packaged newer `.10` and source tag `.13` retain that decision. On master,
commit `fafb95ee8d5828f064c68826294ed41d7313581d` removed the obsolete product
helper and simplified the default result to `false`; the override remains.
This is not a newly enabled BMG feature in the versions inspected.

Actual query results with default settings:

| Allocation category | Installed `.14` flags | Newer `.10` flags |
| --- | ---: | ---: |
| Host | 3 | 3 |
| Device | 3 | 3 |
| Shared, single device | 3 | 3 |
| Shared, cross device | 0 | 0 |
| Shared system | 15 | 15 |

Bits: 1 = read/write, 2 = atomic, 4 = concurrent, 8 = concurrent atomic.
SYCL with the newer library also reported `system=1`, `atomic_host=0`,
`atomic_shared=0`, and system atomic scope available.

This identifies a software implementation/default-policy restriction, not a
missing setting that has been shown to restore the desired behavior. It also
does not exclude a future driver implementation changing the result.

## Kernel and BIOS-related conditions

The distinct **shared-system** path is enabled by all of the following:

- Linux BMG's product helper opts into shared-system USM and its hardware info
  advertises all four memory-access bits.
- `IoctlHelperXe::queryDeviceIdAndRevision` requires the kernel's
  `DRM_XE_QUERY_CONFIG_FLAG_HAS_CPU_ADDR_MIRROR`; disabling recoverable faults or
  shared-system USM through driver debug settings blocks that path.
- `Drm::adjustSharedSystemMemCapabilities` checks CPU versus GPU virtual-address
  widths. Its `no5lvl` diagnostic concerns a CPU address range exceeding the
  GPU's; this is not the current machine's situation (48-bit CPU virtual space).
- Upstream Linux v7.0's `xe_query.c` exposes CPU address mirroring when the GPU
  has USM and `CONFIG_DRM_XE_GPUSVM` is enabled.

On this host both the running and installed newer kernel configs have
`CONFIG_DRM_XE_GPUSVM=y`, `CONFIG_DRM_XE_PAGEMAP=y`, `CONFIG_HMM_MIRROR=y`,
`CONFIG_DEVICE_PRIVATE=y` and `CONFIG_MMU_NOTIFIER=y`.
The actual kernel ioctl on the B570 render node returned config flags `0xf`,
including CPU address mirroring. Creating and destroying a temporary
`LR_MODE | FAULT_MODE` GPU VM also succeeded. Therefore this kernel is not
missing those prerequisites.

The command line is `root=ZFS=... ro init_on_alloc=0 amd_pstate=active` besides
the boot-image path. No relevant USM overrides were found in the process
environment. Reading `xe` module parameters `force_probe` and
`svm_notifier_size` was denied by permissions; their values were not inferred.
Their presence or absence does not bypass the user-mode driver's false
KMD-migration default. No evidence was found that ReBAR, IOMMU or an ordinary
BIOS toggle changes the host/shared concurrent capability decision above.

Booting the already installed newer kernel might affect bugs or performance,
but it was not tested. It cannot, by itself, alter the hardcoded default in the
inspected user-mode driver. The kernel source analysis uses upstream v7.0;
Ubuntu's exact patchset was not audited in full, so runtime ioctl results are
the evidence for the installed kernel's capabilities.

## Internal overrides: reporting versus functionality

These are observations of diagnostic experiments, not recommended settings.
Only individual child processes received the variables.

| Experiment | Result |
| --- | --- |
| Unprefixed override names alone | No change in the installed release's capability query |
| `NEOReadDebugKeys=1`, `NEO_EnableUsmConcurrentAccessSupport=5` | Host and single-device shared flags change from 3 to 15 |
| `NEOReadDebugKeys=1`, `NEO_UseKmdMigration=1` | Single-device shared flags change from 3 to 15; host remains 3 |
| SYCL execution with the KMD-migration override | Did not complete; instrumented run stopped at queue creation, before mailbox allocation or the test kernel |

The concurrent-access override is referenced by the capability helper; it does
not supply the missing coherence/migration implementation. The KMD-migration
override additionally changes shared-allocation machinery in
`unified_memory_manager.cpp` from the user-mode page-fault path to
`createUnifiedKmdMigratedAllocation`. It is therefore not just a reporting knob,
but the forced path failed in this environment.

The first forced execution hit a 20-second timeout. A second instrumented run
printed only `stage=queue-create` and required the timeout's kill after 12+2
seconds. Later ordinary event/system tests returned
`UR_RESULT_ERROR_DEVICE_LOST` and a glibc `sysmalloc` assertion respectively.
Kernel messages at 13:24:52 JST included kernel-submitted job timeouts and
automatic resets. No causal diagnosis beyond this sequence was established;
these failures must not be described as a working workaround or blamed on the
ordinary system-memory probe alone. No successful post-failure compute run was
obtained, including the later minimal health check. Further GPU tests need a
known-good recovered device first. No reboot, manual device reset or driver
unbind was performed, because those can interrupt other work on this host.

## Handoff measurements before the forced-path failure

Standalone programs are in `tools/sycl/system_handoff.cpp` and
`tools/sycl/event_handoff.cpp`. They ran on the installed default driver and
kernel before the override experiment. Each performs 16 round trips, checking
16 changing `uint32_t` payload words in each direction (64 bytes each way).

The system test uses one 4-KiB-aligned ordinary allocation, host
`std::atomic_ref`, and GPU system-scope `sycl::atomic_ref` with release/acquire
publication. One GPU kernel serves all rounds; the host yields while polling.
It gates execution on the Level Zero shared-system concurrent-atomic bit and
the SYCL system-allocation aspect. GPU polling is bounded at 1,000,000 loads per
round, and host polling has a five-second deadline. These limits do not protect
against hangs inside driver initialization or kernel faults; use an external
timeout as well and never combine the test with forced capability overrides.

The event test uses `malloc_host`, alternating CPU/GPU access with
`wait_and_throw`. It submits 17 small kernels: each validates the preceding CPU
response and publishes the next GPU payload, with a final validation kernel.
There is no explicit bulk payload memcpy in this path. Both tests copy a small
device result back after completion.

| Run | System atomic, 16 rounds | Host USM with events, 16 rounds |
| --- | ---: | ---: |
| First (includes first-use overhead) | 226.503 ms | 29.7562 ms |
| Second | 121.252 ms | 0.215877 ms |
| Third | 120.900 ms | 0.214603 ms |

All these completed runs reported zero host and GPU payload errors; the system
test confirmed 16 completed rounds on both sides. The two warm runs are about
7.6 ms/round versus 13.4 us/round including submission/completion overhead.
These are tiny synthetic tests, not a broad atomic stress test, a best-case
latency bound, a full Graph test or an inference speed prediction. Both control
directions and payloads share one page in the system test; separating them or
using a different allocation/placement strategy was not benchmarked. Kernel and
submission topology differ deliberately; the timing identifies a poor default
doorbell candidate, not the isolated cost of one atomic instruction.

The result is consistent with the upstream kernel implementation:
`xe_vma_need_vram_for_atomic` requires VRAM for discrete-GPU global/unspecified
atomic accesses, and the SVM fault handler may migrate pages and use an atomic
timeslice. `xe_device.c` initializes that timeslice to 5 ms. CPU access can then
require migration back. This is an explanation supported by source, not a
measured attribution: GPU page-fault/migration counters were not captured.
GPU-only atomics on system memory are a different case from global CPU/GPU
atomic sharing and do not invalidate this distinction.

## Implementation consequence

Proceed with a typed synchronization boundary whose initial implementation uses
events and host-USM or explicit DMA staging, retaining concurrent CPU experts
and independent GPU work. Do not require turning the two atomic aspects true.
Do not put ordinary system-USM spin flags on every layer's critical path based
only on their advertised support.

Keep a future specialized handoff implementation possible. Before adopting one,
measure changing-payload visibility, bounded forward progress, graph replay,
CPU/GPU overlap and actual layer timings on this machine. The current evidence
does not justify driver debug overrides, an immediate OS upgrade for these
flags, or assuming that an atomic doorbell will outperform event synchronization.

## Reproduction and references

The two successful-path programs can be compiled without running the engine:

```sh
source /opt/intel/oneapi/setvars.sh
icpx -std=c++20 -fsycl -O2 -I/usr/include tools/sycl/system_handoff.cpp -lze_loader -o /tmp/system-handoff
icpx -std=c++20 -fsycl -O2 tools/sycl/event_handoff.cpp -o /tmp/event-handoff
```

They were run with `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` and an external
20-second timeout. Do not run them on a device still recovering from the failed
override. The forced-path variant is not included as a recommended runnable
recipe. The formatted committed sources were compiled again after the failure,
but the handoff benchmarks were not repeated again on the degraded device.

- [Installed-tag capability helper](https://github.com/intel/compute-runtime/blob/26.31.39395.14/shared/source/os_interface/product_helper.inl)
- [Installed-tag DRM decisions](https://github.com/intel/compute-runtime/blob/26.31.39395.14/shared/source/os_interface/linux/drm_neo.cpp)
- [Installed-tag allocation paths](https://github.com/intel/compute-runtime/blob/26.31.39395.14/shared/source/memory_manager/unified_memory_manager.cpp)
- [Installed-tag Linux BMG product helper](https://github.com/intel/compute-runtime/blob/26.31.39395.14/shared/source/xe2_hpg_core/linux/product_helper_bmg.cpp)
- [Default-false simplification](https://github.com/intel/compute-runtime/commit/fafb95ee8d5828f064c68826294ed41d7313581d)
- [Packaged newer release and SHA256](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10)
- [SYCL aspect mapping](https://github.com/intel/llvm/blob/16fff99e152f8485cda1905fcd9bbd18b347849c/sycl/source/detail/device_impl.hpp)
- [UR Level Zero mapping](https://github.com/intel/llvm/blob/16fff99e152f8485cda1905fcd9bbd18b347849c/unified-runtime/source/adapters/level_zero/common/device.cpp)
- [Kernel CPU-address-mirror query](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/xe/xe_query.c)
- [Kernel atomic migration decision](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/xe/xe_vm.c)
- [Kernel SVM fault handling](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/xe/xe_svm.c)
- [Kernel atomic timeslice default](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/xe/xe_device.c)
