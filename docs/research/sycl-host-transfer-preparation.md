# SYCL host-to-device preparation: source-audit delta

**Use:** retrieve only for a Main-owned host-transfer comparison after the current path's CPU attribution identifies a decision. This is a source audit and measurement-method note; no adapter branch or B570 transfer mechanism was measured.

## Current Strata path

The immutable private Arena view bypasses Strata's layer-wide application repack. `stage_expert_bank()` passes the expert-range pointer into `ExpertCache::fill_slot()`, which calls the compatibility `cudaMemcpyAsync`; SYCL compatibility submits `queue.memcpy()` and records the event without an explicit wait. This proves the caller passes the Arena pointer. SYCL permits ordinary host pointers for explicit copy operations when valid, but neither the API nor this source proves that the adapter avoids an internal pinned staging copy, synchronous preparation, or registration lookup.

The SYCL `PinnedArena` keeps the large Arena in ordinary RAM, ignores the slice/cap arguments used by registering backends, and attempts an OS residency lock unless disabled. Residency locking is not host-USM allocation or Level Zero registration. In this compatibility layer, `cudaHostRegister`/unregister reject arbitrary mappings; `cudaHostGetDevicePointer` aliases only host USM. `cudaHostAlloc` maps to `sycl::malloc_host`, but the expert upload source above does not use it. Thus bounded host-USM expert staging or importing the Arena is a candidate path, not a current capability.

The frozen comparison reports 40,102,860,800 requested H2D payload bytes and 4.94585 s wall / 4.88016 s CPU in the upload-submission region. That region includes the wrapper; it cannot separate CPU packing, runtime staging/registration, polling, device copy service or physical PCIe bytes. Do not turn requested payload into wire bandwidth.

## Decision gates if a comparison becomes eligible

Compare (A) the current direct Arena view, (B) a bounded reusable host-USM slab with the full Arena-to-slab CPU copy included, and (C) existing-range import only if the active runtime exposes and supports it. Preserve same-queue ordering and the workstation's qualified copy-offload/direct-submission settings. Bound staging by in-flight chunks and host-memory headroom; include allocation/import and release costs, actual accepted/pinned bytes and failures.

Separate caller preparation, `queue.memcpy` call, explicit/implicit completion, device copy service, and request critical-path wall time. Report thread CPU separately from wall time and polling; do not add overlapping queue spans. Keep first-use import/allocation distinct from warm requests and independent processes. Requested bytes remain logical when physical counters are unavailable. A transfer microbenchmark is not request-throughput evidence.

### Registered source and sparse runs

The Level Zero external-memory mapping extension, introduced in 1.14, requires both the system pointer and size to be host-page aligned. Use the OS page size. Set `ZE_HOST_MEM_ALLOC_FLAG_MEM_READ_ONLY` for read-only backing. Keep the source alive until the copy list has drained and `zeMemFree` releases the mapping. Assign each imported page range to one owner. Releasing overlapping imports can invalidate device access to the other range. Treat file mappings and other backing outside the specified storage classes as provider-dependent. [Level Zero external-memory mapping v1.21.26](https://oneapi-src.github.io/level-zero-spec/level-zero/1.21.26/core/EXT_ExternalMemMap.html)

The B570 admission check at `37017156e` observed `HOST=1` for the tested source with zero application copy or kernel commands. Validate the actual provider and allocator backing for each new import route. Copy service, physical traffic and model speed require separate evidence. See `prefill_pinned_host_contract_cl_20261116` and `prefill_mapped_source_contract_cj_20261116` in the [research index](../../bench/results/2026-10-15-pr6-recovery-audit/research-index.json).

For a sparse-copy batching comparison, begin with exact adjacency in both the source and destination, with the same source owner, format and useful bytes. Measure the reduction in commands and the effect on first-ready time. Treat copying across gaps or adding packing for scattered destinations as separate changes that add work. Check alignment after page rounding before importing neighboring ranges.

## Provenance and primary references

This note summarizes local source paths in the assigned host-transfer report; no code was copied. Current online specifications are not pins of the installed compiler/runtime, and the report establishes no active UR branch. Intel transfer/import advice is vendor-stack-specific; FPGA pinning and NVIDIA pageable staging supply only transferable cost examples.

- Khronos [SYCL `handler::memcpy`](https://github.khronos.org/SYCL_Reference/iface/command-group-handler.html) permits host pointers or accessible USM pointers for the copy operation.
- Intel [Optimizing Data Transfers](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/optimize-sycl-data-transfers.html) describes host USM and experimental import/release; it is guidance, not evidence of this B570 path.
- Unified Runtime [Level Zero controls](https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html) list host-pointer import disabled by default in that reference; confirm the installed version and child environment before applying it.
- Level Zero [external-memory mapping](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/EXT_ExternalMemMap.html) requires page-aligned ranges, valid backing lifetime and nonoverlapping mappings; support for nonstandard backing is platform/driver-dependent.
- Unified Runtime [experimental USM import/release](https://oneapi-src.github.io/unified-runtime/core/EXP-USM-IMPORT-RELEASE.html) is explicitly experimental.

Retrieval tags: `host-arena-direct-view`, `sycl-queue-memcpy`, `ordinary-host-pointer`, `host-usm-staging`, `usm-import-lifetime`, `requested-not-wire-bytes`, `same-queue-safety`.
