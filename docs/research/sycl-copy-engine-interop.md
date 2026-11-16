# SYCL and Level Zero copy-engine interop reference

This source collection maps possible Level Zero queue routes for explicit copy-engine placement. It is a contract and retrieval reference, not a qualified implementation, hardware-attribution result, or speed result.

## What the captured SDK establishes

The installed oneAPI DPC++ 2026.2 Level Zero backend header declares queue interop from a native Level Zero command queue or command list. Its make_queue wrapper passes the native handle, SYCL context and device, ownership choice, properties, and async handler into the runtime. The paired header defines ownership values keep and transfer. This establishes an API surface; it does not establish that the installed Unified Runtime adapter can safely import and operate on an externally created list.

The installed queue-properties header also declares sycl::ext::intel::property::queue::compute_index. This is a compute-index property, not a copy-engine selector. Current upstream extension documentation says compute_index is built into the native queue/list and cannot be set among make_queue properties. That documentation is a moving reference, not proof for the captured runtime.

The Level Zero header exposes copy-capable queue-group flags and the immediate-list descriptor's group ordinal and index. Those fields let an application request a logical placement. A descriptor value does not prove which physical engine executes the work.

## Candidate routes and their contracts

**SYCL Level Zero interop** is the narrowest candidate when a SYCL queue must wrap an explicitly created native immediate command list. The current upstream backend extension says the v2 adapter accepts only an in-order command list and that the native object carries its compute index. Use a context and device compatible with the existing SYCL runtime. Cross-queue order is not implied by an in-order queue; bridge producer and consumer dependencies explicitly, preferably with SYCL events where supported.

The ownership choice must have one owner. With keep, the application retains and destroys the native handle after the last SYCL queue copy is gone. With transfer, SYCL owns destruction and the application must stop using the handle. Keep the context, device, events, list, and host bytes valid through completion. Check native return codes and drain work before teardown. A wrapped list does not prove equivalent event profiling, adapter queue tracking, async error reporting, or completion behavior.

**Runtime queue properties and environment controls** describe runtime-selected work, not a documented per-queue CE selector. The current environment-variable reference labels SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE as legacy and describes it for SYCL buffer/image transfers. It does not prove behavior for this installed v2 adapter or under UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1.

**Raw Level Zero submission** gives more direct control over requested group/index, with greater responsibility for SYCL interoperation. The captured Level Zero API does not settle pageable-host-pointer eligibility, host registration/visibility, source lifetime, event bridging, or error/drain behavior for the desired path. Ordinary host memory being legal for standard SYCL memcpy does not establish those raw-API contracts.

## Installed-version and measurement limits

The captured runtime observation identifies oneAPI DPC++ 2026.2, Intel Arc B570, and libur_adapter_level_zero_v2.so.0.12.0, with UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 and EnableDirectSubmission=0. Current upstream documentation and implementation links below are not version-matched evidence for that adapter. Previous CX curation identifies the installed Level Zero runtime as 1.17; latest specification links do not prove its behavior.

A Q1 trace captured successful immediate-list creation requests for ordinal 0/index 0. This records requested descriptor values only. It does not attribute the copies to a physical engine, prove copy-engine saturation, or show a performance loss. Queue placement and timing are separate questions.

Before considering an implementation trial, retrieve an immutable, license-checked source revision or vendor document matching UR adapter 0.12.0 for native-list import, ownership, synchronization, events, profiling, and errors. If measured results later require engine attribution, establish the device topology and observe actual command execution. Compare the complete qualified path; do not infer benefit from API availability or a logical queue descriptor.

## Pinned project sources

These exact commit links document the reviewed Strata snapshot and original runtime. They are source context, not adoption evidence.

- [Copy lane](https://github.com/Niko1221/Strata/blob/148e6dac73b8cc964503be9decef0e6912abc8d4/include/strata/prefill/sycl_copy_lane.hpp)
- [Upload event lane](https://github.com/Niko1221/Strata/blob/148e6dac73b8cc964503be9decef0e6912abc8d4/include/strata/prefill/upload_event_lane.hpp)
- [Full-math SYCL copy fixture](https://github.com/Niko1221/Strata/blob/148e6dac73b8cc964503be9decef0e6912abc8d4/bench/sycl/prefill_sycl_copy_math.cpp)
- [Original SYCL runtime implementation](https://github.com/Niko1221/Strata/blob/14508acbf1b70eb994fdf37f97fb6d0f5cff7d43/src/sycl/runtime.cpp)
- [Original SYCL runtime interface](https://github.com/Niko1221/Strata/blob/14508acbf1b70eb994fdf37f97fb6d0f5cff7d43/include/strata/sycl/runtime.hpp)

## Installed header capture

The following hashes identify the captured local SDK declarations; they are not source commits:

- DPC++ 2026.2, /opt/intel/oneapi/compiler/2026.2/include/sycl/ext/oneapi/backend/level_zero.hpp — SHA256 30f86adea4a4be42714a6bef7b637b5a305afcb272cfcc31ad2547c6dd13de4b
- DPC++ 2026.2, /opt/intel/oneapi/compiler/2026.2/include/sycl/ext/oneapi/backend/level_zero_ownership.hpp — SHA256 082d86b5b159c4f17eb62ef549fd3d6afbbacfa80e7e3fe9b31958c2e3bd9080
- DPC++ 2026.2, /opt/intel/oneapi/compiler/2026.2/include/sycl/properties/queue_properties.hpp — SHA256 19377e14ba0c5704b4de949ce1b04d536e012c43e625fba3270a0b53e20d27e4
- DPC++ 2026.2, /opt/intel/oneapi/compiler/2026.2/include/unified-runtime/ur_api.h — SHA256 2c757749049df95a8794b48e7c2a651d79792d170636373d3ce626caa4af4549
- Level Zero, /usr/include/level_zero/ze_api.h — SHA256 095a0eb442ae21946f29147c209f270b93dd0eb5b12a2d0c67e5f4138b533712

## Primary references

These upstream URLs are version-moving reference material. Check the exact installed release before using them as a compatibility contract.

- [Intel SYCL Level Zero backend extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md)
- [Intel SYCL environment variables](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md)
- [Unified Runtime Level Zero adapter example](https://github.com/oneapi-src/unified-runtime/blob/main/source/adapters/level_zero/image.cpp)
- [Level Zero command queues](https://spec.oneapi.com/level-zero/latest/core/PROG.html#command-queues)
