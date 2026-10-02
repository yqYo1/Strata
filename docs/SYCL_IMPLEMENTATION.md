# SYCL backend implementation

The backend is under development. The runtime and device arena run on Intel Arc
B570; the SYCL build does not yet provide the `strata` inference executable.
The implementation follows [the port research](SYCL_RESEARCH.md) and
[the operation inventory](SYCL_BATTLEMAGE_OPERATIONS.md).

## Build and validate

On Linux with Intel oneAPI DPC++ installed:

```sh
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
cmake -S . -B build-sycl -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_ENABLE_SYCL=ON -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF
cmake --build build-sycl --target strata-device sycl_runtime_test -j2
ctest --test-dir build-sycl -R '^sycl_' --output-on-failure
build-sycl/strata-device --list-devices
```

`STRATA_SYCL_TESTS` controls the independent SYCL tests. The native CPU expert
library is disabled in this runtime-only command to avoid fetching ggml before
the inference path is connected. CUDA, HIP and SYCL are mutually exclusive.

For the separate large-address check, with at least 5 GiB of free VRAM:

```sh
timeout 60 build-sycl/sycl_runtime_test --large
```

This allocates 4 GiB + 64 KiB, writes different 64-bit values below and above
the 4 GiB boundary, and reads both back. It does not measure memory bandwidth.

## Runtime contract

- Compute and transfer queues use the same device and context. Each queue is
  ordered; dependencies between queues use explicit events.
- CPU access to host USM follows completion of its producing GPU command. No
  concurrent CPU/GPU atomic access to host/shared USM is required.
- Device and bounded host-staging allocations own their context lifetime and
  complete pending work before freeing storage. Bulk expert RAM will remain
  ordinary host memory rather than a shared-USM allocation.
- Asynchronous errors are retained and rethrown by runtime checks and waits.
  A timeout reports incomplete work; it does not cancel commands or authorize
  releasing their buffers.
- `DeviceArena` preserves the engine's bump-allocation interface, with checked
  arithmetic and alignment of actual pointers. Memory planning queries free
  VRAM rather than treating total capacity as available memory.

## Validation recorded on 2026-10-02

Intel Arc B570, DPC++ 2026.1.1, Level Zero driver `1.17.39395+14`, normal driver
settings; see the research document for the machine configuration:

- 64 changing-input CPU → transfer queue → compute queue → CPU round trips,
  each with 4,097 integer elements: all results matched.
- Invalid subranges, impossible device allocation size and move ownership:
  expected rejection/ownership behavior.
- A deliberately throwing host task: the asynchronous error reached the caller
  and remained observable on subsequent checks.
- Bounded completion wait and the existing `strata-device --selftest`: passed.
- Distinct 64-bit writes below and above 4 GiB: both values matched.

These checks establish runtime behavior, not model correctness or inference
performance. Remaining work includes the model kernels, CPU expert scheduling,
prefill, cache/sequence state, verification and launch integration, followed by
layer/logit comparisons and real-model inference validation.
