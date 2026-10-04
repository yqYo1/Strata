# Original device admission and arena

The whole original `src/core/device.cu` and `src/core/device_main.cpp` now build
and run with explicit native SYCL metadata. The whole original
`src/program/generate.cpp` also compiles to an object with native metadata
bindings and native experts, MMQ and fused declarations enabled. It is not yet
a linked or executed inference engine. The 256 pinned original files remain
byte-identical. [Recorded evidence](native-device-results.json) contains source,
generated-file, object, binary, SDK-header and loaded-library hashes and logs.

## Native metadata and admission

`DeviceFacts` reads the visible device's native name, architecture enum, driver,
device and platform version strings, global/local memory, compute units, maximum
clock, workgroup/subgroup limits and FP16/matrix/USM aspects. Ordinal inspection
uses the existing frontend's error handling without changing the current device.
Failed queries leave their output unchanged. Free VRAM continues to use the
[native Sysman binding](runtime-devices.md).

The original public `DeviceInfo` receives the native name, architecture and
memory. Its CUDA-only compute-capability, multiprocessor and integer version
fields remain zero. Native compute units and version strings have explicit
fields in `DeviceFacts`. No CUDA property struct, fabricated compute capability,
CUDA runtime version or HIP compiler macro is supplied.

Admission checks the actual architecture against this binary's AOT target,
where present. It also checks Intel Level Zero, FP16/matrix/USM support,
subgroups 16 and 32, workgroups of at least 1024 and at least 16,640 local bytes.
These are requirements of the components bound so far; this check does not
establish that every remaining original kernel can run.

The original poison kernel has a named SYCL kernel identity. `device_code_error`
queries its executable bundle and its availability on the selected device. It
does not launch the initializer. A successful query can compile an image at
runtime; it is not a throughput measurement. This currently establishes image
availability for that one bound kernel. The original CUDA comment that its
image stands for every engine translation unit does not yet apply to this port.
The [SYCL kernel-bundle reference](https://github.khronos.org/SYCL_Reference/iface/kernel-bundles.html)
defines executable-image queries. Architecture identity uses the installed SDK's
[experimental architecture extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc);
its interface can change between SDK versions. The exact installed headers are
hashed in the evidence.

## Original processing retained

The arena retains its original single allocation, bump offsets, power-of-two
alignment checks, capacity refusal, destructor and poison launch loop. The
initializer retains its original `0x7fc00000` float bits and boundary predicate.
The generated launch captures the original arguments on the host and uses the
existing stream frontend. It is a native kernel, not a host fill.

The generated diagnostic program retains the original options, device listing,
memory plan and 64 MiB selftest. Display labels use native architecture, compute
units and version strings. The original CLI selftest checks alignment and
capacity refusal; it does not read poison values back. A separate test does.

The whole generation driver retains the planner, feature branches and engine
calls. Four property sites use native facts: split-owned-buffer capacity,
skip-if-fits display, startup admission and the limited-stage warning. The
automatic layer-split hardware query uses actual native compute units and MHz.
Its original `0.33 * (84 * 2.617) / speed` formula remains CUDA-calibrated, and
the generated diagnostic says so. This is an unvalidated estimate on Intel GPUs.
Multi-GPU execution has not been measured here.

The driver target is an object library. No stub engine definitions or linker
section deletion make it into a working executable. Its unresolved symbols are
recorded for the remaining integration work. The older unchanged-source host
syntax census remains its historical 19/20 checkpoint; the generated native
driver compile is separate evidence.

## Validation and reproduction

Measured on Intel Arc B570 / Ryzen 5 5600X, oneAPI 2026.1.1, compute-runtime
26.35.39758.10, IGC 2.41.5, GMM 22.10.0 and Linux 7.0.0-38. The GPU reports
`intel_gpu_bmg_g21`, 160 compute units, a 2750 MHz maximum clock and
10,666,115,072 global-memory bytes. These are native query values.

Both spir64 JIT and `bmg_g21` AOT pass 25 related CTest programs. Four new
programs cover native metadata/arena/image admission, original CLI listing,
original CLI selftest and wrong-architecture rejection. The core test checks
seven scenarios, 1,065,218 exact NaN words and ten invalid queries. It includes
partial workgroups, allocations not divisible by a float, bump offsets,
suballocation copies, unchanged outputs on query errors and zero captured nodes
around image admission on a nonblocking stream. An existing buffer remains
unchanged by admission.

The architecture rejection fixture compiles the metadata helper with the real
`intel_gpu_pvc` enum while running on the actual B570. It has no PVC device
image; it tests the metadata admission branch. The error names both the real
card/architecture and the mismatched target.

The previous GEMM, native product and native state/attention tests also pass
after the frontend-header rebuild. Their generated source hashes are unchanged
from the [native primitive checkpoint](native-primitives.md). The broad suite
does not repeat standalone large MMQ or historical real-model dequant samples.

```sh
source /opt/intel/oneapi/setvars.sh
export LD_LIBRARY_PATH=/home/yayoi/.local/opt/strata-compute-runtime-26.35.39758.10/usr/lib/x86_64-linux-gnu:/home/yayoi/.local/opt/strata-compute-runtime-26.35.39758.10/usr/local/lib:$LD_LIBRARY_PATH
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0
cmake -S tests/sycl_upstream -B build-upstream-sycl -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_GGML_DIR=/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned
cmake --build build-upstream-sycl -j3
ctest --test-dir build-upstream-sycl -V \
  -R 'original_|mmq_public_stages|mmq_context_lifetime|runtime_'
build-upstream-sycl/strata_device_sycl --selftest
```

Use a separate build directory with `-DSTRATA_SYCL_DEVICE_ARCH=bmg_g21` for AOT.
Initial build failure from the missing GGML include path and an invalid test
attempt to capture the legacy default stream are retained in the evidence.
The corrected test uses a supported nonblocking capture stream. Existing AOT
product register-spill warnings and the original missing-override warning remain
in the build logs. The final suites took 9.81 seconds JIT and 9.68 seconds AOT;
these correctness-test durations are not inference benchmarks.

Remaining work includes the other original kernel families, native binding of
the NVIDIA tensor score algorithm, fixed BLAS workspace binding, CPU ISA
bindings, full engine linking and whole-model state validation. Windows, other
devices and multi-GPU execution remain untested. PP1000/TG70 remain unachieved.
