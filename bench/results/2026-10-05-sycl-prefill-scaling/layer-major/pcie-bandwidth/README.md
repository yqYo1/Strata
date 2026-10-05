# Standalone host/VRAM bandwidth probe

Status: compiled successfully with oneAPI 2026.1. GPU execution is pending
recovery from the xe kernel fault. No standalone bandwidth result is
claimed. The [preparation record](preparation.json) preserves the source
and binary hashes and the reported PCIe path.

[probe.cpp](probe.cpp) compares ordinary host memory and `sycl::malloc_host`,
the allocation API used by the engine's expert staging. It also measures
CPU copying into an eight-slot host-USM ring followed by H2D copies. That
mode has one issuer; the engine has a worker pool. Copies use the three
native expert sizes, a 128-row FP32 residual, 4K/8K residual chunks and a
1 GiB bulk transfer. It includes both H2D and D2H, except for the H2D
staging ring.

The probe uses one in-order queue with event profiling. H2D source offsets
rotate within a resident 1 GiB host arena. Each case has a validated
warmup and five measured rounds. Queue completion is included in host
wall time. Allocation, source initialization and complete final-copy byte
comparison are outside the timers. Event durations and wall durations
are both recorded; event timing is not an independent measurement of
the electrical link. GB/s uses decimal bytes. The staging buffers remain
alive until DMA completes, and a slot is not overwritten while in use.

Compile with:

```sh
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -O2 -std=c++17 -fp-model=precise probe.cpp -o pcie-bandwidth
```

[controller.py](controller.py) records the selected driver environment,
raw output, all 175 rounds, 35 medians and PCIe metadata before and after
the run. Its recorded paths are specific to this benchmark workspace.
It refuses to start while the known blocked inference thread remains or
another inference measurement runs. Run the probe without an inference
or disk workload.
