# Standalone host/VRAM bandwidth on the Arc B570

The probe completed after host restart and GPU access was restored. All
35 cases passed, with five measured rounds each and complete final-copy
byte checks. The second pass runs after the user stops the resident embedding service;
it is verified inactive before measurement, and no other Arc GPU clients
are found. The probe itself performs copies without inference or disk reads.
The [first pass](shared-gpu-first/run.json) remains archived with its shared
GPU condition and agrees closely with the repeat. The device is Intel Arc B570 10 GB, with Ryzen 5 5600X
and 128 GB installed RAM, oneAPI 2026.1 and compute driver
1.17.39758+10. The [machine record](machine.json) includes the kernel;
the [run record](run.json) includes driver environment and PCIe metadata.
Source and binary hashes match the earlier [preparation](preparation.json).

Each entry below is decimal GB/s from total copied bytes divided by the
median of five host wall times. Queue submission, completion waits and,
for the staged mode, serial CPU memcpy into the ring are included.

| Copy size | Ordinary RAM → VRAM | Host USM → VRAM | CPU staging + H2D, ring 8 | VRAM → ordinary RAM | VRAM → host USM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.5104 MB expert | 5.961 | 6.288 | 6.125 | 4.155 | 6.570 |
| 2.1760 MB expert | 5.927 | 6.343 | 6.141 | 4.210 | 6.571 |
| 2.6624 MB expert | 5.970 | 6.365 | 6.192 | 4.625 | 6.571 |
| 5.24288 MB residual (128 rows) | 6.024 | 6.428 | 6.261 | 4.758 | 6.572 |
| 167.77216 MB residual (4K rows) | 6.098 | 6.466 | 6.057 | 6.309 | 6.567 |
| 335.54432 MB residual (8K rows) | 6.123 | 6.463 | 5.863 | 6.185 | 6.568 |
| 1 GiB bulk | 5.888 | 6.466 | 5.716 | 6.049 | 6.572 |

At the three native expert sizes, staging plus H2D reaches 6.125–6.192
GB/s. Direct host-USM H2D reaches 6.288–6.365 GB/s, and reaches 6.466
GB/s at 1 GiB. Host-USM D2H reaches about 6.57 GB/s. Thus the engine's
previous expert-DMA observation, about 50.292 GB in 8.1 seconds, is close
to the standalone copy rate. Those engine event durations overlap other
work: they cannot be added to wall time to establish the fraction of
inference spent waiting. The previous 64K/8K-chunk comparison reduced
weight plus host residual bytes from 382.987 to 176.457 GB without an
overall elapsed-time gain. Transfer volume and critical-path waits both
matter for subsequent tuning.

Three upstream bridges report 16 GT/s ×4. The endpoint and its immediate
internal bridge report 2.5 GT/s ×1; those reports do not explain the
measured 6+ GB/s. All raw reports are preserved before and after the run,
without treating the endpoint value as an established physical-link rate.

## Method and reproduction

[probe.cpp](probe.cpp) compares ordinary host memory and `sycl::malloc_host`,
the allocation API used by the engine's expert staging. It also measures
CPU copying into an eight-slot host-USM ring followed by H2D copies. That
mode has one issuer; the engine has a worker pool. Copies use the three
native expert sizes, a 128-row FP32 residual, 4K/8K residual chunks and a
1 GiB bulk transfer. It includes both H2D and D2H, except for the H2D
staging ring.

The probe uses one in-order queue with event profiling. H2D source offsets
rotate within a resident 1 GiB host arena. Each case has a validated
warmup and five measured rounds. Allocation, source initialization and
complete final-copy byte comparison are outside the timers. The check
covers the final overwritten destination, rather than separately checking
every intermediate copy. Event and wall durations are both recorded;
event timing is not an independent measurement of the electrical link.
The staging buffers remain alive until DMA completes, and a slot is not
overwritten while in use.

Compile with:

```sh
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -O2 -std=c++17 -fp-model=precise probe.cpp -o pcie-bandwidth
```

[controller.py](controller.py) records the selected driver environment,
[raw output](stdout.jsonl), all 175 rounds, 35 medians and PCIe metadata
before and after the run. Its paths are specific to this benchmark
workspace. It refuses to start while the known blocked inference thread
remains or another inference measurement runs. Run the probe without an
inference or disk workload.
