# Actual transfer checks

2026-10-14, Arc B570, current primary LLVM/SYCL9/Level Zero v2 stack.
This adds only an optional standalone target to the primary CMake backend;
the native inference executable's SHA256 remained unchanged.

Main built and tested the proposal after fixing host-source lifetime during
exception unwinding and the asynchronous poison-upload source mutation.
The host lifetime fixture passed all three expected exit/destruction cases.
The initial logged diagnostic passed all 12 cases; the separate quiet check
passed all 156 samples, full payload/corpus/guards and checked cleanup.
Both GPU windows ended normally with no fresh faults or driver dump.

All individual values are in the exact receipt. [results.json](results.json)
groups the three repetitions without discarding variability.

| 256 MiB transfer | Median of the three per-run burst medians, GB/s | Individual per-run medians, GB/s |
| --- | ---: | --- |
| Host-USM to VRAM | 2.73 | 2.7293, 2.7076, 2.7485 |
| Ordinary host to VRAM | 2.25 | 2.2487, 2.2401, 3.0572 |
| VRAM to Host-USM | 5.66 | 5.6624, 5.6630, 5.6612 |
| VRAM to ordinary host | 5.05 | 5.0467, 5.0490, 5.1714 |

Each repetition has four consecutive bursts. Those event rates include the
ordered compatibility event brackets. Host group wall also includes enqueue
and final completion overhead; it is recorded separately. The shared bulk
group wall/counters must not be added four times.

Actual expert extents were obtained from metadata alone: 1,510,400 / 1,971,200 /
2,662,400 bytes. For example, eight copies of the 1,971,200-byte extent from
ordinary host to VRAM achieved host group rates 5.3031 / 5.4888 / 5.4021 GB/s.
The same extent in batch32 achieved 2.5212 / 3.3371 / 2.9521 GB/s. Host-USM
does not win consistently; size, allocation kind, direction and batch differ.
The variable H2D arms need additional focused repetition before a small
performance difference is used to adopt a transfer scheme.

The safety settings include disabled direct submission and disabled Level Zero
v2 copy offload. These results describe that actual stack and configuration.
They do not establish the physical PCIe maximum, a model critical path,
overlap, RAM bandwidth or inference speed. Logical selected model bytes are
also not measured DRAM/PCIe traffic. A single earlier bulk rate cannot price
all current transfers.
