# Dequantization work-group probe on Arc B570

The unchanged engine dequantization formulas are tested with one, two,
four or eight 256-value blocks in a work-group, with and without the
experimental root-sync property. The program compiles with oneAPI 2026.1,
precise floating point and default subgroup size 32, on the recorded
B570 driver. The [probe](probe.cpp) includes a preserved copy of the
engine's [original source](source/sycl/src/kernels/cuda/iq_kernels.dp.cpp).
There is no engine integration or full-model speed claim yet.

All 432 complete FP16 byte comparisons pass across nine quantization
types, flat and interleaved gate/up matrices and row counts 1, 640 and
641. Random finite blocks use seed 7, with each actual quant block's
scale initialized to 0.125. The small and odd shapes exercise guarded
work-group tails. Original functions are the comparison, rather than
a second implementation of their formulas.

Timings use the 640-row, 2,560-column shapes, ten warmup calls per
implementation and five alternating original/candidate pairs, 128 calls
per pair. Wall timing includes queue submission and completion. Allocation,
initialization and all-byte checking are outside timers. The original
API also performs its existing device capability check; the candidate
wrapper does not. Thus timing differences do not isolate the root-sync
property from host wrapper and compiled-kernel differences. These are
microbenchmarks on resident buffers, not complete prefill timings.

The planned peer annotation in `run.json` says the embedding service
remains resident, but the actual `peer_before` and `peer_after` observations
are both null: the user had already stopped it before the probe started.
The service is also verified inactive immediately afterward. These
observations take precedence over the controller's generic note.

The lowest median candidate for each type and role is listed below;
all 720 raw timing rounds and all geometry comparisons are in the
[run record](run.json). Selecting the minimum is exploratory rather than
an independently confirmed performance estimate.

| GGML type | Role | Blocks per group | Root property | Original µs | Candidate µs | Original/candidate |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 11 | Flat | 1 | No | 10.637 | 10.672 | 0.997 |
| 11 | Gate/up | 1 | No | 20.922 | 22.480 | 0.931 |
| 16 | Flat | 1 | No | 18.596 | 16.470 | 1.129 |
| 16 | Gate/up | 1 | No | 37.228 | 31.729 | 1.173 |
| 17 | Flat | 1 | No | 11.963 | 10.929 | 1.095 |
| 17 | Gate/up | 1 | No | 21.584 | 21.085 | 1.024 |
| 18 | Flat | 1 | No | 19.219 | 16.676 | 1.152 |
| 18 | Gate/up | 1 | No | 38.016 | 31.933 | 1.190 |
| 20 | Flat | 1 | No | 26.726 | 26.357 | 1.014 |
| 20 | Gate/up | 1 | No | 50.968 | 50.720 | 1.005 |
| 21 | Flat | 1 | No | 12.063 | 11.046 | 1.092 |
| 21 | Gate/up | 1 | No | 21.839 | 21.395 | 1.021 |
| 22 | Flat | 1 | No | 12.065 | 11.154 | 1.082 |
| 22 | Gate/up | 1 | No | 21.666 | 21.219 | 1.021 |
| 23 | Flat | 1 | No | 29.653 | 25.016 | 1.185 |
| 23 | Gate/up | 1 | No | 54.532 | 48.186 | 1.132 |
| 42 | Flat | 1 | No | 3.708 | 4.335 | 0.855 |
| 42 | Gate/up | 2 | No | 9.612 | 13.300 | 0.723 |

Larger groups offer no consistent gain. The one-block candidates without
the root property are worth a separate follow-up, especially IQ3_XXS and
IQ4_XS; the Q2_0 cases regress. A production choice needs a new matched
comparison and complete model head/state proof. The [build record](build.json)
and [build log](build.log) preserve compiler arguments and the warnings
inherited from including the complete engine kernel source.
