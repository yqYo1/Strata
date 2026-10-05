# GCC IQ4_NL down: rejected

On Ryzen 5 5600X / 128 GiB RAM (2026-10-05), the unchanged IQ4_NL down source
compiled with GCC 13.3 is slower than the actual precise icpx 2026.1 production
object. The existing optional GCC gate/up object already contains the renamed
down entry point, but the engine does not select it. No down dispatch is added.

The baseline is the actual production IQ object with its four global exports
renamed by `objcopy`, without recompilation. The icpx object inlines its IQ4_NL
templates and has no external template definitions/references. GCC's emitted
template symbols therefore cannot replace the baseline loops in this build.
Both entry points are called in the same CPU-only process, pinned to CPU 2.
The stalled GPU-load control remains live and uses one CPU core; no GPU queue
is created or reset by this probe. No inherited `STRATA_*` variables are set.

Three runs each pass 19,285,056 finite exact float comparisons including
inactive outputs and guards: 117 real experts (0/7/31 over all 39 IQ4_NL down
layers), 1–8 tokens, plus synthetic widths 32/96/640/768/2560, five byte offsets,
three row paddings, partial output ranges, signed finite half scales and
zero/constant/alternating/random quantized activations. This compares to the
original Strata kernel's bits, retaining its group-size rounding policy.

Median original/GCC time ratio, 27 alternating-order pairs over three fresh
processes; values below one are slower:

| Tokens | Reused expert | 96 experts |
| ---: | ---: | ---: |
| 1 | 0.883 | 0.877 |
| 2 | 0.943 | 0.943 |
| 3 | 0.971 | 0.972 |
| 4 | 0.968 | 0.972 |
| 8 | 0.971 | 0.967 |

The native driver's default one-token path uses ggml rather than this kernel;
the one-token row here is not a one-token engine optimization. Real down
geometry is 640 inputs / 2560 output rows. The 96-expert dataset is 88,473,600
bytes, above the CPU's 32 MiB L3. Timings repeat one expert 128 times or all 96
twice. All outliers and per-process medians are retained in the raw JSONL and
`summary.json`; this is not TG.

`probe.cpp` and `build.sh` show the exact harness/compiler/link setup. The
actual renamed baseline and alternate CMake GCC object, executable and their
hashes are preserved in `~/.local/state/strata-sycl/cpu-gcc-down-probe` and
`manifest.json`. Source of the baseline symbol mapping is the previous
[production-object comparison](../cpu-gcc-probe/dispatch-check/README.md).
