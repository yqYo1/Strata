# IQ4_NL activation scale caching: not adopted

Ryzen 5 5600X / 128 GiB RAM, precise icpx 2026.1, real GSQ-RCO IQ3_S
Qwen3.8 Flash Next, 2026-10-05. Two implementations convert read-only Q8_0
activation scales from FP16 to FP32 once per row group, retaining every weight
decode, `dx * dy` multiplication, FMA and final reduction in the original order.

The generic version caches up to 80 blocks, with the original conversion as
the fallback for longer reductions. It uses at most 1.25 KiB of scale scratch.
The fixed version specializes the real 640-input / 360-byte IQ4_NL row, uses
20 blocks and at most 320 bytes of scales, and falls back to the original
kernel for other widths/strides. Neither changes the native one-token rule,
which normally uses ggml rather than Strata's multi-token kernel.

Both were integrated in temporary off-by-default SYCL builds for measurement.
Both pass exact CPU comparisons, but a stable speed improvement does not
survive the full expert pool measurement. The production option, dispatch and
source are removed again; the executable is restored byte-for-byte to the
previous GCC gate/up build. These experiments are preserved for reproduction,
not presented as a TG optimization.

## Isolated down arithmetic

The baseline is the actual production IQ object with exported symbols renamed
without recompilation. `prepare.py` derives the first candidate from unchanged
upstream CPU source; `candidate.diff` shows the complete change. Processes pin
to CPU 2. Each of three runs passes 19,305,216 finite exact FP32 comparisons,
including guards and inactive outputs: 117 real experts from all 39 IQ4_NL
down layers, 1–8 tokens, plus six widths through 4096, five byte offsets, three
row paddings, partial ranges, signed finite half scales and varied activations.
4096 tests the generic scale-array fallback. Actual CMake helper checks pass
the same comparisons for both integrated implementations.

Median original/candidate ratio for the generic isolated change, 27
alternating-order pairs over three fresh processes:

| Tokens | Reused expert | 96 experts |
| ---: | ---: | ---: |
| 1 | 1.078 | 1.079 |
| 2 | 1.112 | 1.107 |
| 3 | 1.087 | 1.096 |
| 4 | 1.070 | 1.075 |
| 8 | 1.079 | 1.079 |

The streaming dataset is 88,473,600 bytes, above the 32 MiB L3. The baseline
repeats one expert 128 times or 96 experts twice. Every outlier is retained.
The fixed production helper also looks faster in one nine-pair isolated
series: ratios 1.077 / 1.122 / 1.088 at two / three / four tokens on 96 experts.
None of these kernel-only numbers demonstrates a whole-engine gain.

## Full production expert pool

Both arms retain `STRATA_IQ2S_GCC=ON`; only the scale experiment differs. Each
variant is tested against the actual saved pre-change archive, through five
pinned workers plus the pinned host, including gate/up, intermediate
quantization and down. For each variant, 396 cases over all 48 layers and
mixed/changing batches up to 97 experts compare 34,412,784 finite FP32 values
bit-for-bit. Complete 137,651,136-byte dumps have the same SHA-256 as the
previous full-pool reference:
`092acfb9cb9d0b96931490e96d709afd53b56f95fff07e9e33da252555489482`.
Every timed dataset is also byte-compared before its timing series.

There are 27 alternating-order pairs per case, in three fresh process pairs,
for each variant (864 pairs each). The table shows median original/candidate
ratios for IQ4_NL-down cases. Whole-pool / down-phase are shown separately:

| Layer | Experts | Tokens | Generic: pool / down | Fixed: pool / down |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 12 | 2 | 1.085 / 1.071 | 0.994 / 0.987 |
| 2 | 48 | 2 | 0.970 / 1.015 | 1.014 / 0.971 |
| 17 | 12 | 2 | 0.971 / 0.998 | 1.025 / 0.984 |
| 17 | 48 | 2 | 0.979 / 1.012 | 1.022 / 0.992 |
| 2 | 12 | 4 | 0.954 / 0.934 | 0.996 / 1.019 |
| 2 | 48 | 4 | 0.983 / 0.937 | 0.980 / 1.011 |
| 17 | 12 | 4 | 1.008 / 0.966 | 0.999 / 1.032 |
| 17 | 48 | 4 | 0.991 / 0.920 | 0.997 / 1.017 |

The generic version regresses four-token down by about 3–8%. The fixed version
reduces that regression but has no two-token down gain. Mixed 1/2/3/4 groups
show fixed whole-pool ratios 1.007–1.043, with changes in the untouched gate/up
phase too. Untouched Q2_0-down whole-pool controls range 0.959–1.051 in the
fixed series; individual processes are noisier. The stuck GPU-load control
PID 1074147 uses one CPU core throughout. Its affinity is not changed.
All sampled benchmark processes have VmSwap 0. This evidence is insufficient
to choose the scale change as a TG speed improvement. Its cause is not proved.

Full protocol, timings, phase counters, placement observations, hashes and
summaries are retained in `pool-generic` and `pool-fixed`. The first pool
runner stopped at a filename mismatch after both C++ validations completed.
The two initial dumps already match; the corrected runner was rerun to finish
comparisons and timing. `initial-production-check.log` and `initial-run.py`
retain that harness failure rather than hiding it.

## Reproduction and state

`iq4nl-scales-generic.cpp` / `iq4nl-scales-fixed.cpp` are the measured production
components. Apply the corresponding integration patch and copy the component
to `sycl/src/kernels/cpu/iq4nl_scales.cpp`, then configure with
`STRATA_IQ2S_GCC=ON` and the experimental `STRATA_IQ4NL_SCALES=ON`. Preserve
the existing archive first. `prepare-production.py`, `production-check.sh`
and `fixed-production-check.sh` describe the harness generation, actual-object
check and complete pool run. Build and failure logs accompany them.

Frozen archives, executables and full dumps are preserved outside `/tmp` under
`~/.local/state/strata-sycl/cpu-down-scales*`. `frozen-engines.json` identifies
the control, generic and fixed engines. After rejection, `restored.json`
confirms restoration of engine SHA-256
`0d12548e844349821453fde748eb10bfab8f0388317a493d065f76a1046f1542`.
The generic experimental engine also passes the two 256K capacity refusals;
that does not validate full-length GPU prefill/decode. Actual 256K tests and
whole-engine PP/TG remain pending GPU recovery. PP 1000 / TG 70 are unachieved.
