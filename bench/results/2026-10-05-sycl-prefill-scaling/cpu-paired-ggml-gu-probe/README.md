# Pairing one-token original-ggml gate/up: not selected

Ryzen 5 5600X / 128 GiB RAM, 2026-10-06. Sharing activation loads between
gate and up preserves the original one-token output bits, but neither precise
icx 2026.1 nor GCC 13.3 establishes a useful streaming speed improvement.
Production source, dispatch and engine binary are unchanged.

This differs from the earlier two-token Strata-kernel probe: one token normally
uses ggml, with its own reduction order. `generate.py` copies the original
AVX2 loop bodies for IQ3_XXS, IQ3_S and IQ2_S from the pinned ggml source. It
duplicates each weight lane's integer partial sums, per-block FP32 FMA and
final horizontal reduction, and shares only the two 32-byte activation loads
inside each 64-element subblock. Structure field names are retained during
variable renaming. The original source is included unchanged to provide its
tables/static helpers; all its global exports are renamed so they cannot
interpose the reference dots or activation quantizers.

The reference is the actual current `native_gu_rows` production archive, with
the default one-token ggml policy. Both helper compilers retain native AVX2,
FMA and non-associative math settings. The C++ SwiGLU expression is compiled
with precise icpx and matches the production expression. Inherited `STRATA_*`
variables are cleared. The existing optional multi-token IQ2_S GCC production
configuration is retained, but is not reached by this one-token probe.

Each of three fresh processes passes 31,056,048 finite exact comparisons,
including both compiler candidates and guards: 26,604,000 raw gate/up dot
values and 4,452,048 SwiGLU values. Raw dots are compared independently so a
zero SwiGLU result cannot conceal a changed gate or up. Coverage includes
141 real experts (0/7/31 over all 47 supported layers), all 96 experts of every
timed dataset, synthetic widths 256/768/2560, five byte offsets, three row
paddings, finite signed/subnormal half scales and eight original quantized
activation vectors. Partial output ranges and their untouched guards are
checked. These are eight separate one-token inputs, not eight-token groups.

The caller is pinned to CPU 2. Every case has 27 alternating-order pairs over
three fresh processes. The benchmark includes the full gate/up plus SwiGLU
row pass. It repeats a reused expert 128 times, or all 96 experts twice.
Streaming gate/up buffers are 120,422,400 bytes (IQ3_XXS), 135,168,000
(IQ3_S) and 100,761,600 (IQ2_S), above the CPU's 32 MiB L3.

Median original/candidate time ratio; below one is slower:

| Format | icx, reused | icx, 96 experts | GCC, reused | GCC, 96 experts |
| --- | ---: | ---: | ---: | ---: |
| IQ3_XXS | 0.962 | 0.968 | 0.988 | 1.004 |
| IQ3_S | 1.012 | 0.987 | 0.958 | 0.963 |
| IQ2_S | 0.995 | 1.000 | 0.867 | 0.875 |

All observations, process medians and outliers are retained. The stalled GPU
model-load control uses one CPU core during this CPU-only measurement. No GPU
queue or reset is created. These results do not measure whole-engine TG or
actual 256K context behavior. A full-pool run is not justified by these isolated
results, so this gate/up selection is not integrated into the engine.

`assembly-summary.json` records vector stack accesses in the paired helper
objects. These include deliberate vector-to-scalar index materialization in
IQ3_S and may include spills. Counts alone do not establish the cause of a
speed regression. Full disassemblies, objects, executable and logs are
preserved in `~/.local/state/strata-sycl/cpu-paired-ggml-gu-probe`.
`generation.json` and `manifest.json` record source/object hashes and symbol
renaming. `build.sh` contains the exact compiler/link commands. Reproduce:

```sh
bash build.sh
```
