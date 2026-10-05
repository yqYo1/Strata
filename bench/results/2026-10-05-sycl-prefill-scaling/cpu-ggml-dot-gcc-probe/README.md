# One-token ggml compiler and half-scale probes

Measured on 2026-10-06: Ryzen 5 5600X, 128 GiB RAM, precise oneAPI 2026.1
production objects, GCC 13.3. No one-token dispatch is adopted. Whole-engine
PP/TG and actual full-256K GPU validation remain pending GPU recovery.

Native expert groups of one token normally use ggml's `vec_dot`; the earlier
Strata multi-token compiler results do not cover that path. This probe extracts
the actual production x86 C object and renames all its global exports with
`objcopy`, without recompiling it. The alternate object also has every global
export renamed, including activation quantizers, so it cannot replace the
original quantization or shadow an unrelated dot. In every validation call the
renamed reference must match the original CPU traits' actual dot, including
guards. The pinned ggml checkout is clean at
`3cf03257f219afbe7334045ff7c6a06ac68c627d`.

Three alternatives:

- GCC compiles the unchanged C source with `-ffp-contract=off` and
  `-fno-associative-math` (parent results).
- GCC allows FMA contraction with `-ffp-contract=fast`, still disabling
  reassociation (`gcc-contract`).
- icx retains the production precise/native flags. A wrapper overrides the
  half-to-float macro with `_cvtsh_ss` before including the original source,
  using F16C for scale conversions (`icx-f16c`). Upstream source is unchanged.

Every fresh run checks 23,607,000 output/guard floats across nine IQ formats,
synthetic widths, five byte offsets, three row paddings and eight original
quantized activation vectors including zero/constant/alternating inputs.
Synthetic half scales include finite signs and subnormals; IQ1_M's packed
scale nibbles are set explicitly. Real validation covers 405 experts: indices
0/7/31 in every supported gate/up/down tensor over all 48 layers. Q2_0 down is
outside this C-object probe and is covered by the full-pool check below.
Every one of the 96 experts used by a timing is checked separately as well.

Each alternative has three fresh runs, nine alternating-order pairs per case,
with the caller pinned to CPU 2. One expert is repeated 128 times, or all 96
twice. Streaming datasets are 50.4–88.5 MB, above the CPU's 32 MiB L3. The
following table uses the 96-expert cases; values below one are slower. This is
single-token dot time, not TG.

| Format | GCC, contraction off | GCC, contraction allowed | icx F16C |
| --- | ---: | ---: | ---: |
| IQ3_XXS | 1.026 | 1.032 | 1.021 |
| IQ3_S | 1.014 | 1.022 | 0.992 |
| IQ2_S | 0.959 | 0.984 | 0.996 |
| IQ4_XS | 0.934 | 0.930 | 1.013 |
| IQ4_NL | 0.927 | 0.921 | 0.689 |

Both GCC configurations differ from the original on 214 IQ4_NL comparisons
per run, including short odd-block inputs. Contraction off also differs on
255 IQ1_S comparisons. Those formats fail the unchanged-bits requirement.
All other tested GCC formats match. The icx F16C variant matches every tested
output and separately checks all 63,488 finite half values against the actual
CPU conversion table. Its IQ4_NL speed regression rules out replacing scale
conversions generally.

## Selective IQ3_XXS full-pool check

The repeatable isolated IQ3_XXS gain is checked in the real production expert
pool, selecting only that format's GCC dot. Both arms link the same current
CPU archive, including the existing optional two/four-token IQ2_S GCC kernel,
and the same original ggml libraries. A linker wrapper returns original CPU
traits for every type except IQ3_XXS; that copy changes only `vec_dot`. The
normal group-size policy, activation/intermediate quantizers and other dots
remain unchanged. This selection is a measurement harness, not engine code.

The separate selection audit checks all 43 CPU trait entries: zero changed
entries in the baseline and exactly IQ3_XXS in the alternate, retaining its
quantizer, activation type and row count. Compiled calls to the wrapper are
recorded in `pool-iq3xxs/compiled-dispatch.txt`.

Both arms pass 34,412,784 finite output/guard comparisons over 396 cases: all
48 layers, three experts, 1–8 tokens and changing mixed batches up to 97
experts. Stress batches include IQ3_XXS. The complete 137,651,136-byte dumps
compare exactly, SHA-256:
`24ac38306a79350a575a4bb6fd5c648ae7280f1f9ca9a7ef7fef636373f276f9`.
Five pinned workers plus the pinned host run gate/up, intermediate quantization
and down. All 96 timed datasets also have an actual byte comparison.

The pool measurement has 864 alternating-order pairs: layers 0/46 (IQ3_XXS),
1 (IQ2_S) and 17 (IQ3_S), expert counts 12/48, token patterns 1/2/4/mixed,
27 pairs per case over three fresh process pairs. For one-token IQ3_XXS cases,
wall-time ratio medians are 0.984–1.071; individual process medians cross one.
Mixed-case medians are 0.991–1.059. Unchanged one-token controls are
0.952–0.999, and unchanged multi-token phases also vary. This does not establish
a stable complete-pool gain, so the selective dispatch is not adopted.

The stalled GPU model-load control continues using one CPU core during these
CPU-only measurements. All controls and outliers are retained. Every sampled
pool process has VmSwap 0; this is not a claim about global host swap. These
programs create no GPU queues or resets.

`manifest.json`, each variant's manifest, raw JSONL, stderr and summary preserve
flags, object/symbol mappings, hashes and all observations. The pool selection
manifest adds the compiled harness/wrapper hashes. Executables, objects,
archives and full output dumps are kept outside Git in
`~/.local/state/strata-sycl/cpu-ggml-dot-gcc-probe`.
`decision.json` verifies that the production engine remains byte-for-byte
SHA-256 `0d12548e844349821453fde748eb10bfab8f0388317a493d065f76a1046f1542`.

Reproduce from this directory, with the current precise/GCC-gate-up build:

```sh
bash build.sh
source /opt/intel/oneapi/setvars.sh
python3 variant.py gcc-contract
python3 variant.py icx-f16c
bash pool-build.sh
```
