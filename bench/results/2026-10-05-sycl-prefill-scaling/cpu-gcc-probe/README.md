# Alternate GCC code generation for selected IQ2_S groups

On Ryzen 5 5600X, compiling the unchanged CPU i-quant source with GCC 13.3
helps the two/four-token IQ2_S gate/up cases, while IQ3_S regresses. The
experimental SYCL option `STRATA_IQ2S_GCC=ON` therefore selects the alternate
object only for IQ2_S gate/up with exactly two or four tokens. It is off by
default; other groups, dot products, GPU kernels and rounding policy retain
the upstream implementation. Whole-engine TG and output validation are pending.

The candidate uses `-O3 -ffp-contract=off -mavx2 -mfma -mf16c`; explicit FMA
intrinsics remain fused. The original uses precise icpx 2026.1. Both use the
same pinned ggml quantizer and real 2560 / 640 gate/up geometry. Ninety expert
samples over the pack's 30 IQ3_S/IQ2_S layers, six synthetic i-quant formats,
1–8 tokens, three widths, five offsets and partial output ranges give
7,591,680 finite exact float comparisons per run, including untouched guards.

Initial nine alternating-order pairs, median original/candidate time ratio:

| Format | Tokens/expert | Reused expert | 96 experts |
| --- | ---: | ---: | ---: |
| IQ2_S | 2 | 1.076 | 1.079 |
| IQ2_S | 4 | 1.086 | 1.093 |
| IQ3_S | 2 | 0.903 | 0.897 |
| IQ3_S | 4 | 0.927 | 0.920 |

The IQ2_S four-token series has large early outliers in both directions;
none are removed. No engine-level speed improvement follows from this table
alone. Validation against the actual production CPU objects and repeated
observations are recorded separately in `dispatch-check`.

The upstream CPU source is not edited. A small SYCL wrapper includes it and
selects the alternate compiler only for the observed groups. Default build
SHA-256 is exactly the prior candidate's
`b1a7db41d1225d024d3455b9db58e71aebd13c8f88b8e737ae11049fe257b56d`.
The opt-in frozen executable currently has SHA-256
`0d12548e844349821453fde748eb10bfab8f0388317a493d065f76a1046f1542`.
Both executables are preserved outside `/tmp` under
`~/.local/state/strata-sycl/cpu-gcc-probe`. GPU recovery, full heads/IDs/logprobs,
checkpoint/MTP and actual 256K boundary checks are still required before merge.
