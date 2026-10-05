# Actual engine CPU dispatch comparison

The opt-in `STRATA_IQ2S_GCC=ON` CPU archive passes complete output-bit checks
against the exact default engine IQ object. Compiler versions are icpx 2026.1 and GCC 13.3.
CPU affinity is logical CPU 2 on Ryzen 5 5600X; geometry is 2560 / 640. The control object's four exported
symbols are renamed with objcopy so both implementations can link into one
harness; it is not recompiled. `symbol-renames.json` records that step.

Four completed CPU-only processes compare 90 real expert samples across all
30 affected-format layers, six synthetic formats, 1–8 tokens, three widths,
five byte offsets and partial output bounds. Every run has 7,591,680 finite
exact float comparisons including untouched guard outputs. No GPU queue is
created. The one-time sign-pattern check is omitted here because the actual
production archive does not export the probe's sign helper.

Three fresh IQ2_S processes each run nine alternating-order pairs per case.
The original/candidate time-ratio median over all 27 pairs is:

| Tokens per expert | Reused expert | 96 experts, above L3 |
| --- | ---: | ---: |
| 2 | 1.139 | 1.130 |
| 4 | 1.092 | 1.086 |

In the 96-expert case, the per-process medians are 1.110 / 1.141 / 1.132 for
two tokens, and 1.084 / 1.090 / 1.082 for four. This is a local CPU gate/up
speed observation, not a whole-engine TG measurement. The native model has
mixed quantization types: the option selects only IQ2_S groups of size 2/4;
IQ3_S and other group sizes retain the icpx path. Those unchanged paths are
also covered by equality checks and timed controls.

All timings and outliers are retained. A separately stalled GPU control
process remains alive; no GPU correctness or full-context conclusion follows
from these CPU tests. The initial standalone-reference prototype records,
including a noisy third two-token series with no improvement, remain under
`preliminary-standalone-reference`. They are not pooled with the exact
production-object comparison above.

Default and optional full engine builds succeed. The default executable is
byte-for-byte the prior frozen candidate (`b1a7db41...`); the GCC selection
executable is `0d12548e...`. Complete hashes, production-source hashes and build
commands are in `summary.json`. The alternate build leaves GPU code under
precise oneAPI compilation. Upstream CPU source is included unchanged.

A shell's `set -u` initially rejected an unset variable inside oneAPI's
`setvars.sh`; the corrected runner keeps `set -e` and `pipefail`. Its initial
environment error is preserved. Build scripts and both successful option
configurations are recorded.

Before merging or selecting this option for normal use, require matched
whole-engine finite heads, IDs/logprobs, normal MTP and checkpoint results,
actual full-256K checks, and repeated warm TG timings after GPU recovery.
