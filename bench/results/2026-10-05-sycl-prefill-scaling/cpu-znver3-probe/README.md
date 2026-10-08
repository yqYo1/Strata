# Explicit Zen 3 target: rejected global compiler change

Adding `-march=znver3` to the CPU i-quant translation unit does not improve
its important two-token cases on Ryzen 5 5600X. One-token IQ2_S is faster in
this probe, but that group size uses ggml's own single-token dot in the
unchanged engine. No production compiler setting is changed for this result.

Median paired original/candidate ratios for the 96-expert two-token case are
1.005 (IQ3_S) and 0.985 (IQ2_S); four-token IQ3_S regresses to 0.938. Some
IQ2_S four-token times fluctuate substantially, so that observation does not
justify a global flag change. All pairs are preserved.

Each format completes 90 real-expert sample comparisons, 1–8 tokens, and the
same synthetic coverage as [the sign probe](../cpu-sign-probe/README.md).
All 7,591,680 float comparisons per run, including output guards, are finite
and bit-exact. CPU-only microbenchmarks do not establish whole-engine TG.
