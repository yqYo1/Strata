# Independent integer accumulators: rejected candidate

On Ryzen 5 5600X, splitting an IQ3_S/IQ2_S block's integer sum into independent
even/odd-half sums regresses the important two-token gate/up case. Each sum
is combined before the original per-block FP32 conversion and FMA; every
synthetic and real output bit matches. No production kernel is changed.

This follows the accumulator layout already present in Strata's IQ2_XS
multi-token path and the inspected pinned llama.cpp IQ3_S CPU implementation.
It is applied only for one through three tokens; four tokens retain the old
algorithm. More independent integer chains do not establish a speed gain.

Each format has nine alternating-order pairs on a reused expert and a
96-expert dataset above L3. Median paired original/candidate time ratios for
two tokens are:

| Format | Reused expert | 96 experts |
| --- | ---: | ---: |
| IQ3_S | 0.957 | 0.955 |
| IQ2_S | 0.950 | 0.952 |

A roughly 3% one-token IQ3_S observation is not adopted: the unchanged engine
uses this multi-token kernel from two tokens by default, and the affected
IQ2_S cases also regress. Changing the engine's rounding policy to make that
observation applicable would not preserve the measured original configuration.

Both completed runs check 90 real expert samples over 30 affected layers,
1–8 tokens, six synthetic formats, three widths, five byte offsets and partial
output ranges: 7,591,680 exact finite float comparisons per run, including
untouched output guards. All comparisons pass.

Compiler, affinity, source-generation dependencies and GPU limitations are
the same as [the sign probe](../cpu-sign-probe/README.md). Whole-engine TG is
not measured. Raw observations, complete source and build scripts are retained.
