# Shared activation loads for gate/up: rejected candidate

On Ryzen 5 5600X, interleaving independent gate/up block decoding so both use
one activation load regresses the affected two-token CPU cases. No production
kernel is changed. Gate and up retain their original integer addition and
FP32 FMA order; every synthetic and real output bit matches the control.

The probe enables this path only for IQ3_S and IQ2_S with at most two tokens.
Its three/four-token cases keep the original algorithm and provide comparison
noise controls. Each format has nine alternating-order pairs for a reused
expert and a 96-expert dataset larger than L3. Real geometry is 2560 / 640;
streaming weights occupy 135,168,000 bytes (IQ3_S) or 100,761,600 (IQ2_S).

Median paired original/candidate time ratios for two tokens are:

| Format | Reused expert | 96 experts |
| --- | ---: | ---: |
| IQ3_S | 0.938 | 0.944 |
| IQ2_S | 0.918 | 0.923 |

Values below one mean the candidate takes longer. Both completed runs also
check 90 real expert samples, 1–8 tokens, six synthetic i-quant formats,
three widths, five byte offsets and partial output bounds: 7,591,680 exact
finite float comparisons per run, including untouched guard outputs.

Use `summary.json` and both raw JSONL files for individual observations.
Compiler/math, affinity, source-generation dependencies and GPU limitations
are the same as [the sign probe](../cpu-sign-probe/README.md). These CPU-only
observations do not measure whole-engine TG. Extra simultaneous accumulators
can increase register pressure, but that is an explanation to investigate,
not a measured cause of this regression.
