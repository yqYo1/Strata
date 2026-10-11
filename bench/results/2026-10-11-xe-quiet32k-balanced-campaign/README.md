# Quiet 32K balanced campaign

Main completed the planned four independent fresh-process pairs. All eight
samples passed strict GEN16/token/selected LP5 checks, normal QUIT, stable
source/runtime/compiler mapping checks and fresh before/after GPU health and
kernel fault gates. No retry, reset or warmup was used.

The source is 0740b43488d812117f021ccbe4bd0e15d223f955, with the actual free
release built from fa3e20e8 plus the content-identical published five-line
optional preload guard. The original historical build identity is preserved.

Each arm uses the same32768-token input, actual chunk6144, G16, context262144,
KV INT8/resident32768, cache request128/actual144, workers5, MTP4/min-p0,
FUSED0 and preload0. Only STRATA_PREFILL_NATIVE_DEQUANT_BATCH differs.
Warning/error logs remain; API/validation/transfer/census/GPU timers are absent.
Engine DONE phase times are used. Mapping hashes occur before GEN and after
DONE; their CPU/I/O/cache effects are not calibrated.

| Pair | Order | OFF prefill tok/s | ON prefill tok/s | OFF cold decode tok/s | ON cold decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | OFF / ON | 263.845 | 260.107 | 6.887 | 7.105 |
| 2 | ON / OFF | 256.312 | 252.552 | 8.135 | 7.911 |
| 3 | OFF / ON | 259.268 | 271.105 | 7.338 | 7.901 |
| 4 | ON / OFF | 260.678 | 270.475 | 9.070 | 7.780 |

| Phase | OFF mean | OFF sample SD | ON mean | ON sample SD |
| --- | ---: | ---: | ---: | ---: |
| Prefill tok/s | 260.026 | 3.129 | 263.560 | 8.904 |
| Cold decode tok/s | 7.858 | 0.959 | 7.674 | 0.384 |

Keep native dequant batching default OFF. Prefill and decode are evaluated
separately; these four pairs do not establish a robust benefit. This flag does
not change decode kernels. All individual phase times, paired differences,
medians, SD/ranges, exact command/environment/input and process/fault outcomes
are in [main-comparison-v1.json](main-comparison-v1.json) and the byte-exact samples.
The raw receipts retain performance_eligible=false; Main qualifies only this
complete recipe-specific timing comparison in a separate derived record.

GEN16 cold decode is a short sample, not sustained warm or restored decode.
One canonical prompt and selected LP5 equality do not qualify overall model
ability, RESTORE or a completed262144-token context. Those remain open.

Actual CPU controller qualification v3 passed 9 original positive and35
negative fixtures,39 additional cases (26 rejections), and4 malformed CLI cases.
The immutable launch manifest and the unchanged pending template remain here.

Research registry v136 contains568 completed reports. Original reports are
lossless; [main-research-corrections-v2.json](main-research-corrections-v2.json)
records the serve/non-serve scope and route-row-bound corrections. Reusable
FP16 Xs/FP32 GU/FP16 Hh scratch could lower the T8192 owned request by721.283MiB
after the shared max(GDN,QSA,MoE) limit, leaving the new T8192 request only
44.389MiB above the current old T6144 request. This is unimplemented and does
not measure physical VRAM released or added dispatch costs. Row-capped groups
can preserve grouping conditionally; descriptor lifetime and all callers
need review before execution. The earlier GDN y-removal capacity claim remains
rejected: its view lies inside an MoE-dominated shared arena, so owned bytes
saved are zero in this recipe. Source-owned T6144 is2815.941MiB (about2.750GiB).

The [upstream settings audit TODO](../2026-10-11-upstream-parameter-audit/TODO.md)
remains open: past settings were explicit, but an independent search for the
best official prefill/decode settings and its verification is not established.
Current quiet samples are the Xe candidate, not new official-pristine timings.

Next: implement and qualify bounded fallback scratch reuse, then measure actual
8K capacity and64K pass cost. Do not weaken the512MiB READY reserve or full-context
requirements. A separate serve workspace-retirement candidate preserves session
and decoder state; cache enlargement still needs its own pointer/graph contract.
