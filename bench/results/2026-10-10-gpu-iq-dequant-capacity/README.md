# Actual-pack GPU IQ dequant capacity

Measured2026-10-10 on ArcB570 with unchanged safe LevelZero controls. Three fresh clean processes, seven samples per pair per process:147 sample rows. Every process completed normally, all selected output/source/guard gates passed, and fresh kernel fault intervals were empty. One separate detailed qualifier compared34,406,400 output half bits plus48 device conversion edges. The three timing processes compared4,748,083,200 halves outside timers.

| GU / Down type | Host-completion µs / expert | Process-median range | Logical input+output GB/s |
| --- | ---: | ---: | ---: |
| 18 / 20 | 129.134 | 121.169–129.537 | 92.98 |
| 18 / 42 | 99.681 | 96.865–102.506 | 115.83 |
| 21 / 20 | 115.101 | 114.924–115.745 | 105.65 |
| 21 / 42 | 75.876 | 73.079–79.373 | 154.19 |
| 22 / 20 | 114.156 | 109.464–114.419 | 103.38 |
| 22 / 42 | 76.033 | 73.670–78.708 | 149.16 |
| 23 / 20 | 207.700 | 203.421–210.490 | 60.15 |

The timer includes submitting32 distinct actual expert blobs and one terminal queue wait; each blob uses one GU and one Down wrapper, with two reusable output slots. Parsing, H2D staging, warmup/JIT, oracle checks and readbacks are outside samples. Source data are device-resident; cache state is not proved cold. The wrappers return void: these are host-completion service rates, not exclusive kernel durations. Packed input plus9,830,400 FP16 output bytes/expert is logical traffic, not a DRAM-controller counter. Large GEMM capacity does not include this work.

The precise GPU flags/default subgroup32 and runtime safety flags are preserved; standalone GPU compile optimization is O2, not an absolute maximum search. Input source is the exact inspected7-pair pack;32 selected experts/pair are not512-expert/model routing coverage. Generic rates do not apply to MMQ/fused/peer routes. Actual per-request branch/row/copy census remains required.

Original cached GGML half conversion failed a CPU overflow-domain test. Exact original ggml.c/quants.c with precise/no-contraction passes independentCPUtests; original library/failed receipts remain unchanged. The repaired reference passes all actual selected GPU output bits. This is diagnostic reference qualification, not engine adoption/fullcontext inference.

[All individual samples and closure evidence](three-process-summary.json). [CPU admission and preserved failures](CPU-admission/README.md).
