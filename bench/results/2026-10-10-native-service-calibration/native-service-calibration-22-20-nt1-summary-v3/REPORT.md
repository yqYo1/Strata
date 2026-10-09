# Actual native CPU service calibration,22/20/NT1

Four new fresh processes r3/r4, counterbalanced6/18/18/6 task profiles with six verified distinct singleton CPUs0..5;25 measured rounds per arm/cohort in each process. All recorded per-round samples remain. This is a layer-balanced host-resident weight characterization with synthetic activations, not model throughput.

| profile | cohort | GU ms/call | quant ms/call | Down ms/call | complete ms/call | pool cohort ms | pool fit gate |
|---|---|---:|---:|---:|---:|---:|---|
| tasks6 r3 | train | 0.195446 | 0.000145 | 0.084638 | 0.283233 | 17.473221 | False |
| tasks6 r3 | holdout | 0.203454 | 0.000152 | 0.083745 | 0.289265 | 17.424228 | False |
| tasks18 r3 | train | 0.198026 | 0.000135 | 0.080741 | 0.271364 | 15.897815 | False |
| tasks18 r3 | holdout | 0.202764 | 0.000130 | 0.081067 | 0.270254 | 14.820987 | False |
| tasks18 r4 | train | 0.197529 | 0.000126 | 0.080945 | 0.270180 | 14.992602 | False |
| tasks18 r4 | holdout | 0.197946 | 0.000131 | 0.080505 | 0.271452 | 15.021326 | False |
| tasks6 r4 | train | 0.199146 | 0.000129 | 0.082624 | 0.282296 | 17.496126 | False |
| tasks6 r4 | holdout | 0.199627 | 0.000139 | 0.082902 | 0.281337 | 17.208084 | False |

95% bootstrap intervals, relative widths, CV, every fit/holdout gate and per-process ratios are in summary.json. Failing gates remain failed; additional repeats do not turn distinct prompts or experts into independent workloads. Exact native/independent NT1 GU+Down20, quantizer, repeat and partition checks passed in each process.

If any fit/holdout gates fail, those cells are unresolved and cannot weight route-demand or establish a task candidate. Whole-engine >=32K and full262144 gates remain mandatory. No pool-ms sum or cohort-ms division is an inference/per-job latency estimate.
