# Actual native CPU service calibration,22/20/NT1

Four fresh processes, counterbalanced6/18/18/6 task profiles;25 measured rounds per arm/cohort in each process. All recorded per-round samples remain. This is a layer-balanced host-resident weight characterization with synthetic activations, not model throughput.

| profile | cohort | GU ms/call | quant ms/call | Down ms/call | complete ms/call | pool cohort ms | pool fit gate |
|---|---|---:|---:|---:|---:|---:|---|
| tasks6 r1 | train | 0.267884 | 0.000181 | 0.086712 | 0.297165 | 296.229418 | False |
| tasks6 r1 | holdout | 0.262696 | 0.000150 | 0.085063 | 0.283812 | 302.236611 | False |
| tasks18 r1 | train | 0.260450 | 0.000140 | 0.084263 | 0.276803 | 298.121039 | True |
| tasks18 r1 | holdout | 0.255477 | 0.000189 | 0.082527 | 0.288142 | 299.958012 | True |
| tasks18 r2 | train | 0.255925 | 0.000172 | 0.082572 | 0.282314 | 293.210051 | False |
| tasks18 r2 | holdout | 0.260528 | 0.000177 | 0.086476 | 0.280055 | 297.353540 | False |
| tasks6 r2 | train | 0.262410 | 0.000136 | 0.085008 | 0.282631 | 302.232368 | False |
| tasks6 r2 | holdout | 0.263438 | 0.000133 | 0.081693 | 0.279707 | 298.702040 | False |

95% bootstrap intervals, relative widths, CV, every fit/holdout gate and per-process ratios are in summary.json. Failing gates remain failed; additional repeats do not turn distinct prompts or experts into independent workloads. Exact native/independent NT1 GU+Down20, quantizer, repeat and partition checks passed in each process.

If any fit/holdout gates fail, those cells are unresolved and cannot weight route-demand or establish a task candidate. Whole-engine >=32K and full262144 gates remain mandatory. No pool-ms sum or cohort-ms division is an inference/per-job latency estimate.
