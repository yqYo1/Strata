# CPU attainable capacity and native expert service

Ryzen5 5600X, unchanged powersave/EPPpower policy. No GPU or model inference. Three fresh processes per condition; no optimization adopted.

| FP32 AVX2 register FMA | Median GFLOP/s | Range of process medians |
| --- | ---: | ---: |
| 1 physical cores | 146.85 | 146.65–147.23 |
| 6 physical cores | 818.22 | 792.51–819.01 |

The loop contains exactly12 independent packed FP32 FMA instructions, no loop spill or memory operand. One warmup and seven measurements per core-count perprocess; full96float outputs compare to untimed scalarFMA references. This is attained arithmetic throughput under a register-only workload, not a quantized integer-dot ceiling.

The native service fixture uses384 actual GU22/IQ2_S and Down20/IQ4_NL experts, deterministic synthetic activations, H2560/FF640. Fresh NT1–4 qualifiers preserve exact NT1 controls and predeclared maxabs/NRMS budgets; worst observed errors are below2.4e-7absolute GU and4.5e-8Down/fullchain. Every measured process repeats all references, guards, direct/pool parity, placement and full row counts.

| NT | Active set | GU µs/expert-token | Down µs/expert-token | Direct complete µs/expert-token | Pool complete µs/expert-token |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | hot | 181.971 | 78.108 | 261.405 | 72.204 |
| 1 | streaming | 201.152 | 83.925 | 286.466 | 90.978 |
| 2 | hot | 139.496 | 61.718 | 198.422 | 48.452 |
| 2 | streaming | 137.526 | 61.863 | 197.137 | 54.215 |
| 3 | hot | 110.070 | 54.482 | 163.160 | 39.076 |
| 3 | streaming | 110.439 | 54.139 | 161.989 | 42.931 |
| 4 | hot | 95.919 | 50.760 | 147.419 | 34.736 |
| 4 | streaming | 96.966 | 50.722 | 148.362 | 38.738 |

Table native values summarize the six process/cohort medians (3processes×2disjointcohorts). Structured results retain each separate process/cohort median and all6000 raw measurement rows plus720warmups. These are service intervals, not token throughput of the whole model. Hot uses8selectedexperts/cohort≈15.04MiB packedblobs; streaming uses192≈360.94MiB. Differentexpert subsets and jobcounts prevent a pure cache-causality inference; cache residency and physicalDRAM traffic are unmeasured.

Direct runs use one pinned caller. Pool runs use five workers plus host, tasks0→18rowpartitions, batch1, synchronous phases with serial FFquant. This is no worker-count sweep. Dense-equivalent FLOPs normalize mathematical dot work; they cannot be divided by register FP32 capacity to claim a percent CPU utilization. NT4 emitted spills do not prevent it being faster pertoken in this fixture; spill dominance remains unmeasured.

The first untimed process passed all numerical/placement checks and normal0, but its controller incorrectly expected the timed finalmarker; its original receipt remainsFAILED. Controller2 changed only stage-marker parsing, pinned the actual manifest cohort_sha256 and added traceback reporting; all four fresh qualifiers and24timingprocesses pass unchanged source/gates.

Compact archives preserve original receipt hashes/statuses and each timing row, unique canonicalpayload identities, exact count/worst-error summaries, commands/ownership/policy/build/source/binarypins. Repeated successful reference-vector text is represented by verified count/worst summaries and a hash; no failed status is rewritten. Current four original qualifiers remain fixtures for the calibration controller. No full262144-position model boundary is qualified.
