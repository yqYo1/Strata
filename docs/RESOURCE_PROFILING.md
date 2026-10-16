# Hardware resources in prefill tuning

Measure which resource limits the dependency path before selecting an optimization.
Stage times alone do not identify PCIe, RAM, VRAM, matrix arithmetic, local memory,
synchronization or CPU submission as the limiting resource.

For each fixed long-input case, record source/binary/runtime/driver, actual tokens,
chunk/context, precision, KV format, placement and effective pre-exec environment.
Use the official benchmark workload and keep instrumented profiles separate from
repeated clean speed samples. Tiny runs are numerical/collector safety gates.

| Resource | Application evidence | Capacity comparison and limits |
| --- | --- | --- |
| PCIe | Copy direction, requested bytes, intervals, dependencies and copy/compute overlap | Measure both directions with matching sizes/allocation types. Requested bytes are not wire traffic; topology is not attained bandwidth. A fast busy copy can occupy a small fraction of the request. |
| RAM | Staging/packing bytes when source-proved, CPU time, measured memory traffic and data access pattern | Match single/multiple threads and streaming versus gather/copy. Separate logical read/write accounting from physical traffic. |
| VRAM/cache | Per-kernel physical read/write/cache counters when available, plus copy measurements | A logical device-copy roof is operation-specific. Do not infer kernel DRAM saturation from source load counts or copy bandwidth. |
| GPU arithmetic | Actual kernel work, dimensions, datatype/accumulation, supported arithmetic activity counters | Compare matching kernels/shapes. Large GEMM throughput and vendor peak are not utilization measurements for a tiled attention kernel. |
| SLM/registers/synchronization | Actual supported stall/occupancy/conflict counters or controlled component comparisons | Compiled SLM/register metadata and barrier counts do not measure occupancy or barrier wait. |
| CPU/control | Issuing-thread CPU time inside waits versus outside waits, submission/packing/grouping, other threads, observed command gaps | Completion polling can saturate one core. Gaps are not proof of hardware idle or CPU causality; check library fallback explicitly. |

Calculate service sums and interval unions separately; report overlapping work
and uncovered intervals. Name exact observed windows. An attention envelope is
not the whole prefill, and startup/decode/teardown must not silently enter its
utilization denominator. Do not bridge unrelated host/device clocks by assumption.

Measure resource capacities for the operations actually used. Label older-stack
or different-shape figures as references. Use controlled changes to distinguish
competing explanations: lower bytes, better overlap, fewer submissions or better
arithmetic should predict a measured dependency-path change. Record uncertainty;
"GPU compute commands dominate" does not identify the resource inside kernels.

Do not declare counters unavailable solely from a policy file. Test bounded
capability enumeration without changing permissions, then separately qualify
activation/collection and interpretation. Conversely, enumeration does not prove
collection works. Preserve API failures and stop before unqualified model runs.

Keep numerical/lifetime/fault gates, physical262144 and RESTORE requirements.
Serialize builds/tests/GPU/cleanup, fix provider controls before native exec,
and follow [artifact retention](ARTIFACT_RETENTION.md). Commit compact evidence
and all samples used by decisions; retain large captures only for a named current
question with a finite budget and review point.
