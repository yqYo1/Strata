Terminal evidence and retention review — 2026-10-09

T telemetry: four fresh 32768-token reads, exact head/live state/output/LP/MTP checks pass; normal exit 0, no new GPU fault, no forced cleanup/survivor. No full-context or speed qualification.

| Read | Windows | GU ms/window | FF quant ms/window | Down ms/window | PP diagnostic ms | Decode diagnostic ms |
| --- | --- | --- | --- | --- | --- | --- |
| A-read0 | 23 | 111.743 | 0.389 | 41.691 | 81391.9 | 4688.4 |
| B-read1 | 26 | 111.869 | 0.675 | 40.617 | 74448.1 | 4842.5 |
| A-read2 | 23 | 113.595 | 0.349 | 42.321 | 74544.2 | 4272.0 |
| B-read3 | 26 | 116.66 | 0.36 | 43.898 | 74411.7 | 5001.4 |

The diagnostics run with validation/logging. These are individual diagnostic samples, not clean performance estimates. Host phases overlap GPU work; packed byte counters are submitted bytes, not measured DRAM traffic.

C physical-context repeat remains rejected for numerical differences. Keep the original failed receipt and current divergent/reference captures for localization. Do not run the clean C controller until its full lifecycle gate is genuinely qualified.

Both profiler smokes remain failed. VTune rejected ptrace scope 1 before target work. gprofng completed CPU work but reports itimer initialization failure and empty frame data; no model profiling is admitted.

Research round 17 is source-only: paired activation loads and dispatch-cost-aware task shaping are conditional hypotheses requiring actual format/NT and worker-tail evidence.

Artifact policy is published at feature/sycl commit 4fa478f50f8da396e651e3ae862f8b1849d49b54 in AGENTS.md and docs/ARTIFACT_RETENTION.md. The compact receipts here preserve the measured results, conditions, rejected outcomes and open questions. Redundant successful T tensors have no new current RESTORE/comparison consumer after extraction; canonical baseline fixtures remain.

Next concrete work: resolve timer collection feasibility without changing system settings; build and CPU-validate corrected repeat capture before serial full-context diagnosis. Root owns all execution.
