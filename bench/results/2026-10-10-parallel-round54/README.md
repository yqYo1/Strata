# Measured hardware capacity and the remaining service attribution

Measured on the Ryzen 5 5600X (6 cores, 12 threads, 128 GB RAM) and Intel Arc B570 at
0000:05:00.0. Three fresh CPU processes and three fresh GPU processes ran serially,
with seven measured samples per case in each process. All six processes passed
full output/checksum validation and exited normally with closed owned sessions.
The GPU runs recorded no new visible xe fault or devcoredump. Direct submission
and Level Zero v2 copy offload stayed disabled, as in the qualified production
configuration. See [all samples](hardware/validated-three-process-summary.json)
and the original build/qualification/source records in
[round53](../2026-10-10-parallel-round53/README.md).

These are observed effective rates for specific workloads, not absolute hardware
maxima. The table uses the median of the three process medians; the range is the
range of those medians. GB and TFLOP use decimal units.

| Resource and operation | Effective rate | Process median range |
| --- | ---: | ---: |
| RAM read, six physical cores, 512 MiB arrays | 36.87 GB/s | 36.74–37.69 |
| RAM non-temporal copy, six cores | 37.58 GB/s | 36.81–38.24 |
| RAM cached copy, six cores | 22.95 GB/s | 22.36–23.44 |
| Host USM to VRAM, 256 MiB per call | 6.447 GB/s | 6.446–6.449 |
| VRAM to host USM, 256 MiB per call | 5.643 GB/s | 5.6428–5.6430 |
| Pageable host memory to VRAM, 256 MiB per call | 4.605 GB/s | 4.590–4.612 |
| VRAM queue copy, 256 MiB buffers | 330.07 GB/s | 329.76–330.12 |
| FP16-input/FP32-output GU GEMM, 8192 rows | 52.69 TFLOP/s | 52.67–52.79 |
| FP16-input/FP32-output Down GEMM, 8192 rows | 41.61 TFLOP/s | 41.56–41.77 |

Copy rates count source reads plus destination writes, except PCIe rates which
count the one-way payload. Cached-copy traffic does not include write allocation;
these are logical-byte rates, not memory-controller counters. The RAM read has a
reduction dependency. GPU durations include host submission and completion wait;
profiling is disabled. Synthetic GEMM inputs validate every output exactly and
use production dimensions/layout/dtypes, but do not reproduce production data or
the application's wrapper overhead. Eight rotating weights do not prove cold
cache residency. Independent capacity tests also do not establish performance
when CPU staging, DMA and GPU compute run simultaneously.

## Shape and work determine which rate is relevant

The production GU shape has 1280 outputs and K=2560; Down has 2560 outputs and
K=640. With eight input rows, hot GU process medians are 1.92–2.70 TFLOP/s and
Down 1.22–1.48 TFLOP/s. Down also varies between processes at 80 rows: hot
9.86–14.40 TFLOP/s and rotating 10.04–14.21. Preserve this variation rather than
turning one process into a universal service estimate. The actual expert-row
histogram still needs measurement.

For 32768 positions, 48 layers and top-10 routing, the source-defined expert
dense-equivalent work is 103.079 TFLOP GU plus 51.540 TFLOP Down. This excludes
attention, dequantization, shared experts, grouping and other work. Sustaining
1000 tok/s would require about 4.72 TFLOP/s for these two expert products, but
the large-row GEMM rate cannot stand in for small routed products.

A historical same-route 32K receipt logged about 190 GB of queued expert-copy
payload. If that payload stays unchanged, 6.447 GB/s implies about 29.5 seconds
of copy service against a 32.768-second budget at 1000 tok/s. It is a conditional
estimate: the current diagnostic did not collect its own copy-byte/event ledger,
and queued payload is not physical DRAM traffic. At hypothetical all-80-row
rotating GEMM shapes, the three capacity probes imply 11.5–13.0 seconds of GU
plus Down service. Serializing those products with the copy estimate exceeds
the target; overlap, reuse or lower transfer volume would be necessary under
that assumed workload. Those shapes are probe points, not observed route counts,
and measured probe rates are not mathematical hardware upper bounds.

For decode, the existing same-request pool0 diagnostic reports restored CPU
dispatch spans of 145.04 and 145.81 ms/window, largely GU plus Down, versus a
36.57 ms/window budget for 64 outputs in 25 windows at 70 tok/s. Its approximately
3.18 GB/window expert-blob extent is logical submitted work, not measured RAM
reads. Neither these nested host spans nor division by the RAM rate proves an
architectural limit. Actual format/NT counts, worker-active/tail/wait intervals
and physical traffic remain missing. The closed actual-weight NT1 calibration
is useful component evidence; it does not qualify NT2–4 service or a CPU peak.

## Valid prefill intervals from an overall failed controller

The timer guard at source commit 185fa780 passed its GPU-free extracted-source
contract tests and isolated compile/link. With timing enabled, the 32769-token
request performed 32768 batched positions; fresh and two checkpoint-restored
64-output requests each matched baseline IDs, logprobs, MTP and first heads.
All 66 fresh state parts also matched. The marker ledger admitted all 469556
marks and 469555 intervals, with 939110 successful queries and no missing,
backward or incomplete interval.

The original whole controller remains **failed**: after normal exit it compared
the current source commit against a dictionary frozen before the diagnostic
build override. V1 separately failed an environment assertion before model
launch. Both original receipts are preserved byte-for-byte. An
[independent audit](prefill/independent-evidence-audit-v1-record.json) rechecked
source/build/binary pins, outputs, finite heads, every state-part hash, process
closure, stderr and the full phase ledger. It admits the narrow diagnostic
observation without rewriting either original result as a pass. V3 repairs the
expected-commit capture and passed a CPU-only preflight; the model was not
repeated merely to correct metadata.

The admitted marker timeline is 86.687 seconds. Down-labelled intervals account
for 32.893 seconds, QSA attention 10.697, dequantization 5.973, PLE read wait
5.943 and GU 4.834. **These are marker-to-marker intervals, not exclusive kernel
execution.** Static source review confirms host staging/lookahead delay before
the next marker can charge compute-queue idle to the preceding Down label.
The 0.850-second wait-copy interval is a compute dependency span, not DMA service.
Do not infer low PCIe utilization or GPU-compute saturation from these labels.
Profiling/validation makes this run ineligible for clean speed comparisons.

## Next work and retention

Hold new optimizations until actual GEMM service, copy bytes/service, host-submit
gaps and route geometry are joined for the same request. Existing Gemm::f16 and
DPCT gemm return void, so capturing their oneMKL event needs an explicit diagnostic
channel or a qualified profiler. The existing transfer timer also needs timestamp
and complete-ledger validation before its data is authoritative. Then use
timer-off wall comparisons to determine whether reordering removes exposed idle,
or whether service demand requires different kernels, residency or data volume.
Keep CPU native service by format and NT separate from pool scheduling time.

No optimization is adopted here. A full 262144-position input boundary remains
an adoption requirement; these 32K diagnostics do not inherit that qualification.
The returned R210–R213 reports were fully read, archived and corrected separately;
registry v77 contains 320 reviewed research reports. Source, commands, environment,
every sample, original failures and decisions are committed. Redundant successful
candidate tensor captures can retire after this archive is committed; keep the
canonical baseline head/state and matched32832 session for future comparisons.
