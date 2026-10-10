# Measured hardware capacity and workload limits

Measured2026-10-10 on Ryzen5 5600X(6physical cores),128GB RAM and Arc B570. CPU and GPU capacity controls each used three fresh processes with seven samples/cell. Values below are medians of process medians; all individual samples and process ranges are in [the validated summary](../2026-10-10-parallel-round54/hardware/validated-three-process-summary.json). These are attained effective rates for the stated operations, not absolute hardware maxima. CPU power/governor policy and the previously qualified GPU safety settings were left unchanged.

| Component / operation | Measured rate | Scope |
| --- | ---: | --- |
| RAM read |36.87 GB/s|6physical cores;512MiB arrays exceed LLC|
| RAM non-temporal copy |37.58 GB/s|Logical read+write bytes|
| RAM cached copy |22.95 GB/s|Logical read+write bytes|
| RAM→GPU host USM |6.447 GB/s|256MiB, one-way payload, host-inclusive completion|
| GPU→RAM host USM |5.643 GB/s|256MiB, one-way payload|
| RAM→GPU pageable host |4.605 GB/s|256MiB; different memory path|
| VRAM copy |330.07 GB/s|256MiB, logical read+write bytes|
| GU GEMM /M8192 |52.69 TFLOP/s|Hot FP16 inputs, FP32 output;N1280,K2560|
| Down GEMM /M8192 |41.61 TFLOP/s|Hot FP16 inputs, FP32 output;N2560,K640|
| GU GEMM /M80 rotating8weights |13.16 TFLOP/s|Actual production-shaped kernel geometry, synthetic inputs|
| Down GEMM /M80 rotating8weights |10.08 TFLOP/s|Process medians10.04–14.21; variation retained|
| GU GEMM /M160 rotating8weights |25.78 TFLOP/s|Synthetic row grouping, not measured production row histogram|
| Down GEMM /M160 rotating8weights |20.03 TFLOP/s|Process medians19.19–22.55|

Different logical traffic conventions are explicit: RAM/VRAM copy counts both read and write; PCIe rates count transferred payload once. These are not directly comparable utilization ratios. Eight rotating weights do not prove cold caches. Large-shape rates cannot be applied to each small expert GEMM.

## What changes the architectural estimate

At32768 tokens,48layers,topK10,H2560 andFF640, routed GU/Down dense arithmetic is154.619TFLOP(4.718592GFLOP/token), excluding attention, shared experts, dequantization and other work. A1000token/s target therefore requires4.719TFLOP/s for these products, but attained expert shape and dependencies matter much more than the large GEMM peak.

An older matched32K route reported about190.2GB of logical expert copies. Holding that volume fixed, the attained host-USM rate6.447GB/s gives about29.5seconds of transfer. The1000token/s total budget is32.768seconds. This is a conditional estimate from a prior route and a separate capacity control, not the latest model's measured DMA service or a mathematical physical minimum. The current route's bytes, shapes and reuse must be reconciled before predicting speed.

The separate two-queue control found essentially serial GU wall time, and its PTI trace found zero intersection of144target copies and3168GEMM kernels with the safe runtime settings unchanged. Reordering submissions cannot be assumed to hide transfer at those settings. A scheduler change needs actual engine-overlap evidence; otherwise reducing transferred bytes, increasing useful work per weight load, or changing the data representation is required to recover that budget. Layer-major prefill already retains each loaded layer across token chunks, so an extra reuse claim must account for existing reuse.

RAM bandwidth alone does not characterize the native quantized CPU decoder. NT1 dispatches a different fallback from groupedNT2–4. The independent emitted-code audit found NT4 vector accumulator stack spills, but did not measure their cost. Grouped actual-payload hot/streaming service and worker-tail data remain to be qualified; no CPU instruction ceiling or decoder impossibility is claimed. PLE's foreground gather/wait markers also do not measure storage service: an actual row/page trace and matching read-only storage controls remain needed. Those missing capacities are explicit open work.

The existingDown phase-marker duration includes host waits and must not be equated to exclusive GPU execution. A default-off production returned-event ledger has passed host contracts, but still needs SYCL build, small device qualification, profiler coverage and a matched32K trace. Full physical262144-position validation is a separate candidate gate; a configured context length is not that validation.


The separate diagnostic source7c5ad75a subsequently passed the root-owned isolated SYCL build and119float B570 layout/completion/profiling-field check. Exact receipts are committed on `diag/sycl-prefill-service-qualification-20261010`; production oneMKL internal-kernel span, matched32K route service and physical262144 positions remain unqualified.
