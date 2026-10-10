# Measured hardware capacity and workload limits

Measured2026-10-10 on Ryzen5 5600X(6physical cores),128GB RAM and Arc B570. RAM and GPU controls and register-FMA controls each used three fresh processes with seven samples/cell. Native quantized CPU controls used 24 fresh processes (four group sizes, hot/streaming, three repeats), with 25 samples per arm/cohort. Same-file storage controls used 15 fresh processes (five patterns, three repeats). Values below are medians of process medians; all individual samples and process ranges are in [the validated summary](../2026-10-10-parallel-round54/hardware/validated-three-process-summary.json). These are attained effective rates for the stated operations, not absolute hardware maxima. CPU power/governor policy and the previously qualified GPU safety settings were left unchanged.

| Component / operation | Measured rate | Scope |
| --- | ---: | --- |
| CPU register FP32 FMA |0.818 TFLOP/s|6 physical cores; process medians 0.793–0.819; 12 independent AVX2 accumulators|
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

| SSD sequential / 1 MiB |2.266 GB/s|16 workers; application payload; range 2.137–2.273|
| SSD random / 4 KiB |0.05099 GB/s; 12,450 IOPS|16 workers; same-file ZFS path; range 0.05055–0.05127|
| SSD random / 4 KiB |0.05307 GB/s; 12,958 IOPS|64 workers; range 0.03706–0.05313; not SSD queue depth|

CPU operation details and all samples are in [CPU capacity](../2026-10-10-cpu-capacity/README.md); storage patterns, three repeats and whole-process counter boundaries are in [same-file storage capacity](../2026-10-10-storage-capacity/README.md).

Different logical traffic conventions are explicit: RAM/VRAM copy counts both read and write; PCIe rates count transferred payload once. These are not directly comparable utilization ratios. Eight rotating weights do not prove cold caches. Large-shape rates cannot be applied to each small expert GEMM.

## What changes the architectural estimate

At32768 tokens,48layers,topK10,H2560 andFF640, routed GU/Down dense arithmetic is154.619TFLOP(4.718592GFLOP/token), excluding attention, shared experts, dequantization and other work. A1000token/s target therefore requires4.719TFLOP/s for these products, but attained expert shape and dependencies matter much more than the large GEMM peak.

An older matched32K route reported about190.2GB of logical expert copies. Holding that volume fixed, the attained host-USM rate6.447GB/s gives about29.5seconds of transfer. The1000token/s total budget is32.768seconds. This is a conditional estimate from a prior route and a separate capacity control, not the latest model's measured DMA service or a mathematical physical minimum. The current route's bytes, shapes and reuse must be reconciled before predicting speed.

The separate two-queue control found essentially serial GU wall time, and its PTI trace found zero intersection of144target copies and3168GEMM kernels with the safe runtime settings unchanged. Reordering submissions cannot be assumed to hide transfer at those settings. A scheduler change needs actual engine-overlap evidence; otherwise reducing transferred bytes, increasing useful work per weight load, or changing the data representation is required to recover that budget. Layer-major prefill already retains each loaded layer across token chunks, so an extra reuse claim must account for existing reuse.

Native quantized CPU capacity was measured with the actual selected IQ2_S/IQ4_NL payloads. In the streaming direct complete-chain fixture, NT1 costs 286.47 microseconds/expert-token versus 148.36 at NT4; the fixed five-worker-plus-host pool costs 90.98 versus 38.74. Grouping already improves attained per-token service substantially. This is not whole-model decoder latency or a physical-DRAM/cycle census. Hot/streaming use different expert subsets and job counts, and native quantized dense-equivalent FLOPs cannot be divided by register FP32 FMA capacity to infer CPU utilization. The NT4 emitted stack spills remain a source fact with unmeasured cost; NT4 is still the fastest tested per-token group. No decoder impossibility follows from logical packed bytes or RAM bandwidth alone.

PLE's foreground gather/wait markers do not measure storage service. The same-file control now shows about 2.27 GB/s for 1 MiB requests but only 0.051 GB/s for 4 KiB random requests at 16 workers; 64 workers offers a small median difference with a much wider range. All random cells show approximately 32 direct-DMU bytes per application byte, versus approximately one for aligned sequential requests. These pool-global whole-process counters support testing filesystem-record coalescing; they do not attribute NAND bytes or prove the actual PLE route's bottleneck. Reordering tiny jobs and increasing workers cannot be assumed to recover the contiguous-read rate. Actual PLE reader statistics and duplicate-record census remain necessary.

The existingDown phase-marker duration includes host waits and must not be equated to exclusive GPU execution. A default-off production returned-event ledger has passed host contracts, isolated SYCL build and small device qualification, but still needs profiler coverage and a matched32K trace. Full physical262144-position validation is a separate candidate gate; a configured context length is not that validation.


The separate diagnostic source7c5ad75a subsequently passed the root-owned isolated SYCL build and119float B570 layout/completion/profiling-field check. Exact receipts are committed on `diag/sycl-prefill-service-qualification-20261010`; production oneMKL internal-kernel span, matched32K route service and physical262144 positions remain unqualified.
