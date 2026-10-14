# Attained component capacities and the decision budget

Measured2026-10-10 on Ryzen5 5600X,128GB RAM and ArcB570. [Complete matrix and process ranges](component-capacity-matrix-v2.json). These operation-specific attained rates are a practical reference, not absolute hardware maxima.

| Component | Attained operation rate |
| --- | ---: |
| CPU FP32 registerFMA /6physicalcores |0.818TFLOP/s|
| RAM read /6cores |36.87GB/s|
| RAM non-temporal copy |37.58GB/s, logical read+write|
| Host packed-copy pool /3workers |13.74GB/s, one-way payload|
| HostUSM RAM→GPU /GPU→RAM |6.447 /5.643GB/s, one-way payload|
| Pageable RAM→GPU /GPU→RAM |4.605 /5.274GB/s|
| VRAM copy |330.07GB/s, logical read+write|
| GPU GU/Down GEMM /8192rows |52.69 /41.61TFLOP/s, FP16inputs/FP32output|
| GPU GU/Down GEMM /80rows, rotating8weights |13.16 /10.08TFLOP/s|
| CPU native22/20 fullservice /NT1→NT4 |90.98→38.74µs per expert-token, streaming fixture/pool|
| SSD sequential /random4KiB,16workers |2.266 /0.05099GB/s, same-fileZFSpath|

The observed external PCIe path isGen4x4. Ideal128/130-encoded data is7.877GB/s per direction before packet overhead; attained H2D is6.447GB/s. InternalGPUfunction05:00.0 reports2.5GT/x1; external link evidence comes from the upstream03:00.0/AMDbridge/rootport chain. Vendor memorybandwidth is380GB/s, versus attained logical-copy330GB/s. Different links/traffic conventions must stay separate. [Topology receipt and vendor reference](physical-bandwidth-reference.json). Manufacturer INT8TOPS cannot substitute for FP16 GEMM or quantized CPU capacity.

[Actual seven-pair GPU IQ expansion](../2026-10-10-gpu-iq-dequant-capacity/README.md) now has qualified three-process rates, about76–208µs per expert for isolated generic GU+Down materialization. Each expert writes9.83MB FP16 weights regardless of routed row count. Large denseGEMM rates exclude this cost. Useful repeated rows amortize expansion/copy, but generic capacity does not apply to MMQ/fused/peer branches. GPUus/expert and CPUus/expert-token have different denominators.

For32768positions,1000token/s allows32.768s total. Historical expert-copy190.241GB at the separate attained H2Drate gives29.50s. At ideal encoded PCIe rate those bytes would still require about24.15s, if they cross that link; protocol/othertransfers/compute are additional. This is not a currentroute measurement or a physicalmin derived from an attainedrate. Actual expertshapes, branch, copycounts anddependencies decide the budget.

Submission reordering preserves bytes/work. The safe two-queue control foundzero intersection of144targetcopies and3168GEMMs; it does not prove every futureengine cannot overlap. Scheduling needs matching engine timeline/dependencies. If overlap is unavailable, reducing repeated transfers/materialization and increasing useful rows per weightload changes the architecture budget.

Layer-major all512experts/layer would reduce historical190.241GB packedcopy to currentinspected-header50.292GB, about3.78times less, or7.80s at separateH2Drate. Generic dequant stillruns perchunk when selected; packed weightreuse is not decoded weightreuse. With residualD10240, all-RAM32K residual adds25.66s of conditionalpageable roundtrips over47boundaries. Serialweight+residual is33.46s beforecompute. An effective8KGPU prefix needs320MiB residualbacking and lowers hostresidual estimate to19.24s; inplace can skip itsD2D copies. Peak/numericalqualification is absent. [Corrected placement worksheet](mixed-residual-placement-estimates.json).

At262144positions the entire residualplane is10,737,418,240B=10GiB, larger than devicecapacity10,666,115,072B beforeanyother allocations. Currentlayer-major requiresresidentQSAKV; QSA/indexer/RoPEsubtotal is6,912,212,992B. Originaldecodecache, MTP, session/prefillscratch, denseweights, temporarylayercache andgraph/runtime can overlap. Cache/MTPrelease andRAMrestore require graphretirement, drainedqueues andrestoration. [Allocation audit andmetadata correction](../2026-10-10-parallel-round75/root-review.json). Partialplacement is a candidate, not a fitclaim.

For decode70token/s, whole emitted-token budget is14.286ms. The measured22/20CPU NT1..4 fixtures cannot be assigned to all480routeentries or theother6pairs. Grouping/verifieracceptance, GPUresidentcalls andactualCPUmisses needtheirown census. Prefillgrouping doesnot establishdecoderthroughput.

Nextqualify default-off hostroute accounting andmatched>=32K call/byte/ne/service evidence, thenselect thelargest critical-path gapagainst matchingcomponent capacity. Separate cleanlatency frominstrumented receipts. Fullphysical262144position/state/capacity qualification remainsrequired beforeadoption. No newmodelthroughput orperformance candidate isadopted here.
