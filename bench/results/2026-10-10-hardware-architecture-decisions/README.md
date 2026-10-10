# Decisions enabled by the hardware capacity controls

The measured capacity map and original samples are frozen in the preceding commit. This worksheet distinguishes the effect of submission order from data reuse. It adds no model/GPU measurement and adopts no candidate.

For32768positions, the1000token/s budget is32.768seconds. Historical190.2GB expert payload at the separate attained6.447GB/s host-USM rate takes29.50seconds. Assigning every expert GEMM hypothetically M80/160/640 yields12.95/6.57/3.85seconds of GEMM arithmetic respectively. With serial execution and no other work, corresponding target payload budgets are127.79/168.89/186.43GB. These are conditional scenarios, not actual model times or physical lower bounds: the current route shape histogram, actual payload/path/dependencies and all remaining service must be measured. M8192hot is not an attainable assumption for every expert.

| Change | What can change | Evidence gate |
| --- | --- | --- |
| Reorder independent submissions | Queue overhead/overlap, with the same bytes and work | Exact operation timeline and dependencies; safe two-queue control had zero target overlap |
| Keep one layer's packed weights across all chunks | Fewer packed transfers and potentially fewer host copies | Selected source route, actual expert reuse, peak live VRAM, full-context numerical/state proof |
| Group more useful routed rows per weight load | GEMM attained rate and transfer/dequant amortization | Actual per-expert `ne` histogram and same-request correctness/latency |
| Keep intermediate rows in VRAM | Fewer intermediate H2D/D2H transfers | Full simultaneous allocation/KV budget, ownership/lifetime proof and full262144-position validation |
| Change packed representation/device kernels | Packed bytes or dequant/GEMM service | Exact numerical and full-context tests, separate prefill/decode samples |
| Group PLE reads by filesystem record | Fewer source calls but more returned logical bytes | Actual cache/reader geometry and service; cold census alone does not select an I/O policy |

Frozen layer-major source loads every expert's packed blob once per layer, then runs all token chunks while retaining those weights. Historical layer-specific packed sizes imply50,292,326,400bytes for the explicit512-expert/layer scenario, versus historical A190,240,998,400bytes (about3.78x less). The largest layer/max-size temporary allocation is1,363,148,800bytes. This shows that a processing-order change can change the byte budget substantially; it is different from submitting the same copies earlier.

The source also hands off FP32 H2560 intermediate rows across47layer boundaries. If all rows reside in RAM,32768positions require335,544,320bytes of host row storage and15,770,583,040bytes in each PCIe direction. The separate pageable H2D/D2H controls give a conditional6.41seconds for those residual copies, plus7.80seconds for the once-per-layer packed weights at the separate host-USM rate. Actual paths/chunk sizes/dependencies/otherwork are unmeasured here. Keeping all intermediate rows in VRAM removes these PCIe handoffs but uses the row storage in addition to layer cache, dequant scratch, KV, decode state and other allocations. The existing mixed/inplace source is not validated by this count.

At262144hypotheticalpositions, the residual store is2,684,354,560bytes and each direction's47-boundary payload is126,164,664,320bytes. The conditional pageable residual time scales to about51.32seconds. This count is not evidence that all these tensors/KV fit the10GiB device or that the candidate produces correct output at the full physical context boundary. Existing alternative numerical rejection and its reproducer remain current fixtures. No reset/driver safety setting is changed to improve overlap.

Host staging copies packed data; its CPU service must be measured separately from GPU IQ dequantization. The384selected IQ2_S/IQ4_NL experts in the CPU capacity fixture do not mean384experts in every production layer. The source RAM-copy thread default is max(2,min(4,hardware_concurrency/4)); with12logical CPUs that is3unless explicitly overridden. Source reads and their32thread/ring128 default are a different arm. Register-only0.818TFLOP/s does not determine this staging or quantized-dot utilization.

The next production attribution needs actual expert rows, bytes/source/cache hits and distinct host-stager, H2D, device dequant, GU/Down, PLE and graph/cache restoration service. Returned oneMKL events still need internal-operation coverage qualification before they are called entire service spans. Then compare an isolated change using fresh >=32K inputs and independent repetitions, and validate all262144physical positions before adoption.
