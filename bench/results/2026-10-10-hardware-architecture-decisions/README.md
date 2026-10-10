# Hardware capacities and the architectural budget

Measured on Ryzen 5 5600X, 128GB RAM and Arc B570 on 2026-10-10. The [capacity matrix](component-capacity-matrix.json) pins individual samples, units and three-process ranges. These are attained rates for specified operations, not absolute hardware maxima or new model throughput.

| Component / operation | Attained rate | What the number covers |
| --- | ---: | --- |
| CPU FP32 FMA | 0.818 TFLOP/s | Six physical cores, register arithmetic |
| RAM read | 36.87 GB/s | Six physical cores, arrays larger than LLC |
| RAM copy | 37.58 GB/s | Non-temporal, logical source read plus destination write |
| Packed host-copy jobs | 13.74 GB/s | One-way payload, three workers; ordinary RAM, source/copy/wakeup/ack control |
| RAM→GPU | 6.447 GB/s | Host USM, one-way 256MiB payload, host-inclusive completion |
| GPU→RAM | 5.643 GB/s | Host USM, one-way 256MiB payload |
| Pageable RAM→GPU / GPU→RAM | 4.605 / 5.274 GB/s | The path used in the residual estimate below |
| VRAM copy | 330.07 GB/s | Logical source read plus destination write |
| GPU GU / Down GEMM, M8192 | 52.69 / 41.61 TFLOP/s | FP16 inputs, FP32 output, hot weights |
| GPU GU / Down GEMM, M80 | 13.16 / 10.08 TFLOP/s | Eight rotating weights, synthetic useful row grouping |
| SSD sequential, 1MiB | 2.266 GB/s | Same-file ZFS path, 16 workers |
| SSD random, 4KiB | 0.05099 GB/s; 12,450 IOPS | Same-file ZFS path, 16 workers |

RAM/VRAM copy counts read and write; PCIe and host packed-copy rates count payload once. They are not directly comparable utilization ratios. Hot large GEMM rates cannot predict small expert service. Register FP32 FMA does not set the native quantized CPU dot-product roof. Device IQ dequantization and actual model service remain gaps.

For 32,768 positions, the 1000 token/s budget is 32.768 seconds. Historical 190.2GB expert payload at the separate attained 6.447GB/s host-USM rate takes 29.50 seconds. Assigning every expert GEMM hypothetically M80/160/640 yields 12.95/6.57/3.85 seconds of GEMM arithmetic respectively. With serial execution and no other work, corresponding target payload budgets are 127.79/168.89/186.43GB. These are conditional scenarios, not measured model times or physical lower bounds. Current route shape/format/source histograms, actual bytes and all remaining work must be measured.

| Change | What can change | Evidence needed |
| --- | --- | --- |
| Reorder independent submissions | Queue overhead and overlap with unchanged bytes/work | Exact operation timeline and dependencies; the safe two-queue control had zero target overlap |
| Keep one layer's packed weights across all chunks | Fewer packed transfers and host copies | Actual route/reuse, simultaneous memory peaks and full-context state/numerical validation |
| Group useful routed rows per weight load | Attained GEMM rate and transfer/dequant amortization | Actual per-expert row histogram, correctness and latency |
| Keep some intermediate rows in VRAM | Fewer residual H2D/D2H transfers | Simultaneous allocations/KV budget and full-context lifetime/numerical validation |
| Change packed representation/device kernels | Packed bytes or dequant/GEMM service | Operation-specific attribution, numerical tests and separate prefill/decode comparisons |
| Group PLE reads by filesystem record | Fewer source calls but potentially more returned bytes | Actual cache/reader geometry and service; the cold census alone does not select I/O policy |

The frozen layer-major source loads every expert's packed blob once per layer and runs all chunks while retaining that layer. Historical layer-specific sizes imply 50,292,326,400 bytes for the explicit 512-expert/layer scenario, versus historical A's 190,240,998,400 bytes: about 3.78 times less. Its largest historical temporary layer-cache allocation was 1,363,148,800 bytes. These are historical pack counts, not current-pack metadata or evidence that the alternative is adopted. Reuse changes the byte budget; submitting the same copies earlier does not.

The residual plane has **D=10240 FP32 elements per position**: expert hidden N=2560 times four streams. The previous worksheet incorrectly used N for this plane. Its preserved original commit/hash and the fourfold correction are in [the correction receipt](residual-dimension-correction.json); original benchmark results are unchanged.

| Positions | Complete residual plane | Payload per PCIe direction over 47 boundaries | Conditional all-RAM residual copies | Once-per-layer packed copies | Serial copy sum |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 1,342,177,280 B (1.25GiB) | 63,082,332,160 B | 25.66s | 7.80s | 33.46s |
| 262,144 | 10,737,418,240 B (10GiB) | 504,658,657,280 B | 205.27s | 7.80s | 213.07s |

These combine source counts with separate pageable H2D/D2H and host-USM capacity measurements. Actual chunk-sized service, cache/source path, copy overlap and other work are not measured here. Under this serial scenario, all-RAM residual plus weight reuse alone already exceeds the 32K target budget before computation. Partial GPU placement can lower residual PCIe bytes but consumes VRAM; grouping/reuse must therefore be evaluated with placement, not from weight savings alone.

Current layer-major explicitly rejects streamed QSA K/V (`kv_mode != 0`). For the cited geometry its resident K/V, indexer and RoPE subtotal is 864,038,912 bytes at 32K and 6,912,212,992 bytes at 262,144 positions. The latter leaves 3,753,902,080 bytes below the whole-device capacity of 10,666,115,072 bytes, before GDN, all weights, caches, prefill/dequant/session scratch, graphs and runtime. This is not free memory. The complete 10GiB residual plane alone exceeds that device capacity by 71,303,168 bytes. Full GPU placement cannot fit; mixed placement has no complete admission receipt yet. Historical streamed-KV C is a different, ineligible route for this source. Existing alternative numerical rejection remains unresolved.

The packed host-copy control reaches 13.74GB/s with three workers, with no clear improvement from four or six. This is ordinary aligned RAM with a synthetic acknowledgment, not production host-USM/DMA/source service. Production allocation kind, actual source path and dependencies still need attribution. The current host-copy default is three workers on 12 logical CPUs; source-reading threads are a separate setting.

Next measure actual expert rows, role formats, bytes/source/cache hits and distinct host stager, H2D, GPU IQ dequant, GU/Down, PLE and graph/cache restoration service. Returned oneMKL events need internal-operation coverage qualification before being called complete service spans. Then compare isolated changes with fresh inputs of at least 32K and independent repetitions; qualify all 262,144 physical positions before adoption. GPU safety settings are unchanged and no candidate is adopted by this worksheet.

The [mixed-placement worksheet](mixed-residual-placement-estimates.json) gives the host/device copy tradeoff. With an effective 8K GPU prefix in a 32K call, GPU backing is 320MiB and the conditional host residual time falls from 25.66s to 19.24s. Ordinary placement adds about 0.19s of device copies under the separate attained VRAM-copy rate. Enabling the existing in-place flag also aliases GPU-prefix compute in a mixed layout and skips those device copies; only scratch allocation reuse requires all rows on GPU. This corrects R249's overly narrow all-GPU-only alias claim. Both placements require numerical and memory admission checks.

The [current-pack geometry reconciliation](current-pack-memory-geometry-reconciliation.json) also corrects R248's head decomposition: current metadata/source have two KV heads of width256, rather than four of width128. Their product is the same, so the quoted FP16KV and QSA subtotal bytes remain unchanged; other head-sensitive formulas must use the actual geometry. Source GDN recurrence and convolution history add 113,246,208 and 4,423,680 bytes before other buffers. The current local pack census verifies seven paired formats and MAXBLOB2,662,400; it is a header inspection, not a runtime admission result.
