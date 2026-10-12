# Production prefill command-event measurement

On Arc B570 with LLVM7.1.1/Clang22.1/libSYCL9, the32K+GEN16 production diagnostic captured129820 valid events. The sixteen output IDs and all reported LP5 values exactly matched the selected baseline. Startup, normal QUIT, fresh integer health and kernel fault checks passed; no reset or retry occurred.

| Selected category | Commands | Event spans | Host submit wall |
| --- | ---: | ---: | ---: |
| Host-USM to device-USM copy |97732|69.651 s|11.092 s|
| Grouped XMX GU |5944|3.345 s|0.681 s|
| Grouped XMX Down |5944|1.662 s|0.628 s|
| Native GU dequant |10096|2.545 s|1.096 s|
| Native Down dequant |10096|1.961 s|1.094 s|

Copy request extents were200326113280 bytes. These are command-action spans, not physical PCIe bandwidth; different queues may overlap. Dense projections, GDN, QSA and other uncaptured work are outside this summary. Host submission covers the existing wrapper/API interval, excluding pointer classification and collector insertion. Query/aggregation took4.002 s. Engine prefill105936.4 ms and decode2555.9 ms belong to this instrumented diagnostic, not a clean speed distribution. See MODEL_RESULTS.json and each original category for exact values.

The transfer path and submission count now have a measured reason to take priority over grouped GEMM tile tuning. Copy cohorts and larger chunks remain candidates, with no implementation or speedup claimed. Next qualify quiet instrumentation overhead and the remaining dense/GDN/QSA leaf coverage, then test bounded native Copy2D cohorts with the existing per-slot last-reader/credit protocol.

Both free and contrib-llvm builds passed with no new lint finding after correcting the original failed lint. Host-only ASan/UBSan checks passed cap200001, unavailable profiling, negative host time and scope reset. Seven finite GPU fixtures passed exact opt-in values and the simulated DP4a configuration, including1280 outputs and16 canaries. Integrated graphics, NVIDIA, full smaller-model memory admission,262144-token input and RESTORE remain unverified.

Fresh remote main and peeled tags are unchanged: officialv0.1.43 at7695859a and Xexe0.1.42 atda3d4260. Prior pristine32K measurements and the upstream server completion atfcb86a77 remain the update evidence. The displayed engine label is stillxe0.1.40.2.2; use the Git revision and binary hash to identify this curated build. The selected normal model binary0691/ed65 remains unchanged.

Committed receipts retain individual outcomes, exact categories, token/LP5 fingerprints, provider/command/environment and fault/closure evidence. Duplicate protocol/event strings and full successful traces are removed from archival derivatives; original receipt hashes and derivations are recorded. Raw artifact cleanup follows under the exclusive measurement lock.
