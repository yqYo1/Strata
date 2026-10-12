# Production prefill command-event instrumentation

The opt-in diagnostic retains exact returned events for copies, native grouped dequantization and grouped GEMM. It enables queue profiling only for exact `STRATA_PREFILL_SERVICE_EVENTS=1`. It adds no production wait or profiling tag. The final existing checked sync precedes event queries; zero or invalid timestamps, caps and query failures make the diagnostic incomplete.

Both free and contrib-llvm builds passed with LLVM7.1.1, Clang22.1 and libSYCL9. The initial lint failed; Main corrected the empty catch and supplied actual SYCL compiler flags for the new header. The subsequent lint has no new finding. The original failure is retained.

The host-only ASan/UBSan fixture passed cap200001, overlapping-scope rejection, unavailable profiling, negative host clock and scope reset checks. Seven finite GPU fixtures passed on Arc B570: profiling1 in both modes, unset/0/01/true in free mode, and the simulated smaller configuration with XMX disabled. All1280 outputs and16 canaries matched; all captured command timestamps were positive. No fresh GPU fault or reset occurred. The smaller fixture checks DP4a and queue behavior; full smaller-model memory admission, integrated graphics and NVIDIA remain unverified.

The current selected model binary remains0691/ed65. The new instrumentation has no32K model qualification or speed result at this boundary. Its counts cover selected leaf commands, and event spans can overlap. They do not establish physical PCIe bandwidth, total GPU work or idle gaps. Next: candidate startup, then one supervised32K event diagnostic with exact16-token/LP5 comparison.262144-token input and RESTORE remain unqualified.

Receipts, source/controller pins and original reports are archived here. Raw successful logs will be reviewed after the model boundary under the measurement lock.
