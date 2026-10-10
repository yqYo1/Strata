# Safe runtime hardware concurrency controls

This probe uses two in-order queues on the same B570 device/context, with queue profiling disabled in the application. It copies independent next-slot host-USM buffers while running production-shaped FP16-input/FP32-output oneMKL GEMMs. Each copy-enabled batch moves four 8 MiB buffers. Small M80/M160 shapes run128 rotating-weight GEMMs; M8192 runs8. This is an independent capacity control, not Strata model or production ring performance.

## Closed clean repeats

A separate logged qualification and three fresh clean processes passed normal exit, output and fresh kernel-fault checks. All504 samples are preserved in the three process receipts and summary. Each cell has seven samples/process, with rotating mode order; seven rounds do not give exactly balanced position counts.

The following are medians of the three process medians, in milliseconds per batch.

| Role / M | Copy only | GEMM only | Forced serial | Concurrent submission |
| --- | ---: | ---: | ---: | ---: |
| GU /80 |5.263|5.146|10.366|10.323|
| GU /160 |5.250|5.201|10.438|10.376|
| GU /8192 |5.298|8.184|13.454|13.356|
| Down /80 |5.258|3.303|8.489|7.515|
| Down /160 |5.259|3.392|8.781|8.223|
| Down /8192 |5.270|5.176|10.439|10.352|

GU has little wall benefit. Small Down shapes have variable wall benefit, which alone does not prove device overlap. Final validation checks both copy buffers and the final surviving product; overwritten products are not checked. No optimization is adopted.

## Separate PTI qualification

The closed instrumented process used pinned unitrace with kernel and host-call logging. Source, binary, runtime, safe flags, owner lifetime, all output hashes and correctness/fault outcomes are in `profile/record.json`. Instrumented timings are excluded from the clean table.

All3330 GPU operations were reconciled to unique host append calls using flow IDs and timestamps. Source counts account for144 target8MiB copies,3168 GEMM kernels, and18 setup/validation operations. The full compact interval table preserves all3330 operations. Target copy/GEMM interval intersection is0ns; their closest cross-role gap is5084ns, above the conservative1us serialization guard. Both tracked device lanes are named `L0 Compute Engine<0,0>` by PTI. Pinned profiler source shows that this label uses queue group ordinal/index and recorded group properties, not a physical utilization counter. Both device flow directions anchor command_start, so D2H flow is not a completion timestamp.

This qualifies lack of device overlap in this specific independent probe with `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` and `EnableDirectSubmission=0`. It does not establish production engine coverage, identify the precise serialization cause, or justify changing either safety setting. Production oneMKL returned-event/internal-kernel coverage remains a separate qualification.

## Reproduction and retention

The controller stages are build, qualify, r1, r2, r3, profile; each stage writes a unique finite owned directory under the shared measurement lock. Historical receipts retain their original controller/source commits. The offline analyzer reproduces the full event join and intersection from the closed raw profile. The raw5.16MB timeline has a named next-audit consumer and review point in `profile/retention.json`; compact results are committed in Git.
