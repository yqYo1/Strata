# Same-file read-only storage capacity control

Original IQ4_NL PLE shard, tableoffset192,28800138240bytes. KernelZFS2.4.1/userspace2.2.2; datasetcompressionoff/recordsize128KiB/primarycachemetadata/secondarynone, NVMeCSSD-M2L2TRGAXN2000GB firmwareVC2S038D, PCIeGen3x4. Facts pinned in metadata. No property, power/cache, data or device state changed.

Source-only Sol handoffv1 is preserved unchanged. Rootv2 changes only sequentialstart alignment to1MiB and associated bounds/omitted-prefix receipts before anyfile run, avoiding extra boundary-record reads from the4KiB start. Root compiled source, verifiednoGPUdependencies and serially qualifiedall5shapes: sequential1MiB threads1/16, random4KiB threads1/16/64. All fullread counts/guards/finalsurvivingbuffer exact equality/fileidentity/normalexit/ownerclosure pass. These short qualifiers emit no rateclaim. DirectDMU byte counters are1x alignedsequential or~32xrandom, a wholeprocesspoolwide observation, not media service.

Full capacity measurements require3freshprocesses percell. The source measures queuehandoff/pread/latencyclock bookkeeping/canarychecks/joins; finalbuffer comparisons/hashes/output areuntimed. Every read returnsfullbytes, only finalsurvivingbuffer perworker getsbyteequality. Activeworker highwater isnot physicalSSDqueue depth. Wholeprocesspool/block counterdeltas includestartup, untimedbufferedreferences andunrelatedactivity; preserveARC/direct/leaf views separately, no subtraction orNAND claim. This standalone uniform/contiguous workload isnot actualPleReader, actual32Kroute, model performance or absolutehardwaremaximum.

## Closed repeated capacities

All 15 fresh processes closed PASS with unchanged source, binary, shard, boot and properties. Repeat 2 reversed the cell order; all random cells use the same 65,536 replacement offsets and all sequential cells the same 27,464 planned 1 MiB offsets. No cache state was reset. Values are medians and ranges of three whole-process application rates. Full original receipts are preserved byte-identically in `measure/`; structured summary retains each measurement and latency quantile.

| Read pattern | Workers | Payload GB/s median [range] | Logical IOPS median [range] |
| --- | ---: | ---: | ---: |
|sequential / 1 MiB|1|1.150462 [1.129518, 1.206633]|1097.2 [1077.2, 1150.7]|
|sequential / 1 MiB|16|2.265974 [2.137457, 2.272824]|2161.0 [2038.4, 2167.5]|
|random / 4 KiB|1|0.009112 [0.009080, 0.010243]|2224.6 [2216.8, 2500.6]|
|random / 4 KiB|16|0.050993 [0.050550, 0.051273]|12449.6 [12341.3, 12517.9]|
|random / 4 KiB|64|0.053074 [0.037055, 0.053134]|12957.6 [9046.6, 12972.1]|

The 64-worker first repeat is retained, including its lower 0.037055 GB/s and longer tail; it is not discarded as an outlier. Increasing 16 to 64 workers does not establish a reliable gain. Worker high-water values are worker `pread` regions, not SSD queue depth.

Every cell reports exactly 131,072 bytes per counted direct-DMU read. Whole-process direct-DMU bytes / timed logical payload is 31.9956–31.9971 for 4 KiB random reads, and 0.999964–0.999977 for aligned sequential reads. These are pool-global counter observations over the whole supervised process; they are not timed per-file physical read amplification or NAND traffic. ARC and leaf-byte deltas remain separately recorded and unattributed. Pinned upstream 2.4.1 source suppresses predicted DIO data prefetch, but can still issue ARC indirect-metadata reads; it does not attribute this run's ARC bytes.

This supports testing record-coalesced PLE reads as a hypothesis. Sorting page jobs alone is not proof that each filesystem record is read once. Actual PLE row-cache hit rate, duplicate records, page/straddler shapes, reader return bytes, critical-path wait and matched 32K model correctness/performance must be measured before adopting a change. No source, runtime or production tuning was adopted by these controls.
