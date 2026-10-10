# IQ4NL inclusive host-wall component comparison

Source984fb5f7e7762ea66c3bbd023021c3c4bd3e7cec; B57010GiB PCI0000:05:00.0, Ryzen5600X, oneAPI2026.1, pinnedURLevelZerov2, in-order queue without profiling. Incremental build and logged qualification passed. Existing library and numerical fixture hashes remained unchanged. Native qualification contains384 generic and384 private launches, all6400groups/local32, zero event profiling/native timestamp queries.

Three fresh clean processes each passed all guards/input immutability/independent-RNE pre/post replay, exact-event metadata, known-completion checks and normal queue/USM teardown. Each has8 paired blocks per regime,32 timed cells,64 calls/cell,2304 totalreceiptcalls,160batches,2560 successfulwrappers and324waits. Every owned process/session closed and reaped normally; complete kernel journal intervals show no new GPU fault or dump. No model was opened.

| Address regime | Generic median64-call wall | Private median64-call wall | Private minus generic aggregate wall | Descriptive95% hierarchical bootstrap |
|---|---:|---:|---:|---:|
| Same address |1.274708ms|1.2701595ms|-0.25714%|[-0.77737%,+0.34229%]|
|64-address rotation|1.382606ms|1.366972ms|-1.23719%|[-1.55528%,-0.90795%]|

Rotation per-process differences are-1.23857%,-1.18094%,-1.29150%;21/24 paired blocks shorter. Same-address19/24 shorter but interval includes zero. Bootstrap resamples whole fresh processes (3) and paired blocks within each (8),20000replicates,seed197196; only3 process clusters and fixed order families limit inference. Calls within a batch are not independent trials.

The metric brackets host submission loop through normal completion-wait return, including host clock reads and exact-event retention. It excludes post-wait logging/queries/readback/numerical checks. Williams0132,1203,2310,3021 orders repeat twice: position and directed predecessors balanced within blocks; cross-block predecessors recorded but not universally balanced. Full64-input/2-output D2H outside each measured interval conditions subsequent cache history. No pure-device-service/cold-cache/uninstrumented/model claim.

Decision: rotation component result supports an actual caller census and real-model qualification, not adoption. Same-address improvement is unestablished. Required actual model>=32768batchedpositions, physical262144 lifecycle/tailcontrols and independent repeated decode remain pending. Original device-profile start-before-submit failure remains FAILED at fcf0f04e; this independent metric does not repair it.

Compact derivative receipts preserve original statuses, command/source/binary/runtime/environment/ownership/fault evidence, all96 cell samples and all6144 individual timed submit durations. Original hashes are listed; repeated per-call pointer/tuple text need not remain after its strict validator and native-trace consumers have closed. Source-based strict transcript parser/controller are included.
