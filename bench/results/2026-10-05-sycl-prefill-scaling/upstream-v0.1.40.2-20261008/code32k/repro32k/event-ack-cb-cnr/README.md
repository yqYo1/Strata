# Actual-DMA completion candidate with the32K CNR settings

This uses the same private binary4b805390b3f732d21607f846826ee54c2f36eb650b2ddc611d087626938e12bf
as the earlier event-status-query candidate. Only the process-local implicit
counter conversion setting and documented MKL_CBWR=AUTO are added, as in the
five completed production controls. Model input/configuration remain32,768
tokens,8192-token chunks, normal MTP4, int8 KV and explicit128cache slots.

The logged first process completes64finite-logprob outputs and normal exit,
and matches the production CNR logged control in all66prefill state parts,
all248,320 first-head float bytes, all64IDs and every protocol logprob. No new
xe fault is recorded. This is functional equality evidence, not a timing job.

The next unlogged state/head process stops in its first chunk, before a full
state or head is captured. The main thread waits in Stager::wait. Three workers
query the actual DMA event in queryCounterBasedEventStatus ->
synchronizeTimestampCompletionWithTimeout -> assignKernelEventCompletionData.
Thus disabling implicit conversion does not remove every observed counter-event
completion path. Neither this setting nor CNR is proved to prevent the stall.
The kernel journal records no new xe fault.

The owned inferior/debugger are both terminated and verified absent before a
fresh logged H2D/kernel/D2H health check passes on the same boot. No reset,
rebind, reboot, package or service change is performed. The sequence stops
before its second unlogged repeat. The candidate remains unadopted, and no
timing is accepted. The first logged equality pass cannot establish reliability
of the unlogged completion path or full262,144-cell capacity.

The observed stack motivates a separate private candidate that omits copy-queue
profiling unless transfer profiling is explicitly requested, while retaining
compute-queue properties, arithmetic, in-order dependencies and explicit drains.
That candidate is not GPU-validated by this record.
