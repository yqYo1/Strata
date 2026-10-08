# Full-context failure on the installed legacy adapter

After its successful detailed four-request short comparison, the installed
legacy Level Zero adapter was used for an actual normal-MTP 262,144-cell
capacity diagnostic on 2026-10-07. The frozen engine remained `a63f66eb…`,
expert waits were disabled, and the legacy copy-engine setting remained 0.
The model stopped while finishing the chunk from token 83,968 in layer 1.
The application watchdog aborted; no request completed. The diagnostic
ended after 183.80 seconds.

The owned debugger preserved the first watchdog `SIGABRT`, without a
preceding requested pause. The main thread was in
`L0::EventImp<unsigned long>::hostSynchronize`, querying event packet
completion through legacy `ur_queue_handle_t_::synchronize` and
`urQueueFinish`, then SYCL queue wait and a prefill lambda. The event pointer
was `0x54c5100`; the query frame reported 62,310,513 microseconds since wait.
No main-register `EAGAIN` was observed. This is an event-completion wait,
not the preceding V2 CSR wait or proof of a common root cause.

The prepared CSR reader ran but found no CSR frame in this event wait.
No live CSR counters or completion-tag value were captured. Main registers,
stack words and instructions were saved separately from the watchdog's
registers. The supervisor's broader I/O/GPU activity did not satisfy its
unchanged-progress snapshot trigger; only the real abort stop was inspected.

A later [CPU-code-matched line-table mapping](../prefill-wait-map-20261007/README.md)
locates the prefill frame at `prefill.cpp:4298`, the compute queue wait before
the chunk callbacks. It does not identify the preceding unfinished operation.

The runtime also logged unsupported `urQueueIsGraphCaptureEnabledExp`
queries while earlier chunks continued. Those messages are retained, not
silently classified as successful API queries. They do not by themselves
identify the eventual event wait's cause. Both small adapter probes used
immediate command lists, with different kernel-launch APIs; this full failure
does not establish that the V2 argument-launch path alone caused the problem.

Owned cleanup removed both engine and debugger, with no new xe fault/reset.
The unchanged logged V2 small GPU probe then passed three rounds of 16,384
exact words through H2D, kernel and D2H without reset. No rebind, reboot,
service change or driver update was performed.

The diagnostic retained warnings, parameter validation and Strata progress,
chunk 1,024, compact mode 2, layer-major mode 1 and normal MTP. This is not a
throughput result. The legacy adapter cannot be adopted as a demonstrated
long-workload prevention method. Its full-cell CLI gate also remains pending.

`full-context-legacy-l0-serve/` contains the terminal result, metrics, protocol,
stderr, main/watchdog snapshot and explicitly unavailable CSR read.
`post-legacy-full-stall-health/` contains the reset-free follow-up.
`sources.json` and `manifest.json` preserve source and receipt digests.
