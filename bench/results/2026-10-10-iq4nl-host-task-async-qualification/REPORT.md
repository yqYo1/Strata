# Completed host-origin asynchronous error boundary

On Arc B57010GiB rootPCI0000:05:00.0 with source000f072da7d3f5b179173e81028cacb5f528e0af, the root built the isolated fixture and unchanged productionprefill archive using oneAPI2026.1 precise-math SPIR64/subgroup32/per-kernel split. Build and all owned processes closed normally with empty observed/reaped sessions and no cleanup/survivors.

The control and injected-host-task cases passed. Both submitted one valid private IQ4NL GPU kernel. After an existing successful wait, the injected host exception was delivered once through the actual production DPCT rethrowing handler, translated to false, and skipped the fixture callback. Generic retry count stayed zero; the later wait_and_throw did not redeliver it. All input bytes and output/canary words matched; both queues and USM buffers were explicitly torn down before final terminal. All21 stages and two CASE records were present. The GPU child exited0 in0.322125seconds, with sampled peakRSS50,368,512bytes.

The complete new kernel interval had no GPU entries and no devcoredump. The diagnostic API summary identifies two actual private launches and the exact single handler error line. This is legal host-origin SYCL asynchronous error injection. It does not simulate a device fault, establish recovery, test production callback/layer-major error paths, measure model speed, or qualify full physical262144 context.

Production error integration is withheld: prefill.cpp remains exact2abbaseline bytes. The original draft was preserved at4f22db5c and removed because its pre-unwind/default-token coverage was incomplete. The completed-boundary helper contract now has this narrow runtime evidence; production safety remains pending. No candidate is adopted.
