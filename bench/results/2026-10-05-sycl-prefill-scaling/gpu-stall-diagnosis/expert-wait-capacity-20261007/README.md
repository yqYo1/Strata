# Full-context normal-MTP check with 32-expert queue waits

The `a63f66eb…` executable's actual normal-MTP capacity diagnostic used
`STRATA_PREFILL_EXPERT_WAIT_BATCH=32` on 2026-10-07. It passed the preceding
stall position, but stopped during the chunk starting at token 216,064 in
layer 19. The application's 60-second watchdog reported no progress and
called `abort()` two seconds later. No request completed. The diagnostic
ended after 739.55 seconds.

The owned debugger preserved the first `SIGABRT` stop in watchdog thread 34
and cleaned up both engine and debugger. There was no requested debugger
pause before the abort. The final log shows a completed 32-expert group in
the active chunk, so the optional waits were executing. This establishes
that waits every 32 direct-FP16 experts are insufficient to prevent this
long-workload failure. The short four-request arithmetic comparison remains
valid; it was not full-capacity proof.

The main thread's final backtrace has only five unresolved frames. The
register dump is for the selected watchdog thread. Unlike the preceding
failed b81 serving run, this receipt has no stopped CSR inspection or
resolved NEO retry stack. It does not establish that this failure was at
the same ioctl, had the same completion-counter gap or returned `EAGAIN`.

The terminal receipt records no new xe fault/reset and no surviving engine
or debugger. After checking their identities were gone, the unchanged
small GPU health executable passed all three rounds of 16,384 integer words
through H2D, kernel and D2H with full API logging and parameter validation.
No GPU reset, rebind, reboot, service change or driver update was performed.
The small follow-up proves current basic execution, not full-model safety.

The failed capacity run retained warning-level runtime logs, parameter
validation and Strata progress after the exact executable's detailed short
check. It used 262,144 allocated context cells, chunk 1,024, compact mode 2,
layer-major mode 1 and the normal MTP drafter/verifier. It is not a throughput
measurement. The full-cell CLI gate for this queue-wait configuration is
still pending; full normal-MTP serving has failed rather than passed.

`full-context-expert-wait-32-serve/` preserves the terminal record, metrics,
protocol bytes, stderr and failure snapshot. `post-expert-wait-stall-health/`
preserves the reset-free follow-up. Only relevant xe/B570 journal rows are
published. `sources.json` and `manifest.json` preserve source and receipt
digests. No prevention or speed claim is made.
