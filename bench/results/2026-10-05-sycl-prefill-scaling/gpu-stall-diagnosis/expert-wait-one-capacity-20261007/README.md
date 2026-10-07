# Full-context normal-MTP failure with one-expert queue waits

The same frozen `a63f66eb…` executable used
`STRATA_PREFILL_EXPERT_WAIT_BATCH=1` for an actual normal-MTP 262,144-cell
capacity diagnostic on 2026-10-07. It stopped in layer 10 during the chunk
starting at token 139,264. The application's watchdog reported no progress
and aborted. No request completed. The diagnostic ended after 543.52 seconds.

The log records 1,187,353 completed one-expert queue waits before the stop.
This condition is insufficient to prevent the long-workload failure. The
preceding four-request short arithmetic comparison remains valid; it was
not capacity proof. No throughput or general prevention claim is made.

At 40 seconds of unchanged progress, the owned debugger interrupted the
main thread and found `NEO::CommandStreamReceiver::baseWaitFunction`, via
Level Zero immediate-command-list `hostSynchronize`, UR V2 `queueFinish`
and the newly added SYCL `wait_and_throw`. Its completion target was
8,582,366. This is a wait target from the stack, not a measurement of live
kernel jobs or a captured completion-tag value. The main registers, stack
words and instructions were read successfully. No main-register `EAGAIN`
was observed in either snapshot.

The requested pause was resumed. The later application watchdog abort
includes its timing effect; it is not an untouched reproduction. The first
real `SIGABRT` stop in watchdog thread 34 was preserved. The main thread's
final backtrace again did not unwind fully, but its registers, stack words
and instructions were saved separately from the watchdog's registers.

Both attempted CSR reads explicitly report `NO CSR FRAME`: the original
reader searches for a submission/flush frame, not this completion-wait
frame. This receipt therefore has no live CSR counter or completion-tag
measurement. Missing values must not be interpreted as zero or as the
preceding b81 run's 1,101-counter difference.

Owned cleanup removed both engine and debugger. No new xe fault/reset was
recorded. The unchanged small GPU probe then passed all three rounds of
16,384 exact integer words through H2D, kernel and D2H with full API logging
and parameter validation. No GPU reset, rebind, reboot, service change or
driver update was performed. This establishes current basic execution,
not full-model safety.

The failed model run retained warnings, parameter validation and Strata
progress after the exact binary/condition passed its detailed short check.
It used chunk 1,024, compact mode 2, layer-major mode 1 and normal MTP. The
84,005,807-byte full stderr remains private; its SHA-256, line/wait counts,
all 2,697 chunk-start lines, other Strata progress and final 64 KiB are
archived. Finite request, overall and log limits were retained.

`full-context-expert-wait-1-serve/` preserves the terminal record, metrics,
protocol, both snapshots and both explicitly unavailable CSR reads.
`post-expert-wait-one-stall-health/` preserves the reset-free follow-up.
`sources.json` and `manifest.json` preserve source and receipt digests.
Normal-MTP full capacity failed; this condition's own full-cell CLI gate
remains pending.
