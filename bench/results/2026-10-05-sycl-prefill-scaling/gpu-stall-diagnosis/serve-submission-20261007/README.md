# Normal-MTP submission stall and healthy follow-up

The real 262,144-cell normal-MTP serving diagnostic stopped before completing
its first prompt on 2026-10-07. It used the same frozen `b81a7d6f…` executable
as the completed full-cell CLI check, but also loaded the normal MTP weights
and serving verifier. Initial free VRAM was 29 MiB. Draft-weight release
allowed layer-major prefill to start.

It reached the chunk starting at token 91,136 in layer 15. After no progress,
the application's watchdog called `abort()`. The owned debugger preserved
the first `SIGABRT` stop and then cleaned up both engine and debugger. This
was a failed serving-capacity check, not a completed prompt or throughput
result. No new xe fault/reset was recorded. The diagnostic ended after
620.02 seconds.

## Captured submission and completion values

At the requested interrupt, the main thread was inside NEO's
`gemExecbuffer2` ioctl retry path, entered from oneMKL's FP16 GEMM submission
through UR V2 and Level Zero validation. The syscall register was `-11`
(`EAGAIN`). The main thread's final abort-time stack retained the same
submission argument, 10,448,090. The watchdog thread's abort stack is also
preserved. The expert-pool report showed all five workers parked, with no
CPU expert jobs in this request.

The read-only CSR inspection captured:

| Field | Value |
| --- | ---: |
| Current submission argument | 10,448,090 |
| `taskCount` | 10,448,089 |
| `latestSentTaskCount` | 10,448,090 |
| `latestFlushedTaskCount` | 10,447,653 |
| Value at `tagAddress` | 10,446,989 |
| `completionFenceValuePointer` | null; dereference unavailable |

The submission argument is 1,101 ahead of the observed completion-tag value.
This is a CSR counter difference, not a measurement of the kernel's live job
count. The tag was read once; it was not independently sampled again at the
abort. Queue pressure is consistent with the previously inspected xe
1,000-job branch, but this does not prove that branch was the live cause or
explain the lack of workload progress.

The CSR inspector never calls inferior functions or writes inferior memory.
The requested debugger pause was resumed. The watchdog's later reports and
abort include that pause's effect on wall-clock time; they are not an
untouched reproduction. The observed lack of progress preceded the requested
pause; `EAGAIN` was observed in the first stopped stack/register snapshot.

## GPU remains usable after owned cleanup

After checking that both failed children were gone, the unchanged small B570
health executable passed all three rounds of 16,384 exact integer words,
including host-to-device copy, kernel work and device-to-host copy. The probe
retained full API logging and parameter validation and recorded no new xe
fault/reset. The agent performed no GPU reset, rebind, reboot, service change
or driver update.

This establishes a failed workload with a healthy small follow-up, not a
permanent loss of GPU availability or a proven general prevention method.
The next comparison adds optional host backpressure between groups of direct
FP16 experts, preserving data and model arithmetic. It must pass actual head
comparisons and full capacity before a performance or prevention claim.

`full-context-layer-trace-serve/` contains the terminal record, metrics,
protocol bytes, stderr, both thread snapshots and the stopped CSR read.
`post-full-serve-stall-health/` contains the actual follow-up. Only relevant
xe/B570 host-journal rows are archived. `sources.json` and `manifest.json`
preserve the sources and receipt digests.
