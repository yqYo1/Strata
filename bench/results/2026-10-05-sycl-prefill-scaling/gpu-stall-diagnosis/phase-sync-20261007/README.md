# Logged short model check with phase completion waits

On 2026-10-07, the frozen `a63f66eb…` engine completed four actual normal-MTP
requests with the existing `STRATA_PREFILL_SYNC=1`. Expert batch waits were
disabled. Every output ID, printed logprob and complete finite 248,320-float
head matched the preceding released-weight control, including the repeated,
different and restored requests. Six release/restore pairs completed, the
engine exited normally, and both owned children were removed without force.
No new xe fault/reset was recorded.

The case used context 128, chunk 32, compact mode 2, forced layer-major mode,
normal MTP and five CPU workers. Direct submission, persistent caching and
V2 copy offload were disabled. The installed Level Zero validation and
parameter checks, API entry/result logging and flushed UR tracing were all
enabled. The exact environment and source digests are in the result.

The private API log contains 1,078,292,501 bytes, 1,171,712 API entries and
1,021,864 successful Level Zero results. Unsupported graph-capture queries
are counted separately. It contains 73,692 completed phase marks. Their
counts and final marks, the full-log digest, other Strata progress and the
final 64 KiB are archived in `owned-phase-sync-short/api-log.json` and its
tail file. The complete API log and binary heads remain private.

Each phase message is printed after waiting for earlier compute work; its
name is the next phase. A completed `gemm down` mark therefore does not
establish completion of the down GEMM that is submitted afterward. The
[source mapping](../prefill-wait-map-20261007/README.md) explains the stopped
chunk wait and why the coarser step-sync setting does not cover this case.

The diagnostic lasted 179.14 seconds. This is output-parity and log-delivery
evidence, not clean throughput or stall prevention. The configuration needs
its own full-cell CLI and normal-MTP serving gates. A separate long serving
diagnostic used warning/validation/progress logging plus the same phase
waits after this detailed short check. Its
[result](../phase-sync-capacity-20261007/README.md) reached the end of prefill
but failed during decode-weight restoration; full capacity remains unproven.

`sources-used/` preserves the exact helper, engine source and controllers.
`sources.json` and `manifest.json` preserve source and receipt digests.
