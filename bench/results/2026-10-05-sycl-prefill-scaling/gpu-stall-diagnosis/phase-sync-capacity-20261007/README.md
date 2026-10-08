# Full prefill completion followed by a decode restoration failure

On 2026-10-07, the B570 `a63f66eb…` engine used the preceding validated
phase-sync profile with context 262,144, input 262,140 and four requested
output tokens. It processed all 262,139 prefill tokens through 48 layers.
The last chunk was 261,120 through 262,139 in layer range `[47,48)`.
It then closed the temporary layer cache and recaptured verifier graphs for
windows 1 through 4. Restoring the MTP decode weights failed allocating an
8,388,608-byte SYCL physical segment.

The owned main-thread snapshot records `DecodeLease::~DecodeLease()` calling
`std::terminate`, followed by SIGABRT. No requested diagnostic pause preceded
this abort. The CSR reader reported no matching frame; this is not a zero
counter measurement or a queue-completion stall. The controller then removed
both owned children. No new xe fault/reset message was recorded. The logged
follow-up probe passed three rounds of 16,384 exact integer words each,
including H2D, kernel and D2H, without reset, service or driver changes.

No token-generation reply or complete head was produced. The clipped tail,
capacity refusal and later valid-request cases were not reached. Thus this
run proves completion of the full prefill followed by restoration failure;
it does not pass the normal-MTP full-cell gate or prove general hang
prevention. It does not identify why the physical allocation could no longer
be satisfied. The original error and main-thread evidence are preserved.

The diagnostic lasted 2,680.60 seconds and printed 20,839,201 synchronized
marks. It used direct submission off, V2 copy offload off, expert wait batch
zero, cache off, compact mode 2, layer-major mode, chunk 1,024, attention batch
128 and five CPU workers. Loader warnings, parameter validation and Strata
progress were enabled. This is not a clean throughput measurement.

The complete private phase log is 1,480,096,559 bytes. `phase-log.json` records
its SHA-256, counts and final marks. `strata-progress.txt` preserves non-phase
progress and the failure, and `phase-log.tail.txt` preserves the final 64 KiB.
Phase names refer to the next phase after completion of earlier work.
The exact environment, metrics, protocol, snapshots, health check and source
digests are archived with `sources.json` and `manifest.json`.
