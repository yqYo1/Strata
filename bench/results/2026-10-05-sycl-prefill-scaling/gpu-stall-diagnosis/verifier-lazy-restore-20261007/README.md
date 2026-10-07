# Lazy verifier capture after decode-cache restoration

Measured on 2026-10-07 with the Arc B570, Ryzen 5600X and runtime recorded
in the [profiling receipts](../profiling-20261007/README.md). This is a
private candidate. Production sources, objects, archives and executable
were unchanged; the candidate has not passed full-context occupancy gates.

The earlier 256K request finished prefill but failed an 8 MiB physical
allocation while restoring MTP weights. Temporary-buffer release left
1,411,559,424 bytes free. Main-cache restoration plus eager verifier
recapture reduced that to 131,244,032 bytes. This combined decrease was
larger than the main cache's 402,653,184 bytes of released physical backing.
It does not by itself identify a leaked allocation or a driver defect.

Source inspection found that serve initially captures a window lazily in
`Verifier::run()`, whereas `rebuild_cache_graphs()` called `warm()` to
capture every window from one to four tokens immediately after the main
cache was restored, before MTP's physical backing was restored. The private
candidate keeps the idle graph-discard checks and refreshed cache address,
then lets the existing `run()` capture each requested window before use.
It changes no kernel body or model weights. The commit graph does not use
the expert-cache address and retains its existing lifetime.

The executable SHA256 is
`882858b2e3b6c14986694a824c6a1093456d895b355ee00472573c341903d4d9`.
Only `verify.cpp.o` in the engine archive was replaced for this build.
The first offline preflight used the wrong CMake archive/object path and
failed before compilation or GPU execution. The corrected build passed;
its receipts verify every other archive member and production link input.

The first logged check used context 128, FP16 KV, 32-token chunks,
layer-major mode 2 and 600 expert slots. Four first/repeat/other/restored
requests matched all IDs, printed logprobs and all 248,320 finite float
logits. Six MTP release/restore pairs completed. This configuration did
not enable main-cache release, so it checks the new executable and MTP
behavior but does not exercise the changed main-cache callback.
Original phase-wait counts were retained, including 73,692 printed waits.
Only one-token and four-token verifier windows were captured.

The next check allocated **all 262,144 KV cells**, with INT8 KV,
1,024-token chunks, layer-major mode 1, 128 expert slots, original phase
waits and normal MTP. It consumed only 2,048 prompt tokens and generated
four tokens. Main-cache and MTP release were both enabled. Captures were
deferred until the requested one-token and four-token windows ran.

| Memory checkpoint | Free device-memory bytes |
| --- | ---: |
| Before decode-cache release | 72,466,432 |
| After main-cache release | 475,119,616 |
| After MTP-weight release | 1,414,582,272 |
| After temporary-buffer release | 1,411,559,424 |
| After main-cache restoration, before lazy capture | 1,008,906,240 |
| After MTP-weight restoration | 69,369,856 |

The private prefill log retains the old checkpoint text "verifier graphs
rebuilt". In this candidate the checkpoint only refreshes the cache address;
the actual capture lines occur later. The 877,662,208-byte difference from
the earlier post-recapture checkpoint is an observation across separate
runs, not a measured per-object allocation size or proof that full
occupancy is fixed.

An unchanged workspace-reclamation executable was then run with its
existing `STRATA_VERIFY_EAGER=1` path and identical context, input, chunks,
weights, expert-cache and MTP settings. It matched the lazy candidate's
four IDs, every printed logprob and the complete finite head, SHA256
`f9a00ae50962f9fec51da97eb1058c9b2d685a3b65e86ee13759aea773c799ea`.
Both runs exited normally, needed no forced cleanup and left no owned
process or new xe fault/reset. The comparison against the earlier 4K
profiling configuration matches IDs but not logprobs or heads; those runs
are not the same configuration and are not used as this parity gate.
The diagnostic durations include cold capture and phase prints and do not
establish a clean throughput gain.

The frozen full-occupancy controller retains 262,140 input plus four output
tokens, a clipped two-token tail, context-overflow refusals and a later
valid request. Its source is archived, but its result is pending and is
not included as a completed measurement here. Allocating a 256K KV region
and consuming a 2K input does not replace using every context cell.

Controllers, original/candidate source, build records and completed run
receipts are retained here. Large private logs have hashes, byte counts
and bounded tails. Complete raw logs and heads remain at their private
paths. No driver, service, reset or global security setting was changed.
