# Retained decode backing comparison on 2026-10-07

On Arc B570 10 GiB / Ryzen 5600X / 128 GiB RAM, the same private executable
33bef89f as the preceding regular-launch comparison retains both main and
MTP decode backing and recorded graphs during prefill. Only
STRATA_PREFILL_RELEASE_CACHE and STRATA_PREFILL_RELEASE_DRAFT change to 0;
the context-4096, int8-KV, FP16 expert path and phase waits are retained.
This is a diagnostic comparison, not a proposed full-context memory policy.

The first of two repeated 2048-token inputs stops at layer 17 of the chunk
starting at token 256. No request returns an output. The last completed
phase is dequant, mark 71177. After 60 seconds without progress the watchdog
raises SIGABRT. Owned GDB records the stop and cleanup removes both owned
processes. The trace contains 107,755 successful native/UR enqueue
associations and zero cooperative enqueue flags. API success is submission
evidence and does not establish device completion.

The failure therefore also occurs without main/MTP prefill retirement.
Removing cooperative properties and retaining decode backing are each
insufficient to prevent this wait. The exact failing resource or operation
remains unidentified. The shutdown message saying the verify-window GPU
finished polls the verifier's own queues through ext_oneapi_empty(); it does
not demonstrate that the separate prefill queue finished.

No new xe fault is recorded. A fresh bounded logged H2D/kernel/D2H test
then passes all three rounds of 16,384 exact words on the same boot, without
a reset or service/driver change. No diagnostic time is treated as clean
throughput, and the complete repeated 262144-cell validation remains open.
The terminal record, exact launch audit, bounded logs and full private-log
hashes are preserved. The previous archive stays frozen at its earlier
point, when this controller had not yet run.
