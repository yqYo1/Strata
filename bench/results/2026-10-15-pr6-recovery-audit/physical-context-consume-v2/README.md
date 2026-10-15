# physical-context-consume-v2

Unapplied source-only extension of the explicit --consume-prompt diagnostic to both resident and actual streamed target KV. See source-review.md for the audited ordinary lifecycle and observation limits. V1 and NativeDense bank v2 remain unchanged; source.diff applies against current production, without either proposal applied.

Mandatory diagnostic shape remains fresh standalone SYCL --native --spec4 --mtp PATH --suffix-draft0 --prefillN --no-token-graph, PLE enabled, no split/peer/remote/window/competing diagnostics. Only the explicit flag relaxes output reservation; --max-new0 ordinary/API meanings do not change. Input and configured context never exceed262144. N−1 batched rows plus one real target verifier T1/commit1/wait consume the complete prompt, with output0, no MTP draft, no adaptation launch and no position advancement. Success reaches existing manual cleanup and emits schema2; external process-owner/driver checks remain required.

Unlike V1, --kv-resident32768 is admitted. Actual SessionState mode/resident/context and host/map ownership are checked, normal host-authoritative Prefill append→Verifier terminal append→selected block resolve→attention→commit paths are reused, and real device map counters plus one exact packed selected prior host-vs-slot block comparison per streamed QSA are observed after checked completion. No full KV owner or new GPU allocation is added. This bounded probe is not all-context payload/numerical parity.

At512 input/context, requested32768 legitimately resolves to resident mode0; it cannot count as a streamed gate. Use a true small streamed context (e.g32768 input/context with requested20480) before the actual262144/32768 target gate. The independent receipt checker requires expected actual mode and effective resident cells and rejects silent resident substitution.

Artifacts: proposed source/header mirrors, production→proposal source.diff, exact base/proposed hashes, source-review.md, actual-helper host_fixture.cpp, bounded mode-aware verify_receipt.py and Main execution recipe. Nothing was compiled/tested/run. No quality, RESTORE, physical pass, throughput or integration claim is made.
