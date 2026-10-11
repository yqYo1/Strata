# CompactMoE8192/64K GPU diagnostics

All three serial cases passed:32768/chunk8192/on,65536/chunk6144/on,
65536/chunk8192/on. Same-N8192 and6144 emitted identical16 IDs and displayed LP5.
Each case closed by normal QUIT, passed fresh integer GPU health before/after,
source/runtime mapping guards and a fresh kernel fault gate without reset or retry.
READY observed713MiB free for8192 and1286MiB for6144 with owned stage buffers.
This proves these bounded invocations fit; it is not a physical peak/maximum-fit test.

[Results](results.json) retain individual diagnostic times, settings, source/input/
binary/runtime provenance, exact equality and fault/closure evidence. All timings
include diagnostic logging/validation and are performance-ineligible. Compact
remains defaultOFF; quiet repeated performance, broad full-distribution quality,
full262144 physical occupancy and RESTORE remain unqualified.

Receipts use byte-verified gzip+base64, scanned as exact plaintext before encoding.
Original private receipts stay for current reference/controller consumers. Successful
full logs should be retired after extraction/publication and review; no deletion
or active-run mutation is performed here. Researchers continue distinct source/
external-kernel and bounded-GDN questions; a separate Sol implementer prepares
quiet64K/coarse-phase controller source only. Main owns all tests and GPU work.

本家単体の最適設定探索・独立検証は未完了。
[TODO](../2026-10-11-upstream-parameter-audit/TODO.md)を維持する。
