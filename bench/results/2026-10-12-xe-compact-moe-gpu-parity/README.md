# Compact MoE GPU output and lifetime checks

At actual chunk6144, fresh OFF/ON checks at64,2049 and32768 input tokens all closed
normally with16 generated IDs and exact selected LP5. Fresh OFF also matched the
same-N retained historical OFF. Every process passed fresh integer GPU health,
runtime mapping/source guards and kernel fault checks. Arithmetic precision did
not change. These are diagnostics; their timings do not qualify performance.

The same configuration observed745MiB free VRAM OFF and1286MiB ON at READY.
This is a6144 observation, not8192 fit. The feature remains defaultOFF;8192,
64K performance, broad model quality, RESTORE and full262144 occupancy are unqualified.

[Individual results](results.json) link each original receipt. Receipt archives
use gzip+base64 with verified decoded byte count and SHA256. Source qualification
and exact controller source are in the preceding committed source report. Retain
private originals for current model/reference consumers; no cleanup is performed here.
[Research corrections](main-research-corrections-v2.json) preserve original reports.

本家の測定は箱出し設定だけではないが、本家単体の最適設定探索・独立検証は
未完了。[TODO](../2026-10-11-upstream-parameter-audit/TODO.md)を維持する。
