# Full-cell CLI capacity and the next scheduling candidate

The real CLI capacity diagnostic completed on this Arc B570/Ryzen 5 5600X
host on 2026-10-07. It prefills 262,141 tokens, verifies the last prompt token,
then uses a clipped two-token window. The final executed KV cell is 262,143
in a context of 262,144 cells. It outputs `[40, 3172]`, has a complete finite
248,320-float first head and exits normally. Neither the debugger nor engine
survives cleanup; no new xe faults or resets are recorded.

This establishes the tested CLI capacity path. Full-context normal-MTP serving
remains a separate required check; the supervisor prepared here is not evidence
that serving has completed. Nor does one completed diagnostic establish
prevention of the earlier intermittent `EAGAIN` submission stall.

## Actual CLI evidence

`full-context-layer-trace-csr/record.json` records the exact argv, selected
environment, binary and source hashes, 12,288 chunk-start records across all
48 distinct layer ranges, finite-head digest, executed windows and cleanup.
The frozen executable SHA-256 is
`b81a7d6fbc1c6d524cf3b64e196f5866c91ec56c76e460a66b63b28ce3aed461`.
The first head SHA-256 is
`8050c1e0655924cd013229dde06343632208d12a25c9b82ae1d35c72c93f33b9`.

The original full-input tuning is retained: prefill 1,024, compact layout 2,
layer-major mode 1, int8 KV, normal confidence clipping disabled, direct
submission disabled and V2 copy offload disabled. The detailed short check
had already passed on this exact executable. This longer check retained
parameter validation, warning/error logs and Strata layer/chunk progress,
with an owned debugger for stopped-submission inspection.

It completed in 1,426.65 diagnostic seconds. This is not a clean throughput
measurement. The 743,413-byte engine/GDB stderr and GDB/MI are archived in
full. No debugger pause or stalled-CSR inspection was needed. The head binary,
input token fixture and complete host kernel journal remain private; their
digests and relevant xe/B570 journal rows are recorded. The metrics JSONL is
retained; separate per-sample command outputs remain private because they
duplicate those samples.

## CPU scheduling candidate

`layer-trace-tasks9-link/` relinks the same current objects and libraries first
with the original CPU archive, reproducing the exact `b81a7d6f…` binary hash,
and then with the previously measured task-factor-9 CPU archive. The candidate
SHA-256 is
`f5ef7bce303a7123819357eb86e5d9461c4073acbffb205650fc2546b510457d`.
All other archive members and all original link-input digests are unchanged.
Only `pool.cpp.o` is replaced by `native-pool-task-factor.cpp.o`.

This is a build receipt, not a GPU correctness or engine-speed result. It keeps
the scheduling experiment on the current traced engine instead of the older
executable. The previously recorded CPU microbenchmark does not establish
prefill or decode improvement for this new engine.

## Serving supervisor checks

Ten real CPU-child/GDB/PTY cases check the new supervisor without loading a
model, GPU adapter or GPU driver. Host journal/service operations are fakes.
The first six passing cases are in
`full-layer-trace-serve-cpu-fixed-fixture/record.json`; the final four are in
`full-layer-trace-serve-cpu-final-four/record.json`. They cover the five valid
capacity/refusal/retry requests, overrun, last-cell omission, early EOS, stale
head, execution on refusal, 2 MiB of QUIT output, ignored QUIT, missing READY
and blocked request input. The QUIT payload is checked exactly, and no test
debugger or inferior survives cleanup.

The initial fake diagnostic environment omitted a required UR-layer key and
failed before an engine launched. Its harness and failure output are retained.
A later QUIT fixture gave only 0.5 seconds to drain 2 MiB through a PTY; its
failed receipt is retained too. That case passes with a 10-second drain deadline;
the real supervisor retains a 30-second shutdown allowance. The other six
already-passing cases were not rerun. These are fixture corrections and CPU
supervision results, not GPU recovery or capacity proof.

`sources.json` preserves the exact actual CLI sources and the prepared serving
supervisor, CPU harness and relink controller. `manifest.json` covers the
archived receipts and source snapshots. Raw debugger prompts are preserved
unchanged.
