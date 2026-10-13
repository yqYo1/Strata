<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Decode CPU service diagnostic

`STRATA_DECODE_SERVICE_CPU=1` records CPU time in the existing native expert pool
per request. It targets ordinary single-stage decode without batch, MTP, suffix,
or lookup. Raw verifier capacity differs from actual NT; dispatch checks actual
NT=1. Ineligible requests retain their inference selection and report incomplete
at close. The envelope is at most 64 layers and 128 workers.

Linux uses `CLOCK_THREAD_CPUTIME_ID` and a host steady clock. Other operating
systems report CPU-clock unavailable and cannot produce a complete result.
When the flag is off, there are no new clock reads, per-job allocations, or output.
When on, fixed-capacity state is allocated at request start.

Separate issuer (the caller performing inline drain) and worker buckets record
CPU nanoseconds, wall nanoseconds, calls, and clock-read attempts per native GU,
`native_quant_h`, and native Down invocation/range. Existing Q2 GU/SwiGLU, ActQ
quantization, and Q2 Down have separate fallback buckets. A Q2 GU bucket counts
one range containing the existing two row calls and SwiGLU. Claimed row tasks
and native call ranges need not have equal counts. There are no clocks inside rows.

The same clocks bracket the issuer's complete per-layer dispatch and existing
GU/Down parked-before, publish-to-done, and done-to-parked boundaries. Each
worker has a fixed slot with one writer; reads occur after the existing
release/acquire publication and worker-park completion boundaries. There are no
new pool atomics, GPU events, copies, waits, or device branches. Request binding
and close mutexes synchronize host code only.

These wall spans overlap. Do not add worker wall sums, phase wall sums, or GPU
ticks to request wall. Wall minus CPU is a residual that includes blocked or
preempted time; it is not pure wait. Existing `decode_wall_ms` is request decode
wall before the commit wait and excludes CPU diagnostic formatting. This
diagnostic alone cannot establish GPU overlap or the critical path.

One request-close block is bounded at 65,536 bytes. There is no per-job or per-row
logging. Clock failure, reversal, counter overflow, task/quant count mismatch,
missing layer coverage, and abnormal close remain incomplete. The original
close status is preserved. `cpu_jobs` counts coalesced expert jobs;
`cpu_activation_rows` sums their `nt` values and is the quantizer-call denominator.
Dispatch NT and a job's `nt` are separate values. Footer `request_invalid` and
`request_overflow` describe request flags only; use the final footer's `incomplete`
for the overall result. Even `length`/`stop` close requires completed dispatch
and clock receipts for each layer to match emitted tokens. The final close row
states the result. An abort or forced termination without a close block also
cannot provide a complete receipt.

`STRATA_DECODE_ROUTE_LEDGER=1` can accompany this diagnostic as complementary
owner, NT, and call coverage. Its logical bytes are not measured DRAM bytes,
transfers, or time. The ledger has its own output budget. `STRATA_DECODE_TIMING`
is the existing host diagnostic. `STRATA_VERIFY_PROFILE` changes the eligible DF
branch and must remain off for this branch-preserving measurement.

The CPU fixture is `tools/test_decode_service_budget.cpp`. It checks clock
accounting, aggregation, overflow, reset, availability, normal/failed coverage,
independent parallel worker slots, and zero clock reads when off. Assertions
require `-UNDEBUG`. Separately check model output/logit parity and decode wall
for the same request with the diagnostic off/on. Clock-read attempts account
for instrumentation; they do not themselves measure instrumentation overhead.
