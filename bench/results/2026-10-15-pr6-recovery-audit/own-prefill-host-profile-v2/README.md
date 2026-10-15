# Own long-prefill host attribution v2

Source-only proposal, 2026-10-16 JST. **Untested**: no compiler, test,
GPU, profiler or model process was run. Active source was not edited.
Root repository LICENSE (MIT) applies to newly authored proposal files.

Parent supplied immutable base commit:
`9b83fbf8e05115fa70b136c083896e7841608ba1`.
`manifest.json` pins the exact base/proposed SHA256 for each of four touched
files and the patch SHA256. No Git commands were used to independently verify
commit membership; Main must verify these source hashes against that commit
before integration. `source.diff` is generated against the retained exact base
snapshots, not a moving checkout. `prepare.py` initially captures active files;
subsequent executions read and verify only these pinned base snapshots.

Provenance: independently authored using the active repository MIT source and
its own `cpu_wait_profile.hpp`, plus the requested own CPU investigation report.
No XeStrata source, archived observer/helper, or other engine implementation was
read or used. Existing CPU v1 proposal remains untouched.

## Behavior

The existing `STRATA_PREFILL_CPU_WAIT_PROFILE=1` exact-match long-prefill scope
remains the sole switch. Supply it in the native child's environment before
exec; never mutate the environment after runtime initialization. Default remains
OFF. No clock, lock, allocation or environment query is added per disabled call;
the region guard checks TLS and returns. Its local initialization and finish
branch remain small ordinary instructions.

The original summary becomes version 2 with the same field names. One bounded
additional host summary line reports actual region-entry counts, unwind throws,
inclusive caller CPU/wall and explicit-wait-subtracted caller CPU/wall for:

- `source_copy`: each layer's whole expert `copy_blob` slab loop, including
  cancellation callbacks and error handling inside the loop. It excludes
  residency clearing, blob-size lookup and subsequent slab uploads.
- `host_group`: after existing router/peer waits and cancellation checking,
  starting immediately before counts clear, through counts/offsets/fill/
  slot/source/order construction. Includes census callbacks, peer bookkeeping
  and associated slot/src map copy submissions, including their implicit
  blocking. Excludes router ID transfer/wait and later grouping audit capture,
  gather kernels, bounds creation and stage/job maps.
- `gemm_submit`: the actual oneMKL row-major GEMM call, including inexpensive
  scalar/pointer argument evaluation and returned event construction. Queue
  resolution and shape/span validation occur before timing; `finish()` occurs
  after timing. Synchronous library internal work or blocking remains included.
  This does not prove internal arithmetic location or distinguish its causes.

Explicit wait durations are snapshotted at each region boundary and subtracted.
`unclassified_cpu_s` = total CPU - explicit wait CPU - sum(region outside-wait
CPU), with the same wall equation. Thus outside-wait region values plus
unclassified partition total outside-wait time; inclusive values overlap the
existing wait totals and must not be added to them. Region timing does not
include its ending bookkeeping. Dense-bank activation and other submission work
remain unclassified. No exclusive nested-region metric is needed: overlapping
host regions, a host region beginning inside a wait, or an open region at scope
end invalidate the parent summary. A wait inside a host region is supported.
Explicit `finish()` preserves existing variable lifetimes and avoids reindent.
Destructor finish is idempotent and records early returns/unwinding.

All reported durations are checked for finite/nonnegative values, including
accumulated totals and partition remainders. Any clock/accounting failure makes
all durations -1 with available=0; actual counts remain available. Existing wait
exception counts and completion semantics remain intact. No inference math,
placement, wait policy or synchronization is changed.

## Overhead and Main validation

No timing overhead has been measured. This first diagnostic records every call,
without sampling or another environment variable. At 500,000 GEMMs it adds
1,000,000 stamps, each reading steady wall and thread CPU clocks: approximately
2,000,000 clock reads, plus two five-element wait-total loops per GEMM and small
counter/branch costs. The disabled path has no clock reads. This overhead can
materially perturb many short submissions; do not treat enabled measurements as
clean speed samples or infer that measured GEMM time is removable latency.

Main should measure stamp/guard overhead in a bounded host-only harness, test
disabled/enabled/exception/nesting/nonfinite accounting and a synthetic wait
inside a host region, then run its small GPU gates before a separate owned model
diagnostic. Compare enabled versus disabled in separate preconfigured processes
with identical source/build/input settings. Record all samples and actual call
counts; estimate added clock cost from measured per-stamp overhead and observed
counts. Sampling is deliberately deferred until this cost is known.

No tests or harness are supplied by this source-only task. Main owns build,
test, GPU scheduling, exact commit verification, patch review and integration.
Keep this compact proposal and subsequent individual results under artifact
retention policy; there are no raw traces or tensor captures in this package.
