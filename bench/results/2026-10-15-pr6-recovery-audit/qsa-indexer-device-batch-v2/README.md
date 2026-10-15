Native SYCL QSA indexer device-batch proposal v2, 2026-10-15

V2 addresses the fixture gaps identified in R682. It is a complete standalone
replacement patch against the actual root: apply V2 alone, not V1 followed by
V2. V1 and every other proposal remain untouched. No root source or source pin
was changed. No compiler, build, test (including host), GPU, process/environment
or private-state inspection, Git mutation or cleanup was performed. Main's live
head-lifecycle diagnostic continues independently.

The four existing root files were read and SHA-checked against V1's base pins:
native SYCL indexer source, its public header, SYCL CMake and root CMake all
match. `source-manifest.json` records these actual checks and every base/proposed
SHA. The native kernel, API/header changes, production planner and both CMake
changes are byte-identical to V1's proposed versions; the CPU fixture is also
unchanged. The ONLY changed proposed source is the GPU fixture. The manifest
lists each unchanged V1/V2 proposed SHA explicitly. No algorithm modification
or integration with head release, BF16 or KV-prefix reuse is included.

Production remains default-off under immutable
`STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH=1`; absent/0 retains the original per-cell
batch loop. The original single-cell kernel body and decode are unchanged.
For a multi-cell candidate batch, the existing position-zero append seeds dead
when needed, independent completed-block workgroups write their own pooled rows,
then a separate finalizer writes the latest tail slots, terminal spare and last
block_pos on the same validated in-order queue. Half rounding, sum/FMA/RMS tree
and RoPE/M-RoPE expressions remain V1's exactness candidate. There is no new
production allocation, global scratch, readback or wait. This design has not
been compiled or shown bitwise-correct here.

New GPU fixture source coverage, all still unexecuted:

| Addition | Implemented assertion path |
| --- | --- |
| Fresh p0=0,n=2 | A new `sequence()` creates clear, independently allocated oracle/candidate states. Candidate receives n=2 directly, forcing the original zero seed plus finalizer with no completed block. Complete pooled allocation (including unwritten rows/spare), tail, dead, block_pos and both guard rows must match the sequential oracle. Linked score/top-k comparisons also run. |
| Fresh p0=0,n=3 | The same checks in a separate new pair of clear states. It covers all three partial tail slots while pooled[0] remains the dead seed and the next pooled row remains untouched. |
| Fresh p0=0,n=4 | Another independent pair runs the seed, actual completed-block workgroup and finalizer. It checks the completed row, terminal spare, tail, dead, block_pos, guards and causal IDs. |
| Distinct n=1 fallback | The existing split suite still begins its p0=0 case with n=1, which uses the original single-cell fallback. The three new cases are n>1 and cannot pass through that fallback. `fresh_zero_multicell_batches` counts them only after each complete comparison succeeds. |
| Gamma/state alignment | Negative cases now include +2-byte misalignment of gamma, tail, dead, pooled and block_pos, in addition to the existing raw misalignment. |
| Representative state-state overlap | Negative cases add dead within tail, pooled overlapping tail, and block_pos within pooled. The existing block_pos-within-dead and raw/gamma-to-state alias cases remain. These represent specific overlap classes; the fixture does not claim to enumerate every possible alias. |
| Per-negative-case no mutation | The shared rejection wrapper now takes a complete state image BEFORE EACH case, checks its guards, invokes the candidate, fences every fixture queue, reads another full image, checks its guards and compares every state byte. Only then does it count the expected invalid_argument rejection. The old suite-wide initial/final check also remains. |

`small` now executes the fresh 2/3/4 cases first, followed by the unchanged
p0=0/1/3/4/5 split 1/2/3/5/4/1/7/2 histories and their synthetic saved-image
continuation. Each call creates or uses the same independently allocated states
as V1; no math is copied into the fixture. `invalid` adds the specified cases
and per-case assertions while preserving the existing invalid ranges, bases,
capacities, shapes, epsilon/scaling, null pointers and queue/context cases.

Existing coverage remains proposed source, not a PASS:

- CPU fixture uses the unchanged production parser/planner for strict helper
  flags, checked ranges, completed-block launch geometry and addressable extents.
- GPU large suite compares full 4096/8192 batches and partial tails. Selection
  uses linked production scores/top-k on both states, query-specific causal
  steps, identity through 2051 and sparse selection at 2052, future-block score
  sentinels, selected-ID range/order and guard checks.
- Per invocation, none/linear/YaRN, analytic/full/short registered RoPE tables
  and plain/M-RoPE are explicit fixed parameters. Short tables have 1024
  positions. Table registration occurs before comparisons and is retired after
  checked queue completion; no environment arm is switched at runtime.
- Physical suite allocates all 65537 pooled rows in each state plus full-row
  guards, builds the contiguous 262140-cell prefix with the original single-cell
  oracle, clones it, and compares the final four cells plus spare 65536 and
  next-row 65537 canary. This is unexecuted component allocation coverage, not a
  physically qualified 262144-context model run.
- The saved-image continuation copies component buffers into fresh allocations
  and republishes the moving spare from dead. It remains synthetic and cannot
  qualify real snapshot serialization, session RESTORE or callback policy.

The fixture's compact JSON adds `fresh_zero_multicell_batches`,
`invalid_rejections` and `invalid_unchanged_checks`. A successful future small
suite should count 3 fresh multi-cell zero-start batches; a successful invalid
suite must have one full no-mutation check per rejection. These fields describe
executed checks only when Main actually runs and closes a process successfully.
All full-model physical-context, RESTORE and performance qualification fields
remain false. No receipt or run result has been produced here.

Files:

| File | Purpose |
| --- | --- |
| `source.diff` | Full V2 replacement payload, including unchanged V1 production changes and updated GPU fixture. |
| `source-manifest.json` | Actual root base reads, base/proposed hashes, explicit V1 production/planner byte identity and remaining qualification gates. |
| `qsa_indexer_device_batch_plan.hpp` | Unchanged new production planner source from V1. |
| `qsa_indexer_device_batch_plan_test.cpp` | Unchanged CPU fixture source from V1. |
| `qsa_indexer_device_batch.cpp` | Updated linkable GPU fixture source; CMake target links actual `strata_kernels`. |
| `artifact-sha256.json` | Proposal, V1 and R682 read-only input identities. |

Main's later manual reproduction, not executed here:

```text
cmake --build <configured-host-build> --target qsa_indexer_device_batch_plan_test
<configured-host-build>/qsa_indexer_device_batch_plan_test
cmake --build <configured-SYCL-build> --target sycl_qsa_indexer_device_batch_test
<configured-SYCL-build>/sycl_qsa_indexer_device_batch_test small none analytic plain direct
<configured-SYCL-build>/sycl_qsa_indexer_device_batch_test small yarn short mrope dispatch
<configured-SYCL-build>/sycl_qsa_indexer_device_batch_test invalid none analytic plain direct
<configured-SYCL-build>/sycl_qsa_indexer_device_batch_test invalid yarn full mrope dispatch
<configured-SYCL-build>/sycl_qsa_indexer_device_batch_test large linear full plain direct
<configured-SYCL-build>/sycl_qsa_indexer_device_batch_test physical none analytic plain direct
```

The host target is under `STRATA_BUILD_PREFILL_HOST_TESTS`; the GPU target is
manual/EXCLUDE_FROM_ALL, with no automatic CTest run. Set all runtime/driver/
profiler variables in each owned native child's environment BEFORE exec.
Dispatch mode requires exact `STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH=1`; registered
full/short tables require `STRATA_ROPE_TABLE=1`. Run comparison configurations
in separate processes. The explicit direct API provides the same candidate
without changing an environment flag during a process.

Remaining Main gates:

1. Recheck base hashes at the integration boundary; integrate only this QSA
   proposal against Main's intended pinned baseline. Preserve the live head
   lifecycle run and do no concurrent build/GPU work. Build the host fixture,
   actual SYCL source/library and new GPU fixture with recorded command/flags,
   source, binary, toolchain, runtime and hardware identities.
2. Execute and close all original/new exact oracle comparisons and negative
   assertions under fresh fault/journal boundaries. The zero-start 2/3/4 and
   per-case alignment/alias checks are prepared source only. Require exact
   state, guards, scores and causal IDs before any adoption decision.
3. Production-dispatch n=0 and invalid-environment-flag behavior remain separate
   UNEXECUTED gates. The invalid suite contains a zero-length call, and dispatch
   mode can route it with flag 1, but it has not been run. Dispatch mode itself
   requires flag 1, so it does not test invalid environment flag values; the
   host parser-helper checks are not production-dispatch evidence. Main still
   needs separate immutable-process strict/default/zero-dispatch checks.
4. Run actual-model parity, MTP/callback ordering, decode continuation,
   cancellation/fault closure and true session RESTORE, preserving existing
   layer-major checkpoint limitations. Physically admitted 262144-context model
   correctness is a separate later gate from the synthetic last-row fixture.
5. Obtain independent clean matched repeated long-input timings and decode
   regressions before any performance/adoption claim. No speedup or reduction
   of the visible GPU gap is predicted. 600/1000 tok/s remain goals.

Retention follows `docs/ARTIFACT_RETENTION.md`. No run artifact was created or
cleaned. Keep this compact source/fixture/identity proposal for Main's next idle
review; retain individual future outcomes and fault closure. Retire successful
full traces/duplicate logs only after verified compact evidence, under the
measurement lock with path/hash/reason manifests. Failed receipts stay failed.
