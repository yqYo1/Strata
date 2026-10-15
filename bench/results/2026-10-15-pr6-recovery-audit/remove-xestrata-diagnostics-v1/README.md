# Remove adopted XeStrata route/service diagnostics

Source-only proposal against immutable `0f816385a0b2ed4cc856202ebd9d7c4e53a44a63`.
`compactsource.diff` deletes the implementations, not merely their notices. `pins.json`
records base/proposed hashes and deletion status. Only proposed modified files are
copied; deleted files remain recoverable from Git. No live product source was edited.
No build, fixture, model, GPU, service or process-environment operation was performed.
No functional, quality, safety or speed pass is claimed.

Own history inspected: `f5903fdcddc2e86fd88b75d2a04ac276c7f62a1c`
(the actual adoption), `332cc37242fb9456dbb3dcee7437945fd1605a6d`
(the archived adaptation beside an independent upload lifetime fix), and
`1cd442981faf137e4370fffdc9d73ab91cce8c7a` (original archived proposal).
No reference-fork clone implementation was inspected or used.

## Removal inventory

Entire headers `decode_route_ledger.hpp`, `decode_service_budget.hpp`, CPU
`service_budget.hpp` and their derived `tests/cpu/decode_joined_mtp4.cpp` fixture
are deleted. This includes all fields, nested types and methods in those files;
`removed-symbols.json` records every identifier appearing in deleted source as
an exhaustive token inventory (including ordinary C++ words).

Public types removed: DecodeRouteLedger, DecodeLedgerSnapshot, DecodeLedgerSources,
DecodeLedgerOutput, DecodeServiceBudgetScope, ServiceCounter, ServiceClockScope,
NativeServiceBudget. Removed helpers: decode_joined_requested,
decode_route_ledger_eligible, decode_route_ledger_dispatch_eligible,
decode_native_owner_call, service_layer_complete, service_activation_rows_add,
service_cpu_now, service_wall_now. Removed caller state/hooks: joined_decode_requested,
joined_request_sequence, ExpertDispatch::decode_route_ledger,
ExpertPool::service_budget/service_budget_, all route attempt/record/window/commit/
emission/final-wait/close hooks, worker service clocks, task/phase accounting.

The old STRATA_DECODE_ROUTE_LEDGER and STRATA_DECODE_SERVICE_CPU diagnostics cease
having an implementation. No replacement collector is added. No CMake or non-archive
scripts register the removed fixture at this base. Historical bench build scripts
still name it and are retired recipes, not runnable validation for this proposal;
Main must not invoke those recipes unchanged. Historical reports remain evidence.

## Preserved safety and provenance limitation

Native pool exceptions now use a minimal local catch that logs, releases GPU waits,
flushes and aborts before borrowed dispatch owners can unwind. This uses existing
own-fork fail-stop operations without retaining the diagnostic owner wrapper.
Existing wait_done/wait_parked, parking epochs/completion atomics, MTP upload leases,
drain, final verifier wait_commit, FP32/reference and normal decode timings remain.

GU reuse provenance is unresolved. Its header, ExpertDispatch pointer and job touch/
completed hooks remain; the removed ledger owned its only request binding and its
cross-checking receipt, so request-level GU collection is unavailable in this proposal.
Do not silently restore it or claim retained GU behavior. R716/Main must decide its
origin and whether an independent allowed owner may be restored. This proposal does
not certify the remainder of the tree as free of other reference-derived code.

## Main-owned validation recipe (not run)

1. Review complete diff, compare R715 caller inventory and R716 GU findings; use
   pins.json to reject any base mismatch. Resolve concurrent product changes before
   applying; never overwrite current files from proposed copies.
2. At a serialized build/GPU boundary run `git apply --check` on the patch, then apply
   after review. Search active include/src/tests/cmake/tools for removed type/helper/
   header names and for STRATA_DECODE_ROUTE_LEDGER/STRATA_DECODE_SERVICE_CPU consumers.
3. Trace configured SYCL CMake source resolver and compile its actual generate/Arc
   entrypoint plus expert_source/pool objects using Main's configured build command.
   Remove fixture consumers from any current local harness. Do not rebuild archives.
4. Run independent existing CPU pool parking/completion, native layout/kernel and
   MTP lifetime/drain fixtures. Add an independently authored failure injection at the
   native call boundary if Main needs executable owner-retention evidence; require
   abort before caller-owner destructors. No worker clocks/counters should remain.
5. In fresh owned child processes, verify normal short and long input, MTP4/coupled
   and applicable cancellation/error paths, final commit completion, output quality,
   262144 physical context requirements and repeated long-prefill performance using
   Main's existing model/fixtures and artifact retention policy. Explicitly report GU
   unavailability and absence of route/service receipts; old env toggles do not restore
   retired diagnostics. No new performance inference follows from removal alone.
