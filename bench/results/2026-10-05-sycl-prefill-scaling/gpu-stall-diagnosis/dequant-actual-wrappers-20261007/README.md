# Actual dequant wrapper comparison on 2026-10-07

This follows the [CPU-only preparation](../dequant-launch-properties-20261007/README.md)
on the same B570 host and boot, after the failed full-context process was
removed and a logged GPU health probe passed without reset. The revised
controller records those prerequisites and owns each child through GDB.

Both executables use one shared host fixture object. Their actual engine
kernel objects differ only in the two launch-property declarations. All
144 pairs of guarded FP16 outputs match every byte, totaling 354,511,872
bytes per side. Nine types, seeds 7/23, small/640/641-row shapes, flat/GU
layouts and zero/positive/negative scales produce finite values and intact
guards. Both processes exit normally without force, surviving owners or
new xe faults.

The [kernel-specific API analysis](analysis.json) observes 72 flat and
72 GU successful launches in each process. All 144 original launches use
the cooperative flag; none of the changed launches do. The [complete
receipt](run/record.json) preserves output sizes/hashes, environment,
binary/object/controller provenance and lifecycle checks. API log tails
are bounded; their full private hashes are retained.

These are isolated actual-wrapper checks. The private engine executable
has not run a model, and these logged durations are not clean speed
measurements. Full-model heads/state, repeated performance, capacity and
a causal connection to earlier stalls remain unproven. No production
kernel change is adopted by this evidence.
