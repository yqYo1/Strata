# prefill-canonical-dense-bank-v1

Unapplied source-only proposal on immutable feature commit1025e0ed04816891c1df84c26ebf1bbc062784eb. No production/previous artifact changes, builds/tests/GPU or branch actions. See source-review.md for ownership, reader hazards and scope; source.diff applies only to the pinned commit.

New default-off STRATA_PREFILL_CANONICAL_DENSE_BANK=1 requires native bank1 and skipped/disjoint canonical/native rows. Startup captures exact original converted WeightTable upload bytes into bounded owned RAM. One existing outer reader transaction prepares both destinations before withdrawal, preserves a charged canonical HC/non-layer/PLE pool, composes private canonical+native layer views, and restores both full owners/PLE/graphs before decode. The canonical norm-bearing Verifier commit graph is explicitly retired too. keep-canonical/overlaps/unsupported ownership reject before retirement; unknown completion or restore failure fail-stops.

This conservative implementation uses separate canonical/native banks plus persistent pool, not an adaptive combined expert/dense bank. Current target steady dense payload is166,024,192B, plus6912B scratch; persistent PLE/HC/non-layer bytes are included. See payload-ledger.json for actual names, corrected142,024,576B original dense maximum and transition/RAM peaks. No physical recovery, speed, quality, RESTORE or262144 pass is claimed.

Artifacts include exact immutable-base hashes, source mirrors/diff, source review, payload ledger, unexecuted host policy and actual conversion/device-payload/metadata/allocator-fault fixture sources, plus Main serial recipe. All compile/runtime qualification belongs to Main.
