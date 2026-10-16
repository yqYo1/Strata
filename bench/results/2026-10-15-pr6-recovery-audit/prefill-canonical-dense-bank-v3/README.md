# Canonical dense bank v3: clean-base source reconstruction

Fixed base: `5942f1853f006b0df45e15fd4f1c60bc37b7b3b3`.
Source-only, unapplied, uncompiled and untested. Main alone owns execution and integration.
No live product file, original v1/v2 artifact, environment, service or active process was
changed. No GPU, build or fixture execution occurred. No speed, quality, numerical,
restore or runtime-safety qualification is claimed.

The SHA-pinned v2 source.diff (`a4a8258bc087a8e79fbce233866385f36dd6cd9941c1f0a7498acffe77ed77cc`)
contains independent own-fork canonical-bank changes. Every one of its 34 hunks matched
one exact context in the new Git snapshot; no fuzzy merge or old generator mirror was
used. New source.diff is generated against this fixed clean base only. hunk-ledger.json
records each original hunk and before/after context hashes; proposal-manifest.json pins
every affected base/proposed file. No reference-fork clone or implementation was used.
The reconstructed modified files contain no route/service/GU collector types, includes,
hooks or reference notices. Source archives of the removed diagnostics were not saved.

## Source and ABI review

The complete canonical diff was reviewed by component: 28-line policy, 210-line bank,
NativeDense prepare/cancel declarations and implementations, WeightTable capture
signature and upload hooks, Verifier canonical commit retirement, private Prefill PLE
selection, startup ownership and NativeDensePhase transaction connections. The v2
class-scope correction remains: both switches and snapshot are local before canonical
pool sizing/load; GpuStage is byte-identical. Main's complete TRACE observer and MTP
prompt helper are also byte-identical. Their current-base hashes, and untouched current
expert/pool fail-stop, MTP, SessionGraphs and selected SYCL kernel hashes are recorded
in retained-invariants.json. These are source comparisons, not executed tests.

WeightTable::load has two additional arguments with defaults. Existing source calls
remain compatible, but old compiled objects have an incompatible ABI and must all be
rebuilt against candidate declarations/definitions. NativeDense prepare/cancel and
Verifier retirement declaration/definition/calls match in source. The canonical bank
is header-only; no new CMake translation unit is required. Main must trace the actual
configured SYCL source resolver and entrypoint before compiling.

Canonical capture records the actual converted H2D bytes at startup without D2H.
The existing native bank remains separate. Canonical prepare and native prepare occur
before retirement/withdrawal under one NativeDensePhase lease. Canonical projections,
96 HC norms and lookahead references stay in the persistent pool; private per-layer
views combine native and canonical entries. Verifier commit_exec_ carrying canonical
GDN/indexer norm pointers is retired after drained dense retirement. Existing complete
SessionGraphs retirement and recapture are unchanged. Restore republishes both Full
owner sets, rebinds canonical and native PLE fields, frees the persistent pool, recaptures
session graphs and resumes the verifier. MTP, NativeEmbed, all session state and short
input's Full layout remain required. Source order does not prove reader retirement.

Unknown allocation or contradictory pointer/status results fail-stop. Confirmed
OOM-null during prepare leaves original publications intact with checked reservation
rollback. Standalone restore OOM retains Bank owners; generator restore failure is
fatal and cannot resume decode. Upload/drain/free/graph uncertainty also fail-stops.
The base's new minimal native expert exception fail-stop is wholly untouched.

## Planning limits and execution handoff

Default is OFF. Canonical payload capture limit is 3GiB, native snapshots separately
3GiB; aggregate payload bound is 6GiB, excluding maps/staging/allocator/model/session
owners. Metadata rows are capped at 16384. Target canonical/native snapshot payload is
3654986240 B. Persistent pool is 82821120 B; separate canonical/native banks are
32284672/50918400 B. Expert+dense+pool reservation is 1529172992 B, not the hypothetical
adaptive 1440161792 B layout. Target weight entry/restore peaks are 3821010432 and
3770092032 B; all other live owners remain charged. payload-ledger.json is unchanged
historical index-derived planning, with a v3 annotation, not a fresh measurement.

Main's existing 48-layer/78453 native-bank model result belongs to the base and does
not validate this canonical extension. main-execution-recipe.md retains the serial ABI,
host/component/fault, actual MTP/PLE/NativeEmbed, physical 262144-context RESTORE,
quality and repeated >600 toward1000 tok/s gates. Included host_fixture.cpp and
component_fixture.cpp are unchanged own proposal fixtures and remain unexecuted.
Their allocator fault shim covers allocation only; other uncertainty paths still need
Main-owned injection. Current free memory, rollback/restore peaks and whole dependency
cost must be measured; historical unexplained headroom cannot be reused as capacity.

## Retention

Keep source.diff, this README, proposal-manifest.json, hunk-ledger.json,
retained-invariants.json, payload-ledger.json, main-execution-recipe.md,
reconstruct.py, preservation.py and the two unexecuted fixtures as compact evidence.
main-review-consumer/ is the sole redundant full-source set: owner Main, purpose
immediate complete-source/ABI review and application comparison, byte budget 2MiB,
next review at Main's compile/integration boundary. It was assembled from clean Git
base plus canonical hunks, never old full mirrors. Delete it after that consumer ends
with path/hash/reason manifest and serialized artifact cleanup. No base mirrors or
raw tensor/profile/run captures were created.
