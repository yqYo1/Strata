# Artifact review, 2026-10-18

The active checkout had 6,304 tracked benchmark files (175,116,113 bytes).
The PR6 investigation alone had 4,930 files and 4,096 immediate directory
entries, exceeding GitHub's listing limit. Only 854,257 bytes were exact
duplicates: removing duplicates alone would not address the problem.

The archive boundary is **8063aacc6749d18089f7a53db5d3e47b745f348c**,
already reachable from the fork's default branch. Its tree equals the reviewed
feature-branch snapshot. Shared Git history is retained without rewriting it.
For a particular historical file:

```sh
git show 8063aacc6749d18089f7a53db5d3e47b745f348c:bench/results/<path>
```

Retirement removes 6,274 tracked files (173,056,486 original bytes). These are
closed dispatch/retry/ack records, old reports and reviews, unsuccessful or
superseded proposals, source mirrors, patches, one-run controllers, repeated
environment/library maps and routine output. They are not recombined into
archives. Git's deletion diff is the path/blob manifest. Historical Markdown
links are pinned to the archive revision rather than keeping obsolete trees.

The active evidence has actual consumers:

- Long-prefill comparison: all eight samples, inputs, configuration, current
  build identity and limits. Serial median 725.7 tok/s versus candidate 733.5;
  candidate adoption remains on hold. This is not a long-decode, quality,
  physical-262144-context or RESTORE qualification.
- Hardware budgeting: individual host-transfer samples and timing breakdowns,
  current attention/resource attribution and the dense-model reference caveat.
  Command duration is not hardware utilization.
- Current numerical/fault questions: representative original failing receipts,
  first divergence evidence and distinct timeout/reset contexts. Retirement
  does not turn an original failure into a pass.
- Current GDN work: maintained independent candidate source, actual model
  geometry, numerical/provenance contract and the focused review corrections.
  Only two research bodies have an immediate retrieval use. The short index
  points to Git history before any repeated investigation.

The review used inventories, executable dependencies, result consumers,
supersession/retrieval metadata and representative evidence. Reviewers did not
read every historical research body. Main did not load knowledge collections
merely because they arrived.

Before deleting dated fixtures, the live QSA test received the byte-identical
previously checked DMA buffer-lifetime fix. The transfer-ceiling CMake source
was moved to `bench/sycl`; reusable measurement ownership/guard code was moved
to `tools/sycl`. Equality was checked before deleting the duplicate originals.
No new GPU run or inference performance is claimed by this cleanup.

The current engine's build receipt has 476 project input hashes. All 476 match
the same paths at the archive revision, including its build recipe. External
compiler/runtime/dependency inputs are separate pinned assets. A binary hash
or nearby source commit alone is insufficient evidence of recoverable code.

The binary policy keeps the latest known-good engine generation and only
additional generations with an imminent named comparison, normally two total.
Measured source, generated inputs, local changes and the build recipe must be
committed and verified before retiring an old build. Unreconciled inputs hold
up deletion; keeping an orphan binary does not resolve missing source.

Separately, 23 obsolete session/state payloads totaling 58,278,952,972 bytes
were removed from closed, superseded Oct 8/9 runs. Their original result records
were preserved; the retained deletion record contains paths, hashes and reasons.
Those historical runs do not qualify the current engine's 256K/RESTORE path.

The retention rules are in [ARTIFACT_RETENTION.md](ARTIFACT_RETENTION.md).
Review actual consumers at every experiment boundary. Do not grow another
per-run diary, keep all successful logs, or use compression to avoid selection.

A further 193 untracked copies (8,163,471 bytes) were removed only after exact
recovery from reachable Git blobs or committed patches was verified. Another
172 closed transport/cache/obsolete fixture outputs (6,065,631 bytes) were
retired. The compact path/hash/reason ledger is
`bench/results/2026-10-15-pr6-recovery-audit/build-retirement-20261018.json`.
It contains deletion metadata, not bundled original contents. Unreconciled
legacy source copies remain on hold; they must not be described as recoverable
merely because another nearby source version is committed.

The host-transfer reference also needed a provenance correction: its reported
base `57477f26` is not the exact built code. All 380 project pins were retrieved
and matched at **7d2d344fafc6d64b1baffcafa4904b966dea20fe**. The original
receipt is unchanged; the current reference explicitly records the correction.

Legacy external builds have a separate recovery gate. The original inventory
has 4,065 ELF/archive files (4,078,627,544 bytes, 326 roots), now individually
hashed. Twenty old generations passed source/recipe recovery checks. Their
3,187 generated build files (1,961,721,285 bytes) were removed under the
exclusive measurement lock, after committing the recovery proof and recording
paths/hashes/reasons. The first four generations account for 1,676 verified
project inputs; subsequent generations also required exact input checks,
including overrides recovered from Git history or committed patches. Original
root receipts remain unchanged. This is retirement, not a new numerical or
performance qualification.

The canonical-bank source needed reconstruction rather than a nearby commit:
applying the archived patch to its recorded base reproduced all 374 project
input hashes, including five pins absent from the initial lookup. The pre-Xe
reference matched all 595 source inputs at its recorded commit and all five
recorded executable hashes. Its original compressed receipt and build recipe
are recoverable from the archive. These checks recover historical inputs;
they do not adopt either historical implementation into the current engine.
The host-transfer reference now also records its effective build settings.

The latest retained engine, `main-qsa-score-gemm-entry-engine-build-v2`, still
matches its measured hash and retains the runtime archive needed by the current
GDN work. Remaining legacy roots await a complete source/recipe join before
deletion. Inventory alone is insufficient: 82 original roots had no nearby
JSON. Further receipt joins and source recovery are needed where identity or
configuration remains unresolved. An unexecuted ABI-only record is not a
measured baseline even when its source is recoverable.

Main owns the remaining provenance audit. Its bounded inventories and reviews
are in `~/.local/state/strata-sycl/artifact-retention-audit-20261018`; they are
current audit inputs, not resume backups. Review and retire them when this gate
closes. Unreconciled inputs block deletion of the affected files; current tuning
with separately verified source can continue. Serialize actual cleanup with
builds, tests and GPU work. Preserve unreconciled measured source first; do not
conceal legacy provenance gaps by discarding their last code copy.
