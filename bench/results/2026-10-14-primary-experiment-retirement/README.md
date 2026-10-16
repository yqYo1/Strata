# Experiment preservation and retirement

Niko1221/Strata is the sole upstream; `feature/sycl` is the primary working branch. Eight closed local
experiment branches/worktrees and their three remote branches were retired. The fork now has three branches
locally and remotely: primary, default `main`, and one explicitly retained bulk reference experiment.
Separate-clone compiler/ggml/fixture dependency worktrees were not removed.

Five dirty source-only proposals, 73 file instances, were archived in commits. All eight original heads and
archive commits, their reports and the earlier 104 preserved tips are reachable from primary preservation
merge `4526800257fe49c0db02a414f21b38dac2586e54`. This is archival, not inference adoption.
Primary's engine tree is unchanged. [preservation.json](preservation.json) records original heads, archive
commits and source hashes; post-retirement source restoration and Git connectivity were verified.

Example restoration:

```sh
git show 1cd442981faf137e4370fffdc9d73ab91cce8c7a:include/strata/core/decode_service_budget.hpp
```

Closed reproducible build outputs and duplicate metadata retired here total 750,271,455 logical bytes.
`artifact-deletion-manifest.jsonl.gz` contains path/hash/reason records; [retirement.json](retirement.json)
records compressed/decoded hashes. Symlink targets were retained. The reviewable plan was committed/pushed
before deletion, the shared measurement/build/test lock serialized cleanup, and no living process referenced
the retired worktrees. Original failed statuses were preserved.

The retained bulk branch is a component source donor and historical benchmark reference. Review it at the
first primary-side lease/diagnostic port and matched comparison boundary. New implementation children must
start from current `feature/sycl`. Superseded skill-publication drafts were removed from the reference tree;
the corrected actual-primary project skill is committed and the reference worktree is clean.

Main 0.1.39 functionality still needs adaptation to primary's 0.1.38 inference family before default promotion.
Because main ancestry is archived, an already-up-to-date merge cannot establish functional integration.
Use the pinned pre-archive trees, original common ancestor and seven-file merge preview in the
[source audit](../2026-10-14-primary-strata-source-audit/README.md). Preserve both primary optimization and
main CLI/server behavior, validate, promote through an own-fork PR, then check/evaluate current Niko updates.
This boundary does not claim a new upstream check, inference adoption or achievement of performance goals.
