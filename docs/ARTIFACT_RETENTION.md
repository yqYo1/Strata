# Measurement and diagnostic artifact retention

Keep information because a named consumer is likely to use it. A completed
debugging step, a successful command, a report review, or an unresolved question
alone does not justify another permanent file. Apply this policy to Git, the
working tree and local state directories at each experiment boundary.

Decide the minimum permanent result fields before collection and review them
again before staging. Full parser output, repeated diagnostic strings, per-layer
arrays and complete histograms remain raw evidence even when placed in a single
JSON file or report. Keep the original values only when a named question needs
them; otherwise retain useful individual comparison samples, conservation
checks, conclusions, limitations and pinned source/history references. A larger
file is not a substitute for deciding which information has a realistic use.

## Active files and Git history

The working tree is for maintained code, current decisions and reusable
references. **Normal Git history is the archive for completed work.** Once a
report, rejected patch or old fixture is superseded, remove it from the active
tree if restoring its original commit is sufficient. Do not retain a second
source snapshot, progress backup, compressed copy, archive directory or bundle.
Another obsolete file linking to it is not a current consumer.

Before retaining a file, identify the decision, regression test, fault mechanism
or concrete upcoming comparison that will use its information, and when to
review that need again. Remove an obsolete index row instead of preserving an
entire chain of stale references. Link historical evidence to a pinned Git
revision. Keep the archive commit reachable; do not rewrite shared history.

Preserve the original source commits referenced by retained measurements when
integrating branches. Use a merge commit for such work and verify those commits
are reachable from the fork's retained default branch before deleting their
branches or recoverable build artifacts. A squash or rebase with equivalent
files does not preserve the recorded commit identities.

Useful active content is normally:

- The current comparison's individual samples, including rejected/failed speed
  samples, exact configuration, source/binary identity and decision limits.
  One compact result is sufficient; the original per-process receipts need not
  also remain checked out when Git holds them.
- Maintained benchmark/test source used by current build targets or an explicit
  near-term implementation. Put it in `bench/sycl`, `tests` or `tools`, rather
  than keeping duplicate proposals under dated results directories.
- The smallest useful reproducer and diagnostic for each **current** unresolved
  failure. Retain a tensor/session capture only if a named current consumer
  needs those actual bytes. Retire superseded-version captures when that
  consumer ends; an old failure remains a failure after its payload is retired.
- A research reference with substantive, nonduplicated findings and a realistic
  retrieval use. Keep essential source/license corrections and a short topic
  index. Completed specs, dispatch/ready/heartbeat/ack/retry/release records and
  full review append proposals are transaction history, not a reference library.

Record enough provenance to assess the result: source, command, effective
environment, hardware/runtime, actual workload, individual measurements,
correctness/fault outcome and limitations. Refer to shared maintained setup
instead of embedding the same full environment, loaded-library maps, compiler
headers and process bookkeeping in every result. Do not commit every successful
health probe, build transcript or debugging output. Worthless transient output
does not need to be committed merely so it can subsequently be deleted.

Keep current state in maintained documentation and committed decisions. Resume
from those and Git; do not create external resume checkpoints.

## Rebuildable binaries

Keep one current known-good engine build by default. Retain an additional
baseline/candidate generation only for a named, imminent comparison; normally
this means at most two generations. Record their consumer and remove the extra
build when the comparison closes, rather than waiting for storage pressure.
Close in this order:

1. Commit the useful results and their individual comparison samples.
2. Select the validated generation to retain.
3. Retrieve the old build's original source and recipe from reachable Git
   history and compare them with its recorded build inputs.
4. Record the deletion decision, retire the extra build and remove stale active
   references. If recovery is unresolved, carry the explicit hold described below.

A separate latest build for every old experiment is not a retention policy.
Count retained generations across project worktrees and local output directories;
copying or renaming a binary does not
create a separate allowance. This applies to project executables,
static libraries, object files, JIT/native dumps and their disposable build trees;
installed toolchains, runtime dependencies and model weights are separate assets.
Keep these project-generated binaries and dumps outside Git, including the
retained current generation. Commit the required source, recipe and useful
measurement evidence instead.

Before creating another build generation, identify the current known-good build
and the named comparison that needs the extra one across all project worktrees.
When the comparison closes, select the validated build to retain and retire the
extra after verifying provenance. Record this in the existing compact result;
do not create a separate build archive or progress backup. An unresolved source
recovery problem remains an explicit provenance hold until reconciled.

A rebuild that reuses the current output path must not overwrite the last
validated build before a fallback is preserved. Hash-check the retained
executable and required same-build fixtures at a named external path, record
their source/recipe and next qualification consumer in the current result, and
count that generation in the same retention quota. A successful compilation
alone does not qualify the replacement. At qualification or comparison closure,
explicitly select the validated generation to keep and retire the extra.

**Commit the measured code before treating a binary as disposable.** Before
building a measurement executable or running a fixture, pin a reachable source
commit, all local changes and generated source inputs, compiler/dependency
versions, build command/options and relevant environment. Check that the
committed inputs actually match those used to build
the measured executable, and record its hash. A nearby commit or a source path
alone is insufficient. If a measurement used an uncommitted snapshot, reconcile
and commit its necessary original inputs before deleting its last build/source
copy. Do not claim reproducibility from a binary hash alone.
Record the tested source revision separately from later result/documentation
commits. A newly committed report or the current HEAD does not change the source
identity of an existing executable.
Include benchmark/controller changes and generated workloads among the pinned
inputs. A later commit does not qualify a different snapshot retroactively;
preserve the original inputs and resolve any mismatch explicitly.
Controllers must check the pinned inputs, build record and actual executable
before launch, and stop if evidence is missing or inconsistent. Scope this check
to the complete measurement inputs; unrelated worktree edits need not be added
to that commit. Do not update hashes to make an inconsistent run pass admission.

Before removing an old build, verify that its measured source and recipe can be
retrieved from the recorded commits and that no current comparison needs the
executable. Match the exact final binary and all required build inputs. A
matching binary/receipt hash proves artifact continuity, not recoverable source;
a seed build, similar root name, later build or review eligibility label cannot
substitute for that check. Record a review's missing recovery steps in the
current compact result, keeping unresolved legacy holds distinct from qualified
comparison generations. Do not claim global retirement from a current-build-only
inventory. Preserve original evidence and source provenance without retaining
duplicate source snapshots. Build reproducibility means recovering the correct
inputs and recipe; byte-identical binaries may additionally require the original
toolchain. Do not retain an unusable orphan binary as a substitute for source.

For each temporary provenance hold, record the exact missing inputs, current
consumer, next bounded recovery action and review point in the existing compact
result. Preserve unique unreconciled source/recipe evidence; a hold does not
justify retaining unrelated logs, caches or duplicate build outputs.
If bounded recovery establishes that the required original inputs cannot be
recovered, retain the compact provenance failure and retire the unusable binary.
Do not renew the hold indefinitely or treat that executable as a validated
comparison reference.

## Temporary diagnostic collection

Before launch, use a private output directory, finite process deadline and a
supervised size budget. The default total raw text budget is **64 MiB per job**;
record any finite replacement and the question requiring it before launch.
A budget-triggered stop is incomplete. Keep all speed samples used by a
comparison, but extract them from routine logs and discard the logs after
verification. A different extension does not make generated output source.

Set SYCL, Level Zero, driver, profiler and experiment variables before native
`exec`. Keep them immutable after runtime initialization. For a small first
check use the qualified tracing and validation settings:

```sh
UR_ENABLE_LAYERS=UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION
UR_LOG_TRACING='level:info;flush:info;output:stderr'
```

The layer list uses commas; logger options use semicolons. For a long model
check prefer parameter validation, warning/error logging and `STRATA_TRACE=1`
without the full API trace. Obtain speed samples from separate clean processes.

After closure, retire successful API/synchronization logs, repeated profiler
windows, raw timelines, duplicate kernel dumps and normal build output once
their relevant evidence is verified. Keep a complete trace only for a named
question requiring its history, with an owner, finite byte budget and next
review point. Failure excerpts normally fit within **256 KiB per distinct
mechanism**. The same signal name alone does not establish the same cause.

## Cleanup and deletion records

Main serializes cleanup with builds, tests, GPU work and profiling. Do not touch
an active run or remove a current consumer's last required fixture. Preserve
original success/failure status and numerical criteria. Retirement is not a
new validation result.

At each work boundary, reconcile the actual retained paths and build generations
with their named consumers. A written policy, deletion plan or eligibility label
does not establish completed cleanup. Keep any remaining consumer, missing
recovery evidence, next bounded action and review point in the existing compact
result. After verified deletion, remove stale active index/path references;
do not create a separate resume checkpoint or permanent inventory for this check.

For tracked files, record the original reachable commit, selection/reason rules,
exceptions and counts/bytes. **Git's deletion diff already records each path and
blob hash; do not duplicate thousands of those rows in another permanent
manifest.** Restore a particular file with
`git show <original-commit>:<path>`. Never rewrite a pinned historical receipt
into a success or silently replace its contents.

For untracked files or large external captures, record explicit paths, hashes
and reasons before deletion because Git cannot recover them. Verify no active
consumer needs them. Keep only the compact deletion decision in Git; retire
the audit's temporary inventories and verbose review output after reconciliation.

The benchmark results directory must remain browsable. A directory approaching
GitHub's listing limit is a review alarm, not a reason to split or bundle the
same unnecessary information into more directories or larger files.
