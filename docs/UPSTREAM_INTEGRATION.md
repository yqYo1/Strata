# Personal fork: upstream integration

This personal fork uses **Niko1221/Strata** as its only upstream and targets this
PC's Intel Arc B570, Ryzen 5 5600X and 128 GB RAM. CUDA/HIP inference work and
upstream contribution are outside this task. XeStrata is a performance reference
only; its code, helpers, tests and design must not be copied or adapted.

Use the project-only [upstream sync skill](../.agents/skills/strata-upstream-sync/SKILL.md)
and [AGENTS.md](../AGENTS.md). Work remains on `feature/sycl`; promote completed
work through the fork's PR while preserving measured source commit identities.

## Current checked state

The [upstream status](../bench/results/2026-10-15-pr6-recovery-audit/upstream-status.json)
records actual fetch and remote-ref observations, separately from integration.

| Item | Recorded state |
| --- | --- |
| Last actual check and fetch | 2026-11-11 19:42:07 UTC / November 12 04:42:07 JST |
| Observed upstream | `c1ff07f1a1817594811c960039571c6e268fe89a`, `v0.2.13.1-152-gc1ff07f1` |
| Next check at the approximate 48-hour cadence | 2026-11-13 19:42:07 UTC / November 14 04:42:07 JST |
| Current upstream Python/setup/control/server/API/CLI/Web import | Partial import from `146df3c4`; full validation remains open. Later changes through `c1ff07f1` are fetched, with the plain GPU-list ordering fix imported and tested |
| Complete native function ledger and required-function port | Incomplete |
| Unmodified, hardware-tuned upstream performance evaluation | Incomplete |
| New inference adoption from this check | None |

The import record covers 630 changed paths in its named scope, including the
current setup/model catalog, Python control and server/API/Web code, CLI
integrations, packing/tokenizer helpers and upstream host tests. It preserves
the local owned-file console shutdown repair and existing setup/benchmark
adaptations. Focused configuration, streaming, API and CLI changes have their
own host evidence; that does not complete the native function ledger or full
host validation. The interval's changed source and partial status are recorded
under `current_sync.functional_import` in the status file.

The latest observation adds151commits and303changed paths beyond `504b1433`,
or371commits and995changed paths beyond the partial `146df3c4` import. Bounded
source reviews cover router, control/launch and server/reader changes; they do
not complete the function ledger. The plain `--gpu` fix from Niko
`2b9df3fe71cb24cef3052c332e032ea681729abe` now clears stale GPU ordering when
replacing a saved selection, while preserving explicit roles and unchanged
selections. Its four regression tests and14existing GPU-role tests pass with
no skips; help text is updated. The source and evidence are recorded under
`current_sync.functional_import.incremental_function_ports.plain_gpu_order`.

The Control/App Intel Arc setup fix and expert-arena shortfall/warning changes
remain required-function candidates. Fixed-order skinny BF16 GEMMs and compiled
model geometry need component-level decisions; upstream A750 timings and
default-build checksum claims are not measurements or qualification on this
B570. No new inference component from the observed interval is adopted. The
remaining source and pending review are recorded under `new_interval`; older
unfinished scopes remain open.

The [Arc setup source map](research/upstream-arc-control-setup.md) identified
that the broad import lost this fork's native setup routing. Source `b8f0a5bbf`
restores the Linux checkout entry to `tools/setup_intel.py` and sends installed/
previous native configurations through it before upstream update/migration.
The focused10-test host plan is committed and pending;no actual install is
qualified. Control/App/MCP setup and the CLI launcher's direct `adopt.plan` /
`adopt.write` config replacement still need their native adapters. Control cannot
invoke the source builder while its component route is pending. The
[unmodified Arc benchmark source map](research/upstream-arc-benchmark-readiness.md)
records B570 build/workload prerequisites; its helper dependency closure,
target configuration, runtime qualification and actual measurement remain open.
These source reviews did not perform another upstream fetch or import.

The current Unsloth v3.1 catalog/pinned metadata and stock DFlash2 pack are
present. The focused v3.1 setup/pins/shard-discovery and DFlash2 packing tests
passed77host cases with1skip; this does not qualify native v3.1 model/type
execution. The native DFlash2 candidate is opt-in. Its corrected32,769-token
route admission and32K-plus sustained thinking generation have been exercised;
the cache-loan comparison also closed in six fresh processes. Those measurements
charge validation/observation and do not qualify a clean, repeated DFlash2-versus-
MTP speed comparison. Task quality, physical262144/RESTORE and default adoption
remain open. Existing configurations retain their MTP selection, and the
experimental cache loan remains off by default. See `current_sync` in the status
file for the original checks and limits; component success is not model adoption.

## Remaining integration work

The status file's `function_review`, `latest_interval_review`,
`new_interval_function_slices`, `pending_interval` and `new_interval` retain the reviewed scopes,
source identities, feature decisions and unfinished deltas. They are partial
reviews, not a certified complete feature ledger. Preserve the original ported
and tested records without converting pending functions into completed imports.

Outstanding work includes the new Control/App Arc setup fix, the complete native
function ledger and actual Arc entrypoint review; disk-session, BGEN/capability, tracking/recovery/alias and
remaining API/CLI behaviors; complete host validation; and the unmodified
upstream's hardware-tuned comparison. Reconcile each applicable function
through the observed target. Required server/API/CLI/UI/setup functions must be
preserved or adapted to this SYCL engine even when inference components are
retained locally. Keep DeepSeek and 35B execution paths excluded unless the user
explicitly requests them.

Evaluate unmodified upstream with hardware-tuned settings and the maintained
`bench.sh` workloads: long input, agent session, and thinking-enabled long
generation with observed strong and weak MTP acceptance. Choose inference
changes per component and separately for prefill/decode; an overall gain does
not justify replacing a faster local part. Do not substitute ancestry or a
history-only merge for implementation and validation.

Current engine selection, performance and open correctness gates are maintained
in [SYCL_NATIVE16_PREFILL.md](SYCL_NATIVE16_PREFILL.md) and the single
[performance CSV](../bench/performance-records.csv). The October 14 source audit
and its old speed figures remain in [their original document revision](https://github.com/yqYo1/Strata/blob/0820688a4ff14947b659d96bff112ed293559ca8/docs/UPSTREAM_INTEGRATION.md);
they do not describe the current selected engines.

Default-branch promotion remains pending through PR 180. Main ruleset `24638744`
permits only squash PR merges. Before merging, verify that every original
source/evidence commit referenced by the performance CSV remains an ancestor
of the pushed, permanent `feature/sycl`, and keep automatic head deletion
disabled. Under the project workflow, the default may receive the PR content
snapshot while the primary retains those original identities; merge the updated
default back into the primary normally and verify reachability again. This changes no protection
or measured source history. The last verified default before that operation is
`05069aeedea43f778b97dce30c729739cb3bce5c`; do not claim integration until its
remote ref and clean ghq mirror are verified.
