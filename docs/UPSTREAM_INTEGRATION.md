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
| Last actual check and fetch | 2026-11-07 20:27:59 UTC / November 8 05:27:59 JST |
| Observed upstream | `146df3c468739147f520ac3816de806f71232571`, `v0.2.11-98-g146df3c4` |
| Next check at the approximate 48-hour cadence | 2026-11-09 20:27:59 UTC / November 10 05:27:59 JST |
| Current upstream Python/setup/control/server/API/CLI/Web import | Present as a partial functional import from `146df3c4`; full validation remains open |
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

The current Unsloth v3.1 catalog/pinned metadata and stock DFlash2 pack are
present. The native DFlash2 candidate is opt-in and its corrected 32,769-token
route admission passed, but sustained thinking generation, clean paired speed,
task quality, physical 262144/RESTORE and default adoption remain open. MTP
selection for existing configs is preserved. These component and route checks
do not qualify a model route or a new inference adoption.

## Remaining integration work

The status file's `function_review`, `latest_interval_review`,
`new_interval_function_slices` and `pending_interval` retain the reviewed scopes,
source identities, feature decisions and unfinished deltas. They are partial
reviews, not a certified complete feature ledger. Preserve the original ported
and tested records without converting pending functions into completed imports.

Outstanding work includes the complete native function ledger and actual Arc
entrypoint review; disk-session, BGEN/capability, tracking/recovery/alias and
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

Default-branch promotion remains pending. PR 180 is mergeable but blocked by
main ruleset `24638744`, whose pull-request rule permits only squash merges.
Repository-wide permission for merge commits does not override that rule.
The last verified default ref is `05069aeedea43f778b97dce30c729739cb3bce5c`;
keep original measured commits reachable on `feature/sycl` without bypassing the
rule or claiming default integration.
