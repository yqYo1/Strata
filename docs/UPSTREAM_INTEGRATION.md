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
| Last actual check and fetch | 2026-11-02 21:17:40 UTC / November 3 06:17 JST |
| Observed upstream | `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`, tagged `v0.2.9` |
| Preceding actual check | 2026-10-31 21:13:43 UTC; the observed revision was unchanged |
| Next check at the approximate 48-hour cadence | 2026-11-04 21:17 UTC / November 5 06:17 JST |
| Complete feature ledger | Incomplete |
| All required non-inference functions ported | Incomplete |
| Unmodified, hardware-tuned upstream performance evaluation | Incomplete |
| New inference adoption from the last check | None; that check did not change the engine |

These dates describe the last actual observations, not a new remote check on
November 4. Fetching a release does not mean that all its functions were ported.

## Remaining integration work

The status file's `function_review`, `latest_interval_review`,
`new_interval_function_slices` and `pending_interval` retain the reviewed scopes,
source identities, feature decisions and unfinished deltas. They are partial
reviews, not a certified complete feature ledger. Preserve the original ported
and tested records without converting pending functions into completed imports.

Outstanding work includes sampling/configuration precedence and UI migration,
thinking-budget/answer-room behavior, stream parsing and recovery, native
batch/cache/restore protocol dependencies, CLI policies and subsequent API,
setup and download changes. Reconcile each applicable function through the
observed target. Required server/API/CLI/UI/setup functions must be preserved or
adapted to this SYCL engine even when inference components are retained locally.

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

Default-branch promotion remains separate: PR 180 is blocked by the squash-only
ruleset. The last verified default ref is `05069aeedea43f778b97dce30c729739cb3bce5c`;
keep the measured commits reachable and do not claim default integration.
