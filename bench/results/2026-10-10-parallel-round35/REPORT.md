# Recurring parallel research and main-agent execution

At the user's status check, both round34 Luna assignments had returned and no
research agent was active. Root resumed two separate round35 assignments using
the shared prior-report directory and registry. They returned while root was
running the bounded D numerical capture. Root reviewed both reports and assigned
two distinct round36 investigations plus a separate Sol source implementation.
The continuation rule is now explicit in AGENTS.md.

| Assignment at registry v33 | Model | Scope | Status |
| --- | --- | --- | --- |
| Cache count audit | gpt-6-luna | Pair-keyed route counts, evaluated speculative work, DONE totals and callback ownership | Running, read-only |
| GDN kernel audit | gpt-6-luna | Actual T8192 kernel writers, scratch, collectives and reduction determinism | Running, read-only |
| Cache pair diagnostic | gpt-6.1-sol | Opt-in finite per-request host counters in an isolated worktree | Running, source only |
| Main agent | Root | D two-full capture and returned-report review | Runtime diagnostic active |

Registry v32 preserves the earlier snapshot with round35 running; v33 records its
completion and the subsequent assignments. There are75 completed Luna reports.
Past reports and decisions are shared before each assignment. Root does not
intervene mid-task. Researchers write only their assigned report files; root
owns all builds, tests, model/profiler runs and cleanup, serialized by the common
measurement lock. No test of the new Sol code has run.

## Returned findings and decisions

Round34 confirmed that D already captures layer9 input, post-GDN residual and
post-MoE residual for rows98304–98335. No new hook or rebuild is required before
the first comparison. Round35 maps those records to the actual in-order compute
queue and explains their limits: final recurrent state includes all8192 chunk
rows, and synchronous capture may change timing. Equality of the fixed window
cannot clear the earlier uninstrumented C mathematical rejection.

The cache audit confirms the actual model/profile has48 layers and512 experts.
The explicit128-slot per-layer policy grants2 slots to layers0–46 and34 to
layer47. The existing196632-byte STRP file contains24576 unique ranked pairs and
a rank-to-slot table, with no route counts or provenance. A rank-based redistribution
changes the selected set but has no measured hit or speed benefit. Root holds
that policy change. The separate Sol assignment supplies exact pair-keyed
classification evidence before any policy experiment; count semantics require
independent review and root-managed qualification.

## Main work and evidence limits

Root's capture controller v3 CPU preflight passed in2.230s with normal exit0 and
no cleanup or survivors. Its SHA256 is
`2a81a49356e687df078adfe97b53d4094047c18797db560cc21b306752cc0194`.
The actual root-owned GPU job is active under executor session28794, with a5400s
overall deadline,2400s per-protocol deadline,64MiB text budget and128MiB binary
capture budget. Both the initial32K control and resumed32K reference passed the
existing math gates; the first full256K request is running at this snapshot.
No final numerical/capture/health result is yet claimed. Its authoritative live
receipt is named in the registry. Root keeps supervising the same process.

The original C full256K receipt remains mathematically rejected. Adoption,
full lifecycle and clean performance eligibility stay false. Logged diagnostic
elapsed times do not support speed claims. Full-head, all66 live state parts,
IDs/logprobs, MTP, physical262144 context, fresh fault/health and normal ownership
checks remain required. Successful comparison fixtures and large live captures
stay outside Git for named consumers; compact reports, sources and receipts are
committed. No artifact cleanup runs during the live model.
