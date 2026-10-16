# Orca R764–766: depth, scope and Main review

3 independent Luna tasks started through Orca in the current feature/sycl worktree, with no new branches.
All3 effective launch models gpt-6-luna; worker_done Task/Dispatch and succeeded settlement checked.
Main read every completed report fully and requested worker release before acknowledging the full delivery.
No steering/interrupt during research. Main implemented, built and ran14 serialized GPU transfer processes in parallel.

The user's new scope instruction arrived after the3 tasks started. Main recorded it in AGENTS.md for new tasks:
one coherent investigation with enough breadth for alternatives, applicability and counterevidence; no trivial
fact-lookup flood, no filler repeating known context. Future report budgets allow substantive analysis rather
than a universal hard short-wordcap. Existing tasks were not steered or extended in flight.

R764: Qwen's official same-family report provides useful residual FP8-storage evidence, unlike generic mixed
precision alone. Main opened arxiv2608.30320v1 and verified section2.2 Inference Efficiency: FP8 residual storage
and the quality caution for sparse branches are actually present. This supports a trial, not a measured FP16/BF16
comparison on this IQ3 fork. Its proposed context-clustered NLL/task screen is a design;16 contexts and unknown
acceptance margins do not prove broad nearzero capability loss. No quality/admission threshold was adopted.

R765: immutable v7.1.1 and current-head sources identify a specific possible queue back-pressure host drain.
The installed library's filename/hash and compiler directory do not establish that source commit or actual queue
mode. Our transfer component tested idle queues, persistent addresses and one copy/wait per sample; it did not
exercise a deep pre-existing kernel queue or issued-command threshold. It cannot decide the model's exact
runtime staging/cache/driver mechanism. Main's fresh data verifies timing behavior rather than this cause.

R766: useful non-LLM register threshold examples, qualified across PVC/CDNA2; not B570 performance evidence.
The smaller heads8 SLM estimate remains source-only. Main opened Intel's register-spill guide; exact compiler
resource and kernel/full-phase timing remain necessary. Recomputing cheap values can add work and a source
SLM delta is not an occupancy measurement. Original report is unchanged; no external code adopted.

All3 investigations contain concrete primary evidence and actionable limits. R764 overlaps older questions but
adds same-family evidence verification and a matched screen design; next scope should advance an actual
quality result rather than ask the same no-direct-FP16-data question again. R765 should not be reduced to a
one-function lookup without surrounding queue/driver behavior. R766 connects compiler thresholds to a full
kernel tradeoff; a future scope can compare several independently sourced implementations after license review.
