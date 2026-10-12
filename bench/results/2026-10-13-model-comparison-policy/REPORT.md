<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Model comparison criteria and long-output controller checks

Generated output identity is not the general adoption criterion for different
inference implementations. Even greedy decoding can choose a different next
token after a small floating-point change; later histories then differ.
The Xe b9 ninth-token divergence is an observed difference, not evidence that
quality regressed. Its closed diagnostic remains operationally successful;
quality and repeated clean performance comparisons remain open.

Main created a separate v3 long-output controller. It records exact first16
and full128 output fingerprints while allowing a normally closed, finite,
fault-free request to contribute a speed sample when outputs differ. Quality
stays explicitly unevaluated and unqualified; no inference port is adopted.
Different generated histories can change decode work, so those timing samples
are not a claim of identical internal workloads. Operational faults, NaN,
incomplete output, wrong context, source/provider mismatch and abnormal closure
remain hard failures. Arithmetic-preserving CPU bit-parity fixtures are kept.
The previous source-only v2 packet and all original result statuses are unchanged.

Quality evaluation will compare full distributions on identical fixed token
histories, including target-token likelihood/perplexity and top-token margins,
then representative held-out task scores with declared task-specific margins
and uncertainty. Top5 logprobs alone cannot establish full-distribution KL or
model quality. No universal numeric tolerance is adopted. R318 records reusable
metric methods and the still-missing verified production teacher-forcing hook.

Actual host checks: v2 and v3 each pass eight parser/admission/comparison tests
twice, then two owned CPU children cover normal GEN128/QUIT and once-only
STOP/cancel/QUIT. Cancellation stays failed/incomplete. No GPU/model request
was executed at this boundary. Both existing free and contrib-llvm engine
builds are up to date, with unchanged executable hashes. The new source capture
pins 288 local source/header/test/build files and 194/201 compile-command entries;
it is a current build-closure observation, not a completed long-model admission.
Supplemental provenance review and Python source lints precede any GEN128 launch.

R313/R314 correct R311/R312, which overlooked completed committed measurements.
The existing production profile really captured 129820 valid events, about200GB
of requested H2D copy extents and complete32K output/fault evidence. Event-span
sums can overlap and do not establish an additive wall fraction or PCIe rate.
R315 audits the proposed Xe copy-only queue; cleanup and event-failure handling
need qualification before a port. No copy-queue performance gain is claimed.
R317 confirms that original official cache diagnostics are not an optimized
parameter search. That TODO, full262144-input coverage and RESTORE remain open.

Compact commands, environments, source/binary identities, individual host
outcomes and comparison decisions are committed here. Private original receipt
bytes remain pinned for current controller/provenance consumers. Repeated
successful command logs can be retired once this report is verified.
