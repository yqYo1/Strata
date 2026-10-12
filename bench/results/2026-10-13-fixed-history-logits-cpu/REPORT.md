<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Fixed-history full-logit comparator: CPU qualification

Main reviewed and applied the SourceOnly comparator in an isolated worktree.
It validates complete finite full-vocabulary captures, exact common histories,
positions, input identities, bounded geometry and declared lifecycle evidence.
It computes target NLL/PPL, both KL directions, ranks and probability margins
using float64 log-softmax. Changed generated argmax is permitted and measured.
No generation, inference kernel, GPU, model or service behavior changed.

All 11 analytical CPU tests passed. These cover known distributions, additive
logit shifts, same argmax with changed likelihood, changed argmax, overflow,
NaN/Inf and incomplete captures, mismatched IDs/positions/receipts, duplicate
JSON keys, valid big-endian captures, analytical forward/reverse KL and CLI
acceptance/rejection. The original nine-test source passed numerical fixtures
but had six lint findings. Main corrected them and added endian/CLI checks;
the final source passes flake8, ruff, mypy, codespell, gitleaks and reuse.
Both initial lint failures remain recorded; no rules or tests were weakened.

Receipt-schema checks do not attest actual production capture semantics.
The output explicitly keeps production_capture_qualified and quality_qualified
false. A reviewed adapter deriving fields from actual supervisor, provider,
fault-window and capture-close evidence is still required. No model degradation
rate, task-quality result, precision adoption or performance gain is claimed.
Next: qualify that capture path and score common held-out fixed histories plus
tasks. Generated-text equality remains a diagnostic rather than a quality gate.

Individual commands, source identities, test/lint results and original outcomes
are retained. Duplicate successful CPU command logs were retired after verified
extraction. Temporary analytical captures were closed with their unittest owners;
no production tensor/session fixture was removed. Objective remains active.
