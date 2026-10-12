<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Upstream conversation-cache Monitor port

The shared v0.1.43 Monitor now shows the conversation-cache budget, parked slots,
RAM use, reuse totals and last observed park/restore. Authorized JSON metrics
include the cache view. The Xe model switcher and existing Reused column remain.

Current-run byte offsets and unique run identities reset observations on restart.
A mutex prevents concurrent metrics readers from counting one event twice.
Each poll reads at most 1 MiB; parsing skips lines over 4096 characters and refuses
uint64 overflow. Lost backlog or a replaced/truncated log makes occupancy unknown
until another snapshot, with incomplete history reported. Dropped entries without
an updated byte total report unknown bytes. No raw log or path is exposed.

Main ran 566 host/fake-HTTP tests: 555 passed, 11 existing optional-dependency skips.
The added 11 cases cover incremental/partial log records, a real fake-child restart,
concurrent metrics, bounded reads, file replacement, lost history and numeric limits.
A Node fixture ran the actual renderer against actual page IDs for seven states.
Fresh Python, JavaScript, CSS, HTML, spelling, secret and license checks have no
new findings; existing findings remain. Three original failed lint receipts are
retained, including tool setup and comparison-controller failures.

All 413 engine/build/setup/tool/data files remain byte-identical to ba6bc783.
No GPU/model ran for this source change. Browser layout and native cache/SAVE/
RESTORE correctness are outside these host checks. Model settings and inference
behavior are unchanged. Exact source pins, commands, environment, process closure,
individual test results, lint findings and derivations are in the receipt index.
