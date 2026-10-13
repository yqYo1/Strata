<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Fixed-history capture receipt CPU checks

A companion validates provenance and closure for full-vocabulary FP32 logits
from the same history. It checks owner exit/reap, source/command/environment/
provider identity, exact row geometry and finite values. Incomplete captures
fail. Exact generated prose is not a quality requirement.

Main ran nine new schema/negative-input tests and eleven existing comparator
tests; all passed. All new-file lints passed after fixing annotation and wrapping
findings. The original failed lint outcome is retained. No model or GPU ran.
The real producer registry remains empty, with production and quality flags
false. Actual captures, GDN/QSA/PLE RESTORE comparisons and held-out quality are
unverified.

Corrected research confirms that stock ckpt omission is accepted with actual
fresh state and zero cache capacity, and that the existing native copy stream
already has only a prefill production caller. Generic decode queue selection
does not prove identical shared CPU guard overhead. No extra selector is added.
Performance comparisons continue through unchanged upstream bench.sh.

Original and corrected reports, commands/environment and failed/successful
checks are in the registry and receipts. Closed CPU-check duplicate logs retire
only after verification; pinned receipts remain unchanged. Next are Main-owned
phase model admission and actual production capture.
