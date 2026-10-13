<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Actual phase cache model lifetime check

On Arc B570 / Ryzen 5600X in free mode, fresh 1024 and 32768 inputs in one
process passed checked mapping, copy, readback and comparison, second-request
detach and workspace recreation, normal QUIT and exit0. Model/provider hashes
matched before and after, with no new GPU fault. Logging and validation were
enabled; these are lifetime diagnostics and supply no performance samples.

Single-GPU serial mode uses pipeline-windows=0. Correcting the erroneous1 gate
passed free and contrib-llvm builds with zero new lint findings. Original v1
startup failure and v2 harness failure retain failed outcomes. v2 incorrectly
rejected a valid early EOS and sent owned SIGTERM; v3 accepts EOS within the
output cap and checks semantics after normal QUIT.

Performance comparisons use unchanged upstream bench.sh long with3 runs.
Default remains OFF. Quality, full physical262144 input and head-lease effects
are not qualified. Exact prose identity is not a universal quality criterion.

R374 shows that the earlier1.18 GB was only a shared-region candidate, not the
whole workspace. Prior fit implications based on that total are withdrawn.
R375 copy/readback/hash proposals are unmeasured. JSON retains original reports
and Main's errata. Only closed duplicate logs retire after hash verification,
under the deletion manifest.
