<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Optional native copy queue: closed host qualification

Exact frozen STRATA_COPY_ENGINE=1 opts into a COPY-only Level Zero immediate
list imported with ownership::keep; default is OFF/generic. The native owner
outlives its borrowed queue. Ambiguous import/insertion exceptions fail-stop
before owners unwind. Checked drain/list/loader failures retain resources until
process exit. This does not fix or prove cleanup of the possible rawUR wrapper
leak. Public rawqueue access remains outside the supported-facade role guarantee.
Prefill changes only copy-queue creation; arithmetic and staging dependencies
are unchanged. Timing tags/compute/host-task/graphs are refused on native copy
streams, untimed markers are barriers, profiling diagnostics use generic queues.

Main built and tested the final source in free and contrib-llvm with source-built
intel/llvm7.1.1, WERROR and unchanged math settings. Each mode passes five fresh
exact/frozen policy processes and ten actual shared import/release-helper cases.
The same15CPU cases pass ASan/UBSan. Actual compile commands and ELF symbols
prove host instrumentation. These45checks are for the final source; the earlier
packet revision's separate successful checks remain recorded too. CPU fake
resources test the production helper boundary, not native adapter internals.

A first publication attempt stopped before deletion/commit on an incorrect
packet filename assumption; its original failed receipt is retained.
Initial lint review failed because cppcheck was absent from the chosen PATH and
clang-tidy lacked the sourceLLVM SYCL include path. Corrected environment reaches
every required tool. Main then corrected four new findings: explicit fatal
handling in catch, valid initial Level Zero descriptor enum values, assignment
outside an if condition, and external callback fault injection in the CPU fixture.
Both builds and all45CPU checks were repeated after those source changes.
Final before/after lints have no new findings by exact diagnostic and unchanged
source-line mapping. Existing baseline findings/nonzero exits remain recorded;
no rules were disabled. Original failed receipts and source packets are unchanged.

No GPU/model request was executed for this candidate at this boundary. Native
selection, actual handle/event/role/provider/teardown behavior, Prefill/Stager
qualification, speed and fullcontext/RESTORE remain open. R334 confirms no raw
Prefill kernel submission onto its copy stream, and leaves driver failure/source
credit behavior for actual qualification. Next is an owned tiny debugGPU check
with fresh integer health and fault cursors, then a distinct>=32K model check.
No performance gain or inference adoption is claimed.

R335 finds a potential16-bit down-intermediate saving of600MiB at12288rows,
conditional on MoE controlling the shared workspace max. No capacity or quality
gain was measured. The strict fixed-history comparator is still source-only and
needs Main CPU checks plus a reviewed producer receipt adapter. The separate
persistent phase-cache caller implementation is in progress, not completed here.

Commands/env/source/binary/individual CPU outcomes and original failures are
committed. Raw successful host command logs duplicated in verified receipts are
retired with path/hash/reason evidence. Private receipt bytes stay pinned for
future admission; source packets retain named finite consumers. Objective remains
1000prefill/70decode tok/s, with physical262144-input and quality work still open.
