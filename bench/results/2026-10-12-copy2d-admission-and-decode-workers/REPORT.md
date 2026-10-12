<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Copy2D admission and fixed-task decode worker comparison

The fresh remote check still resolves official main/v0.1.43 to 7695859 and
XeStrata main/xe0.1.42 to da3d426. Both revisions were already measured on 32K.
The combined source at 9b03c18 includes the required upstream server features.
There is no additional update to import. The pristine official measurements
remain diagnostics, not a ranking against fully optimized official settings.

Main reviewed the proposed expert Copy2D cohorts and added a shared production
admission scan for owned buffers, 16-row/tail/credit limits and source gaps.
The completed runtime investigation then exposed a decisive constraint:
the inspected UR Level Zero v2 context reports native 2D USM copies unsupported.
SYCL's later memory-manager branch emits one linear copy per row and an event
join. An extension call therefore cannot establish one native transfer command.
See the [memory manager source](https://github.com/intel/llvm/blob/504366f4b8/sycl/source/detail/memory_manager.cpp)
and [context capability](https://github.com/intel/llvm/blob/504366f4b8/unified-runtime/source/adapters/level_zero/v2/context.cpp).

The proposal was neither built nor executed on a GPU. It is retained as an
unqualified patch, including the separate Sol allocation-owner fixture.
All active engine/build files in this report branch were restored to their
byte-identical combined base. No capability bypass or hidden fallback was run.
The unresolved fallback event-pool concern in R291 is source analysis, not an
observed device fault. R292's unchanged-layout 1D option applies only to one
maximum-width layer: its command-count reduction ceiling is 1.95%, before
actual source gaps; it is not a speed measurement.

The next measurement uses the existing qualified binary, identical 32K input
and 16 output tokens. It compares workers 5/3/4/4/3/5 in fresh processes with
18 row tasks fixed, preserving every other setting. It checks effective INFO,
actual task/participant logging, CPU topology/affinity, exact token/LP5 output,
normal QUIT/reap and fresh kernel faults. Main's 36 CPU INFO checks and two
prelaunch rejection checks passed. No model ran for those host checks.

This is screening with two samples per worker count. Prefill and decode stay
separate, and no recipe promotion follows from short/noisy decode results.
The subsequent individual measurements will be recorded here. Full physical
262144-token input, RESTORE and the phase-lifetime proposal remain unqualified.
